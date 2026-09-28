CardioCare is a real-time ECG and PPG monitoring SaaS. I am the sole developer. It is Innovate UK funded.

This repository is a cut-down copy of the signal path: a 0.5–40 Hz bandpass with powerline notches, then Pan-Tompkins R-peak detection, then R-R intervals and a quality score. The FastAPI wrapper (`POST /analyze`, `GET /health`) was written for this sample. Production does not expose this route.

Left out, one line each:

- Auth, database, and billing: they are production concerns and are not part of the signal path.
- Arrhythmia models (AFib, PVC, and the rest of `app/ml_models/`): those were built by the team's ML engineer.
- Patient recordings: nothing here is a real ECG. Tests build a synthetic pulse train.
- Hamilton-Tompkins detector: not selected. The live setting is `rpeak_detector = "pan_tompkins"`.
- Ensemble detector: consensus of Pan-Tompkins and Hamilton. Not the live detector.
- NeuroKit2 peak detection: the other implementation in `SignalProcessor`. This sample does not depend on it.
- Scipy `find_peaks` fallback (80 ms window, threshold at mean + 0.5 standard deviations): that was the stand-in when NeuroKit2 was missing. It is a different algorithm, so I did not swap it in. Pan-Tompkins already runs on SciPy.
- Standard filter chain (0.5 Hz high-pass, single notch with Q=30, then another 0.5–40 Hz bandpass): used only when `use_advanced_filters` is false. The live flag is true.
- Median-filter baseline removal (200 ms window): `enable_baseline_removal` is false on the live path.
- Multi-channel transpose: `POST /analyze` takes one list of samples.
- `matlab_loader.py`: production input adapter for MATLAB structs. This route takes JSON samples.
- HRV (SDNN, RMSSD, pNN50, LF/HF) and PR/QRS/QT interval measurement: not part of this response.
- Production request and response schemas for stored recordings, strips, and arrhythmia payloads: they do not describe this JSON body.
- Async wrappers on the filter itself: the route calls `asyncio.to_thread` once around the whole pipeline.

Detector choice: Pan-Tompkins, because that is what the service runs by default. SciPy is the numeric library. NeuroKit2 is not imported.

```
JSON samples + sampling_rate_hz
        |
        v
ecg_bandpass
  Butterworth 0.5–40 Hz, order 4, zero phase
  notches at 60 Hz and 120 Hz, bandwidth 2 Hz
        |
        v
pan_tompkins
  5–15 Hz QRS band, derivative, square, 150 ms integrate
  adaptive threshold, 200 ms refractory, searchback
        |
        v
R-R intervals (ms), mean of beat-to-beat heart rates, signal quality
        |
        v
JSON response
```

`ecg_bandpass` takes a `FilterOptions` object, so the cutoffs and the notch frequency are arguments rather than a second function. The live call passes 60 Hz. The dataclass default is still 50 Hz, which is what you get if you call `ecg_bandpass` with no options.

Filtering and peak detection are ordinary numeric code. They do not await. Running them directly inside an async route would stall every other request on that worker. `asyncio.to_thread` runs that work on a thread, and NumPy releases the GIL during the filter.

```bash
make install
make test
make run
```

`make lint` runs ruff. The server listens on `$PORT`, or 8000 if `$PORT` is unset. Then `GET /health` and `POST /analyze` with `{"sampling_rate_hz": 250, "samples": [...]}`. Accepted rates are 100 to 1000 Hz. The sample list must contain at least 50 finite numbers.

With more time I would make the notch default one number. Settings use 60 Hz and `FilterOptions` still defaults to 50 Hz. I would also stop taking the higher of the two quality scores, because a trace with almost no peaks can still score 40 from the filter SNR. The searchback and T-wave switches in settings are not read by the detector; I would wire them or delete them.
