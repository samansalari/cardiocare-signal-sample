"""The path POST /analyze runs: filter, Pan-Tompkins, intervals, quality."""

import logging

import numpy as np

from app.filters import (
    FilterOptions,
    ecg_bandpass,
    estimate_signal_quality_from_filter,
)
from app.peaks import pan_tompkins

logger = logging.getLogger(__name__)


def analyze_signal(samples: list[float] | np.ndarray, sampling_rate_hz: float) -> dict:
    """Filter one ECG channel and summarise beats.

    Filter numbers are the ones ``SignalProcessor`` passes from settings:
    0.5-40 Hz, order 4, notch 60 Hz, bandwidth 2 Hz, two harmonics.
    Baseline removal stays off.
    """
    raw = np.asarray(samples, dtype=np.float64).ravel()
    options = FilterOptions(
        low_cut=0.5,
        high_cut=40.0,
        order=4,
        notch=True,
        notch_freq=60.0,
        notch_bw=2.0,
        notch_harmonics=2,
    )
    filtered = ecg_bandpass(raw, sampling_rate_hz, options)
    peaks = np.asarray(pan_tompkins(filtered, sampling_rate_hz)["r_peaks"], dtype=int)
    rr_ms = _rr_intervals_ms(peaks, sampling_rate_hz)
    quality = _signal_quality(raw, filtered, peaks, sampling_rate_hz)
    logger.info(
        "Analyzed %s samples at %s Hz: %s peaks, quality %.1f",
        len(raw),
        sampling_rate_hz,
        len(peaks),
        quality,
    )
    return {
        "filtered_length": int(len(filtered)),
        "r_peaks": [int(index) for index in peaks],
        "rr_intervals_ms": [float(interval) for interval in rr_ms],
        "mean_heart_rate_bpm": _mean_heart_rate_bpm(rr_ms),
        "signal_quality": float(quality),
    }


def _rr_intervals_ms(peaks: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    """Successive R-R intervals in milliseconds."""
    if len(peaks) < 2:
        return np.array([], dtype=np.float64)
    return np.diff(peaks) / sampling_rate_hz * 1000.0


def _mean_heart_rate_bpm(rr_ms: np.ndarray) -> float | None:
    """Mean of the beat-to-beat rates. None when there is no interval."""
    if len(rr_ms) == 0:
        return None
    return float(np.mean(60_000.0 / rr_ms))


def _signal_quality(
    raw: np.ndarray,
    filtered: np.ndarray,
    peaks: np.ndarray,
    sampling_rate_hz: float,
) -> float:
    """Higher of the peak-regularity SQI and the filter SNR score."""
    sqi = _peak_sqi(filtered, peaks, sampling_rate_hz)
    filter_score = estimate_signal_quality_from_filter(raw, filtered, sampling_rate_hz)
    return max(sqi, filter_score["quality_score"])


def _peak_sqi(ecg_signal: np.ndarray, peaks: np.ndarray, sampling_rate_hz: float) -> float:
    """0-100 score from RR regularity (weight 0.6) and a coarse SNR (weight 0.4)."""
    if len(peaks) < 5:
        logger.warning("Insufficient peaks for SQI calculation")
        return 0.0

    rr_intervals = np.diff(peaks) / sampling_rate_hz
    rr_mean = float(np.mean(rr_intervals))
    rr_std = float(np.std(rr_intervals))
    regularity = 1.0 - min(rr_std / rr_mean, 1.0) if rr_mean > 0 else 0.0

    signal_power = float(np.mean(ecg_signal**2))
    noise_estimate = float(np.std(np.diff(ecg_signal)))
    snr = 10.0 * np.log10(signal_power / (noise_estimate**2 + 1e-10))
    snr_normalized = min(snr / 20.0, 1.0)
    sqi = (0.6 * regularity + 0.4 * snr_normalized) * 100.0
    return float(max(0.0, min(100.0, sqi)))
