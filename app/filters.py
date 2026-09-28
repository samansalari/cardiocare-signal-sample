"""Zero-phase ECG bandpass and a filter-based quality score."""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

logger = logging.getLogger(__name__)


@dataclass
class FilterOptions:
    """Bandpass and powerline notch settings.

    ``notch_freq`` defaults to 50 Hz. The live pipeline passes 60 Hz.
    """

    low_cut: float = 0.5
    high_cut: float = 40.0
    order: int = 4
    notch: bool = True
    notch_freq: float = 50.0
    notch_bw: float = 2.0
    notch_harmonics: int = 2


def ecg_bandpass(
    ecg: np.ndarray,
    fs: float,
    options: FilterOptions | None = None,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass, then optional harmonic notches.

    The result has the same length as ``ecg``. One channel only.
    """
    if options is None:
        options = FilterOptions()

    ecg = np.asarray(ecg, dtype=np.float64).ravel()
    nyquist = fs / 2.0

    # Keep the low-pass edge under Nyquist. This writes back onto options.
    if options.high_cut >= nyquist:
        options.high_cut = nyquist * 0.95
        logger.warning("High cutoff adjusted to %.1f Hz (Nyquist limit)", options.high_cut)

    wn = [options.low_cut / nyquist, options.high_cut / nyquist]
    sos_bp = butter(options.order, wn, btype="bandpass", output="sos")

    notches: list[tuple[np.ndarray, np.ndarray]] = []
    if options.notch:
        for harmonic in range(1, options.notch_harmonics + 1):
            notch_freq = options.notch_freq * harmonic
            if notch_freq < nyquist:
                quality = notch_freq / options.notch_bw
                b_notch, a_notch = iirnotch(notch_freq / nyquist, quality)
                notches.append((b_notch, a_notch))

    filtered = sosfiltfilt(sos_bp, ecg)
    for b_notch, a_notch in notches:
        filtered = filtfilt(b_notch, a_notch, filtered)
    return filtered


def estimate_signal_quality_from_filter(
    ecg_raw: np.ndarray,
    ecg_filtered: np.ndarray,
    fs: float,
) -> dict[str, float]:
    """Compare raw and filtered signals and map SNR onto a 0-100 score."""
    noise_estimate = np.asarray(ecg_raw, dtype=np.float64).ravel() - np.asarray(
        ecg_filtered, dtype=np.float64
    ).ravel()
    signal_power = float(np.var(ecg_filtered))
    noise_power = float(np.var(noise_estimate))

    if noise_power > 0:
        snr_db = float(10 * np.log10(signal_power / noise_power))
    else:
        snr_db = float("inf")

    nyquist = fs / 2.0
    sos_lp = butter(2, 0.5 / nyquist, btype="low", output="sos")
    baseline = sosfiltfilt(sos_lp, np.asarray(ecg_raw, dtype=np.float64).ravel())
    baseline_amplitude = float(np.ptp(baseline))

    if snr_db > 20:
        quality_score = 95.0
    elif snr_db > 15:
        quality_score = 85.0
    elif snr_db > 10:
        quality_score = 70.0
    elif snr_db > 5:
        quality_score = 55.0
    else:
        quality_score = 40.0

    return {
        "snr_db": snr_db,
        "signal_power": signal_power,
        "noise_power": noise_power,
        "baseline_amplitude": baseline_amplitude,
        "quality_score": quality_score,
    }
