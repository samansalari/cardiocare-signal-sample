"""Pan-Tompkins QRS detection with searchback and polarity-aware refinement."""

import logging

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, convolve, find_peaks, sosfiltfilt

logger = logging.getLogger(__name__)


def pan_tompkins(ecg: np.ndarray, fs: float) -> dict[str, np.ndarray | str]:
    """Detect R-peak sample indices.

    Bandpass 5-15 Hz, five-point derivative, square, 150 ms integration,
    then adaptive thresholding. Returns peaks on the original sample index.
    """
    ecg = np.asarray(ecg, dtype=np.float64).ravel()
    nyquist = fs / 2.0
    low = 5.0 / nyquist
    high = min(15.0, nyquist * 0.95) / nyquist
    sos_bp = butter(2, [low, high], btype="bandpass", output="sos")
    ecg_bp = sosfiltfilt(sos_bp, ecg)

    # H(z) = (1/8T) (-z^-2 - 2z^-1 + 2z + z^2), T = 1/fs.
    derivative_kernel = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) * (fs / 8.0)
    derivative = convolve(ecg_bp, derivative_kernel, mode="same")
    squared = derivative**2

    window_width = max(1, int(0.150 * fs))
    integrated = uniform_filter1d(squared, size=window_width, mode="constant")
    qrs_polarity = _determine_qrs_polarity(ecg_bp, integrated, fs)
    r_peaks = _adaptive_threshold_improved(integrated, ecg_bp, fs, qrs_polarity)

    logger.info("Pan-Tompkins detected %s R-peaks", len(r_peaks))
    return {
        "r_peaks": r_peaks,
        "r_times": r_peaks / fs,
        "filtered": ecg_bp,
        "derivative": derivative,
        "squared": squared,
        "integrated": integrated,
        "qrs_polarity": qrs_polarity,
    }


def _determine_qrs_polarity(ecg_bp: np.ndarray, integrated: np.ndarray, fs: float) -> str:
    """Return ``upright``, ``inverted``, or ``mixed`` from a sample of beats."""
    min_distance = int(0.3 * fs)
    peaks, _props = find_peaks(integrated, distance=min_distance, height=0)
    if len(peaks) < 3:
        return "upright"

    n_samples_check = min(20, len(peaks))
    sample_indices = np.linspace(0, len(peaks) - 1, n_samples_check, dtype=int)
    search_window = int(0.100 * fs)
    n_upright = 0
    n_inverted = 0

    for peak in peaks[sample_indices]:
        start = max(0, peak - search_window)
        end = min(len(ecg_bp), peak + search_window)
        segment = ecg_bp[start:end]
        if len(segment) == 0:
            continue
        pos_max = float(np.max(segment))
        neg_min = float(np.min(segment))
        if pos_max > abs(neg_min) * 0.7:
            n_upright += 1
        elif abs(neg_min) > pos_max * 0.7:
            n_inverted += 1
        else:
            n_upright += 1

    total = n_upright + n_inverted
    if total == 0:
        return "upright"
    upright_ratio = n_upright / total
    if upright_ratio > 0.7:
        return "upright"
    if upright_ratio < 0.3:
        return "inverted"
    return "mixed"


def _adaptive_threshold_improved(
    integrated: np.ndarray,
    ecg_bp: np.ndarray,
    fs: float,
    qrs_polarity: str = "upright",
) -> np.ndarray:
    """Adaptive QRS threshold, T-wave rejection, searchback, then R-peak snap."""
    n_samples = len(integrated)
    min_rr = int(0.2 * fs)  # 300 bpm
    max_rr = int(2.0 * fs)  # 30 bpm
    refractory_period = int(0.200 * fs)
    t_wave_start = int(0.150 * fs)
    t_wave_end = int(0.360 * fs)

    peaks, properties = find_peaks(
        integrated,
        distance=min_rr,
        height=0,
        prominence=0,
    )
    if len(peaks) == 0:
        return np.array([], dtype=int)

    peak_heights = properties["peak_heights"]
    learning_samples = int(2.0 * fs)
    learning_peaks_mask = peaks < learning_samples
    if np.sum(learning_peaks_mask) >= 2:
        learning_heights = peak_heights[learning_peaks_mask]
        spki = float(np.max(learning_heights) * 0.25)
        sorted_heights = np.sort(learning_heights)
        npki = float(np.mean(sorted_heights[: len(sorted_heights) // 2 + 1]) * 0.5)
    else:
        spki = float(np.percentile(peak_heights, 75) * 0.25)
        npki = float(np.percentile(peak_heights, 25) * 0.5)

    threshold_i1 = npki + 0.25 * (spki - npki)
    threshold_i2 = 0.5 * threshold_i1
    rr_intervals: list[int] = []
    rr_low = 0.92 * max_rr
    rr_high = 1.16 * min_rr
    rr_missed = 1.66 * max_rr

    qrs_peaks: list[int] = []
    last_qrs_idx = -max_rr

    for peak_idx, height in zip(peaks, peak_heights, strict=True):
        peak_idx = int(peak_idx)
        height = float(height)
        if peak_idx - last_qrs_idx < refractory_period:
            continue

        if height > threshold_i1:
            is_t_wave = False
            if qrs_peaks:
                time_since_last = peak_idx - last_qrs_idx
                if t_wave_start < time_since_last < t_wave_end:
                    current_slope = _get_max_slope(ecg_bp, peak_idx, fs)
                    last_qrs_slope = _get_max_slope(ecg_bp, qrs_peaks[-1], fs)
                    if current_slope < 0.5 * last_qrs_slope:
                        is_t_wave = True
            if not is_t_wave:
                qrs_peaks.append(peak_idx)
                spki = 0.125 * height + 0.875 * spki
                if len(qrs_peaks) >= 2:
                    rr = qrs_peaks[-1] - qrs_peaks[-2]
                    rr_intervals.append(rr)
                    if len(rr_intervals) >= 8:
                        rr_average1 = float(np.mean(rr_intervals[-8:]))
                        normal_rr = [r for r in rr_intervals[-8:] if rr_low < r < rr_high]
                        if len(normal_rr) >= 2:
                            rr_average2 = float(np.mean(normal_rr))
                        else:
                            rr_average2 = rr_average1
                        rr_low = 0.92 * rr_average2
                        rr_high = 1.16 * rr_average2
                        rr_missed = 1.66 * rr_average2
                last_qrs_idx = peak_idx
        else:
            npki = 0.125 * height + 0.875 * npki

        threshold_i1 = npki + 0.25 * (spki - npki)
        threshold_i2 = 0.5 * threshold_i1

        if len(qrs_peaks) >= 2 and len(rr_intervals) >= 8:
            current_rr = peak_idx - last_qrs_idx
            if current_rr > rr_missed:
                searchback_start = last_qrs_idx + refractory_period
                searchback_end = peak_idx - refractory_period
                searchback_mask = (peaks > searchback_start) & (peaks < searchback_end)
                missed_candidates = [
                    (int(sb_peak), float(sb_height))
                    for sb_peak, sb_height in zip(
                        peaks[searchback_mask],
                        peak_heights[searchback_mask],
                        strict=True,
                    )
                    if sb_height > threshold_i2
                ]
                if missed_candidates:
                    missed_candidates.sort(key=lambda item: item[1], reverse=True)
                    missed_peak, missed_height = missed_candidates[0]
                    if all(abs(missed_peak - qp) > refractory_period for qp in qrs_peaks):
                        qrs_peaks.append(missed_peak)
                        qrs_peaks.sort()
                        spki = 0.25 * missed_height + 0.75 * spki

    qrs_peaks_arr = np.unique(np.array(qrs_peaks, dtype=int))
    search_window = int(0.100 * fs)
    refined_peaks: list[int] = []
    for peak in qrs_peaks_arr:
        start = max(0, int(peak) - search_window)
        end = min(n_samples, int(peak) + search_window)
        segment_bp = ecg_bp[start:end]
        if len(segment_bp) == 0:
            continue
        local_max_idx = _local_r_index(segment_bp, qrs_polarity)
        refined_peak = start + local_max_idx
        if 0 <= refined_peak < n_samples:
            refined_peaks.append(refined_peak)
        else:
            refined_peaks.append(int(peak))

    refined = np.unique(np.array(refined_peaks, dtype=int))
    if len(refined) > 1:
        cleaned = [int(refined[0])]
        for peak in refined[1:]:
            if int(peak) - cleaned[-1] >= min_rr:
                cleaned.append(int(peak))
        refined = np.array(cleaned, dtype=int)
    return refined


def _local_r_index(segment_bp: np.ndarray, qrs_polarity: str) -> int:
    """Index of the R wave inside a window on the 5-15 Hz signal."""
    if qrs_polarity == "upright":
        local_max_idx = int(np.argmax(segment_bp))
        if segment_bp[local_max_idx] <= 0:
            return int(np.argmax(np.abs(segment_bp)))
        return local_max_idx
    if qrs_polarity == "inverted":
        return int(np.argmin(segment_bp))

    pos_max_idx = int(np.argmax(segment_bp))
    pos_max_val = float(segment_bp[pos_max_idx])
    neg_min_idx = int(np.argmin(segment_bp))
    neg_min_val = abs(float(segment_bp[neg_min_idx]))
    if pos_max_val > 0 and pos_max_idx < neg_min_idx:
        return pos_max_idx
    if pos_max_val > neg_min_val * 0.5:
        return pos_max_idx
    segment_before_s = segment_bp[:neg_min_idx] if neg_min_idx > 0 else segment_bp
    if len(segment_before_s) > 0 and float(np.max(segment_before_s)) > 0:
        return int(np.argmax(segment_before_s))
    return int(np.argmax(np.abs(segment_bp)))


def _get_max_slope(signal: np.ndarray, peak_idx: int, fs: float) -> float:
    """Largest absolute sample-to-sample slope in a 40 ms window."""
    window = int(0.040 * fs)
    start = max(0, peak_idx - window)
    end = min(len(signal), peak_idx + window)
    if end - start < 3:
        return 0.0
    derivative = np.abs(np.diff(signal[start:end]))
    if len(derivative) == 0:
        return 0.0
    return float(np.max(derivative))
