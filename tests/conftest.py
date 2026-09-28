"""Synthetic ECG-like traces. Not recordings."""

import numpy as np
import pytest


@pytest.fixture
def sampling_rate_hz() -> float:
    return 250.0


@pytest.fixture
def heart_rate_bpm() -> float:
    return 72.0


@pytest.fixture
def synthetic_ecg(sampling_rate_hz: float, heart_rate_bpm: float) -> np.ndarray:
    """Thirty seconds of a pulse train at a known rate."""
    return make_synthetic_ecg(sampling_rate_hz, heart_rate_bpm, duration_s=30.0)


def make_synthetic_ecg(
    sampling_rate_hz: float,
    heart_rate_bpm: float,
    duration_s: float,
) -> np.ndarray:
    """Gaussian P, QRS, and T waves plus a slow drift and a 60 Hz tone."""
    n_samples = int(duration_s * sampling_rate_hz)
    times = np.arange(n_samples) / sampling_rate_hz
    signal = np.zeros(n_samples, dtype=np.float64)
    rr_s = 60.0 / heart_rate_bpm
    beat = 0.4
    while beat < duration_s - 0.4:
        signal += 1.2 * np.exp(-0.5 * ((times - beat) / 0.008) ** 2)
        signal += 0.15 * np.exp(-0.5 * ((times - (beat - 0.16)) / 0.018) ** 2)
        signal += 0.30 * np.exp(-0.5 * ((times - (beat + 0.20)) / 0.040) ** 2)
        beat += rr_s
    signal += 0.05 * np.sin(2 * np.pi * 0.15 * times)
    signal += 0.02 * np.sin(2 * np.pi * 60.0 * times)
    return signal
