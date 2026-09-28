"""Filter length and finite output."""

import numpy as np

from app.filters import ecg_bandpass
from tests.conftest import make_synthetic_ecg


def test_bandpass_keeps_length_and_finite_values(synthetic_ecg: np.ndarray) -> None:
    filtered = ecg_bandpass(synthetic_ecg, 250.0)
    assert filtered.shape == synthetic_ecg.shape
    assert np.isfinite(filtered).all()


def test_short_trace_stays_finite() -> None:
    raw = make_synthetic_ecg(250.0, 72.0, duration_s=0.2)
    assert len(raw) == 50
    filtered = ecg_bandpass(raw, 250.0)
    assert len(filtered) == len(raw)
    assert np.isfinite(filtered).all()
