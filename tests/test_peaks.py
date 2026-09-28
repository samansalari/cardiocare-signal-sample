"""R-peak rate on a synthetic trace."""

import numpy as np
import pytest

from app.pipeline import analyze_signal


def test_recovers_known_heart_rate(synthetic_ecg: np.ndarray, heart_rate_bpm: float) -> None:
    result = analyze_signal(synthetic_ecg, 250.0)
    assert result["filtered_length"] == len(synthetic_ecg)
    assert len(result["r_peaks"]) >= 10
    assert len(result["rr_intervals_ms"]) == len(result["r_peaks"]) - 1
    assert result["mean_heart_rate_bpm"] == pytest.approx(heart_rate_bpm, abs=5.0)
    assert 0.0 <= result["signal_quality"] <= 100.0
