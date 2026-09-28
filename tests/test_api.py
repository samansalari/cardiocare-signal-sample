"""HTTP contract for the sample API."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import make_synthetic_ecg

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_happy_path() -> None:
    samples = make_synthetic_ecg(250.0, 72.0, duration_s=30.0)
    response = client.post(
        "/analyze",
        json={"sampling_rate_hz": 250, "samples": samples.tolist()},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filtered_length"] == len(samples)
    assert body["mean_heart_rate_bpm"] == pytest.approx(72.0, abs=5.0)
    assert len(body["rr_intervals_ms"]) == len(body["r_peaks"]) - 1


def test_rejects_sampling_rate_below_range() -> None:
    response = client.post(
        "/analyze",
        json={"sampling_rate_hz": 40, "samples": [0.0] * 50},
    )
    assert response.status_code == 422


def test_rejects_sampling_rate_above_range() -> None:
    response = client.post(
        "/analyze",
        json={"sampling_rate_hz": 2000, "samples": [0.0] * 50},
    )
    assert response.status_code == 422


def test_rejects_short_trace() -> None:
    response = client.post(
        "/analyze",
        json={"sampling_rate_hz": 250, "samples": [0.1, 0.2, 0.3]},
    )
    assert response.status_code == 422

