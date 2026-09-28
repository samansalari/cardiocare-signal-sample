"""Request and response models for the sample API."""

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AnalyzeRequest(BaseModel):
    """One single-channel ECG trace."""

    model_config = ConfigDict(extra="forbid")

    sampling_rate_hz: float = Field(ge=100, le=1000)
    samples: list[float] = Field(min_length=50)

    @field_validator("samples")
    @classmethod
    def samples_must_be_finite(cls, samples: list[float]) -> list[float]:
        """Reject NaN and infinity before they reach the filter."""
        if any(not math.isfinite(value) for value in samples):
            raise ValueError("samples must be finite")
        return samples


class AnalyzeResponse(BaseModel):
    """Beat summary. No waveform is returned."""

    filtered_length: int
    r_peaks: list[int]
    rr_intervals_ms: list[float]
    mean_heart_rate_bpm: float | None
    signal_quality: float


class HealthResponse(BaseModel):
    """Liveness only."""

    status: str
