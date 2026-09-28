"""Sample HTTP wrapper. The processing underneath is synchronous NumPy/SciPy."""

import asyncio
import logging

from fastapi import FastAPI

from app.pipeline import analyze_signal
from app.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="CardioCare signal sample", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Process is up. This does not check a database."""
    return HealthResponse(status="ok")


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeRequest) -> AnalyzeResponse:
    """Run the filter and Pan-Tompkins path off the event loop."""
    logger.info("POST /analyze samples=%s fs=%s", len(body.samples), body.sampling_rate_hz)
    result = await asyncio.to_thread(analyze_signal, body.samples, body.sampling_rate_hz)
    return AnalyzeResponse(**result)
