"""FastAPI serving surface: /health, /metrics, /predict."""

from __future__ import annotations

from src.models.runtime_env import load_runtime_env

load_runtime_env()

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Annotated, cast

from fastapi import FastAPI, File, HTTPException, UploadFile
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from src.config_types import AppConfig
from src.deployment.infer import predict_clip
from src.deployment.metrics import (
    DRIFT_KS_CONFIDENCE,
    DRIFT_PSI_CLASS,
    DRIFT_WINDOW_SIZE,
    PREDICT_CLASS,
    PREDICT_CONFIDENCE,
    PREDICT_ERRORS,
    PREDICT_LATENCY,
    PREDICT_REQUESTS,
)
from src.deployment.runtime import ServingState, load_serving_state
from src.deployment.schemas import HealthResponse, PredictResponse
from src.monitoring.logger import log_prediction

logger: logging.Logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
WAV_SUFFIXES = {".wav"}
WAV_MIMES = {"audio/wav", "audio/x-wav", "audio/wave"}

_STATE = ServingState(
    model=None, config=cast(AppConfig, {}), models_dir=Path("models")
)


def current_state() -> ServingState:
    return _STATE


def set_state(state: ServingState) -> None:
    global _STATE
    _STATE = state


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    set_state(load_serving_state())
    yield


app: FastAPI = FastAPI(title="da5402w", lifespan=lifespan)


def _observe_drift(state: ServingState, label: str, confidence: float) -> None:
    monitor = state.drift_monitor
    if monitor is None:
        return
    snapshot = monitor.update(label, confidence)
    DRIFT_WINDOW_SIZE.set(monitor.filled)
    if snapshot is None:
        return
    DRIFT_PSI_CLASS.set(snapshot.psi_class)
    if snapshot.ks_confidence is not None:
        DRIFT_KS_CONFIDENCE.set(snapshot.ks_confidence)


def _validate_upload(
    filename: str | None, content_type: str | None, data: bytes
) -> None:
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="upload exceeds 8 MiB")
    suffix = Path(filename or "").suffix.lower()
    mime = (content_type or "").split(";")[0].strip().lower()
    if suffix not in WAV_SUFFIXES and mime not in WAV_MIMES:
        raise HTTPException(status_code=400, detail="expected a .wav file")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
async def predict(file: Annotated[UploadFile, File()]) -> PredictResponse:
    PREDICT_REQUESTS.inc()
    started = perf_counter()
    state = current_state()
    filename = file.filename
    try:
        data = await file.read()
        _validate_upload(filename, file.content_type, data)
        if state.model is None:
            raise HTTPException(
                status_code=503,
                detail=state.error or "winning model is not loaded",
            )
        result = predict_clip(state.model, data, state.config)
        latency_ms = (perf_counter() - started) * 1000.0
        PREDICT_LATENCY.observe(latency_ms / 1000.0)
        PREDICT_CLASS.labels(class_name=result["label"]).inc()
        PREDICT_CONFIDENCE.observe(float(result["confidence"]))
        log_prediction(
            label=result["label"],
            confidence=result["confidence"],
            model_name=result["model_name"],
            latency_ms=latency_ms,
            filename=filename,
            status="ok",
        )
        _observe_drift(state, result["label"], float(result["confidence"]))
        return PredictResponse(latency_ms=round(latency_ms, 3), **result)
    except HTTPException as exc:
        PREDICT_ERRORS.inc()
        PREDICT_LATENCY.observe(perf_counter() - started)
        log_prediction(
            label=None,
            confidence=None,
            model_name=state.model.model_name if state.model else None,
            latency_ms=(perf_counter() - started) * 1000.0,
            filename=filename,
            status=f"http_{exc.status_code}",
        )
        raise
    except (ValueError, OSError, RuntimeError) as exc:
        PREDICT_ERRORS.inc()
        PREDICT_LATENCY.observe(perf_counter() - started)
        log_prediction(
            label=None,
            confidence=None,
            model_name=state.model.model_name if state.model else None,
            latency_ms=(perf_counter() - started) * 1000.0,
            filename=filename,
            status="error",
        )
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
