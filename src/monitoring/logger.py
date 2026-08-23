"""Stdout JSON prediction log (one object per line)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

logger: logging.Logger = logging.getLogger(__name__)


def log_prediction(
    *,
    label: str | None,
    confidence: float | None,
    model_name: str | None,
    latency_ms: float,
    filename: str | None = None,
    status: str = "ok",
) -> None:
    record: dict[str, object] = {
        "event": "prediction",
        "ts": datetime.now(UTC).isoformat(),
        "status": status,
        "label": label,
        "confidence": confidence,
        "model_name": model_name,
        "latency_ms": round(latency_ms, 3),
        "filename": filename,
    }
    logger.info("%s", json.dumps(record, default=str))
