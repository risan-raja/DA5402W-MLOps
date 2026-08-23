"""UrbanSound8K evaluation cross-validation helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mlflow
import numpy as np

from src.models.mlflow_logging import save_json

# Metrics aggregated across folds (subset of compute_metrics keys).
_AGG_KEYS = (
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "roc_auc_ovr",
)


def aggregate_fold_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, float]:
    """Compute ``cv_{metric}_mean`` / ``cv_{metric}_std`` over fold metric dicts."""
    if not fold_metrics:
        raise ValueError("fold_metrics must be non-empty")

    aggregate: dict[str, float] = {}
    for key in _AGG_KEYS:
        values: list[float] = []
        for row in fold_metrics:
            if key not in row:
                continue
            val = float(row[key])
            if math.isfinite(val):
                values.append(val)
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        aggregate[f"cv_{key}_mean"] = float(np.mean(arr))
        aggregate[f"cv_{key}_std"] = float(np.std(arr, ddof=0))
    return aggregate


def log_cv_results(
    fold_rows: list[dict[str, Any]],
    aggregate: dict[str, float],
    out_dir: Path | str,
) -> Path:
    """Write ``cv_metrics.json``, log aggregate + per-fold F1 to the active MLflow run."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"folds": fold_rows, "aggregate": aggregate}
    path = out_dir / "cv_metrics.json"
    save_json(path, payload)

    for key, value in aggregate.items():
        if math.isfinite(value):
            mlflow.log_metric(key, value)
    for row in fold_rows:
        fold = int(row["fold"])
        f1 = row.get("f1_macro")
        if f1 is not None and math.isfinite(float(f1)):
            mlflow.log_metric(f"fold_{fold}_f1_macro", float(f1))
    mlflow.log_artifact(str(path))
    return path
