"""TypedDicts for dataset lineage, train results, and winner payloads."""

from __future__ import annotations

from typing import TypedDict


class FileFingerprint(TypedDict, total=False):
    path: str
    exists: bool
    size_bytes: int
    mtime: float
    sha256: str
    digest_mode: str


class DatasetLineage(TypedDict, total=False):
    hf_repo_id: str | None
    hf_repo_type: str | None
    raw_revision: str | None
    raw_downloaded_at: str | None
    interim_created_at: str | None
    interim_num_rows: int | None
    processed_created_at: str | None
    processed_num_rows: int | None
    processed_num_input_rows: int | None
    tabular: FileFingerprint
    mels: FileFingerprint
    processed_manifest: dict[str, object]
    interim_manifest: dict[str, object]
    raw_manifest: dict[str, object]


class TrainResult(TypedDict):
    model_name: str
    run_id: str
    metrics: dict[str, float]
    out_dir: str


class WinnerDatasetInfo(TypedDict, total=False):
    hf_repo_id: str | None
    processed_created_at: str | None
    tabular_sha256: str | None


class WinnerPayload(TypedDict, total=False):
    model_name: str
    f1_macro: float | None
    run_id: str | None
    dataset: WinnerDatasetInfo
    cv_f1_macro_mean: float | None
    cv_f1_macro_std: float | None


class FoldMetricRow(TypedDict, total=False):
    fold: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    roc_auc_ovr: float


class SchemaValidationResult(TypedDict):
    num_rows: int
    num_classes: int


class MelStats(TypedDict):
    mean: float
    std: float


class LineageStub(TypedDict, total=False):
    hf_repo_id: str | None
    processed_created_at: str | None
    tabular: FileFingerprint


class RunResultPayload(TrainResult, total=False):
    dataset: LineageStub | WinnerDatasetInfo


class CnnSuggestParams(TypedDict):
    lr: float
    batch_size: int
    seed: int


class CnnHistory(TypedDict):
    best_val_f1_macro: float
    best_val_accuracy: float
    best_val_f1_weighted: float
    epochs_run: int


__all__ = [
    "CnnHistory",
    "CnnSuggestParams",
    "DatasetLineage",
    "FileFingerprint",
    "FoldMetricRow",
    "LineageStub",
    "MelStats",
    "RunResultPayload",
    "SchemaValidationResult",
    "TrainResult",
    "WinnerDatasetInfo",
    "WinnerPayload",
]
