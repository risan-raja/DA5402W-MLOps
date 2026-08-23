"""Load processed tabular/mel features and build train/val/eval splits."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.artifact_types import DatasetLineage, FileFingerprint, MelStats
from src.config import (
    DEFAULT_CONFIG_PATH,
    load_app_config,
)
from src.config import (
    load_training_config as _load_training_config,
)
from src.config_types import AppConfig, TrainingConfig
from src.data_pipeline.audio_features import (
    normalize_mels as _normalize_mels,
)
from src.data_pipeline.audio_features import tabular_feature_names
from src.data_pipeline.spark_feature_extractor import MELS_FILENAME, TABULAR_FILENAME

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = DEFAULT_CONFIG_PATH


def load_full_config(config_path: Path = CONFIG_PATH) -> AppConfig:
    return load_app_config(config_path)


def load_training_config(config_path: Path = CONFIG_PATH) -> TrainingConfig:
    return _load_training_config(config_path)


def _read_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return cast(dict[str, object], payload)


def _fingerprint(path: Path) -> FileFingerprint:
    """Size/mtime plus a fast content digest (head+tail) — full SHA of ~1GB mels is too slow."""
    if not path.is_file():
        return {"path": str(path), "exists": False}
    st = path.stat()
    h = hashlib.sha256()
    size: int = st.st_size
    chunk: int = 1024 * 1024
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if size > chunk * 2:
            f.seek(max(0, size - chunk))
            h.update(f.read(chunk))
        elif size > chunk:
            h.update(f.read())
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": size,
        "mtime": st.st_mtime,
        "sha256": h.hexdigest(),
        "digest_mode": "head_tail_1mb",
    }


def collect_dataset_lineage(
    processed_dir: Path | str,
    full_config: AppConfig | None = None,
) -> DatasetLineage:
    """Provenance for MLflow: HF repo + interim/processed manifests + file fingerprints."""
    processed_path: Path = Path(processed_dir)
    cfg: AppConfig = full_config if full_config is not None else load_full_config()
    dataset_cfg = cfg["dataset"]

    interim_dir: Path = Path(cfg["preprocessing"]["local_interim_dir"])
    if not interim_dir.is_absolute():
        interim_dir = ROOT / interim_dir
    raw_dir: Path = Path(dataset_cfg["local_raw_dir"])
    if not raw_dir.is_absolute():
        raw_dir = ROOT / raw_dir

    processed_manifest: dict[str, object] = (
        _read_json_if_exists(processed_path / ".manifest.json") or {}
    )
    interim_manifest: dict[str, object] = (
        _read_json_if_exists(interim_dir / ".manifest.json") or {}
    )
    raw_manifest: dict[str, object] = (
        _read_json_if_exists(raw_dir / ".manifest.json") or {}
    )

    return {
        "hf_repo_id": dataset_cfg.get("hf_repo_id"),
        "hf_repo_type": dataset_cfg.get("hf_repo_type", "dataset"),
        "raw_revision": cast(str | None, raw_manifest.get("revision")),
        "raw_downloaded_at": cast(str | None, raw_manifest.get("downloaded_at")),
        "interim_created_at": cast(str | None, interim_manifest.get("created_at")),
        "interim_num_rows": cast(int | None, interim_manifest.get("num_rows_written")),
        "processed_created_at": cast(str | None, processed_manifest.get("created_at")),
        "processed_num_rows": cast(int | None, processed_manifest.get("num_written")),
        "processed_num_input_rows": cast(
            int | None, processed_manifest.get("num_input_rows")
        ),
        "tabular": _fingerprint(processed_path / TABULAR_FILENAME),
        "mels": _fingerprint(processed_path / MELS_FILENAME),
        "processed_manifest": processed_manifest,
        "interim_manifest": interim_manifest,
        "raw_manifest": raw_manifest,
    }


def filter_split(
    frame: pd.DataFrame,
    folds: list[int] | tuple[int, ...],
    *,
    include_augmented: bool,
    fold_column: str = "fold",
) -> pd.DataFrame:
    mask = frame[fold_column].isin(list(folds))
    if not include_augmented and "is_augmented" in frame.columns:
        mask = mask & ~frame["is_augmented"].astype(bool)
    return frame.loc[mask].reset_index(drop=True)


def feature_columns(frame: pd.DataFrame, n_mfcc: int = 13) -> list[str]:
    expected: list[str] = tabular_feature_names(n_mfcc=n_mfcc)
    missing: list[str] = [c for c in expected if c not in frame.columns]
    if missing:
        raise ValueError(f"tabular frame missing feature columns: {missing[:5]}...")
    return expected


def build_label_maps(labels: pd.Series) -> tuple[dict[str, int], dict[int, str]]:
    classes: list[str] = sorted(labels.astype(str).unique())
    label_to_id: dict[str, int] = {c: i for i, c in enumerate(classes)}
    id_to_label: dict[int, str] = {i: c for c, i in label_to_id.items()}
    return label_to_id, id_to_label


def encode_labels(labels: pd.Series, label_to_id: dict[str, int]) -> np.ndarray:
    return labels.astype(str).map(label_to_id).to_numpy(dtype=np.int64)


def class_weight_vector(
    y: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Inverse-frequency weights shaped ``(n_classes,)`` for CE / sample weights."""
    counts: np.ndarray = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights: np.ndarray = counts.sum() / (n_classes * counts)
    return weights.astype(np.float64)


def sample_weights_from_y(y: np.ndarray, class_weights: np.ndarray) -> np.ndarray:
    return class_weights[y].astype(np.float64)


def load_tabular(processed_dir: Path | str) -> pd.DataFrame:
    path: Path = Path(processed_dir) / TABULAR_FILENAME
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def load_mels(processed_dir: Path | str) -> pd.DataFrame:
    path: Path = Path(processed_dir) / MELS_FILENAME
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def tabular_xy(
    frame: pd.DataFrame,
    feature_cols: list[str],
    label_to_id: dict[str, int],
    label_column: str = "class",
) -> tuple[np.ndarray, np.ndarray]:
    x: np.ndarray = frame[feature_cols].to_numpy(dtype=np.float32)
    y: np.ndarray = encode_labels(frame[label_column], label_to_id)
    return x, y


def fit_scaler(x_train: np.ndarray) -> StandardScaler:
    scaler: StandardScaler = StandardScaler()
    scaler.fit(x_train)
    return scaler


def transform_features(scaler: StandardScaler, x: np.ndarray) -> np.ndarray:
    return scaler.transform(x).astype(np.float32)


def reshape_mel_row(row: pd.Series) -> np.ndarray:
    h: int = int(row["mel_height"])
    w: int = int(row["mel_width"])
    mel: np.ndarray = np.asarray(row["mel"], dtype=np.float32).reshape(h, w)
    return mel


def mels_array(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.empty((0, 0, 0), dtype=np.float32)
    arrays: list[np.ndarray] = [reshape_mel_row(row) for _, row in frame.iterrows()]
    return np.stack(arrays, axis=0)


def fit_mel_stats(mels: np.ndarray) -> MelStats:
    if mels.size == 0:
        raise ValueError("cannot fit mel stats on empty array")
    return {
        "mean": float(np.mean(mels)),
        "std": float(max(np.std(mels), 1e-8)),
    }


def normalize_mels(mels: np.ndarray, stats: MelStats) -> np.ndarray:
    return _normalize_mels(mels, stats)


def split_frames(
    frame: pd.DataFrame,
    train_folds: list[int],
    val_fold: int,
    eval_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return optuna_train (1-8+aug), val (9 orig), refit (1-9+aug), eval (10 orig)."""
    optuna_train: pd.DataFrame = filter_split(frame, train_folds, include_augmented=True)
    val: pd.DataFrame = filter_split(frame, [val_fold], include_augmented=False)
    refit: pd.DataFrame = filter_split(
        frame, [*train_folds, val_fold], include_augmented=True
    )
    eval_df: pd.DataFrame = filter_split(frame, [eval_fold], include_augmented=False)
    return optuna_train, val, refit, eval_df


def iter_us8k_cv_folds(n_folds: int = 10) -> Iterator[tuple[int, list[int]]]:
    """Yield ``(test_fold, train_folds)`` for official UrbanSound8K fold CV.

    For each fold ``k`` in ``1..n_folds``, train on all other folds and evaluate on ``k``.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    folds: list[int] = list(range(1, n_folds + 1))
    for test_fold in folds:
        train_folds: list[int] = [f for f in folds if f != test_fold]
        yield test_fold, train_folds
