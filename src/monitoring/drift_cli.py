"""Build fold-10 drift references and score offline prediction or feature logs."""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pandas as pd

from src.config import load_app_config
from src.config_types import AppConfig, DriftMonitoringConfig
from src.data_pipeline.audio_features import tabular_feature_names
from src.models.runtime_env import ROOT
from src.monitoring.drift_detector import (
    DEFAULT_FEATURE_BINS,
    OFFICIAL_FOLD10_COUNTS,
    DriftReference,
    build_feature_histograms,
    dump_reference,
    load_reference,
    official_fold10_reference,
    reference_from_labels,
    score_features,
    score_predictions,
    with_confidence_samples,
    with_feature_histograms,
)

TABULAR_FILENAME = "tabular.parquet"
METADATA_FILENAME = "metadata.parquet"
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


def _load_config(path: Path) -> AppConfig:
    return load_app_config(path)


def _drift_cfg(config: AppConfig) -> DriftMonitoringConfig | dict[str, object]:
    monitoring = config.get("monitoring") or {}
    return cast(
        DriftMonitoringConfig | dict[str, object], monitoring.get("drift") or {}
    )


def resolve_reference_path(
    config: AppConfig,
    *,
    models_root: Path | None = None,
    explicit: Path | None = None,
) -> Path:
    if explicit is not None:
        return explicit
    dcfg = _drift_cfg(config)
    raw = Path(dcfg.get("reference_path", "models/drift_reference.json"))
    if raw.is_absolute():
        return raw
    if models_root is not None:
        named = models_root / raw.name
        if named.is_file() or models_root.exists():
            return named
    return ROOT / raw


def _processed_dir(config: AppConfig) -> Path:
    path = Path(config.get("spark", {}).get("local_processed_dir", "data/processed"))
    return path if path.is_absolute() else ROOT / path


def _interim_dir(config: AppConfig) -> Path:
    path = Path(
        config.get("preprocessing", {}).get("local_interim_dir", "data/interim")
    )
    return path if path.is_absolute() else ROOT / path


def _eval_fold(config: AppConfig) -> int:
    return int(config.get("training", {}).get("eval_fold", 10))


def _fold_labels(frame: pd.DataFrame, fold: int) -> pd.DataFrame:
    if "fold" not in frame.columns or "class" not in frame.columns:
        raise ValueError("expected fold and class columns")
    subset = frame.loc[frame["fold"].astype(int) == fold]
    if "is_augmented" in subset.columns:
        subset = subset.loc[~subset["is_augmented"].astype(bool)]
    return subset.reset_index(drop=True)


def _load_fold10_frame(config: AppConfig) -> tuple[pd.DataFrame | None, str]:
    fold = _eval_fold(config)
    processed = _processed_dir(config) / TABULAR_FILENAME
    if processed.is_file():
        frame = _fold_labels(pd.read_parquet(processed), fold)
        if not frame.empty:
            return frame, "processed_tabular"
    interim = _interim_dir(config) / METADATA_FILENAME
    if interim.is_file():
        frame = _fold_labels(pd.read_parquet(interim), fold)
        if not frame.empty:
            return frame, "interim_metadata"
    return None, "official_urbansound8k_fold10"


def _winner_meta(config: AppConfig) -> dict[str, object]:
    models_dir = Path(config.get("training", {}).get("models_dir", "models"))
    if not models_dir.is_absolute():
        models_dir = ROOT / models_dir
    payload_path = models_dir / "winner.json"
    if not payload_path.is_file():
        return {}
    with open(payload_path) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return {}
    dataset = cast(dict[str, object], payload.get("dataset") or {})
    return {
        "dataset_id": dataset.get("hf_repo_id")
        or config.get("dataset", {}).get("hf_repo_id"),
        "tabular_sha256": dataset.get("tabular_sha256"),
        "winner_model": payload.get("model_name"),
        "winner_run_id": payload.get("run_id"),
    }


def _feature_columns(frame: pd.DataFrame, n_mfcc: int) -> list[str]:
    expected = tabular_feature_names(n_mfcc=n_mfcc)
    return [name for name in expected if name in frame.columns]


def _stored_confidences(frame: pd.DataFrame) -> list[float]:
    for column in ("confidence", "max_proba"):
        if column not in frame.columns:
            continue
        values = [float(v) for v in frame[column].tolist() if pd.notna(v)]
        if values:
            return values
    return []


def _read_confidence_file(path: Path) -> list[float]:
    values: list[float] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            values.append(float(line))
    return values


def build_reference(
    config: AppConfig,
    *,
    with_features: bool = False,
    confidence_path: Path | None = None,
    n_bins: int | None = None,
) -> DriftReference:
    extra = _winner_meta(config)
    extra["dataset_id"] = extra.get("dataset_id") or config.get("dataset", {}).get(
        "hf_repo_id"
    )
    extra["eval_fold"] = _eval_fold(config)
    frame, source = _load_fold10_frame(config)
    if frame is None:
        warnings.warn(
            "fold-10 labels not found under data/processed or data/interim; "
            "using official UrbanSound8K fold-10 class counts",
            stacklevel=2,
        )
        reference = official_fold10_reference(extra_meta=extra)
    else:
        extra["source"] = source
        extra["n_rows"] = len(frame)
        reference = reference_from_labels(
            frame["class"].astype(str).tolist(), extra_meta=extra
        )
    if confidence_path is not None:
        samples = _read_confidence_file(confidence_path)
        if samples:
            reference = with_confidence_samples(reference, samples)
        else:
            warnings.warn(f"no confidence samples in {confidence_path}", stacklevel=2)
    elif frame is not None:
        samples = _stored_confidences(frame)
        if samples:
            reference = with_confidence_samples(reference, samples)
        else:
            warnings.warn(
                "no fold-10 confidence samples; class-prior-only reference",
                stacklevel=2,
            )
    else:
        warnings.warn(
            "no fold-10 confidence samples; class-prior-only reference",
            stacklevel=2,
        )
    if with_features:
        if frame is None:
            raise FileNotFoundError(
                "build-reference --with-features needs data/processed/tabular.parquet"
            )
        n_mfcc = int(config.get("spark", {}).get("n_mfcc", 13))
        cols = _feature_columns(frame, n_mfcc)
        if not cols:
            raise ValueError("processed fold-10 frame has no tabular feature columns")
        histograms = build_feature_histograms(
            {name: frame[name].to_numpy(dtype=float) for name in cols},
            n_bins=n_bins
            or int(_drift_cfg(config).get("n_feature_bins", DEFAULT_FEATURE_BINS)),
        )
        reference = with_feature_histograms(reference, histograms)
    return reference


def _iter_prediction_records(path: Path) -> Iterator[tuple[str, float]]:
    with open(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not JSON") from exc
            if not isinstance(record, dict):
                continue
            if record.get("event") not in (None, "prediction"):
                continue
            if record.get("status", "ok") != "ok":
                continue
            label = record.get("label")
            if label is None:
                continue
            yield str(label), float(record.get("confidence") or 0.0)


def fold10_prediction_labels(config: AppConfig) -> list[str]:
    frame, _source = _load_fold10_frame(config)
    if frame is not None:
        return frame["class"].astype(str).tolist()
    labels: list[str] = []
    for name, count in OFFICIAL_FOLD10_COUNTS.items():
        labels.extend([name] * int(count))
    return labels


def write_fold10_prediction_log(path: Path, config: AppConfig) -> int:
    """Write a synthetic JSONL of fold-10 labels so score-predictions can run."""
    labels = fold10_prediction_labels(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (
        json.dumps(
            {
                "event": "prediction",
                "status": "ok",
                "label": label,
                "confidence": None,
            }
        )
        + "\n"
        for label in labels
    )
    with open(path, "w") as handle:
        handle.writelines(lines)
    return len(labels)


def ensure_prediction_jsonl(path: Path, config: AppConfig) -> Path:
    if path.is_file():
        return path
    n_rows = write_fold10_prediction_log(path, config)
    warnings.warn(
        f"{path} was missing; wrote {n_rows} fold-10 labels as a demo prediction log",
        stacklevel=2,
    )
    return path


def score_prediction_file(path: Path, reference: DriftReference) -> dict:
    labels: list[str] = []
    confidences: list[float] = []
    for label, confidence in _iter_prediction_records(path):
        labels.append(label)
        confidences.append(confidence)
    snapshot = score_predictions(labels, confidences, reference)
    return {
        "psi_class": snapshot.psi_class,
        "ks_confidence": snapshot.ks_confidence,
        "n_observations": snapshot.n_observations,
    }


def score_feature_file(path: Path, reference: DriftReference, n_mfcc: int) -> dict:
    if not reference.has_features:
        raise ValueError(
            "reference has no feature_histograms; rebuild with --with-features"
        )
    frame = pd.read_parquet(path)
    if "is_augmented" in frame.columns:
        originals = frame.loc[~frame["is_augmented"].astype(bool)]
        if not originals.empty:
            frame = originals
    cols = [
        name
        for name in tabular_feature_names(n_mfcc=n_mfcc)
        if name in frame.columns and name in reference.feature_histograms
    ]
    return score_features(
        {name: reference.feature_histograms[name] for name in cols},
        {name: frame[name].to_numpy(dtype=float) for name in cols},
    )


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or score KS/PSI drift references."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to config.yaml",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser(
        "build-reference", help="Write models/drift_reference.json from fold 10"
    )
    build_p.add_argument("--output", type=Path, default=None)
    build_p.add_argument(
        "--with-features",
        action="store_true",
        help="Include tabular feature histograms (requires processed parquet)",
    )
    build_p.add_argument(
        "--confidences",
        type=Path,
        default=None,
        help="Optional text file of one fold-10 confidence per line",
    )

    pred_p = sub.add_parser(
        "score-predictions",
        help="Score a JSONL of prediction logs against the reference",
    )
    pred_p.add_argument("jsonl", type=Path)
    pred_p.add_argument("--reference", type=Path, default=None)

    feat_p = sub.add_parser(
        "score-features",
        help="Score a tabular parquet against fold-10 feature histograms",
    )
    feat_p.add_argument("parquet", type=Path)
    feat_p.add_argument("--reference", type=Path, default=None)
    feat_p.add_argument(
        "--with-features",
        action="store_true",
        help="Build fold-10 feature histograms into the reference if they are missing",
    )

    args = parser.parse_args(argv)
    config = _load_config(args.config)
    if args.command == "build-reference":
        reference = build_reference(
            config,
            with_features=args.with_features,
            confidence_path=args.confidences,
        )
        dest = resolve_reference_path(config, explicit=args.output)
        dump_reference(dest, reference)
        _print_json({"wrote": str(dest), "meta": reference.meta})
        return 0
    reference_path = resolve_reference_path(config, explicit=args.reference)
    reference = load_reference(reference_path)
    if args.command == "score-predictions":
        jsonl = ensure_prediction_jsonl(args.jsonl, config)
        _print_json(score_prediction_file(jsonl, reference))
        return 0
    if args.with_features and not reference.has_features:
        reference = build_reference(config, with_features=True)
        dump_reference(reference_path, reference)
    n_mfcc = int(config.get("spark", {}).get("n_mfcc", 13))
    _print_json(score_feature_file(args.parquet, reference, n_mfcc))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
