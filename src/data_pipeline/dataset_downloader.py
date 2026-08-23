"""Fetch UrbanSound8K raw, interim, and/or processed from one HF dataset repo."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import pyarrow.compute as pc
import pyarrow.dataset as ds
from huggingface_hub import HfApi, snapshot_download

from src.artifact_types import SchemaValidationResult
from src.config import DEFAULT_CONFIG_PATH, load_app_config, load_dataset_config
from src.config_types import (
    AppConfig,
    DatasetConfig,
    PreprocessingConfig,
    SparkConfig,
    VersioningConfig,
)

logger: logging.Logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = DEFAULT_CONFIG_PATH
MANIFEST_FILENAME = ".manifest.json"
METADATA_FILENAME = "metadata.parquet"
AUDIO_DIRNAME = "audio"
TABULAR_FILENAME = "tabular.parquet"
MELS_FILENAME = "mels.parquet"
VALID_TARGETS: frozenset[str] = frozenset({"raw", "interim", "processed"})


class _ArrowTableLike(Protocol):
    def column(self, name: str) -> object: ...


class _ArrowDatasetLike(Protocol):
    """Minimal surface of ``pyarrow.dataset.Dataset`` used by schema checks."""

    @property
    def schema(self) -> object: ...

    def count_rows(self) -> int: ...

    def to_table(self, columns: list[str] | None = None) -> _ArrowTableLike: ...


def load_full_config(config_path: Path = CONFIG_PATH) -> AppConfig:
    return load_app_config(config_path)


def load_config(config_path: Path = CONFIG_PATH) -> DatasetConfig:
    return load_dataset_config(config_path)

def _manifest_path(local_dir: Path) -> Path:
    return local_dir / MANIFEST_FILENAME


def raw_data_present(local_raw_dir: Path | str | None = None) -> bool:
    """True when local raw has a manifest and at least one parquet under ``data/``."""
    if local_raw_dir is None:
        local_raw_dir = Path(load_config()["local_raw_dir"])
    else:
        local_raw_dir = Path(local_raw_dir)
    parquet_dir = local_raw_dir / "data"
    if not _manifest_path(local_raw_dir).is_file():
        return False
    if not parquet_dir.is_dir():
        return False
    return any(parquet_dir.glob("*.parquet"))


def interim_data_present(local_interim_dir: Path | str | None = None) -> bool:
    """True when interim has ``metadata.parquet`` and at least one wav under ``audio/``."""
    if local_interim_dir is None:
        local_interim_dir = Path(
            load_full_config()["preprocessing"]["local_interim_dir"]
        )
    else:
        local_interim_dir = Path(local_interim_dir)
    metadata_path = local_interim_dir / METADATA_FILENAME
    audio_dir = local_interim_dir / AUDIO_DIRNAME
    if not metadata_path.is_file():
        return False
    if not audio_dir.is_dir():
        return False
    return any(audio_dir.rglob("*.wav"))


def processed_data_present(local_processed_dir: Path | str | None = None) -> bool:
    """True when processed has both ``tabular.parquet`` and ``mels.parquet``."""
    if local_processed_dir is None:
        local_processed_dir = Path(load_full_config()["spark"]["local_processed_dir"])
    else:
        local_processed_dir = Path(local_processed_dir)
    return (local_processed_dir / TABULAR_FILENAME).is_file() and (
        local_processed_dir / MELS_FILENAME
    ).is_file()


def _read_existing_manifest(local_dir: Path) -> dict[str, object] | None:
    manifest_path = _manifest_path(local_dir)
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        return cast(dict[str, object], json.load(f))


def _write_manifest(local_dir: Path, manifest: dict[str, object]) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    with open(_manifest_path(local_dir), "w") as f:
        json.dump(manifest, f, indent=2)


def validate_schema(
    table: _ArrowDatasetLike, config: DatasetConfig
) -> SchemaValidationResult:
    num_rows = table.count_rows()
    if num_rows != config["expected_rows"]:
        raise ValueError(
            f"Row count mismatch: expected {config['expected_rows']}, got {num_rows}"
        )

    schema = table.schema
    columns = cast(list[str], getattr(schema, "names", []))
    fold_col = config["fold_column"]
    label_col = config["label_column"]
    if fold_col not in columns:
        raise ValueError(
            f"Missing expected column '{fold_col}'. Columns present: {columns}"
        )
    if label_col not in columns:
        raise ValueError(
            f"Missing expected column '{label_col}'. Columns present: {columns}"
        )

    fold_values = table.to_table(columns=[fold_col]).column(fold_col)
    fold_min, fold_max = config["fold_range"]
    observed_min = pc.min(fold_values).as_py()
    observed_max = pc.max(fold_values).as_py()
    if observed_min < fold_min or observed_max > fold_max:
        raise ValueError(
            f"Fold values out of expected range {config['fold_range']}: "
            f"observed [{observed_min}, {observed_max}]"
        )

    label_values = table.to_table(columns=[label_col]).column(label_col)
    num_classes = pc.count_distinct(label_values).as_py()
    if num_classes != config["expected_num_classes"]:
        raise ValueError(
            f"Expected {config['expected_num_classes']} classes, found {num_classes}"
        )

    return {"num_rows": num_rows, "num_classes": num_classes}


def _normalize_targets(targets: list[str] | tuple[str, ...] | None) -> list[str]:
    if not targets:
        return ["raw"]
    normalized: list[str] = []
    for t in targets:
        key = t.strip().lower()
        if key not in VALID_TARGETS:
            raise ValueError(
                f"Unknown download target '{t}'. Expected one of {sorted(VALID_TARGETS)}"
            )
        if key not in normalized:
            normalized.append(key)
    return normalized


def _download_raw(
    config: DatasetConfig | dict[str, object],
    revision: str,
    *,
    force: bool,
    allow_patterns: list[str] | None,
) -> dict[str, object]:
    local_raw_dir: Path = Path(str(config["local_raw_dir"]))
    local_raw_dir.mkdir(parents=True, exist_ok=True)

    existing: dict[str, object] | None = _read_existing_manifest(local_raw_dir)
    if not force and existing is not None and existing.get("revision") == revision:
        logger.info("Raw already at revision %s, skipping download", revision)
        return existing

    patterns: list[str] = allow_patterns or cast(
        list[str],
        config.get(
            "raw_allow_patterns",
            ["data/*.parquet", "UrbanSound8K.csv"],
        ),
    )
    logger.info(
        "Downloading raw from %s (revision %s) into %s",
        config["hf_repo_id"],
        revision,
        local_raw_dir,
    )
    snapshot_download(
        repo_id=str(config["hf_repo_id"]),
        repo_type=str(config["hf_repo_type"]),
        revision=revision,
        local_dir=str(local_raw_dir),
        allow_patterns=patterns,
    )

    table = ds.dataset(str(local_raw_dir / "data"), format="parquet")
    stats: SchemaValidationResult = validate_schema(
        table, cast(DatasetConfig, config)
    )
    manifest: dict[str, object] = {
        "target": "raw",
        "hf_repo_id": config["hf_repo_id"],
        "revision": revision,
        "downloaded_at": datetime.now(UTC).isoformat(),
        **stats,
    }
    _write_manifest(local_raw_dir, manifest)
    logger.info("Raw validated and manifest written: %s", manifest)
    return manifest


def _download_interim(
    full_config: AppConfig,
    revision: str,
    *,
    force: bool,
) -> dict[str, object]:
    dataset_cfg: DatasetConfig = full_config["dataset"]
    prep_cfg: PreprocessingConfig = full_config["preprocessing"]
    local_interim_dir: Path = Path(prep_cfg["local_interim_dir"])
    data_root: Path = local_interim_dir.parent
    data_root.mkdir(parents=True, exist_ok=True)

    existing = _read_existing_manifest(local_interim_dir)
    if not force and existing is not None and existing.get("revision") == revision:
        logger.info("Interim already at revision %s, skipping download", revision)
        return existing

    patterns: list[str] = dataset_cfg.get("interim_allow_patterns", ["interim/**"])
    versioning: VersioningConfig | dict[str, object] = full_config.get(
        "versioning", {}
    )
    path_in_repo: str = str(versioning.get("path_in_repo", "interim"))
    logger.info(
        "Downloading interim from %s (revision %s) into %s",
        dataset_cfg["hf_repo_id"],
        revision,
        local_interim_dir,
    )
    snapshot_download(
        repo_id=dataset_cfg["hf_repo_id"],
        repo_type=dataset_cfg["hf_repo_type"],
        revision=revision,
        local_dir=str(data_root),
        allow_patterns=patterns,
    )

    metadata_path: Path = local_interim_dir / METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"interim download finished but {metadata_path} is missing. "
            f"Push interim to {dataset_cfg['hf_repo_id']} under {path_in_repo}/ first."
        )

    manifest = {
        "target": "interim",
        "hf_repo_id": dataset_cfg["hf_repo_id"],
        "revision": revision,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "local_interim_dir": str(local_interim_dir),
        "has_metadata": True,
    }
    _write_manifest(local_interim_dir, manifest)
    logger.info("Interim downloaded and manifest written: %s", manifest)
    return manifest


def _download_processed(
    full_config: AppConfig,
    revision: str,
    *,
    force: bool,
) -> dict[str, object]:
    dataset_cfg: DatasetConfig = full_config["dataset"]
    spark_cfg: SparkConfig = full_config["spark"]
    local_processed_dir: Path = Path(spark_cfg["local_processed_dir"])
    data_root: Path = local_processed_dir.parent
    data_root.mkdir(parents=True, exist_ok=True)

    existing = _read_existing_manifest(local_processed_dir)
    if not force and existing is not None and existing.get("revision") == revision:
        logger.info("Processed already at revision %s, skipping download", revision)
        return existing

    patterns = dataset_cfg.get("processed_allow_patterns", ["processed/**"])
    versioning = full_config.get("versioning", {})
    path_in_repo = str(versioning.get("processed_path_in_repo", "processed"))
    logger.info(
        "Downloading processed from %s (revision %s) into %s",
        dataset_cfg["hf_repo_id"],
        revision,
        local_processed_dir,
    )
    snapshot_download(
        repo_id=dataset_cfg["hf_repo_id"],
        repo_type=dataset_cfg["hf_repo_type"],
        revision=revision,
        local_dir=str(data_root),
        allow_patterns=patterns,
    )

    tabular_path: Path = local_processed_dir / TABULAR_FILENAME
    mels_path: Path = local_processed_dir / MELS_FILENAME
    missing: list[str] = [p.name for p in (tabular_path, mels_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"processed download finished but missing {missing} under "
            f"{local_processed_dir}. Push processed to {dataset_cfg['hf_repo_id']} "
            f"under {path_in_repo}/ first."
        )

    manifest = {
        "target": "processed",
        "hf_repo_id": dataset_cfg["hf_repo_id"],
        "revision": revision,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "local_processed_dir": str(local_processed_dir),
        "has_tabular": True,
        "has_mels": True,
    }
    _write_manifest(local_processed_dir, manifest)
    logger.info("Processed downloaded and manifest written: %s", manifest)
    return manifest


def download_dataset(
    config: DatasetConfig | dict[str, object] | None = None,
    force: bool = False,
    allow_patterns: list[str] | None = None,
    targets: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Download from ``dataset.hf_repo_id`` (raw, ``interim/``, and/or ``processed/``).

    ``targets`` defaults to ``["raw"]``. Use ``["interim"]``, ``["processed"]``, or
    combinations to pull versioned pipeline outputs. Raw-only returns the raw
    manifest dict (tests / existing callers). Multi-target returns
    ``{revision, hf_repo_id, raw?, interim?, processed?}``.
    """
    full_config: AppConfig = load_full_config()
    if config is not None:
        full_config["dataset"] = cast(
            DatasetConfig, {**full_config["dataset"], **config}
        )
    dataset_cfg: DatasetConfig = full_config["dataset"]
    selected: list[str] = _normalize_targets(targets)

    revision: str = HfApi().dataset_info(dataset_cfg["hf_repo_id"]).sha
    results: dict[str, object] = {
        "revision": revision,
        "hf_repo_id": dataset_cfg["hf_repo_id"],
    }
    if "raw" in selected:
        results["raw"] = _download_raw(
            dataset_cfg,
            revision,
            force=force,
            allow_patterns=allow_patterns,
        )
    if "interim" in selected:
        results["interim"] = _download_interim(full_config, revision, force=force)
    if "processed" in selected:
        results["processed"] = _download_processed(full_config, revision, force=force)

    if selected == ["raw"]:
        return cast(dict[str, object], results["raw"])
    return results


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description=(
            "Download raw, interim, and/or processed from the project HF dataset repo"
        )
    )
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        choices=sorted(VALID_TARGETS),
        help=(
            "Repeatable. Default: raw. "
            "Example: --target raw --target interim --target processed"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_dataset(force=args.force, targets=args.targets)


if __name__ == "__main__":
    main()
