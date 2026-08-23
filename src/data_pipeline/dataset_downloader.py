"""Fetch UrbanSound8K raw and/or versioned interim from one HF dataset repo."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds
import yaml
from huggingface_hub import HfApi, snapshot_download

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"
MANIFEST_FILENAME = ".manifest.json"
VALID_TARGETS = frozenset({"raw", "interim"})


def load_full_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    return load_full_config(config_path)["dataset"]


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


def _read_existing_manifest(local_dir: Path) -> dict | None:
    manifest_path = _manifest_path(local_dir)
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        return json.load(f)


def _write_manifest(local_dir: Path, manifest: dict) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    with open(_manifest_path(local_dir), "w") as f:
        json.dump(manifest, f, indent=2)


def validate_schema(table: ds.Dataset, config: dict) -> dict:
    num_rows = table.count_rows()
    if num_rows != config["expected_rows"]:
        raise ValueError(
            f"Row count mismatch: expected {config['expected_rows']}, got {num_rows}"
        )

    columns = table.schema.names
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
    config: dict,
    revision: str,
    *,
    force: bool,
    allow_patterns: list[str] | None,
) -> dict:
    local_raw_dir = Path(config["local_raw_dir"])
    local_raw_dir.mkdir(parents=True, exist_ok=True)

    existing = _read_existing_manifest(local_raw_dir)
    if not force and existing is not None and existing.get("revision") == revision:
        logger.info("Raw already at revision %s, skipping download", revision)
        return existing

    patterns = allow_patterns or config.get(
        "raw_allow_patterns",
        ["data/*.parquet", "UrbanSound8K.csv"],
    )
    logger.info(
        "Downloading raw from %s (revision %s) into %s",
        config["hf_repo_id"],
        revision,
        local_raw_dir,
    )
    snapshot_download(
        repo_id=config["hf_repo_id"],
        repo_type=config["hf_repo_type"],
        revision=revision,
        local_dir=str(local_raw_dir),
        allow_patterns=patterns,
    )

    table = ds.dataset(str(local_raw_dir / "data"), format="parquet")
    stats = validate_schema(table, config)
    manifest = {
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
    full_config: dict,
    revision: str,
    *,
    force: bool,
) -> dict:
    dataset_cfg = full_config["dataset"]
    prep_cfg = full_config["preprocessing"]
    local_interim_dir = Path(prep_cfg["local_interim_dir"])
    data_root = local_interim_dir.parent
    data_root.mkdir(parents=True, exist_ok=True)

    existing = _read_existing_manifest(local_interim_dir)
    if not force and existing is not None and existing.get("revision") == revision:
        logger.info("Interim already at revision %s, skipping download", revision)
        return existing

    patterns = dataset_cfg.get("interim_allow_patterns", ["interim/**"])
    path_in_repo = full_config.get("versioning", {}).get("path_in_repo", "interim")
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

    metadata_path = local_interim_dir / "metadata.parquet"
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


def download_dataset(
    config: dict | None = None,
    force: bool = False,
    allow_patterns: list[str] | None = None,
    targets: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Download from ``dataset.hf_repo_id`` (raw parquet and/or ``interim/``).

    ``targets`` defaults to ``["raw"]``. Use ``["interim"]`` or ``["raw", "interim"]``
    to pull versioned cleaned audio. Raw-only returns the raw manifest dict (tests /
    existing callers). Multi-target returns ``{revision, hf_repo_id, raw?, interim?}``.
    """
    full_config = load_full_config()
    if config is not None:
        full_config["dataset"] = {**full_config["dataset"], **config}
    dataset_cfg = full_config["dataset"]
    selected = _normalize_targets(targets)

    revision = HfApi().dataset_info(dataset_cfg["hf_repo_id"]).sha
    results: dict = {
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

    if selected == ["raw"]:
        return results["raw"]
    return results


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Download raw and/or interim from the project HF dataset repo"
    )
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        choices=sorted(VALID_TARGETS),
        help="Repeatable. Default: raw. Example: --target raw --target interim",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_dataset(force=args.force, targets=args.targets)


if __name__ == "__main__":
    main()
