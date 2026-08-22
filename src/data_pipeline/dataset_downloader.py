import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds
import yaml
from huggingface_hub import HfApi, snapshot_download

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
MANIFEST_FILENAME = ".manifest.json"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)["dataset"]


def _manifest_path(local_raw_dir: Path) -> Path:
    return local_raw_dir / MANIFEST_FILENAME


def _read_existing_manifest(local_raw_dir: Path) -> dict | None:
    manifest_path = _manifest_path(local_raw_dir)
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        return json.load(f)


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
        raise ValueError(f"Missing expected column '{fold_col}'. Columns present: {columns}")
    if label_col not in columns:
        raise ValueError(f"Missing expected column '{label_col}'. Columns present: {columns}")

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


def download_dataset(
    config: dict | None = None,
    force: bool = False,
    allow_patterns: list[str] | None = None,
) -> dict:
    config = config or load_config()
    local_raw_dir = Path(config["local_raw_dir"])
    local_raw_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    revision = api.dataset_info(config["hf_repo_id"]).sha

    existing = _read_existing_manifest(local_raw_dir)
    if not force and existing is not None and existing.get("revision") == revision:
        logger.info("Dataset already at revision %s, skipping download", revision)
        return existing

    allow_patterns = allow_patterns or ["data/*.parquet", "UrbanSound8K.csv"]
    logger.info("Downloading %s (revision %s) into %s", config["hf_repo_id"], revision, local_raw_dir)
    snapshot_download(
        repo_id=config["hf_repo_id"],
        repo_type=config["hf_repo_type"],
        local_dir=str(local_raw_dir),
        allow_patterns=allow_patterns,
    )

    table = ds.dataset(str(local_raw_dir / "data"), format="parquet")
    stats = validate_schema(table, config)

    manifest = {
        "hf_repo_id": config["hf_repo_id"],
        "revision": revision,
        "downloaded_at": datetime.now(UTC).isoformat(),
        **stats,
    }
    with open(_manifest_path(local_raw_dir), "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Dataset validated and manifest written: %s", manifest)
    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_dataset()
