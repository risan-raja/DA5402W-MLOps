import pyarrow as pa
import pyarrow.dataset as ds
import pytest

from src.data_pipeline.dataset_downloader import (
    _normalize_targets,
    download_dataset,
    load_config,
    validate_schema,
)

CONFIG = {
    "expected_rows": 4,
    "fold_column": "fold",
    "label_column": "class",
    "fold_range": [1, 10],
    "expected_num_classes": 2,
}


def _table(rows: dict) -> ds.Dataset:
    return ds.dataset(pa.Table.from_pydict(rows))


def test_validate_schema_accepts_matching_data():
    table = _table(
        {
            "fold": [1, 2, 3, 4],
            "class": ["dog_bark", "siren", "dog_bark", "siren"],
        }
    )
    stats = validate_schema(table, CONFIG)
    assert stats == {"num_rows": 4, "num_classes": 2}


def test_validate_schema_rejects_wrong_row_count():
    table = _table({"fold": [1, 2, 3], "class": ["dog_bark", "siren", "dog_bark"]})
    with pytest.raises(ValueError, match="Row count mismatch"):
        validate_schema(table, CONFIG)


def test_validate_schema_rejects_missing_fold_column():
    table = _table(
        {
            "not_fold": [1, 2, 3, 4],
            "class": ["dog_bark", "siren", "dog_bark", "siren"],
        }
    )
    with pytest.raises(ValueError, match="Missing expected column 'fold'"):
        validate_schema(table, CONFIG)


def test_validate_schema_rejects_fold_out_of_range():
    table = _table(
        {
            "fold": [1, 2, 3, 11],
            "class": ["dog_bark", "siren", "dog_bark", "siren"],
        }
    )
    with pytest.raises(ValueError, match="Fold values out of expected range"):
        validate_schema(table, CONFIG)


def test_validate_schema_rejects_wrong_class_count():
    table = _table(
        {
            "fold": [1, 2, 3, 4],
            "class": ["dog_bark", "siren", "gun_shot", "engine_idling"],
        }
    )
    with pytest.raises(ValueError, match="Expected 2 classes"):
        validate_schema(table, CONFIG)


def test_normalize_targets_defaults_and_rejects_unknown():
    assert _normalize_targets(None) == ["raw"]
    assert _normalize_targets(["raw", "interim", "raw"]) == ["raw", "interim"]
    with pytest.raises(ValueError, match="Unknown download target"):
        _normalize_targets(["processed"])


@pytest.mark.integration
def test_download_dataset_end_to_end(tmp_path):
    config = load_config()
    config["local_raw_dir"] = str(tmp_path)
    config["expected_rows"] = 546
    config["expected_num_classes"] = 10
    manifest = download_dataset(
        config,
        allow_patterns=["data/train-00000-of-00016-e478d7cccca6a095.parquet"],
    )
    assert manifest["num_rows"] == config["expected_rows"]
