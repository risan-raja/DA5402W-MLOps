from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pytest

from src.data_pipeline.dataset_downloader import (
    _normalize_targets,
    download_dataset,
    load_config,
    raw_data_present,
    validate_schema,
)
from src.data_processing.versioning import (
    TRAINED_MODEL_DIRS,
    config_enabled,
    pull_winner_artifacts,
    push_all_trained_models,
    push_winner_artifacts,
    resolve_model_repo_id,
    versioning_push_enabled,
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


def test_raw_data_present_requires_manifest_and_parquet(tmp_path):
    assert raw_data_present(tmp_path) is False

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "part.parquet").write_bytes(b"x")
    assert raw_data_present(tmp_path) is False

    (tmp_path / ".manifest.json").write_text("{}")
    assert raw_data_present(tmp_path) is True


def test_versioning_push_enabled(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "dataset:\n  hf_repo_id: example/repo\n"
        "versioning:\n  push_interim: true\n  push_processed: false\n"
        "  push_models: true\n"
    )
    assert versioning_push_enabled("push_interim", config_path=cfg) is True
    assert versioning_push_enabled("push_processed", config_path=cfg) is False
    assert versioning_push_enabled("push_models", config_path=cfg) is True
    assert versioning_push_enabled("missing_key", config_path=cfg) is False


def test_config_enabled_section_flags(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("preprocessing:\n  enabled: false\nspark:\n  enabled: true\n")
    assert config_enabled("preprocessing", config_path=cfg) is False
    assert config_enabled("spark", config_path=cfg) is True
    assert config_enabled("missing_section", config_path=cfg) is True


def test_resolve_model_repo_id(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "dataset:\n  hf_repo_id: example/ds\n"
        "versioning:\n  hf_model_repo_id: org/models\n  hf_model_repo_type: model\n"
    )
    assert resolve_model_repo_id(config_path=cfg) == ("org/models", "model")


def test_resolve_model_repo_id_requires_id(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "dataset:\n  hf_repo_id: example/ds\nversioning:\n  path_in_repo: interim\n"
    )
    with pytest.raises(ValueError, match="hf_model_repo_id"):
        resolve_model_repo_id(config_path=cfg)


def test_push_all_trained_models_requires_each_dir(tmp_path):
    (tmp_path / "rf").mkdir()
    with pytest.raises(FileNotFoundError):
        push_all_trained_models(tmp_path)


def test_push_winner_artifacts_requires_json_and_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        push_winner_artifacts(tmp_path)
    (tmp_path / "winner.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        push_winner_artifacts(tmp_path)


def test_push_all_trained_models_uploads_each_dir(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "dataset:\n  hf_repo_id: example/ds\n"
        "versioning:\n  hf_model_repo_id: org/models\n  hf_model_repo_type: model\n"
    )
    models_dir = tmp_path / "models"
    for name in TRAINED_MODEL_DIRS:
        (models_dir / name).mkdir(parents=True)
    uploaded: list[str] = []

    def _fake_push(local_dir, path_in_repo, **_kwargs):
        uploaded.append(path_in_repo)
        return "ok"

    monkeypatch.setattr("src.data_processing.versioning.push_model_tree", _fake_push)
    push_all_trained_models(models_dir, config_path=cfg)
    assert uploaded == list(TRAINED_MODEL_DIRS)


def test_pull_winner_artifacts_downloads_allow_patterns(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "dataset:\n  hf_repo_id: example/ds\n"
        "versioning:\n  hf_model_repo_id: org/models\n  hf_model_repo_type: model\n"
    )
    dest = tmp_path / "models"
    called: dict = {}

    def _fake_snapshot(**kwargs):
        called.update(kwargs)
        Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "src.data_processing.versioning.snapshot_download", _fake_snapshot
    )
    pull_winner_artifacts(dest, config_path=cfg)
    assert called["repo_id"] == "org/models"
    assert called["allow_patterns"] == ["winner.json", "winner/**"]
    assert dest.is_dir()


def test_dag_project_root_is_repo_root():
    dag_file = (
        Path(__file__).resolve().parents[1]
        / "airflow"
        / "dags"
        / "audio_classification_dag.py"
    )
    project_root = dag_file.parents[2]
    assert (project_root / "src" / "models" / "train.py").is_file()
    assert (project_root / "config" / "config.yaml").is_file()


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
