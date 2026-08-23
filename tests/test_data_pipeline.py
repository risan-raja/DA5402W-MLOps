from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.config_helpers import write_app_config

import pyarrow as pa
import pyarrow.dataset as ds
import pytest

from src.artifact_types import SchemaValidationResult
from src.config_types import DatasetConfig
from src.data_pipeline.dataset_downloader import (
    _normalize_targets,
    download_dataset,
    interim_data_present,
    load_config,
    processed_data_present,
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

CONFIG: DatasetConfig | dict[str, object] = {
    "expected_rows": 4,
    "fold_column": "fold",
    "label_column": "class",
    "fold_range": [1, 10],
    "expected_num_classes": 2,
}


def _table(rows: dict[str, list[object]]) -> ds.Dataset:
    return ds.dataset(pa.Table.from_pydict(rows))


def test_validate_schema_accepts_matching_data() -> None:
    table: ds.Dataset = _table(
        {
            "fold": [1, 2, 3, 4],
            "class": ["dog_bark", "siren", "dog_bark", "siren"],
        }
    )
    stats: SchemaValidationResult = validate_schema(table, CONFIG)
    assert stats == {"num_rows": 4, "num_classes": 2}


def test_validate_schema_rejects_wrong_row_count() -> None:
    table: ds.Dataset = _table(
        {"fold": [1, 2, 3], "class": ["dog_bark", "siren", "dog_bark"]}
    )
    with pytest.raises(ValueError, match="Row count mismatch"):
        validate_schema(table, CONFIG)


def test_validate_schema_rejects_missing_fold_column() -> None:
    table: ds.Dataset = _table(
        {
            "not_fold": [1, 2, 3, 4],
            "class": ["dog_bark", "siren", "dog_bark", "siren"],
        }
    )
    with pytest.raises(ValueError, match="Missing expected column 'fold'"):
        validate_schema(table, CONFIG)


def test_validate_schema_rejects_fold_out_of_range() -> None:
    table: ds.Dataset = _table(
        {
            "fold": [1, 2, 3, 11],
            "class": ["dog_bark", "siren", "dog_bark", "siren"],
        }
    )
    with pytest.raises(ValueError, match="Fold values out of expected range"):
        validate_schema(table, CONFIG)


def test_validate_schema_rejects_wrong_class_count() -> None:
    table: ds.Dataset = _table(
        {
            "fold": [1, 2, 3, 4],
            "class": ["dog_bark", "siren", "gun_shot", "engine_idling"],
        }
    )
    with pytest.raises(ValueError, match="Expected 2 classes"):
        validate_schema(table, CONFIG)


def test_normalize_targets_defaults_and_rejects_unknown() -> None:
    assert _normalize_targets(None) == ["raw"]
    assert _normalize_targets(["raw", "interim", "raw"]) == ["raw", "interim"]
    assert _normalize_targets(["processed", "processed"]) == ["processed"]
    assert _normalize_targets(["raw", "interim", "processed"]) == [
        "raw",
        "interim",
        "processed",
    ]
    with pytest.raises(ValueError, match="Unknown download target"):
        _normalize_targets(["unknown"])


def test_raw_data_present_requires_manifest_and_parquet(tmp_path: Path) -> None:
    assert raw_data_present(tmp_path) is False

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "part.parquet").write_bytes(b"x")
    assert raw_data_present(tmp_path) is False

    (tmp_path / ".manifest.json").write_text("{}")
    assert raw_data_present(tmp_path) is True


def test_interim_data_present_requires_metadata_and_wav(tmp_path: Path) -> None:
    assert interim_data_present(tmp_path) is False

    (tmp_path / "metadata.parquet").write_bytes(b"x")
    assert interim_data_present(tmp_path) is False

    audio: Path = tmp_path / "audio" / "fold1"
    audio.mkdir(parents=True)
    assert interim_data_present(tmp_path) is False

    (audio / "clip.wav").write_bytes(b"RIFF")
    assert interim_data_present(tmp_path) is True


def test_processed_data_present_requires_both_parquets(tmp_path: Path) -> None:
    assert processed_data_present(tmp_path) is False

    (tmp_path / "tabular.parquet").write_bytes(b"x")
    assert processed_data_present(tmp_path) is False

    (tmp_path / "mels.parquet").write_bytes(b"y")
    assert processed_data_present(tmp_path) is True


def test_versioning_push_enabled(tmp_path: Path) -> None:
    cfg: Path = tmp_path / "config.yaml"
    write_app_config(
        cfg,
        dataset={"hf_repo_id": "example/repo"},
        versioning={
            "push_interim": True,
            "push_processed": False,
            "push_models": True,
        },
    )
    assert versioning_push_enabled("push_interim", config_path=cfg) is True
    assert versioning_push_enabled("push_processed", config_path=cfg) is False
    assert versioning_push_enabled("push_models", config_path=cfg) is True
    assert versioning_push_enabled("missing_key", config_path=cfg) is False


def test_config_enabled_section_flags(tmp_path: Path) -> None:
    cfg: Path = tmp_path / "config.yaml"
    write_app_config(
        cfg,
        preprocessing={"enabled": False},
        spark={"enabled": True},
    )
    assert config_enabled("preprocessing", config_path=cfg) is False
    assert config_enabled("spark", config_path=cfg) is True
    assert config_enabled("missing_section", config_path=cfg) is True


def test_resolve_model_repo_id(tmp_path: Path) -> None:
    cfg: Path = tmp_path / "config.yaml"
    write_app_config(
        cfg,
        dataset={"hf_repo_id": "example/ds"},
        versioning={"hf_model_repo_id": "org/models", "hf_model_repo_type": "model"},
    )
    assert resolve_model_repo_id(config_path=cfg) == ("org/models", "model")


def test_resolve_model_repo_id_requires_id(tmp_path: Path) -> None:
    cfg: Path = tmp_path / "config.yaml"
    write_app_config(
        cfg,
        dataset={"hf_repo_id": "example/ds"},
        versioning={"path_in_repo": "interim"},
    )
    with pytest.raises(ValueError, match="hf_model_repo_id"):
        resolve_model_repo_id(config_path=cfg)


def test_push_all_trained_models_requires_each_dir(tmp_path: Path) -> None:
    (tmp_path / "rf").mkdir()
    with pytest.raises(FileNotFoundError):
        push_all_trained_models(tmp_path)


def test_push_winner_artifacts_requires_json_and_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        push_winner_artifacts(tmp_path)
    (tmp_path / "winner.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        push_winner_artifacts(tmp_path)


def test_push_all_trained_models_uploads_each_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg: Path = tmp_path / "config.yaml"
    write_app_config(
        cfg,
        dataset={"hf_repo_id": "example/ds"},
        versioning={"hf_model_repo_id": "org/models", "hf_model_repo_type": "model"},
    )
    models_dir: Path = tmp_path / "models"
    for name in TRAINED_MODEL_DIRS:
        (models_dir / name).mkdir(parents=True)
    uploaded: list[str] = []

    def _fake_push(local_dir: Path, path_in_repo: str, **_kwargs: Any) -> str:
        uploaded.append(path_in_repo)
        return "ok"

    monkeypatch.setattr("src.data_processing.versioning.push_model_tree", _fake_push)
    push_all_trained_models(models_dir, config_path=cfg)
    assert uploaded == list(TRAINED_MODEL_DIRS)


def test_pull_winner_artifacts_downloads_allow_patterns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg: Path = tmp_path / "config.yaml"
    write_app_config(
        cfg,
        dataset={"hf_repo_id": "example/ds"},
        versioning={"hf_model_repo_id": "org/models", "hf_model_repo_type": "model"},
    )
    dest: Path = tmp_path / "models"
    called: dict[str, object] = {}

    def _fake_snapshot(**kwargs: Any) -> None:
        called.update(kwargs)
        Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "src.data_processing.versioning.snapshot_download", _fake_snapshot
    )
    pull_winner_artifacts(dest, config_path=cfg)
    assert called["repo_id"] == "org/models"
    assert called["allow_patterns"] == ["winner.json", "winner/**"]
    assert dest.is_dir()


def test_dag_project_root_is_repo_root() -> None:
    dag_file: Path = (
        Path(__file__).resolve().parents[1]
        / "airflow"
        / "dags"
        / "audio_classification_dag.py"
    )
    project_root: Path = dag_file.parents[2]
    assert (project_root / "src" / "models" / "train.py").is_file()
    assert (project_root / "config" / "config.yaml").is_file()


@pytest.mark.integration
def test_download_dataset_end_to_end(tmp_path: Path) -> None:
    config: DatasetConfig = load_config()
    config["local_raw_dir"] = str(tmp_path)
    config["expected_rows"] = 546
    config["expected_num_classes"] = 10
    manifest: dict[str, object] = download_dataset(
        config,
        allow_patterns=["data/train-00000-of-00016-e478d7cccca6a095.parquet"],
    )
    assert manifest["num_rows"] == config["expected_rows"]
