from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from src.data_pipeline.spark_feature_extractor import tabular_feature_names
from src.models.cnn_model import build_resnet18, mel_to_3ch
from src.models.cross_validation import aggregate_fold_metrics
from src.models.data import (
    build_label_maps,
    class_weight_vector,
    feature_columns,
    filter_split,
    fit_mel_stats,
    fit_scaler,
    iter_us8k_cv_folds,
    normalize_mels,
    sample_weights_from_y,
    split_frames,
    tabular_xy,
    transform_features,
)
from src.models.evaluate import compute_metrics
from src.models.runtime_env import _THREAD_DEFAULTS, load_runtime_env
from src.models.train import (
    materialize_winner_dir,
    persist_run_result,
    result_score,
    select_winner,
    select_winner_from_artifacts,
    winner_payload,
)
from tests.config_helpers import write_app_config


def _synthetic_tabular(n_per_fold: int = 4) -> pd.DataFrame:
    feat_cols: list[str] = tabular_feature_names(n_mfcc=13)
    rows: list[dict[str, object]] = []
    classes: list[str] = [f"c{i}" for i in range(10)]
    for fold in range(1, 11):
        for i in range(n_per_fold):
            row: dict[str, object] = {
                "path": f"audio/fold{fold}/x{i}.wav",
                "slice_file_name": f"x{i}.wav",
                "fold": fold,
                "class": classes[i % 10],
                "classID": i % 10,
                "is_augmented": False,
                "aug_index": -1,
            }
            for c in feat_cols:
                row[c] = float(fold + i)
            rows.append(row)
            if fold <= 8:
                aug: dict[str, object] = dict(row)
                aug["path"] = f"audio/fold{fold}/x{i}_aug0.wav"
                aug["is_augmented"] = True
                aug["aug_index"] = 0
                for c in feat_cols:
                    aug[c] = float(fold + i) + 0.1
                rows.append(aug)
    return pd.DataFrame(rows)


def test_filter_split_excludes_augments_on_val_eval() -> None:
    frame: pd.DataFrame = _synthetic_tabular()
    val: pd.DataFrame = filter_split(frame, [9], include_augmented=False)
    assert val["is_augmented"].astype(bool).sum() == 0
    assert set(val["fold"].unique()) == {9}

    train: pd.DataFrame = filter_split(frame, [1, 2], include_augmented=True)
    assert train["is_augmented"].astype(bool).sum() > 0


def test_split_frames_shapes() -> None:
    frame: pd.DataFrame = _synthetic_tabular()
    optuna_train, val, refit, eval_df = split_frames(frame, list(range(1, 9)), 9, 10)
    assert set(optuna_train["fold"].unique()) <= set(range(1, 9))
    assert set(val["fold"].unique()) == {9}
    assert val["is_augmented"].astype(bool).sum() == 0
    assert set(eval_df["fold"].unique()) == {10}
    assert 9 in set(refit["fold"].unique())


def test_scaler_fit_on_train_only() -> None:
    frame: pd.DataFrame = _synthetic_tabular()
    optuna_train, val, _, _ = split_frames(frame, list(range(1, 9)), 9, 10)
    label_to_id, _ = build_label_maps(optuna_train["class"])
    feat_cols: list[str] = feature_columns(frame, n_mfcc=13)
    x_tr, y_tr = tabular_xy(optuna_train, feat_cols, label_to_id)
    x_va, _ = tabular_xy(val, feat_cols, label_to_id)
    scaler = fit_scaler(x_tr)
    x_tr_s = transform_features(scaler, x_tr)
    x_va_s = transform_features(scaler, x_va)
    assert x_tr_s.shape == x_tr.shape
    assert x_va_s.shape == x_va.shape
    assert abs(float(np.mean(x_tr_s))) < 1e-5
    cw = class_weight_vector(y_tr, n_classes=len(label_to_id))
    sw = sample_weights_from_y(y_tr, cw)
    assert sw.shape == y_tr.shape


def test_mel_stats_and_normalize() -> None:
    mels: np.ndarray = np.random.randn(8, 128, 126).astype(np.float32)
    stats = fit_mel_stats(mels)
    normed: np.ndarray = normalize_mels(mels, stats)
    assert abs(float(np.mean(normed))) < 1e-5
    assert abs(float(np.std(normed)) - 1.0) < 1e-3


def test_mel_to_3ch_and_resnet_forward() -> None:
    mel: np.ndarray = np.random.randn(128, 126).astype(np.float32)
    stacked: np.ndarray = mel_to_3ch(mel)
    assert stacked.shape == (3, 128, 126)

    model = build_resnet18(n_classes=10, pretrained=False)
    model.eval()
    x = torch.from_numpy(stacked).unsqueeze(0)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    logits = model(x)
    assert logits.shape == (1, 10)


def test_collect_dataset_lineage_reads_manifests(tmp_path: Path) -> None:
    from src.models.data import collect_dataset_lineage

    processed: Path = tmp_path / "processed"
    processed.mkdir()
    (processed / ".manifest.json").write_text(
        '{"created_at": "2026-01-01T00:00:00+00:00", "num_written": 10}'
    )
    (processed / "tabular.parquet").write_bytes(b"abc")
    lineage = collect_dataset_lineage(processed)
    assert lineage["processed_created_at"] == "2026-01-01T00:00:00+00:00"
    assert lineage["processed_num_rows"] == 10
    assert lineage["tabular"]["exists"] is True
    assert lineage["tabular"]["sha256"]


def test_compute_metrics_keys() -> None:
    y_true: np.ndarray = np.array([0, 1, 2, 0, 1, 2])
    y_pred: np.ndarray = np.array([0, 1, 1, 0, 2, 2])
    proba: np.ndarray = np.eye(3)[y_pred]
    metrics: dict[str, float] = compute_metrics(
        y_true, y_pred, y_proba=proba, labels=[0, 1, 2]
    )
    for key in (
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "roc_auc_ovr",
    ):
        assert key in metrics


def test_load_runtime_env_sets_thread_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file: Path = tmp_path / ".env"
    env_file.write_text("OMP_NUM_THREADS=2\n")
    for key in _THREAD_DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    load_runtime_env(env_path=env_file)
    assert os.environ["OMP_NUM_THREADS"] == "2"
    for key, value in _THREAD_DEFAULTS.items():
        if key == "OMP_NUM_THREADS":
            continue
        assert os.environ.get(key) == value


def test_register_dataset_input_retries_unique_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    from mlflow.exceptions import MlflowException

    from src.models.mlflow_logging import register_dataset_input

    calls = {"n": 0}

    def _log_input(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise MlflowException(
                "UNIQUE constraint failed: datasets.experiment_id, "
                "datasets.name, datasets.digest"
            )

    monkeypatch.setattr("src.models.mlflow_logging.mlflow.log_input", _log_input)
    monkeypatch.setattr("src.models.mlflow_logging.time.sleep", lambda *_: None)
    register_dataset_input(object())
    assert calls["n"] == 3


def test_register_dataset_input_continues_after_persistent_unique(monkeypatch: pytest.MonkeyPatch) -> None:
    from mlflow.exceptions import MlflowException

    from src.models.mlflow_logging import register_dataset_input

    calls = {"n": 0}

    def _log_input(*_args, **_kwargs):
        calls["n"] += 1
        raise MlflowException("UNIQUE constraint failed: datasets.digest")

    monkeypatch.setattr("src.models.mlflow_logging.mlflow.log_input", _log_input)
    monkeypatch.setattr("src.models.mlflow_logging.time.sleep", lambda *_: None)
    register_dataset_input(object(), attempts=2)
    assert calls["n"] == 2


def test_result_score_prefers_cv_mean() -> None:
    assert result_score({"metrics": {"f1_macro": 0.9, "cv_f1_macro_mean": 0.5}}) == 0.5
    assert result_score({"metrics": {"f1_macro": 0.8}}) == 0.8
    assert result_score({"metrics": {}}) == float("-inf")


def test_winner_payload_includes_cv_and_lineage() -> None:
    payload = winner_payload(
        {
            "model_name": "xgboost",
            "run_id": "abc",
            "metrics": {
                "f1_macro": 0.6,
                "cv_f1_macro_mean": 0.7,
                "cv_f1_macro_std": 0.01,
            },
        },
        {
            "hf_repo_id": "org/ds",
            "processed_created_at": "t0",
            "tabular": {"sha256": "deadbeef"},
        },
    )
    assert payload["model_name"] == "xgboost"
    assert payload["cv_f1_macro_mean"] == 0.7
    assert payload["dataset"]["tabular_sha256"] == "deadbeef"


def test_select_winner_writes_json_and_copies_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.models.train.tag_mlflow_winners", lambda *a, **k: None)
    (tmp_path / "rf").mkdir()
    (tmp_path / "rf" / "model.joblib").write_text("rf-weights")
    (tmp_path / "xgboost").mkdir()
    (tmp_path / "xgboost" / "model.joblib").write_text("xgb-weights")
    results = [
        {
            "model_name": "rf",
            "run_id": "a",
            "metrics": {"f1_macro": 0.4},
            "out_dir": str(tmp_path / "rf"),
        },
        {
            "model_name": "xgboost",
            "run_id": "b",
            "metrics": {"cv_f1_macro_mean": 0.7, "f1_macro": 0.6},
            "out_dir": str(tmp_path / "xgboost"),
        },
    ]
    lineage = {
        "hf_repo_id": "org/ds",
        "processed_created_at": "t0",
        "tabular": {"sha256": "abc"},
    }
    winner = select_winner(results, tmp_path, lineage)
    assert winner["model_name"] == "xgboost"
    written = (tmp_path / "winner.json").read_text()
    assert "xgboost" in written
    assert (tmp_path / "winner" / "model.joblib").read_text() == "xgb-weights"
    dest = materialize_winner_dir(tmp_path, "rf")
    assert dest.name == "winner"
    assert (dest / "model.joblib").read_text() == "rf-weights"


def test_select_winner_from_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.models.train.tag_mlflow_winners", lambda *a, **k: None)
    monkeypatch.setattr("src.models.train.ensure_mlflow", lambda cfg: None)
    models_dir = tmp_path / "models"
    lineage = {
        "hf_repo_id": "org/ds",
        "processed_created_at": "t0",
        "tabular": {"sha256": "abc"},
    }
    for name, score in (("rf", 0.2), ("xgboost", 0.9)):
        out = models_dir / name
        out.mkdir(parents=True)
        persist_run_result(
            {
                "model_name": name,
                "run_id": name,
                "metrics": {"f1_macro": score},
                "out_dir": str(out),
            },
            lineage,
        )
    cfg = tmp_path / "config.yaml"
    write_app_config(
        cfg,
        training={
            "models_dir": str(models_dir),
            "models": ["rf", "xgboost"],
            "mlflow_tracking_uri": "sqlite:///unused.db",
            "mlflow_artifact_root": str(tmp_path / "mlruns"),
            "processed_dir": "unused",
        },
    )
    winner = select_winner_from_artifacts(
        config_path=cfg, models=["rf", "xgboost"]
    )
    assert winner["model_name"] == "xgboost"
    assert (models_dir / "winner.json").is_file()


def test_train_resnet_history_has_val_metrics() -> None:
    """Optuna trials log these keys; history must return finite values."""
    from src.models.cnn_model import train_resnet

    rng = np.random.default_rng(0)
    n_cls = 3
    x_tr = rng.standard_normal((8, 32, 32), dtype=np.float32)
    y_tr = rng.integers(0, n_cls, size=8)
    x_va = rng.standard_normal((4, 32, 32), dtype=np.float32)
    y_va = rng.integers(0, n_cls, size=4)
    cw = np.ones(n_cls, dtype=np.float64)

    _, history = train_resnet(
        x_tr,
        y_tr,
        x_va,
        y_va,
        class_weights=cw,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        patience=1,
        seed=0,
        pretrained=False,
        early_stop=True,
    )
    for key in ("best_val_f1_macro", "best_val_accuracy", "best_val_f1_weighted"):
        value = history[key]
        assert not math.isnan(value)
        assert 0.0 <= value <= 1.0

def test_iter_us8k_cv_folds_disjoint() -> None:
    pairs = list(iter_us8k_cv_folds(10))
    assert len(pairs) == 10
    for test_fold, train_folds in pairs:
        assert test_fold not in train_folds
        assert len(train_folds) == 9
        assert set(train_folds) | {test_fold} == set(range(1, 11))


def test_aggregate_fold_metrics_mean_std() -> None:
    folds = [
        {"fold": 1, "f1_macro": 0.5, "accuracy": 0.6},
        {"fold": 2, "f1_macro": 0.7, "accuracy": 0.8},
    ]
    agg = aggregate_fold_metrics(folds)
    assert abs(agg["cv_f1_macro_mean"] - 0.6) < 1e-9
    assert abs(agg["cv_accuracy_mean"] - 0.7) < 1e-9
    assert agg["cv_f1_macro_std"] >= 0.0

