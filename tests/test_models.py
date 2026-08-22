"""Smoke tests for model data splits, CNN forward, and metrics."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data_pipeline.spark_feature_extractor import tabular_feature_names
from src.models.cnn_model import build_resnet18, mel_to_3ch
from src.models.data import (
    build_label_maps,
    class_weight_vector,
    feature_columns,
    filter_split,
    fit_mel_stats,
    fit_scaler,
    normalize_mels,
    sample_weights_from_y,
    split_frames,
    tabular_xy,
    transform_features,
)
from src.models.evaluate import compute_metrics
from src.models.runtime_env import _THREAD_DEFAULTS, load_runtime_env
from src.models.train import ALL_MODELS, run_training


def _synthetic_tabular(n_per_fold: int = 4) -> pd.DataFrame:
    feat_cols = tabular_feature_names(n_mfcc=13)
    rows = []
    classes = [f"c{i}" for i in range(10)]
    for fold in range(1, 11):
        for i in range(n_per_fold):
            row = {
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
                aug = dict(row)
                aug["path"] = f"audio/fold{fold}/x{i}_aug0.wav"
                aug["is_augmented"] = True
                aug["aug_index"] = 0
                for c in feat_cols:
                    aug[c] = float(fold + i) + 0.1
                rows.append(aug)
    return pd.DataFrame(rows)


def test_filter_split_excludes_augments_on_val_eval():
    frame = _synthetic_tabular()
    val = filter_split(frame, [9], include_augmented=False)
    assert val["is_augmented"].astype(bool).sum() == 0
    assert set(val["fold"].unique()) == {9}

    train = filter_split(frame, [1, 2], include_augmented=True)
    assert train["is_augmented"].astype(bool).sum() > 0


def test_split_frames_shapes():
    frame = _synthetic_tabular()
    optuna_train, val, refit, eval_df = split_frames(
        frame, list(range(1, 9)), 9, 10
    )
    assert set(optuna_train["fold"].unique()) <= set(range(1, 9))
    assert set(val["fold"].unique()) == {9}
    assert val["is_augmented"].astype(bool).sum() == 0
    assert set(eval_df["fold"].unique()) == {10}
    assert 9 in set(refit["fold"].unique())


def test_scaler_fit_on_train_only():
    frame = _synthetic_tabular()
    optuna_train, val, _, _ = split_frames(frame, list(range(1, 9)), 9, 10)
    label_to_id, _ = build_label_maps(optuna_train["class"])
    feat_cols = feature_columns(frame, n_mfcc=13)
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


def test_mel_stats_and_normalize():
    mels = np.random.randn(8, 128, 126).astype(np.float32)
    stats = fit_mel_stats(mels)
    normed = normalize_mels(mels, stats)
    assert abs(float(np.mean(normed))) < 1e-5
    assert abs(float(np.std(normed)) - 1.0) < 1e-3


def test_mel_to_3ch_and_resnet_forward():
    mel = np.random.randn(128, 126).astype(np.float32)
    stacked = mel_to_3ch(mel)
    assert stacked.shape == (3, 128, 126)

    model = build_resnet18(n_classes=10, pretrained=False)
    model.eval()
    x = torch.from_numpy(stacked).unsqueeze(0)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    logits = model(x)
    assert logits.shape == (1, 10)


def test_collect_dataset_lineage_reads_manifests(tmp_path):
    from src.models.data import collect_dataset_lineage

    processed = tmp_path / "processed"
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


def test_compute_metrics_keys():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    proba = np.eye(3)[y_pred]
    metrics = compute_metrics(y_true, y_pred, y_proba=proba, labels=[0, 1, 2])
    for key in (
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "roc_auc_ovr",
    ):
        assert key in metrics


def test_load_runtime_env_sets_thread_defaults(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OMP_NUM_THREADS=2\n")
    for key in _THREAD_DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    load_runtime_env(env_path=env_file)
    assert os.environ["OMP_NUM_THREADS"] == "2"
    for key, value in _THREAD_DEFAULTS.items():
        if key == "OMP_NUM_THREADS":
            continue
        assert os.environ.get(key) == value


def test_run_training_import_and_all_models():
    assert "rf" in ALL_MODELS
    assert "resnet18" in ALL_MODELS
    assert callable(run_training)
