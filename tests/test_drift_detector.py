from __future__ import annotations

import json
from pathlib import Path

from tests.config_helpers import write_app_config

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config_types import AppConfig
from src.deployment.app import app
from src.monitoring.drift_cli import (
    build_reference,
    ensure_prediction_jsonl,
    main,
    score_feature_file,
    score_prediction_file,
)
from src.monitoring.drift_detector import (
    CLASS_NAMES,
    OFFICIAL_FOLD10_COUNTS,
    DriftMonitor,
    DriftReference,
    DriftSnapshot,
    FeatureHistogram,
    build_feature_histograms,
    dump_reference,
    ks_statistic,
    official_fold10_reference,
    population_stability_index,
    proportions_from_counts,
    score_features,
    score_predictions,
    with_confidence_samples,
    with_feature_histograms,
)
from tests.test_api import _wav_bytes, _write_tabular_winner

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def test_psi_zero_when_distributions_match() -> None:
    prior: dict[str, float] = proportions_from_counts(
        OFFICIAL_FOLD10_COUNTS, CLASS_NAMES
    )
    assert population_stability_index(prior, prior) == pytest.approx(0.0, abs=1e-12)


def test_psi_rises_when_one_class_dominates() -> None:
    prior: dict[str, float] = proportions_from_counts(
        OFFICIAL_FOLD10_COUNTS, CLASS_NAMES
    )
    skewed: dict[str, float] = {
        name: (1.0 if name == "siren" else 0.0) for name in CLASS_NAMES
    }
    assert population_stability_index(prior, skewed) > 1.0


def test_ks_zero_for_identical_samples() -> None:
    samples: list[float] = [0.1, 0.4, 0.7, 0.9, 0.95]
    assert ks_statistic(samples, samples) == pytest.approx(0.0)


def test_ks_larger_for_shifted_confidences() -> None:
    rng = np.random.default_rng(0)
    ref: np.ndarray = rng.beta(8, 2, size=200)
    live: np.ndarray = rng.beta(2, 8, size=200)
    assert ks_statistic(ref, live) > 0.4


def test_monitor_none_until_window_full() -> None:
    reference: DriftReference = official_fold10_reference()
    monitor: DriftMonitor = DriftMonitor(window_size=3, reference=reference)
    assert monitor.update("dog_bark", 0.9) is None
    assert monitor.update("siren", 0.8) is None
    assert monitor.filled == 2
    snapshot: DriftSnapshot | None = monitor.update("dog_bark", 0.7)
    assert snapshot is not None
    assert snapshot.n_observations == 3
    assert snapshot.psi_class > 0.0
    assert snapshot.ks_confidence is None


def test_monitor_confidence_ks_when_quantiles_present() -> None:
    reference: DriftReference = with_confidence_samples(
        official_fold10_reference(),
        [0.9, 0.92, 0.88, 0.95, 0.91],
    )
    monitor: DriftMonitor = DriftMonitor(window_size=2, reference=reference)
    assert monitor.update("dog_bark", 0.2) is None
    snapshot: DriftSnapshot | None = monitor.update("dog_bark", 0.1)
    assert snapshot is not None
    assert snapshot.ks_confidence is not None
    assert snapshot.ks_confidence > 0.5


def test_score_predictions_batch() -> None:
    reference: DriftReference = official_fold10_reference()
    labels: list[str] = ["dog_bark"] * 50 + ["siren"] * 50
    snapshot: DriftSnapshot = score_predictions(labels, [0.8] * 100, reference)
    assert snapshot.n_observations == 100
    assert snapshot.psi_class > 0.5


def test_feature_psi_near_zero_on_same_draw() -> None:
    rng = np.random.default_rng(1)
    values: np.ndarray = rng.normal(size=400)
    histograms: dict[str, FeatureHistogram] = build_feature_histograms(
        {"mfcc_0_mean": values}, n_bins=8
    )
    scored: dict[str, float | str] = score_features(
        histograms, {"mfcc_0_mean": values}
    )
    assert scored["mean_psi"] < 0.05
    assert scored["max_psi_feature"] == "mfcc_0_mean"


def test_cli_build_reference_from_processed(tmp_path: Path) -> None:
    frame: pd.DataFrame = pd.DataFrame(
        {
            "fold": [10, 10, 9, 10],
            "class": ["dog_bark", "siren", "drilling", "dog_bark"],
            "is_augmented": [False, False, False, True],
            "mfcc_0_mean": [1.0, 2.0, 3.0, 4.0],
        }
    )
    processed: Path = tmp_path / "processed"
    processed.mkdir()
    frame.to_parquet(processed / "tabular.parquet")
    config: AppConfig | dict[str, object] = {
        "dataset": {"hf_repo_id": "test/repo"},
        "training": {"eval_fold": 10, "models_dir": str(tmp_path / "models")},
        "spark": {"local_processed_dir": str(processed), "n_mfcc": 13},
        "preprocessing": {"local_interim_dir": str(tmp_path / "interim")},
        "monitoring": {"drift": {"n_feature_bins": 4}},
    }
    reference: DriftReference = build_reference(config, with_features=True)
    assert reference.class_counts["dog_bark"] == 1
    assert reference.class_counts["siren"] == 1
    assert reference.has_features
    dest: Path = tmp_path / "drift_reference.json"
    dump_reference(dest, reference)
    loaded: DriftReference = DriftReference.from_dict(json.loads(dest.read_text()))
    assert loaded.class_counts["dog_bark"] == 1


def test_cli_score_predictions_jsonl(tmp_path: Path) -> None:
    reference: DriftReference = official_fold10_reference()
    ref_path: Path = tmp_path / "ref.json"
    dump_reference(ref_path, reference)
    log_path: Path = tmp_path / "preds.jsonl"
    rows: list[dict[str, object]] = [
        {"event": "prediction", "status": "ok", "label": "siren", "confidence": 0.9},
        {"event": "prediction", "status": "ok", "label": "siren", "confidence": 0.8},
        {
            "event": "prediction",
            "status": "http_400",
            "label": None,
            "confidence": None,
        },
    ]
    log_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    scored: dict[str, object] = score_prediction_file(log_path, reference)
    assert scored["n_observations"] == 2
    assert scored["psi_class"] > 0.0
    assert scored["ks_confidence"] is None


def test_cli_writes_missing_prediction_jsonl(tmp_path: Path) -> None:
    config: AppConfig | dict[str, object] = {
        "dataset": {"hf_repo_id": "test/repo"},
        "training": {"eval_fold": 10, "models_dir": str(tmp_path / "models")},
        "spark": {"local_processed_dir": str(tmp_path / "processed")},
        "preprocessing": {"local_interim_dir": str(tmp_path / "interim")},
    }
    path: Path = tmp_path / "predictions.jsonl"
    with pytest.warns(UserWarning, match="was missing"):
        ensure_prediction_jsonl(path, config)
    rows: list[dict[str, object]] = [
        json.loads(line) for line in path.read_text().splitlines() if line
    ]
    assert len(rows) == 837
    assert rows[0]["event"] == "prediction"
    scored: dict[str, object] = score_prediction_file(
        path, official_fold10_reference()
    )
    assert scored["n_observations"] == 837
    assert scored["psi_class"] == pytest.approx(0.0, abs=1e-12)


def test_cli_score_features(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    ref_values: np.ndarray = rng.normal(loc=0.0, scale=1.0, size=200)
    live_values: np.ndarray = rng.normal(loc=3.0, scale=1.0, size=80)
    histograms: dict[str, FeatureHistogram] = build_feature_histograms(
        {"mfcc_0_mean": ref_values}, n_bins=8
    )
    reference: DriftReference = with_feature_histograms(
        official_fold10_reference(), histograms
    )
    ref_path: Path = tmp_path / "ref.json"
    dump_reference(ref_path, reference)
    frame: pd.DataFrame = pd.DataFrame(
        {"mfcc_0_mean": live_values, "fold": [10] * 80}
    )
    parquet: Path = tmp_path / "live.parquet"
    frame.to_parquet(parquet)
    scored: dict[str, float | str] = score_feature_file(parquet, reference, n_mfcc=13)
    assert float(scored["mean_psi"]) > 0.2


def test_cli_score_features_accepts_with_features(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rng = np.random.default_rng(3)
    frame: pd.DataFrame = pd.DataFrame(
        {
            "fold": [10] * 40,
            "class": (["dog_bark"] * 20) + (["siren"] * 20),
            "is_augmented": [False] * 40,
            "mfcc_0_mean": rng.normal(size=40),
        }
    )
    processed: Path = tmp_path / "processed"
    processed.mkdir()
    parquet: Path = processed / "tabular.parquet"
    frame.to_parquet(parquet)
    config_path: Path = tmp_path / "config.yaml"
    write_app_config(
        config_path,
        dataset={"hf_repo_id": "test/repo"},
        training={
            "eval_fold": 10,
            "models_dir": str(tmp_path / "models"),
        },
        spark={
            "local_processed_dir": str(processed),
            "n_mfcc": 13,
        },
        preprocessing={"local_interim_dir": str(tmp_path / "interim")},
        monitoring={
            "drift": {
                "reference_path": str(tmp_path / "drift_reference.json"),
                "n_feature_bins": 4,
            }
        },
    )
    dump_reference(tmp_path / "drift_reference.json", official_fold10_reference())
    code: int = main(
        [
            "--config",
            str(config_path),
            "score-features",
            str(parquet),
            "--with-features",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    payload: dict[str, object] = json.loads(captured.out)
    assert int(payload["n_features"]) == 1
    loaded: DriftReference = DriftReference.from_dict(
        json.loads((tmp_path / "drift_reference.json").read_text())
    )
    assert loaded.has_features


def test_predict_updates_drift_gauges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DA5402W_CONFIG", str(REPO_CONFIG))
    monkeypatch.setenv("DA5402W_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("DA5402W_MODEL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("DA5402W_PULL_WINNER", "0")
    monkeypatch.setenv("DA5402W_DRIFT_WINDOW", "2")
    models_dir: Path = tmp_path / "models"
    models_dir.mkdir()
    _write_tabular_winner(models_dir)
    dump_reference(models_dir / "drift_reference.json", official_fold10_reference())
    wav: bytes = _wav_bytes()
    with TestClient(app) as client:
        first = client.post("/predict", files={"file": ("a.wav", wav, "audio/wav")})
        second = client.post("/predict", files={"file": ("b.wav", wav, "audio/wav")})
        metrics = client.get("/metrics")
    assert first.status_code == 200
    assert second.status_code == 200
    text: str = metrics.text
    assert "drift_psi_class" in text
    assert "drift_window_size" in text
