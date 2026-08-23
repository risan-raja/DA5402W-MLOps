from __future__ import annotations

import io
from pathlib import Path

import joblib
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data_pipeline.audio_features import tabular_feature_names
from src.deployment.app import app
from src.models.mlflow_logging import save_json

CLASS_NAMES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]
REPO_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def _wav_bytes(sr: int = 16000, seconds: float = 0.25) -> bytes:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    y = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


def _write_tabular_winner(models_dir: Path) -> None:
    names = tabular_feature_names(n_mfcc=13)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, len(names))).astype(np.float32)
    y = np.resize(np.arange(len(CLASS_NAMES)), 40)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(max_iter=200).fit(scaler.transform(x), y)
    dest = models_dir / "rf"
    dest.mkdir(parents=True)
    joblib.dump(model, dest / "model.joblib")
    joblib.dump(scaler, dest / "scaler.joblib")
    save_json(
        dest / "label_map.json",
        {
            "label_to_id": {name: idx for idx, name in enumerate(CLASS_NAMES)},
            "id_to_label": {str(idx): name for idx, name in enumerate(CLASS_NAMES)},
        },
    )
    save_json(models_dir / "winner.json", {"model_name": "rf", "f1_macro": 0.5})


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DA5402W_CONFIG", str(REPO_CONFIG))
    monkeypatch.setenv("DA5402W_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("DA5402W_MODEL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("DA5402W_PULL_WINNER", "0")
    (tmp_path / "models").mkdir()
    return tmp_path


def test_health_ok_without_model(isolated_env: Path) -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        missing = client.get("/")
        metrics = client.get("/metrics")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert missing.status_code == 404
    assert metrics.status_code == 200
    assert "predict_requests_total" in metrics.text


def test_predict_503_without_model(isolated_env: Path) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
        )
    assert response.status_code == 503


def test_predict_rejects_non_wav(isolated_env: Path) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("notes.txt", b"not audio", "text/plain")},
        )
    assert response.status_code == 400
    assert "wav" in response.json()["detail"].lower()


def test_predict_rejects_oversize(isolated_env: Path) -> None:
    payload = b"RIFF" + b"\x00" * (8 * 1024 * 1024)
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("huge.wav", payload, "audio/wav")},
        )
    assert response.status_code == 413


def test_predict_tabular_winner_contract(isolated_env: Path) -> None:
    _write_tabular_winner(isolated_env / "models")
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
        )
        metrics = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in CLASS_NAMES
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_name"] == "rf"
    assert body["latency_ms"] >= 0.0
    assert set(body["probabilities"]) == set(CLASS_NAMES)
    assert "predict_requests_total" in metrics.text
    assert "predict_latency_seconds" in metrics.text
    assert "predict_class_total" in metrics.text
    assert "predict_confidence" in metrics.text
    assert f'class_name="{body["label"]}"' in metrics.text
