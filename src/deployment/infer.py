"""Run a single-clip prediction against the loaded winner."""

from __future__ import annotations

import numpy as np
import torch

from src.data_pipeline.audio_features import (
    decode_wav,
    extract_log_mel,
    extract_tabular_features,
    normalize_mels,
    prepare_waveform,
    tabular_feature_names,
)
from src.deployment.runtime import LoadedModel
from src.models.cnn_model import predict_proba as cnn_predict_proba


def _spark_cfg(config: dict) -> dict:
    spark = config.get("spark") or {}
    return {
        "sample_rate": int(spark.get("sample_rate", 16000)),
        "target_duration_sec": float(spark.get("target_duration_sec", 4.0)),
        "n_mfcc": int(spark.get("n_mfcc", 13)),
        "n_mels": int(spark.get("n_mels", 128)),
        "n_fft": int(spark.get("n_fft", 1024)),
        "hop_length": int(spark.get("hop_length", 512)),
        "mel_frames": int(spark.get("mel_frames", 126)),
    }


def _waveform(wav_bytes: bytes, spark_cfg: dict) -> np.ndarray:
    y, native_sr = decode_wav(wav_bytes)
    return prepare_waveform(
        y,
        native_sr,
        sample_rate=int(spark_cfg["sample_rate"]),
        duration_sec=float(spark_cfg["target_duration_sec"]),
    )


def _cnn_proba(loaded: LoadedModel, y: np.ndarray, spark_cfg: dict) -> np.ndarray:
    if loaded.torch_model is None or loaded.mel_stats is None:
        raise RuntimeError("CNN artifacts are not loaded")
    log_mel = extract_log_mel(
        y,
        int(spark_cfg["sample_rate"]),
        n_mels=int(spark_cfg["n_mels"]),
        n_fft=int(spark_cfg["n_fft"]),
        hop_length=int(spark_cfg["hop_length"]),
        mel_frames=int(spark_cfg["mel_frames"]),
    )
    mels = normalize_mels(log_mel[np.newaxis, ...], loaded.mel_stats)
    return cnn_predict_proba(
        loaded.torch_model,
        mels,
        batch_size=1,
        device=torch.device("cpu"),
    )


def _tabular_proba(loaded: LoadedModel, y: np.ndarray, spark_cfg: dict) -> np.ndarray:
    if loaded.sklearn_model is None or loaded.scaler is None:
        raise RuntimeError("tabular artifacts are not loaded")
    feats = extract_tabular_features(
        y, int(spark_cfg["sample_rate"]), n_mfcc=int(spark_cfg["n_mfcc"])
    )
    names = tabular_feature_names(n_mfcc=int(spark_cfg["n_mfcc"]))
    row = np.array([[feats[name] for name in names]], dtype=np.float32)
    scaled = loaded.scaler.transform(row)
    return np.asarray(loaded.sklearn_model.predict_proba(scaled), dtype=np.float32)


def predict_clip(loaded: LoadedModel, wav_bytes: bytes, config: dict) -> dict:
    spark_cfg = _spark_cfg(config)
    y = _waveform(wav_bytes, spark_cfg)
    if loaded.family == "cnn":
        proba = _cnn_proba(loaded, y, spark_cfg)[0]
    elif loaded.family == "tabular":
        proba = _tabular_proba(loaded, y, spark_cfg)[0]
    else:
        raise ValueError(f"unsupported family {loaded.family!r}")
    class_ids = sorted(loaded.id_to_label)
    probabilities = {
        loaded.id_to_label[idx]: float(proba[idx])
        for idx in class_ids
        if idx < len(proba)
    }
    best_idx = int(np.argmax(proba))
    label = loaded.id_to_label[best_idx]
    return {
        "label": label,
        "confidence": float(proba[best_idx]),
        "model_name": loaded.model_name,
        "probabilities": probabilities,
    }
