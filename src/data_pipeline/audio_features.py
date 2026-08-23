"""Spark-free audio helpers shared by feature extraction and the serving API."""

from __future__ import annotations

import io
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def tabular_feature_names(n_mfcc: int = 13) -> list[str]:
    names: list[str] = []
    for i in range(n_mfcc):
        names.extend(
            [
                f"mfcc_{i}_mean",
                f"mfcc_{i}_std",
                f"mfcc_delta_{i}_mean",
                f"mfcc_delta_{i}_std",
            ]
        )
    for i in range(12):
        names.extend([f"chroma_{i}_mean", f"chroma_{i}_std"])
    for base in (
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
        "zcr",
    ):
        names.extend([f"{base}_mean", f"{base}_std"])
    return names


def pad_or_truncate(y: np.ndarray, sample_rate: int, duration_sec: float) -> np.ndarray:
    target = int(sample_rate * duration_sec)
    if len(y) < target:
        return np.pad(y, (0, target - len(y)))
    return y[:target]


def _mean_std_feats(prefix: str, matrix: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for i in range(matrix.shape[0]):
        out[f"{prefix}_{i}_mean"] = float(np.mean(matrix[i]))
        out[f"{prefix}_{i}_std"] = float(np.std(matrix[i]))
    return out


def extract_tabular_features(
    y: np.ndarray, sample_rate: int, n_mfcc: int = 13
) -> dict[str, float]:
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sample_rate)
    feats: dict[str, float] = {}
    feats.update(_mean_std_feats("mfcc", mfcc))
    feats.update(_mean_std_feats("mfcc_delta", mfcc_delta))
    feats.update(_mean_std_feats("chroma", chroma))
    for name, arr in (
        ("spectral_centroid", librosa.feature.spectral_centroid(y=y, sr=sample_rate)),
        ("spectral_bandwidth", librosa.feature.spectral_bandwidth(y=y, sr=sample_rate)),
        ("spectral_rolloff", librosa.feature.spectral_rolloff(y=y, sr=sample_rate)),
        ("zcr", librosa.feature.zero_crossing_rate(y)),
    ):
        feats[f"{name}_mean"] = float(np.mean(arr))
        feats[f"{name}_std"] = float(np.std(arr))
    return feats


def extract_log_mel(
    y: np.ndarray,
    sample_rate: int,
    *,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    mel_frames: int,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    if log_mel.shape[1] < mel_frames:
        pad = mel_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)))
    elif log_mel.shape[1] > mel_frames:
        log_mel = log_mel[:, :mel_frames]
    return log_mel


def normalize_mels(mels: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    return ((mels - stats["mean"]) / stats["std"]).astype(np.float32)


def decode_wav(source: Path | str | bytes) -> tuple[np.ndarray, int]:
    """Read a wav from a path or byte string. Mix to mono. Raise on empty audio."""
    if isinstance(source, bytes):
        y, sr = sf.read(io.BytesIO(source), always_2d=False)
    else:
        y, sr = sf.read(str(source), always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if y.size == 0:
        raise ValueError("empty audio")
    return y, int(sr)


def prepare_waveform(
    y: np.ndarray,
    native_sr: int,
    *,
    sample_rate: int,
    duration_sec: float,
) -> np.ndarray:
    """Resample to ``sample_rate`` and pad/truncate to ``duration_sec``."""
    y = np.asarray(y, dtype=np.float32)
    if native_sr != sample_rate:
        y = librosa.resample(y, orig_sr=native_sr, target_sr=sample_rate)
    return pad_or_truncate(y, sample_rate, duration_sec)
