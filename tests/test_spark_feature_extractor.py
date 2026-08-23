from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.data_pipeline.audio_features import (
    extract_log_mel,
    extract_tabular_features,
    pad_or_truncate,
    tabular_feature_names,
)
from src.data_pipeline.spark_feature_extractor import (
    extract_clip_features,
    extract_features,
)


def _sine(sr: int = 16000, seconds: float = 4.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    return (0.25 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def test_tabular_feature_names_count() -> None:
    names = tabular_feature_names(n_mfcc=13)
    # 13*4 mfcc(+delta) + 12*2 chroma + 4*2 spectral/zcr
    assert len(names) == 13 * 4 + 12 * 2 + 4 * 2
    assert "mfcc_delta_0_mean" in names
    assert "zcr_std" in names


def test_pad_or_truncate_and_mel_shape() -> None:
    short = _sine(seconds=0.5)
    padded = pad_or_truncate(short, 16000, 4.0)
    assert len(padded) == 64000
    mel = extract_log_mel(
        padded,
        16000,
        n_mels=128,
        n_fft=1024,
        hop_length=512,
        mel_frames=126,
    )
    assert mel.shape == (128, 126)


def test_extract_tabular_features_keys() -> None:
    y = pad_or_truncate(_sine(), 16000, 4.0)
    feats = extract_tabular_features(y, 16000, n_mfcc=13)
    names = tabular_feature_names(13)
    assert set(feats) == set(names)
    assert all(np.isfinite(v) for v in feats.values())


def test_extract_clip_features(tmp_path: Path) -> None:
    wav = tmp_path / "clip.wav"
    sf.write(wav, _sine(), 16000)
    spark_cfg = {
        "sample_rate": 16000,
        "target_duration_sec": 4.0,
        "n_mfcc": 13,
        "n_mels": 128,
        "n_fft": 1024,
        "hop_length": 512,
        "mel_frames": 126,
    }
    meta = {
        "path": "audio/fold1/clip.wav",
        "slice_file_name": "clip.wav",
        "fold": 1,
        "class": "dog_bark",
        "classID": 3,
        "is_augmented": False,
        "aug_index": -1,
    }
    out = extract_clip_features(wav, meta, spark_cfg)
    assert out is not None
    assert out["mel"]["mel_height"] == 128
    assert out["mel"]["mel_width"] == 126
    assert len(out["mel"]["mel"]) == 128 * 126
    assert out["tabular"]["path"] == meta["path"]


def _write_mini_interim(interim_dir: Path, n: int = 2) -> None:
    audio_root = interim_dir / "audio" / "fold1"
    audio_root.mkdir(parents=True)
    rows = []
    for i in range(n):
        name = f"clip{i}.wav"
        rel = Path("audio") / "fold1" / name
        sf.write(audio_root / name, _sine(seconds=1.0), 16000)
        rows.append(
            {
                "slice_file_name": name,
                "path": str(rel),
                "fold": 1 if i == 0 else 10,
                "class": "dog_bark" if i % 2 == 0 else "siren",
                "classID": 3 if i % 2 == 0 else 8,
                "fsID": 100 + i,
                "start": 0.0,
                "end": 1.0,
                "salience": 1,
                "sample_rate": 16000,
                "is_augmented": False,
                "aug_index": -1,
            }
        )
    pd.DataFrame(rows).to_parquet(interim_dir / "metadata.parquet", index=False)


def test_extract_features_spark_end_to_end(tmp_path: Path) -> None:
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    _write_mini_interim(interim_dir, n=2)
    config = {
        "preprocessing": {"local_interim_dir": str(interim_dir)},
        "spark": {
            "local_processed_dir": str(processed_dir),
            "master": "local[2]",
            "app_name": "test-features",
            "num_partitions": 2,
            "batch_size": 2,
            "sample_rate": 16000,
            "target_duration_sec": 4.0,
            "n_mfcc": 13,
            "n_mels": 128,
            "n_fft": 1024,
            "hop_length": 512,
            "mel_frames": 126,
            "driver_memory": "1g",
            "max_result_size": "512m",
        },
        "versioning": {"processed_path_in_repo": "processed"},
    }
    manifest = extract_features(config=config, force=True, push=False)
    assert manifest["num_written"] == 2
    assert manifest["num_dropped"] == 0
    tab_path = processed_dir / "tabular.parquet"
    mels_path = processed_dir / "mels.parquet"
    assert tab_path.is_file()
    assert mels_path.is_file()
    assert not (processed_dir / "_spark_tabular").exists()
    assert not list(processed_dir.glob("**/*.crc"))
    tab = pd.read_parquet(tab_path)
    mels = pd.read_parquet(mels_path)
    assert len(tab) == 2
    assert len(mels) == 2
    assert set(tab["path"]) == set(mels["path"])
    assert len(mels.iloc[0]["mel"]) == 128 * 126
