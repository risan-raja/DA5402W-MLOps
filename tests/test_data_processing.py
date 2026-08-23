from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from src.data_processing.audio_augmentor import augment_waveform, build_augmentations
from src.data_processing.preprocessor import (
    class_counts,
    clean_clip,
    decode_audio_bytes,
    fold_split,
    process_raw_to_interim,
)


def _sine(sr: int = 22050, seconds: float = 0.2, freq: float = 440.0) -> np.ndarray:
    t: np.ndarray = np.linspace(
        0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32
    )
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_clean_clip_mono_resample_and_peak_norm() -> None:
    y: np.ndarray = np.stack([_sine(), _sine(freq=880.0)], axis=1)  # stereo (n, 2)
    cleaned, sr = clean_clip(y, 22050, target_sr=16000)
    assert cleaned.ndim == 1
    assert sr == 16000
    assert cleaned.dtype == np.float32
    assert pytest.approx(float(np.max(np.abs(cleaned))), abs=1e-5) == 1.0


def test_clean_clip_rejects_empty_and_silent() -> None:
    with pytest.raises(ValueError, match="empty audio"):
        clean_clip(np.array([], dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="silent audio"):
        clean_clip(np.zeros(1000, dtype=np.float32), 16000)


def test_decode_audio_bytes_roundtrip() -> None:
    raw: np.ndarray = _sine(sr=16000)
    y, sr = decode_audio_bytes(_wav_bytes(raw, 16000))
    assert sr == 16000
    assert y.ndim == 1
    assert len(y) == len(raw)


def test_augment_waveform_is_seeded_and_changes_signal() -> None:
    y: np.ndarray = _sine(sr=16000)
    pipeline = build_augmentations(
        gaussian_noise_p=1.0,
        pitch_shift_p=1.0,
        time_stretch_p=1.0,
    )
    a: np.ndarray = augment_waveform(
        y, 16000, key="a.wav", copy_index=0, base_seed=7, augmentations=pipeline
    )
    b: np.ndarray = augment_waveform(
        y, 16000, key="a.wav", copy_index=0, base_seed=7, augmentations=pipeline
    )
    c: np.ndarray = augment_waveform(
        y, 16000, key="b.wav", copy_index=0, base_seed=7, augmentations=pipeline
    )
    assert a.shape[0] > 0
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_fold_split_default_eval_fold_excludes_augmented_eval() -> None:
    frame: pd.DataFrame = pd.DataFrame(
        {
            "fold": [1, 1, 10, 10],
            "class": ["dog_bark", "dog_bark", "siren", "siren"],
            "is_augmented": [False, True, False, True],
        }
    )
    train, eval_df = fold_split(frame, eval_fold=10)
    assert set(train["fold"]) == {1}
    assert len(train) == 2
    assert list(eval_df["fold"]) == [10]
    assert list(eval_df["is_augmented"]) == [False]


def test_class_counts() -> None:
    frame = pd.DataFrame({"class": ["dog_bark", "siren", "dog_bark"]})
    assert class_counts(frame) == {"dog_bark": 2, "siren": 1}


def _write_fake_raw(raw_dir: Path, n: int = 4) -> None:
    data_dir = raw_dir / "data"
    data_dir.mkdir(parents=True)
    rows = {
        "audio": [],
        "slice_file_name": [],
        "fsID": [],
        "start": [],
        "end": [],
        "salience": [],
        "fold": [],
        "classID": [],
        "class": [],
    }
    for i in range(n):
        fold = 10 if i == n - 1 else 1
        name = f"clip{i}.wav"
        rows["audio"].append(
            {"bytes": _wav_bytes(_sine(sr=22050), 22050), "path": name}
        )
        rows["slice_file_name"].append(name)
        rows["fsID"].append(100 + i)
        rows["start"].append(0.0)
        rows["end"].append(0.2)
        rows["salience"].append(1)
        rows["fold"].append(fold)
        rows["classID"].append(i % 2)
        rows["class"].append("dog_bark" if i % 2 == 0 else "siren")

    audio_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.Table.from_pydict(
        rows,
        schema=pa.schema(
            [
                ("audio", audio_type),
                ("slice_file_name", pa.string()),
                ("fsID", pa.int64()),
                ("start", pa.float64()),
                ("end", pa.float64()),
                ("salience", pa.int64()),
                ("fold", pa.int64()),
                ("classID", pa.int64()),
                ("class", pa.string()),
            ]
        ),
    )
    pq.write_table(table, data_dir / "train-00000.parquet")


def test_process_raw_to_interim(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    _write_fake_raw(raw_dir, n=4)

    config = {
        "dataset": {
            "local_raw_dir": str(raw_dir),
            "fold_column": "fold",
            "label_column": "class",
        },
        "preprocessing": {
            "local_interim_dir": str(interim_dir),
            "target_sample_rate": 16000,
            "eval_fold": 10,
            "augment_copies": 1,
            "fold_column": "fold",
            "label_column": "class",
            "seed": 42,
        },
    }
    manifest = process_raw_to_interim(config=config)
    # 3 train originals + 3 aug + 1 eval original = 7
    assert manifest["num_original"] == 4
    assert manifest["num_augmented"] == 3
    assert manifest["num_rows_written"] == 7
    assert manifest["num_dropped"] == 0
    assert (interim_dir / "metadata.parquet").exists()
    meta = pd.read_parquet(interim_dir / "metadata.parquet")
    assert set(meta["sample_rate"]) == {16000}
    assert meta.loc[meta["fold"] == 10, "is_augmented"].sum() == 0
