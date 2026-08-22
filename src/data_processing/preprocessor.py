"""Clean raw UrbanSound8K parquet clips into versionable interim waveforms."""

from __future__ import annotations

import io
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import soundfile as sf
import yaml

from src.data_processing.audio_augmentor import augment_waveform, build_augmentations

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"
MANIFEST_FILENAME = ".manifest.json"
METADATA_FILENAME = "metadata.parquet"
AUDIO_DIRNAME = "audio"


def load_full_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_preprocessing_config(config_path: Path = CONFIG_PATH) -> dict:
    return load_full_config(config_path)["preprocessing"]


def clean_clip(
    y: np.ndarray,
    sr: int,
    target_sr: int = 16000,
) -> tuple[np.ndarray, int]:
    if y.size == 0:
        raise ValueError("empty audio")
    if y.ndim == 2:
        y = np.mean(y, axis=1)
    elif y.ndim != 1:
        raise ValueError(f"unsupported audio shape {y.shape}")

    y = np.asarray(y, dtype=np.float32)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    peak = float(np.max(np.abs(y)))
    if peak == 0.0:
        raise ValueError("silent audio")
    y = (y / peak).astype(np.float32)
    return y, sr


def decode_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    if not audio_bytes:
        raise ValueError("missing audio bytes")
    y, sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)
    return np.asarray(y, dtype=np.float32), int(sr)


def fold_split(
    frame: pd.DataFrame,
    eval_fold: int = 10,
    fold_column: str = "fold",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fold_column not in frame.columns:
        raise ValueError(f"missing fold column '{fold_column}'")
    eval_mask = frame[fold_column] == eval_fold
    if "is_augmented" in frame.columns:
        eval_mask = eval_mask & ~frame["is_augmented"].astype(bool)
        train = frame.loc[frame[fold_column] != eval_fold].reset_index(drop=True)
    else:
        train = frame.loc[~eval_mask].reset_index(drop=True)
    eval_df = frame.loc[eval_mask].reset_index(drop=True)
    return train, eval_df


def class_counts(
    frame: pd.DataFrame,
    label_column: str = "class",
) -> dict[str, int]:
    if label_column not in frame.columns:
        raise ValueError(f"missing label column '{label_column}'")
    return {str(k): int(v) for k, v in frame[label_column].value_counts().items()}


def _stem_from_slice_name(slice_file_name: str) -> str:
    return Path(slice_file_name).stem


def _write_wav(path: Path, y: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, sr, subtype="PCM_16")


def process_raw_to_interim(
    raw_dir: Path | str | None = None,
    interim_dir: Path | str | None = None,
    config: dict | None = None,
    *,
    max_rows: int | None = None,
    force: bool = False,
) -> dict:
    """Decode/clean raw parquet audio into interim wavs + metadata.parquet.

    Train folds get ``augment_copies`` on-disk siblings; the eval fold does not.
    """
    full = config or load_full_config()
    dataset_cfg = full["dataset"]
    prep = full["preprocessing"]

    raw_dir = Path(raw_dir or dataset_cfg["local_raw_dir"])
    interim_dir = Path(interim_dir or prep["local_interim_dir"])
    audio_root = interim_dir / AUDIO_DIRNAME
    metadata_path = interim_dir / METADATA_FILENAME
    manifest_path = interim_dir / MANIFEST_FILENAME

    if metadata_path.exists() and not force:
        logger.info("Interim metadata already present at %s (pass force=True to redo)", metadata_path)
        with open(manifest_path) as f:
            return json.load(f)

    parquet_dir = raw_dir / "data"
    if not parquet_dir.exists():
        raise FileNotFoundError(f"raw parquet directory not found: {parquet_dir}")

    target_sr = int(prep["target_sample_rate"])
    eval_fold = int(prep["eval_fold"])
    augment_copies = int(prep["augment_copies"])
    fold_column = prep.get("fold_column", dataset_cfg["fold_column"])
    label_column = prep.get("label_column", dataset_cfg["label_column"])
    base_seed = int(prep.get("seed", 42))

    table = ds.dataset(str(parquet_dir), format="parquet")
    columns = [
        "audio",
        "slice_file_name",
        "fsID",
        "start",
        "end",
        "salience",
        fold_column,
        "classID",
        label_column,
    ]
    scanner = table.scanner(columns=columns)
    augmentations = build_augmentations()

    interim_dir.mkdir(parents=True, exist_ok=True)
    if force and audio_root.exists():
        for wav in audio_root.rglob("*.wav"):
            wav.unlink()

    rows: list[dict] = []
    dropped = 0
    seen = 0

    for batch in scanner.to_batches():
        batch_dict = batch.to_pydict()
        n = len(batch_dict["slice_file_name"])
        for i in range(n):
            if max_rows is not None and seen >= max_rows:
                break
            seen += 1
            slice_name = batch_dict["slice_file_name"][i]
            fold = int(batch_dict[fold_column][i])
            label = batch_dict[label_column][i]
            class_id = int(batch_dict["classID"][i])
            audio_struct = batch_dict["audio"][i]
            audio_bytes = audio_struct["bytes"] if isinstance(audio_struct, dict) else None

            try:
                y, sr = decode_audio_bytes(audio_bytes)
                y, sr = clean_clip(y, sr, target_sr=target_sr)
            except (ValueError, OSError, RuntimeError) as exc:
                dropped += 1
                logger.warning("Dropping %s: %s", slice_name, exc)
                continue

            stem = _stem_from_slice_name(slice_name)
            rel_path = Path(f"fold{fold}") / f"{stem}.wav"
            abs_path = audio_root / rel_path
            _write_wav(abs_path, y, sr)
            rows.append(
                {
                    "slice_file_name": slice_name,
                    "path": str(Path(AUDIO_DIRNAME) / rel_path),
                    "fold": fold,
                    "class": label,
                    "classID": class_id,
                    "fsID": int(batch_dict["fsID"][i]),
                    "start": float(batch_dict["start"][i]),
                    "end": float(batch_dict["end"][i]),
                    "salience": int(batch_dict["salience"][i]),
                    "sample_rate": sr,
                    "is_augmented": False,
                    "aug_index": -1,
                }
            )

            if fold == eval_fold or augment_copies <= 0:
                continue

            for aug_i in range(augment_copies):
                aug = augment_waveform(
                    y,
                    sr,
                    key=slice_name,
                    copy_index=aug_i,
                    base_seed=base_seed,
                    augmentations=augmentations,
                )
                peak = float(np.max(np.abs(aug)))
                if peak > 0:
                    aug = (aug / peak).astype(np.float32)
                aug_rel = Path(f"fold{fold}") / f"{stem}_aug{aug_i}.wav"
                _write_wav(audio_root / aug_rel, aug, sr)
                rows.append(
                    {
                        "slice_file_name": slice_name,
                        "path": str(Path(AUDIO_DIRNAME) / aug_rel),
                        "fold": fold,
                        "class": label,
                        "classID": class_id,
                        "fsID": int(batch_dict["fsID"][i]),
                        "start": float(batch_dict["start"][i]),
                        "end": float(batch_dict["end"][i]),
                        "salience": int(batch_dict["salience"][i]),
                        "sample_rate": sr,
                        "is_augmented": True,
                        "aug_index": aug_i,
                    }
                )
        if max_rows is not None and seen >= max_rows:
            break

    if not rows:
        raise ValueError("no clips written to interim; check raw data and drop logs")

    frame = pd.DataFrame(rows)
    frame.to_parquet(metadata_path, index=False)

    train_df, eval_df = fold_split(frame, eval_fold=eval_fold, fold_column="fold")
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "raw_dir": str(raw_dir),
        "interim_dir": str(interim_dir),
        "target_sample_rate": target_sr,
        "eval_fold": eval_fold,
        "augment_copies": augment_copies,
        "num_rows_written": len(frame),
        "num_original": int((~frame["is_augmented"]).sum()),
        "num_augmented": int(frame["is_augmented"].sum()),
        "num_dropped": dropped,
        "num_train_rows": len(train_df),
        "num_eval_rows": len(eval_df),
        "class_counts_all": class_counts(frame.loc[~frame["is_augmented"]], label_column),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Wrote interim dataset: %s", manifest)
    return manifest


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build data/interim from data/raw")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    process_raw_to_interim(max_rows=args.max_rows, force=args.force)


if __name__ == "__main__":
    main()
