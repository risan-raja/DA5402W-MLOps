"""PySpark feature extraction from interim wavs → tabular + mel parquets."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from src.data_processing.versioning import push_dataset_tree

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"
MANIFEST_FILENAME = ".manifest.json"
TABULAR_FILENAME = "tabular.parquet"
MELS_FILENAME = "mels.parquet"
META_COLS = (
    "path",
    "slice_file_name",
    "fold",
    "class",
    "classID",
    "is_augmented",
    "aug_index",
)


def load_full_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


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


def extract_tabular_features(y: np.ndarray, sample_rate: int, n_mfcc: int = 13) -> dict[str, float]:
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sample_rate)
    feats = {}
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


def extract_clip_features(abs_wav_path: Path | str, meta: dict, spark_cfg: dict) -> dict | None:
    """Return ``{"tabular": dict, "mel": dict}`` or None if the clip cannot be read."""
    try:
        y, sr = sf.read(str(abs_wav_path), always_2d=False)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if y.size == 0:
            raise ValueError("empty audio")
        sample_rate = int(spark_cfg["sample_rate"])
        if sr != sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
        y = pad_or_truncate(y, sample_rate, float(spark_cfg["target_duration_sec"]))
        tabular_feats = extract_tabular_features(y, sample_rate, n_mfcc=int(spark_cfg["n_mfcc"]))
        log_mel = extract_log_mel(
            y,
            sample_rate,
            n_mels=int(spark_cfg["n_mels"]),
            n_fft=int(spark_cfg["n_fft"]),
            hop_length=int(spark_cfg["hop_length"]),
            mel_frames=int(spark_cfg["mel_frames"]),
        )
    except (ValueError, OSError, RuntimeError) as exc:
        logger.warning("Dropping %s: %s", abs_wav_path, exc)
        return None

    meta_out = {k: meta[k] for k in META_COLS}
    tabular = {**meta_out, **tabular_feats}
    mel = {
        **meta_out,
        "mel": log_mel.reshape(-1).astype(np.float32).tolist(),
        "mel_height": int(log_mel.shape[0]),
        "mel_width": int(log_mel.shape[1]),
    }
    return {"tabular": tabular, "mel": mel}


def _extract_partition(rows, spark_cfg: dict):
    for row in rows:
        result = extract_clip_features(row["_abs_path"], row, spark_cfg)
        if result is not None:
            yield result


def _tabular_schema(feature_names: list[str]) -> StructType:
    fields = [
        StructField("path", StringType(), False),
        StructField("slice_file_name", StringType(), False),
        StructField("fold", LongType(), False),
        StructField("class", StringType(), False),
        StructField("classID", LongType(), False),
        StructField("is_augmented", BooleanType(), False),
        StructField("aug_index", LongType(), False),
    ]
    fields.extend(StructField(name, DoubleType(), False) for name in feature_names)
    return StructType(fields)


def _mel_schema() -> StructType:
    return StructType(
        [
            StructField("path", StringType(), False),
            StructField("slice_file_name", StringType(), False),
            StructField("fold", LongType(), False),
            StructField("class", StringType(), False),
            StructField("classID", LongType(), False),
            StructField("is_augmented", BooleanType(), False),
            StructField("aug_index", LongType(), False),
            StructField("mel", ArrayType(FloatType(), False), False),
            StructField("mel_height", IntegerType(), False),
            StructField("mel_width", IntegerType(), False),
        ]
    )


def _to_tabular_row(record: dict, feature_names: list[str]) -> Row:
    t = record["tabular"]
    values = [
        t["path"],
        t["slice_file_name"],
        int(t["fold"]),
        t["class"],
        int(t["classID"]),
        bool(t["is_augmented"]),
        int(t["aug_index"]),
    ]
    values.extend(float(t[name]) for name in feature_names)
    return Row(*values)


def _to_mel_row(record: dict) -> Row:
    m = record["mel"]
    return Row(
        m["path"],
        m["slice_file_name"],
        int(m["fold"]),
        m["class"],
        int(m["classID"]),
        bool(m["is_augmented"]),
        int(m["aug_index"]),
        m["mel"],
        int(m["mel_height"]),
        int(m["mel_width"]),
    )


def _finalize_spark_parquet_dir(spark_out_dir: Path, dest_file: Path) -> None:
    """Collapse Spark's part-*/_SUCCESS/.crc directory into one parquet file."""
    parts = sorted(spark_out_dir.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no parquet parts under {spark_out_dir}")
    if dest_file.exists():
        if dest_file.is_dir():
            shutil.rmtree(dest_file)
        else:
            dest_file.unlink()
    if len(parts) == 1:
        shutil.move(str(parts[0]), str(dest_file))
    else:
        table = pd.read_parquet(spark_out_dir)
        table.to_parquet(dest_file, index=False)
    shutil.rmtree(spark_out_dir)


def extract_features(
    interim_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
    config: dict | None = None,
    *,
    push: bool = False,
    max_rows: int | None = None,
    force: bool = False,
) -> dict:
    full = config or load_full_config()
    prep = full["preprocessing"]
    spark_cfg = full["spark"]
    interim_dir = Path(interim_dir or prep["local_interim_dir"])
    processed_dir = Path(processed_dir or spark_cfg["local_processed_dir"])
    if not interim_dir.is_absolute():
        interim_dir = ROOT / interim_dir
    if not processed_dir.is_absolute():
        processed_dir = ROOT / processed_dir

    metadata_path = interim_dir / "metadata.parquet"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing interim metadata: {metadata_path}")

    tabular_path = processed_dir / TABULAR_FILENAME
    mels_path = processed_dir / MELS_FILENAME
    manifest_path = processed_dir / MANIFEST_FILENAME
    if tabular_path.exists() and mels_path.exists() and not force:
        logger.info("Processed outputs already exist (pass force=True to redo)")
        with open(manifest_path) as f:
            return json.load(f)

    meta = pd.read_parquet(metadata_path)
    if max_rows is not None:
        meta = meta.head(max_rows)
    records = meta.to_dict(orient="records")
    for row in records:
        row["_abs_path"] = str(interim_dir / row["path"])

    feature_names = tabular_feature_names(n_mfcc=int(spark_cfg["n_mfcc"]))
    processed_dir.mkdir(parents=True, exist_ok=True)

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = (
        SparkSession.builder.master(spark_cfg["master"])
        .appName(spark_cfg["app_name"])
        .config("spark.driver.memory", spark_cfg.get("driver_memory", "4g"))
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    tab_tmp = processed_dir / "_spark_tabular"
    mel_tmp = processed_dir / "_spark_mels"
    try:
        num_partitions = int(spark_cfg.get("num_partitions", 8))
        rdd = spark.sparkContext.parallelize(records, num_partitions)
        extracted = rdd.mapPartitions(lambda part: _extract_partition(part, spark_cfg)).cache()
        kept = extracted.count()
        dropped = len(records) - kept
        if kept == 0:
            raise ValueError("no clips produced features; check interim audio paths")

        tab_rdd = extracted.map(lambda rec: _to_tabular_row(rec, feature_names))
        mel_rdd = extracted.map(_to_mel_row)
        tab_df = spark.createDataFrame(tab_rdd, schema=_tabular_schema(feature_names))
        mel_df = spark.createDataFrame(mel_rdd, schema=_mel_schema())

        tab_df.coalesce(1).write.mode("overwrite").parquet(str(tab_tmp))
        mel_df.coalesce(1).write.mode("overwrite").parquet(str(mel_tmp))
        extracted.unpersist()
    finally:
        spark.stop()

    _finalize_spark_parquet_dir(tab_tmp, tabular_path)
    _finalize_spark_parquet_dir(mel_tmp, mels_path)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "interim_dir": str(interim_dir),
        "processed_dir": str(processed_dir),
        "num_input_rows": len(records),
        "num_written": kept,
        "num_dropped": dropped,
        "n_mfcc": int(spark_cfg["n_mfcc"]),
        "n_mels": int(spark_cfg["n_mels"]),
        "mel_shape": [int(spark_cfg["n_mels"]), int(spark_cfg["mel_frames"])],
        "feature_names": feature_names,
        "tabular_path": TABULAR_FILENAME,
        "mels_path": MELS_FILENAME,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote processed features: %s", manifest)

    should_push = push or os.environ.get("PUSH_PROCESSED", "").strip() in {"1", "true", "True"}
    if should_push:
        path_in_repo = full.get("versioning", {}).get("processed_path_in_repo", "processed")
        push_dataset_tree(processed_dir, path_in_repo)
        manifest["pushed_path_in_repo"] = path_in_repo
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    return manifest


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Spark feature extraction → data/processed")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--push",
        action="store_true",
        help="Upload data/processed to HF under versioning.processed_path_in_repo",
    )
    args = parser.parse_args()
    extract_features(max_rows=args.max_rows, force=args.force, push=args.push)


if __name__ == "__main__":
    main()
