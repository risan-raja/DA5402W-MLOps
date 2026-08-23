"""PySpark feature extraction from interim wavs → tabular + mel parquets."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
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

from src.data_pipeline.audio_features import (
    decode_wav,
    extract_log_mel,
    extract_tabular_features,
    prepare_waveform,
    tabular_feature_names,
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


def extract_clip_features(
    abs_wav_path: Path | str, meta: dict, spark_cfg: dict
) -> dict | None:
    """Return ``{"tabular": dict, "mel": dict}`` or None if the clip cannot be read."""
    try:
        y, sr = decode_wav(abs_wav_path)
        sample_rate = int(spark_cfg["sample_rate"])
        y = prepare_waveform(
            y,
            sr,
            sample_rate=sample_rate,
            duration_sec=float(spark_cfg["target_duration_sec"]),
        )
        tabular_feats = extract_tabular_features(
            y, sample_rate, n_mfcc=int(spark_cfg["n_mfcc"])
        )
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
        writer = None
        try:
            for part in parts:
                table = pq.read_table(part)
                if writer is None:
                    writer = pq.ParquetWriter(
                        dest_file, table.schema, compression="snappy"
                    )
                writer.write_table(table)
                del table
        finally:
            if writer is not None:
                writer.close()
    shutil.rmtree(spark_out_dir)


def _chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _stream_merge_files(parts: list[Path], dest_file: Path) -> None:
    if dest_file.exists():
        dest_file.unlink()
    if len(parts) == 1:
        shutil.move(str(parts[0]), str(dest_file))
        return
    writer = None
    try:
        for part in parts:
            table = pq.read_table(part)
            if writer is None:
                writer = pq.ParquetWriter(dest_file, table.schema, compression="snappy")
            writer.write_table(table)
            del table
    finally:
        if writer is not None:
            writer.close()
    for part in parts:
        part.unlink(missing_ok=True)


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
    work_dir = processed_dir / "_spark_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    batch_size = int(spark_cfg.get("batch_size", 256))
    num_partitions = int(spark_cfg.get("num_partitions", 2))
    driver_memory = spark_cfg.get("driver_memory", "1g")
    max_result_size = spark_cfg.get("max_result_size", "512m")

    spark = (
        SparkSession.builder.master(spark_cfg["master"])
        .appName(spark_cfg["app_name"])
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", driver_memory)
        .config("spark.driver.maxResultSize", max_result_size)
        .config("spark.sql.shuffle.partitions", str(max(2, num_partitions)))
        .config("spark.local.dir", str(work_dir / "scratch"))
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    kept = 0
    dropped = 0
    tab_batch_files: list[Path] = []
    mel_batch_files: list[Path] = []
    try:
        for batch_idx, batch in enumerate(_chunked(records, batch_size)):
            logger.info(
                "Spark batch %s: %s clips (batch_size=%s)",
                batch_idx,
                len(batch),
                batch_size,
            )
            parts = min(num_partitions, len(batch))
            rdd = spark.sparkContext.parallelize(batch, parts)
            # Cache only this small batch (~batch_size clips), then drop it.
            extracted = rdd.mapPartitions(
                lambda part: _extract_partition(part, spark_cfg)
            ).cache()
            batch_kept = extracted.count()
            batch_dropped = len(batch) - batch_kept
            kept += batch_kept
            dropped += batch_dropped
            if batch_kept == 0:
                extracted.unpersist()
                continue

            tab_rdd = extracted.map(lambda rec: _to_tabular_row(rec, feature_names))
            mel_rdd = extracted.map(_to_mel_row)
            tab_df = spark.createDataFrame(
                tab_rdd, schema=_tabular_schema(feature_names)
            )
            mel_df = spark.createDataFrame(mel_rdd, schema=_mel_schema())

            tab_tmp = work_dir / f"tab_batch_{batch_idx}"
            mel_tmp = work_dir / f"mel_batch_{batch_idx}"
            tab_out = work_dir / f"tabular_batch_{batch_idx}.parquet"
            mel_out = work_dir / f"mels_batch_{batch_idx}.parquet"
            tab_df.write.mode("overwrite").parquet(str(tab_tmp))
            mel_df.write.mode("overwrite").parquet(str(mel_tmp))
            extracted.unpersist()
            _finalize_spark_parquet_dir(tab_tmp, tab_out)
            _finalize_spark_parquet_dir(mel_tmp, mel_out)
            tab_batch_files.append(tab_out)
            mel_batch_files.append(mel_out)
    finally:
        spark.stop()

    if kept == 0 or not tab_batch_files:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise ValueError("no clips produced features; check interim audio paths")

    _stream_merge_files(tab_batch_files, tabular_path)
    _stream_merge_files(mel_batch_files, mels_path)
    shutil.rmtree(work_dir, ignore_errors=True)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "interim_dir": str(interim_dir),
        "processed_dir": str(processed_dir),
        "num_input_rows": len(records),
        "num_written": kept,
        "num_dropped": dropped,
        "batch_size": batch_size,
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

    should_push = push or bool(full.get("versioning", {}).get("push_processed", False))
    if should_push:
        path_in_repo = full.get("versioning", {}).get(
            "processed_path_in_repo", "processed"
        )
        push_dataset_tree(processed_dir, path_in_repo)
        manifest["pushed_path_in_repo"] = path_in_repo
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    return manifest


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Spark feature extraction → data/processed"
    )
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
