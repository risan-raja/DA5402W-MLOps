"""Shared typed loader for ``config/config.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml

from src.config_types import (
    AppConfig,
    DatasetConfig,
    PreprocessingConfig,
    SparkConfig,
    TrainingConfig,
    VersioningConfig,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "config.yaml"

_REQUIRED_TOP_LEVEL: tuple[str, ...] = (
    "dataset",
    "preprocessing",
    "spark",
    "versioning",
    "training",
    "monitoring",
)


def load_app_config(path: Path | None = None) -> AppConfig:
    """Load YAML config, require top-level sections, return as ``AppConfig``."""
    config_path: Path = path if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path) as handle:
        loaded: object = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"invalid config (expected mapping): {config_path}")
    missing: list[str] = [key for key in _REQUIRED_TOP_LEVEL if key not in loaded]
    if missing:
        raise KeyError(f"config missing required top-level keys {missing}: {config_path}")
    return cast(AppConfig, loaded)


def load_dataset_config(path: Path | None = None) -> DatasetConfig:
    return load_app_config(path)["dataset"]


def load_preprocessing_config(path: Path | None = None) -> PreprocessingConfig:
    return load_app_config(path)["preprocessing"]


def load_spark_config(path: Path | None = None) -> SparkConfig:
    return load_app_config(path)["spark"]


def load_versioning_config(path: Path | None = None) -> VersioningConfig:
    return load_app_config(path)["versioning"]


def load_training_config(path: Path | None = None) -> TrainingConfig:
    return load_app_config(path)["training"]
