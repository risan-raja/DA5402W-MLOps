"""TypedDict shapes for ``config/config.yaml`` sections."""

from __future__ import annotations

from typing import TypedDict


class DatasetConfig(TypedDict):
    hf_repo_id: str
    hf_repo_type: str
    local_raw_dir: str
    expected_rows: int
    fold_range: list[int]
    fold_column: str
    label_column: str
    expected_num_classes: int
    raw_allow_patterns: list[str]
    interim_allow_patterns: list[str]
    processed_allow_patterns: list[str]


class PreprocessingConfig(TypedDict):
    enabled: bool
    local_interim_dir: str
    target_sample_rate: int
    eval_fold: int
    augment_copies: int
    fold_column: str
    label_column: str
    seed: int


class SparkConfig(TypedDict):
    enabled: bool
    local_processed_dir: str
    driver_memory: str
    max_result_size: str
    master: str
    app_name: str
    num_partitions: int
    batch_size: int
    sample_rate: int
    target_duration_sec: float
    n_mfcc: int
    n_mels: int
    n_fft: int
    hop_length: int
    mel_frames: int


class VersioningConfig(TypedDict):
    path_in_repo: str
    processed_path_in_repo: str
    push_interim: bool
    push_processed: bool
    hf_model_repo_id: str
    hf_model_repo_type: str
    push_models: bool


class TrainingCvConfig(TypedDict):
    enabled: bool
    n_folds: int


class CnnTrainingConfig(TypedDict):
    epochs: int
    batch_size: int
    patience: int
    lr: float


class TrainingConfig(TypedDict):
    processed_dir: str
    models_dir: str
    mlflow_experiment: str
    mlflow_tracking_uri: str
    mlflow_artifact_root: str
    train_folds: list[int]
    val_fold: int
    eval_fold: int
    n_trials: int
    cv: TrainingCvConfig
    seed: int
    models: list[str]
    cnn: CnnTrainingConfig


class DriftMonitoringConfig(TypedDict):
    enabled: bool
    window_size: int
    reference_path: str
    psi_warn: float
    psi_alert: float
    n_feature_bins: int


class MonitoringConfig(TypedDict):
    drift: DriftMonitoringConfig


class AppConfig(TypedDict):
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    spark: SparkConfig
    versioning: VersioningConfig
    training: TrainingConfig
    monitoring: MonitoringConfig
