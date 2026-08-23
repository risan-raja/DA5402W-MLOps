"""Train RF / XGBoost / LightGBM / ResNet-18 with Optuna + MLflow."""

from __future__ import annotations

# OpenMP / thread env must be applied before sklearn / xgboost / lightgbm / torch.
from src.models.runtime_env import load_runtime_env

load_runtime_env()

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import cast

import pandas as pd
from mlflow.tracking import MlflowClient

from src.artifact_types import (
    DatasetLineage,
    LineageStub,
    TrainResult,
    WinnerPayload,
)
from src.config_types import AppConfig, SparkConfig, TrainingConfig, TrainingCvConfig
from src.models.baseline_model import TABULAR_MODELS
from src.models.data import (
    collect_dataset_lineage,
    load_full_config,
    load_mels,
    load_tabular,
)
from src.models.mlflow_logging import ensure_mlflow, save_json
from src.models.runtime_env import ROOT
from src.models.train_cnn import train_resnet_model
from src.models.train_tabular import train_tabular_model

logger: logging.Logger = logging.getLogger(__name__)

CONFIG_PATH = ROOT / "config" / "config.yaml"
ALL_MODELS = ("rf", "xgboost", "lightgbm", "resnet18")
WINNER_DIRNAME = "winner"
RUN_RESULT_FILENAME = "run_result.json"


def result_score(result: TrainResult | dict[str, object]) -> float:
    metrics: dict[str, float] = cast(
        dict[str, float], result.get("metrics") or {}
    )
    if "cv_f1_macro_mean" in metrics:
        return float(metrics["cv_f1_macro_mean"])
    return float(metrics.get("f1_macro", float("-inf")))


def _lineage_stub(lineage: DatasetLineage) -> LineageStub:
    return {
        "hf_repo_id": lineage.get("hf_repo_id"),
        "processed_created_at": lineage.get("processed_created_at"),
        "tabular": {"sha256": (lineage.get("tabular") or {}).get("sha256")},
    }


def winner_payload(result: TrainResult, lineage: DatasetLineage) -> WinnerPayload:
    metrics: dict[str, float] = result.get("metrics") or {}
    payload: WinnerPayload = {
        "model_name": result["model_name"],
        "f1_macro": metrics.get("f1_macro"),
        "run_id": result.get("run_id"),
        "dataset": {
            "hf_repo_id": lineage.get("hf_repo_id"),
            "processed_created_at": lineage.get("processed_created_at"),
            "tabular_sha256": (lineage.get("tabular") or {}).get("sha256"),
        },
    }
    if "cv_f1_macro_mean" in metrics:
        payload["cv_f1_macro_mean"] = metrics.get("cv_f1_macro_mean")
        payload["cv_f1_macro_std"] = metrics.get("cv_f1_macro_std")
    return payload


def persist_run_result(result: TrainResult, lineage: DatasetLineage) -> Path:
    out_dir: Path = Path(result["out_dir"])
    payload: dict[str, object] = {**result, "dataset": _lineage_stub(lineage)}
    path: Path = out_dir / RUN_RESULT_FILENAME
    save_json(path, payload)
    return path


def materialize_winner_dir(models_dir: Path, winner_name: str) -> Path:
    source: Path = models_dir / winner_name
    dest: Path = models_dir / WINNER_DIRNAME
    if not source.is_dir():
        raise FileNotFoundError(source)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)
    return dest


def tag_mlflow_winners(
    results: list[TrainResult], winner: TrainResult
) -> None:
    try:
        client: MlflowClient = MlflowClient()
    except Exception:
        logger.exception("MLflow client unavailable; skipping winner tags")
        return
    for result in results:
        is_winner: bool = result["model_name"] == winner["model_name"]
        try:
            versions = client.search_model_versions(f"name='{result['model_name']}'")
            run_versions = [v for v in versions if v.run_id == result["run_id"]]
            if run_versions:
                ver: str = run_versions[0].version
                client.set_model_version_tag(
                    result["model_name"],
                    ver,
                    "winner",
                    "true" if is_winner else "false",
                )
                client.set_registered_model_tag(
                    result["model_name"], "winner", "true" if is_winner else "false"
                )
                if is_winner:
                    client.set_registered_model_alias(
                        result["model_name"], "production", ver
                    )
        except Exception:
            logger.exception("Winner tagging failed for %s", result["model_name"])


def select_winner(
    results: list[TrainResult], models_dir: Path, lineage: DatasetLineage
) -> TrainResult:
    if not results:
        raise ValueError("no training results to select a winner from")
    winner: TrainResult = max(results, key=result_score)
    tag_mlflow_winners(results, winner)
    payload: WinnerPayload = winner_payload(winner, lineage)
    save_json(models_dir / "winner.json", payload)
    materialize_winner_dir(models_dir, winner["model_name"])
    logger.info("Winner: %s (score=%s)", winner["model_name"], result_score(winner))
    return winner


def _resolve_models_dir(train_cfg: TrainingConfig | dict[str, object]) -> Path:
    models_dir: Path = Path(str(train_cfg["models_dir"]))
    if not models_dir.is_absolute():
        models_dir = ROOT / models_dir
    train_cfg["models_dir"] = str(models_dir)
    return models_dir


def _prepare_training(
    config_path: Path,
    n_trials: int | None,
    enable_cv: bool | None,
) -> tuple[
    AppConfig,
    TrainingConfig | dict[str, object],
    SparkConfig | dict[str, object],
    int,
    int,
    Path,
    Path,
    DatasetLineage,
]:
    full: AppConfig = load_full_config(config_path)
    train_cfg: TrainingConfig | dict[str, object] = dict(full["training"])
    if enable_cv is not None:
        cv_cfg: TrainingCvConfig | dict[str, object] = dict(train_cfg.get("cv") or {})
        cv_cfg["enabled"] = bool(enable_cv)
        train_cfg["cv"] = cv_cfg
    spark_cfg: SparkConfig | dict[str, object] = full.get("spark", {})
    seed: int = int(train_cfg.get("seed", 42))
    trials: int = int(n_trials if n_trials is not None else train_cfg.get("n_trials", 20))

    processed_dir: Path = Path(str(train_cfg["processed_dir"]))
    if not processed_dir.is_absolute():
        processed_dir = ROOT / processed_dir
    models_dir: Path = _resolve_models_dir(train_cfg)
    artifact_root: Path = Path(str(train_cfg["mlflow_artifact_root"]))
    if not artifact_root.is_absolute():
        artifact_root = ROOT / artifact_root
    train_cfg["mlflow_artifact_root"] = str(artifact_root)

    ensure_mlflow(train_cfg)
    lineage: DatasetLineage = collect_dataset_lineage(processed_dir, full_config=full)
    logger.info(
        "Dataset lineage: hf=%s processed_at=%s tabular_sha=%s",
        lineage.get("hf_repo_id"),
        lineage.get("processed_created_at"),
        (lineage.get("tabular") or {}).get("sha256", "")[:12],
    )
    return full, train_cfg, spark_cfg, seed, trials, processed_dir, models_dir, lineage


def _fit_named_model(
    name: str,
    *,
    train_cfg: TrainingConfig | dict[str, object],
    spark_cfg: SparkConfig | dict[str, object],
    tabular: pd.DataFrame | None,
    mels_df: pd.DataFrame | None,
    n_trials: int,
    seed: int,
    lineage: DatasetLineage,
) -> TrainResult:
    if name not in ALL_MODELS:
        raise ValueError(f"unknown model '{name}'")
    logger.info("Training %s (%s trials)", name, n_trials)
    if name in TABULAR_MODELS:
        if tabular is None:
            raise ValueError(f"tabular features required for {name}")
        return train_tabular_model(
            name,
            train_cfg=train_cfg,
            spark_cfg=spark_cfg,
            tabular=tabular,
            n_trials=n_trials,
            seed=seed,
            lineage=lineage,
        )
    if mels_df is None:
        raise ValueError("mel features required for resnet18")
    return train_resnet_model(
        train_cfg=train_cfg,
        mels_df=mels_df,
        n_trials=n_trials,
        seed=seed,
        lineage=lineage,
    )


def train_one_model(
    model_name: str,
    config_path: Path = CONFIG_PATH,
    n_trials: int | None = None,
    *,
    enable_cv: bool | None = None,
) -> TrainResult:
    """Train a single model and persist ``run_result.json`` under its artifact dir."""
    (
        _,
        train_cfg,
        spark_cfg,
        seed,
        trials,
        processed_dir,
        _models_dir,
        lineage,
    ) = _prepare_training(config_path, n_trials, enable_cv)
    tabular: pd.DataFrame | None = (
        load_tabular(processed_dir) if model_name in TABULAR_MODELS else None
    )
    mels_df: pd.DataFrame | None = (
        load_mels(processed_dir) if model_name == "resnet18" else None
    )
    result: TrainResult = _fit_named_model(
        model_name,
        train_cfg=train_cfg,
        spark_cfg=spark_cfg,
        tabular=tabular,
        mels_df=mels_df,
        n_trials=trials,
        seed=seed,
        lineage=lineage,
    )
    persist_run_result(result, lineage)
    return result


def select_winner_from_artifacts(
    config_path: Path = CONFIG_PATH,
    models: list[str] | None = None,
) -> TrainResult:
    """Pick the winner from on-disk ``run_result.json`` files written by train tasks."""
    full: AppConfig = load_full_config(config_path)
    train_cfg: TrainingConfig | dict[str, object] = dict(full["training"])
    models_dir: Path = _resolve_models_dir(train_cfg)
    artifact_root: Path = Path(str(train_cfg["mlflow_artifact_root"]))
    if not artifact_root.is_absolute():
        artifact_root = ROOT / artifact_root
    train_cfg["mlflow_artifact_root"] = str(artifact_root)
    ensure_mlflow(train_cfg)

    selected: list[str] = models or list(train_cfg.get("models", ALL_MODELS))
    results: list[TrainResult] = []
    for name in selected:
        path: Path = models_dir / name / RUN_RESULT_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"missing run result for {name}: {path}")
        with open(path) as f:
            loaded: object = json.load(f)
        results.append(cast(TrainResult, loaded))
    lineage_raw: object = results[0].get("dataset") or {}
    if not isinstance(lineage_raw, dict):
        lineage_raw = {}
    lineage: DatasetLineage = {
        "hf_repo_id": cast(str | None, lineage_raw.get("hf_repo_id")),
        "processed_created_at": cast(
            str | None, lineage_raw.get("processed_created_at")
        ),
        "tabular": cast(dict[str, object], lineage_raw.get("tabular") or {}),
    }
    return select_winner(results, models_dir, lineage)


def run_training(
    config_path: Path = CONFIG_PATH,
    models: list[str] | None = None,
    n_trials: int | None = None,
    *,
    enable_cv: bool | None = None,
) -> list[TrainResult]:
    (
        _,
        train_cfg,
        spark_cfg,
        seed,
        trials,
        processed_dir,
        models_dir,
        lineage,
    ) = _prepare_training(config_path, n_trials, enable_cv)
    selected: list[str] = models or list(train_cfg.get("models", ALL_MODELS))

    results: list[TrainResult] = []
    need_tabular: bool = any(m in TABULAR_MODELS for m in selected)
    need_mels: bool = "resnet18" in selected
    tabular: pd.DataFrame | None = load_tabular(processed_dir) if need_tabular else None
    mels_df: pd.DataFrame | None = load_mels(processed_dir) if need_mels else None

    for name in selected:
        result: TrainResult = _fit_named_model(
            name,
            train_cfg=train_cfg,
            spark_cfg=spark_cfg,
            tabular=tabular,
            mels_df=mels_df,
            n_trials=trials,
            seed=seed,
            lineage=lineage,
        )
        persist_run_result(result, lineage)
        results.append(result)

    if results:
        select_winner(results, models_dir, lineage)
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train UrbanSound8K models")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated subset, e.g. rf,xgboost",
    )
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Enable official UrbanSound8K 10-fold eval CV after Optuna",
    )
    args = parser.parse_args()
    models: list[str] | None = (
        [m.strip() for m in args.models.split(",")] if args.models else None
    )
    run_training(
        config_path=args.config,
        models=models,
        n_trials=args.n_trials,
        enable_cv=True if args.cv else None,
    )


if __name__ == "__main__":
    main()
