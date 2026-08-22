"""Train RF / XGBoost / LightGBM / ResNet-18 with Optuna + MLflow."""

from __future__ import annotations

# OpenMP / thread env must be applied before sklearn / xgboost / lightgbm / torch.
from src.models.runtime_env import load_runtime_env

load_runtime_env()

import argparse
import logging
from pathlib import Path

from mlflow.tracking import MlflowClient

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

logger = logging.getLogger(__name__)

CONFIG_PATH = ROOT / "config" / "config.yaml"
ALL_MODELS = ("rf", "xgboost", "lightgbm", "resnet18")


def run_training(
    config_path: Path = CONFIG_PATH,
    models: list[str] | None = None,
    n_trials: int | None = None,
) -> list[dict]:
    full = load_full_config(config_path)
    train_cfg = dict(full["training"])
    spark_cfg = full.get("spark", {})
    seed = int(train_cfg.get("seed", 42))
    selected = models or list(train_cfg.get("models", ALL_MODELS))
    trials = int(n_trials if n_trials is not None else train_cfg.get("n_trials", 20))

    processed_dir = Path(train_cfg["processed_dir"])
    if not processed_dir.is_absolute():
        processed_dir = ROOT / processed_dir
    models_dir = Path(train_cfg["models_dir"])
    if not models_dir.is_absolute():
        models_dir = ROOT / models_dir
    train_cfg["models_dir"] = str(models_dir)
    artifact_root = Path(train_cfg["mlflow_artifact_root"])
    if not artifact_root.is_absolute():
        artifact_root = ROOT / artifact_root
    train_cfg["mlflow_artifact_root"] = str(artifact_root)

    ensure_mlflow(train_cfg)
    lineage = collect_dataset_lineage(processed_dir, full_config=full)
    logger.info(
        "Dataset lineage: hf=%s processed_at=%s tabular_sha=%s",
        lineage.get("hf_repo_id"),
        lineage.get("processed_created_at"),
        (lineage.get("tabular") or {}).get("sha256", "")[:12],
    )

    results: list[dict] = []
    need_tabular = any(m in TABULAR_MODELS for m in selected)
    need_mels = "resnet18" in selected
    tabular = load_tabular(processed_dir) if need_tabular else None
    mels_df = load_mels(processed_dir) if need_mels else None

    for name in selected:
        if name not in ALL_MODELS:
            raise ValueError(f"unknown model '{name}'")
        logger.info("Training %s (%s trials)", name, trials)
        if name in TABULAR_MODELS:
            results.append(
                train_tabular_model(
                    name,
                    train_cfg=train_cfg,
                    spark_cfg=spark_cfg,
                    tabular=tabular,
                    n_trials=trials,
                    seed=seed,
                    lineage=lineage,
                )
            )
        else:
            results.append(
                train_resnet_model(
                    train_cfg=train_cfg,
                    mels_df=mels_df,
                    n_trials=trials,
                    seed=seed,
                    lineage=lineage,
                )
            )

    if not results:
        return results

    winner = max(results, key=lambda r: r["metrics"].get("f1_macro", float("-inf")))
    client = MlflowClient()
    for r in results:
        is_winner = r["model_name"] == winner["model_name"]
        try:
            versions = client.search_model_versions(f"name='{r['model_name']}'")
            run_versions = [v for v in versions if v.run_id == r["run_id"]]
            if run_versions:
                ver = run_versions[0].version
                client.set_model_version_tag(
                    r["model_name"], ver, "winner", "true" if is_winner else "false"
                )
                client.set_registered_model_tag(
                    r["model_name"], "winner", "true" if is_winner else "false"
                )
                if is_winner:
                    client.set_registered_model_alias(
                        r["model_name"], "production", ver
                    )
        except Exception:
            logger.exception("Winner tagging failed for %s", r["model_name"])
        if is_winner:
            save_json(
                models_dir / "winner.json",
                {
                    "model_name": r["model_name"],
                    "f1_macro": r["metrics"].get("f1_macro"),
                    "run_id": r["run_id"],
                    "dataset": {
                        "hf_repo_id": lineage.get("hf_repo_id"),
                        "processed_created_at": lineage.get("processed_created_at"),
                        "tabular_sha256": (lineage.get("tabular") or {}).get("sha256"),
                    },
                },
            )
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
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    run_training(config_path=args.config, models=models, n_trials=args.n_trials)


if __name__ == "__main__":
    main()
