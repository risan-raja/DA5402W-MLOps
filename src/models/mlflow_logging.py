"""MLflow setup, dataset lineage, Optuna artifacts, and model registry helpers."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.lightgbm
import mlflow.pytorch
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.data.meta_dataset import MetaDataset
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def ensure_mlflow(train_cfg: dict) -> None:
    tracking = os.environ.get("MLFLOW_TRACKING_URI") or train_cfg["mlflow_tracking_uri"]
    mlflow.set_tracking_uri(tracking)
    Path(train_cfg["mlflow_artifact_root"]).mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment(train_cfg["mlflow_experiment"])


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def jsonable_params(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if isinstance(v, (np.integer, int)):
            out[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            out[k] = float(v)
        elif isinstance(v, (str, bool)):
            out[k] = v
    return out


def log_dataset_lineage(lineage: dict) -> MetaDataset:
    """Attach dataset provenance and register an MLflow Dataset input.

    Returns the MetaDataset so metrics can be logged with ``dataset=...``
    (fills the Metrics table Dataset column in the MLflow UI).
    """
    scalar_keys = (
        "hf_repo_id",
        "hf_repo_type",
        "raw_revision",
        "raw_downloaded_at",
        "interim_created_at",
        "interim_num_rows",
        "processed_created_at",
        "processed_num_rows",
        "processed_num_input_rows",
    )
    params: dict = {}
    tags: dict = {}
    for key in scalar_keys:
        val = lineage.get(key)
        if val is None or val == "":
            continue
        s = str(val)
        params[f"data_{key}"] = s[:250]
        tags[f"data.{key}"] = s[:5000]

    tab = lineage.get("tabular") or {}
    mel = lineage.get("mels") or {}
    if tab.get("sha256"):
        params["data_tabular_sha256"] = tab["sha256"][:64]
        tags["data.tabular_sha256"] = tab["sha256"]
    if mel.get("sha256"):
        params["data_mels_sha256"] = mel["sha256"][:64]
        tags["data.mels_sha256"] = mel["sha256"]
    if tab.get("size_bytes") is not None:
        params["data_tabular_bytes"] = int(tab["size_bytes"])
    if mel.get("size_bytes") is not None:
        params["data_mels_bytes"] = int(mel["size_bytes"])

    if params:
        mlflow.log_params(
            {k: v for k, v in params.items() if not isinstance(v, (dict, list))}
        )
    for k, v in tags.items():
        mlflow.set_tag(k, v)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dataset_lineage.json"
        slim = {
            k: v
            for k, v in lineage.items()
            if k not in {"processed_manifest", "interim_manifest", "raw_manifest"}
        }
        slim["processed_manifest"] = {
            kk: vv
            for kk, vv in (lineage.get("processed_manifest") or {}).items()
            if kk != "feature_names"
        }
        slim["interim_manifest"] = lineage.get("interim_manifest") or {}
        slim["raw_manifest"] = lineage.get("raw_manifest") or {}
        with open(path, "w") as f:
            json.dump(slim, f, indent=2)
        mlflow.log_artifact(str(path), artifact_path="dataset")

    repo = lineage.get("hf_repo_id") or "unknown"
    # MLflow rejects digests longer than 36 chars.
    digest = (tab.get("sha256") or mel.get("sha256") or "unknown")[:36]
    source_url = f"https://huggingface.co/datasets/{repo}"
    dataset = MetaDataset(
        source=HTTPDatasetSource(source_url),
        name="urbansound8k-processed",
        digest=digest,
    )
    mlflow.log_input(dataset, context="training")
    return dataset


def log_metrics(metrics: dict[str, float], *, dataset: MetaDataset | None = None) -> None:
    for k, v in metrics.items():
        if v == v:  # skip NaN
            mlflow.log_metric(k, v, dataset=dataset)


def log_optuna_study(
    study: optuna.Study,
    parent_run_id: str,
    *,
    dataset: MetaDataset | None = None,
) -> None:
    """Full Optuna report on the parent run (trials table + best summary)."""
    mlflow.log_metric("optuna_best_value", float(study.best_value), dataset=dataset)
    mlflow.log_param("optuna_best_trial", int(study.best_trial.number))
    mlflow.log_param("optuna_n_trials", len(study.trials))
    mlflow.set_tag("optuna.best_value", f"{study.best_value:.6f}")
    mlflow.set_tag("optuna.best_trial", str(study.best_trial.number))

    rows = []
    for t in study.trials:
        row = {
            "number": t.number,
            "state": str(t.state),
            "value": t.value,
            **{f"param_{k}": v for k, v in t.params.items()},
        }
        rows.append(row)
        if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE:
            mlflow.log_metric(
                "optuna_val_f1_macro",
                float(t.value),
                step=t.number,
                dataset=dataset,
            )

    frame = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "optuna_trials.csv"
        json_path = Path(tmp) / "optuna_best.json"
        frame.to_csv(csv_path, index=False)
        save_json(
            json_path,
            {
                "best_trial": study.best_trial.number,
                "best_value": float(study.best_value),
                "best_params": study.best_params,
                "parent_run_id": parent_run_id,
            },
        )
        mlflow.log_artifact(str(csv_path), artifact_path="optuna")
        mlflow.log_artifact(str(json_path), artifact_path="optuna")


def register_and_tag(
    model_name: str,
    run_id: str,
    metrics: dict[str, float],
    lineage: dict,
    *,
    is_winner: bool,
) -> None:
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, name=model_name)
    client = MlflowClient()
    client.set_registered_model_alias(model_name, "champion", mv.version)
    tags = {
        "f1_macro": f"{metrics.get('f1_macro', float('nan')):.6f}",
        "winner": "true" if is_winner else "false",
        "hf_repo_id": str(lineage.get("hf_repo_id") or ""),
        "processed_created_at": str(lineage.get("processed_created_at") or ""),
        "tabular_sha256": str((lineage.get("tabular") or {}).get("sha256") or ""),
    }
    for k, v in tags.items():
        if not v:
            continue
        client.set_model_version_tag(model_name, mv.version, k, v)
        client.set_registered_model_tag(model_name, k, v)
    logger.info("Registered %s version %s (winner=%s)", model_name, mv.version, is_winner)


def log_sklearn_family_model(model_name: str, model, signature) -> None:
    # Use artifact_path so runs:/{id}/model works with Model Registry (MLflow 3
    # `name=` alone does not create the classic run artifact tree).
    if model_name == "rf":
        mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)
    elif model_name == "xgboost":
        mlflow.xgboost.log_model(model, artifact_path="model", signature=signature)
    else:
        mlflow.lightgbm.log_model(model, artifact_path="model", signature=signature)
