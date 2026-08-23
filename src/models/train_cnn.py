"""Train ResNet-18 on mel-spectrograms with Optuna + MLflow."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import cast

import mlflow
import mlflow.pytorch
import numpy as np
import optuna
import pandas as pd
import torch

from src.artifact_types import (
    CnnHistory,
    CnnSuggestParams,
    DatasetLineage,
    FoldMetricRow,
    MelStats,
    TrainResult,
)
from src.config_types import CnnTrainingConfig, TrainingConfig, TrainingCvConfig
from src.models.cnn_model import (
    ensure_resnet18_weights,
    suggest_cnn_params,
    train_resnet,
)
from src.models.cnn_model import (
    predict_proba as cnn_predict_proba,
)
from src.models.cross_validation import aggregate_fold_metrics, log_cv_results
from src.models.data import (
    build_label_maps,
    class_weight_vector,
    encode_labels,
    filter_split,
    fit_mel_stats,
    iter_us8k_cv_folds,
    mels_array,
    normalize_mels,
    split_frames,
)
from src.models.evaluate import compute_metrics, confusion_matrix_figure
from src.models.mlflow_logging import (
    jsonable_params,
    log_dataset_lineage,
    log_metrics,
    log_optuna_study,
    register_and_tag,
    save_json,
)

logger: logging.Logger = logging.getLogger(__name__)


def _xy_mels(
    frame: pd.DataFrame, label_to_id: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    y: np.ndarray = encode_labels(frame["class"], label_to_id)
    x: np.ndarray = mels_array(frame)
    return x, y


def train_resnet_model(
    *,
    train_cfg: TrainingConfig | dict[str, object],
    mels_df: pd.DataFrame,
    n_trials: int,
    seed: int,
    lineage: DatasetLineage,
) -> TrainResult:
    train_folds: list[int] = list(train_cfg["train_folds"])
    val_fold: int = int(train_cfg["val_fold"])
    eval_fold: int = int(train_cfg["eval_fold"])
    cnn_cfg: CnnTrainingConfig | dict[str, object] = cast(
        CnnTrainingConfig | dict[str, object], train_cfg.get("cnn", {})
    )

    optuna_train, val_df, refit_df, eval_df = split_frames(
        mels_df, train_folds, val_fold, eval_fold
    )
    label_to_id, id_to_label = build_label_maps(optuna_train["class"])
    n_classes: int = len(label_to_id)
    class_names: list[str] = [id_to_label[i] for i in range(n_classes)]

    x_tr, y_tr = _xy_mels(optuna_train, label_to_id)
    x_va, y_va = _xy_mels(val_df, label_to_id)
    x_rf, y_rf = _xy_mels(refit_df, label_to_id)
    x_ev, y_ev = _xy_mels(eval_df, label_to_id)

    mel_stats: MelStats = fit_mel_stats(x_tr)
    x_tr_n: np.ndarray = normalize_mels(x_tr, mel_stats)
    x_va_n: np.ndarray = normalize_mels(x_va, mel_stats)
    cw: np.ndarray = class_weight_vector(y_tr, n_classes)

    epochs: int = int(cnn_cfg.get("epochs", 30))
    patience: int = int(cnn_cfg.get("patience", 5))
    default_bs: int = int(cnn_cfg.get("batch_size", 32))
    default_lr: float = float(cnn_cfg.get("lr", 1e-4))

    def objective(trial: optuna.Trial) -> float:
        params: CnnSuggestParams = suggest_cnn_params(trial, seed=seed)
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.set_tag("optuna.trial_number", str(trial.number))
            mlflow.log_params(jsonable_params(dict(params)))
            _, history = train_resnet(
                x_tr_n,
                y_tr,
                x_va_n,
                y_va,
                class_weights=cw,
                epochs=epochs,
                batch_size=int(params["batch_size"]),
                lr=float(params["lr"]),
                patience=patience,
                seed=seed,
                pretrained=True,
                early_stop=True,
            )
            history_typed: CnnHistory = history
            trial.set_user_attr("epochs_run", history_typed["epochs_run"])
            # Same val metric names as tabular Optuna trials (required for UI compare).
            trial_metrics: dict[str, float] = {
                "val_f1_macro": float(history_typed["best_val_f1_macro"]),
                "val_accuracy": float(history_typed["best_val_accuracy"]),
                "val_f1_weighted": float(history_typed["best_val_f1_weighted"]),
            }
            missing: list[str] = [k for k, v in trial_metrics.items() if math.isnan(v)]
            if missing:
                raise RuntimeError(
                    f"resnet Optuna trial missing finite val metrics: {missing}; "
                    f"history={history_typed}"
                )
            mlflow.log_metrics(
                {
                    **trial_metrics,
                    "epochs_run": float(history_typed["epochs_run"]),
                }
            )
            return trial_metrics["val_f1_macro"]

    with mlflow.start_run(run_name="resnet18") as parent:
        mlflow.log_param("model_name", "resnet18")
        mlflow.log_param("n_trials", n_trials)
        mlflow.set_tag("model_family", "cnn")
        ml_dataset = log_dataset_lineage(lineage, model_name="resnet18")

        # Prefetch weights once (avoids SSL failures / re-downloads inside Optuna trials).
        ensure_resnet18_weights()

        study: optuna.Study = optuna.create_study(
            direction="maximize",
            study_name="resnet18-macro-f1",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        log_optuna_study(study, parent.info.run_id, dataset=ml_dataset)

        best = study.best_trial
        lr: float = float(best.params.get("lr", default_lr))
        batch_size: int = int(best.params.get("batch_size", default_bs))
        epochs_run: int = int(best.user_attrs.get("epochs_run", epochs))

        final_stats: MelStats = fit_mel_stats(x_rf)
        x_rf_n: np.ndarray = normalize_mels(x_rf, final_stats)
        x_ev_n: np.ndarray = normalize_mels(x_ev, final_stats)
        cw_rf: np.ndarray = class_weight_vector(y_rf, n_classes)

        model, _ = train_resnet(
            x_rf_n,
            y_rf,
            None,
            None,
            class_weights=cw_rf,
            epochs=epochs_run,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            seed=seed,
            pretrained=True,
            early_stop=False,
        )

        proba: np.ndarray = cnn_predict_proba(model, x_ev_n, batch_size=batch_size)
        pred: np.ndarray = np.argmax(proba, axis=1)
        metrics: dict[str, float] = compute_metrics(
            y_ev, pred, y_proba=proba, labels=list(range(n_classes))
        )
        log_metrics(metrics, dataset=ml_dataset)
        mlflow.log_params(
            {"best_lr": lr, "best_batch_size": batch_size, "epochs_run": epochs_run}
        )

        out_dir: Path = Path(train_cfg["models_dir"]) / "resnet18"
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict(), "n_classes": n_classes},
            out_dir / "model.pt",
        )
        save_json(out_dir / "mel_stats.json", final_stats)
        save_json(
            out_dir / "label_map.json",
            {
                "label_to_id": label_to_id,
                "id_to_label": {str(k): v for k, v in id_to_label.items()},
            },
        )
        save_json(out_dir / "metrics.json", metrics)
        save_json(
            out_dir / "params.json",
            {"lr": lr, "batch_size": batch_size, "epochs_run": epochs_run},
        )
        cm_path: Path = out_dir / "confusion_matrix.png"
        confusion_matrix_figure(y_ev, pred, class_names, cm_path)
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(out_dir / "mel_stats.json"))
        mlflow.log_artifact(str(out_dir / "label_map.json"))
        mlflow.log_artifact(str(out_dir / "metrics.json"))

        # MLflow 3 defaults to pt2 (torch.export); that format requires input_example.
        example: np.ndarray = np.zeros((1, 3, 224, 224), dtype=np.float32)
        model_cpu = model.cpu().eval()
        mlflow.pytorch.log_model(
            model_cpu,
            name="model",
            serialization_format="pt2",
            input_example=example,
        )

        cv_cfg: TrainingCvConfig | dict[str, object] = cast(
            TrainingCvConfig | dict[str, object], train_cfg.get("cv") or {}
        )
        if cv_cfg.get("enabled"):
            n_folds: int = int(cv_cfg.get("n_folds", 10))
            cv_label_to_id, _ = build_label_maps(mels_df["class"])
            cv_n_classes: int = len(cv_label_to_id)
            fold_rows: list[FoldMetricRow] = []
            logger.info("Running UrbanSound8K %s-fold eval CV for resnet18", n_folds)
            for test_fold, cv_train_folds in iter_us8k_cv_folds(n_folds):
                train_df: pd.DataFrame = filter_split(
                    mels_df, cv_train_folds, include_augmented=True
                )
                test_df: pd.DataFrame = filter_split(
                    mels_df, [test_fold], include_augmented=False
                )
                x_tr_cv, y_tr_cv = _xy_mels(train_df, cv_label_to_id)
                x_te_cv, y_te_cv = _xy_mels(test_df, cv_label_to_id)
                fold_stats: MelStats = fit_mel_stats(x_tr_cv)
                x_tr_cv_n: np.ndarray = normalize_mels(x_tr_cv, fold_stats)
                x_te_cv_n: np.ndarray = normalize_mels(x_te_cv, fold_stats)
                cw_cv: np.ndarray = class_weight_vector(y_tr_cv, cv_n_classes)
                fold_model, _ = train_resnet(
                    x_tr_cv_n,
                    y_tr_cv,
                    None,
                    None,
                    class_weights=cw_cv,
                    epochs=epochs_run,
                    batch_size=batch_size,
                    lr=lr,
                    patience=patience,
                    seed=seed,
                    pretrained=True,
                    early_stop=False,
                )
                proba_cv: np.ndarray = cnn_predict_proba(
                    fold_model, x_te_cv_n, batch_size=batch_size
                )
                pred_cv: np.ndarray = np.argmax(proba_cv, axis=1)
                fold_metrics: dict[str, float] = compute_metrics(
                    y_te_cv,
                    pred_cv,
                    y_proba=proba_cv,
                    labels=list(range(cv_n_classes)),
                )
                fold_rows.append(
                    cast(FoldMetricRow, {"fold": int(test_fold), **fold_metrics})
                )
            aggregate: dict[str, float] = aggregate_fold_metrics(fold_rows)
            metrics.update(aggregate)
            log_cv_results(fold_rows, aggregate, out_dir)
            save_json(out_dir / "metrics.json", metrics)
            mlflow.log_artifact(str(out_dir / "metrics.json"))

        register_and_tag(
            "resnet18",
            parent.info.run_id,
            metrics,
            lineage,
            is_winner=False,
        )

        result: TrainResult = {
            "model_name": "resnet18",
            "run_id": parent.info.run_id,
            "metrics": metrics,
            "out_dir": str(out_dir),
        }
        return result
