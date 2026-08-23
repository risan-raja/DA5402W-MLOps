"""Train RF / XGBoost / LightGBM with Optuna + MLflow."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import optuna
from mlflow.models import infer_signature

import mlflow
from src.models.baseline_model import fit_predict_proba, suggest_params
from src.models.cross_validation import aggregate_fold_metrics, log_cv_results
from src.models.data import (
    build_label_maps,
    class_weight_vector,
    feature_columns,
    filter_split,
    fit_scaler,
    iter_us8k_cv_folds,
    sample_weights_from_y,
    split_frames,
    tabular_xy,
    transform_features,
)
from src.models.evaluate import compute_metrics, confusion_matrix_figure
from src.models.mlflow_logging import (
    jsonable_params,
    log_dataset_lineage,
    log_metrics,
    log_optuna_study,
    log_sklearn_family_model,
    register_and_tag,
    save_json,
)

logger = logging.getLogger(__name__)


def train_tabular_model(
    model_name: str,
    *,
    train_cfg: dict,
    spark_cfg: dict,
    tabular,
    n_trials: int,
    seed: int,
    lineage: dict,
) -> dict:
    train_folds = list(train_cfg["train_folds"])
    val_fold = int(train_cfg["val_fold"])
    eval_fold = int(train_cfg["eval_fold"])
    n_mfcc = int(spark_cfg.get("n_mfcc", 13))

    optuna_train, val_df, refit_df, eval_df = split_frames(
        tabular, train_folds, val_fold, eval_fold
    )
    label_to_id, id_to_label = build_label_maps(optuna_train["class"])
    n_classes = len(label_to_id)
    class_names = [id_to_label[i] for i in range(n_classes)]
    feat_cols = feature_columns(tabular, n_mfcc=n_mfcc)

    x_tr, y_tr = tabular_xy(optuna_train, feat_cols, label_to_id)
    x_va, y_va = tabular_xy(val_df, feat_cols, label_to_id)
    x_rf, y_rf = tabular_xy(refit_df, feat_cols, label_to_id)
    x_ev, y_ev = tabular_xy(eval_df, feat_cols, label_to_id)

    scaler = fit_scaler(x_tr)
    x_tr_s = transform_features(scaler, x_tr)
    x_va_s = transform_features(scaler, x_va)
    cw = class_weight_vector(y_tr, n_classes)
    sw = sample_weights_from_y(y_tr, cw)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(model_name, trial, seed=seed)
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.set_tag("optuna.trial_number", str(trial.number))
            mlflow.log_params(jsonable_params(params))
            _, pred, _ = fit_predict_proba(
                model_name,
                params,
                x_tr_s,
                y_tr,
                x_va_s,
                sample_weight=sw,
                n_classes=n_classes,
            )
            metrics = compute_metrics(y_va, pred)
            mlflow.log_metrics(
                {
                    "val_f1_macro": metrics["f1_macro"],
                    "val_accuracy": metrics["accuracy"],
                    "val_f1_weighted": metrics["f1_weighted"],
                }
            )
            return metrics["f1_macro"]

    with mlflow.start_run(run_name=model_name) as parent:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_trials", n_trials)
        mlflow.set_tag("model_family", "tabular")
        ml_dataset = log_dataset_lineage(lineage, model_name=model_name)

        study = optuna.create_study(
            direction="maximize",
            study_name=f"{model_name}-macro-f1",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        log_optuna_study(study, parent.info.run_id, dataset=ml_dataset)

        best_params = suggest_params(
            model_name,
            optuna.trial.FixedTrial(study.best_params),
            seed=seed,
        )

        final_scaler = fit_scaler(x_rf)
        x_rf_s = transform_features(final_scaler, x_rf)
        x_ev_s = transform_features(final_scaler, x_ev)
        cw_rf = class_weight_vector(y_rf, n_classes)
        sw_rf = sample_weights_from_y(y_rf, cw_rf)

        model, pred, proba = fit_predict_proba(
            model_name,
            best_params,
            x_rf_s,
            y_rf,
            x_ev_s,
            sample_weight=sw_rf,
            n_classes=n_classes,
        )
        metrics = compute_metrics(
            y_ev, pred, y_proba=proba, labels=list(range(n_classes))
        )
        log_metrics(metrics, dataset=ml_dataset)
        mlflow.log_params(
            {f"best_{k}": v for k, v in jsonable_params(best_params).items()}
        )

        out_dir = Path(train_cfg["models_dir"]) / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "model.joblib"
        scaler_path = out_dir / "scaler.joblib"
        joblib.dump(model, model_path)
        joblib.dump(final_scaler, scaler_path)
        save_json(
            out_dir / "label_map.json",
            {
                "label_to_id": label_to_id,
                "id_to_label": {str(k): v for k, v in id_to_label.items()},
            },
        )
        save_json(out_dir / "metrics.json", metrics)
        save_json(out_dir / "params.json", jsonable_params(best_params))
        save_json(
            out_dir / "dataset_lineage.json",
            {
                k: v
                for k, v in lineage.items()
                if k not in {"processed_manifest", "interim_manifest", "raw_manifest"}
            },
        )

        cm_path = out_dir / "confusion_matrix.png"
        confusion_matrix_figure(y_ev, pred, class_names, cm_path)
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(scaler_path))
        mlflow.log_artifact(str(out_dir / "label_map.json"))
        mlflow.log_artifact(str(out_dir / "metrics.json"))

        signature = infer_signature(x_ev_s[:5], proba[:5])
        log_sklearn_family_model(model_name, model, signature)


        cv_cfg = train_cfg.get("cv") or {}
        if cv_cfg.get("enabled"):
            n_folds = int(cv_cfg.get("n_folds", 10))
            # Global label map so fold-local class gaps do not renumber IDs.
            cv_label_to_id, _ = build_label_maps(tabular["class"])
            cv_n_classes = len(cv_label_to_id)
            fold_rows: list[dict] = []
            logger.info(
                "Running UrbanSound8K %s-fold eval CV for %s", n_folds, model_name
            )
            for test_fold, cv_train_folds in iter_us8k_cv_folds(n_folds):
                train_df = filter_split(
                    tabular, cv_train_folds, include_augmented=True
                )
                test_df = filter_split(
                    tabular, [test_fold], include_augmented=False
                )
                x_tr_cv, y_tr_cv = tabular_xy(train_df, feat_cols, cv_label_to_id)
                x_te_cv, y_te_cv = tabular_xy(test_df, feat_cols, cv_label_to_id)
                fold_scaler = fit_scaler(x_tr_cv)
                x_tr_cv_s = transform_features(fold_scaler, x_tr_cv)
                x_te_cv_s = transform_features(fold_scaler, x_te_cv)
                cw_cv = class_weight_vector(y_tr_cv, cv_n_classes)
                sw_cv = sample_weights_from_y(y_tr_cv, cw_cv)
                _, pred_cv, proba_cv = fit_predict_proba(
                    model_name,
                    best_params,
                    x_tr_cv_s,
                    y_tr_cv,
                    x_te_cv_s,
                    sample_weight=sw_cv,
                    n_classes=cv_n_classes,
                )
                fold_metrics = compute_metrics(
                    y_te_cv,
                    pred_cv,
                    y_proba=proba_cv,
                    labels=list(range(cv_n_classes)),
                )
                fold_rows.append({"fold": int(test_fold), **fold_metrics})
            aggregate = aggregate_fold_metrics(fold_rows)
            metrics.update(aggregate)
            log_cv_results(fold_rows, aggregate, out_dir)
            save_json(out_dir / "metrics.json", metrics)
            mlflow.log_artifact(str(out_dir / "metrics.json"))

        register_and_tag(
            model_name,
            parent.info.run_id,
            metrics,
            lineage,
            is_winner=False,
        )

        return {
            "model_name": model_name,
            "run_id": parent.info.run_id,
            "metrics": metrics,
            "out_dir": str(out_dir),
        }
