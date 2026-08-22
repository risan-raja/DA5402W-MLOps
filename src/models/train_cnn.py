"""Train ResNet-18 on mel-spectrograms with Optuna + MLflow."""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import optuna
import torch

from src.models.cnn_model import (
    ensure_resnet18_weights,
    predict_proba as cnn_predict_proba,
    suggest_cnn_params,
    train_resnet,
)
from src.models.data import (
    build_label_maps,
    class_weight_vector,
    encode_labels,
    fit_mel_stats,
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

logger = logging.getLogger(__name__)


def _xy_mels(frame, label_to_id: dict[str, int]):
    y = encode_labels(frame["class"], label_to_id)
    x = mels_array(frame)
    return x, y


def train_resnet_model(
    *,
    train_cfg: dict,
    mels_df,
    n_trials: int,
    seed: int,
    lineage: dict,
) -> dict:
    train_folds = list(train_cfg["train_folds"])
    val_fold = int(train_cfg["val_fold"])
    eval_fold = int(train_cfg["eval_fold"])
    cnn_cfg = train_cfg.get("cnn", {})

    optuna_train, val_df, refit_df, eval_df = split_frames(
        mels_df, train_folds, val_fold, eval_fold
    )
    label_to_id, id_to_label = build_label_maps(optuna_train["class"])
    n_classes = len(label_to_id)
    class_names = [id_to_label[i] for i in range(n_classes)]

    x_tr, y_tr = _xy_mels(optuna_train, label_to_id)
    x_va, y_va = _xy_mels(val_df, label_to_id)
    x_rf, y_rf = _xy_mels(refit_df, label_to_id)
    x_ev, y_ev = _xy_mels(eval_df, label_to_id)

    mel_stats = fit_mel_stats(x_tr)
    x_tr_n = normalize_mels(x_tr, mel_stats)
    x_va_n = normalize_mels(x_va, mel_stats)
    cw = class_weight_vector(y_tr, n_classes)

    epochs = int(cnn_cfg.get("epochs", 30))
    patience = int(cnn_cfg.get("patience", 5))
    default_bs = int(cnn_cfg.get("batch_size", 32))
    default_lr = float(cnn_cfg.get("lr", 1e-4))

    def objective(trial: optuna.Trial) -> float:
        params = suggest_cnn_params(trial, seed=seed)
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.set_tag("optuna.trial_number", str(trial.number))
            mlflow.log_params(jsonable_params(params))
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
            trial.set_user_attr("epochs_run", history["epochs_run"])
            mlflow.log_metrics(
                {
                    "val_f1_macro": float(history["best_val_f1_macro"]),
                    "epochs_run": float(history["epochs_run"]),
                }
            )
            return float(history["best_val_f1_macro"])

    with mlflow.start_run(run_name="resnet18") as parent:
        mlflow.log_param("model_name", "resnet18")
        mlflow.log_param("n_trials", n_trials)
        mlflow.set_tag("model_family", "cnn")
        ml_dataset = log_dataset_lineage(lineage)

        # Prefetch weights once (avoids SSL failures / re-downloads inside Optuna trials).
        ensure_resnet18_weights()

        study = optuna.create_study(
            direction="maximize",
            study_name="resnet18-macro-f1",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        log_optuna_study(study, parent.info.run_id, dataset=ml_dataset)

        best = study.best_trial
        lr = float(best.params.get("lr", default_lr))
        batch_size = int(best.params.get("batch_size", default_bs))
        epochs_run = int(best.user_attrs.get("epochs_run", epochs))

        final_stats = fit_mel_stats(x_rf)
        x_rf_n = normalize_mels(x_rf, final_stats)
        x_ev_n = normalize_mels(x_ev, final_stats)
        cw_rf = class_weight_vector(y_rf, n_classes)

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

        proba = cnn_predict_proba(model, x_ev_n, batch_size=batch_size)
        pred = np.argmax(proba, axis=1)
        metrics = compute_metrics(
            y_ev, pred, y_proba=proba, labels=list(range(n_classes))
        )
        log_metrics(metrics, dataset=ml_dataset)
        mlflow.log_params(
            {"best_lr": lr, "best_batch_size": batch_size, "epochs_run": epochs_run}
        )

        out_dir = Path(train_cfg["models_dir"]) / "resnet18"
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
        cm_path = out_dir / "confusion_matrix.png"
        confusion_matrix_figure(y_ev, pred, class_names, cm_path)
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(out_dir / "mel_stats.json"))
        mlflow.log_artifact(str(out_dir / "label_map.json"))
        mlflow.log_artifact(str(out_dir / "metrics.json"))

        # MLflow 3 defaults to pt2 (torch.export); that format requires input_example.
        example = np.zeros((1, 3, 224, 224), dtype=np.float32)
        model_cpu = model.cpu().eval()
        mlflow.pytorch.log_model(
            model_cpu,
            artifact_path="model",
            serialization_format="pt2",
            input_example=example,
        )

        register_and_tag(
            "resnet18",
            parent.info.run_id,
            metrics,
            lineage,
            is_winner=False,
        )

        return {
            "model_name": "resnet18",
            "run_id": parent.info.run_id,
            "metrics": metrics,
            "out_dir": str(out_dir),
        }
