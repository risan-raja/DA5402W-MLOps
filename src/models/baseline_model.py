"""Tabular baselines: Random Forest, XGBoost, LightGBM + Optuna spaces."""

from __future__ import annotations

import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.model_protocols import SupportsPredictProba

TABULAR_MODELS = ("rf", "xgboost", "lightgbm")
# Keep single-threaded: sklearn/xgboost/lightgbm OpenMP stacks segfault on macOS
# when n_jobs=-1 after a previous model family has already loaded libomp.
_N_JOBS = 1


def suggest_params(
    model_name: str, trial: optuna.Trial, seed: int = 42
) -> dict[str, object]:
    if model_name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 24),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.5]
            ),
            "random_state": seed,
            "n_jobs": _N_JOBS,
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "random_state": seed,
            "n_jobs": _N_JOBS,
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "objective": "multiclass",
            "random_state": seed,
            "n_jobs": _N_JOBS,
            "verbosity": -1,
        }
    raise ValueError(f"unknown tabular model '{model_name}'")


def build_model(
    model_name: str, params: dict[str, object], n_classes: int
) -> SupportsPredictProba:
    params = dict(params)
    if model_name == "rf":
        return RandomForestClassifier(**params)
    if model_name == "xgboost":
        params.setdefault("num_class", n_classes)
        return XGBClassifier(**params)
    if model_name == "lightgbm":
        params.setdefault("num_class", n_classes)
        return LGBMClassifier(**params)
    raise ValueError(f"unknown tabular model '{model_name}'")


def fit_predict_proba(
    model_name: str,
    params: dict[str, object],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    sample_weight: np.ndarray | None,
    n_classes: int,
) -> tuple[SupportsPredictProba, np.ndarray, np.ndarray]:
    model: SupportsPredictProba = build_model(model_name, params, n_classes=n_classes)
    model.fit(x_train, y_train, sample_weight=sample_weight)
    proba: np.ndarray = model.predict_proba(x_eval)
    pred: np.ndarray = np.argmax(proba, axis=1)
    return model, pred, proba
