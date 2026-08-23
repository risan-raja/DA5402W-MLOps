"""Classification metrics and confusion-matrix artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_proba: np.ndarray | None = None,
    labels: list[int] | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }
    if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[0] == len(y_true):
        try:
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(
                    y_true,
                    y_proba,
                    multi_class="ovr",
                    average="macro",
                    labels=labels,
                )
            )
        except ValueError:
            metrics["roc_auc_ovr"] = float("nan")
    return metrics


def confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_path: Path | str | None = None,
) -> Path | None:
    cm: np.ndarray = confusion_matrix(
        y_true, y_pred, labels=list(range(len(class_names)))
    )
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(8, 7))
    im: AxesImage = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    if out_path is None:
        plt.close(fig)
        return None
    dest: Path = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    return dest
