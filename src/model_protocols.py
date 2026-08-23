"""Protocols for sklearn-family classifiers used in training and serving."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike


@runtime_checkable
class SupportsPredictProba(Protocol):
    """Minimal fit / predict_proba surface shared by RF, XGBoost, LightGBM."""

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> SupportsPredictProba: ...

    def predict_proba(self, X: ArrayLike) -> np.ndarray: ...


@runtime_checkable
class SupportsTransform(Protocol):
    """Scaler-like transform used at tabular serve time."""

    def transform(self, X: ArrayLike) -> np.ndarray: ...
