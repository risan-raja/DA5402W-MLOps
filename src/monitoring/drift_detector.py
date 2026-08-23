"""KS and PSI drift scores against a fold-10 reference."""

from __future__ import annotations

import json
import math
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

CLASS_NAMES = (
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
)

# Official UrbanSound8K fold-10 counts (UrbanSound8K.csv / Salamon et al.).
OFFICIAL_FOLD10_COUNTS = {
    "air_conditioner": 100,
    "car_horn": 33,
    "children_playing": 100,
    "dog_bark": 100,
    "drilling": 100,
    "engine_idling": 93,
    "gun_shot": 32,
    "jackhammer": 96,
    "siren": 83,
    "street_music": 100,
}

N_QUANTILES = 101
DEFAULT_FEATURE_BINS = 10
PSI_EPS = 1e-6


@dataclass(frozen=True)
class FeatureHistogram:
    edges: tuple[float, ...]
    density: tuple[float, ...]

    def to_dict(self) -> dict:
        return {"edges": list(self.edges), "density": list(self.density)}

    @classmethod
    def from_dict(cls, payload: Mapping) -> FeatureHistogram:
        edges = tuple(float(v) for v in payload["edges"])
        density = tuple(float(v) for v in payload["density"])
        if len(edges) < 2:
            raise ValueError("feature histogram needs at least two edges")
        if len(density) != len(edges) - 1:
            raise ValueError("feature histogram density length must be n_edges - 1")
        return cls(edges=edges, density=density)


@dataclass(frozen=True)
class DriftReference:
    class_prior: dict[str, float]
    class_counts: dict[str, int]
    confidence_quantiles: tuple[float, ...] = ()
    feature_histograms: dict[str, FeatureHistogram] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    class_names: tuple[str, ...] = CLASS_NAMES

    @property
    def has_confidence(self) -> bool:
        return len(self.confidence_quantiles) >= 2

    @property
    def has_features(self) -> bool:
        return bool(self.feature_histograms)

    def to_dict(self) -> dict:
        return {
            "class_prior": dict(self.class_prior),
            "class_counts": dict(self.class_counts),
            "confidence_quantiles": list(self.confidence_quantiles),
            "feature_histograms": {
                name: hist.to_dict() for name, hist in self.feature_histograms.items()
            },
            "meta": dict(self.meta),
            "class_names": list(self.class_names),
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> DriftReference:
        names = tuple(str(n) for n in payload.get("class_names") or CLASS_NAMES)
        raw_prior = payload.get("class_prior") or {}
        raw_counts = payload.get("class_counts") or {}
        counts = {str(k): int(v) for k, v in raw_counts.items()}
        if raw_prior:
            prior = {str(k): float(v) for k, v in raw_prior.items()}
        else:
            prior = proportions_from_counts(counts, names)
        histograms = {
            str(name): FeatureHistogram.from_dict(raw)
            for name, raw in (payload.get("feature_histograms") or {}).items()
        }
        quantiles = tuple(
            float(v) for v in (payload.get("confidence_quantiles") or [])
        )
        meta = dict(payload.get("meta") or {})
        return cls(
            class_prior=prior,
            class_counts=counts,
            confidence_quantiles=quantiles,
            feature_histograms=histograms,
            meta=meta,
            class_names=names,
        )


@dataclass(frozen=True)
class DriftSnapshot:
    psi_class: float
    ks_confidence: float | None
    window_size: int
    n_observations: int


def proportions_from_counts(
    counts: Mapping[str, float],
    keys: Sequence[str],
) -> dict[str, float]:
    aligned = {str(key): float(counts.get(key, 0.0)) for key in keys}
    total = sum(aligned.values())
    if total <= 0:
        raise ValueError("class counts must sum to a positive value")
    return {key: value / total for key, value in aligned.items()}


def class_proportions(
    labels: Sequence[str],
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict[str, float]:
    counts = Counter(str(label) for label in labels)
    return proportions_from_counts(counts, class_names)


def population_stability_index(
    expected: Mapping[str, float],
    actual: Mapping[str, float],
    *,
    eps: float = PSI_EPS,
) -> float:
    keys = sorted(set(expected) | set(actual))
    if not keys:
        raise ValueError("PSI requires at least one category")
    psi = 0.0
    for key in keys:
        exp = max(float(expected.get(key, 0.0)), eps)
        act = max(float(actual.get(key, 0.0)), eps)
        psi += (act - exp) * math.log(act / exp)
    return float(psi)


def ks_statistic(
    ref_samples: Sequence[float],
    live_samples: Sequence[float],
) -> float:
    ref = np.asarray(ref_samples, dtype=np.float64)
    live = np.asarray(live_samples, dtype=np.float64)
    if ref.size == 0 or live.size == 0:
        raise ValueError("KS requires non-empty reference and live samples")
    return float(ks_2samp(ref, live, method="asymp").statistic)


def confidence_quantiles(
    samples: Sequence[float],
    n_quantiles: int = N_QUANTILES,
) -> tuple[float, ...]:
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        return ()
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be >= 2")
    probs = np.linspace(0.0, 1.0, n_quantiles)
    return tuple(float(v) for v in np.quantile(values, probs))


def build_feature_histograms(
    columns: Mapping[str, Sequence[float]],
    *,
    n_bins: int = DEFAULT_FEATURE_BINS,
) -> dict[str, FeatureHistogram]:
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    histograms: dict[str, FeatureHistogram] = {}
    for name, raw in columns.items():
        values = np.asarray(raw, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        counts, edges = np.histogram(values, bins=n_bins)
        total = float(counts.sum())
        if total <= 0:
            continue
        density = tuple(float(c) / total for c in counts)
        histograms[str(name)] = FeatureHistogram(
            edges=tuple(float(e) for e in edges),
            density=density,
        )
    return histograms


def feature_column_psi(
    histogram: FeatureHistogram,
    values: Sequence[float],
    *,
    eps: float = PSI_EPS,
) -> float:
    live = np.asarray(values, dtype=np.float64)
    live = live[np.isfinite(live)]
    if live.size == 0:
        raise ValueError("feature PSI requires non-empty live values")
    edges = np.asarray(histogram.edges, dtype=np.float64)
    clipped = np.clip(live, edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=edges)
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("feature PSI histogram is empty")
    expected = {
        str(i): float(density) for i, density in enumerate(histogram.density)
    }
    actual = {str(i): float(count) / total for i, count in enumerate(counts)}
    return population_stability_index(expected, actual, eps=eps)


def feature_column_ks(
    histogram: FeatureHistogram,
    values: Sequence[float],
) -> float:
    """KS against the midpoints of the reference histogram, weighted by density."""
    live = np.asarray(values, dtype=np.float64)
    live = live[np.isfinite(live)]
    if live.size == 0:
        raise ValueError("feature KS requires non-empty live values")
    edges = np.asarray(histogram.edges, dtype=np.float64)
    mids = 0.5 * (edges[:-1] + edges[1:])
    weights = np.asarray(histogram.density, dtype=np.float64)
    n_ref = max(len(live), 32)
    counts = np.maximum(np.round(weights / weights.sum() * n_ref), 1).astype(int)
    ref = np.repeat(mids, counts)
    return ks_statistic(ref, live)


def score_features(
    histograms: Mapping[str, FeatureHistogram],
    columns: Mapping[str, Sequence[float]],
) -> dict[str, float | str]:
    if not histograms:
        raise ValueError("reference has no feature histograms")
    psi_values: list[float] = []
    ks_values: list[float] = []
    max_psi = -1.0
    max_feature = ""
    for name, histogram in histograms.items():
        if name not in columns:
            continue
        psi = feature_column_psi(histogram, columns[name])
        ks = feature_column_ks(histogram, columns[name])
        psi_values.append(psi)
        ks_values.append(ks)
        if psi > max_psi:
            max_psi = psi
            max_feature = name
    if not psi_values:
        raise ValueError("no overlapping feature columns to score")
    return {
        "n_features": len(psi_values),
        "mean_psi": float(np.mean(psi_values)),
        "max_psi": float(max_psi),
        "max_psi_feature": max_feature,
        "mean_ks": float(np.mean(ks_values)),
    }


def official_fold10_reference(*, extra_meta: Mapping | None = None) -> DriftReference:
    counts = dict(OFFICIAL_FOLD10_COUNTS)
    prior = proportions_from_counts(counts, CLASS_NAMES)
    meta = {
        "eval_fold": 10,
        "n_rows": int(sum(counts.values())),
        "source": "official_urbansound8k_fold10",
        "built_at": datetime.now(UTC).isoformat(),
        "has_confidence": False,
        "has_features": False,
    }
    if extra_meta:
        meta.update(dict(extra_meta))
    return DriftReference(
        class_prior=prior,
        class_counts=counts,
        meta=meta,
    )


def reference_from_labels(
    labels: Sequence[str],
    *,
    class_names: Sequence[str] = CLASS_NAMES,
    extra_meta: Mapping | None = None,
) -> DriftReference:
    counts_raw = Counter(str(label) for label in labels)
    counts = {name: int(counts_raw.get(name, 0)) for name in class_names}
    prior = proportions_from_counts(counts, class_names)
    meta = {
        "eval_fold": 10,
        "n_rows": int(sum(counts.values())),
        "source": "fold10_labels",
        "built_at": datetime.now(UTC).isoformat(),
        "has_confidence": False,
        "has_features": False,
    }
    if extra_meta:
        meta.update(dict(extra_meta))
    return DriftReference(
        class_prior=prior,
        class_counts=counts,
        meta=meta,
        class_names=tuple(class_names),
    )


def with_confidence_samples(
    reference: DriftReference,
    samples: Sequence[float],
) -> DriftReference:
    quantiles = confidence_quantiles(samples)
    meta = dict(reference.meta)
    meta["has_confidence"] = bool(quantiles)
    meta["n_confidence_samples"] = len(samples)
    return DriftReference(
        class_prior=reference.class_prior,
        class_counts=reference.class_counts,
        confidence_quantiles=quantiles,
        feature_histograms=reference.feature_histograms,
        meta=meta,
        class_names=reference.class_names,
    )


def with_feature_histograms(
    reference: DriftReference,
    histograms: Mapping[str, FeatureHistogram],
) -> DriftReference:
    stored = dict(histograms)
    meta = dict(reference.meta)
    meta["has_features"] = bool(stored)
    meta["n_feature_histograms"] = len(stored)
    return DriftReference(
        class_prior=reference.class_prior,
        class_counts=reference.class_counts,
        confidence_quantiles=reference.confidence_quantiles,
        feature_histograms=stored,
        meta=meta,
        class_names=reference.class_names,
    )


def load_reference(path: Path) -> DriftReference:
    with open(path) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected object in {path}")
    return DriftReference.from_dict(payload)


def dump_reference(path: Path, reference: DriftReference) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(reference.to_dict(), handle, indent=2)
        handle.write("\n")


def score_predictions(
    labels: Sequence[str],
    confidences: Sequence[float],
    reference: DriftReference,
) -> DriftSnapshot:
    if not labels:
        raise ValueError("prediction score requires at least one label")
    actual = class_proportions(labels, reference.class_names)
    psi = population_stability_index(reference.class_prior, actual)
    ks: float | None = None
    if reference.has_confidence and confidences:
        ks = ks_statistic(reference.confidence_quantiles, confidences)
    return DriftSnapshot(
        psi_class=psi,
        ks_confidence=ks,
        window_size=len(labels),
        n_observations=len(labels),
    )


class DriftMonitor:
    """Rolling window of live labels and confidences."""

    def __init__(self, window_size: int, reference: DriftReference) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size
        self.reference = reference
        self._labels: deque[str] = deque(maxlen=window_size)
        self._confidences: deque[float] = deque(maxlen=window_size)

    @property
    def filled(self) -> int:
        return len(self._labels)

    def snapshot(self) -> DriftSnapshot:
        return score_predictions(list(self._labels), list(self._confidences), self.reference)

    def update(self, label: str, confidence: float) -> DriftSnapshot | None:
        self._labels.append(str(label))
        self._confidences.append(float(confidence))
        if self.filled < self.window_size:
            return None
        return self.snapshot()
