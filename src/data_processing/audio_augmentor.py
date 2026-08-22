"""Train-fold waveform augmentation via audiomentations."""

from __future__ import annotations

import hashlib
import random

import numpy as np
from audiomentations import AddGaussianNoise, Compose, PitchShift, TimeStretch


def build_augmentations(
    gaussian_noise_p: float = 0.5,
    pitch_shift_p: float = 0.5,
    time_stretch_p: float = 0.5,
) -> Compose:
    return Compose(
        transforms=[
            AddGaussianNoise(
                min_amplitude=0.001,
                max_amplitude=0.015,
                p=gaussian_noise_p,
            ),
            PitchShift(
                min_semitones=-2.0,
                max_semitones=2.0,
                p=pitch_shift_p,
            ),
            TimeStretch(
                min_rate=0.9,
                max_rate=1.1,
                p=time_stretch_p,
            ),
        ],
        shuffle=False,
    )


def _clip_seed(base_seed: int, key: str, copy_index: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{key}:{copy_index}".encode()).hexdigest()
    return int(digest[:8], 16)


def augment_waveform(
    y: np.ndarray,
    sample_rate: int,
    *,
    key: str = "",
    copy_index: int = 0,
    base_seed: int = 42,
    augmentations: Compose | None = None,
) -> np.ndarray:
    """Return one augmented copy. RNG seed is derived from base_seed + key + copy_index."""
    seed = _clip_seed(base_seed, key, copy_index)
    random.seed(seed)
    np.random.seed(seed)
    pipeline = augmentations or build_augmentations()
    out = pipeline(samples=y.astype(np.float32, copy=False), sample_rate=sample_rate)
    return np.asarray(out, dtype=np.float32)
