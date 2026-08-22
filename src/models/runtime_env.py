"""Load repo .env and apply OpenMP/thread defaults before heavy ML imports."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

# Avoid OpenMP double-init segfaults (sklearn ↔ xgboost ↔ lightgbm on macOS).
_THREAD_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
}


def load_runtime_env(*, env_path: Path | None = None) -> Path:
    """Load ``.env`` (no override of existing vars), then setdefault thread knobs.

    Call this before importing sklearn / xgboost / lightgbm / torch.
    Returns the path that was considered for dotenv loading.
    """
    path = env_path if env_path is not None else ROOT / ".env"
    load_dotenv(path, override=False)
    for key, value in _THREAD_DEFAULTS.items():
        os.environ.setdefault(key, value)
    return path
