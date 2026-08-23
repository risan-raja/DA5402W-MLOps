"""Helpers for writing full ``AppConfig`` YAML fixtures in unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.config import _REQUIRED_TOP_LEVEL


def write_app_config(path: Path, **sections: dict[str, Any]) -> Path:
    """Write a YAML config that includes every required top-level section.

    Pass section overrides as kwargs (e.g. ``dataset={"hf_repo_id": "..."}``).
    Unspecified required sections become empty mappings so ``load_app_config``
    accepts the fixture.
    """
    cfg: dict[str, Any] = {key: {} for key in _REQUIRED_TOP_LEVEL}
    for key, value in sections.items():
        if key not in _REQUIRED_TOP_LEVEL:
            raise KeyError(f"unknown config section {key!r}; expected one of {_REQUIRED_TOP_LEVEL}")
        if not isinstance(value, dict):
            raise TypeError(f"section {key!r} must be a mapping, got {type(value).__name__}")
        cfg[key] = value
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path
