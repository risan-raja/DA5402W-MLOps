"""Resolve and load the winning model for serving."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import torch
import yaml
from huggingface_hub import snapshot_download

from src.models.cnn_model import build_resnet18
from src.models.runtime_env import ROOT
from src.monitoring.drift_detector import DriftMonitor, load_reference

logger = logging.getLogger(__name__)

CNN_NAMES = frozenset({"resnet18"})
TABULAR_NAMES = frozenset({"rf", "xgboost", "lightgbm"})
CNN_FILES = ("model.pt", "mel_stats.json", "label_map.json")
TABULAR_FILES = ("model.joblib", "scaler.joblib", "label_map.json")
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "da5402w"


@dataclass
class LoadedModel:
    model_name: str
    family: str
    artifact_dir: Path
    id_to_label: dict[int, str]
    torch_model: Any = None
    mel_stats: dict[str, float] | None = None
    sklearn_model: Any = None
    scaler: Any = None


@dataclass
class ServingState:
    model: LoadedModel | None
    config: dict
    models_dir: Path
    error: str | None = None
    drift_monitor: DriftMonitor | None = None


def config_path() -> Path:
    override = os.environ.get("DA5402W_CONFIG")
    if override:
        return Path(override)
    return ROOT / "config" / "config.yaml"


def models_dir() -> Path:
    override = os.environ.get("DA5402W_MODELS_DIR")
    if override:
        return Path(override)
    return ROOT / "models"


def cache_dir() -> Path:
    override = os.environ.get("DA5402W_MODEL_CACHE")
    if override:
        return Path(override)
    return DEFAULT_CACHE_DIR


def pull_enabled() -> bool:
    return os.environ.get("DA5402W_PULL_WINNER", "1") != "0"


def load_config(path: Path | None = None) -> dict:
    path = path or config_path()
    if not path.is_file():
        raise FileNotFoundError(f"config not mounted or missing: {path}")
    with open(path) as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"invalid config: {path}")
    return loaded


def _read_json(path: Path) -> dict:
    with open(path) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected object in {path}")
    return payload


def _label_map(path: Path) -> dict[int, str]:
    raw = _read_json(path)["id_to_label"]
    return {int(key): str(value) for key, value in raw.items()}


def _dir_complete(directory: Path, filenames: tuple[str, ...]) -> bool:
    return directory.is_dir() and all(
        (directory / name).is_file() for name in filenames
    )


def family_for_name(model_name: str) -> str:
    if model_name in CNN_NAMES:
        return "cnn"
    if model_name in TABULAR_NAMES:
        return "tabular"
    raise ValueError(f"unknown model family for {model_name!r}")


def infer_family(directory: Path) -> str | None:
    if _dir_complete(directory, CNN_FILES):
        return "cnn"
    if _dir_complete(directory, TABULAR_FILES):
        return "tabular"
    return None


def winner_payload(models_root: Path) -> dict | None:
    path = models_root / "winner.json"
    if not path.is_file():
        return None
    return _read_json(path)


def load_artifact_dir(directory: Path, model_name: str) -> LoadedModel:
    family = family_for_name(model_name)
    inferred = infer_family(directory)
    if inferred is None:
        raise FileNotFoundError(f"incomplete artifact dir: {directory}")
    if inferred != family:
        raise ValueError(
            f"{directory} looks like {inferred} but winner is {model_name} ({family})"
        )
    id_to_label = _label_map(directory / "label_map.json")
    if family == "cnn":
        ckpt = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        n_classes = int(ckpt["n_classes"])
        model = build_resnet18(n_classes=n_classes, pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        stats = _read_json(directory / "mel_stats.json")
        return LoadedModel(
            model_name=model_name,
            family=family,
            artifact_dir=directory,
            id_to_label=id_to_label,
            torch_model=model,
            mel_stats={"mean": float(stats["mean"]), "std": float(stats["std"])},
        )
    return LoadedModel(
        model_name=model_name,
        family=family,
        artifact_dir=directory,
        id_to_label=id_to_label,
        sklearn_model=joblib.load(directory / "model.joblib"),
        scaler=joblib.load(directory / "scaler.joblib"),
    )


def _named_dir(models_root: Path, model_name: str) -> Path | None:
    candidate = models_root / model_name
    family = family_for_name(model_name)
    needed = CNN_FILES if family == "cnn" else TABULAR_FILES
    if _dir_complete(candidate, needed):
        return candidate
    return None


def download_winner(
    dest: Path,
    *,
    config: dict,
    token: str | None = None,
) -> Path:
    vcfg = config.get("versioning") or {}
    repo_id = vcfg.get("hf_model_repo_id")
    if not repo_id:
        raise ValueError("versioning.hf_model_repo_id is not set")
    repo_type = str(vcfg.get("hf_model_repo_type", "model"))
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=str(repo_id),
        repo_type=repo_type,
        local_dir=str(dest),
        allow_patterns=["winner.json", "winner/**"],
        token=token,
    )
    logger.info("Downloaded winner artifacts from %s into %s", repo_id, dest)
    return dest


def resolve_artifact_dir(
    models_root: Path,
    *,
    config: dict,
    pull: bool,
    cache: Path,
) -> tuple[Path, str]:
    """Return (artifact_dir, model_name)."""
    payload = winner_payload(models_root)
    winner_dir = models_root / "winner"
    winner_family = infer_family(winner_dir)
    if payload and winner_family is not None:
        return winner_dir, str(payload["model_name"])
    if winner_family is not None:
        name = "resnet18" if winner_family == "cnn" else "rf"
        return winner_dir, name
    if payload:
        named = _named_dir(models_root, str(payload["model_name"]))
        if named is not None:
            return named, str(payload["model_name"])
    if pull:
        download_winner(cache, config=config, token=os.environ.get("HF_TOKEN"))
        cache_payload = winner_payload(cache)
        cache_winner = cache / "winner"
        if infer_family(cache_winner) is None:
            raise FileNotFoundError(f"Hub snapshot missing winner/ under {cache}")
        name = (
            str(cache_payload["model_name"])
            if cache_payload
            else ("resnet18" if infer_family(cache_winner) == "cnn" else "rf")
        )
        return cache_winner, name
    raise FileNotFoundError(
        f"no complete winner artifacts under {models_root}; "
        "run `make pull-winner` or mount models/"
    )


def drift_enabled(config: dict) -> bool:
    dcfg = (config.get("monitoring") or {}).get("drift") or {}
    return bool(dcfg.get("enabled", True))


def drift_window_size(config: dict) -> int:
    raw = os.environ.get("DA5402W_DRIFT_WINDOW")
    if raw:
        return max(1, int(raw))
    dcfg = (config.get("monitoring") or {}).get("drift") or {}
    return max(1, int(dcfg.get("window_size", 100)))


def resolve_drift_reference_path(config: dict, models_root: Path) -> Path:
    dcfg = (config.get("monitoring") or {}).get("drift") or {}
    raw = Path(dcfg.get("reference_path", "models/drift_reference.json"))
    if raw.is_absolute():
        return raw
    candidates = (models_root / raw.name, models_root / raw, ROOT / raw)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return models_root / raw.name


def load_drift_monitor(config: dict, models_root: Path) -> DriftMonitor | None:
    if not drift_enabled(config):
        return None
    path = resolve_drift_reference_path(config, models_root)
    if not path.is_file():
        logger.warning("drift reference missing at %s; drift gauges disabled", path)
        return None
    try:
        reference = load_reference(path)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        logger.warning("failed to load drift reference %s: %s", path, exc)
        return None
    return DriftMonitor(drift_window_size(config), reference)


def load_serving_state(
    *,
    config: dict | None = None,
    models_root: Path | None = None,
    cache: Path | None = None,
    pull: bool | None = None,
) -> ServingState:
    models_root = models_root or models_dir()
    cache = cache or cache_dir()
    pull = pull_enabled() if pull is None else pull
    try:
        cfg = config if config is not None else load_config()
    except (OSError, TypeError, ValueError) as exc:
        logger.exception("Failed to load serving config")
        return ServingState(
            model=None,
            config={},
            models_dir=models_root,
            error=str(exc),
        )
    try:
        artifact_dir, model_name = resolve_artifact_dir(
            models_root, config=cfg, pull=pull, cache=cache
        )
        loaded = load_artifact_dir(artifact_dir, model_name)
        logger.info(
            "Loaded %s (%s) from %s", loaded.model_name, loaded.family, artifact_dir
        )
        return ServingState(
            model=loaded,
            config=cfg,
            models_dir=models_root,
            drift_monitor=load_drift_monitor(cfg, models_root),
        )
    except Exception as exc:
        logger.exception("Winner model not loaded")
        return ServingState(
            model=None,
            config=cfg,
            models_dir=models_root,
            error=str(exc),
            drift_monitor=load_drift_monitor(cfg, models_root),
        )
