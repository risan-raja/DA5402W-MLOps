"""ImageNet ResNet-18 on log-mel spectrograms ({log, delta, delta-delta})."""

from __future__ import annotations

import copy
import logging
import ssl
import urllib.request
from pathlib import Path

import certifi
import librosa
import numpy as np
import torch
import torch.nn as nn  # noqa: PLR0402
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

logger = logging.getLogger(__name__)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def mel_to_3ch(mel: np.ndarray) -> np.ndarray:
    """Stack log-mel, delta, delta-delta → (3, H, W)."""
    mel = np.asarray(mel, dtype=np.float32)
    delta = librosa.feature.delta(mel)
    delta2 = librosa.feature.delta(mel, order=2)
    return np.stack([mel, delta, delta2], axis=0).astype(np.float32)


def _resnet18_weight_path() -> Path:
    url = ResNet18_Weights.IMAGENET1K_V1.url
    filename = url.rsplit("/", 1)[-1]
    cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / filename


def ensure_resnet18_weights() -> Path:
    """Download ImageNet ResNet-18 weights using certifi CA bundle (macOS Python.org SSL)."""
    cached = _resnet18_weight_path()
    if cached.is_file() and cached.stat().st_size > 0:
        return cached

    url = ResNet18_Weights.IMAGENET1K_V1.url
    logger.info("Downloading ResNet-18 ImageNet weights to %s", cached)
    ctx = ssl.create_default_context(cafile=certifi.where())
    tmp = cached.with_suffix(cached.suffix + ".tmp")
    try:
        with (
            urllib.request.urlopen(url, context=ctx, timeout=120) as resp,
            open(tmp, "wb") as out,
        ):
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        tmp.replace(cached)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    return cached


def build_resnet18(n_classes: int = 10, pretrained: bool = True) -> nn.Module:
    model = resnet18(weights=None)
    if pretrained:
        path = ensure_resnet18_weights()
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


class MelDataset(Dataset):
    def __init__(self, mels: np.ndarray, labels: np.ndarray):
        self.mels = mels
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(mel_to_3ch(self.mels[idx]))
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def _resize_batch(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    mels: np.ndarray,
    *,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or resolve_device()
    model = model.to(device)
    model.eval()
    loader = DataLoader(
        MelDataset(mels, np.zeros(len(mels), dtype=np.int64)),
        batch_size=batch_size,
        shuffle=False,
        # num_workers=2,
        # pin_memory=True,
    )
    probs: list[np.ndarray] = []
    for xb, _ in loader:
        xb = _resize_batch(xb.to(device))
        logits = model(xb)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    if not probs:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(probs, axis=0)


def train_resnet(
    mels_train: np.ndarray,
    y_train: np.ndarray,
    mels_val: np.ndarray | None,
    y_val: np.ndarray | None,
    *,
    class_weights: np.ndarray,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
    patience: int = 5,
    seed: int = 42,
    pretrained: bool = True,
    device: torch.device | None = None,
    early_stop: bool = True,
) -> tuple[nn.Module, dict[str, float]]:
    device = device or resolve_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_classes = int(class_weights.shape[0])
    model = build_resnet18(n_classes=n_classes, pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    train_loader = DataLoader(
        MelDataset(mels_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        # num_workers=4,
        # pin_memory=True,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    stale = 0
    history = {
        "best_val_f1_macro": 0.0,
        "best_val_accuracy": float("nan"),
        "best_val_f1_weighted": float("nan"),
        "epochs_run": 0,
    }

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = _resize_batch(xb.to(device))
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        history["epochs_run"] = epoch + 1
        if not early_stop or mels_val is None or y_val is None or len(y_val) == 0:
            best_state = copy.deepcopy(model.state_dict())
            continue

        proba = predict_proba(model, mels_val, batch_size=batch_size, device=device)
        pred = np.argmax(proba, axis=1)
        val_f1 = float(f1_score(y_val, pred, average="macro", zero_division=0))
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    if early_stop and mels_val is not None and y_val is not None and len(y_val) > 0:
        # Recompute on the restored best checkpoint so trial metrics always match.
        proba = predict_proba(model, mels_val, batch_size=batch_size, device=device)
        pred = np.argmax(proba, axis=1)
        history["best_val_f1_macro"] = float(
            f1_score(y_val, pred, average="macro", zero_division=0)
        )
        history["best_val_accuracy"] = float(accuracy_score(y_val, pred))
        history["best_val_f1_weighted"] = float(
            f1_score(y_val, pred, average="weighted", zero_division=0)
        )
    else:
        history["best_val_f1_macro"] = float("nan")
        history["best_val_accuracy"] = float("nan")
        history["best_val_f1_weighted"] = float("nan")
    return model, history


def suggest_cnn_params(trial, seed: int = 42) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-5, 3e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
        "seed": seed,
    }
