"""ImageNet-1k loaders for timm DeiT-Tiny (surgery/distill/PTQ)."""

from __future__ import annotations

import pathlib
from typing import Any, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets

from transformer_surgery.internal.util import make_loaders
from transformer_surgery.models.pet import build_vit_transforms

IMAGENET_NUM_CLASSES = 1000


def build_imagenet_loaders(cfg: Any) -> Tuple[DataLoader, DataLoader]:
    """Train/val ImageNet-1k loaders rooted at ``cfg.data_dir`` with ``train`` and ``val`` subdirs."""
    root = pathlib.Path(cfg.data_dir).expanduser().resolve()
    train_dir, val_dir = root / "train", root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(f"ImageNet dataset not found. Expected directories: {train_dir} and {val_dir}")
    print(f"Using ImageNet-1k from {root}", flush=True)
    train_tf, val_tf = build_vit_transforms(cfg)
    return make_loaders(
        datasets.ImageFolder(root=str(train_dir), transform=train_tf),
        datasets.ImageFolder(root=str(val_dir), transform=val_tf),
        cfg,
    )
