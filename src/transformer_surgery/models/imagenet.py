"""Shared helpers for timm DeiT-Tiny on ImageNet-1k (surgery/distill/PTQ loaders + checkpoint load)."""

from __future__ import annotations

import os
import pathlib
from typing import Any, Tuple

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from transformer_surgery.internal.util import get_device
from transformer_surgery.models.pet import IMAGENET_MEAN, IMAGENET_STD

IMAGENET_NUM_CLASSES = 1000


def create_deit_tiny_imagenet(pretrained: bool = True) -> nn.Module:
    """Construct timm DeiT-Tiny classifier for ImageNet-1k."""
    return timm.create_model(
        "deit_tiny_patch16_224",
        pretrained=pretrained,
        num_classes=IMAGENET_NUM_CLASSES,
    )


def build_imagenet_transforms(
    img_size: int = 224,
    *,
    randaugment: bool = True,
    ra_magnitude: int = 9,
    random_erasing_prob: float = 0.0,
) -> Tuple[transforms.Compose, transforms.Compose]:
    import warnings

    try:
        from torchvision.transforms import RandAugment
    except ImportError:
        RandAugment = None  # type: ignore

    train_steps: list = [
        transforms.RandomResizedCrop(img_size),
    ]
    if randaugment and RandAugment is not None:
        train_steps.append(RandAugment(num_ops=2, magnitude=ra_magnitude))
    elif randaugment and RandAugment is None:
        warnings.warn(
            "RandAugment requested but torchvision.transforms.RandAugment is unavailable; continuing without it.",
            stacklevel=2,
        )
    train_steps.extend(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    if random_erasing_prob and random_erasing_prob > 0:
        train_steps.append(transforms.RandomErasing(p=random_erasing_prob))
    train_steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    train_tf = transforms.Compose(train_steps)

    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def _imagenet_split_dir(data_dir: str, split: str) -> pathlib.Path:
    root = pathlib.Path(data_dir).expanduser().resolve()
    return root / split


def build_imagenet_loaders(cfg: Any) -> Tuple[DataLoader, DataLoader]:
    """
    Train/val ImageNet-1k loaders rooted at ``cfg.data_dir`` with ``train`` and ``val`` subdirs.
    """
    device = get_device()
    data_dir = str(cfg.data_dir)
    batch_size = int(cfg.batch_size)
    workers = int(cfg.workers)
    randaugment = bool(cfg.randaugment)
    ra_magnitude = int(cfg.ra_magnitude)
    random_erasing_prob = float(cfg.random_erasing_prob)

    train_dir = _imagenet_split_dir(data_dir, "train")
    val_dir = _imagenet_split_dir(data_dir, "val")
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            "ImageNet dataset not found. Expected directories: "
            f"{train_dir} and {val_dir}"
        )
    print(f"Using ImageNet-1k from {os.path.abspath(data_dir)}", flush=True)

    train_tf, val_tf = build_imagenet_transforms(
        224,
        randaugment=randaugment,
        ra_magnitude=ra_magnitude,
        random_erasing_prob=random_erasing_prob,
    )
    train_set = datasets.ImageFolder(root=str(train_dir), transform=train_tf)
    val_set = datasets.ImageFolder(root=str(val_dir), transform=val_tf)
    pin = device.type == "cuda"
    persistent = workers > 0
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    return train_loader, val_loader


def load_timm_deit_imagenet_checkpoint(path: str) -> nn.Module:
    """Load DeiT-Tiny ImageNet checkpoint from local file."""
    device = get_device()
    model = create_deit_tiny_imagenet(pretrained=False).to(device)
    payload = torch.load(os.path.abspath(path), map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model
