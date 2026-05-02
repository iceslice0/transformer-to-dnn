"""
Shared helpers for timm DeiT-Tiny on Oxford-IIIT Pet (pretrain + surgery load).

Training recipe (transforms, LR schedule, grad clip, defaults) follows the same spirit as
https://github.com/salimkhazem/adaptertune — `src/datasets/torchvision_datasets.py` + `src/train/sched.py`.
Using Resize-only aug + per-epoch cosine (our earlier defaults) is far from common ViT fine-tuning on Pet.
"""

from __future__ import annotations

import copy
import os
import pathlib
from typing import Any, Optional, Tuple

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from transformer_surgery.util import accuracy_and_loss, get_device, warmup_cosine_scheduler

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PET_NUM_CLASSES = 37


def oxford_iiit_pet_is_present(data_root: str) -> bool:
    """
    True if torchvision's Oxford-IIIT Pet layout already exists under data_root
    (so we can pass download=False and never hit the network).
    """
    base = pathlib.Path(data_root).expanduser().resolve() / "oxford-iiit-pet"
    return (base / "images").is_dir() and (base / "annotations").is_dir()


def create_deit_tiny_pet(pretrained: bool = True) -> nn.Module:
    """
    Same construction as pretraining: must match before/after checkpoint load.
    `pretrained=True` uses ImageNet backbone init (then checkpoints overwrite weights).
    """
    return timm.create_model(
        "deit_tiny_patch16_224",
        pretrained=pretrained,
        num_classes=PET_NUM_CLASSES,
    )


def build_pet_transforms(
    img_size: int = 224,
    *,
    randaugment: bool = True,
    ra_magnitude: int = 9,
    random_erasing_prob: float = 0.0,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Train: RandomResizedCrop, optional RandAugment (+ RandomErasing), flip, normalize.
    Eval: Resize 256 + CenterCrop (standard ViT eval).

    RandAugment + layer decay are typical for pushing ViT fine-tune toward published numbers.
    """
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
            "RandAugment requested but torchvision.transforms.RandAugment is unavailable; "
            "continuing without it.",
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


def build_pet_loaders(
    cfg: Any,
) -> Tuple[DataLoader, DataLoader]:
    """
    Train/val Oxford-IIIT Pet loaders from a run config: ``data_dir``, ``batch_size``, ``workers``,
    ``randaugment``, ``ra_magnitude``, plus random erasing (field name depends on config type).
    """
    device = get_device()
    data_dir = cfg.data_dir
    batch_size = cfg.batch_size
    workers = cfg.workers
    randaugment = bool(cfg.randaugment)
    ra_magnitude = int(cfg.ra_magnitude)
    random_erasing_prob = float(cfg.random_erasing_prob)

    os.makedirs(data_dir, exist_ok=True)
    need_download = not oxford_iiit_pet_is_present(data_dir)
    if need_download:
        print(
            f"Oxford-IIIT Pet not found under {os.path.abspath(data_dir)}/oxford-iiit-pet; downloading once.",
            flush=True,
        )
    else:
        print(
            f"Using existing Oxford-IIIT Pet at {os.path.abspath(data_dir)}/oxford-iiit-pet (download=False).",
            flush=True,
        )
    train_tf, val_tf = build_pet_transforms(
        224,
        randaugment=randaugment,
        ra_magnitude=ra_magnitude,
        random_erasing_prob=random_erasing_prob,
    )
    train_set = OxfordIIITPet(root=data_dir, split="trainval", transform=train_tf, download=need_download)
    val_set = OxfordIIITPet(root=data_dir, split="test", transform=val_tf, download=need_download)
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


def load_timm_deit_pet_checkpoint(path: str) -> nn.Module:
    """
    Load Pet fine-tuned weights. Built with ``pretrained=False`` then ``load_state_dict`` — no
    ImageNet download; full weights come from the checkpoint.
    """
    device = get_device()
    path = os.path.abspath(path)
    model = create_deit_tiny_pet(pretrained=False).to(device)
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        sd = payload["model_state_dict"]
    elif isinstance(payload, dict):
        sd = payload
    else:
        sd = payload
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def _fmt_best_epoch_for_log(best_ep: Optional[int]) -> str:
    """Training epoch (1-based), ``resume`` when baseline was seeded from a checkpoint (0), else ``—``."""
    if best_ep is None:
        return "—"
    if best_ep == 0:
        return "resume"
    return str(best_ep)


def train_timm_deit_on_pet(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: "PretrainPetConfig",
    *,
    resume_val_acc: Optional[float] = None,
) -> Tuple[float, float, Optional[int], int]:
    """Pet timm DeiT-Tiny classifier-head training."""
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("head")
    head_params = [p for n, p in model.named_parameters() if n.startswith("head")]
    if not head_params:
        raise ValueError("Expected timm ViT parameters named 'head*' for classifier fine-tune.")
    opt = torch.optim.AdamW(head_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps_per_epoch = len(train_loader)
    if cfg.max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, cfg.max_train_batches)
    epochs = cfg.epochs
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = min(cfg.warmup_epochs * steps_per_epoch, max(total_steps - 1, 0))

    scheduler = warmup_cosine_scheduler(
        opt,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        eta_min=max(0.0, float(cfg.cosine_eta_min)),
    )
    device = get_device()
    use_cuda = device.type == "cuda"
    pf = "pet ref "
    best_acc = float("-inf")
    best_state: Optional[dict] = None
    best_ep: Optional[int] = None
    gap_revert_count = 0
    if resume_val_acc is not None:
        best_acc = float(resume_val_acc)
        best_state = copy.deepcopy(model.state_dict())
        best_ep = 0
        print(
            f"  {pf}seed best from resume: val_acc={best_acc:.4f} (checkpoint baseline weights)",
            flush=True,
        )
    for ep in range(epochs):
        model.train()
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            if cfg.grad_clip > 0:
                trainable = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            opt.step()
            scheduler.step()
            n_batches += 1
            if cfg.max_train_batches is not None and n_batches >= cfg.max_train_batches:
                break
        acc, loss_v = accuracy_and_loss(model, val_loader, criterion)
        print(f"  {pf}epoch {ep + 1}/{epochs} | val acc={acc:.4f} loss={loss_v:.4f}", flush=True)
        if acc > best_acc:
            best_acc = acc
            best_ep = ep + 1
            best_state = copy.deepcopy(model.state_dict())

        gap_th = cfg.gap_th
        if (
            gap_th is not None
            and best_state is not None
            and float(acc) < float(best_acc) - float(gap_th)
        ):
            model.load_state_dict(best_state)
            acc, loss_v = accuracy_and_loss(model, val_loader, criterion)
            gap_revert_count += 1
            print(
                f"  {pf}gap revert #{gap_revert_count}: val acc below best by > {float(gap_th)} "
                f"(best={best_acc:.4f} @ {_fmt_best_epoch_for_log(best_ep)}) — restored best weights "
                f"(val acc={acc:.4f} loss={loss_v:.4f})",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"  {pf}kept best val acc={best_acc:.4f} (epoch {_fmt_best_epoch_for_log(best_ep)}/{epochs})",
            flush=True,
        )
    acc_f, loss_f = accuracy_and_loss(model, val_loader, criterion)
    return acc_f, loss_f, best_ep, gap_revert_count
