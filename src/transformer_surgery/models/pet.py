"""
timm DeiT-Tiny reference helpers (construction, checkpoint load, ViT transforms) and the Oxford-IIIT Pet task.

Training recipe (transforms, LR schedule, grad clip, defaults) follows the same spirit as
https://github.com/salimkhazem/adaptertune — `src/datasets/torchvision_datasets.py` + `src/train/sched.py`.
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

from transformer_surgery.internal.util import accuracy_and_loss, get_device, make_loaders, warmup_cosine_scheduler

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PET_NUM_CLASSES = 37


def create_timm_deit_tiny(num_classes: int, pretrained: bool = False) -> nn.Module:
    """timm ``deit_tiny_patch16_224`` with a ``num_classes`` head; ``pretrained`` loads the ImageNet backbone."""
    return timm.create_model("deit_tiny_patch16_224", pretrained=pretrained, num_classes=num_classes)


def load_timm_deit_checkpoint(path: str, num_classes: int) -> nn.Module:
    """DeiT-Tiny with all weights from ``path`` (``model_state_dict``), in eval mode on the run device."""
    device = get_device()
    model = create_timm_deit_tiny(num_classes).to(device)
    payload = torch.load(os.path.abspath(path), map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval()


def build_vit_transforms(cfg: Any, img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Train: RandomResizedCrop, optional RandAugment, flip, optional RandomErasing, normalize.
    Eval: Resize 256 + CenterCrop (standard ViT eval).
    """
    train_steps: list = [transforms.RandomResizedCrop(img_size)]
    if cfg.randaugment:
        train_steps.append(transforms.RandAugment(num_ops=2, magnitude=cfg.ra_magnitude))
    train_steps += [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    if cfg.random_erasing_prob > 0:
        train_steps.append(transforms.RandomErasing(p=cfg.random_erasing_prob))
    train_steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(train_steps), eval_tf


def oxford_iiit_pet_is_present(data_root: str) -> bool:
    """True if torchvision's Oxford-IIIT Pet layout already exists under data_root (no network access needed)."""
    base = pathlib.Path(data_root).expanduser().resolve() / "oxford-iiit-pet"
    return (base / "images").is_dir() and (base / "annotations").is_dir()


def build_pet_loaders(cfg: Any) -> Tuple[DataLoader, DataLoader]:
    """Train/val Oxford-IIIT Pet loaders under ``cfg.data_dir``; downloads the dataset once when missing."""
    os.makedirs(cfg.data_dir, exist_ok=True)
    download = not oxford_iiit_pet_is_present(cfg.data_dir)
    where = f"{os.path.abspath(cfg.data_dir)}/oxford-iiit-pet"
    print(f"Oxford-IIIT Pet {'not found, downloading once into' if download else 'found at'} {where}", flush=True)
    train_tf, val_tf = build_vit_transforms(cfg)
    train_set = OxfordIIITPet(root=cfg.data_dir, split="trainval", transform=train_tf, download=download)
    val_set = OxfordIIITPet(root=cfg.data_dir, split="test", transform=val_tf, download=download)
    return make_loaders(train_set, val_set, cfg)


def _fmt_best_epoch(best_ep: Optional[int]) -> str:
    """Training epoch (1-based), ``resume`` when baseline was seeded from a checkpoint (0), else ``—``."""
    if best_ep is None:
        return "—"
    return "resume" if best_ep == 0 else str(best_ep)


def train_timm_deit_on_pet(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Any,
    *,
    resume_val_acc: Optional[float] = None,
) -> Tuple[float, float, Optional[int], int]:
    """Pet timm DeiT-Tiny classifier-head training; returns final val acc/loss, best epoch and gap-revert count."""
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("head")
    head_params = [p for p in model.parameters() if p.requires_grad]
    if not head_params:
        raise ValueError("Expected timm ViT parameters named 'head*' for classifier fine-tune.")
    opt = torch.optim.AdamW(head_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps_per_epoch = len(train_loader)
    if cfg.max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, cfg.max_train_batches)
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    scheduler = warmup_cosine_scheduler(
        opt,
        total_steps=total_steps,
        warmup_steps=min(cfg.warmup_epochs * steps_per_epoch, max(total_steps - 1, 0)),
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
        print(f"  {pf}seed best from resume: val_acc={best_acc:.4f} (checkpoint baseline weights)", flush=True)
    for ep in range(cfg.epochs):
        model.train()
        for n_batches, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head_params, cfg.grad_clip)
            opt.step()
            scheduler.step()
            if cfg.max_train_batches is not None and n_batches >= cfg.max_train_batches:
                break
        acc, loss_v = accuracy_and_loss(model, val_loader, criterion)
        print(f"  {pf}epoch {ep + 1}/{cfg.epochs} | val acc={acc:.4f} loss={loss_v:.4f}", flush=True)
        if acc > best_acc:
            best_acc = acc
            best_ep = ep + 1
            best_state = copy.deepcopy(model.state_dict())

        if cfg.gap_th is not None and best_state is not None and acc < best_acc - cfg.gap_th:
            model.load_state_dict(best_state)
            acc, loss_v = accuracy_and_loss(model, val_loader, criterion)
            gap_revert_count += 1
            print(
                f"  {pf}gap revert #{gap_revert_count}: val acc below best by > {cfg.gap_th} "
                f"(best={best_acc:.4f} @ {_fmt_best_epoch(best_ep)}) — restored best weights "
                f"(val acc={acc:.4f} loss={loss_v:.4f})",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  {pf}kept best val acc={best_acc:.4f} (epoch {_fmt_best_epoch(best_ep)}/{cfg.epochs})", flush=True)
    acc_f, loss_f = accuracy_and_loss(model, val_loader, criterion)
    return acc_f, loss_f, best_ep, gap_revert_count
