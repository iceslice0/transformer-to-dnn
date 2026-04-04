"""
Shared helpers for timm DeiT-Tiny on Oxford-IIIT Pet (pretrain + surgery load).

Training recipe (transforms, LR schedule, grad clip, defaults) follows the same spirit as
https://github.com/salimkhazem/adaptertune — `src/datasets/torchvision_datasets.py` + `src/train/sched.py`.
Using Resize-only aug + per-epoch cosine (our earlier defaults) is far from common ViT fine-tuning on Pet.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import random
import time
from dataclasses import dataclass, fields, replace
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import timm
import torch
import torch.nn as nn
from timm.optim import create_optimizer_v2
from torchmetrics.classification import MulticlassAccuracy
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from surgery_utils import CALIBRATION_LEGEND_TEXT, SurgeryMeta, jeffreys_divergence_dense

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PET_NUM_CLASSES = 37

TConfig = TypeVar("TConfig")


def load_dataclass_from_json(
    cls: Type[TConfig],
    json_path: str,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    config_json_path_field: str = "config_json_path",
) -> TConfig:
    """
    Instantiate ``cls()`` (defaults), merge JSON keys, then ``overrides`` (e.g. CLI).
    Only keys that match dataclass fields are applied; ``config_json_path_field`` is set to the
    loaded JSON path when the file exists.
    """
    cfg = cls()
    allowed = {f.name for f in fields(cls)}
    skip = {config_json_path_field}
    loaded: Optional[str] = None
    ap = os.path.abspath(os.path.expanduser(json_path))
    if os.path.isfile(ap):
        with open(ap, encoding="utf-8") as f:
            raw = json.load(f)
        kwargs = {k: v for k, v in raw.items() if k in allowed and k not in skip}
        if kwargs:
            cfg = replace(cfg, **kwargs)
        loaded = ap
    if overrides:
        kwargs = {k: v for k, v in overrides.items() if k in allowed and k not in skip}
        if kwargs:
            cfg = replace(cfg, **kwargs)
    return replace(cfg, **{config_json_path_field: loaded})


def cli_overrides_from_namespace(
    args: Any,
    cls: Type[Any],
    *,
    exclude: frozenset[str] = frozenset({"config", "config_json_path"}),
) -> Dict[str, Any]:
    """Collect ``argparse`` overrides: only attributes present on ``args`` (e.g. not ``SUPPRESS``)."""
    patchable = {f.name for f in fields(cls)} - exclude
    out: Dict[str, Any] = {}
    for name in patchable:
        if hasattr(args, name):
            out[name] = getattr(args, name)
    return out


def _snake_to_kebab(name: str) -> str:
    return name.replace("_", "-")


def _strip_optional(tp: Any) -> Any:
    origin = get_origin(tp)
    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def add_dataclass_cli_args(
    parser: argparse.ArgumentParser,
    cls: Type[Any],
    *,
    exclude: frozenset[str] = frozenset({"config_json_path"}),
    field_help: Optional[Dict[str, str]] = None,
) -> None:
    """
    One ``--kebab-case`` flag per dataclass field. Omitted flags do not override JSON (``SUPPRESS``).

    Booleans: default ``False`` → ``--field`` with ``store_true``; default ``True`` →
    ``BooleanOptionalAction`` (``--field`` / ``--no-field``).
    """
    field_help = field_help or {}
    hints = get_type_hints(cls)
    for f in fields(cls):
        if f.name in exclude:
            continue
        name = f.name
        flag = f"--{_snake_to_kebab(name)}"
        h = field_help.get(name)
        tp = _strip_optional(hints.get(name, type(f.default)))
        if tp is bool:
            if f.default is True:
                parser.add_argument(
                    flag,
                    dest=name,
                    action=argparse.BooleanOptionalAction,
                    default=argparse.SUPPRESS,
                    help=h,
                )
            else:
                parser.add_argument(
                    flag,
                    dest=name,
                    action="store_true",
                    default=argparse.SUPPRESS,
                    help=h,
                )
        elif tp is int:
            parser.add_argument(flag, dest=name, type=int, default=argparse.SUPPRESS, help=h)
        elif tp is float:
            parser.add_argument(flag, dest=name, type=float, default=argparse.SUPPRESS, help=h)
        elif tp is str:
            parser.add_argument(flag, dest=name, type=str, default=argparse.SUPPRESS, help=h)
        else:
            raise TypeError(f"Unsupported CLI field type {cls.__name__}.{name}: {tp}")


def build_config_cli_parser(
    description: str,
    config_cls: Type[Any],
    *,
    config_default: str,
    config_help: str = "JSON hyperparameters (merged with dataclass defaults).",
    config_dest: str = "config",
    field_help: Optional[Dict[str, str]] = None,
) -> argparse.ArgumentParser:
    """``--config`` plus one option per dataclass field (except ``config_json_path``)."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--config",
        dest=config_dest,
        type=str,
        default=config_default,
        help=config_help,
    )
    add_dataclass_cli_args(p, config_cls, field_help=field_help)
    return p


def parse_cli_config(
    config_cls: Type[TConfig],
    *,
    description: str,
    config_default: str,
    config_help: str = "JSON hyperparameters (merged with dataclass defaults).",
    field_help: Optional[Dict[str, str]] = None,
    argv: Optional[Sequence[str]] = None,
) -> TConfig:
    """
    Build the standard ``--config`` + dataclass-flag parser, ``parse_args``, and return
    ``config_cls.load(config, cli_overrides)``. Used by all pipeline CLIs for one pattern.
    """
    parser = build_config_cli_parser(
        description,
        config_cls,
        config_default=config_default,
        config_help=config_help,
        field_help=field_help,
    )
    args = parser.parse_args(argv)
    return config_cls.load(args.config, cli_overrides_from_namespace(args, config_cls))


def resolve_path_under_script(path: str, script_file: str) -> str:
    """If ``path`` is relative, resolve under ``dirname(script_file)``; absolute paths unchanged."""
    if os.path.isabs(path):
        return path
    base = os.path.dirname(os.path.abspath(script_file))
    return os.path.join(base, path)


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    # Full determinism may require CUBLAS_WORKSPACE_CONFIG etc.


def _resolve_device_string(name: str) -> torch.device:
    n = (name or "auto").strip().lower()
    if n == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if n == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device cuda but torch.cuda.is_available() is False. "
                "Install a CUDA build of PyTorch or use device cpu."
            )
        return torch.device("cuda")
    if n == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    return torch.device(n)


# Process-wide default for loaders, checkpoints, and training loops. Call ``set_default_device`` once at startup.
TORCH_DEVICE: torch.device = torch.device("cpu")


def set_default_device(name: str = "auto") -> torch.device:
    """
    Set :data:`TORCH_DEVICE` from ``\"auto\"`` / ``\"cuda\"`` / ``\"cpu\"`` / …; enables cudnn benchmark on CUDA.

    Prefer :func:`apply_device_from_config` at CLI entry points so the device string comes from merged
    JSON/CLI config in one place.
    """
    global TORCH_DEVICE
    TORCH_DEVICE = _resolve_device_string(name)
    if TORCH_DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return TORCH_DEVICE


def get_device() -> torch.device:
    """Current default device (set with :func:`set_default_device`)."""
    return TORCH_DEVICE


def apply_device_from_config(
    cfg: Union[
        Mapping[str, Any],
        "PretrainPetConfig",
        "SurgeryRunConfig",
        "JeffreysDistillConfig",
    ],
) -> torch.device:
    """
    Resolve ``device`` from merged run configuration and set the process default exactly once.

    Use this after :func:`load_dataclass_from_json` / ``*.load(...)`` — reads ``device`` from a
    mapping or from any run config dataclass. Internally calls :func:`set_default_device`.
    """
    if isinstance(cfg, Mapping):
        name = str(cfg.get("device", "auto"))
    else:
        name = str(getattr(cfg, "device", "auto"))
    return set_default_device(name)


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        try:
            return f"cuda ({torch.cuda.get_device_name(device)})"
        except Exception:
            return "cuda"
    return str(device)


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


def pet_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Alias: strong aug on by default."""
    return build_pet_transforms(img_size)


@torch.no_grad()
def accuracy_and_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    num_classes: Optional[int] = None,
) -> Tuple[float, float]:
    """Validation accuracy via ``torchmetrics.MulticlassAccuracy``; mean CE loss weighted by batch size."""
    model.eval()
    device = get_device()
    use_cuda = device.type == "cuda"
    acc_metric: Optional[MulticlassAccuracy] = None
    loss_sum, n_samples = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=use_cuda)
        y = y.to(device, non_blocking=use_cuda)
        logits = model(x)
        loss = criterion(logits, y)
        bs = y.size(0)
        loss_sum += loss.item() * bs
        n_samples += bs
        if acc_metric is None:
            nc = int(num_classes) if num_classes is not None else int(logits.shape[-1])
            acc_metric = MulticlassAccuracy(num_classes=nc, average="micro").to(device)
        acc_metric.update(logits, y)
    if acc_metric is None or n_samples == 0:
        return 0.0, 0.0
    return float(acc_metric.compute().item()), loss_sum / n_samples


def _pet_loader_random_erasing_prob(cfg: Any) -> float:
    """``random_erasing_prob`` (surgery / pretrain) or ``random_erasing`` (Jeffreys distill config)."""
    if hasattr(cfg, "random_erasing_prob"):
        return float(cfg.random_erasing_prob)
    return float(getattr(cfg, "random_erasing", 0.0))


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
    randaugment = bool(getattr(cfg, "randaugment", True))
    ra_magnitude = int(getattr(cfg, "ra_magnitude", 9))
    random_erasing_prob = _pet_loader_random_erasing_prob(cfg)

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
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
    )
    return train_loader, val_loader


def load_timm_deit_pet_checkpoint(path: str) -> nn.Module:
    """
    Load Pet fine-tuned weights. Built with ``pretrained=False`` then ``load_state_dict`` — no
    ImageNet download; full weights come from the checkpoint.
    """
    device = get_device()
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pet reference checkpoint not found: {path}")
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


def _warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    eta_min: float,
) -> LRScheduler:
    """
    Linear warmup (``LinearLR``) then ``CosineAnnealingLR``, chained with ``SequentialLR``.
    Step once per optimizer step. Matches common ViT fine-tuning schedules without hand-written cos.
    """
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    warmup_steps = min(warmup_steps, total_steps)

    if warmup_steps <= 0:
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)

    if warmup_steps >= total_steps:
        return LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=total_steps,
        )

    cosine_steps = max(1, total_steps - warmup_steps)
    warmup = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=eta_min)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])


def _fmt_best_epoch_for_log(best_ep: Optional[int]) -> str:
    """Training epoch (1-based), ``resume`` when baseline was seeded from a checkpoint (0), else ``—``."""
    if best_ep is None:
        return "—"
    if best_ep == 0:
        return "resume"
    return str(best_ep)


def finetune_model_adaptertune_style(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    max_train_batches: Optional[int] = None,
    weight_decay: float = 0.05,
    backbone_lr_mult: float = 1.0,
    warmup_epochs: int = 5,
    grad_clip: float = 1.0,
    log_prefix: str = "",
    layer_decay: Optional[float] = None,
    label_smoothing: float = 0.0,
    head_only: bool = False,
    keep_best_val: bool = False,
    cosine_eta_min: float = 0.0,
    val_gap_th: Optional[float] = None,
    resume_val_acc: Optional[float] = None,
) -> Tuple[float, float, Optional[int], int]:
    """
    AdamW + ``LinearLR`` warmup (optional) + ``CosineAnnealingLR`` (``SequentialLR``), stepped once
    per batch. ``cosine_eta_min`` is passed as ``eta_min`` on cosine.

    If ``head_only`` is True, freezes all parameters except those whose names start with ``head``
    (timm ViT classifier) and trains only that layer — typical linear probe / head fine-tune.

    If ``layer_decay`` is set (e.g. 0.75), uses timm ``create_optimizer_v2`` BEiT-style grouping
    (ignored when ``head_only`` or combined with ``backbone_lr_mult != 1``).

    If ``keep_best_val`` is True, keeps weights from the epoch with highest validation accuracy
    (returns that accuracy/loss and the 1-based best epoch index; else third return is None).

    If ``resume_val_acc`` is set and ``keep_best_val`` is True, ``best_acc`` / ``best_state`` are
    seeded before epoch 1 from that value and the model's current weights (e.g. checkpoint
    ``val_acc`` when continuing training).

    If ``val_gap_th`` is set and ``keep_best_val`` is True, after each epoch when validation
    accuracy is more than ``val_gap_th`` below the best-so-far, model weights are restored to the
    best checkpoint only (optimizer state unchanged — simple hook for random search). The fourth
    return value counts how many such reverts occurred.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if head_only:
        for n, p in model.named_parameters():
            p.requires_grad = n.startswith("head")
        head_params = [p for n, p in model.named_parameters() if n.startswith("head")]
        if not head_params:
            raise ValueError(
                "head_only=True but no parameters named 'head*' — expected timm ViT classifier weights."
            )
        opt = torch.optim.AdamW(head_params, lr=lr, weight_decay=weight_decay)
    elif layer_decay is not None and backbone_lr_mult == 1.0:
        opt = create_optimizer_v2(
            model,
            "adamw",
            lr=lr,
            weight_decay=weight_decay,
            layer_decay=layer_decay,
        )
    elif backbone_lr_mult != 1.0:
        head_params, bb_params = [], []
        for n, p in model.named_parameters():
            (head_params if n.startswith("head") else bb_params).append(p)
        opt = torch.optim.AdamW(
            [
                {"params": bb_params, "lr": lr * backbone_lr_mult},
                {"params": head_params, "lr": lr},
            ],
            weight_decay=weight_decay,
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = len(train_loader)
    if max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, max_train_batches)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = min(warmup_epochs * steps_per_epoch, max(total_steps - 1, 0))

    scheduler = _warmup_cosine_scheduler(
        opt,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        eta_min=max(0.0, float(cosine_eta_min)),
    )
    device = get_device()
    use_cuda = device.type == "cuda"
    pf = f"{log_prefix} " if log_prefix else ""
    best_acc = float("-inf")
    best_state: Optional[dict] = None
    best_ep: Optional[int] = None
    gap_revert_count = 0
    if keep_best_val and resume_val_acc is not None:
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
            if grad_clip > 0:
                trainable = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            opt.step()
            scheduler.step()
            n_batches += 1
            if max_train_batches is not None and n_batches >= max_train_batches:
                break
        acc, loss_v = accuracy_and_loss(model, val_loader, criterion)
        print(f"  {pf}epoch {ep + 1}/{epochs} | val acc={acc:.4f} loss={loss_v:.4f}", flush=True)
        if keep_best_val and acc > best_acc:
            best_acc = acc
            best_ep = ep + 1
            best_state = copy.deepcopy(model.state_dict())

        if (
            val_gap_th is not None
            and keep_best_val
            and best_state is not None
            and float(acc) < float(best_acc) - float(val_gap_th)
        ):
            model.load_state_dict(best_state)
            acc, loss_v = accuracy_and_loss(model, val_loader, criterion)
            gap_revert_count += 1
            print(
                f"  {pf}gap revert #{gap_revert_count}: val acc below best by > {float(val_gap_th)} "
                f"(best={best_acc:.4f} @ {_fmt_best_epoch_for_log(best_ep)}) — restored best weights "
                f"(val acc={acc:.4f} loss={loss_v:.4f})",
                flush=True,
            )
    if keep_best_val and best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"  {pf}kept best val acc={best_acc:.4f} (epoch {_fmt_best_epoch_for_log(best_ep)}/{epochs})",
            flush=True,
        )
    acc_f, loss_f = accuracy_and_loss(model, val_loader, criterion)
    best_i = best_ep if keep_best_val else None
    return acc_f, loss_f, best_i, gap_revert_count


def train_timm_deit_on_pet(
    ref: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: "PretrainPetConfig",
    *,
    resume_val_acc: Optional[float] = None,
) -> Tuple[float, float, Optional[int], int]:
    """Timm DeiT-Tiny: train only the Pet classification head (backbone frozen); keeps best val checkpoint.

    Hyperparameters from ``cfg`` (:class:`PretrainPetConfig`). Uses ``LinearLR`` + ``CosineAnnealingLR``
    (``SequentialLR``) per step; ``eta_min=cfg.cosine_eta_min``. Optional ``cfg.gap_th`` enables
    per-epoch gap revert inside the training loop (see :func:`finetune_model_adaptertune_style`).

    Pass ``resume_val_acc`` (e.g. ``val_acc`` from the loaded checkpoint) to seed best-so-far before
    epoch 1 so gap revert compares against that baseline.
    """
    c = cfg
    return finetune_model_adaptertune_style(
        ref,
        train_loader,
        val_loader,
        c.epochs,
        c.lr,
        max_train_batches=c.max_train_batches,
        weight_decay=c.weight_decay,
        backbone_lr_mult=1.0,
        warmup_epochs=c.warmup_epochs,
        grad_clip=c.grad_clip,
        log_prefix="pet ref",
        layer_decay=None,
        label_smoothing=c.label_smoothing,
        head_only=True,
        keep_best_val=True,
        cosine_eta_min=c.cosine_eta_min,
        val_gap_th=c.gap_th,
        resume_val_acc=resume_val_acc,
    )


@torch.no_grad()
def eval_distillation_metrics(
    teacher: nn.Module,
    student: nn.Module,
    val_loader: DataLoader,
    temperature: float,
    *,
    progress_batches: int = 0,
    progress_prefix: str = "",
) -> Tuple[float, float, float]:
    """Student val accuracy, mean CE vs labels, mean Jeffreys J(teacher, student) on val."""
    device = get_device()
    teacher.eval()
    student.eval()
    use_cuda = device.type == "cuda"
    ce_sum, j_sum = 0.0, 0.0
    n = 0
    correct = 0
    n_val = len(val_loader)
    t0 = time.perf_counter()
    for bi, (x, y) in enumerate(val_loader):
        x = x.to(device, non_blocking=use_cuda)
        y = y.to(device, non_blocking=use_cuda)
        t_log = teacher(x)
        s_log = student(x)
        j_sum += jeffreys_divergence_dense(t_log, s_log, temperature=temperature).sum().item()
        ce_sum += torch.nn.functional.cross_entropy(s_log, y, reduction="sum").item()
        correct += (s_log.argmax(dim=-1) == y).sum().item()
        n += y.size(0)
        if progress_batches > 0 and n_val > 0:
            if bi == 0 or (bi + 1) % progress_batches == 0 or (bi + 1) == n_val:
                elapsed = time.perf_counter() - t0
                rate = (bi + 1) / elapsed if elapsed > 0 else 0.0
                eta = (n_val - bi - 1) / rate if rate > 0 else 0.0
                print(
                    f"  {progress_prefix}val {bi + 1}/{n_val} batches "
                    f"~{rate:.2f} batch/s eta~{eta:.0f}s",
                    flush=True,
                )
    acc = correct / max(n, 1)
    return acc, ce_sum / max(n, 1), j_sum / max(n, 1)


def distill_surgery_from_teacher_jeffreys(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: "JeffreysDistillConfig",
    *,
    log_prefix: str = "distill",
) -> Tuple[float, float, float, Optional[int]]:
    """
    Train student to match frozen timm teacher class distributions using dense Jeffreys J(p,q).
    Hyperparameters come from ``cfg`` (:class:`JeffreysDistillConfig`). Warmup length follows the
    same rule as the Jeffreys CLI: ``min(5, max(epochs, 1))`` when ``cfg.warmup_epochs`` is None.

    Returns (val_acc, val_ce_mean, val_jeffreys_mean, best_epoch_or_None).

    ``cfg.train_progress_interval``: print every N training batches. ``0`` = only per-epoch val lines.

    ``cfg.val_progress_batches``: val eval progress every N batches; ``0`` = silent until metrics done.
    """
    c = cfg
    epochs = c.epochs
    lr = c.lr
    temperature = c.temperature
    max_train_batches = int(c.max_train_batches) if c.max_train_batches is not None else None
    weight_decay = c.weight_decay
    warmup_epochs = min(5, max(c.epochs, 1)) if c.warmup_epochs is None else int(c.warmup_epochs)
    grad_clip = c.grad_clip
    cosine_eta_min = c.cosine_eta_min
    keep_best_val = c.keep_best
    train_progress_interval = c.train_progress_interval
    val_progress_batches = c.val_progress_batches

    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    device = get_device()
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = len(train_loader)
    if max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, max_train_batches)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = min(warmup_epochs * steps_per_epoch, max(total_steps - 1, 0))
    scheduler = _warmup_cosine_scheduler(
        opt,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        eta_min=max(0.0, float(cosine_eta_min)),
    )
    use_cuda = device.type == "cuda"
    pf = f"{log_prefix} " if log_prefix else ""
    best_acc = float("-inf")
    best_state: Optional[dict] = None
    best_ep: Optional[int] = None

    print(
        f"  {pf}schedule: {epochs} epoch(s) × {steps_per_epoch} train batches "
        f"→ {total_steps} optimizer steps | train log every {train_progress_interval} batch(es)"
        + (f" | val log every {val_progress_batches} batch(es)" if val_progress_batches > 0 else " | val silent"),
        flush=True,
    )

    global_step = 0
    for ep in range(epochs):
        student.train()
        n_batches = 0
        ep_t0 = time.perf_counter()
        running_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device, non_blocking=use_cuda)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                t_log = teacher(x)
            s_log = student(x)
            loss = jeffreys_divergence_dense(t_log, s_log, temperature=temperature).mean()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            opt.step()
            scheduler.step()
            n_batches += 1
            global_step += 1
            li = float(loss.item())
            running_loss += li
            if train_progress_interval > 0:
                do_log = (
                    n_batches == 1
                    or n_batches % train_progress_interval == 0
                    or n_batches == steps_per_epoch
                )
                if max_train_batches is not None and n_batches >= max_train_batches:
                    do_log = True
                if do_log:
                    elapsed = time.perf_counter() - ep_t0
                    avg = running_loss / n_batches
                    rate = n_batches / elapsed if elapsed > 0 else 0.0
                    left = steps_per_epoch - n_batches
                    eta_s = left / rate if rate > 0 else 0.0
                    lr_c = opt.param_groups[0]["lr"]
                    print(
                        f"  {pf}epoch {ep + 1}/{epochs} train {n_batches}/{steps_per_epoch} "
                        f"step {global_step}/{total_steps} loss={li:.6f} loss_avg={avg:.6f} "
                        f"lr={lr_c:.2e} {rate:.2f} batch/s epoch_eta~{eta_s / 60.0:.1f}m",
                        flush=True,
                    )
            if max_train_batches is not None and n_batches >= max_train_batches:
                break
        acc, ce_v, j_v = eval_distillation_metrics(
            teacher,
            student,
            val_loader,
            temperature,
            progress_batches=val_progress_batches,
            progress_prefix=pf,
        )
        print(
            f"  {pf}epoch {ep + 1}/{epochs} | val acc={acc:.4f} ce={ce_v:.4f} jeffreys={j_v:.4f}",
            flush=True,
        )
        if keep_best_val and acc > best_acc:
            best_acc = acc
            best_ep = ep + 1
            best_state = copy.deepcopy(student.state_dict())
    if keep_best_val and best_state is not None:
        student.load_state_dict(best_state)
        print(
            f"  {pf}kept best val acc={best_acc:.4f} (epoch {best_ep}/{epochs})",
            flush=True,
        )
    acc_f, ce_f, j_f = eval_distillation_metrics(
        teacher,
        student,
        val_loader,
        temperature,
        progress_batches=val_progress_batches,
        progress_prefix=pf,
    )
    best_i = best_ep if keep_best_val else None
    return acc_f, ce_f, j_f, best_i


# ---------------------------------------------------------------------------
# Checkpoint I/O (shared by surgery run + distillation)
# ---------------------------------------------------------------------------


def save_deit_checkpoint(path: str, model: nn.Module, extra: Optional[Dict[str, Any]] = None) -> None:
    """Save ``{model_state_dict, extra}`` in the same format as legacy ``save_surgery_checkpoint``."""
    torch.save({"model_state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_surgery_student_checkpoint(
    path: str,
    top_k: Optional[int],
    eps_ln: Optional[float],
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a surgery DeiT student from ``surgery_pre_ft.pt`` (or compatible) for distillation / eval.
    Architecture flags and ``top_k`` / ``eps_ln`` default from checkpoint ``extra`` when overrides are None.
    """
    from deit_tiny_surgery_model import DeiTTinySurgeryModel, freeze_eps_parameters

    device = get_device()
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    ex = payload.get("extra") or {}
    tk = int(top_k) if top_k is not None else int(ex.get("top_k", 32))
    ep = float(eps_ln) if eps_ln is not None else float(ex.get("eps_ln", 1e-5))
    dlr = bool(ex.get("disable_layernorm_replacement", False))
    das = bool(ex.get("disable_attention_surgery", False))
    dsr = bool(ex.get("disable_softmax_replacement", False))
    ams = bool(ex.get("allow_matmul_scores", False))
    aev = bool(ex.get("allow_elementwise_attn_value_mul", False))
    model = DeiTTinySurgeryModel(
        num_classes=PET_NUM_CLASSES,
        top_k=tk,
        eps_ln=ep,
        use_surgery_layernorm=not dlr,
        use_attention_surgery=not das,
        use_surgery_softmax=not dsr,
        allow_matmul_scores=ams,
        allow_elementwise_attn_value_mul=aev,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    freeze_eps_parameters(model)
    return model, ex


def merge_post_distill_into_surgery_meta(
    meta_path: str,
    val_acc: float,
    val_ce: float,
    val_jeffreys: float,
) -> None:
    """Merge post-distill metrics into ``surgery_meta.json`` (create stub if missing)."""
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = {
            "patient": "DeiT-Tiny",
            "dataset": "Oxford-IIIT Pet",
            "calibration": {},
            "pwl_knees": {
                "usage": {
                    "note": (
                        "No surgery run meta on disk; run run_deit_tiny_surgery.py for full "
                        "pwl_knees (exp_knots, log_x_knots, usage)."
                    ),
                },
            },
            "meta_note": "Stub created before distill (no prior surgery_meta at this path).",
        }
    cal = raw.get("calibration") or {}
    cal["student_post_distill_val_acc"] = float(val_acc)
    cal["student_post_distill_mean_ce"] = float(val_ce)
    cal["student_post_distill_mean_jeffreys"] = float(val_jeffreys)
    cal["val_acc_post_ft"] = float(val_acc)
    cal["val_ce_post_ft"] = float(val_ce)
    cal["val_jeffreys_post_ft"] = float(val_jeffreys)
    cal.pop("val_loss_post_ft", None)
    raw["calibration"] = cal
    raw["calibration_legend"] = CALIBRATION_LEGEND_TEXT
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)


@dataclass
class PretrainPetConfig:
    """Pet classifier head training on timm DeiT-Tiny; JSON + CLI via :meth:`load`."""

    data_dir: str = "./data"
    output: str = "./pet_timm_deit_tiny.pt"
    epochs: int = 50
    batch_size: int = 128
    workers: int = 2
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    label_smoothing: float = 0.05
    seed: int = 42
    max_train_batches: Optional[int] = None
    device: str = "auto"
    cosine_eta_min: float = 1e-6
    gap_th: Optional[float] = None
    randaugment: bool = True
    ra_magnitude: int = 9
    random_erasing_prob: float = 0.0
    config_json_path: Optional[str] = None

    @classmethod
    def load(cls, json_path: str, overrides: Optional[Dict[str, Any]] = None) -> "PretrainPetConfig":
        """Defaults → JSON (if present) → ``overrides``; sets ``config_json_path``."""
        return load_dataclass_from_json(cls, json_path, overrides)


FIELD_HELP_PRETRAIN: Dict[str, str] = {
    "gap_th": (
        "During training: after each epoch, if val acc is more than this far below the best-so-far "
        "val acc, restore model weights to that best checkpoint only (optimizer state unchanged)."
    ),
}


def pretrain_train_config_record(
    cfg: PretrainPetConfig,
    *,
    output_abs: str,
    best_val_epoch: Optional[int] = None,
    best_acc_reference: Optional[float] = None,
    gap_revert_count: int = 0,
) -> Dict[str, Any]:
    """JSON-serializable ``train_config`` block for the Pet head checkpoint."""
    rec: Dict[str, Any] = {
        "data_dir": cfg.data_dir,
        "output": output_abs,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "warmup_epochs": cfg.warmup_epochs,
        "grad_clip": cfg.grad_clip,
        "label_smoothing": cfg.label_smoothing,
        "seed": cfg.seed,
        "workers": cfg.workers,
        "device": cfg.device,
        "cosine_eta_min": cfg.cosine_eta_min,
    }
    if cfg.config_json_path:
        rec["config_json"] = cfg.config_json_path
    if cfg.max_train_batches is not None:
        rec["max_train_batches"] = cfg.max_train_batches
    if best_val_epoch is not None:
        rec["best_val_epoch"] = best_val_epoch
    if cfg.gap_th is not None:
        rec["gap_th"] = float(cfg.gap_th)
    if best_acc_reference is not None:
        rec["best_acc_reference"] = best_acc_reference
    if gap_revert_count > 0:
        rec["gap_revert_count"] = gap_revert_count
    return rec


@dataclass
class SurgeryRunConfig:
    """Surgery transform + calibration run (``run_deit_tiny_surgery.py``); JSON + CLI via :meth:`load`."""

    data_dir: str = "./data"
    batch_size: int = 128
    workers: int = 2
    top_k: int = 32
    eps: float = 1e-5
    pet_ref_checkpoint: str = "./pet_timm_deit_tiny.pt"
    device: str = "auto"
    disable_layernorm_replacement: bool = False
    disable_attention_surgery: bool = False
    disable_softmax_replacement: bool = False
    allow_matmul_scores: bool = False
    allow_elementwise_attn_value_mul: bool = False
    randaugment: bool = True
    ra_magnitude: int = 9
    random_erasing_prob: float = 0.0
    meta_json: str = "surgery_meta.json"
    pre_ft_checkpoint: str = "surgery_pre_ft.pt"
    config_json_path: Optional[str] = None

    @classmethod
    def load(cls, json_path: str, overrides: Optional[Dict[str, Any]] = None) -> "SurgeryRunConfig":
        """Defaults → JSON (if present) → ``overrides``; sets ``config_json_path``."""
        return load_dataclass_from_json(cls, json_path, overrides)


FIELD_HELP_SURGERY_RUN: Dict[str, str] = {
    "disable_layernorm_replacement": (
        "Use nn.LayerNorm instead of RewrittenLayerNormAbsSign (isolates LN PWL path)."
    ),
    "disable_attention_surgery": (
        "Use timm-like attention (scaled QK^T, full softmax, dense @ V); no PairwiseDotBySquare."
    ),
    "disable_softmax_replacement": (
        "When attention surgery is on: full softmax @ V instead of Gibbs top-k + sparse mix."
    ),
    "allow_matmul_scores": (
        "When attention surgery is on: fused (q/sqrt(d))@k^T for scores (fast; not plan.md explicit square)."
    ),
    "allow_elementwise_attn_value_mul": (
        "With Gibbs sparse mix: p*v instead of square identity (fast; not strict demo)."
    ),
}


def surgery_meta_for_pre_ft(
    cfg: SurgeryRunConfig,
    *,
    calibration: Dict[str, float],
    pet_ref_checkpoint_abs: str,
    pwl_knees: Dict[str, Any],
    module_mapping: Dict[str, str],
) -> SurgeryMeta:
    """Build :class:`surgery_utils.SurgeryMeta` for the pre–Jeffreys surgery run."""
    return SurgeryMeta(
        eps=float(cfg.eps),
        top_k=int(cfg.top_k),
        pwl_knees=pwl_knees,
        calibration=dict(calibration),
        module_mapping=module_mapping,
        pet_ref_checkpoint=pet_ref_checkpoint_abs,
        allow_matmul_scores=cfg.allow_matmul_scores,
        allow_elementwise_attn_value_mul=cfg.allow_elementwise_attn_value_mul,
    )


def pre_ft_checkpoint_extra(
    cfg: SurgeryRunConfig,
    *,
    mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """``extra`` dict for :func:`save_deit_checkpoint` after surgery transform."""
    return {
        "meta_ref": os.path.basename(cfg.meta_json),
        "mapping": mapping,
        "top_k": int(cfg.top_k),
        "eps_ln": float(cfg.eps),
        "config_json": cfg.config_json_path,
        "disable_layernorm_replacement": cfg.disable_layernorm_replacement,
        "disable_attention_surgery": cfg.disable_attention_surgery,
        "disable_softmax_replacement": cfg.disable_softmax_replacement,
        "allow_matmul_scores": cfg.allow_matmul_scores,
        "allow_elementwise_attn_value_mul": cfg.allow_elementwise_attn_value_mul,
    }


@dataclass
class JeffreysDistillConfig:
    """Single container for Jeffreys distillation (JSON file + optional CLI overrides)."""

    data_dir: str = "./data"
    epochs: int = 2
    batch_size: int = 128
    workers: int = 2
    lr: float = 5e-4
    weight_decay: float = 0.05
    warmup_epochs: Optional[int] = None
    grad_clip: float = 1.0
    cosine_eta_min: float = 0.0
    temperature: float = 1.0
    max_train_batches: Optional[int] = None
    keep_best: bool = True
    pet_ref_checkpoint: str = "./pet_timm_deit_tiny.pt"
    pre_checkpoint: str = "./surgery_pre_ft.pt"
    output: str = "./surgery_post_ft.pt"
    meta_json: Optional[str] = "./surgery_meta.json"
    randaugment: bool = True
    ra_magnitude: int = 9
    random_erasing: float = 0.0
    device: str = "auto"
    train_progress_interval: int = 10
    val_progress_batches: int = 20
    top_k: Optional[int] = None
    eps: Optional[float] = None
    config_json_path: Optional[str] = None
    quiet: bool = False

    @classmethod
    def load(cls, json_path: str, overrides: Optional[Dict[str, Any]] = None) -> "JeffreysDistillConfig":
        """Defaults → JSON file (if present) → ``overrides`` (e.g. CLI). Sets ``config_json_path``."""
        cfg = load_dataclass_from_json(cls, json_path, overrides)
        mj = cfg.meta_json
        if mj is not None and isinstance(mj, str) and not mj.strip():
            cfg = replace(cfg, meta_json=None)
        return cfg


FIELD_HELP_JEFFREYS: Dict[str, str] = {
    "meta_json": "Path to surgery_meta.json; empty string disables merge.",
    "quiet": "Less pipeline logging.",
}

# Default ``--config`` paths and parser descriptions (single source for all three CLIs).
CLI_PRETRAIN_DESCRIPTION = "Pet head training on timm DeiT-Tiny"
CLI_PRETRAIN_CONFIG_DEFAULT = "conf/pretrain_config.json"
CLI_SURGERY_RUN_DESCRIPTION = (
    "DeiT-Tiny surgery: timm Pet checkpoint → surgery student + surgery_meta.json"
)
CLI_SURGERY_RUN_CONFIG_DEFAULT = "conf/surgery_run_config.json"
CLI_JEFFREYS_DESCRIPTION = "Jeffreys distillation: timm teacher → surgery student"
CLI_JEFFREYS_CONFIG_DEFAULT = "conf/surgery_distill_config.json"
CLI_JEFFREYS_CONFIG_HELP = "JSON hyperparameters (merged with JeffreysDistillConfig defaults)."


def parse_pretrain_pet_config(argv: Optional[Sequence[str]] = None) -> PretrainPetConfig:
    """Parse argv (default ``sys.argv``) into a merged :class:`PretrainPetConfig`."""
    return parse_cli_config(
        PretrainPetConfig,
        description=CLI_PRETRAIN_DESCRIPTION,
        config_default=CLI_PRETRAIN_CONFIG_DEFAULT,
        field_help=FIELD_HELP_PRETRAIN,
        argv=argv,
    )


def parse_surgery_run_config(argv: Optional[Sequence[str]] = None) -> SurgeryRunConfig:
    """Parse argv into a merged :class:`SurgeryRunConfig`."""
    return parse_cli_config(
        SurgeryRunConfig,
        description=CLI_SURGERY_RUN_DESCRIPTION,
        config_default=CLI_SURGERY_RUN_CONFIG_DEFAULT,
        field_help=FIELD_HELP_SURGERY_RUN,
        argv=argv,
    )


# Backward-compatible alias (prefer :func:`set_default_device` + :func:`get_device`).
resolve_device = _resolve_device_string
