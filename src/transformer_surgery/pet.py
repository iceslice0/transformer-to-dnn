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
from contextlib import nullcontext
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
from torchmetrics.classification import MulticlassAccuracy
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from transformer_surgery.ops import (
    get_surgery_dtype,
    jeffreys_divergence_dense,
    set_surgery_dtype,
)

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
    Load JSON from ``json_path``, merge keys into ``cls()`` defaults, then ``overrides`` (e.g. CLI).
    Only keys that match dataclass fields are applied; ``config_json_path_field`` is the resolved
    absolute path. Missing/invalid files or JSON raise from ``open`` / ``json.load`` (no silent skip).
    """
    cfg = cls()
    allowed = {f.name for f in fields(cls)}
    skip = {config_json_path_field}
    ap = os.path.abspath(os.path.expanduser((json_path or "").strip()))
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
    """Collect ``argparse`` overrides: only keys present on ``args`` (omitted ``SUPPRESS`` flags are absent)."""
    patchable = {f.name for f in fields(cls)} - exclude
    avars = vars(args)
    out: Dict[str, Any] = {}
    for name in patchable:
        if name in avars:
            out[name] = avars[name]
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


# Process-wide default for loaders, checkpoints, and training loops. Call ``set_default_device`` once at startup.
TORCH_DEVICE: torch.device = torch.device("cpu")


def set_default_device(name: str) -> torch.device:
    """
    Store the training device in :data:`TORCH_DEVICE` (``torch.device(...)`` only).

    Does **not** call ``torch.set_default_device``: that API makes ops like ``torch.randperm`` used
    inside DataLoader shuffling expect a CUDA RNG and raises at sampler init. Training code should
    keep using ``.to(get_device())`` for model/tensors.

    ``name`` is any string accepted by ``torch.device`` (e.g. ``\"cuda\"``, ``\"cpu\"``, ``\"cuda:0\"``).
    Enables cudnn benchmark when the device is CUDA.
    """
    global TORCH_DEVICE
    TORCH_DEVICE = torch.device(name.strip())
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
        name = str(cfg.get("device", "cuda"))
    else:
        name = str(cfg.device)
    return set_default_device(name)


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        try:
            return f"cuda ({torch.cuda.get_device_name(device)})"
        except Exception:
            return "cuda"
    return str(device)


def describe_dtype(dt: torch.dtype) -> str:
    return str(dt).replace("torch.", "")


def apply_dtype_from_config(
    cfg: Union[Mapping[str, Any], Any],
) -> torch.dtype:
    """
    Resolve ``surgery_dtype`` from a run config mapping or dataclass and set the process default
    (``set_surgery_dtype``), mirroring :func:`apply_device_from_config`.
    Default name is ``bfloat16`` when the field is absent.
    """
    if isinstance(cfg, Mapping):
        name = str(cfg.get("surgery_dtype", "bfloat16"))
    else:
        name = str(getattr(cfg, "surgery_dtype", "bfloat16"))
    dt = getattr(torch, name.strip())
    set_surgery_dtype(dt)
    return dt


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
    try:
        input_dtype = next(model.parameters()).dtype
    except StopIteration:
        input_dtype = torch.float32
    for x, y in loader:
        x = x.to(device, dtype=input_dtype, non_blocking=use_cuda)
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
    v = vars(cfg)
    if "random_erasing_prob" in v:
        return float(v["random_erasing_prob"])
    return float(v["random_erasing"])


def _maybe_cuda_autocast(device: torch.device, dt: torch.dtype):
    """CUDA autocast for half/bfloat16 distillation; no-op elsewhere."""
    if device.type == "cuda" and dt in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dt)
    return nullcontext()


def _copy_trainable_state(src: nn.Module, dst: nn.Module) -> None:
    """Copy trainable floating-point state from ``src`` into ``dst`` preserving ``dst`` dtype."""
    with torch.no_grad():
        src_sd = src.state_dict()
        dst_sd = dst.state_dict()
        for name, tensor in dst_sd.items():
            if name not in src_sd:
                continue
            src_t = src_sd[name]
            if torch.is_floating_point(tensor):
                tensor.copy_(src_t.to(device=tensor.device, dtype=tensor.dtype))
            else:
                tensor.copy_(src_t.to(device=tensor.device))


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
    cfg: "PretrainPetConfig",
    *,
    resume_val_acc: Optional[float] = None,
) -> Tuple[float, float, Optional[int], int]:
    """
    Pet timm DeiT-Tiny: train classifier head only (AdamW + warmup + cosine per step).
    Hyperparameters from ``cfg``; optional ``resume_val_acc`` seeds best-so-far before epoch 1.
    Optional ``cfg.gap_th`` enables per-epoch gap revert vs best val acc.
    """
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

    scheduler = _warmup_cosine_scheduler(
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


def train_timm_deit_on_pet(
    ref: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: "PretrainPetConfig",
    *,
    resume_val_acc: Optional[float] = None,
) -> Tuple[float, float, Optional[int], int]:
    """Timm DeiT-Tiny Pet head training; delegates to :func:`finetune_model_adaptertune_style`."""
    return finetune_model_adaptertune_style(ref, train_loader, val_loader, cfg, resume_val_acc=resume_val_acc)


@torch.no_grad()
def eval_distillation_metrics(
    teacher: nn.Module,
    student: nn.Module,
    val_loader: DataLoader,
    temperature: float = 1.0,
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
    dt = get_surgery_dtype()
    for bi, (x, y) in enumerate(val_loader):
        x = x.to(device, dtype=dt, non_blocking=use_cuda)
        y = y.to(device, non_blocking=use_cuda)
        with _maybe_cuda_autocast(device, dt):
            t_log = teacher(x)
            s_log = student(x)
        ce_sum += torch.nn.functional.cross_entropy(s_log.float(), y, reduction="sum").item()
        j_sum += jeffreys_divergence_dense(t_log, s_log, temperature=temperature).sum().item()
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
    Train student with a mixed objective: hard-label cross-entropy plus Jeffreys teacher matching.
    Hyperparameters come from ``cfg`` (:class:`JeffreysDistillConfig`). Warmup length follows the
    same rule as the Jeffreys CLI: ``min(5, max(epochs, 1))`` when ``cfg.warmup_epochs`` is None.

    Returns (val_acc, val_ce_mean, val_jeffreys_mean, best_epoch_or_None).

    ``cfg.train_progress_interval``: print every N training batches. ``0`` = only per-epoch val lines.

    ``cfg.val_progress_batches``: val eval progress every N batches; ``0`` = silent until metrics done.
    """
    device = get_device()
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = len(train_loader)
    if cfg.max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, cfg.max_train_batches)
    epochs = cfg.epochs
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_epochs = min(5, max(cfg.epochs, 1)) if cfg.warmup_epochs is None else int(cfg.warmup_epochs)
    warmup_steps = min(warmup_epochs * steps_per_epoch, max(total_steps - 1, 0))
    scheduler = _warmup_cosine_scheduler(
        opt,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        eta_min=max(0.0, float(cfg.cosine_eta_min)),
    )
    use_cuda = device.type == "cuda"
    pf = f"{log_prefix} " if log_prefix else ""
    best_acc = float("-inf")
    best_state: Optional[dict] = None
    best_ep: Optional[int] = None

    print(
        f"  {pf}schedule: {epochs} epoch(s) × {steps_per_epoch} train batches "
        f"→ {total_steps} optimizer steps | train log every {cfg.train_progress_interval} batch(es)"
        + (
            f" | val log every {cfg.val_progress_batches} batch(es)"
            if cfg.val_progress_batches > 0
            else " | val silent"
        ),
        flush=True,
    )

    global_step = 0
    dt = get_surgery_dtype()
    use_master_fp32 = use_cuda and dt == torch.float16
    if use_master_fp32:
        train_student = copy.deepcopy(student).float()
        train_opt = torch.optim.AdamW(train_student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        train_scheduler = _warmup_cosine_scheduler(
            train_opt,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            eta_min=max(0.0, float(cfg.cosine_eta_min)),
        )
        # fp16 checkpoints remain the saved format; fp32 masters just drive the optimizer.
        train_student.train()
    else:
        train_student = student
        train_opt = opt
        train_scheduler = scheduler
    scaler = None
    baseline_student = train_student if use_master_fp32 else student
    baseline_teacher = teacher
    for p in baseline_teacher.parameters():
        p.requires_grad = False
    baseline_teacher.eval()
    baseline_acc, baseline_ce, baseline_j = eval_distillation_metrics(
        baseline_teacher,
        baseline_student,
        val_loader,
        temperature=cfg.temperature,
        progress_batches=cfg.val_progress_batches,
        progress_prefix=pf,
    )
    print(
        f"  {pf}baseline val acc={baseline_acc:.4f} ce={baseline_ce:.4f} jeffreys={baseline_j:.4f}",
        flush=True,
    )
    best_acc = baseline_acc
    best_ep = 0
    best_state = copy.deepcopy(train_student.state_dict() if use_master_fp32 else student.state_dict())
    for ep in range(epochs):
        train_student.train()
        n_batches = 0
        ep_t0 = time.perf_counter()
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device, dtype=dt, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)
            train_opt.zero_grad(set_to_none=True)
            if use_master_fp32:
                with torch.no_grad():
                    with _maybe_cuda_autocast(device, dt):
                        t_log = baseline_teacher(x)
                s_log = train_student(x.float())
            else:
                with torch.no_grad():
                    with _maybe_cuda_autocast(device, dt):
                        t_log = baseline_teacher(x)
                with _maybe_cuda_autocast(device, dt):
                    s_log = train_student(x)
            ce_loss = torch.nn.functional.cross_entropy(s_log.float(), y, reduction="mean")
            j_loss = jeffreys_divergence_dense(t_log, s_log, temperature=cfg.temperature).mean()
            mix = float(cfg.distill_weight)
            loss = (1.0 - mix) * ce_loss + mix * j_loss
            if use_master_fp32:
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(train_student.parameters(), cfg.grad_clip)
                train_opt.step()
            else:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if cfg.grad_clip > 0:
                        scaler.unscale_(train_opt)
                        torch.nn.utils.clip_grad_norm_(train_student.parameters(), cfg.grad_clip)
                    scaler.step(train_opt)
                    scaler.update()
                else:
                    loss.backward()
                    if cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(train_student.parameters(), cfg.grad_clip)
                    train_opt.step()
            train_scheduler.step()
            n_batches += 1
            global_step += 1
            li = float(loss.item())
            running_loss += li
            if cfg.train_progress_interval > 0:
                do_log = (
                    n_batches == 1
                    or n_batches % cfg.train_progress_interval == 0
                    or n_batches == steps_per_epoch
                )
                if cfg.max_train_batches is not None and n_batches >= cfg.max_train_batches:
                    do_log = True
                if do_log:
                    elapsed = time.perf_counter() - ep_t0
                    avg = running_loss / n_batches
                    rate = n_batches / elapsed if elapsed > 0 else 0.0
                    left = steps_per_epoch - n_batches
                    eta_s = left / rate if rate > 0 else 0.0
                    lr_c = train_opt.param_groups[0]["lr"]
                    print(
                        f"  {pf}epoch {ep + 1}/{epochs} train {n_batches}/{steps_per_epoch} "
                        f"step {global_step}/{total_steps} loss={li:.6f} loss_avg={avg:.6f} "
                        f"ce={float(ce_loss.item()):.6f} j={float(j_loss.item()):.6f} "
                        f"lr={lr_c:.2e} {rate:.2f} batch/s epoch_eta~{eta_s / 60.0:.1f}m",
                        flush=True,
                    )
            if cfg.max_train_batches is not None and n_batches >= cfg.max_train_batches:
                break
        if use_master_fp32:
            _copy_trainable_state(train_student, student)
        else:
            student = train_student
        acc, ce_v, j_v = eval_distillation_metrics(
            baseline_teacher,
            student,
            val_loader,
            temperature=cfg.temperature,
            progress_batches=cfg.val_progress_batches,
            progress_prefix=pf,
        )
        print(
            f"  {pf}epoch {ep + 1}/{epochs} | val acc={acc:.4f} ce={ce_v:.4f} jeffreys={j_v:.4f}",
            flush=True,
        )
        if cfg.keep_best and acc > best_acc:
            best_acc = acc
            best_ep = ep + 1
            best_state = copy.deepcopy(train_student.state_dict() if use_master_fp32 else student.state_dict())
    if cfg.keep_best and best_state is not None:
        if use_master_fp32:
            train_student.load_state_dict(best_state)
            _copy_trainable_state(train_student, student)
        else:
            student.load_state_dict(best_state)
        print(
            f"  {pf}kept best val acc={best_acc:.4f} (epoch {best_ep}/{epochs})",
            flush=True,
        )
    if use_master_fp32:
        _copy_trainable_state(train_student, student)
    acc_f, ce_f, j_f = eval_distillation_metrics(
        baseline_teacher,
        student,
        val_loader,
        temperature=cfg.temperature,
        progress_batches=cfg.val_progress_batches,
        progress_prefix=pf,
    )
    best_i = best_ep if cfg.keep_best else None
    return acc_f, ce_f, j_f, best_i


# Backward-compatible aliases for generic runtime and processing helpers.
#
# Pretraining remains the concrete Pet/DeiT recipe in this module. Surgery, distillation, and PTQ
# use the adapter-aware implementations from ``pipeline`` / ``model_adapters``; the aliases keep
# existing imports from ``transformer_surgery.pet`` working.
from transformer_surgery.cli.distill_config import (
    CLI_JEFFREYS_CONFIG_DEFAULT as CLI_JEFFREYS_CONFIG_DEFAULT,
    CLI_JEFFREYS_CONFIG_HELP as CLI_JEFFREYS_CONFIG_HELP,
    CLI_JEFFREYS_DESCRIPTION as CLI_JEFFREYS_DESCRIPTION,
    FIELD_HELP_JEFFREYS as FIELD_HELP_JEFFREYS,
    JeffreysDistillConfig as JeffreysDistillConfig,
)
from transformer_surgery.cli.pretrain_config import (
    CLI_PRETRAIN_CONFIG_DEFAULT as CLI_PRETRAIN_CONFIG_DEFAULT,
    CLI_PRETRAIN_DESCRIPTION as CLI_PRETRAIN_DESCRIPTION,
    FIELD_HELP_PRETRAIN as FIELD_HELP_PRETRAIN,
    PretrainPetConfig as PretrainPetConfig,
    parse_pretrain_pet_config as parse_pretrain_pet_config,
    pretrain_train_config_record as pretrain_train_config_record,
)
from transformer_surgery.cli.surgery_config import (
    CLI_SURGERY_RUN_CONFIG_DEFAULT as CLI_SURGERY_RUN_CONFIG_DEFAULT,
    CLI_SURGERY_RUN_DESCRIPTION as CLI_SURGERY_RUN_DESCRIPTION,
    FIELD_HELP_SURGERY_RUN as FIELD_HELP_SURGERY_RUN,
    SurgeryRunConfig as SurgeryRunConfig,
    parse_surgery_run_config as parse_surgery_run_config,
)
from transformer_surgery.model_adapters import load_surgery_student_checkpoint as load_surgery_student_checkpoint
from transformer_surgery.pipeline import (
    accuracy_and_loss as accuracy_and_loss,
    add_dataclass_cli_args as add_dataclass_cli_args,
    apply_device_from_config as apply_device_from_config,
    apply_dtype_from_config as apply_dtype_from_config,
    build_config_cli_parser as build_config_cli_parser,
    cli_overrides_from_namespace as cli_overrides_from_namespace,
    describe_device as describe_device,
    describe_dtype as describe_dtype,
    distill_student_from_teacher_jeffreys as distill_surgery_from_teacher_jeffreys,
    eval_distillation_metrics as eval_distillation_metrics,
    get_device as get_device,
    load_dataclass_from_json as load_dataclass_from_json,
    merge_post_distill_into_surgery_meta as merge_post_distill_into_surgery_meta,
    parse_cli_config as parse_cli_config,
    save_model_checkpoint as save_deit_checkpoint,
    set_default_device as set_default_device,
    set_seed as set_seed,
)
