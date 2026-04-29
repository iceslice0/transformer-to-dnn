"""
Model-agnostic pipeline utilities for surgery, distillation, and PTQ.

Concrete datasets and patient architectures live behind model adapters. This module owns only
shared config parsing, runtime device/dtype state, generic evaluation, distillation, and checkpoint
metadata helpers.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import fields, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union, get_args, get_origin, get_type_hints

import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR
from torch.utils.data import DataLoader

from transformer_surgery.ops import CALIBRATION_LEGEND_TEXT, jeffreys_divergence_dense, set_surgery_dtype


TConfig = TypeVar("TConfig")
DEFAULT_MODEL_KEY = "deit_tiny_pet"


def load_dataclass_from_json(
    cls: Type[TConfig],
    json_path: str,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    config_json_path_field: str = "config_json_path",
) -> TConfig:
    """
    Load JSON into dataclass defaults, then apply explicit overrides.

    Unknown JSON keys are ignored so older model-specific configs continue to load while generic
    configs can add adapter fields incrementally.
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
    if overrides:
        kwargs = {k: v for k, v in overrides.items() if k in allowed and k not in skip}
        if kwargs:
            cfg = replace(cfg, **kwargs)
    return replace(cfg, **{config_json_path_field: ap})


def cli_overrides_from_namespace(
    args: Any,
    cls: Type[Any],
    *,
    exclude: frozenset[str] = frozenset({"config", "config_json_path"}),
) -> Dict[str, Any]:
    patchable = {f.name for f in fields(cls)} - exclude
    avars = vars(args)
    return {name: avars[name] for name in patchable if name in avars}


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
    """Add one ``--kebab-case`` argparse flag for each supported dataclass field."""
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
                parser.add_argument(flag, dest=name, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, help=h)
            else:
                parser.add_argument(flag, dest=name, action="store_true", default=argparse.SUPPRESS, help=h)
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
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", dest=config_dest, type=str, default=config_default, help=config_help)
    add_dataclass_cli_args(parser, config_cls, field_help=field_help)
    return parser


def parse_cli_config(
    config_cls: Type[TConfig],
    *,
    description: str,
    config_default: str,
    config_help: str = "JSON hyperparameters (merged with dataclass defaults).",
    field_help: Optional[Dict[str, str]] = None,
    argv: Optional[Sequence[str]] = None,
) -> TConfig:
    parser = build_config_cli_parser(
        description,
        config_cls,
        config_default=config_default,
        config_help=config_help,
        field_help=field_help,
    )
    args = parser.parse_args(argv)
    return config_cls.load(args.config, cli_overrides_from_namespace(args, config_cls))


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


TORCH_DEVICE: torch.device = torch.device("cpu")


def set_default_device(name: str) -> torch.device:
    global TORCH_DEVICE
    TORCH_DEVICE = torch.device(name.strip())
    if TORCH_DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return TORCH_DEVICE


def get_device() -> torch.device:
    return TORCH_DEVICE


def apply_device_from_config(cfg: Union[Mapping[str, Any], Any]) -> torch.device:
    name = str(cfg.get("device", "cuda")) if isinstance(cfg, Mapping) else str(cfg.device)
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


def dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).strip().replace("torch.", "")
    return getattr(torch, normalized)


def apply_dtype_from_config(cfg: Union[Mapping[str, Any], Any]) -> torch.dtype:
    if isinstance(cfg, Mapping):
        name = str(cfg.get("surgery_dtype", "bfloat16"))
    else:
        name = str(getattr(cfg, "surgery_dtype", "bfloat16"))
    dt = dtype_from_name(name)
    set_surgery_dtype(dt)
    return dt


@torch.no_grad()
def accuracy_and_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    num_classes: Optional[int] = None,
) -> Tuple[float, float]:
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


def _maybe_cuda_autocast(device: torch.device, dt: torch.dtype):
    if device.type == "cuda" and dt in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dt)
    return nullcontext()


def _copy_trainable_state(src: nn.Module, dst: nn.Module) -> None:
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


def _warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    eta_min: float,
) -> LRScheduler:
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    warmup_steps = min(warmup_steps, total_steps)
    if warmup_steps <= 0:
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)
    if warmup_steps >= total_steps:
        return LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=total_steps)
    cosine_steps = max(1, total_steps - warmup_steps)
    warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=eta_min)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])


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
    """Student accuracy, mean CE, and mean Jeffreys divergence against a teacher."""
    from transformer_surgery.ops import get_surgery_dtype

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


def distill_student_from_teacher_jeffreys(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Any,
    *,
    log_prefix: str = "distill",
) -> Tuple[float, float, float, Optional[int]]:
    """Train any classifier student with mixed hard-label CE plus Jeffreys teacher matching."""
    from transformer_surgery.ops import get_surgery_dtype

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
    print(
        f"  {pf}schedule: {epochs} epoch(s) x {steps_per_epoch} train batches "
        f"-> {total_steps} optimizer steps | train log every {cfg.train_progress_interval} batch(es)"
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
        train_student.train()
    else:
        train_student = student
        train_opt = opt
        train_scheduler = scheduler
    baseline_teacher = teacher
    for p in baseline_teacher.parameters():
        p.requires_grad = False
    baseline_teacher.eval()
    baseline_student = train_student if use_master_fp32 else student
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
    best_ep: Optional[int] = 0
    best_state: Optional[dict] = copy.deepcopy(train_student.state_dict() if use_master_fp32 else student.state_dict())
    for ep in range(epochs):
        train_student.train()
        n_batches = 0
        ep_t0 = time.perf_counter()
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device, dtype=dt, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)
            train_opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                with _maybe_cuda_autocast(device, dt):
                    t_log = baseline_teacher(x)
            if use_master_fp32:
                s_log = train_student(x.float())
            else:
                with _maybe_cuda_autocast(device, dt):
                    s_log = train_student(x)
            ce_loss = torch.nn.functional.cross_entropy(s_log.float(), y, reduction="mean")
            j_loss = jeffreys_divergence_dense(t_log, s_log, temperature=cfg.temperature).mean()
            mix = float(cfg.distill_weight)
            loss = (1.0 - mix) * ce_loss + mix * j_loss
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
        print(f"  {pf}epoch {ep + 1}/{epochs} | val acc={acc:.4f} ce={ce_v:.4f} jeffreys={j_v:.4f}", flush=True)
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
        print(f"  {pf}kept best val acc={best_acc:.4f} (epoch {best_ep}/{epochs})", flush=True)
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
    return acc_f, ce_f, j_f, best_ep if cfg.keep_best else None


def save_model_checkpoint(path: str, model: nn.Module, extra: Optional[Dict[str, Any]] = None) -> None:
    torch.save({"model_state_dict": model.state_dict(), "extra": extra or {}}, path)


def merge_post_distill_into_surgery_meta(
    meta_path: str,
    val_acc: float,
    val_ce: float,
    val_jeffreys: float,
    *,
    model_key: str = DEFAULT_MODEL_KEY,
    patient: str = "unknown",
    dataset: str = "unknown",
) -> None:
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = {
            "model_key": model_key,
            "patient": patient,
            "dataset": dataset,
            "calibration": {},
            "pwl": {
                "note": "No surgery run meta on disk; run the surgery CLI first for PWL metadata.",
            },
            "meta_note": "Stub created before distill (no prior surgery_meta at this path).",
        }
    raw.setdefault("model_key", model_key)
    raw.setdefault("patient", patient)
    raw.setdefault("dataset", dataset)
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
    os.makedirs(os.path.dirname(os.path.abspath(meta_path)) or ".", exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)


_CLI_COMPAT_EXPORTS: Dict[str, str] = {
    "SurgeryRunConfig": "transformer_surgery.cli.surgery_config",
    "FIELD_HELP_SURGERY_RUN": "transformer_surgery.cli.surgery_config",
    "CLI_SURGERY_RUN_DESCRIPTION": "transformer_surgery.cli.surgery_config",
    "CLI_SURGERY_RUN_CONFIG_DEFAULT": "transformer_surgery.cli.surgery_config",
    "parse_surgery_run_config": "transformer_surgery.cli.surgery_config",
    "JeffreysDistillConfig": "transformer_surgery.cli.distill_config",
    "FIELD_HELP_JEFFREYS": "transformer_surgery.cli.distill_config",
    "CLI_JEFFREYS_DESCRIPTION": "transformer_surgery.cli.distill_config",
    "CLI_JEFFREYS_CONFIG_DEFAULT": "transformer_surgery.cli.distill_config",
    "CLI_JEFFREYS_CONFIG_HELP": "transformer_surgery.cli.distill_config",
}


def __getattr__(name: str) -> Any:
    module_name = _CLI_COMPAT_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
