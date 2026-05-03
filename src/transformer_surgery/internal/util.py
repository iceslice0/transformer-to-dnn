"""Device/seed, namespaces, schedules, checkpoints, and validation metrics (no CLI logging)."""


from __future__ import annotations

import os
import random
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

from .runtime import get_surgery_dtype, maybe_surgery_cuda_autocast

DEFAULT_MODEL_KEY = "deit_tiny_pet"


def namespace_from_mapping(obj: Any) -> Any:
    """Recursively turn dict trees into ``types.SimpleNamespace`` (lists preserved element-wise)."""
    if isinstance(obj, Mapping) and not isinstance(obj, (str, bytes, bytearray)):
        return SimpleNamespace(**{str(k): namespace_from_mapping(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [namespace_from_mapping(v) for v in obj]
    return obj


def namespace_to_mapping(obj: Any) -> Any:
    """Inverse of :func:`namespace_from_mapping` for JSON / ``torch.save`` payloads."""
    if isinstance(obj, SimpleNamespace):
        return {k: namespace_to_mapping(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [namespace_to_mapping(v) for v in obj]
    return obj


def ensure_mapping(obj: Any) -> Dict[str, Any]:
    """Shallow ``dict`` from a ``Mapping``, or recursive plain dict from a nested ``SimpleNamespace``."""
    if isinstance(obj, SimpleNamespace):
        return namespace_to_mapping(obj)
    if isinstance(obj, Mapping):
        return dict(obj)
    raise TypeError(f"expected Mapping or SimpleNamespace, got {type(obj).__name__}")


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


def set_default_device(device: torch.device) -> torch.device:
    global TORCH_DEVICE
    TORCH_DEVICE = device
    if TORCH_DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return TORCH_DEVICE


def get_device() -> torch.device:
    return TORCH_DEVICE


def torch_dtype_from_name(name: str) -> torch.dtype:
    key = str(name).strip().replace("torch.", "")
    try:
        value = getattr(torch, key)
    except AttributeError as exc:
        raise ValueError(f"Unknown torch dtype {name!r}") from exc
    return value


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
    loss_sum_t = torch.zeros((), device=device, dtype=torch.float64)
    n_samples = 0
    try:
        input_dtype = next(model.parameters()).dtype
    except StopIteration:
        input_dtype = torch.float32
    dt_eval = get_surgery_dtype()
    for x, y in loader:
        x = x.to(device, dtype=input_dtype, non_blocking=use_cuda)
        y = y.to(device, non_blocking=use_cuda)
        with maybe_surgery_cuda_autocast(device, dt_eval):
            logits = model(x)
        bs = y.size(0)
        loss_sum_t += criterion(logits, y).double() * bs
        n_samples += bs
        if acc_metric is None:
            nc = int(num_classes) if num_classes is not None else int(logits.shape[-1])
            acc_metric = MulticlassAccuracy(num_classes=nc, average="micro").to(device)
        acc_metric.update(logits, y)
    if acc_metric is None or n_samples == 0:
        return 0.0, 0.0
    return float(acc_metric.compute().item()), loss_sum_t.item() / n_samples


def save_model_checkpoint(path: str, model: nn.Module, extra: Optional[Dict[str, Any]] = None) -> None:
    torch.save({"model_state_dict": model.state_dict(), "extra": extra or {}}, path)


def warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    eta_min: float,
) -> LRScheduler:
    """Linear warmup then cosine decay, stepped once per optimizer step."""
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
