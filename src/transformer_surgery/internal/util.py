"""Device/seed, surgery dtype, model inputs, schedules, checkpoints, weight copy, and validation metrics."""


from __future__ import annotations

import random
from contextlib import nullcontext
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR
from torch.utils.data import DataLoader, Dataset

DEFAULT_MODEL_KEY = "deit_tiny_pet"

_SURGERY_DTYPE: torch.dtype = torch.bfloat16
_DEVICE: torch.device = torch.device("cpu")


def get_surgery_dtype() -> torch.dtype:
    return _SURGERY_DTYPE


def set_surgery_dtype(dt: torch.dtype) -> None:
    global _SURGERY_DTYPE
    _SURGERY_DTYPE = dt


def get_device() -> torch.device:
    return _DEVICE


def set_default_device(device: torch.device) -> torch.device:
    global _DEVICE
    _DEVICE = device
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return device


def maybe_surgery_cuda_autocast(device: torch.device, dt: torch.dtype):
    if device.type == "cuda" and dt in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dt)
    return nullcontext()


def torch_dtype_from_name(name: str) -> torch.dtype:
    value = getattr(torch, str(name).strip().replace("torch.", ""), None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"Unknown torch dtype {name!r}")
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def model_dtype(model: nn.Module) -> torch.dtype:
    """Dtype of the first parameter; the surgery dtype for a model without parameters."""
    p = next(model.parameters(), None)
    return get_surgery_dtype() if p is None else p.dtype


def to_model_input(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Move a batch to the run device; float inputs take the model dtype, token ids stay integer."""
    device = get_device()
    dtype = model_dtype(model) if x.is_floating_point() else x.dtype
    return x.to(device, dtype=dtype, non_blocking=device.type == "cuda")


@torch.no_grad()
def accuracy_and_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Tuple[float, float]:
    """Top-1 accuracy and the sample mean of ``criterion`` on ``loader``."""
    model.eval()
    device = get_device()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    correct = torch.zeros((), device=device, dtype=torch.long)
    n = 0
    for x, y in loader:
        x = to_model_input(x, model)
        y = y.to(device, non_blocking=device.type == "cuda")
        with maybe_surgery_cuda_autocast(device, get_surgery_dtype()):
            logits = model(x)
        loss_sum += criterion(logits, y).double() * y.size(0)
        correct += (logits.argmax(dim=-1) == y).sum()
        n += y.size(0)
    return correct.item() / n, loss_sum.item() / n


def save_model_checkpoint(path: str, model: nn.Module, extra: Optional[Dict[str, Any]] = None) -> None:
    torch.save({"model_state_dict": model.state_dict(), "extra": extra or {}}, path)


def copy_matching_weights(
    model: nn.Module, reference_state: Mapping[str, torch.Tensor], reference_key: Callable[[str], str]
) -> Dict[str, str]:
    """Copy each reference tensor into the ``model`` entry whose mapped name and shape match; returns the name map."""
    mapping: Dict[str, str] = {}
    with torch.no_grad():
        for key, tensor in model.state_dict().items():
            ref_key = reference_key(key)
            ref = reference_state.get(ref_key)
            if ref is not None and ref.shape == tensor.shape:
                tensor.copy_(ref)
                mapping[key] = ref_key
    return mapping


def warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    eta_min: float,
) -> LRScheduler:
    """Linear warmup then cosine decay, stepped once per optimizer step."""
    total_steps = max(1, int(total_steps))
    warmup_steps = min(max(0, int(warmup_steps)), total_steps)
    if warmup_steps == 0:
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)
    warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    if warmup_steps == total_steps:
        return warmup
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=eta_min)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])


def make_loaders(
    train_set: Dataset, val_set: Dataset, cfg: Any, *, val_batch_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """Shuffled train and ordered val loaders with the run's batch size and workers."""
    kw = dict(num_workers=cfg.workers, pin_memory=get_device().type == "cuda", persistent_workers=cfg.workers > 0)
    return (
        DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, **kw),
        DataLoader(val_set, batch_size=val_batch_size or cfg.batch_size, shuffle=False, **kw),
    )
