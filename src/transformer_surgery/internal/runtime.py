"""Runtime dtype helpers shared by surgery, distill, and PTQ."""

from __future__ import annotations

from contextlib import nullcontext

import torch

_SURGERY_DTYPE: torch.dtype = torch.bfloat16


def get_surgery_dtype() -> torch.dtype:
    return _SURGERY_DTYPE


def set_surgery_dtype(dt: torch.dtype) -> None:
    global _SURGERY_DTYPE
    _SURGERY_DTYPE = dt


def maybe_surgery_cuda_autocast(device: torch.device, dt: torch.dtype):
    if device.type == "cuda" and dt in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dt)
    return nullcontext()
