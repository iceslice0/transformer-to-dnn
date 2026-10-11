"""Super-resolution metrics (Y-channel PSNR/SSIM) and spatial alignment.

Ported from the reference experiment ``mambal/softmax/softmax_surgery.py`` so the framework's
MambaIR SR path reports the same numbers. Tensors are ``[B, 3, H, W]`` in ``[0, 1]``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    """BT.601 luma from RGB in ``[0, 1]``; input ``[B, 3, H, W]`` -> ``[B, 1, H, W]``."""
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return (16.0 / 255.0) + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0


def _prepare_metric_arrays(
    pred: torch.Tensor, target: torch.Tensor, crop_border: int, y_channel: bool
) -> Tuple[np.ndarray, np.ndarray]:
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    if crop_border > 0:
        pred = pred[..., crop_border:-crop_border, crop_border:-crop_border]
        target = target[..., crop_border:-crop_border, crop_border:-crop_border]
    if y_channel:
        pred = rgb_to_y(pred)
        target = rgb_to_y(target)
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    if pred_np.shape[1] == 1:
        return pred_np[:, 0], target_np[:, 0]
    return np.transpose(pred_np, (0, 2, 3, 1)), np.transpose(target_np, (0, 2, 3, 1))


def calc_psnr(pred: torch.Tensor, target: torch.Tensor, crop_border: int, y_channel: bool = True) -> torch.Tensor:
    pred_np, target_np = _prepare_metric_arrays(pred, target, crop_border, y_channel)
    scores = [peak_signal_noise_ratio(target_np[i], pred_np[i], data_range=1.0) for i in range(pred_np.shape[0])]
    return torch.tensor(float(np.mean(scores)), dtype=pred.dtype, device=pred.device)


def calc_ssim(pred: torch.Tensor, target: torch.Tensor, crop_border: int, y_channel: bool = True) -> torch.Tensor:
    """Standard skimage SSIM averaged over batch."""
    pred_np, target_np = _prepare_metric_arrays(pred, target, crop_border, y_channel)
    channel_axis = None if pred_np.ndim == 3 else -1
    scores = [
        structural_similarity(target_np[i], pred_np[i], data_range=1.0, channel_axis=channel_axis)
        for i in range(pred_np.shape[0])
    ]
    return torch.tensor(float(np.mean(scores)), dtype=pred.dtype, device=pred.device)


def _center_crop(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    dh, dw = t.shape[-2] - h, t.shape[-1] - w
    return t[..., dh // 2 : dh // 2 + h, dw // 2 : dw // 2 + w]


def align_spatial(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Center-crop pred/target to a shared spatial size for safe loss/metric eval."""
    h = min(pred.shape[-2], target.shape[-2])
    w = min(pred.shape[-1], target.shape[-1])
    return _center_crop(pred, h, w), _center_crop(target, h, w)
