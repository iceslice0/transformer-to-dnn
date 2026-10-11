"""Count unary (pointwise) evaluations of a surgery graph in one forward pass.

Every ``NL*`` op module contributes the number of entries it outputs: the raw module-label count, measured on real
inputs. A basis-level count may classify the same kernel by the operation of that basis: extended LayerNorm's
``NLSquare`` computes ``x * x`` like ``AffineSelfHadamard`` and counts there as the Hadamard product ``x ⊙ x``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from transformer_surgery.internal.util import get_device, get_surgery_dtype, maybe_surgery_cuda_autocast, to_model_input


def _is_unary_module(m: nn.Module) -> bool:
    return type(m).__name__.startswith("NL") and type(m).__module__.startswith("transformer_surgery")


@dataclass
class PointwiseCount:
    """Unary evaluations per input example, total and by ``NL*`` op type."""

    per_input: float
    by_type: Dict[str, float]


@torch.no_grad()
def count_pointwise_per_input(model: nn.Module, x: torch.Tensor) -> PointwiseCount:
    """Unary evaluations per input example for one batch ``x``, total and by op type."""
    by_type: Dict[str, int] = {}

    def add(name: str, n: int) -> None:
        by_type[name] = by_type.get(name, 0) + n

    handles = [
        m.register_forward_hook(lambda mod, _i, out: add(type(mod).__name__, out.numel()))
        for m in model.modules()
        if _is_unary_module(m)
    ]
    model.eval()
    try:
        with maybe_surgery_cuda_autocast(get_device(), get_surgery_dtype()):
            model(to_model_input(x, model))
    finally:
        for h in handles:
            h.remove()
    batch = int(x.shape[0])
    return PointwiseCount(sum(by_type.values()) / batch, {k: v / batch for k, v in sorted(by_type.items())})
