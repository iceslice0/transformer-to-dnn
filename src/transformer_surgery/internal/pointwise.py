"""Count unary (pointwise) evaluations of a surgery graph in one forward pass.

Every ``NL*`` op module contributes the number of entries it outputs. This is
the raw module-label count. A basis-level count may classify the same kernel by
the operation available in that basis: extended LayerNorm's ``NLSquare`` executes
``x * x``, identical to ``AffineSelfHadamard``, and therefore counts as the
Hadamard product ``x ⊙ x`` rather than as a unary evaluation in the extended
basis. The strict
:class:`~transformer_surgery.ops.RewrittenLayerNorm` splits signs with two ReLU evaluations per entry
(``relu(u)``, ``relu(-u)``) in its deprecated log-domain form, which run through the routing alias and are added here. This is the
raw pointwise count, measured on real inputs.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from transformer_surgery.internal.util import get_device, get_surgery_dtype, maybe_surgery_cuda_autocast


def _is_unary_module(m: nn.Module) -> bool:
    return type(m).__name__.startswith("NL") and type(m).__module__.startswith("transformer_surgery")


@torch.no_grad()
def count_pointwise_per_input(model: nn.Module, x: torch.Tensor) -> Dict[str, Any]:
    """Unary evaluations per input example for one batch ``x``, total and by op type."""
    from transformer_surgery.ops import RewrittenLayerNorm

    by_type: Dict[str, int] = {}
    handles: List[Any] = []

    def add(name: str, n: int) -> None:
        by_type[name] = by_type.get(name, 0) + int(n)

    for m in model.modules():
        if _is_unary_module(m):
            handles.append(m.register_forward_hook(lambda mod, _i, out: add(type(mod).__name__, out.numel())))
        elif isinstance(m, RewrittenLayerNorm) and not m.allow_matmul and m.strict_unary == "log_exp":
            handles.append(m.register_forward_hook(lambda _mod, inp, _o: add("ReLU", 2 * inp[0].numel())))
    device = get_device()
    model.eval()
    try:
        if x.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            x = x.to(device)
        else:
            x = x.to(device, dtype=next(model.parameters()).dtype)
        with maybe_surgery_cuda_autocast(device, get_surgery_dtype()):
            model(x)
    finally:
        for h in handles:
            h.remove()
    batch = int(x.shape[0])
    return {
        "pointwise_per_input": sum(by_type.values()) / batch,
        "pointwise_per_input_by_type": {k: v / batch for k, v in sorted(by_type.items())},
    }
