"""Text reports and metadata constants."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .runtime import get_surgery_dtype

CALIBRATION_LEGEND_TEXT = (
    "ref_val_acc / ref_val_loss: frozen timm teacher on val (mean CE). "
    "student_pre_ft_val_acc / student_pre_ft_mean_ce: surgery student on val "
    "after transform, before distill. "
    "student_post_distill_*: after Jeffreys distillation (acc and mean CE / Jeffreys). "
    "gibbs_tail_prob_eps_calibrated_*: observed dense-softmax omitted tail mass for top-k scores; "
    "gibbs_tail_prob_eps_applied_*: values copied into GibbsTopKSoftmax parameters."
)


def _forward_output_shape_str(out: Any) -> str:
    if torch.is_tensor(out):
        return str(tuple(out.shape))
    if isinstance(out, (tuple, list)):
        return "(" + ", ".join(_forward_output_shape_str(x) for x in out) + ")"
    return type(out).__name__


def write_model_structure_txt(
    path: str,
    model: nn.Module,
    title: str,
    *,
    example_input: Optional[torch.Tensor] = None,
    default_input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    include_forward_shapes: bool = True,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines: List[str] = [
        title,
        "=" * min(80, max(len(title), 40)),
        f"class: {type(model).__name__}",
        f"parameters: total={n_all:,} trainable={n_train:,}",
        "",
        str(model),
        "",
        "--- named_modules (name: class) ---",
        "",
    ]
    for name, mod in model.named_modules():
        lines.append(f"{name if name else '<root>'}: {type(mod).__name__}")

    if include_forward_shapes:
        out_shapes: Dict[str, str] = {}
        shape_err: Optional[str] = None
        x_log: Optional[torch.Tensor] = None
        try:
            try:
                device = next(model.parameters()).device
                dtype = next(model.parameters()).dtype
            except StopIteration:
                device = torch.device("cpu")
                dtype = get_surgery_dtype()
            x = example_input
            if x is None:
                x = torch.zeros(default_input_shape, device=device, dtype=dtype)
            else:
                x = x.to(device=device, dtype=dtype)
            x_log = x

            hooks: List[Any] = []

            def _make_hook(key: str):
                def _hook(_mod: nn.Module, _inp: Any, out: Any) -> None:
                    out_shapes[key] = _forward_output_shape_str(out)

                return _hook

            for name, mod in model.named_modules():
                key = name if name else "<root>"
                hooks.append(mod.register_forward_hook(_make_hook(key)))

            was_training = model.training
            model.eval()
            with torch.no_grad():
                model(x)
            if was_training:
                model.train()
            for handle in hooks:
                handle.remove()
        except Exception as ex:
            shape_err = repr(ex)

        lines.extend(
            [
                "",
                "--- forward output shapes (one eval batch; dummy input unless example_input set) ---",
            ]
        )
        if shape_err is not None:
            lines.append(f"(forward shape trace failed: {shape_err})")
        elif x_log is not None:
            lines.append(f"example_input: {tuple(x_log.shape)}  dtype={x_log.dtype}  device={x_log.device}")
            lines.append("")
            for name, _mod in model.named_modules():
                key = name if name else "<root>"
                label = name if name else "<root>"
                lines.append(f"{label}: {out_shapes.get(key, '-')}")
        else:
            lines.append("(no example batch; shapes skipped)")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
