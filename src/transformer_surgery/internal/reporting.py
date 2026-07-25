"""Text reports, structure dumps, traceable paths, and CLI-style logging."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .util import get_surgery_dtype

CALIBRATION_LEGEND_TEXT = (
    "ref_val_acc / ref_val_loss: frozen timm teacher on val (mean CE). "
    "student_pre_ft_val_acc / student_pre_ft_mean_ce: surgery student on val "
    "after transform, before distill. "
    "student_post_distill_*: after Jeffreys distillation (acc and mean CE / Jeffreys). "
    "gibbs_tail_prob_eps_exact_*: observed dense-softmax omitted tail mass for top-k scores "
    "(mean/std/min/max on train batches); "
    "gibbs_tail_prob_eps_calibrated_*: per-block means copied into GibbsTopKSoftmax when not using "
    "exact runtime tail mass; "
    "gibbs_tail_prob_eps_applied_*: values copied into GibbsTopKSoftmax parameters; "
    "use_exact_tail_mass: runtime q_tail via N*mean(exp) centroid partition instead of calibrated scalar."
)


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        try:
            return f"cuda ({torch.cuda.get_device_name(device)})"
        except Exception:
            return "cuda"
    return str(device)


def describe_dtype(dt: torch.dtype) -> str:
    return str(dt).replace("torch.", "")


def log_line(msg: str) -> None:
    print(msg, flush=True)


def log_wrote(path: str) -> None:
    print(f"wrote {path}", flush=True)


def log_json_block(title: str, obj: Any, *, indent: int = 2) -> None:
    print(title, json.dumps(obj, indent=indent), flush=True)


def write_json(path: str, data: Any, *, indent: int = 2) -> None:
    ap = os.path.abspath(path)
    os.makedirs(os.path.dirname(ap) or ".", exist_ok=True)
    with open(ap, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def _slug_part(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "artifact"


def config_artifact_stem(cfg_or_path: Any, tool_name: str) -> str:
    """
    Stable artifact stem from the CLI/tool and active JSON config name.

    ``tool_name`` may be an entry point like ``ts-surgery``; filenames use a filesystem-safe
    underscore slug such as ``ts_surgery_topk64_fast``.
    """
    if isinstance(cfg_or_path, (str, os.PathLike)):
        config_path = os.fspath(cfg_or_path)
    else:
        cjp = cfg_or_path.config_json_path
        config_path = os.fspath(cjp) if cjp else ""
    config_name = os.path.splitext(os.path.basename(config_path))[0] if config_path else "config"
    tool_parts = _slug_part(tool_name).split("_")
    config_parts = _slug_part(config_name).split("_")
    if tool_parts and config_parts and tool_parts[-1] == config_parts[0]:
        config_parts = config_parts[1:]
    return "_".join(tool_parts + config_parts)


def traceable_artifact_path(
    path: str,
    cfg_or_path: Any,
    tool_name: str,
    artifact_name: str = "",
    extension: Optional[str] = None,
) -> str:
    """Return ``path``'s directory plus ``<tool>_<config>[_artifact]<extension>``."""
    original = os.path.abspath(path)
    directory = os.path.dirname(original) or "."
    ext = extension if extension is not None else os.path.splitext(original)[1]
    stem = config_artifact_stem(cfg_or_path, tool_name)
    artifact = _slug_part(artifact_name) if artifact_name else ""
    filename = f"{stem}_{artifact}{ext}" if artifact else f"{stem}{ext}"
    return os.path.join(directory, filename)


def traceable_log_path(log_dir: str, cfg_or_path: Any, tool_name: str, log_name: str) -> str:
    return os.path.join(
        os.path.abspath(log_dir),
        f"{config_artifact_stem(cfg_or_path, tool_name)}_{_slug_part(log_name)}.txt",
    )


def metadata_path_for_checkpoint(checkpoint_path: str, metadata_dir: str = "artifacts/metadata") -> str:
    """Return the canonical metadata JSON path for a checkpoint basename."""
    stem = os.path.splitext(os.path.basename(os.path.abspath(checkpoint_path)))[0]
    return os.path.join(os.path.abspath(metadata_dir), f"{stem}.json")


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
