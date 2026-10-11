"""Text reports, structure dumps, traceable paths, and CLI-style logging."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from transformer_surgery.internal.util import to_model_input

CALIBRATION_LEGEND_TEXT = (
    "ref_val_acc / ref_val_loss: frozen reference on val (primary metric named by ref_val_primary_name, mean task "
    "loss). student_pre_ft_val_acc / student_pre_ft_val_loss: surgery student on val after transform, before distill. "
    "student_post_distill_*: after distillation (primary metric, mean task loss, mean teacher match). "
    "gibbs_tail_prob_eps_exact_*: observed dense-softmax omitted tail mass of the top-k support "
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


def traceable_artifact_path(path: str, cfg_or_path: Any, tool_name: str, extension: Optional[str] = None) -> str:
    """Return ``path``'s directory plus ``<tool>_<config><extension>`` (``path``'s own extension by default)."""
    original = os.path.abspath(path)
    ext = extension if extension is not None else os.path.splitext(original)[1]
    return os.path.join(os.path.dirname(original), f"{config_artifact_stem(cfg_or_path, tool_name)}{ext}")


def traceable_log_path(log_dir: str, cfg_or_path: Any, tool_name: str, log_name: str) -> str:
    return os.path.join(
        os.path.abspath(log_dir),
        f"{config_artifact_stem(cfg_or_path, tool_name)}_{_slug_part(log_name)}.txt",
    )


def metadata_path_for_checkpoint(checkpoint_path: str, metadata_dir: str = "artifacts/metadata") -> str:
    """Return the canonical metadata JSON path for a checkpoint basename."""
    stem = os.path.splitext(os.path.basename(os.path.abspath(checkpoint_path)))[0]
    return os.path.join(os.path.abspath(metadata_dir), f"{stem}.json")


def _shape_str(out: Any) -> str:
    if torch.is_tensor(out):
        return str(tuple(out.shape))
    if isinstance(out, (tuple, list)):
        return "(" + ", ".join(_shape_str(x) for x in out) + ")"
    return type(out).__name__


def write_model_structure_txt(
    path: str, model: nn.Module, title: str, *, example_input: Optional[torch.Tensor] = None
) -> None:
    """Parameter counts, the module tree, and every module's output shape on one eval batch (image dummy by default)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    names = [name or "<root>" for name, _ in model.named_modules()]
    lines: List[str] = [
        title,
        "=" * min(80, max(len(title), 40)),
        f"class: {type(model).__name__}",
        f"parameters: total={sum(p.numel() for p in model.parameters()):,} "
        f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
        "",
        str(model),
        "",
        "--- named_modules (name: class) ---",
        "",
    ]
    lines += [f"{name}: {type(mod).__name__}" for name, mod in zip(names, model.modules())]
    lines += ["", "--- forward output shapes (one eval batch; dummy input unless example_input set) ---"]

    out_shapes: Dict[str, str] = {}
    handles = [
        mod.register_forward_hook(lambda _m, _i, out, key=name: out_shapes.__setitem__(key, _shape_str(out)))
        for name, mod in zip(names, model.modules())
    ]
    was_training = model.training
    try:
        x = to_model_input(torch.zeros(1, 3, 224, 224) if example_input is None else example_input, model)
        model.eval()
        with torch.no_grad():
            model(x)
        lines += [f"example_input: {tuple(x.shape)}  dtype={x.dtype}  device={x.device}", ""]
        lines += [f"{name}: {out_shapes.get(name, '-')}" for name in names]
    except Exception as ex:
        lines.append(f"(forward shape trace failed: {ex!r})")
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
