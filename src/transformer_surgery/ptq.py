#!/usr/bin/env python3
"""
Procedural PTQ runner for surgery checkpoints.

Flow: pick nodes -> register hooks -> gather range moments -> remove hooks -> gather dequant
moments -> remove hooks -> install wrappers -> save and validate.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from transformer_surgery.models.adapters import (
    apply_load_cfg_overrides,
    get_model_adapter,
    load_surgery_student_checkpoint,
    surgery_dtype_from_extra,
)
from transformer_surgery.ops import (
    AffineContract,
    AffineFixedMix,
    AffineHadamard,
    AffineMatMul,
    AffineScale,
    AffineScaleBias,
)
from transformer_surgery.internal.calibration import (
    PTQNodeSetup,
    PTQNodeStats,
    build_ptq_node_setup,
    build_ptq_wrapper,
    calibration_input_dtype,
    gather_ptq_dequant_moments,
    gather_ptq_range_moments,
    ptq_activation_bits_for_kind,
    ptq_module_kind,
    ptq_wrapper_from_reload_config,
)
from transformer_surgery.internal.reporting import (
    describe_device,
    describe_dtype,
    log_line,
    log_wrote,
    metadata_path_for_checkpoint,
    traceable_artifact_path,
    traceable_log_path,
    write_json,
    write_model_structure_txt,
)
from transformer_surgery.internal.util import (
    ensure_mapping,
    get_device,
    get_surgery_dtype,
    maybe_surgery_cuda_autocast,
    namespace_from_mapping,
    namespace_to_mapping,
    save_model_checkpoint,
    set_surgery_dtype,
)


SUPPORTED_LINEAR_CONV_TYPES = (nn.Linear, nn.Conv2d)
SUPPORTED_AFFINE_TYPES = (AffineScale, AffineScaleBias, AffineFixedMix, AffineContract)
SUPPORTED_MATMUL_TYPES = (AffineMatMul, AffineHadamard)


def _name_matches(name: str, patterns: Sequence[str]) -> bool:
    return any((needle := pattern.strip()) and needle in name for pattern in patterns)


def _module_type_selected(module: nn.Module, cfg: Any) -> bool:
    return (
        (cfg.wrap_linear_conv and isinstance(module, SUPPORTED_LINEAR_CONV_TYPES))
        or (cfg.wrap_affine and isinstance(module, SUPPORTED_AFFINE_TYPES))
        or (cfg.wrap_matmul and isinstance(module, SUPPORTED_MATMUL_TYPES))
    )


def _set_module(root: nn.Module, name: str, new_module: nn.Module) -> None:
    parent_name, _, leaf = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, leaf, new_module)


def _build_node_selection(model: nn.Module, cfg: Any) -> Dict[str, str]:
    selected: Dict[str, str] = {}
    for name, module in model.named_modules():
        if not name:
            continue
        kind = ptq_module_kind(module)
        if kind is None or _name_matches(name, cfg.exclude_names):
            continue
        if _module_type_selected(module, cfg) or _name_matches(name, cfg.include_names):
            selected[name] = kind
    return selected


def _filter_unobserved_nodes(
    selected: Dict[str, str],
    stats: Dict[str, PTQNodeStats],
) -> Dict[str, str]:
    """
    Drop selected nodes that never emitted calibration outputs.

    Some modules are conditionally inactive on a given model/config path (for example,
    Gibbs tail-scaling when ``top_k == seq_len``). Those modules should be skipped
    rather than failing the full PTQ run.
    """
    active: Dict[str, str] = {}
    skipped: List[str] = []
    for name, kind in selected.items():
        if stats[name].examples_total > 0:
            active[name] = kind
        else:
            skipped.append(name)
    if skipped:
        log_line(
            "Skipping PTQ node(s) with no calibration observations: "
            + ", ".join(skipped)
        )
    return active


def _build_ptq_setup(name: str, module: nn.Module, stats: PTQNodeStats, cfg: Any) -> PTQNodeSetup:
    return build_ptq_node_setup(
        name,
        module,
        stats,
        weight_bits=int(cfg.weight_bits),
        activation_bits=int(cfg.activation_bits),
        affine_activation_bits=cfg.affine_activation_bits,
        matmul_activation_bits=cfg.matmul_activation_bits,
        per_output_channel=bool(cfg.per_output_channel),
    )


@torch.no_grad()
def validate_model(model: nn.Module, loader, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    device = get_device()
    use_cuda = device.type == "cuda"
    loss_sum_t = torch.zeros((), device=device, dtype=torch.float64)
    correct_t = torch.zeros((), device=device, dtype=torch.long)
    n = 0
    input_dtype = calibration_input_dtype(model)
    dt_eval = get_surgery_dtype()

    for x, y in loader:
        x = x.to(device, dtype=input_dtype, non_blocking=use_cuda)
        y = y.to(device, non_blocking=use_cuda)
        with maybe_surgery_cuda_autocast(device, dt_eval):
            logits = model(x)
        loss_sum_t += criterion(logits.float(), y).double() * y.size(0)
        correct_t += (logits.argmax(dim=-1) == y).sum()
        n += y.size(0)
    denom = max(n, 1)
    return correct_t.item() / denom, loss_sum_t.item() / denom


def _build_ptq_model(
    fp_model: nn.Module,
    selected: Dict[str, str],
    calibration_stats: Dict[str, PTQNodeStats],
    setups: Dict[str, PTQNodeSetup],
    cfg: Any,
) -> Tuple[nn.Module, List[Dict[str, Any]], List[Dict[str, Any]]]:
    out = copy.deepcopy(fp_model)
    node_meta: List[Dict[str, Any]] = []
    reload_configs: List[Dict[str, Any]] = []
    for name in selected:
        source_module = out.get_submodule(name)
        built = build_ptq_wrapper(
            source_module,
            setups[name],
            calibration_stats[name],
            dequant_var_eps=float(cfg.dequant_var_eps),
        )
        _set_module(out, name, built.module)
        meta = built.metadata
        node_meta.append(meta)
        reload_configs.append(built.reload_config)
    return out, node_meta, reload_configs


def load_ptq_checkpoint(path: str, cfg: Any) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a PTQ checkpoint: rebuild the surgery template, install skeleton PTQ modules, then load_state_dict."""
    device = get_device()
    payload = torch.load(path, map_location=device, weights_only=False)
    extra_ns = namespace_from_mapping(dict(payload["extra"]))
    adapter = get_model_adapter(extra_ns.model_key)
    apply_load_cfg_overrides(extra_ns, cfg)
    set_surgery_dtype(surgery_dtype_from_extra(extra_ns))
    extra_ns.surgery_dtype = describe_dtype(get_surgery_dtype())
    model = adapter.build_surgery_model_from_extra(ensure_mapping(extra_ns), cfg).to(
        device=device,
        dtype=get_surgery_dtype(),
    )
    for wrapper_cfg in extra_ns.ptq_wrappers:
        wrapper_dict = namespace_to_mapping(wrapper_cfg)
        source_module = model.get_submodule(wrapper_dict["name"])
        _set_module(
            model,
            wrapper_dict["name"],
            ptq_wrapper_from_reload_config(wrapper_dict, source_module).to(device=device),
        )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, ensure_mapping(extra_ns)


def _ptq_summary(
    cfg: Any,
    selected: Dict[str, str],
    fp_acc: float,
    fp_loss: float,
    ptq_acc: float,
    ptq_loss: float,
    node_meta: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "source_checkpoint": os.path.abspath(cfg.fp_checkpoint),
        "output_checkpoint": os.path.abspath(cfg.output),
        "selection": {
            "wrap_linear_conv": bool(cfg.wrap_linear_conv),
            "wrap_affine": bool(cfg.wrap_affine),
            "wrap_matmul": bool(cfg.wrap_matmul),
            "include_names": list(cfg.include_names),
            "exclude_names": list(cfg.exclude_names),
            "selected_nodes": list(selected.keys()),
        },
        "quantization": {
            "weight_bits": int(cfg.weight_bits),
            "activation_bits": int(
                ptq_activation_bits_for_kind(
                    "linear",
                    cfg.activation_bits,
                    affine_activation_bits=cfg.affine_activation_bits,
                    matmul_activation_bits=cfg.matmul_activation_bits,
                )
            ),
            "affine_activation_bits": int(
                ptq_activation_bits_for_kind(
                    "affine_scale",
                    cfg.activation_bits,
                    affine_activation_bits=cfg.affine_activation_bits,
                    matmul_activation_bits=cfg.matmul_activation_bits,
                )
            ),
            "matmul_activation_bits": int(
                ptq_activation_bits_for_kind(
                    "matmul",
                    cfg.activation_bits,
                    affine_activation_bits=cfg.affine_activation_bits,
                    matmul_activation_bits=cfg.matmul_activation_bits,
                )
            ),
            "per_output_channel": bool(cfg.per_output_channel),
            "dequant_var_eps": float(cfg.dequant_var_eps),
        },
        "calibration": {
            "loader_split": "train",
            "mode": "full_train_loader_aggregated",
        },
        "metrics": {
            "fp_val_acc": float(fp_acc),
            "fp_val_loss": float(fp_loss),
            "ptq_val_acc": float(ptq_acc),
            "ptq_val_loss": float(ptq_loss),
            "acc_delta": float(ptq_acc - fp_acc),
            "loss_delta": float(ptq_loss - fp_loss),
        },
        "ptq_nodes": node_meta,
    }


def run_ptq(
    cfg: Any,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    device = get_device() if device is None else device
    out_abs = traceable_artifact_path(cfg.output, cfg, "ts-ptq", "", ".pt")
    meta_abs = metadata_path_for_checkpoint(out_abs)
    model_log_abs = traceable_log_path(cfg.log_dir, cfg, "ts-ptq", "model_after_ptq")
    cfg.output = out_abs

    log_line(f"Using device: {describe_device(device)}")
    if cfg.config_json_path:
        log_line(f"config_json={cfg.config_json_path}")

    criterion = nn.CrossEntropyLoss()
    fp_model, fp_extra = load_surgery_student_checkpoint(os.path.abspath(cfg.fp_checkpoint), cfg, surgery_dtype=dtype)
    log_line(f"Surgery dtype (from checkpoint): {describe_dtype(get_surgery_dtype())}")
    adapter = get_model_adapter(fp_extra["model_key"])
    train_loader, val_loader = adapter.build_loaders(cfg)
    log_line(f"Using model adapter: {adapter.key}")

    calibration_batches = list(range(len(train_loader)))
    if not calibration_batches:
        raise SystemExit("No training batches available for PTQ calibration.")
    log_line(f"PTQ calibration batches: using full train loader ({len(calibration_batches)} batch(es), aggregated stats).")

    selected = _build_node_selection(fp_model, cfg)
    if not selected:
        raise SystemExit("No PTQ-wrappable nodes selected by the current config.")
    log_line(f"Selected {len(selected)} PTQ node(s).")
    for name, kind in selected.items():
        log_line(f"  {name}: {kind}")

    fp_acc, fp_loss = validate_model(fp_model, val_loader, criterion)
    log_line(f"Float model val acc={fp_acc:.4f} loss={fp_loss:.4f}")

    stats = gather_ptq_range_moments(fp_model, train_loader, selected, calibration_batches)
    selected = _filter_unobserved_nodes(selected, stats)
    if not selected:
        raise SystemExit("No PTQ node received calibration statistics on the current run.")

    setups = {name: _build_ptq_setup(name, fp_model.get_submodule(name), stats[name], cfg) for name in selected}
    gather_ptq_dequant_moments(fp_model, train_loader, selected, stats, setups, calibration_batches)
    missing_dequant = [
        name for name in selected if stats[name].bias_fit.count <= 0 and stats[name].affine_fit.count <= 0
    ]
    if missing_dequant:
        raise SystemExit(f"Calibration dequant statistics missing for selected node(s): {missing_dequant}")

    ptq_model, node_meta, reload_configs = _build_ptq_model(fp_model, selected, stats, setups, cfg)

    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    save_model_checkpoint(
        out_abs,
        ptq_model,
        extra={**fp_extra, "ptq_meta_path": os.path.basename(meta_abs), "ptq_wrappers": reload_configs},
    )
    write_model_structure_txt(model_log_abs, ptq_model, "PTQ Surgery Model")
    log_wrote(model_log_abs)
    log_wrote(out_abs)

    del ptq_model
    reloaded_model, _ = load_ptq_checkpoint(out_abs, cfg)
    log_line("reloaded PTQ checkpoint from disk for validation")
    ptq_acc, ptq_loss = validate_model(reloaded_model, val_loader, criterion)
    log_line(f"PTQ model val acc={ptq_acc:.4f} loss={ptq_loss:.4f}")
    log_line(f"Delta acc={ptq_acc - fp_acc:+.4f} loss={ptq_loss - fp_loss:+.4f}")

    write_json(
        meta_abs,
        _ptq_summary(
            cfg,
            selected,
            fp_acc,
            fp_loss,
            ptq_acc,
            ptq_loss,
            node_meta,
        ),
    )
    log_wrote(meta_abs)
