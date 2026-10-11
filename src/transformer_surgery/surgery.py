"""Model-agnostic surgery transform orchestration."""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from transformer_surgery.internal.pointwise import count_pointwise_per_input
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
from transformer_surgery.internal.routing import calibrate_support_routing, log_support_routing
from transformer_surgery.internal.util import get_device, get_surgery_dtype, save_model_checkpoint, set_seed
from transformer_surgery.models.adapters import SurgeryModelAdapter, get_model_adapter

if TYPE_CHECKING:
    from transformer_surgery.cli.surgery_config import SurgeryConfig


def _log_tail_prob_calibration(cal: Dict[str, Any], configured: float) -> None:
    def summary(values_key: str, rows_key: str) -> str:
        values = ", ".join(f"{float(v):.6g}" for v in cal[values_key])
        rows = cal.get(rows_key) or []
        rows_text = f"; rows/block min={min(rows)} max={max(rows)}" if rows else ""
        return f" over {cal['gibbs_tail_calibration_batches']} batch(es){rows_text}: [{values}]"

    if cal.get("gibbs_tail_prob_eps_exact_mean_by_block"):
        mode = "exact runtime" if cal.get("use_exact_tail_mass") else "calibrated"
        stds = ", ".join(f"{float(v):.6g}" for v in cal.get("gibbs_tail_prob_eps_exact_std_by_block", []))
        line = summary("gibbs_tail_prob_eps_exact_mean_by_block", "gibbs_tail_prob_eps_exact_rows_by_block")
        log_line(f"Exact gibbs tail mass by block ({mode}){line} std=[{stds}]")
        if cal.get("use_exact_tail_mass"):
            log_line("use_exact_tail_mass=True: runtime q_tail from centroid partition; calibrated value not applied")
            return
    if cal.get("gibbs_tail_prob_eps_calibrated_by_block"):
        line = summary("gibbs_tail_prob_eps_calibrated_by_block", "gibbs_tail_prob_eps_calibration_rows_by_block")
        log_line(f"Calibrated gibbs_tail_prob_eps by block{line}")
    elif cal.get("disable_calib_gibbs_tail_prob"):
        log_line(f"gibbs_tail_prob_eps calibration disabled; using configured value {configured:.6g}")


def _timed_evaluate(
    adapter: SurgeryModelAdapter, model: nn.Module, val_loader: Any, cfg: Any, device: torch.device
) -> Tuple[float, float, float, Optional[int], Dict[str, float]]:
    """``(primary, loss, wall seconds, peak CUDA memory or None, extra metrics)`` of one validation pass."""
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    primary, loss, extra = adapter.evaluate(model, val_loader, cfg)
    peak_memory = None
    if cuda:
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    return float(primary), float(loss), time.perf_counter() - t0, peak_memory, dict(extra)


def surgery(cfg: "SurgeryConfig") -> None:
    """Load reference, build the surgery student, calibrate, and save traceable artifacts."""
    adapter = get_model_adapter(cfg.model_key)
    device = get_device()
    dtype = get_surgery_dtype()
    if not adapter.strict_graph_options and (
        cfg.hadamard or (not cfg.allow_matmul and cfg.strict_product != "polarization")
    ):
        raise ValueError(f"hadamard and strict_product='norm' are implemented for DeiT-Tiny only, not {adapter.key}")

    log_line(f"Using device: {describe_device(device)}")
    log_line(f"Using surgery dtype: {describe_dtype(dtype)}")
    log_line(f"Using model adapter: {adapter.key}")
    log_line(f"config_json={cfg.config_json_path}")

    if cfg.calibration_seed is not None:
        set_seed(cfg.calibration_seed)
    log_line(f"Calibration seed: {cfg.calibration_seed}")
    train_loader, val_loader = adapter.build_loaders(cfg)

    reference_path = adapter.reference_checkpoint_path(cfg)
    log_line(f"Loading reference from {reference_path} ...")
    ref = adapter.load_reference_checkpoint(reference_path).to(device=device, dtype=dtype)

    example_input = adapter.example_model_input(cfg)
    before_log_path = traceable_log_path(cfg.log_dir, cfg, "ts-surgery", "model_before_surgery")
    write_model_structure_txt(
        before_log_path, ref, "Reference Model (before surgery transform)", example_input=example_input
    )
    log_wrote(before_log_path)

    ref_acc, ref_loss, ref_time, ref_peak_memory, ref_extra = _timed_evaluate(adapter, ref, val_loader, cfg, device)
    log_line(
        f"Reference model val primary={ref_acc:.4f} loss={ref_loss:.4f} ({adapter.primary_metric_name()}) "
        f"time={ref_time:.3f}s" + (f" {ref_extra}" if ref_extra else "")
    )

    cal = adapter.calibrate_reference(ref, train_loader, cfg)

    log_line(
        "Building surgery model | "
        f"disable_layernorm_replacement={cfg.disable_layernorm_replacement} "
        f"disable_attention_surgery={cfg.disable_attention_surgery} "
        f"disable_softmax_replacement={cfg.disable_softmax_replacement} "
        f"allow_matmul={cfg.allow_matmul} "
        f"use_exact_tail_mass={cfg.use_exact_tail_mass}"
    )
    model = adapter.build_surgery_model(cfg).to(device=device, dtype=dtype)
    mapping = adapter.copy_reference_weights(model, ref)
    adapter.freeze_surgery_parameters(model)

    cal.update(adapter.calibrate_after_build(model, train_loader, cfg))
    log_line("Calibration: " + json.dumps(cal, indent=2))
    _log_tail_prob_calibration(cal, cfg.gibbs_tail_prob_eps)

    applied = adapter.apply_calibration(model, cal)
    cal.update(applied)
    applied_mean = applied.get("gibbs_tail_prob_eps_applied_mean")
    if applied_mean is not None:
        log_line(f"Applied gibbs_tail_prob_eps mean={applied_mean:.6g}")

    routing_stats = calibrate_support_routing(model, train_loader, val_loader, cfg)
    log_support_routing(routing_stats)
    cal.update(routing_stats)
    if routing_stats.get("gibbs_tail_prob_eps_applied_mean") is not None:
        applied_mean = routing_stats["gibbs_tail_prob_eps_applied_mean"]
        log_line(f"Applied routed gibbs_tail_prob_eps mean={applied_mean:.6g}")

    after_log_path = traceable_log_path(cfg.log_dir, cfg, "ts-surgery", "model_after_surgery")
    write_model_structure_txt(
        after_log_path, model, "Surgery Model (after transform, pre-finetune checkpoint)", example_input=example_input
    )
    log_wrote(after_log_path)

    pre_acc, pre_loss, pre_time, pre_peak_memory, pre_extra = _timed_evaluate(adapter, model, val_loader, cfg, device)
    rel_slowdown = pre_time / ref_time if ref_time > 0 else 0.0
    pointwise = count_pointwise_per_input(model, next(iter(val_loader))[0])
    log_line(f"Pointwise (unary) evaluations per input: {pointwise['pointwise_per_input']:.6g}")
    log_line(
        f"Post-transform val primary={pre_acc:.4f} loss={pre_loss:.4f} ({adapter.primary_metric_name()}) "
        f"time={pre_time:.3f}s slowdown_vs_ref={rel_slowdown:.3f}x" + (f" {pre_extra}" if pre_extra else "")
    )

    cal.update(
        {
            "ref_val_acc": ref_acc,
            "ref_val_loss": ref_loss,
            "ref_val_primary_name": adapter.primary_metric_name(),
            "ref_val_wall_time_sec": ref_time,
            "student_pre_ft_val_acc": pre_acc,
            "student_pre_ft_val_loss": pre_loss,
            "student_pre_ft_val_wall_time_sec": pre_time,
            "student_pre_ft_val_relative_slowdown_vs_ref": rel_slowdown,
            "student_pointwise_per_input": pointwise["pointwise_per_input"],
            "student_pointwise_per_input_by_type": pointwise["pointwise_per_input_by_type"],
            **{f"ref_val_{k}": float(v) for k, v in ref_extra.items()},
            **{f"student_pre_ft_val_{k}": float(v) for k, v in pre_extra.items()},
        }
    )
    if ref_peak_memory is not None:
        cal["ref_val_peak_gpu_mem_bytes"] = ref_peak_memory
    if pre_peak_memory is not None:
        cal["student_pre_ft_val_peak_gpu_mem_bytes"] = pre_peak_memory

    pre_path = traceable_artifact_path(cfg.pre_ft_checkpoint, cfg, "ts-surgery", extension=".pt")
    cfg.pre_ft_checkpoint = pre_path
    meta_path = metadata_path_for_checkpoint(pre_path)
    # The applied mean replaces the initial tail mass in the run record; the state dict holds the per-block values.
    tail = {} if applied_mean is None else {"gibbs_tail_prob_eps": float(applied_mean)}
    mapping_by_module = adapter.build_module_mapping(cfg, model)
    meta = adapter.build_surgery_meta_dict(cfg, calibration=cal, module_mapping=mapping_by_module)
    write_json(meta_path, {**meta, **tail})
    os.makedirs(os.path.dirname(pre_path), exist_ok=True)
    extra = adapter.pre_ft_checkpoint_extra(cfg, mapping=mapping, metadata_path=meta_path)
    save_model_checkpoint(pre_path, model, extra={**extra, **tail})
    log_wrote(meta_path)
    log_wrote(pre_path)
    log_line(adapter.next_stage_hint())
