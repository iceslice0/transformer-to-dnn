"""Model-agnostic surgery transform orchestration."""

from __future__ import annotations

import os

import time
from typing import TYPE_CHECKING, Any, Dict

import torch
import torch.nn as nn

from transformer_surgery.models.adapters import get_model_adapter
from transformer_surgery.internal.reporting import (
    describe_device,
    describe_dtype,
    log_json_block,
    log_line,
    log_wrote,
    metadata_path_for_checkpoint,
    traceable_artifact_path,
    traceable_log_path,
    write_json,
    write_model_structure_txt,
)
from transformer_surgery.internal.util import (
    get_device,
    get_surgery_dtype,
    namespace_from_mapping,
    namespace_to_mapping,
    save_model_checkpoint,
)

if TYPE_CHECKING:
    from transformer_surgery.cli.surgery_config import SurgeryConfig


def _log_tail_prob_calibration(cal: Dict[str, Any], configured: float) -> None:
    exact_by_block = cal.get("gibbs_tail_prob_eps_exact_mean_by_block")
    if isinstance(exact_by_block, list) and exact_by_block:
        values = ", ".join(f"{float(v):.6g}" for v in exact_by_block)
        batches = cal.get("gibbs_tail_calibration_batches")
        rows = cal.get("gibbs_tail_prob_eps_exact_rows_by_block")
        suffix = f" over {int(batches)} batch(es)" if batches is not None else ""
        if isinstance(rows, list) and rows:
            suffix += f"; rows/block min={min(int(r) for r in rows)} max={max(int(r) for r in rows)}"
        stds = cal.get("gibbs_tail_prob_eps_exact_std_by_block")
        std_suffix = ""
        if isinstance(stds, list) and stds:
            std_vals = ", ".join(f"{float(v):.6g}" for v in stds)
            std_suffix = f" std=[{std_vals}]"
        mode = "exact runtime" if cal.get("use_exact_tail_mass") else "calibrated"
        log_line(f"Exact gibbs tail mass by block ({mode}){suffix}: mean=[{values}]{std_suffix}")
        if cal.get("use_exact_tail_mass"):
            log_line(
                "use_exact_tail_mass=True: runtime q_tail from centroid partition; "
                "calibrated gibbs_tail_prob_eps not applied"
            )
            return
    by_block = cal.get("gibbs_tail_prob_eps_calibrated_by_block")
    if isinstance(by_block, list) and by_block:
        values = ", ".join(f"{float(v):.6g}" for v in by_block)
        batches = cal.get("gibbs_tail_calibration_batches")
        rows = cal.get("gibbs_tail_prob_eps_calibration_rows_by_block")
        suffix = f" over {int(batches)} batch(es)" if batches is not None else ""
        if isinstance(rows, list) and rows:
            suffix += f"; rows/block min={min(int(r) for r in rows)} max={max(int(r) for r in rows)}"
        log_line(f"Calibrated gibbs_tail_prob_eps by block{suffix}: [{values}]")
    elif cal.get("disable_calib_gibbs_tail_prob"):
        log_line(f"gibbs_tail_prob_eps calibration disabled; using configured value {configured:.6g}")


def _timed_evaluate(
    adapter: Any, model: nn.Module, val_loader: Any, cfg: Any, device: Any
) -> tuple[float, float, float, int | None, dict]:
    peak_memory_bytes: int | None = None
    if getattr(device, "type", None) == "cuda":
        torch_device = device
        torch.cuda.synchronize(torch_device)
        torch.cuda.reset_peak_memory_stats(torch_device)
    t0 = time.perf_counter()
    primary, loss, extra = adapter.evaluate(model, val_loader, cfg)
    if getattr(device, "type", None) == "cuda":
        torch_device = device
        torch.cuda.synchronize(torch_device)
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(torch_device))
    elapsed_s = time.perf_counter() - t0
    return float(primary), float(loss), float(elapsed_s), peak_memory_bytes, dict(extra)


def surgery(cfg: "SurgeryConfig") -> None:
    """Load reference, build the surgery student, calibrate, and save traceable artifacts."""
    adapter = get_model_adapter(cfg.model_key)
    device = get_device()
    dtype = get_surgery_dtype()

    log_line(f"Using device: {describe_device(device)}")
    log_line(f"Using surgery dtype: {describe_dtype(dtype)}")
    log_line(f"Using model adapter: {adapter.key}")
    if cfg.config_json_path:
        log_line(f"config_json={cfg.config_json_path}")

    train_loader, val_loader = adapter.build_loaders(cfg)

    reference_path = adapter.reference_checkpoint_path(cfg)
    log_line(f"Loading reference from {reference_path} ...")
    ref = adapter.load_reference_checkpoint(reference_path).to(device=device, dtype=dtype)

    before_log_path = traceable_log_path(cfg.log_dir, cfg, "ts-surgery", "model_before_surgery")
    example_input = adapter.example_model_input(cfg)
    write_model_structure_txt(
        before_log_path,
        ref,
        "Reference Model (before surgery transform)",
        example_input=example_input,
    )
    log_wrote(before_log_path)

    ref_acc, ref_loss, ref_val_time_s, ref_peak_memory, ref_extra = _timed_evaluate(
        adapter, ref, val_loader, cfg, device
    )
    log_line(
        f"Reference model val primary={ref_acc:.4f} loss={ref_loss:.4f} "
        f"({adapter.primary_metric_name()}) "
        f"time={ref_val_time_s:.3f}s"
        + (f" {ref_extra}" if ref_extra else "")
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

    after_build_cal = adapter.calibrate_after_build(model, train_loader, cfg)
    if after_build_cal:
        cal.update(after_build_cal)
    log_json_block("Calibration:", cal)
    _log_tail_prob_calibration(cal, float(cfg.gibbs_tail_prob_eps))

    applied_cal = adapter.apply_calibration(model, cal)
    applied_mean = None
    if applied_cal:
        cal.update(applied_cal)
        applied_mean = applied_cal.get("gibbs_tail_prob_eps_applied_mean")
        if applied_mean is not None:
            log_line(f"Applied gibbs_tail_prob_eps mean={float(applied_mean):.6g}")

    after_log_path = traceable_log_path(cfg.log_dir, cfg, "ts-surgery", "model_after_surgery")
    write_model_structure_txt(
        after_log_path,
        model,
        "Surgery Model (after transform, pre-finetune checkpoint)",
        example_input=example_input,
    )
    log_wrote(after_log_path)

    pre_acc, pre_loss, pre_val_time_s, pre_peak_memory, pre_extra = _timed_evaluate(
        adapter, model, val_loader, cfg, device
    )
    rel_slowdown = pre_val_time_s / ref_val_time_s if ref_val_time_s > 0 else 0.0
    log_line(
        f"Post-transform val primary={pre_acc:.4f} loss={pre_loss:.4f} "
        f"({adapter.primary_metric_name()}) "
        f"time={pre_val_time_s:.3f}s slowdown_vs_ref={rel_slowdown:.3f}x"
        + (f" {pre_extra}" if pre_extra else "")
    )

    pre_path = traceable_artifact_path(cfg.pre_ft_checkpoint, cfg, "ts-surgery", extension=".pt")
    cfg.pre_ft_checkpoint = pre_path
    meta_path = metadata_path_for_checkpoint(pre_path)

    meta = namespace_from_mapping(
        adapter.build_surgery_meta_dict(
            cfg,
            calibration=cal,
            reference_checkpoint_abs=reference_path,
            module_mapping=adapter.build_module_mapping(cfg, model),
        )
    )
    if applied_mean is not None:
        meta.gibbs_tail_prob_eps = float(applied_mean)
    meta.calibration.ref_val_acc = float(ref_acc)
    meta.calibration.ref_val_loss = float(ref_loss)
    meta.calibration.ref_val_primary = float(ref_acc)
    meta.calibration.ref_val_primary_name = adapter.primary_metric_name()
    meta.calibration.ref_val_wall_time_sec = float(ref_val_time_s)
    meta.calibration.student_pre_ft_val_acc = float(pre_acc)
    meta.calibration.student_pre_ft_mean_ce = float(pre_loss)
    meta.calibration.student_pre_ft_val_primary = float(pre_acc)
    meta.calibration.student_pre_ft_val_loss = float(pre_loss)
    meta.calibration.student_pre_ft_val_wall_time_sec = float(pre_val_time_s)
    meta.calibration.student_pre_ft_val_relative_slowdown_vs_ref = float(rel_slowdown)
    if ref_peak_memory is not None:
        meta.calibration.ref_val_peak_gpu_mem_bytes = int(ref_peak_memory)
    if pre_peak_memory is not None:
        meta.calibration.student_pre_ft_val_peak_gpu_mem_bytes = int(pre_peak_memory)
    for key, value in ref_extra.items():
        setattr(meta.calibration, f"ref_val_{key}", float(value))
    for key, value in pre_extra.items():
        setattr(meta.calibration, f"student_pre_ft_val_{key}", float(value))

    write_json(meta_path, namespace_to_mapping(meta))
    os.makedirs(os.path.dirname(pre_path) or ".", exist_ok=True)

    ck = namespace_from_mapping(adapter.pre_ft_checkpoint_extra(cfg, mapping=mapping, metadata_path=meta_path))
    if applied_mean is not None:
        ck.gibbs_tail_prob_eps = float(applied_mean)
    save_model_checkpoint(pre_path, model, extra=namespace_to_mapping(ck))

    log_wrote(meta_path)
    log_wrote(pre_path)
    log_line(adapter.next_stage_hint())
