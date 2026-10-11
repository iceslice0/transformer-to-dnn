"""Model-agnostic surgery transform orchestration."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import torch.nn as nn

from transformer_surgery.internal.calibration import TailCalibration, apply_gibbs_tail_calibration
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


@dataclass
class ValidationRun:
    """One timed validation pass: the adapter's primary metric and loss, wall time, peak CUDA memory, extra metrics."""

    primary: float
    loss: float
    wall_time_sec: float
    peak_gpu_mem_bytes: Optional[int]
    extra: Dict[str, float]

    def metadata(self, prefix: str) -> Dict[str, float]:
        """Metadata keys ``{prefix}acc`` (the primary metric), ``{prefix}loss``, ... of this pass."""
        keys: Dict[str, float] = {f"{prefix}acc": self.primary, f"{prefix}loss": self.loss}
        keys[f"{prefix}wall_time_sec"] = self.wall_time_sec
        if self.peak_gpu_mem_bytes is not None:
            keys[f"{prefix}peak_gpu_mem_bytes"] = self.peak_gpu_mem_bytes
        keys.update({f"{prefix}{k}": float(v) for k, v in self.extra.items()})
        return keys


def _log_tail_prob_calibration(tail: TailCalibration) -> None:
    def summary(values: List[float], rows: Optional[List[int]]) -> str:
        rows_text = f"; rows/block min={min(rows)} max={max(rows)}" if rows else ""
        listed = ", ".join(f"{v:.6g}" for v in values)
        return f" over {tail.gibbs_tail_calibration_batches} batch(es){rows_text}: [{listed}]"

    if tail.gibbs_tail_prob_eps_exact_mean_by_block:
        mode = "exact runtime" if tail.use_exact_tail_mass else "calibrated"
        stds = ", ".join(f"{v:.6g}" for v in tail.gibbs_tail_prob_eps_exact_std_by_block or [])
        line = summary(tail.gibbs_tail_prob_eps_exact_mean_by_block, tail.gibbs_tail_prob_eps_exact_rows_by_block)
        log_line(f"Exact gibbs tail mass by block ({mode}){line} std=[{stds}]")
        if tail.use_exact_tail_mass:
            log_line("use_exact_tail_mass=True: runtime q_tail from centroid partition; calibrated value not applied")
            return
    if tail.gibbs_tail_prob_eps_calibrated_by_block:
        line = summary(tail.gibbs_tail_prob_eps_calibrated_by_block, tail.gibbs_tail_prob_eps_calibration_rows_by_block)
        log_line(f"Calibrated gibbs_tail_prob_eps by block{line}")
    elif tail.disable_calib_gibbs_tail_prob:
        configured = tail.gibbs_tail_prob_eps_configured
        log_line(f"gibbs_tail_prob_eps calibration disabled; using configured value {configured:.6g}")


def _timed_evaluate(
    adapter: SurgeryModelAdapter, model: nn.Module, val_loader: Any, cfg: Any, device: torch.device
) -> ValidationRun:
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
    return ValidationRun(float(primary), float(loss), time.perf_counter() - t0, peak_memory, dict(extra))


def _record(record: Any) -> Dict[str, Any]:
    """Metadata keys of an optional dataclass record."""
    return {} if record is None else asdict(record)


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

    ref_run = _timed_evaluate(adapter, ref, val_loader, cfg, device)
    log_line(
        f"Reference model val primary={ref_run.primary:.4f} loss={ref_run.loss:.4f} ({adapter.primary_metric_name()}) "
        f"time={ref_run.wall_time_sec:.3f}s" + (f" {ref_run.extra}" if ref_run.extra else "")
    )

    tail = adapter.calibrate_reference(ref, train_loader, cfg)

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

    tail = adapter.calibrate_after_build(model, train_loader, cfg) or tail
    if tail is not None:
        log_line("Calibration: " + json.dumps(asdict(tail), indent=2))
        _log_tail_prob_calibration(tail)
    applied = None if tail is None else adapter.apply_calibration(model, tail)
    if applied is not None:
        log_line(f"Applied gibbs_tail_prob_eps mean={applied.gibbs_tail_prob_eps_applied_mean:.6g}")

    routing = calibrate_support_routing(model, train_loader, val_loader, cfg)
    if routing is not None:
        log_support_routing(routing)
        if not cfg.use_exact_tail_mass and not cfg.disable_calib_gibbs_tail_prob:
            # The routed support's mean tail mass replaces the top-k one.
            applied = apply_gibbs_tail_calibration(model, routing.routing_train_tail_routed_by_block)
            if applied is not None:
                values = ", ".join(f"{v:.6g}" for v in applied.gibbs_tail_prob_eps_applied_by_block)
                log_line(f"Applied routed gibbs_tail_prob_eps by block: [{values}]")

    after_log_path = traceable_log_path(cfg.log_dir, cfg, "ts-surgery", "model_after_surgery")
    write_model_structure_txt(
        after_log_path, model, "Surgery Model (after transform, pre-finetune checkpoint)", example_input=example_input
    )
    log_wrote(after_log_path)

    pre_run = _timed_evaluate(adapter, model, val_loader, cfg, device)
    rel_slowdown = pre_run.wall_time_sec / ref_run.wall_time_sec if ref_run.wall_time_sec > 0 else 0.0
    pointwise = count_pointwise_per_input(model, next(iter(val_loader))[0])
    log_line(f"Pointwise (unary) evaluations per input: {pointwise.per_input:.6g}")
    log_line(
        f"Post-transform val primary={pre_run.primary:.4f} loss={pre_run.loss:.4f} ({adapter.primary_metric_name()}) "
        f"time={pre_run.wall_time_sec:.3f}s slowdown_vs_ref={rel_slowdown:.3f}x"
        + (f" {pre_run.extra}" if pre_run.extra else "")
    )

    calibration = {
        **_record(tail),
        **_record(applied),
        **_record(routing),
        **ref_run.metadata("ref_val_"),
        "ref_val_primary_name": adapter.primary_metric_name(),
        **pre_run.metadata("student_pre_ft_val_"),
        "student_pre_ft_val_relative_slowdown_vs_ref": rel_slowdown,
        "student_pointwise_per_input": pointwise.per_input,
        "student_pointwise_per_input_by_type": pointwise.by_type,
    }

    pre_path = traceable_artifact_path(cfg.pre_ft_checkpoint, cfg, "ts-surgery", extension=".pt")
    # The run record carries the applied mean tail mass; the state dict holds the per-block values.
    record = replace(cfg, pre_ft_checkpoint=pre_path)
    if applied is not None:
        record = replace(record, gibbs_tail_prob_eps=applied.gibbs_tail_prob_eps_applied_mean)
    meta_path = metadata_path_for_checkpoint(pre_path)
    meta = adapter.build_surgery_meta_dict(
        record, calibration=calibration, module_mapping=adapter.build_module_mapping(record, model)
    )
    write_json(meta_path, meta)
    os.makedirs(os.path.dirname(pre_path), exist_ok=True)
    extra = adapter.pre_ft_checkpoint_extra(record, mapping=mapping, metadata_path=meta_path)
    save_model_checkpoint(pre_path, model, extra=extra)
    log_wrote(meta_path)
    log_wrote(pre_path)
    log_line(adapter.next_stage_hint())
