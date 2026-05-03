"""Model-agnostic surgery transform orchestration."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict

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
from transformer_surgery.internal.runtime import get_surgery_dtype
from transformer_surgery.internal.util import (
    accuracy_and_loss,
    get_device,
    namespace_from_mapping,
    namespace_to_mapping,
    save_model_checkpoint,
)

if TYPE_CHECKING:
    from transformer_surgery.cli.surgery_config import SurgeryConfig


def _log_tail_prob_calibration(cal: Dict[str, Any], configured: float) -> None:
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

    _, val_loader = adapter.build_loaders(cfg)

    reference_path = adapter.reference_checkpoint_path(cfg)
    log_line(f"Loading reference from {reference_path} ...")
    ref = adapter.load_reference_checkpoint(reference_path).to(device=device, dtype=dtype)

    before_log_path = traceable_log_path(cfg.log_dir, cfg, "ts-surgery", "model_before_surgery")
    write_model_structure_txt(before_log_path, ref, "Reference Model (before surgery transform)")
    log_wrote(before_log_path)

    criterion = nn.CrossEntropyLoss()
    ref_acc, ref_loss = accuracy_and_loss(ref, val_loader, criterion)
    log_line(f"Reference model val acc={ref_acc:.4f} loss={ref_loss:.4f}")

    cal = adapter.calibrate_reference(ref, val_loader, cfg)
    log_json_block("Calibration:", cal)
    _log_tail_prob_calibration(cal, float(cfg.gibbs_tail_prob_eps))

    log_line(
        "Building surgery model | "
        f"disable_layernorm_replacement={cfg.disable_layernorm_replacement} "
        f"disable_attention_surgery={cfg.disable_attention_surgery} "
        f"disable_softmax_replacement={cfg.disable_softmax_replacement} "
        f"allow_matmul={cfg.allow_matmul}"
    )
    model = adapter.build_surgery_model(cfg).to(device=device, dtype=dtype)
    mapping = adapter.copy_reference_weights(model, ref)
    adapter.freeze_surgery_parameters(model)

    applied_cal = adapter.apply_calibration(model, cal)
    applied_mean = None
    if applied_cal:
        cal.update(applied_cal)
        applied_mean = applied_cal.get("gibbs_tail_prob_eps_applied_mean")
        if applied_mean is not None:
            log_line(f"Applied gibbs_tail_prob_eps mean={float(applied_mean):.6g}")

    after_log_path = traceable_log_path(cfg.log_dir, cfg, "ts-surgery", "model_after_surgery")
    write_model_structure_txt(after_log_path, model, "Surgery Model (after transform, pre-finetune checkpoint)")
    log_wrote(after_log_path)

    pre_acc, pre_loss = accuracy_and_loss(model, val_loader, criterion)
    log_line(f"Post-transform val acc={pre_acc:.4f} loss={pre_loss:.4f}")

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
    meta.calibration.student_pre_ft_val_acc = float(pre_acc)
    meta.calibration.student_pre_ft_mean_ce = float(pre_loss)

    write_json(meta_path, namespace_to_mapping(meta))
    os.makedirs(os.path.dirname(pre_path) or ".", exist_ok=True)

    ck = namespace_from_mapping(adapter.pre_ft_checkpoint_extra(cfg, mapping=mapping, metadata_path=meta_path))
    if applied_mean is not None:
        ck.gibbs_tail_prob_eps = float(applied_mean)
    save_model_checkpoint(pre_path, model, extra=namespace_to_mapping(ck))

    log_wrote(meta_path)
    log_wrote(pre_path)
    log_line("Next: run the distill stage.")
