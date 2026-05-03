"""Model-agnostic surgery transform orchestration."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Dict

import torch.nn as nn

from transformer_surgery.models.adapters import get_model_adapter
from transformer_surgery.ops import get_surgery_dtype, write_model_structure_txt
from transformer_surgery.util import (
    accuracy_and_loss,
    describe_device,
    describe_dtype,
    get_device,
    metadata_path_for_checkpoint,
    save_model_checkpoint,
    traceable_artifact_path,
    traceable_log_path,
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
        print(f"Calibrated gibbs_tail_prob_eps by block{suffix}: [{values}]", flush=True)
    elif cal.get("disable_calib_gibbs_tail_prob"):
        print(
            f"gibbs_tail_prob_eps calibration disabled; using configured value {configured:.6g}",
            flush=True,
        )


def surgery(cfg: "SurgeryConfig") -> None:
    """Load reference, build the surgery student, calibrate, and save traceable artifacts."""
    adapter = get_model_adapter(cfg)
    device = get_device()
    dtype = get_surgery_dtype()

    print(f"Using device: {describe_device(device)}", flush=True)
    print(f"Using surgery dtype: {describe_dtype(dtype)}", flush=True)
    print(f"Using model adapter: {adapter.key}", flush=True)
    if cfg.config_json_path:
        print(f"config_json={cfg.config_json_path}", flush=True)

    _, val_loader = adapter.build_loaders(cfg)

    reference_path = adapter.reference_checkpoint_path(cfg)
    print(f"Loading reference from {reference_path} ...", flush=True)
    ref = adapter.load_reference_checkpoint(reference_path).to(device=device, dtype=dtype)

    before_log_path = traceable_log_path(cfg.log_dir, cfg, "ts-surgery", "model_before_surgery")
    write_model_structure_txt(before_log_path, ref, "Reference Model (before surgery transform)")
    print(f"wrote {before_log_path}", flush=True)

    criterion = nn.CrossEntropyLoss()
    ref_acc, ref_loss = accuracy_and_loss(ref, val_loader, criterion)
    print(f"Reference model val acc={ref_acc:.4f} loss={ref_loss:.4f}", flush=True)

    cal = adapter.calibrate_reference(ref, val_loader, cfg)
    print("Calibration:", json.dumps(cal, indent=2), flush=True)
    _log_tail_prob_calibration(cal, float(cfg.gibbs_tail_prob_eps))

    print(
        "Building surgery model | "
        f"disable_layernorm_replacement={cfg.disable_layernorm_replacement} "
        f"disable_attention_surgery={cfg.disable_attention_surgery} "
        f"disable_softmax_replacement={cfg.disable_softmax_replacement} "
        f"allow_matmul={cfg.allow_matmul}",
        flush=True,
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
            print(f"Applied gibbs_tail_prob_eps mean={float(applied_mean):.6g}", flush=True)

    after_log_path = traceable_log_path(cfg.log_dir, cfg, "ts-surgery", "model_after_surgery")
    write_model_structure_txt(after_log_path, model, "Surgery Model (after transform, pre-finetune checkpoint)")
    print(f"wrote {after_log_path}", flush=True)

    pre_acc, pre_loss = accuracy_and_loss(model, val_loader, criterion)
    print(f"Post-transform val acc={pre_acc:.4f} loss={pre_loss:.4f}", flush=True)

    pre_path = traceable_artifact_path(cfg.pre_ft_checkpoint, cfg, "ts-surgery", extension=".pt")
    cfg.pre_ft_checkpoint = pre_path
    meta_path = metadata_path_for_checkpoint(pre_path)

    meta = adapter.build_surgery_meta(
        cfg,
        calibration=cal,
        reference_checkpoint_abs=reference_path,
        module_mapping=adapter.build_module_mapping(cfg, model),
    )
    if applied_mean is not None:
        meta.gibbs_tail_prob_eps = float(applied_mean)
    meta.calibration["ref_val_acc"] = float(ref_acc)
    meta.calibration["ref_val_loss"] = float(ref_loss)
    meta.calibration["student_pre_ft_val_acc"] = float(pre_acc)
    meta.calibration["student_pre_ft_mean_ce"] = float(pre_loss)

    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(pre_path) or ".", exist_ok=True)
    meta.to_json(meta_path)

    checkpoint_extra = adapter.pre_ft_checkpoint_extra(cfg, mapping=mapping, metadata_path=meta_path)
    if applied_mean is not None:
        checkpoint_extra["gibbs_tail_prob_eps"] = float(applied_mean)
    save_model_checkpoint(pre_path, model, extra=checkpoint_extra)

    print(f"wrote {meta_path}", flush=True)
    print(f"wrote {pre_path}", flush=True)
    print("Next: run the distill stage.", flush=True)
