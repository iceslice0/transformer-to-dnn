"""Model-agnostic surgery transform orchestration."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import torch
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


def surgery(cfg: Any, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
    """
    Load a reference checkpoint, build the surgery student, calibrate, and save traceable artifacts.

    Runtime device and dtype are already parsed by the CLI layer; if omitted, the current process
    runtime state is used.
    """
    adapter = get_model_adapter(cfg)
    device = get_device() if device is None else device
    dtype = get_surgery_dtype() if dtype is None else dtype

    reference_path = adapter.reference_checkpoint_path(cfg)
    print(f"Using device: {describe_device(device)}", flush=True)
    print(f"Using surgery dtype: {describe_dtype(dtype)}", flush=True)
    print(f"Using model adapter: {adapter.key}", flush=True)
    if cfg.config_json_path:
        print(f"config_json={cfg.config_json_path}", flush=True)

    _, val_loader = adapter.build_loaders(cfg)

    print(f"Loading reference from {reference_path} ...", flush=True)
    ref = adapter.load_reference_checkpoint(reference_path)
    ref = ref.to(device=device, dtype=dtype)
    before_log_path = traceable_log_path(cfg.log_dir, cfg, "ts-surgery", "model_before_surgery")
    write_model_structure_txt(
        before_log_path,
        ref,
        "Reference Model (before surgery transform)",
    )
    print(f"wrote {before_log_path}", flush=True)

    criterion = nn.CrossEntropyLoss()
    ref.eval()
    ref_acc, ref_loss = accuracy_and_loss(ref, val_loader, criterion)
    print(f"Reference model val acc={ref_acc:.4f} loss={ref_loss:.4f}")

    cal = adapter.calibrate_reference(ref, val_loader, cfg)
    print("Calibration:", json.dumps(cal, indent=2), flush=True)
    tail_cal = cal.get("gibbs_tail_prob_eps_calibrated_by_block")
    if isinstance(tail_cal, list) and tail_cal:
        values = ", ".join(f"{float(v):.6g}" for v in tail_cal)
        print(f"Calibrated gibbs_tail_prob_eps by block: [{values}]", flush=True)
    elif cal.get("disable_calib_gibbs_tail_prob"):
        print(
            f"gibbs_tail_prob_eps calibration disabled; using configured value {float(cfg.gibbs_tail_prob_eps):.6g}",
            flush=True,
        )

    print("Building surgery model...", flush=True)
    print(
        f"  disable_layernorm_replacement={cfg.disable_layernorm_replacement}  "
        f"disable_attention_surgery={cfg.disable_attention_surgery}  "
        f"disable_softmax_replacement={cfg.disable_softmax_replacement}  allow_matmul={cfg.allow_matmul}",
        flush=True,
    )
    model = adapter.build_surgery_model(cfg).to(device=device, dtype=dtype)
    mapping = adapter.copy_reference_weights(model, ref)
    adapter.freeze_surgery_parameters(model)
    applied_cal = adapter.apply_calibration(model, cal)
    if applied_cal:
        cal.update(applied_cal)
        if "gibbs_tail_prob_eps_applied_mean" in applied_cal:
            cfg.gibbs_tail_prob_eps = float(applied_cal["gibbs_tail_prob_eps_applied_mean"])
        print("Applied calibration:", json.dumps(applied_cal, indent=2), flush=True)
    print(f"Loaded {len(mapping)} tensors from reference checkpoint.")

    after_log_path = traceable_log_path(cfg.log_dir, cfg, "ts-surgery", "model_after_surgery")
    write_model_structure_txt(
        after_log_path,
        model,
        "Surgery Model (after transform, pre-finetune checkpoint)",
    )
    print(f"wrote {after_log_path}", flush=True)

    pre_acc, pre_loss = accuracy_and_loss(model, val_loader, criterion)
    print(f"Post-transform val acc={pre_acc:.4f} loss={pre_loss:.4f}")

    mod_map = adapter.build_module_mapping(cfg, model)
    meta = adapter.build_surgery_meta(
        cfg,
        calibration=cal,
        reference_checkpoint_abs=reference_path,
        module_mapping=mod_map,
    )
    meta.calibration["ref_val_acc"] = float(ref_acc)
    meta.calibration["ref_val_loss"] = float(ref_loss)
    meta.calibration["student_pre_ft_val_acc"] = float(pre_acc)
    meta.calibration["student_pre_ft_mean_ce"] = float(pre_loss)
    pre_path = traceable_artifact_path(cfg.pre_ft_checkpoint, cfg, "ts-surgery", extension=".pt")
    cfg.pre_ft_checkpoint = pre_path
    meta_path = metadata_path_for_checkpoint(pre_path)
    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    meta.to_json(meta_path)

    os.makedirs(os.path.dirname(pre_path) or ".", exist_ok=True)
    checkpoint_extra = adapter.pre_ft_checkpoint_extra(cfg, mapping=mapping, metadata_path=meta_path)
    for key in (
        "gibbs_tail_prob_eps_calibrated_by_block",
        "gibbs_tail_prob_eps_calibrated_mean",
        "gibbs_tail_prob_eps_applied_by_block",
        "gibbs_tail_prob_eps_applied_mean",
        "disable_calib_gibbs_tail_prob",
    ):
        if key in cal:
            checkpoint_extra[key] = cal[key]
    save_model_checkpoint(
        pre_path,
        model,
        extra=checkpoint_extra,
    )
    print(f"Wrote {pre_path} and {meta_path}")
    print("Next: run the distill stage.", flush=True)
