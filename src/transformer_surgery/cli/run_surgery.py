#!/usr/bin/env python3
"""
Surgery pipeline: load an adapter reference checkpoint, transform, calibrate, save pre-finetune weights.

This script does **not** run distillation or any finetuning: no teacher–student loss, no optimizer, no ``backward``.
It only builds the surgery student from reference weights, runs **eval** accuracy/loss for reporting, and saves
``surgery_pre_ft.pt`` / ``surgery_meta.json``. (The output name means “before optional Jeffreys distillation,”
not that this step finetunes.)

The default adapter is ``deit_tiny_pet``; its reference checkpoint is produced by
``python -m transformer_surgery.cli.pretrain_pet``. Other models can participate by registering an adapter
and setting ``model_key`` in the config.
"""

from __future__ import annotations

import json
import os

import torch.nn as nn

from transformer_surgery.cli.surgery_config import SurgeryRunConfig, parse_surgery_run_config
from transformer_surgery.model_adapters import get_model_adapter
from transformer_surgery.pipeline import (
    accuracy_and_loss,
    apply_device_from_config,
    apply_dtype_from_config,
    describe_device,
    describe_dtype,
    save_model_checkpoint,
)
from transformer_surgery.ops import (
    build_surgery_pwl_meta,
    write_model_structure_txt,
)


def main() -> None:
    cfg = parse_surgery_run_config()
    adapter = get_model_adapter(cfg)

    device = apply_device_from_config(cfg)
    dtype = apply_dtype_from_config(cfg)

    reference_path = adapter.reference_checkpoint_path(cfg)
    if not os.path.isfile(reference_path):
        hint = f"\nRun first: {adapter.pretrain_command}" if adapter.pretrain_command else ""
        raise SystemExit(
            f"Missing reference checkpoint for model adapter {adapter.key!r}: {reference_path}{hint}"
        )

    print(f"Using device: {describe_device(device)}", flush=True)
    print(f"Using surgery dtype: {describe_dtype(dtype)}", flush=True)
    print(f"Using model adapter: {adapter.key}", flush=True)
    if cfg.config_json_path:
        print(f"config_json={cfg.config_json_path}", flush=True)

    _, val_loader = adapter.build_loaders(cfg)

    print(f"Loading reference from {reference_path} ...", flush=True)
    ref = adapter.load_reference_checkpoint(reference_path)
    ref = ref.to(device=device, dtype=dtype)
    log_dir = os.path.abspath(cfg.log_dir)
    write_model_structure_txt(
        os.path.join(log_dir, "model_before_surgery.txt"),
        ref,
        adapter.reference_log_title,
    )
    print(f"wrote {os.path.join(log_dir, 'model_before_surgery.txt')}", flush=True)

    criterion = nn.CrossEntropyLoss()
    ref.eval()
    ref_acc, ref_loss = accuracy_and_loss(ref, val_loader, criterion)
    print(f"Reference model val acc={ref_acc:.4f} loss={ref_loss:.4f}")

    cal = adapter.calibrate_reference(ref, val_loader, cfg)
    print("Calibration:", json.dumps(cal, indent=2))

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
    print(f"Loaded {len(mapping)} tensors from reference checkpoint.")

    write_model_structure_txt(
        os.path.join(log_dir, "model_after_surgery.txt"),
        model,
        adapter.surgery_log_title,
    )
    print(f"wrote {os.path.join(log_dir, 'model_after_surgery.txt')}", flush=True)

    pre_acc, pre_loss = accuracy_and_loss(model, val_loader, criterion)
    print(f"Post-transform val acc={pre_acc:.4f} loss={pre_loss:.4f}")

    pwl_meta = build_surgery_pwl_meta()
    mod_map = adapter.build_module_mapping(cfg, model)
    meta = adapter.build_surgery_meta(
        cfg,
        calibration=cal,
        reference_checkpoint_abs=reference_path,
        pwl=pwl_meta,
        module_mapping=mod_map,
    )
    meta.calibration["ref_val_acc"] = float(ref_acc)
    meta.calibration["ref_val_loss"] = float(ref_loss)
    meta.calibration["student_pre_ft_val_acc"] = float(pre_acc)
    meta.calibration["student_pre_ft_mean_ce"] = float(pre_loss)
    meta_path = os.path.abspath(cfg.meta_json)
    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    meta.to_json(meta_path)

    pre_path = os.path.abspath(cfg.pre_ft_checkpoint)
    os.makedirs(os.path.dirname(pre_path) or ".", exist_ok=True)
    save_model_checkpoint(
        pre_path,
        model,
        extra=adapter.pre_ft_checkpoint_extra(cfg, mapping=mapping),
    )
    print(f"Wrote {pre_path} and {meta_path}")
    print("Next: python -m transformer_surgery.cli.distill", flush=True)


if __name__ == "__main__":
    main()
