#!/usr/bin/env python3
"""
CLI for Jeffreys distillation. Config: :class:`pet_reference_utils.JeffreysDistillConfig`;
training loop: :func:`pet_reference_utils.distill_surgery_from_teacher_jeffreys`.
"""

from __future__ import annotations

import os
from typing import Optional

from pet_reference_utils import (
    CLI_JEFFREYS_CONFIG_DEFAULT,
    CLI_JEFFREYS_CONFIG_HELP,
    CLI_JEFFREYS_DESCRIPTION,
    FIELD_HELP_JEFFREYS,
    JeffreysDistillConfig,
    apply_device_from_config,
    build_config_cli_parser,
    build_pet_loaders,
    cli_overrides_from_namespace,
    describe_device,
    distill_surgery_from_teacher_jeffreys,
    get_device,
    load_surgery_student_checkpoint,
    load_timm_deit_pet_checkpoint,
    merge_post_distill_into_surgery_meta,
    save_deit_checkpoint,
)


def require_pre_student_checkpoint_path(c: JeffreysDistillConfig) -> str:
    """Absolute path to surgery student checkpoint; raises if missing."""
    p = os.path.abspath(c.pre_checkpoint)
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"Missing student checkpoint: {p} (run run_deit_tiny_surgery.py first)"
        )
    return p


def require_pet_teacher_checkpoint_path(c: JeffreysDistillConfig) -> str:
    """Absolute path to Pet timm teacher checkpoint; raises if missing."""
    p = os.path.abspath(c.pet_ref_checkpoint)
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"Missing teacher checkpoint: {p} (run pretrain_pet_deit_tiny.py first)"
        )
    return p


def log_distill_device_and_config_json(c: JeffreysDistillConfig) -> None:
    if c.quiet:
        return
    print(f"device={describe_device(get_device())}", flush=True)
    if c.config_json_path:
        print(f"config_json={c.config_json_path}", flush=True)


def log_distill_session_line(c: JeffreysDistillConfig, pet_teacher_path: str) -> None:
    if c.quiet:
        return
    print(
        f"distill Jeffreys | teacher={pet_teacher_path} T={c.temperature} "
        f"epochs={c.epochs} lr={c.lr} wd={c.weight_decay} "
        f"train_progress_interval={c.train_progress_interval} val_progress_batches={c.val_progress_batches}",
        flush=True,
    )


def log_distill_final_metrics(
    val_acc: float,
    val_ce: float,
    val_j: float,
    best_ep: Optional[int],
    c: JeffreysDistillConfig,
) -> None:
    if c.quiet:
        return
    print(
        f"final val acc={val_acc:.4f} ce={val_ce:.4f} jeffreys={val_j:.4f}"
        + (f" best_epoch={best_ep}" if best_ep is not None else ""),
        flush=True,
    )


def log_wrote_checkpoint(path: str, c: JeffreysDistillConfig) -> None:
    if c.quiet:
        return
    print(f"wrote {path}", flush=True)


def log_wrote_meta_json(path: str, c: JeffreysDistillConfig) -> None:
    if c.quiet:
        return
    print(f"wrote {path}", flush=True)


def merge_meta_after_distill_if_configured(
    c: JeffreysDistillConfig,
    val_acc: float,
    val_ce: float,
    val_j: float,
) -> None:
    if not c.meta_json:
        return
    meta_out = os.path.abspath(str(c.meta_json).strip())
    merge_post_distill_into_surgery_meta(meta_out, val_acc, val_ce, val_j)
    log_wrote_meta_json(meta_out, c)


def main() -> None:
    parser = build_config_cli_parser(
        CLI_JEFFREYS_DESCRIPTION,
        JeffreysDistillConfig,
        config_default=CLI_JEFFREYS_CONFIG_DEFAULT,
        config_help=CLI_JEFFREYS_CONFIG_HELP,
        field_help=FIELD_HELP_JEFFREYS,
    )
    args = parser.parse_args()
    c = JeffreysDistillConfig.load(
        args.config, cli_overrides_from_namespace(args, JeffreysDistillConfig)
    )

    apply_device_from_config(c)
    pre_path = require_pre_student_checkpoint_path(c)
    pet_path = require_pet_teacher_checkpoint_path(c)
    log_distill_device_and_config_json(c)

    student, _ = load_surgery_student_checkpoint(pre_path, c)
    teacher = load_timm_deit_pet_checkpoint(pet_path)
    train_loader, val_loader = build_pet_loaders(c)

    log_distill_session_line(c, pet_path)
    val_acc, val_ce, val_j, best_ep = distill_surgery_from_teacher_jeffreys(
        student,
        teacher,
        train_loader,
        val_loader,
        c,
    )
    log_distill_final_metrics(val_acc, val_ce, val_j, best_ep, c)

    out_abs = os.path.abspath(c.output)
    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    save_deit_checkpoint(
        out_abs,
        student,
        extra={
            "distill": "jeffreys_dense",
            "temperature": float(c.temperature),
            "teacher_checkpoint": pet_path,
            "student_pre_checkpoint": pre_path,
            "val_acc": val_acc,
            "val_ce_mean": val_ce,
            "val_jeffreys_mean": val_j,
            "best_val_epoch": best_ep,
            "config_json": c.config_json_path,
        },
    )
    log_wrote_checkpoint(out_abs, c)
    merge_meta_after_distill_if_configured(c, val_acc, val_ce, val_j)


if __name__ == "__main__":
    main()
