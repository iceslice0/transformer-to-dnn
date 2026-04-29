#!/usr/bin/env python3
"""CLI for model-adapter mixed CE + teacher matching fine-tuning of a surgery student."""

from __future__ import annotations

import os
from typing import Optional

from transformer_surgery.ops import get_surgery_dtype

from transformer_surgery.cli.distill_config import (
    CLI_JEFFREYS_CONFIG_DEFAULT,
    CLI_JEFFREYS_CONFIG_HELP,
    CLI_JEFFREYS_DESCRIPTION,
    FIELD_HELP_JEFFREYS,
    JeffreysDistillConfig,
)
from transformer_surgery.model_adapters import get_model_adapter, load_surgery_student_checkpoint
from transformer_surgery.pipeline import (
    apply_device_from_config,
    apply_dtype_from_config,
    build_config_cli_parser,
    cli_overrides_from_namespace,
    describe_device,
    describe_dtype,
    distill_student_from_teacher_jeffreys,
    get_device,
    merge_post_distill_into_surgery_meta,
    save_model_checkpoint,
)


def require_pre_student_checkpoint_path(c: JeffreysDistillConfig) -> str:
    """Absolute path to surgery student checkpoint; raises if missing."""
    p = os.path.abspath(c.pre_checkpoint)
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"Missing student checkpoint: {p} (run python -m transformer_surgery.cli.run_surgery first)"
        )
    return p


def require_reference_checkpoint_path(c: JeffreysDistillConfig, adapter) -> str:
    """Absolute path to the adapter reference checkpoint; raises if missing."""
    p = adapter.reference_checkpoint_path(c)
    if not os.path.isfile(p):
        hint = f" (run {adapter.pretrain_command} first)" if adapter.pretrain_command else ""
        raise FileNotFoundError(
            f"Missing teacher checkpoint for model adapter {adapter.key!r}: {p}{hint}"
        )
    return p


def log_distill_device_and_config_json(c: JeffreysDistillConfig) -> None:
    if c.quiet:
        return
    print(
        f"device={describe_device(get_device())} surgery_dtype={describe_dtype(get_surgery_dtype())}",
        flush=True,
    )
    if c.config_json_path:
        print(f"config_json={c.config_json_path}", flush=True)


def log_distill_session_line(c: JeffreysDistillConfig, teacher_path: str) -> None:
    if c.quiet:
        return
    print(
        f"fine-tune CE+distill | teacher={teacher_path} mix={c.distill_weight} "
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
    adapter,
    val_acc: float,
    val_ce: float,
    val_j: float,
) -> None:
    if not c.meta_json:
        return
    meta_out = os.path.abspath(str(c.meta_json).strip())
    merge_post_distill_into_surgery_meta(
        meta_out,
        val_acc,
        val_ce,
        val_j,
        model_key=adapter.key,
        patient=adapter.patient_name,
        dataset=adapter.dataset_name,
    )
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
    apply_dtype_from_config(c)
    pre_path = require_pre_student_checkpoint_path(c)

    student, student_extra = load_surgery_student_checkpoint(pre_path, c)
    adapter = get_model_adapter(student_extra.get("model_key", getattr(c, "model_key", None)))
    teacher_path = require_reference_checkpoint_path(c, adapter)
    log_distill_device_and_config_json(c)
    teacher = adapter.load_reference_checkpoint(teacher_path).to(
        device=get_device(), dtype=get_surgery_dtype()
    )
    train_loader, val_loader = adapter.build_loaders(c)

    log_distill_session_line(c, teacher_path)
    val_acc, val_ce, val_j, best_ep = distill_student_from_teacher_jeffreys(
        student,
        teacher,
        train_loader,
        val_loader,
        c,
    )
    log_distill_final_metrics(val_acc, val_ce, val_j, best_ep, c)

    out_abs = os.path.abspath(c.output)
    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    out_extra = dict(student_extra)
    out_extra.update(
        {
            "distill": "mix_ce_jeffreys",
            "model_key": adapter.key,
            "temperature": float(c.temperature),
            "distill_weight": float(c.distill_weight),
            "teacher_checkpoint": teacher_path,
            "reference_checkpoint": teacher_path,
            "student_pre_checkpoint": pre_path,
            "val_acc": val_acc,
            "val_ce_mean": val_ce,
            "val_jeffreys_mean": val_j,
            "best_val_epoch": best_ep,
            "config_json": c.config_json_path,
            "surgery_dtype": describe_dtype(get_surgery_dtype()),
        }
    )
    save_model_checkpoint(
        out_abs,
        student,
        extra=out_extra,
    )
    log_wrote_checkpoint(out_abs, c)
    merge_meta_after_distill_if_configured(c, adapter, val_acc, val_ce, val_j)


if __name__ == "__main__":
    main()
