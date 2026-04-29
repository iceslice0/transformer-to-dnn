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
from transformer_surgery.models.adapters import get_model_adapter, load_surgery_student_checkpoint
from transformer_surgery.pipeline import (
    apply_device_from_config,
    apply_dtype_from_config,
    build_config_cli_parser,
    cli_overrides_from_namespace,
    describe_device,
    describe_dtype,
    distill_student_from_teacher_jeffreys,
    get_device,
    metadata_path_for_checkpoint,
    merge_post_distill_into_surgery_meta,
    save_model_checkpoint,
    traceable_artifact_path,
)


def log_distill_device_and_config_json(c: JeffreysDistillConfig) -> None:
    print(
        f"device={describe_device(get_device())} surgery_dtype={describe_dtype(get_surgery_dtype())}",
        flush=True,
    )
    if c.config_json_path:
        print(f"config_json={c.config_json_path}", flush=True)


def log_distill_session_line(c: JeffreysDistillConfig, teacher_path: str) -> None:
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
    print(
        f"final val acc={val_acc:.4f} ce={val_ce:.4f} jeffreys={val_j:.4f}"
        + (f" best_epoch={best_ep}" if best_ep is not None else ""),
        flush=True,
    )


def log_wrote_checkpoint(path: str) -> None:
    print(f"wrote {path}", flush=True)


def log_wrote_metadata(path: str) -> None:
    print(f"wrote {path}", flush=True)


def write_distill_metadata(
    meta_out: str,
    adapter,
    val_acc: float,
    val_ce: float,
    val_j: float,
) -> None:
    merge_post_distill_into_surgery_meta(
        meta_out,
        val_acc,
        val_ce,
        val_j,
        model_key=adapter.key,
        patient=adapter.patient_name,
        dataset=adapter.dataset_name,
    )
    log_wrote_metadata(meta_out)


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
    pre_path = os.path.abspath(c.pre_checkpoint)
    out_abs = traceable_artifact_path(c.output, c, "ts-distill", extension=".pt")
    meta_abs = metadata_path_for_checkpoint(out_abs)
    c.output = out_abs

    student, student_extra = load_surgery_student_checkpoint(pre_path, c)
    adapter = get_model_adapter(student_extra.get("model_key", getattr(c, "model_key", None)))
    teacher_path = adapter.reference_checkpoint_path(c)
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

    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    out_extra = dict(student_extra)
    out_extra.update(
        {
            "distill": "mix_ce_jeffreys",
            "model_key": adapter.key,
            "temperature": float(c.temperature),
            "distill_weight": float(c.distill_weight),
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
    log_wrote_checkpoint(out_abs)
    write_distill_metadata(meta_abs, adapter, val_acc, val_ce, val_j)


if __name__ == "__main__":
    main()
