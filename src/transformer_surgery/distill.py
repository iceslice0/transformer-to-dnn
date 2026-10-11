"""Core distillation training and metadata helpers."""

from __future__ import annotations

import copy
import os
import statistics
import time
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer_surgery.internal.reporting import (
    CALIBRATION_LEGEND_TEXT,
    describe_device,
    describe_dtype,
    log_line,
    log_wrote,
    metadata_path_for_checkpoint,
    traceable_artifact_path,
    write_json,
)
from transformer_surgery.internal.util import (
    get_device,
    get_surgery_dtype,
    maybe_surgery_cuda_autocast,
    save_model_checkpoint,
    set_seed,
    warmup_cosine_scheduler,
)
from transformer_surgery.models.adapters import get_model_adapter, load_surgery_student_checkpoint

if TYPE_CHECKING:
    from transformer_surgery.cli.distill_config import JeffreysDistillConfig
    from transformer_surgery.models.adapters import SurgeryModelAdapter


@dataclass
class DistillRun:
    """Validation of one distillation run before training (baseline) and of its kept student."""

    seed: int
    baseline_val_acc: float
    baseline_val_ce_mean: float
    baseline_val_jeffreys_mean: float
    val_acc: float
    val_ce_mean: float
    val_jeffreys_mean: float
    best_val_epoch: Optional[int]
    run_index: int = 0


@dataclass
class DistillSummary:
    """The best of several distillation runs and the spread of their accuracy (distill metadata ``calibration``)."""

    student_post_distill_val_acc: float
    student_post_distill_mean_ce: float
    student_post_distill_mean_jeffreys: float
    student_post_distill_runs: List[DistillRun]
    student_post_distill_val_acc_mean: float
    student_post_distill_val_acc_std: float
    student_post_distill_num_trainings: int
    student_post_distill_best_run_index: int
    student_post_distill_best_seed: int

    @classmethod
    def of(cls, runs: Sequence[DistillRun]) -> "DistillSummary":
        best = max(runs, key=lambda r: r.val_acc)
        accs = [r.val_acc for r in runs]
        return cls(
            student_post_distill_val_acc=best.val_acc,
            student_post_distill_mean_ce=best.val_ce_mean,
            student_post_distill_mean_jeffreys=best.val_jeffreys_mean,
            student_post_distill_runs=list(runs),
            student_post_distill_val_acc_mean=statistics.mean(accs),
            student_post_distill_val_acc_std=statistics.stdev(accs) if len(accs) > 1 else 0.0,
            student_post_distill_num_trainings=len(accs),
            student_post_distill_best_run_index=best.run_index,
            student_post_distill_best_seed=best.seed,
        )


def distill_student_from_teacher(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: "JeffreysDistillConfig",
    *,
    adapter: "SurgeryModelAdapter",
    seed: int,
    baseline: Tuple[float, float, float],
    log_prefix: str = "distill",
) -> DistillRun:
    """Train a student with a task-defined hard loss mixed with teacher matching.

    ``adapter`` supplies ``(hard_loss, match_loss)`` per step and ``(primary, loss, match)`` for
    evaluation, so this loop serves classification (CE + Jeffreys) and SR (L1 + teacher L1) alike.
    ``baseline`` is the student's ``(primary, loss, match)`` before training.
    """
    set_seed(seed)
    device = get_device()
    dt = get_surgery_dtype()
    use_cuda = device.type == "cuda"
    steps_per_epoch = len(train_loader)
    if cfg.max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, cfg.max_train_batches)
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    warmup_epochs = min(5, max(cfg.epochs, 1)) if cfg.warmup_epochs is None else cfg.warmup_epochs
    log_line(
        f"  {log_prefix} schedule: {cfg.epochs} epoch(s) x {steps_per_epoch} train batches "
        f"-> {total_steps} optimizer steps | train log every {cfg.train_progress_interval} batch(es)"
    )

    # fp16 trains an fp32 master copy and copies it back into the student for evaluation.
    master = copy.deepcopy(student).float() if use_cuda and dt == torch.float16 else student
    optimizer = torch.optim.AdamW(master.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=min(warmup_epochs * steps_per_epoch, max(total_steps - 1, 0)),
        eta_min=max(0.0, float(cfg.cosine_eta_min)),
    )
    teacher.eval()
    best_acc, best_ce, best_j = baseline
    best_ep = 0
    best_state = copy.deepcopy(master.state_dict()) if cfg.keep_best else None
    global_step = 0
    for ep in range(cfg.epochs):
        master.train()
        ep_t0 = time.perf_counter()
        running_loss = torch.zeros((), device=device)
        for n_batches, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, dtype=dt, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), maybe_surgery_cuda_autocast(device, dt):
                t_log = teacher(x)
            if master is student:
                with maybe_surgery_cuda_autocast(device, dt):
                    s_log = master(x)
            else:
                s_log = master(x.float())
            hard_loss, match_loss = adapter.distill_step_losses(s_log, t_log, y, temperature=cfg.temperature)
            loss = (1.0 - cfg.distill_weight) * hard_loss + cfg.distill_weight * match_loss
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(master.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            running_loss += loss.detach()
            interval = cfg.train_progress_interval
            if interval > 0 and (n_batches == 1 or n_batches % interval == 0 or n_batches == steps_per_epoch):
                elapsed = time.perf_counter() - ep_t0
                rate = n_batches / elapsed if elapsed > 0 else 0.0
                eta_s = (steps_per_epoch - n_batches) / rate if rate > 0 else 0.0
                log_line(
                    f"  {log_prefix} epoch {ep + 1}/{cfg.epochs} train {n_batches}/{steps_per_epoch} "
                    f"step {global_step}/{total_steps} loss={loss.item():.6f} "
                    f"loss_avg={running_loss.item() / n_batches:.6f} "
                    f"hard={hard_loss.item():.6f} match={match_loss.item():.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} {rate:.2f} batch/s epoch_eta~{eta_s / 60.0:.1f}m"
                )
            if cfg.max_train_batches is not None and n_batches >= cfg.max_train_batches:
                break
        if master is not student:
            student.load_state_dict(master.state_dict())
        acc, ce_v, j_v = adapter.eval_student_vs_teacher(teacher, student, val_loader, temperature=cfg.temperature)
        if not cfg.keep_best or acc > best_acc:
            best_acc, best_ce, best_j, best_ep = acc, ce_v, j_v, ep + 1
            if cfg.keep_best:
                best_state = copy.deepcopy(master.state_dict())
    if best_state is not None:
        master.load_state_dict(best_state)
        if master is not student:
            student.load_state_dict(master.state_dict())
        log_line(f"  {log_prefix} kept best val acc={best_acc:.4f} (epoch {best_ep}/{cfg.epochs})")
    baseline_acc, baseline_ce, baseline_j = baseline
    return DistillRun(
        seed=seed,
        baseline_val_acc=float(baseline_acc),
        baseline_val_ce_mean=float(baseline_ce),
        baseline_val_jeffreys_mean=float(baseline_j),
        val_acc=float(best_acc),
        val_ce_mean=float(best_ce),
        val_jeffreys_mean=float(best_j),
        best_val_epoch=best_ep if cfg.keep_best else None,
    )


def distill_metadata(adapter: "SurgeryModelAdapter", summary: DistillSummary) -> Dict[str, Any]:
    """Distill metadata JSON: patient and dataset, the run summary as the ``calibration`` block."""
    return {
        "model_key": adapter.key,
        "patient": adapter.patient_name,
        "dataset": adapter.dataset_name,
        "calibration": asdict(summary),
        "calibration_legend": CALIBRATION_LEGEND_TEXT,
    }


def run_distill(cfg: "JeffreysDistillConfig") -> None:
    """Load a surgery student, run CE/Jeffreys distillation, and save traceable artifacts."""
    pre_path = os.path.abspath(cfg.pre_checkpoint)
    out_abs = traceable_artifact_path(cfg.output, cfg, "ts-distill", extension=".pt")
    meta_abs = metadata_path_for_checkpoint(out_abs)
    cfg.output = out_abs

    probe_student, student_extra = load_surgery_student_checkpoint(pre_path, cfg)
    adapter = get_model_adapter(student_extra["model_key"])
    teacher_path = adapter.reference_checkpoint_path(cfg)
    log_line(f"device={describe_device(get_device())} surgery_dtype={describe_dtype(get_surgery_dtype())}")
    log_line(f"config_json={cfg.config_json_path}")
    teacher = adapter.load_reference_checkpoint(teacher_path).to(device=get_device(), dtype=get_surgery_dtype())
    teacher.requires_grad_(False)
    teacher.eval()
    log_line(
        f"fine-tune CE+distill | teacher={teacher_path} mix={cfg.distill_weight} "
        f"epochs={cfg.epochs} lr={cfg.lr} wd={cfg.weight_decay} "
        f"num_trainings={cfg.num_trainings} base_seed={cfg.base_seed} "
        f"train_progress_interval={cfg.train_progress_interval}"
    )

    _, baseline_val_loader = adapter.build_loaders(cfg)
    baseline = adapter.eval_student_vs_teacher(teacher, probe_student, baseline_val_loader, temperature=cfg.temperature)
    log_line(f"baseline val primary={baseline[0]:.4f} loss={baseline[1]:.4f} match={baseline[2]:.4f}")
    del probe_student, baseline_val_loader

    runs: List[DistillRun] = []
    best_run: Optional[DistillRun] = None
    best_state: Optional[Dict[str, torch.Tensor]] = None
    for run_index in range(cfg.num_trainings):
        seed = cfg.base_seed + run_index
        run_number = run_index + 1
        log_line(f"starting distill run {run_number}/{cfg.num_trainings} seed={seed}")
        set_seed(seed)
        student, _ = load_surgery_student_checkpoint(pre_path, cfg)
        train_loader, val_loader = adapter.build_loaders(cfg)
        run = distill_student_from_teacher(
            student,
            teacher,
            train_loader,
            val_loader,
            cfg,
            adapter=adapter,
            seed=seed,
            baseline=baseline,
            log_prefix=f"distill[{run_number}/{cfg.num_trainings}]",
        )
        run = replace(run, run_index=run_index)
        runs.append(run)
        log_line(
            f"run {run_number}/{cfg.num_trainings} seed={seed} final val acc={run.val_acc:.4f} "
            f"ce={run.val_ce_mean:.4f} jeffreys={run.val_jeffreys_mean:.4f}"
            + (f" best_epoch={run.best_val_epoch}" if run.best_val_epoch is not None else "")
        )
        if best_run is None or run.val_acc > best_run.val_acc:
            best_run = run
            best_state = {name: t.detach().cpu().clone() for name, t in student.state_dict().items()}
    if best_run is None or best_state is None:
        raise ValueError("num_trainings must be at least 1")
    summary = DistillSummary.of(runs)
    log_line(
        f"validation accuracy over {cfg.num_trainings} training run(s): "
        f"mean={summary.student_post_distill_val_acc_mean:.4f} std={summary.student_post_distill_val_acc_std:.4f}; "
        f"best run={best_run.run_index + 1} seed={best_run.seed} acc={best_run.val_acc:.4f}"
    )

    best_student, _ = load_surgery_student_checkpoint(pre_path, cfg)
    best_student.load_state_dict(best_state, strict=True)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    out_extra = {
        **student_extra,
        "distill": "mix_ce_jeffreys",
        "distill_config": asdict(cfg),
        "teacher_checkpoint": teacher_path,
        "student_pre_checkpoint": pre_path,
        "distill_runs": [asdict(r) for r in runs],
        "distill_val_acc_mean": summary.student_post_distill_val_acc_mean,
        "distill_val_acc_std": summary.student_post_distill_val_acc_std,
        "best_distill_run": asdict(best_run),
    }
    save_model_checkpoint(out_abs, best_student, extra=out_extra)
    log_wrote(out_abs)
    write_json(meta_abs, distill_metadata(adapter, summary))
    log_wrote(meta_abs)
