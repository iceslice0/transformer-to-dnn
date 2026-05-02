"""Core distillation training and metadata merge helpers."""

from __future__ import annotations

import copy
import json
import math
import os
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer_surgery.models.adapters import get_model_adapter, load_surgery_student_checkpoint
from transformer_surgery.ops import CALIBRATION_LEGEND_TEXT, get_surgery_dtype, jeffreys_divergence_dense
from transformer_surgery.util import (
    DEFAULT_MODEL_KEY,
    describe_device,
    describe_dtype,
    get_device,
    metadata_path_for_checkpoint,
    save_model_checkpoint,
    set_seed,
    traceable_artifact_path,
    warmup_cosine_scheduler,
)


def _maybe_cuda_autocast(device: torch.device, dt: torch.dtype):
    if device.type == "cuda" and dt in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dt)
    return nullcontext()


def _copy_trainable_state(src: nn.Module, dst: nn.Module) -> None:
    with torch.no_grad():
        src_sd = src.state_dict()
        dst_sd = dst.state_dict()
        for name, tensor in dst_sd.items():
            if name not in src_sd:
                continue
            src_t = src_sd[name]
            if torch.is_floating_point(tensor):
                tensor.copy_(src_t.to(device=tensor.device, dtype=tensor.dtype))
            else:
                tensor.copy_(src_t.to(device=tensor.device))


def _clone_state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _validation_accuracy_summary(run_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    accs = [float(run["val_acc"]) for run in run_results]
    mean = sum(accs) / max(len(accs), 1)
    std = math.sqrt(sum((acc - mean) ** 2 for acc in accs) / max(len(accs), 1))
    return {
        "val_acc_mean": mean,
        "val_acc_std": std,
        "num_trainings": len(accs),
    }


@torch.no_grad()
def eval_distillation_metrics(
    teacher: nn.Module,
    student: nn.Module,
    val_loader: DataLoader,
    temperature: float = 1.0,
    *,
    progress_batches: int = 0,
    progress_prefix: str = "",
) -> Tuple[float, float, float]:
    """Student accuracy, mean CE, and mean Jeffreys divergence against a teacher."""
    device = get_device()
    teacher.eval()
    student.eval()
    use_cuda = device.type == "cuda"
    ce_sum, j_sum = 0.0, 0.0
    n = 0
    correct = 0
    n_val = len(val_loader)
    t0 = time.perf_counter()
    dt = get_surgery_dtype()
    for bi, (x, y) in enumerate(val_loader):
        x = x.to(device, dtype=dt, non_blocking=use_cuda)
        y = y.to(device, non_blocking=use_cuda)
        with _maybe_cuda_autocast(device, dt):
            t_log = teacher(x)
            s_log = student(x)
        ce_sum += torch.nn.functional.cross_entropy(s_log.float(), y, reduction="sum").item()
        j_sum += jeffreys_divergence_dense(t_log, s_log, temperature=temperature).sum().item()
        correct += (s_log.argmax(dim=-1) == y).sum().item()
        n += y.size(0)
        if progress_batches > 0 and n_val > 0:
            if bi == 0 or (bi + 1) % progress_batches == 0 or (bi + 1) == n_val:
                elapsed = time.perf_counter() - t0
                rate = (bi + 1) / elapsed if elapsed > 0 else 0.0
                eta = (n_val - bi - 1) / rate if rate > 0 else 0.0
                print(
                    f"  {progress_prefix}val {bi + 1}/{n_val} batches "
                    f"~{rate:.2f} batch/s eta~{eta:.0f}s",
                    flush=True,
                )
    acc = correct / max(n, 1)
    return acc, ce_sum / max(n, 1), j_sum / max(n, 1)


def distill_student_from_teacher_jeffreys(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Any,
    *,
    log_prefix: str = "distill",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Train any classifier student with mixed hard-label CE plus Jeffreys teacher matching."""
    if seed is not None:
        set_seed(seed)
    device = get_device()
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = len(train_loader)
    if cfg.max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, cfg.max_train_batches)
    epochs = cfg.epochs
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_epochs = min(5, max(cfg.epochs, 1)) if cfg.warmup_epochs is None else int(cfg.warmup_epochs)
    warmup_steps = min(warmup_epochs * steps_per_epoch, max(total_steps - 1, 0))
    scheduler = warmup_cosine_scheduler(
        opt,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        eta_min=max(0.0, float(cfg.cosine_eta_min)),
    )
    use_cuda = device.type == "cuda"
    pf = f"{log_prefix} " if log_prefix else ""
    print(
        f"  {pf}schedule: {epochs} epoch(s) x {steps_per_epoch} train batches "
        f"-> {total_steps} optimizer steps | train log every {cfg.train_progress_interval} batch(es)"
        + (
            f" | val log every {cfg.val_progress_batches} batch(es)"
            if cfg.val_progress_batches > 0
            else " | val silent"
        ),
        flush=True,
    )

    global_step = 0
    dt = get_surgery_dtype()
    use_master_fp32 = use_cuda and dt == torch.float16
    if use_master_fp32:
        train_student = copy.deepcopy(student).float()
        train_opt = torch.optim.AdamW(train_student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        train_scheduler = warmup_cosine_scheduler(
            train_opt,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            eta_min=max(0.0, float(cfg.cosine_eta_min)),
        )
        train_student.train()
    else:
        train_student = student
        train_opt = opt
        train_scheduler = scheduler
    baseline_teacher = teacher
    for p in baseline_teacher.parameters():
        p.requires_grad = False
    baseline_teacher.eval()
    baseline_student = train_student if use_master_fp32 else student
    baseline_acc, baseline_ce, baseline_j = eval_distillation_metrics(
        baseline_teacher,
        baseline_student,
        val_loader,
        temperature=cfg.temperature,
        progress_batches=cfg.val_progress_batches,
        progress_prefix=pf,
    )
    print(
        f"  {pf}baseline val acc={baseline_acc:.4f} ce={baseline_ce:.4f} jeffreys={baseline_j:.4f}",
        flush=True,
    )
    if seed is not None:
        set_seed(seed)
    best_acc = baseline_acc
    best_ep: Optional[int] = 0
    best_state: Optional[dict] = copy.deepcopy(train_student.state_dict() if use_master_fp32 else student.state_dict())
    for ep in range(epochs):
        train_student.train()
        n_batches = 0
        ep_t0 = time.perf_counter()
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device, dtype=dt, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)
            train_opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                with _maybe_cuda_autocast(device, dt):
                    t_log = baseline_teacher(x)
            if use_master_fp32:
                s_log = train_student(x.float())
            else:
                with _maybe_cuda_autocast(device, dt):
                    s_log = train_student(x)
            ce_loss = torch.nn.functional.cross_entropy(s_log.float(), y, reduction="mean")
            j_loss = jeffreys_divergence_dense(t_log, s_log, temperature=cfg.temperature).mean()
            mix = float(cfg.distill_weight)
            loss = (1.0 - mix) * ce_loss + mix * j_loss
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(train_student.parameters(), cfg.grad_clip)
            train_opt.step()
            train_scheduler.step()
            n_batches += 1
            global_step += 1
            li = float(loss.item())
            running_loss += li
            if cfg.train_progress_interval > 0:
                do_log = (
                    n_batches == 1
                    or n_batches % cfg.train_progress_interval == 0
                    or n_batches == steps_per_epoch
                )
                if cfg.max_train_batches is not None and n_batches >= cfg.max_train_batches:
                    do_log = True
                if do_log:
                    elapsed = time.perf_counter() - ep_t0
                    avg = running_loss / n_batches
                    rate = n_batches / elapsed if elapsed > 0 else 0.0
                    left = steps_per_epoch - n_batches
                    eta_s = left / rate if rate > 0 else 0.0
                    lr_c = train_opt.param_groups[0]["lr"]
                    print(
                        f"  {pf}epoch {ep + 1}/{epochs} train {n_batches}/{steps_per_epoch} "
                        f"step {global_step}/{total_steps} loss={li:.6f} loss_avg={avg:.6f} "
                        f"ce={float(ce_loss.item()):.6f} j={float(j_loss.item()):.6f} "
                        f"lr={lr_c:.2e} {rate:.2f} batch/s epoch_eta~{eta_s / 60.0:.1f}m",
                        flush=True,
                    )
            if cfg.max_train_batches is not None and n_batches >= cfg.max_train_batches:
                break
        if use_master_fp32:
            _copy_trainable_state(train_student, student)
        else:
            student = train_student
        acc, ce_v, j_v = eval_distillation_metrics(
            baseline_teacher,
            student,
            val_loader,
            temperature=cfg.temperature,
            progress_batches=cfg.val_progress_batches,
            progress_prefix=pf,
        )
        print(f"  {pf}epoch {ep + 1}/{epochs} | val acc={acc:.4f} ce={ce_v:.4f} jeffreys={j_v:.4f}", flush=True)
        if cfg.keep_best and acc > best_acc:
            best_acc = acc
            best_ep = ep + 1
            best_state = copy.deepcopy(train_student.state_dict() if use_master_fp32 else student.state_dict())
    if cfg.keep_best and best_state is not None:
        if use_master_fp32:
            train_student.load_state_dict(best_state)
            _copy_trainable_state(train_student, student)
        else:
            student.load_state_dict(best_state)
        print(f"  {pf}kept best val acc={best_acc:.4f} (epoch {best_ep}/{epochs})", flush=True)
    if use_master_fp32:
        _copy_trainable_state(train_student, student)
    acc_f, ce_f, j_f = eval_distillation_metrics(
        baseline_teacher,
        student,
        val_loader,
        temperature=cfg.temperature,
        progress_batches=cfg.val_progress_batches,
        progress_prefix=pf,
    )
    return {
        "baseline_val_acc": float(baseline_acc),
        "baseline_val_ce_mean": float(baseline_ce),
        "baseline_val_jeffreys_mean": float(baseline_j),
        "val_acc": float(acc_f),
        "val_ce_mean": float(ce_f),
        "val_jeffreys_mean": float(j_f),
        "best_val_epoch": best_ep if cfg.keep_best else None,
    }


def merge_post_distill_into_surgery_meta(
    meta_path: str,
    val_acc: float,
    val_ce: float,
    val_jeffreys: float,
    *,
    run_results: Optional[List[Dict[str, Any]]] = None,
    accuracy_summary: Optional[Dict[str, Any]] = None,
    best_run: Optional[Dict[str, Any]] = None,
    model_key: str = DEFAULT_MODEL_KEY,
    patient: str = "unknown",
    dataset: str = "unknown",
) -> None:
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = {
            "model_key": model_key,
            "patient": patient,
            "dataset": dataset,
            "calibration": {},
            "meta_note": "Stub created before distill (no prior surgery metadata at this path).",
        }
    raw.setdefault("model_key", model_key)
    raw.setdefault("patient", patient)
    raw.setdefault("dataset", dataset)
    cal = raw.get("calibration") or {}
    cal["student_post_distill_val_acc"] = float(val_acc)
    cal["student_post_distill_mean_ce"] = float(val_ce)
    cal["student_post_distill_mean_jeffreys"] = float(val_jeffreys)
    cal["val_acc_post_ft"] = float(val_acc)
    cal["val_ce_post_ft"] = float(val_ce)
    cal["val_jeffreys_post_ft"] = float(val_jeffreys)
    if run_results is not None:
        cal["student_post_distill_runs"] = run_results
    if accuracy_summary is not None:
        cal["student_post_distill_val_acc_mean"] = float(accuracy_summary["val_acc_mean"])
        cal["student_post_distill_val_acc_std"] = float(accuracy_summary["val_acc_std"])
        cal["student_post_distill_num_trainings"] = int(accuracy_summary["num_trainings"])
    if best_run is not None:
        cal["student_post_distill_best_run_index"] = int(best_run["run_index"])
        cal["student_post_distill_best_seed"] = int(best_run["seed"])
    cal.pop("val_loss_post_ft", None)
    raw["calibration"] = cal
    raw["calibration_legend"] = CALIBRATION_LEGEND_TEXT
    os.makedirs(os.path.dirname(os.path.abspath(meta_path)) or ".", exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)


def _log_distill_device_and_config_json(cfg: Any) -> None:
    print(
        f"device={describe_device(get_device())} surgery_dtype={describe_dtype(get_surgery_dtype())}",
        flush=True,
    )
    if cfg.config_json_path:
        print(f"config_json={cfg.config_json_path}", flush=True)


def _log_distill_session_line(cfg: Any, teacher_path: str) -> None:
    print(
        f"fine-tune CE+distill | teacher={teacher_path} mix={cfg.distill_weight} "
        f"epochs={cfg.epochs} lr={cfg.lr} wd={cfg.weight_decay} "
        f"num_trainings={cfg.num_trainings} base_seed={cfg.base_seed} "
        f"train_progress_interval={cfg.train_progress_interval} val_progress_batches={cfg.val_progress_batches}",
        flush=True,
    )


def _log_distill_final_metrics(result: Dict[str, Any]) -> None:
    print(
        f"run {int(result['run_number'])}/{int(result['num_trainings'])} "
        f"seed={int(result['seed'])} final val acc={float(result['val_acc']):.4f} "
        f"ce={float(result['val_ce_mean']):.4f} jeffreys={float(result['val_jeffreys_mean']):.4f}"
        + (
            f" best_epoch={result['best_val_epoch']}"
            if result.get("best_val_epoch") is not None
            else ""
        ),
        flush=True,
    )


def _log_distill_summary(summary: Dict[str, Any], best_run: Dict[str, Any]) -> None:
    print(
        f"validation accuracy over {int(summary['num_trainings'])} training run(s): "
        f"mean={float(summary['val_acc_mean']):.4f} std={float(summary['val_acc_std']):.4f}; "
        f"best run={int(best_run['run_number'])} seed={int(best_run['seed'])} "
        f"acc={float(best_run['val_acc']):.4f}",
        flush=True,
    )


def _write_distill_metadata(
    meta_out: str,
    adapter: Any,
    best_run: Dict[str, Any],
    run_results: List[Dict[str, Any]],
    accuracy_summary: Dict[str, Any],
) -> None:
    merge_post_distill_into_surgery_meta(
        meta_out,
        float(best_run["val_acc"]),
        float(best_run["val_ce_mean"]),
        float(best_run["val_jeffreys_mean"]),
        run_results=run_results,
        accuracy_summary=accuracy_summary,
        best_run=best_run,
        model_key=adapter.key,
        patient=adapter.patient_name,
        dataset=adapter.dataset_name,
    )
    print(f"wrote {meta_out}", flush=True)


def run_distill(cfg: Any) -> None:
    """Load a surgery student, run CE/Jeffreys distillation, and save traceable artifacts."""
    pre_path = os.path.abspath(cfg.pre_checkpoint)
    out_abs = traceable_artifact_path(cfg.output, cfg, "ts-distill", extension=".pt")
    meta_abs = metadata_path_for_checkpoint(out_abs)
    cfg.output = out_abs
    num_trainings = int(cfg.num_trainings)
    if num_trainings < 1:
        raise ValueError("num_trainings must be >= 1")
    base_seed = int(cfg.base_seed)

    probe_student, student_extra = load_surgery_student_checkpoint(pre_path, cfg)
    del probe_student
    adapter = get_model_adapter(student_extra.get("model_key", getattr(cfg, "model_key", None)))
    teacher_path = adapter.reference_checkpoint_path(cfg)
    _log_distill_device_and_config_json(cfg)
    teacher = adapter.load_reference_checkpoint(teacher_path).to(
        device=get_device(), dtype=get_surgery_dtype()
    )

    _log_distill_session_line(cfg, teacher_path)
    run_results: List[Dict[str, Any]] = []
    best_run: Optional[Dict[str, Any]] = None
    best_state: Optional[Dict[str, torch.Tensor]] = None
    for run_index in range(num_trainings):
        seed = base_seed + run_index
        run_number = run_index + 1
        print(f"starting distill run {run_number}/{num_trainings} seed={seed}", flush=True)
        set_seed(seed)
        student, _ = load_surgery_student_checkpoint(pre_path, cfg, adapter=adapter)
        train_loader, val_loader = adapter.build_loaders(cfg)
        result = distill_student_from_teacher_jeffreys(
            student,
            teacher,
            train_loader,
            val_loader,
            cfg,
            log_prefix=f"distill[{run_number}/{num_trainings}]",
            seed=seed,
        )
        result.update(
            {
                "run_index": run_index,
                "run_number": run_number,
                "num_trainings": num_trainings,
                "seed": seed,
            }
        )
        run_results.append(result)
        _log_distill_final_metrics(result)
        if best_run is None or float(result["val_acc"]) > float(best_run["val_acc"]):
            best_run = result
            best_state = _clone_state_dict_to_cpu(student)
    if best_run is None or best_state is None:
        raise RuntimeError("distillation produced no runs")
    accuracy_summary = _validation_accuracy_summary(run_results)
    _log_distill_summary(accuracy_summary, best_run)

    best_student, _ = load_surgery_student_checkpoint(pre_path, cfg, adapter=adapter)
    best_student.load_state_dict(best_state, strict=True)

    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    out_extra = dict(student_extra)
    out_extra.update(
        {
            "distill": "mix_ce_jeffreys",
            "model_key": adapter.key,
            "temperature": float(cfg.temperature),
            "distill_weight": float(cfg.distill_weight),
            "reference_checkpoint": teacher_path,
            "student_pre_checkpoint": pre_path,
            "base_seed": base_seed,
            "num_trainings": num_trainings,
            "distill_runs": run_results,
            "distill_val_acc_mean": float(accuracy_summary["val_acc_mean"]),
            "distill_val_acc_std": float(accuracy_summary["val_acc_std"]),
            "best_distill_run": best_run,
            "best_distill_run_index": int(best_run["run_index"]),
            "best_distill_seed": int(best_run["seed"]),
            "val_acc": float(best_run["val_acc"]),
            "val_ce_mean": float(best_run["val_ce_mean"]),
            "val_jeffreys_mean": float(best_run["val_jeffreys_mean"]),
            "best_val_epoch": best_run.get("best_val_epoch"),
            "config_json": cfg.config_json_path,
            "surgery_dtype": describe_dtype(get_surgery_dtype()),
        }
    )
    save_model_checkpoint(
        out_abs,
        best_student,
        extra=out_extra,
    )
    print(f"wrote {out_abs}", flush=True)
    _write_distill_metadata(meta_abs, adapter, best_run, run_results, accuracy_summary)
