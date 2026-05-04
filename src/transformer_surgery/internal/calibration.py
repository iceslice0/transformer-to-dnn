"""Calibration for surgery (LN copy, student hooks, ViT ref pass, tail apply) and PTQ (hook phases, range/dequant gathers, wrappers)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TYPE_CHECKING, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformer_surgery.internal.metrics import jeffreys_distance_sparse_teacher, jeffreys_naive_topk
from transformer_surgery.ops import (
    AffineContract,
    AffineFixedMix,
    AffineHadamard,
    AffineMatMul,
    AffineScale,
    AffineScaleBias,
    CalibratedAffinePTQWrapper,
    RewrittenLayerNorm,
    SurgeryAttention,
    ptq_quantize_proxy,
    ptq_signed_qrange,
)
from transformer_surgery.internal.util import get_device, get_surgery_dtype, maybe_surgery_cuda_autocast

if TYPE_CHECKING:
    from transformer_surgery.cli.surgery_config import SurgeryConfig


def _logit_stable_rows(teacher: torch.Tensor) -> torch.Tensor:
    """Per-row logits with row max subtracted (stable softmax input)."""
    return teacher - teacher.max(dim=-1, keepdim=True).values


def sample_topk_scores(
    scores: torch.Tensor,
    top_k: int,
    *,
    max_rows: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    flat = scores.reshape(-1, scores.shape[-1])
    rows = min(flat.shape[0], max_rows)
    teacher = flat[torch.randperm(flat.shape[0], device=flat.device)[:rows]].float()
    t = _logit_stable_rows(teacher)
    nk = t.shape[-1]
    k_top = min(int(top_k), nk)
    vals, idx = torch.topk(t, k=k_top, dim=-1, largest=True, sorted=True)
    return teacher, vals, idx, nk, k_top


def topk_tail_mass_stats(
    teacher: torch.Tensor,
    idx: torch.Tensor,
    nk: int,
    k_top: int,
) -> Dict[str, Any]:
    t = _logit_stable_rows(teacher)
    if nk > k_top:
        dense = torch.softmax(t, dim=-1)
        top_mass = dense.gather(1, idx).sum(dim=-1)
        tail_mass = (1.0 - top_mass).clamp(0.0, 1.0)
    else:
        tail_mass = torch.zeros(teacher.shape[0], device=teacher.device, dtype=torch.float32)
    return {
        "mean": float(tail_mass.mean().cpu()),
        "min": float(tail_mass.min().cpu()),
        "max": float(tail_mass.max().cpu()),
        "count": int(tail_mass.numel()),
    }


def add_sparse_topk_jeffreys_stats(
    stats: Dict[str, Any],
    teacher: torch.Tensor,
    vals: torch.Tensor,
    idx: torch.Tensor,
    nk: int,
    k_top: int,
    *,
    gibbs_tail_prob_eps: float,
    prefix: str,
) -> None:
    j_gibbs = jeffreys_distance_sparse_teacher(
        teacher,
        vals,
        idx,
        nk,
        k_top,
        gibbs_tail_prob_eps=gibbs_tail_prob_eps,
    ).mean()
    j_naive = jeffreys_naive_topk(teacher, vals, idx, nk, k_top).mean()
    stats[f"jeffreys_gibbs_mean_{prefix}"] = float(j_gibbs.cpu())
    stats[f"jeffreys_naive_mean_{prefix}"] = float(j_naive.cpu())
    stats[f"jeffreys_improvement_naive_minus_gibbs_{prefix}"] = float((j_naive - j_gibbs).cpu())


def _ensure_tail_slot(
    sums: List[float],
    counts: List[int],
    mins: List[float],
    maxs: List[float],
    block_idx: int,
) -> None:
    while len(sums) <= block_idx:
        sums.append(0.0)
        counts.append(0)
        mins.append(float("inf"))
        maxs.append(float("-inf"))


def _accumulate_tail_stats_for_block(
    block_idx: int,
    teacher: torch.Tensor,
    vals: torch.Tensor,
    idx: torch.Tensor,
    nk: int,
    k_top: int,
    *,
    disable_tail_calib: bool,
    tail_eps_sum_by_block: List[float],
    tail_eps_count_by_block: List[int],
    tail_eps_min_by_block: List[float],
    tail_eps_max_by_block: List[float],
) -> None:
    if disable_tail_calib:
        return
    tail_stats = topk_tail_mass_stats(teacher, idx, nk, k_top)
    count = int(tail_stats["count"])
    _ensure_tail_slot(tail_eps_sum_by_block, tail_eps_count_by_block, tail_eps_min_by_block, tail_eps_max_by_block, block_idx)
    tail_eps_sum_by_block[block_idx] += float(tail_stats["mean"]) * count
    tail_eps_count_by_block[block_idx] += count
    tail_eps_min_by_block[block_idx] = min(tail_eps_min_by_block[block_idx], float(tail_stats["min"]))
    tail_eps_max_by_block[block_idx] = max(tail_eps_max_by_block[block_idx], float(tail_stats["max"]))


def _finalize_gibbs_tail_calibration_stats(
    stats: Dict[str, Any],
    *,
    disable_tail_calib: bool,
    gibbs_tail_prob_eps: float,
    cal_batches_cfg: Any,
    processed_batches: int,
    tail_eps_sum_by_block: List[float],
    tail_eps_count_by_block: List[int],
    tail_eps_min_by_block: List[float],
    tail_eps_max_by_block: List[float],
) -> float:
    """Populate shared metadata keys; returns ``metric_tail_prob_eps`` for Jeffreys."""
    stats["disable_calib_gibbs_tail_prob"] = disable_tail_calib
    stats["gibbs_tail_calibration_batches_requested"] = cal_batches_cfg
    stats["gibbs_tail_calibration_batches"] = processed_batches
    stats["gibbs_tail_prob_eps_configured"] = gibbs_tail_prob_eps
    if disable_tail_calib:
        metric = gibbs_tail_prob_eps
    else:
        tail_eps_by_block = [
            tail_sum / max(tail_count, 1)
            for tail_sum, tail_count in zip(tail_eps_sum_by_block, tail_eps_count_by_block)
        ]
        stats["gibbs_tail_prob_eps_calibrated_by_block"] = tail_eps_by_block
        stats["gibbs_tail_prob_eps_calibrated_min_by_block"] = tail_eps_min_by_block
        stats["gibbs_tail_prob_eps_calibrated_max_by_block"] = tail_eps_max_by_block
        stats["gibbs_tail_prob_eps_calibration_rows_by_block"] = tail_eps_count_by_block
        stats["gibbs_tail_prob_eps_calibrated_mean"] = float(sum(tail_eps_by_block) / max(len(tail_eps_by_block), 1))
        metric = tail_eps_by_block[0] if tail_eps_by_block else gibbs_tail_prob_eps
    stats["gibbs_tail_prob_eps_metric"] = float(metric)
    return float(metric)


def _append_jeffreys_reporting(
    stats: Dict[str, Any],
    *,
    device: torch.device,
    dt: torch.dtype,
    block0_nk: int,
    block0_k_top: int,
    metric_tail_prob_eps: float,
) -> None:
    """Append block-0 Jeffreys synthetic check (shape from last block-0 top-k sample)."""
    if block0_nk <= 0 or block0_k_top <= 0:
        return
    teacher2 = torch.randn(4096, block0_nk, device=device, dtype=dt)
    t2_stable = _logit_stable_rows(teacher2)
    vals2, idx2 = torch.topk(t2_stable, k=block0_k_top, dim=-1, largest=True, sorted=True)
    add_sparse_topk_jeffreys_stats(
        stats,
        teacher2,
        vals2,
        idx2,
        block0_nk,
        block0_k_top,
        gibbs_tail_prob_eps=metric_tail_prob_eps,
        prefix="synthetic",
    )


def _fresh_gibbs_tail_eps_lists() -> Tuple[List[float], List[int], List[float], List[float]]:
    return [], [], [], []


def _skipped_gibbs_tail_reporting(
    *,
    disable_tail_calib: bool,
    cal_batches_cfg: Any,
    gibbs_tail_prob_eps: float,
    calibration_mode: str,
    batches: int = 0,
) -> Dict[str, Any]:
    """Reporting-only dict when Gibbs-tail calibration does not run on a forward path."""
    return {
        "disable_calib_gibbs_tail_prob": disable_tail_calib,
        "gibbs_tail_calibration_batches_requested": cal_batches_cfg,
        "gibbs_tail_calibration_batches": batches,
        "gibbs_tail_prob_eps_configured": gibbs_tail_prob_eps,
        "gibbs_tail_prob_eps_metric": float(gibbs_tail_prob_eps),
        "calibration_mode": calibration_mode,
    }


def _write_gibbs_tail_reporting(
    stats: Dict[str, Any],
    *,
    mse_acc: float,
    n_ln: int,
    disable_tail_calib: bool,
    gibbs_tail_prob_eps: float,
    cal_batches_cfg: Any,
    processed_batches: int,
    tail_eps_sum_by_block: List[float],
    tail_eps_count_by_block: List[int],
    tail_eps_min_by_block: List[float],
    tail_eps_max_by_block: List[float],
    calibration_mode: str,
    device: torch.device,
    dt: torch.dtype,
    block0_nk: int,
    block0_k_top: int,
) -> None:
    """LN MSE summary, tail metadata, mode, and Jeffreys reporting (shared by vit and surgery paths)."""
    stats["ln_rewrite_mse_all_norms_mean"] = mse_acc / max(n_ln, 1)
    metric_tail_prob_eps = _finalize_gibbs_tail_calibration_stats(
        stats,
        disable_tail_calib=disable_tail_calib,
        gibbs_tail_prob_eps=gibbs_tail_prob_eps,
        cal_batches_cfg=cal_batches_cfg,
        processed_batches=processed_batches,
        tail_eps_sum_by_block=tail_eps_sum_by_block,
        tail_eps_count_by_block=tail_eps_count_by_block,
        tail_eps_min_by_block=tail_eps_min_by_block,
        tail_eps_max_by_block=tail_eps_max_by_block,
    )
    stats["calibration_mode"] = calibration_mode
    _append_jeffreys_reporting(
        stats,
        device=device,
        dt=dt,
        block0_nk=block0_nk,
        block0_k_top=block0_k_top,
        metric_tail_prob_eps=metric_tail_prob_eps,
    )


def _ln_rewrite_mse_tensor(mod: RewrittenLayerNorm, h: torch.Tensor, y_rw: torch.Tensor, eps: float) -> torch.Tensor:
    """Squared error vs ``F.layer_norm`` using the same affine as ``RewrittenLayerNorm`` (no module cache)."""
    y_ref = F.layer_norm(
        h,
        mod.normalized_shape,
        weight=mod.affine.weight,
        bias=mod.affine.bias,
        eps=float(eps),
    )
    return (y_rw - y_ref).pow(2)


def copy_ln_params_to_rewritten(dst: RewrittenLayerNorm, src: nn.LayerNorm) -> None:
    """Copy ``LayerNorm`` affine weights into :class:`~transformer_surgery.ops.RewrittenLayerNorm`."""
    with torch.no_grad():
        dst.affine.weight.copy_(torch.nan_to_num(src.weight.detach(), nan=1.0, posinf=1.0, neginf=1.0))
        dst.affine.bias.copy_(torch.nan_to_num(src.bias.detach(), nan=0.0, posinf=0.0, neginf=0.0))


def fused_qkv_attention_qk_scores(attn: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """QK attention logits from a fused ``qkv`` linear (B, N, C) layout; ``attn`` must expose ``qkv``, ``num_heads``, ``scale``."""
    b, n, c = x.shape
    head_dim = c // int(attn.num_heads)
    qkv = attn.qkv(x).reshape(b, n, 3, attn.num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    return (q @ k.transpose(-2, -1)) * float(attn.scale)


@torch.no_grad()
def calibrate_vit_reference(
    reference: nn.Module, loader: DataLoader, cfg: "SurgeryConfig"
) -> Dict[str, Any]:
    """LN-rewrite MSE and Gibbs tail / Jeffreys stats from a ViT-shaped ``reference`` (``embed_dim``, ``patch_embed``, ``cls_token``, ``pos_embed``, ``pos_drop``, ``blocks[*].norm1|attn|norm2|mlp``, final ``norm``)."""
    device = get_device()
    dt = get_surgery_dtype()
    reference.eval()
    stats: Dict[str, Any] = {}
    eps = float(cfg.eps)
    gibbs_tail_prob_eps = float(cfg.gibbs_tail_prob_eps)
    disable_tail_calib = bool(cfg.disable_calib_gibbs_tail_prob)
    cal_batches_cfg = cfg.gibbs_tail_calibration_batches
    top_k = int(cfg.top_k)
    use_cuda = device.type == "cuda"
    mse_acc = 0.0
    n_ln = 0
    tail_eps_sum_by_block, tail_eps_count_by_block, tail_eps_min_by_block, tail_eps_max_by_block = _fresh_gibbs_tail_eps_lists()
    block0_nk = 0
    block0_k_top = 0

    batch_indices = _gibbs_cal_batch_indices(len(loader), cal_batches_cfg, random_when_limited=False)
    if not batch_indices:
        raise ValueError("gibbs tail calibration requires at least one training batch")
    batch_set = set(batch_indices)
    last_batch = max(batch_set)
    processed_batches = 0

    for batch_idx, (batch, _) in enumerate(loader):
        if batch_idx > last_batch:
            break
        if batch_idx not in batch_set:
            continue
        processed_batches += 1
        batch = batch.to(device, dtype=dt, non_blocking=use_cuda)

        with maybe_surgery_cuda_autocast(device, dt):
            b = batch.shape[0]
            x = reference.patch_embed(batch)
            x = torch.cat((reference.cls_token.expand(b, -1, -1), x), dim=1) + reference.pos_embed
            x = reference.pos_drop(x)
            if batch_idx == 0:
                y_ref0 = reference.blocks[0].norm1(x)
                rw0 = RewrittenLayerNorm(reference.embed_dim, eps=eps, allow_matmul=cfg.allow_matmul).to(
                    device=device, dtype=dt
                )
                copy_ln_params_to_rewritten(rw0, reference.blocks[0].norm1)
                y_rw0 = rw0(x)
                stats["ln_rewrite_mse_layer0_minibatch"] = float(torch.mean((y_ref0 - y_rw0).pow(2)).cpu())

            h = x
            for block_idx, blk in enumerate(reference.blocks):
                n1 = blk.norm1(h)
                rw = RewrittenLayerNorm(reference.embed_dim, eps=eps, allow_matmul=cfg.allow_matmul).to(
                    device=device, dtype=dt
                )
                copy_ln_params_to_rewritten(rw, blk.norm1)
                mse_acc += torch.mean((rw(h) - n1).pow(2)).item()
                n_ln += 1
                if block_idx == 0 or not disable_tail_calib:
                    scores = fused_qkv_attention_qk_scores(blk.attn, n1)
                    teacher, vals, idx, nk, k_top = sample_topk_scores(scores, top_k)
                    _accumulate_tail_stats_for_block(
                        block_idx,
                        teacher,
                        vals,
                        idx,
                        nk,
                        k_top,
                        disable_tail_calib=disable_tail_calib,
                        tail_eps_sum_by_block=tail_eps_sum_by_block,
                        tail_eps_count_by_block=tail_eps_count_by_block,
                        tail_eps_min_by_block=tail_eps_min_by_block,
                        tail_eps_max_by_block=tail_eps_max_by_block,
                    )
                    if block_idx == 0:
                        block0_nk = nk
                        block0_k_top = k_top
                h = h + blk.attn(n1)
                n2 = blk.norm2(h)
                rw2 = RewrittenLayerNorm(reference.embed_dim, eps=eps, allow_matmul=cfg.allow_matmul).to(
                    device=device, dtype=dt
                )
                copy_ln_params_to_rewritten(rw2, blk.norm2)
                mse_acc += torch.mean((rw2(h) - n2).pow(2)).item()
                n_ln += 1
                h = h + blk.mlp(n2)
            h_pre = h
            h_out = reference.norm(h_pre)
            rwf = RewrittenLayerNorm(reference.embed_dim, eps=eps, allow_matmul=cfg.allow_matmul).to(
                device=device, dtype=dt
            )
            copy_ln_params_to_rewritten(rwf, reference.norm)
            mse_acc += torch.mean((rwf(h_pre) - h_out).pow(2)).item()
            n_ln += 1

    if processed_batches == 0:
        raise ValueError("gibbs tail calibration requires at least one training batch")
    _write_gibbs_tail_reporting(
        stats,
        mse_acc=mse_acc,
        n_ln=n_ln,
        disable_tail_calib=disable_tail_calib,
        gibbs_tail_prob_eps=gibbs_tail_prob_eps,
        cal_batches_cfg=cal_batches_cfg,
        processed_batches=processed_batches,
        tail_eps_sum_by_block=tail_eps_sum_by_block,
        tail_eps_count_by_block=tail_eps_count_by_block,
        tail_eps_min_by_block=tail_eps_min_by_block,
        tail_eps_max_by_block=tail_eps_max_by_block,
        calibration_mode="vit_reference",
        device=device,
        dt=dt,
        block0_nk=block0_nk,
        block0_k_top=block0_k_top,
    )
    stats["calibration_loader_split"] = "train"
    return stats


def apply_gibbs_tail_calibration(model: nn.Module, calibration: Mapping[str, Any]) -> Dict[str, Any]:
    if bool(calibration.get("disable_calib_gibbs_tail_prob", False)):
        return {}
    values = calibration.get("gibbs_tail_prob_eps_calibrated_by_block")
    if not isinstance(values, list) or not values:
        return {}
    applied: List[float] = []
    with torch.no_grad():
        for block_idx, gibbs in enumerate(_gibbs_modules_in_module_order(model)):
            param = gibbs.gibbs_tail_prob_eps
            if not isinstance(param, nn.Parameter):
                continue
            raw_value = float(values[min(block_idx, len(values) - 1)])
            value = max(0.0, min(raw_value, 1.0 - 1e-7))
            param.copy_(torch.tensor(value, device=param.device, dtype=param.dtype))
            applied.append(value)
    if not applied:
        return {}
    mean_eps = float(sum(applied) / len(applied))
    if hasattr(model, "gibbs_tail_prob_eps"):
        setattr(model, "gibbs_tail_prob_eps", mean_eps)
    return {
        "gibbs_tail_prob_eps_applied_by_block": applied,
        "gibbs_tail_prob_eps_applied_mean": mean_eps,
    }


@dataclass
class CurrentBatch:
    """Batch index for forward hooks (shared with PTQ calibration loops)."""

    idx: int = -1


def calibration_input_dtype(model: nn.Module) -> torch.dtype:
    """Input dtype for calibration forwards (first parameter dtype, else surgery dtype)."""
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return get_surgery_dtype()


@torch.no_grad()
def forward_calibration_batches(
    model: nn.Module,
    loader: Any,
    batch_indices: Sequence[int],
    batch_ctx: CurrentBatch,
) -> int:
    """Run ``model`` on selected val minibatches; returns how many batches were executed."""
    batch_set = {int(i) for i in batch_indices}
    if not batch_set:
        return 0
    model.eval()
    device = get_device()
    use_cuda = device.type == "cuda"
    input_dtype = calibration_input_dtype(model)
    dt_eval = get_surgery_dtype()
    last = max(batch_set)
    n_run = 0
    for bi, (x, _y) in enumerate(loader):
        if bi > last:
            break
        if bi not in batch_set:
            continue
        batch_ctx.idx = bi
        x = x.to(device, dtype=input_dtype, non_blocking=use_cuda)
        with maybe_surgery_cuda_autocast(device, dt_eval):
            model(x)
        n_run += 1
    return n_run


def remove_forward_hook_handles(handles: List[Any]) -> None:
    for h in handles:
        h.remove()
    handles.clear()


def run_hook_phase(
    model: nn.Module,
    loader: Any,
    selected_batches: Sequence[int],
    install_hooks: Callable[[CurrentBatch, set[int]], List[Any]],
) -> int:
    batch_ctx = CurrentBatch()
    batch_set = {int(i) for i in selected_batches}
    handles = install_hooks(batch_ctx, batch_set)
    try:
        return forward_calibration_batches(model, loader, selected_batches, batch_ctx)
    finally:
        remove_forward_hook_handles(handles)


def _surgery_score_hook_submodules(model: nn.Module) -> List[Tuple[str, int]]:
    """``(name, block_index)`` for each :class:`~transformer_surgery.ops.SurgeryAttention` dot output (QK scores)."""
    out: List[Tuple[str, int]] = []
    for name, m in model.named_modules():
        if isinstance(m, SurgeryAttention) and m.use_attention_surgery and m.use_surgery_softmax:
            fq = f"{name}.dot" if name else "dot"
            out.append((fq, len(out)))
    return out


def _gibbs_modules_in_module_order(model: nn.Module) -> List[nn.Module]:
    """Gibbs softmax modules in depth-first order (matches typical block indexing)."""
    from transformer_surgery.ops import GibbsTopKSoftmax

    return [m for m in model.modules() if isinstance(m, GibbsTopKSoftmax)]


def _rewritten_layernorm_submodule_names(model: nn.Module) -> List[str]:
    names: List[str] = []
    for name, m in model.named_modules():
        if isinstance(m, RewrittenLayerNorm):
            names.append(name)
    return names


def _effective_gibbs_seq_len(model: nn.Module) -> int:
    if hasattr(model, "seq_len"):
        sl = int(getattr(model, "seq_len"))
        if sl >= 1:
            return sl
    gibbs_list = _gibbs_modules_in_module_order(model)
    return int(gibbs_list[0].seq_len) if gibbs_list else 0


@torch.no_grad()
def calibrate_surgery_student(model: nn.Module, loader: DataLoader, cfg: Any) -> Dict[str, Any]:
    """
    Gibbs tail and Jeffreys stats from the surgery student with ``gibbs.top_k = seq_len`` (dense Gibbs
    path, wrapped reference). Uses the same hook batching as PTQ. Tail / top-k sampling uses
    ``cfg.top_k`` (training k). Restores each block's ``gibbs.top_k`` before return; apply configured
    ``k`` and calibrated tail weights via ``apply_gibbs_tail_calibration`` after this call.

    Discovers :class:`~transformer_surgery.ops.SurgeryAttention` score outputs and
    :class:`~transformer_surgery.ops.RewrittenLayerNorm` modules by tree walk (no ``blocks`` layout required).
    """
    device = get_device()
    dt = get_surgery_dtype()
    stats: Dict[str, Any] = {}
    eps = float(cfg.eps)
    gibbs_tail_prob_eps = float(cfg.gibbs_tail_prob_eps)
    disable_tail_calib = bool(cfg.disable_calib_gibbs_tail_prob)
    cal_batches_cfg = cfg.gibbs_tail_calibration_batches
    top_k_cfg = int(cfg.top_k)

    score_targets = _surgery_score_hook_submodules(model)
    if not score_targets:
        return _skipped_gibbs_tail_reporting(
            disable_tail_calib=disable_tail_calib,
            cal_batches_cfg=cal_batches_cfg,
            gibbs_tail_prob_eps=gibbs_tail_prob_eps,
            calibration_mode="skipped_no_gibbs_attention",
        )

    seq_len = _effective_gibbs_seq_len(model)
    if seq_len < 1:
        raise ValueError("seq_len (model.seq_len or GibbsTopKSoftmax.seq_len) is required for surgery Gibbs calibration")

    gibbs_modules = _gibbs_modules_in_module_order(model)
    saved_top_k = [int(g.top_k) for g in gibbs_modules]
    try:
        for g in gibbs_modules:
            g.top_k = seq_len

        batch_indices = _gibbs_cal_batch_indices(len(loader), cal_batches_cfg, random_when_limited=True)
        if not batch_indices:
            raise ValueError("gibbs tail calibration requires at least one training batch")

        mse_acc = 0.0
        n_ln = 0
        tail_eps_sum_by_block, tail_eps_count_by_block, tail_eps_min_by_block, tail_eps_max_by_block = _fresh_gibbs_tail_eps_lists()
        block0_nk = 0
        block0_k_top = 0

        hook_map: Dict[str, Callable[..., None]] = {}

        for dot_name, block_idx in score_targets:

            def _dot_hook(
                mod: nn.Module,
                _inputs: Tuple[Any, ...],
                output: Any,
                *,
                _bi: int = block_idx,
                batch_idx: int = 0,
            ) -> None:
                nonlocal block0_nk, block0_k_top
                if not torch.is_tensor(output):
                    return
                scores = output
                if _bi == 0 or not disable_tail_calib:
                    teacher, vals, idx, nk, k_top = sample_topk_scores(scores, top_k_cfg)
                    _accumulate_tail_stats_for_block(
                        _bi,
                        teacher,
                        vals,
                        idx,
                        nk,
                        k_top,
                        disable_tail_calib=disable_tail_calib,
                        tail_eps_sum_by_block=tail_eps_sum_by_block,
                        tail_eps_count_by_block=tail_eps_count_by_block,
                        tail_eps_min_by_block=tail_eps_min_by_block,
                        tail_eps_max_by_block=tail_eps_max_by_block,
                    )
                    if _bi == 0:
                        block0_nk = nk
                        block0_k_top = k_top

            hook_map[dot_name] = _dot_hook

        ln_names = _rewritten_layernorm_submodule_names(model)
        layer0_ln_name = ln_names[0] if ln_names else None

        for ln_name in ln_names:

            def _ln_hook(
                mod: nn.Module,
                inputs: Tuple[Any, ...],
                output: Any,
                *,
                _ln_key: str = ln_name,
                batch_idx: int = 0,
            ) -> None:
                nonlocal mse_acc, n_ln
                if not torch.is_tensor(output) or not inputs or not torch.is_tensor(inputs[0]):
                    return
                h = inputs[0]
                mse_t = _ln_rewrite_mse_tensor(cast(RewrittenLayerNorm, mod), h, output, eps)
                mse_acc += float(mse_t.mean().item())
                n_ln += 1
                if batch_idx == 0 and layer0_ln_name is not None and _ln_key == layer0_ln_name:
                    stats["ln_rewrite_mse_layer0_minibatch"] = float(mse_t.mean().cpu())

            hook_map[ln_name] = _ln_hook

        def install(bc: CurrentBatch, bs: set[int]) -> List[Any]:
            handles: List[Any] = []

            def _wrap(fn: Callable[..., None]) -> Callable[..., None]:
                def inner(m: nn.Module, inp: Tuple[Any, ...], out: Any) -> None:
                    if bc.idx not in bs:
                        return
                    fn(m, inp, out, batch_idx=bc.idx)

                return inner

            for name, fn in hook_map.items():
                handles.append(model.get_submodule(name).register_forward_hook(_wrap(fn)))
            return handles

        processed_batches = run_hook_phase(model, loader, batch_indices, install)

        if processed_batches == 0:
            raise ValueError("gibbs tail calibration requires at least one training batch")
        _write_gibbs_tail_reporting(
            stats,
            mse_acc=mse_acc,
            n_ln=n_ln,
            disable_tail_calib=disable_tail_calib,
            gibbs_tail_prob_eps=gibbs_tail_prob_eps,
            cal_batches_cfg=cal_batches_cfg,
            processed_batches=processed_batches,
            tail_eps_sum_by_block=tail_eps_sum_by_block,
            tail_eps_count_by_block=tail_eps_count_by_block,
            tail_eps_min_by_block=tail_eps_min_by_block,
            tail_eps_max_by_block=tail_eps_max_by_block,
            calibration_mode="surgery_student_k_eq_seq_len",
            device=device,
            dt=dt,
            block0_nk=block0_nk,
            block0_k_top=block0_k_top,
        )
        stats["calibration_loader_split"] = "train"
        return stats
    finally:
        for g, tk in zip(gibbs_modules, saved_top_k):
            g.top_k = tk


def sample_calibration_batch_indices(
    total_batches: int,
    requested_batches: Optional[int],
    *,
    generator: Optional[torch.Generator] = None,
) -> List[int]:
    "Pick minibatch indices for PTQ calibration hooks (random subset, or all when requested_batches is None)."
    total = int(total_batches)
    if total < 1:
        return []
    if requested_batches is None:
        return list(range(total))
    keep = min(total, int(requested_batches))
    return sorted(torch.randperm(total, generator=generator)[:keep].tolist())


def _gibbs_cal_batch_indices(
    total_batches: int,
    cal_batches_cfg: Any,
    *,
    random_when_limited: bool,
    generator: Optional[torch.Generator] = None,
) -> List[int]:
    """Minibatch indices for Gibbs-tail calibration (all, first N, or random N)."""
    total = int(total_batches)
    if total < 1:
        return []
    if cal_batches_cfg is None:
        return list(range(total))
    lim = max(1, int(cal_batches_cfg))
    if random_when_limited:
        return sample_calibration_batch_indices(total, lim, generator=generator)
    return list(range(min(total, lim)))


PTQ_MATMUL_KINDS = frozenset({"matmul", "matmul_hadamard"})

_PTQ_KIND_BY_TYPE: Tuple[Tuple[type, str], ...] = (
    (nn.Linear, "linear"),
    (nn.Conv2d, "conv2d"),
    (AffineScale, "affine_scale"),
    (AffineScaleBias, "affine_scale_bias"),
    (AffineFixedMix, "affine_fixed_mix"),
    (AffineContract, "affine_contract"),
    (AffineMatMul, "matmul"),
    (AffineHadamard, "matmul_hadamard"),
)


def ptq_module_kind(module: nn.Module) -> Optional[str]:
    for cls, label in _PTQ_KIND_BY_TYPE:
        if isinstance(module, cls):
            return label
    return None


def ptq_activation_group(kind: str) -> str:
    return "matmul" if kind in PTQ_MATMUL_KINDS else "affine"


def ptq_activation_bits_for_kind(
    kind: str,
    activation_bits: int,
    *,
    affine_activation_bits: Optional[int] = None,
    matmul_activation_bits: Optional[int] = None,
) -> int:
    default_bits = int(activation_bits)
    if kind in PTQ_MATMUL_KINDS:
        return int(matmul_activation_bits) if matmul_activation_bits is not None else default_bits
    return int(affine_activation_bits) if affine_activation_bits is not None else default_bits


@dataclass
class PTQAffineFitStats:
    count: int = 0
    sum_x: Optional[torch.Tensor] = None
    sum_y: Optional[torch.Tensor] = None
    sum_x2: Optional[torch.Tensor] = None
    sum_xy: Optional[torch.Tensor] = None

    def _add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        sx = x.sum(dim=0).to(dtype=torch.float32, device="cpu")
        sy = y.sum(dim=0).to(dtype=torch.float32, device="cpu")
        sx2 = (x * x).sum(dim=0).to(dtype=torch.float32, device="cpu")
        sxy = (x * y).sum(dim=0).to(dtype=torch.float32, device="cpu")
        if self.sum_x is None:
            self.sum_x = sx
            self.sum_y = sy
            self.sum_x2 = sx2
            self.sum_xy = sxy
        else:
            self.sum_x += sx
            self.sum_y += sy
            self.sum_x2 += sx2
            self.sum_xy += sxy
        self.count += int(x.shape[0])

    def update(
        self,
        acc: torch.Tensor,
        out: torch.Tensor,
        *,
        channel_axis: Optional[int],
        per_channel: bool,
    ) -> None:
        if per_channel and channel_axis is not None and acc.ndim > 0:
            axis = channel_axis if channel_axis >= 0 else acc.ndim + channel_axis
            if 0 <= axis < acc.ndim and axis < out.ndim and acc.shape[axis] == out.shape[axis]:
                self._add(
                    acc.movedim(axis, -1).reshape(-1, acc.shape[axis]).to(dtype=torch.float32),
                    out.movedim(axis, -1).reshape(-1, out.shape[axis]).to(dtype=torch.float32),
                )
                return
        self._add(acc.reshape(-1, 1).to(dtype=torch.float32), out.reshape(-1, 1).to(dtype=torch.float32))

    def fit(self, var_eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.count <= 0 or self.sum_x is None or self.sum_y is None or self.sum_x2 is None or self.sum_xy is None:
            return torch.tensor(1.0, dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32)
        denom = float(self.count)
        mx = self.sum_x / denom
        my = self.sum_y / denom
        var = self.sum_x2 / denom - mx * mx
        cov = self.sum_xy / denom - mx * my
        if var.numel() == 1 and float(var.abs().item()) < var_eps:
            second = (self.sum_x2 / denom).clamp_min(var_eps)
            scale = (self.sum_xy / denom) / second
        else:
            scale = torch.where(var.abs() < var_eps, torch.zeros_like(var), cov / var.clamp_min(var_eps))
        return scale.to(dtype=torch.float32), (my - scale * mx).to(dtype=torch.float32)


@dataclass
class PTQBiasFitStats:
    count: int = 0
    residual_sum: Optional[torch.Tensor] = None

    def _add(self, residual_sum: torch.Tensor, count: int) -> None:
        if self.residual_sum is None:
            self.residual_sum = residual_sum.to(dtype=torch.float32, device="cpu")
        else:
            self.residual_sum += residual_sum.to(dtype=torch.float32, device="cpu")
        self.count += int(count)

    def update(
        self,
        acc: torch.Tensor,
        out: torch.Tensor,
        out_scale: torch.Tensor,
        *,
        channel_axis: Optional[int],
        per_channel: bool,
    ) -> None:
        out_scale = out_scale.to(device=acc.device, dtype=torch.float32)
        acc_fp = acc.to(dtype=torch.float32)
        out_fp = out.to(dtype=torch.float32)
        if per_channel and channel_axis is not None and out_scale.ndim > 0 and acc.ndim > 0:
            axis = channel_axis if channel_axis >= 0 else acc.ndim + channel_axis
            if 0 <= axis < acc.ndim and axis < out.ndim and acc.shape[axis] == out.shape[axis]:
                residual = out_fp - _broadcast_axis(out_scale, acc_fp, axis) * acc_fp
                reduce_dims = tuple(i for i in range(residual.ndim) if i != axis)
                count = 1
                for dim in reduce_dims:
                    count *= int(residual.shape[dim])
                self._add(residual.sum(dim=reduce_dims) if reduce_dims else residual, count)
                return
        residual = out_fp - out_scale.reshape(()) * acc_fp
        self._add(residual.sum().reshape(()), residual.numel())

    def bias(self) -> torch.Tensor:
        if self.count <= 0 or self.residual_sum is None:
            return torch.tensor(0.0, dtype=torch.float32)
        return (self.residual_sum / float(self.count)).to(dtype=torch.float32)


@dataclass
class PTQNodeStats:
    name: str
    kind: str
    input_arity: int = 0
    input_max_abs: List[torch.Tensor] = field(default_factory=list)
    output_rank: int = 0
    output_channel_axis: Optional[int] = None
    examples_by_batch: Dict[int, int] = field(default_factory=dict)
    dequant_examples_by_batch: Dict[int, int] = field(default_factory=dict)
    affine_fit: PTQAffineFitStats = field(default_factory=PTQAffineFitStats)
    bias_fit: PTQBiasFitStats = field(default_factory=PTQBiasFitStats)

    def observe_range(self, inputs: Sequence[torch.Tensor], output: torch.Tensor, batch_idx: int) -> None:
        if self.input_arity == 0:
            self.input_arity = len(inputs)
            self.input_max_abs = [torch.tensor(0.0, dtype=torch.float32) for _ in inputs]
        if len(inputs) != self.input_arity:
            raise ValueError(f"Input arity changed during PTQ calibration for node {self.name}")
        for idx, tensor in enumerate(inputs):
            t = tensor.detach().to(dtype=torch.float32)
            self.input_max_abs[idx] = torch.maximum(self.input_max_abs[idx], t.abs().max().to(device="cpu"))
        self.output_rank = output.ndim
        self.output_channel_axis = _default_output_channel_axis(self.kind, output)
        self.examples_by_batch[batch_idx] = int(self.examples_by_batch.get(batch_idx, 0)) + _leading_examples(output)

    def mark_dequant_examples(self, batch_idx: int, output: torch.Tensor) -> None:
        self.dequant_examples_by_batch[batch_idx] = int(
            self.dequant_examples_by_batch.get(batch_idx, 0)
        ) + _leading_examples(output)


@dataclass
class PTQNodeSetup:
    name: str
    kind: str
    activation_bits: int
    weight_bits: int
    einsum_equation: Optional[str]
    stride: Optional[Tuple[int, int]]
    padding: Optional[Tuple[int, int]]
    dilation: Optional[Tuple[int, int]]
    groups: int
    output_channel_axis: Optional[int]
    output_rank: int
    per_channel_output_affine: bool
    per_output_channel_weights_config: bool
    per_output_channel_weights_effective: bool
    input_scale: torch.Tensor
    weight_scale: torch.Tensor
    q_weight: Optional[torch.Tensor]
    module_device: torch.device


@dataclass
class PTQBuiltWrapper:
    module: CalibratedAffinePTQWrapper
    metadata: Dict[str, Any]
    reload_config: Dict[str, Any]


def observe_ptq_range(stats: PTQNodeStats, inputs: Sequence[torch.Tensor], output: torch.Tensor, batch_idx: int) -> None:
    stats.observe_range(inputs, output, batch_idx)


def observe_ptq_dequant(
    module: nn.Module,
    stats: PTQNodeStats,
    setup: PTQNodeSetup,
    inputs: Sequence[torch.Tensor],
    output: torch.Tensor,
    batch_idx: int,
) -> None:
    q_inputs = [ptq_quantize_proxy(inp, setup.input_scale[i], setup.activation_bits) for i, inp in enumerate(inputs)]
    acc = _accumulator_forward(
        module,
        setup.kind,
        _weight_to(setup.q_weight, output.device),
        q_inputs,
    ).to(dtype=torch.float32)
    if setup.kind in ("linear", "conv2d") and setup.q_weight is not None:
        stats.bias_fit.update(
            acc,
            output,
            _linear_conv_out_scale(setup.input_scale, setup.weight_scale),
            channel_axis=setup.output_channel_axis,
            per_channel=setup.per_channel_output_affine,
        )
    else:
        stats.affine_fit.update(
            acc,
            output,
            channel_axis=setup.output_channel_axis,
            per_channel=setup.per_channel_output_affine,
        )
    stats.mark_dequant_examples(batch_idx, output)


def _ptq_hook_tensor_payload(
    inputs: Tuple[Any, ...], output: Any
) -> Optional[Tuple[Tuple[torch.Tensor, ...], torch.Tensor]]:
    if not torch.is_tensor(output):
        return None
    tensor_inputs = tuple(x for x in inputs if torch.is_tensor(x))
    if not tensor_inputs:
        return None
    return tensor_inputs, output


def _register_ptq_forward_hooks(
    model: nn.Module,
    selected: Mapping[str, str],
    hook_for: Callable[[str], Callable[..., None]],
) -> List[Any]:
    return [model.get_submodule(name).register_forward_hook(hook_for(name)) for name in selected]


def _ptq_range_hook_fn(
    name: str,
    stats: Dict[str, PTQNodeStats],
    batch_ctx: CurrentBatch,
    batch_set: set[int],
) -> Callable[..., None]:
    def fn(_module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        bi = batch_ctx.idx
        if bi not in batch_set or stats[name].examples_by_batch.get(bi, 0) > 0:
            return
        payload = _ptq_hook_tensor_payload(inputs, output)
        if payload is None:
            return
        observe_ptq_range(stats[name], payload[0], payload[1], bi)

    return fn


def _ptq_dequant_hook_fn(
    name: str,
    stats: Dict[str, PTQNodeStats],
    setups: Dict[str, PTQNodeSetup],
    batch_ctx: CurrentBatch,
    batch_set: set[int],
) -> Callable[..., None]:
    def fn(_module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        if getattr(_module, "_ptq_accumulator_call", False):
            return
        bi = batch_ctx.idx
        if bi not in batch_set or stats[name].dequant_examples_by_batch.get(bi, 0) > 0:
            return
        payload = _ptq_hook_tensor_payload(inputs, output)
        if payload is None:
            return
        observe_ptq_dequant(_module, stats[name], setups[name], payload[0], payload[1], bi)

    return fn


def _install_ptq_calibration_hooks(
    model: nn.Module,
    selected: Dict[str, str],
    batch_ctx: CurrentBatch,
    batch_set: set[int],
    stats: Dict[str, PTQNodeStats],
    setups: Optional[Dict[str, PTQNodeSetup]],
) -> List[Any]:
    if setups is None:
        return _register_ptq_forward_hooks(
            model, selected, lambda n: _ptq_range_hook_fn(n, stats, batch_ctx, batch_set)
        )
    return _register_ptq_forward_hooks(
        model, selected, lambda n: _ptq_dequant_hook_fn(n, stats, setups, batch_ctx, batch_set)
    )


@torch.no_grad()
def gather_ptq_range_moments(
    model: nn.Module,
    loader: Any,
    selected: Dict[str, str],
    batch_indices: Sequence[int],
    *,
    stats: Optional[Dict[str, PTQNodeStats]] = None,
) -> Dict[str, PTQNodeStats]:
    """Register range hooks, run selected calibration batches, return per-node :class:`PTQNodeStats`."""
    out = stats or {name: PTQNodeStats(name=name, kind=kind) for name, kind in selected.items()}

    def install(batch_ctx: CurrentBatch, batch_set: set[int]) -> List[Any]:
        return _install_ptq_calibration_hooks(model, selected, batch_ctx, batch_set, out, None)

    run_hook_phase(model, loader, batch_indices, install)
    return out


@torch.no_grad()
def gather_ptq_dequant_moments(
    model: nn.Module,
    loader: Any,
    selected: Dict[str, str],
    stats: Dict[str, PTQNodeStats],
    setups: Dict[str, PTQNodeSetup],
    batch_indices: Sequence[int],
) -> None:
    """Register dequant hooks and run selected calibration batches (mutates ``stats``)."""

    def install(batch_ctx: CurrentBatch, batch_set: set[int]) -> List[Any]:
        return _install_ptq_calibration_hooks(model, selected, batch_ctx, batch_set, stats, setups)

    run_hook_phase(model, loader, batch_indices, install)


def build_ptq_node_setup(
    name: str,
    module: nn.Module,
    stats: PTQNodeStats,
    *,
    weight_bits: int,
    activation_bits: int,
    affine_activation_bits: Optional[int] = None,
    matmul_activation_bits: Optional[int] = None,
    per_output_channel: bool = True,
) -> PTQNodeSetup:
    kind = ptq_module_kind(module)
    if kind is None:
        raise ValueError(f"Unsupported PTQ module at {name}: {type(module).__name__}")
    act_bits = ptq_activation_bits_for_kind(
        kind,
        activation_bits,
        affine_activation_bits=affine_activation_bits,
        matmul_activation_bits=matmul_activation_bits,
    )
    weight_axis_used: Optional[int] = None
    weight_fp = _weight_tensor(module, kind)
    q_weight: Optional[torch.Tensor] = None
    if weight_fp is not None:
        weight_axis_used = 0 if kind in ("linear", "conv2d") and per_output_channel else None
        weight_scale = _symmetric_scale(weight_fp, int(weight_bits), axis=weight_axis_used)
        q_weight = ptq_quantize_proxy(weight_fp, weight_scale, int(weight_bits)).to(dtype=torch.float32, device="cpu")
    else:
        weight_scale = torch.tensor(1.0, dtype=torch.float32)

    return PTQNodeSetup(
        name=name,
        kind=kind,
        activation_bits=act_bits,
        weight_bits=int(weight_bits),
        einsum_equation=module.einsum_equation if isinstance(module, (AffineFixedMix, AffineContract)) else None,
        stride=tuple(module.stride) if isinstance(module, nn.Conv2d) else None,
        padding=tuple(module.padding) if isinstance(module, nn.Conv2d) else None,
        dilation=tuple(module.dilation) if isinstance(module, nn.Conv2d) else None,
        groups=int(module.groups) if isinstance(module, nn.Conv2d) else 1,
        output_channel_axis=stats.output_channel_axis,
        output_rank=stats.output_rank,
        per_channel_output_affine=stats.output_channel_axis is not None,
        per_output_channel_weights_config=bool(per_output_channel),
        per_output_channel_weights_effective=weight_axis_used is not None,
        input_scale=_input_scales(stats, act_bits),
        weight_scale=weight_scale.to(dtype=torch.float32, device="cpu"),
        q_weight=q_weight,
        module_device=_module_device(module),
    )


def build_ptq_wrapper(
    module: nn.Module,
    setup: PTQNodeSetup,
    stats: PTQNodeStats,
    *,
    dequant_var_eps: float,
) -> PTQBuiltWrapper:
    q_weight = setup.q_weight.clone() if setup.q_weight is not None else None
    if setup.kind in ("linear", "conv2d") and q_weight is not None:
        out_scale = _linear_conv_out_scale(setup.input_scale, setup.weight_scale)
        out_bias = stats.bias_fit.bias()
        out_scale_mode = "analytical_s_in_times_s_w"
        q_weight = _bake_output_scale_into_weight(q_weight, out_scale)
        out_scale_buf = torch.ones((), dtype=torch.float32)
        skip_out_scale = True
    else:
        out_scale, out_bias = stats.affine_fit.fit(float(dequant_var_eps))
        out_scale_mode = "ols_affine"
        out_scale_buf = out_scale.to(dtype=torch.float32)
        skip_out_scale = False

    out_bcast_shape = _out_bcast_shape(setup, out_scale)
    accumulator = _prepare_accumulator_module(module, setup.kind, q_weight)
    wrapper = CalibratedAffinePTQWrapper(
        activation_bits=setup.activation_bits,
        input_scale=setup.input_scale,
        accumulator=accumulator,
        out_scale=out_scale_buf,
        out_bias=out_bias,
        skip_out_scale=skip_out_scale,
        out_bcast_shape=out_bcast_shape,
        module_device=setup.module_device,
    )
    out_scale_value = out_scale.detach().cpu().tolist()
    metadata = _wrapper_metadata(setup, out_scale_mode, out_scale_value, out_bias, skip_out_scale)
    reload_config = _wrapper_reload_config(setup, wrapper, out_scale_mode, out_scale_value, skip_out_scale, out_bcast_shape)
    return PTQBuiltWrapper(module=wrapper, metadata=metadata, reload_config=reload_config)


def make_ptq_wrapper(
    module: nn.Module,
    setup: PTQNodeSetup,
    stats: PTQNodeStats,
    *,
    dequant_var_eps: float,
) -> CalibratedAffinePTQWrapper:
    return build_ptq_wrapper(module, setup, stats, dequant_var_eps=dequant_var_eps).module


def _wrapper_metadata(
    setup: PTQNodeSetup,
    out_scale_mode: str,
    out_scale_value: Any,
    out_bias: torch.Tensor,
    skip_out_scale: bool,
) -> Dict[str, Any]:
    return {
        "name": setup.name,
        "kind": setup.kind,
        "activation_group": ptq_activation_group(setup.kind),
        "activation_bits": setup.activation_bits,
        "weight_bits": setup.weight_bits,
        "per_output_channel_weights_config": setup.per_output_channel_weights_config,
        "per_output_channel_weights_effective": setup.per_output_channel_weights_effective,
        "per_channel_output_affine": setup.per_channel_output_affine,
        "output_channel_axis": setup.output_channel_axis,
        "input_scale": setup.input_scale.detach().cpu().tolist(),
        "input_zero_point": [0.0] * int(setup.input_scale.numel()),
        "weight_scale": setup.weight_scale.detach().cpu().tolist(),
        "weight_zero_point": torch.zeros_like(setup.weight_scale, dtype=torch.float32).detach().cpu().tolist(),
        "out_scale_mode": out_scale_mode,
        "out_scale": out_scale_value,
        "out_scale_baked_into_weight": skip_out_scale,
        "out_bias": out_bias.detach().cpu().tolist(),
        "stride": list(setup.stride) if setup.stride is not None else None,
        "padding": list(setup.padding) if setup.padding is not None else None,
        "dilation": list(setup.dilation) if setup.dilation is not None else None,
        "groups": setup.groups,
        "einsum_equation": setup.einsum_equation,
    }


def _wrapper_reload_config(
    setup: PTQNodeSetup,
    module: CalibratedAffinePTQWrapper,
    out_scale_mode: str,
    out_scale_value: Any,
    skip_out_scale: bool,
    out_bcast_shape: Optional[Tuple[int, ...]],
) -> Dict[str, Any]:
    return {
        "name": setup.name,
        "kind": setup.kind,
        "activation_bits": setup.activation_bits,
        "weight_bits": setup.weight_bits,
        "einsum_equation": setup.einsum_equation,
        "stride": list(setup.stride) if setup.stride is not None else None,
        "padding": list(setup.padding) if setup.padding is not None else None,
        "dilation": list(setup.dilation) if setup.dilation is not None else None,
        "groups": setup.groups,
        "out_bcast_shape": list(out_bcast_shape) if out_bcast_shape is not None else None,
        "skip_out_scale": skip_out_scale,
        "out_scale_value": out_scale_value,
        "out_scale_mode": out_scale_mode,
        "state_shapes": {name: list(tensor.shape) for name, tensor in module.state_dict().items()},
    }


def ptq_wrapper_from_reload_config(config: Dict[str, Any], module: nn.Module) -> CalibratedAffinePTQWrapper:
    state_shapes = config["state_shapes"]
    accumulator = _prepare_accumulator_module(
        module,
        config["kind"],
        _zero_accumulator_weight(config["kind"], state_shapes),
    )
    return CalibratedAffinePTQWrapper(
        activation_bits=int(config["activation_bits"]),
        input_scale=torch.ones(state_shapes["quantizer.input_scale"], dtype=torch.float32),
        accumulator=accumulator,
        out_scale=torch.ones(state_shapes["out_scale"], dtype=torch.float32),
        out_bias=torch.zeros(state_shapes["out_bias"], dtype=torch.float32),
        skip_out_scale=bool(config["skip_out_scale"]),
        out_bcast_shape=tuple(config["out_bcast_shape"]) if config.get("out_bcast_shape") is not None else None,
        module_device=torch.device("cpu"),
    )


def _weight_to(weight: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    return None if weight is None else weight.to(device=device, dtype=torch.float32)


def _accumulator_forward(
    module: nn.Module,
    kind: str,
    q_weight: Optional[torch.Tensor],
    q_inputs: Sequence[torch.Tensor],
) -> torch.Tensor:
    module._ptq_accumulator_call = True
    try:
        if kind in ("matmul", "matmul_hadamard"):
            return module(*q_inputs)
        if q_weight is None:
            raise ValueError(f"PTQ {kind} accumulator requires q_weight")
        original = _swap_accumulator_weight(module, kind, q_weight)
        try:
            return module(*q_inputs)
        finally:
            _restore_accumulator_weight(module, kind, original)
    finally:
        module._ptq_accumulator_call = False


def _prepare_accumulator_module(module: nn.Module, kind: str, q_weight: Optional[torch.Tensor]) -> nn.Module:
    if kind in ("matmul", "matmul_hadamard"):
        return module
    if q_weight is None:
        raise ValueError(f"PTQ {kind} accumulator requires q_weight")
    _swap_accumulator_weight(module, kind, q_weight)
    return module


def _swap_accumulator_weight(module: nn.Module, kind: str, q_weight: torch.Tensor) -> Tuple[Any, ...]:
    if kind in ("linear", "conv2d"):
        if kind == "linear" and not isinstance(module, nn.Linear):
            raise TypeError(f"expected nn.Linear, got {type(module).__name__}")
        if kind == "conv2d" and not isinstance(module, nn.Conv2d):
            raise TypeError(f"expected nn.Conv2d, got {type(module).__name__}")
        original = (module.weight, module.bias)
        module.weight = nn.Parameter(q_weight.to(device=module.weight.device, dtype=torch.float32), requires_grad=False)
        module.bias = None
        return original
    if kind == "affine_scale":
        original = (module.scale,)
        module.scale = q_weight.to(device=module.scale.device, dtype=torch.float32)
        return original
    if kind == "affine_scale_bias":
        if not isinstance(module, AffineScaleBias):
            raise TypeError(f"expected AffineScaleBias, got {type(module).__name__}")
        original = (module.weight, module.bias)
        module.weight = nn.Parameter(
            q_weight.reshape_as(module.weight).to(device=module.weight.device, dtype=torch.float32),
            requires_grad=False,
        )
        module.bias = nn.Parameter(torch.zeros_like(module.bias, dtype=torch.float32), requires_grad=False)
        return original
    if kind == "affine_fixed_mix":
        original = (module.weight,)
        module.weight = q_weight.to(device=module.weight.device, dtype=torch.float32)
        return original
    if kind == "affine_contract":
        original = (module.coeff,)
        module.coeff = q_weight.to(device=module.coeff.device, dtype=torch.float32)
        return original
    raise ValueError(f"Unsupported PTQ accumulator kind: {kind}")


def _restore_accumulator_weight(module: nn.Module, kind: str, original: Tuple[Any, ...]) -> None:
    if kind in ("linear", "conv2d", "affine_scale_bias"):
        module.weight, module.bias = original
    elif kind == "affine_scale":
        (module.scale,) = original
    elif kind == "affine_fixed_mix":
        (module.weight,) = original
    elif kind == "affine_contract":
        (module.coeff,) = original
    else:
        raise ValueError(f"Unsupported PTQ accumulator kind: {kind}")


def _zero_accumulator_weight(kind: str, state_shapes: Dict[str, List[int]]) -> Optional[torch.Tensor]:
    if kind in ("linear", "conv2d", "affine_scale_bias", "affine_fixed_mix"):
        return torch.zeros(state_shapes["accumulator.weight"], dtype=torch.float32)
    if kind == "affine_scale":
        return torch.zeros(state_shapes["accumulator.scale"], dtype=torch.float32)
    if kind == "affine_contract":
        return torch.zeros(state_shapes["accumulator.coeff"], dtype=torch.float32)
    if kind in ("matmul", "matmul_hadamard"):
        return None
    raise ValueError(f"Unsupported PTQ accumulator kind: {kind}")


def _out_bcast_shape(setup: PTQNodeSetup, out_scale: torch.Tensor) -> Optional[Tuple[int, ...]]:
    if not setup.per_channel_output_affine or setup.output_channel_axis is None:
        return None
    if setup.output_rank <= 0 or out_scale.ndim == 0:
        return None
    axis = setup.output_channel_axis if setup.output_channel_axis >= 0 else setup.output_rank + setup.output_channel_axis
    shape = [1] * setup.output_rank
    shape[axis] = -1
    return tuple(shape)


def _leading_examples(t: torch.Tensor) -> int:
    return 1 if t.ndim == 0 else int(t.shape[0])


def _symmetric_scale(x: torch.Tensor, bits: int, axis: Optional[int] = None) -> torch.Tensor:
    _qmin, qmax = ptq_signed_qrange(bits)
    if axis is None:
        max_abs = x.abs().max()
    else:
        dims = tuple(i for i in range(x.ndim) if i != axis)
        max_abs = x.abs() if not dims else x.abs().amax(dim=dims, keepdim=True)
    return (max_abs / float(max(qmax, 1))).clamp_min(1e-8).to(dtype=torch.float32)


def _default_output_channel_axis(kind: str, out: torch.Tensor) -> Optional[int]:
    if out.ndim == 0:
        return None
    if kind == "conv2d":
        return 1 if out.ndim > 1 else 0
    if kind == "affine_fixed_mix":
        return -2 if out.ndim >= 2 else None
    return -1


_WEIGHT_ATTR: Dict[str, str] = {
    "linear": "weight",
    "conv2d": "weight",
    "affine_scale": "scale",
    "affine_scale_bias": "weight",
    "affine_fixed_mix": "weight",
    "affine_contract": "coeff",
}


def _weight_tensor(module: nn.Module, kind: str) -> Optional[torch.Tensor]:
    attr = _WEIGHT_ATTR.get(kind)
    if attr is None:
        return None
    return getattr(module, attr).detach().to(dtype=torch.float32, device="cpu")


def _broadcast_axis(vec: torch.Tensor, ref: torch.Tensor, axis: Optional[int]) -> torch.Tensor:
    if vec.ndim == 0 or axis is None:
        return vec
    axis = axis if axis >= 0 else ref.ndim + axis
    shape = [1] * ref.ndim
    shape[axis] = vec.numel()
    return vec.view(*shape)


def _linear_conv_out_scale(input_scale: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    s_in = input_scale[0].to(dtype=torch.float32)
    ws = weight_scale.to(dtype=torch.float32)
    return s_in * ws if ws.ndim == 0 else (s_in * ws.reshape(-1)).to(dtype=torch.float32)


def _input_scales(stats: PTQNodeStats, activation_bits: int) -> torch.Tensor:
    if stats.input_arity <= 0 or not stats.input_max_abs:
        raise ValueError(f"No input range statistics collected for node {stats.name}")
    qmax = float(max(ptq_signed_qrange(activation_bits)[1], 1))
    return torch.stack([(max_abs.to(dtype=torch.float32) / qmax).clamp_min(1e-8) for max_abs in stats.input_max_abs])


def _module_device(module: nn.Module) -> torch.device:
    for tensor in list(module.parameters()) + list(module.buffers()):
        return tensor.device
    return torch.device("cpu")


def _bake_output_scale_into_weight(q_weight: torch.Tensor, out_scale: torch.Tensor) -> torch.Tensor:
    if out_scale.ndim == 0:
        return q_weight * out_scale.to(dtype=torch.float32)
    wshape = [1] * q_weight.ndim
    wshape[0] = -1
    return q_weight * out_scale.view(*wshape).to(dtype=torch.float32)
