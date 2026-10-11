"""
Query-centroid support routing: centroid fit, routed tail mass, and support diagnostics.

All passes run the built surgery student with ``gibbs.top_k = seq_len`` (dense Gibbs path, the
wrapped reference), as :func:`~transformer_surgery.internal.calibration.calibrate_surgery_student`
does. Queries and keys are read at the input of each attention ``dot`` node (after rotary for causal
models); the masked scores are read at the input of ``gibbs``.

Per block and head the fit runs k-means on calibration queries and installs a
:class:`~transformer_surgery.ops.QueryCentroidRouter`. Diagnostics compare the routed support with
the exact top-k support of the same size on rows with more valid keys than ``top_k``:

- ``mass``: dense attention mass captured by the support, ``sum_{j in S} pi_ij``;
- ``recall``: fraction of the exact top-k keys that the routed support contains;
- routed tail mass ``1 - mass``, whose calibration mean is the calibrated ``q`` of the routed support.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer_surgery.internal.calibration import (
    _effective_gibbs_seq_len,
    _gibbs_cal_batch_indices,
    run_hook_phase,
)
from transformer_surgery.internal.reporting import log_line
from transformer_surgery.internal.util import get_device

SUPPORT_ROUTINGS = ("topk", "query_centroids")


def routed_attention_modules(model: nn.Module) -> List[nn.Module]:
    """Surgery attention modules with a Gibbs top-k softmax, in module order (block order)."""
    return [
        m
        for m in model.modules()
        if bool(getattr(m, "use_attention_surgery", False))
        and bool(getattr(m, "use_surgery_softmax", False))
        and hasattr(m, "gibbs")
        and hasattr(m, "dot")
        and hasattr(m, "router")
    ]


@torch.no_grad()
def kmeans_per_head(
    x: torch.Tensor, clusters: int, iters: int, generator: torch.Generator
) -> Dict[str, torch.Tensor]:
    """k-means++ then Lloyd iterations, batched over heads. ``x``: [H, n, D] float32."""
    h, n, _ = x.shape
    c_count = min(int(clusters), n)
    heads = torch.arange(h, device=x.device)
    first = torch.randint(n, (h,), device=x.device, generator=generator)
    cents = [x[heads, first].unsqueeze(1)]
    d2 = (x - cents[0]).pow(2).sum(-1)
    for _ in range(1, c_count):
        weights = d2 / d2.sum(-1, keepdim=True).clamp_min(1e-30)
        nxt = torch.multinomial(weights, 1, generator=generator).squeeze(-1)
        c_new = x[heads, nxt].unsqueeze(1)
        cents.append(c_new)
        d2 = torch.minimum(d2, (x - c_new).pow(2).sum(-1))
    cent = torch.cat(cents, dim=1)
    x_sq = x.pow(2).sum(-1, keepdim=True)
    for _ in range(int(iters)):
        dist = x_sq - 2.0 * torch.einsum("hnd,hcd->hnc", x, cent) + cent.pow(2).sum(-1).unsqueeze(1)
        # One-hot contraction instead of scatter_add_: CUDA scatter_add_ sums with atomics in no fixed order, so
        # the centroids, and the routed runs, would change from run to run under the same seed.
        onehot = torch.nn.functional.one_hot(dist.argmin(-1), c_count).to(x.dtype)
        sums = torch.einsum("hnc,hnd->hcd", onehot, x)
        counts = onehot.sum(1)
        cent = torch.where(counts.unsqueeze(-1) > 0, sums / counts.clamp_min(1.0).unsqueeze(-1), cent)
    dist = x_sq - 2.0 * torch.einsum("hnd,hcd->hnc", x, cent) + cent.pow(2).sum(-1).unsqueeze(1)
    counts = torch.nn.functional.one_hot(dist.argmin(-1), c_count).to(x.dtype).sum(1)
    q_sq = x_sq.squeeze(-1).mean(-1)
    inertia = dist.min(-1).values.clamp_min(0.0).mean(-1)
    return {
        "centroids": cent,
        "relative_inertia": inertia / q_sq.clamp_min(1e-30),
        "empty_clusters": (counts == 0).sum(-1),
    }


def _collect_queries(
    model: nn.Module,
    attns: Sequence[nn.Module],
    loader: DataLoader,
    batch_indices: Sequence[int],
    max_samples: int,
    generator: torch.Generator,
) -> List[torch.Tensor]:
    per_batch = max(1, math.ceil(int(max_samples) / max(1, len(batch_indices))))
    samples: List[List[torch.Tensor]] = [[] for _ in attns]

    def install(_batch_ctx: Any, _batch_set: set[int]) -> List[Any]:
        handles = []
        for bi, attn in enumerate(attns):

            def hook(_m: nn.Module, inputs: Any, _out: Any, bi: int = bi) -> None:
                q = inputs[0].detach()
                b, h, n, d = q.shape
                xq = q.permute(1, 0, 2, 3).reshape(h, b * n, d)
                if xq.shape[1] > per_batch:
                    keep = torch.randperm(xq.shape[1], generator=generator)[:per_batch].to(xq.device)
                    xq = xq[:, keep]
                samples[bi].append(xq.float().cpu())

            handles.append(attn.dot.register_forward_hook(hook))
        return handles

    run_hook_phase(model, loader, batch_indices, install)
    return [torch.cat(s, dim=1) for s in samples]


class _SupportStats:
    """Row sums per head for one block."""

    def __init__(self, heads: int) -> None:
        self.rows = torch.zeros(heads, dtype=torch.float64)
        self.mass_topk = torch.zeros(heads, dtype=torch.float64)
        self.mass_routed = torch.zeros(heads, dtype=torch.float64)
        self.recall = torch.zeros(heads, dtype=torch.float64)
        self.tail_routed_sq = torch.zeros(heads, dtype=torch.float64)

    @torch.no_grad()
    def add(self, q: torch.Tensor, k: torch.Tensor, scores: torch.Tensor, router: nn.Module, top_k: int) -> None:
        s = scores.float()
        valid = torch.isfinite(s)
        nk = s.shape[-1]
        kk = min(int(top_k), nk)
        rows = (valid.sum(-1) > kk).to(torch.float64)
        p = torch.softmax(s, dim=-1)
        _vo, idx_o = torch.topk(s, kk, dim=-1)
        mass_o = p.gather(-1, idx_o).sum(-1).double()
        selection = router(q.float(), k.float()).masked_fill(~valid, float("-inf"))
        _vr, idx_r = torch.topk(selection, kk, dim=-1)
        mass_r = p.gather(-1, idx_r).sum(-1).double().clamp(max=1.0)
        in_topk = torch.zeros_like(valid).scatter_(-1, idx_o, True) & valid
        recall = in_topk.gather(-1, idx_r).sum(-1).double() / float(kk)
        tail_r = 1.0 - mass_r
        dims = (0, 2)
        self.rows += rows.sum(dims).cpu()
        self.mass_topk += (mass_o * rows).sum(dims).cpu()
        self.mass_routed += (mass_r * rows).sum(dims).cpu()
        self.recall += (recall * rows).sum(dims).cpu()
        self.tail_routed_sq += (tail_r.pow(2) * rows).sum(dims).cpu()

    def summary(self) -> Dict[str, Any]:
        n = self.rows.clamp_min(1.0)
        mass_o = self.mass_topk / n
        mass_r = self.mass_routed / n
        tail_mean = 1.0 - mass_r
        tail_var = (self.tail_routed_sq / n - tail_mean.pow(2)).clamp_min(0.0)
        total = float(self.rows.sum().clamp_min(1.0))
        return {
            "rows": int(self.rows.sum()),
            "mass_topk": float(self.mass_topk.sum()) / total,
            "mass_routed": float(self.mass_routed.sum()) / total,
            "recall": float(self.recall.sum()) / total,
            "tail_routed": 1.0 - float(self.mass_routed.sum()) / total,
            "tail_routed_std_by_head": [float(v) for v in tail_var.sqrt()],
            "mass_topk_by_head": [float(v) for v in mass_o],
            "mass_routed_by_head": [float(v) for v in mass_r],
            "recall_by_head": [float(v) for v in self.recall / n],
            "mass_ratio_min_head": float((mass_r / mass_o.clamp_min(1e-30)).min()),
        }


def _support_stats_pass(
    model: nn.Module,
    attns: Sequence[nn.Module],
    routers: Sequence[nn.Module],
    loader: DataLoader,
    batch_indices: Sequence[int],
    top_k: int,
) -> List[Dict[str, Any]]:
    acc = [_SupportStats(int(getattr(a, "num_heads"))) for a in attns]
    stash: Dict[int, Any] = {}

    def install(_batch_ctx: Any, _batch_set: set[int]) -> List[Any]:
        handles = []
        for bi, attn in enumerate(attns):

            def dot_hook(_m: nn.Module, inputs: Any, _out: Any, bi: int = bi) -> None:
                stash[bi] = (inputs[0].detach(), inputs[1].detach())

            def gibbs_pre_hook(_m: nn.Module, inputs: Any, bi: int = bi) -> None:
                q, k = stash.pop(bi)
                acc[bi].add(q, k, inputs[0].detach(), routers[bi], top_k)

            handles.append(attn.dot.register_forward_hook(dot_hook))
            handles.append(attn.gibbs.register_forward_pre_hook(gibbs_pre_hook))
        return handles

    run_hook_phase(model, loader, batch_indices, install)
    return [a.summary() for a in acc]


def install_routers_from_state_dict(model: nn.Module, state_dict: Dict[str, Any]) -> int:
    """Create empty :class:`QueryCentroidRouter` modules where ``state_dict`` holds router buffers."""
    from transformer_surgery.ops import QueryCentroidRouter

    installed = 0
    for name, m in model.named_modules():
        if not hasattr(m, "router") or getattr(m, "router") is not None:
            continue
        key = f"{name}.router.assign_scores.coeff" if name else "router.assign_scores.coeff"
        if key in state_dict:
            m.router = QueryCentroidRouter(torch.zeros(tuple(state_dict[key].shape)))
            installed += 1
    return installed


def _pooled(blocks: Sequence[Dict[str, Any]], key: str) -> float:
    rows = sum(b["rows"] for b in blocks)
    return sum(b[key] * b["rows"] for b in blocks) / max(1, rows)


@torch.no_grad()
def calibrate_support_routing(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: Any
) -> Dict[str, Any]:
    """Fit query centroids, install routers, and (calibrated mode) apply the routed tail mass."""
    from transformer_surgery.ops import QueryCentroidRouter

    routing = str(getattr(cfg, "support_routing", "topk"))
    if routing not in SUPPORT_ROUTINGS:
        raise ValueError(f"support_routing must be one of {SUPPORT_ROUTINGS}, got {routing!r}")
    if routing == "topk":
        return {}
    attns = routed_attention_modules(model)
    if not attns:
        raise ValueError("support_routing needs surgery attention with the surgery softmax")
    device = get_device()
    top_k = int(cfg.top_k)
    seed = int(cfg.routing_seed)
    cpu_gen = torch.Generator().manual_seed(seed)
    seq_len = _effective_gibbs_seq_len(model)
    saved_top_k = [a.gibbs.top_k for a in attns]
    fit: List[Dict[str, Any]] = []
    try:
        for a in attns:
            a.gibbs.top_k = seq_len
        fit_batches = _gibbs_cal_batch_indices(
            len(train_loader), cfg.routing_calibration_batches, random_when_limited=True, generator=cpu_gen
        )
        queries = _collect_queries(model, attns, train_loader, fit_batches, cfg.routing_max_samples, cpu_gen)
        routers: List[nn.Module] = []
        for bi, xq in enumerate(queries):
            dev_gen = torch.Generator(device=device).manual_seed(seed + 1000 * (bi + 1))
            km = kmeans_per_head(xq.to(device), int(cfg.routing_clusters), int(cfg.routing_kmeans_iters), dev_gen)
            routers.append(QueryCentroidRouter(km["centroids"]).to(device))
            fit.append(
                {
                    "samples_per_head": int(xq.shape[1]),
                    "relative_inertia_by_head": [float(v) for v in km["relative_inertia"]],
                    "empty_clusters_by_head": [int(v) for v in km["empty_clusters"]],
                }
            )
        cal_batches = _gibbs_cal_batch_indices(
            len(train_loader), cfg.gibbs_tail_calibration_batches, random_when_limited=False
        )
        train_blocks = _support_stats_pass(model, attns, routers, train_loader, cal_batches, top_k)
        val_batches = _gibbs_cal_batch_indices(len(val_loader), cfg.routing_eval_batches, random_when_limited=False)
        val_blocks = _support_stats_pass(model, attns, routers, val_loader, val_batches, top_k)
    finally:
        for a, kk in zip(attns, saved_top_k):
            a.gibbs.top_k = kk
    for a, r in zip(attns, routers):
        a.router = r

    stats: Dict[str, Any] = {
        "support_routing": routing,
        "routing_clusters": int(cfg.routing_clusters),
        "routing_fit_batches": len(fit_batches),
        "routing_fit_by_block": fit,
        "routing_train_batches": len(cal_batches),
        "routing_val_batches": len(val_batches),
        "routing_train_by_block": train_blocks,
        "routing_val_by_block": val_blocks,
        "routing_val_mass_topk": _pooled(val_blocks, "mass_topk"),
        "routing_val_mass_routed": _pooled(val_blocks, "mass_routed"),
        "routing_val_recall": _pooled(val_blocks, "recall"),
        "routing_train_tail_routed_by_block": [b["tail_routed"] for b in train_blocks],
    }
    calibrated_mode = not bool(getattr(cfg, "use_exact_tail_mass", False)) and not bool(
        getattr(cfg, "disable_calib_gibbs_tail_prob", False)
    )
    if calibrated_mode:
        applied: List[float] = []
        for a, b in zip(attns, train_blocks):
            param = a.gibbs.gibbs_tail_prob_eps
            value = max(0.0, min(float(b["tail_routed"]), 1.0 - 1e-7))
            param.data.fill_(value)
            applied.append(value)
        stats["gibbs_tail_prob_eps_applied_by_block"] = applied
        stats["gibbs_tail_prob_eps_applied_mean"] = float(sum(applied) / len(applied))
    return stats


def log_support_routing(stats: Dict[str, Any]) -> None:
    if not stats:
        return
    log_line(
        f"Support routing={stats['support_routing']} clusters={stats['routing_clusters']} "
        f"fit_batches={stats['routing_fit_batches']} val_batches={stats['routing_val_batches']}"
    )
    for bi, (b, f) in enumerate(zip(stats["routing_val_by_block"], stats["routing_fit_by_block"])):
        inertia = sum(f["relative_inertia_by_head"]) / max(1, len(f["relative_inertia_by_head"]))
        log_line(
            f"  block {bi}: val mass topk={b['mass_topk']:.4f} routed={b['mass_routed']:.4f} "
            f"recall={b['recall']:.4f} min_head_ratio={b['mass_ratio_min_head']:.4f} "
            f"rel_inertia={inertia:.4f}"
        )
    log_line(
        f"Support routing val pooled: mass topk={stats['routing_val_mass_topk']:.4f} "
        f"routed={stats['routing_val_mass_routed']:.4f} recall={stats['routing_val_recall']:.4f}"
    )
    applied = stats.get("gibbs_tail_prob_eps_applied_by_block")
    if applied:
        values = ", ".join(f"{v:.6g}" for v in applied)
        log_line(f"Routed calibrated gibbs_tail_prob_eps by block: [{values}]")
