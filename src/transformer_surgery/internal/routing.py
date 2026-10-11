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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, get_args

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer_surgery.internal.calibration import (
    calibration_batch_indices,
    dense_gibbs,
    gibbs_attention_modules,
    run_hooked,
)
from transformer_surgery.cli.surgery_config import SupportRouting
from transformer_surgery.internal.reporting import log_line
from transformer_surgery.internal.util import get_device
from transformer_surgery.ops import QueryCentroidRouter

if TYPE_CHECKING:
    from transformer_surgery.cli.surgery_config import SurgeryConfig

SUPPORT_ROUTINGS = get_args(SupportRouting)


@dataclass
class KMeansFit:
    """Per-head k-means result: centroids [H, C, D], inertia relative to the mean squared query norm, empty clusters."""

    centroids: torch.Tensor
    relative_inertia: torch.Tensor
    empty_clusters: torch.Tensor


@dataclass
class CentroidFit:
    """Fit quality of one block's centroids (metadata ``routing_fit_by_block`` entry)."""

    samples_per_head: int
    relative_inertia_by_head: List[float]
    empty_clusters_by_head: List[int]


@dataclass
class SupportSummary:
    """Routed vs exact top-k support of one block over rows with more valid keys than ``top_k``."""

    rows: int
    mass_topk: float
    mass_routed: float
    recall: float
    tail_routed: float
    tail_routed_std_by_head: List[float]
    mass_topk_by_head: List[float]
    mass_routed_by_head: List[float]
    recall_by_head: List[float]
    mass_ratio_min_head: float


@dataclass
class RoutingCalibration:
    """Query-centroid routing fit and support diagnostics (metadata ``calibration`` keys)."""

    support_routing: Literal["query_centroids"]
    routing_clusters: int
    routing_fit_batches: int
    routing_fit_by_block: List[CentroidFit]
    routing_train_batches: int
    routing_val_batches: int
    routing_train_by_block: List[SupportSummary]
    routing_val_by_block: List[SupportSummary]
    routing_val_mass_topk: float
    routing_val_mass_routed: float
    routing_val_recall: float
    # Mean tail mass of the routed support per block: the calibrated tail masses of the routed model.
    routing_train_tail_routed_by_block: List[float]


@torch.no_grad()
def kmeans_per_head(x: torch.Tensor, clusters: int, iters: int, generator: torch.Generator) -> KMeansFit:
    """k-means++ then Lloyd iterations, batched over heads. ``x``: [H, n, D] float32."""
    h, n, _ = x.shape
    c_count = min(int(clusters), n)
    heads = torch.arange(h, device=x.device)
    first = torch.randint(n, (h,), device=x.device, generator=generator)
    cents = [x[heads, first].unsqueeze(1)]
    d2 = (x - cents[0]).pow(2).sum(-1)
    for _ in range(1, c_count):
        total = d2.sum(-1, keepdim=True)
        weights = d2 / total.clamp_min(1e-30)
        # A head can have fewer distinct queries than requested centroids. Once all
        # remaining distances are zero, any sample is a valid next centroid.
        weights = torch.where(total > 0, weights, torch.full_like(weights, 1.0 / n))
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
    return KMeansFit(cent, inertia / q_sq.clamp_min(1e-30), (counts == 0).sum(-1))


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

    def hook(_m: nn.Module, inputs: Any, _out: Any, bi: int) -> None:
        q = inputs[0].detach()
        b, h, n, d = q.shape
        xq = q.permute(1, 0, 2, 3).reshape(h, b * n, d)
        if xq.shape[1] > per_batch:
            keep = torch.randperm(xq.shape[1], generator=generator)[:per_batch].to(xq.device)
            xq = xq[:, keep]
        samples[bi].append(xq.float().cpu())

    handles = [
        attn.dot.register_forward_hook(lambda m, i, o, bi=bi: hook(m, i, o, bi)) for bi, attn in enumerate(attns)
    ]
    run_hooked(model, loader, batch_indices, handles)
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

    def summary(self) -> SupportSummary:
        n = self.rows.clamp_min(1.0)
        mass_o = self.mass_topk / n
        mass_r = self.mass_routed / n
        tail_var = (self.tail_routed_sq / n - (1.0 - mass_r).pow(2)).clamp_min(0.0)
        total = float(self.rows.sum().clamp_min(1.0))
        return SupportSummary(
            rows=int(self.rows.sum()),
            mass_topk=float(self.mass_topk.sum()) / total,
            mass_routed=float(self.mass_routed.sum()) / total,
            recall=float(self.recall.sum()) / total,
            tail_routed=1.0 - float(self.mass_routed.sum()) / total,
            tail_routed_std_by_head=tail_var.sqrt().tolist(),
            mass_topk_by_head=mass_o.tolist(),
            mass_routed_by_head=mass_r.tolist(),
            recall_by_head=(self.recall / n).tolist(),
            mass_ratio_min_head=float((mass_r / mass_o.clamp_min(1e-30)).min()),
        )


def _support_stats_pass(
    model: nn.Module,
    attns: Sequence[nn.Module],
    routers: Sequence[nn.Module],
    loader: DataLoader,
    batch_indices: Sequence[int],
    top_k: int,
) -> List[SupportSummary]:
    acc = [_SupportStats(a.num_heads) for a in attns]
    stash: Dict[int, Any] = {}

    def dot_hook(inputs: Any, bi: int) -> None:
        stash[bi] = (inputs[0].detach(), inputs[1].detach())

    def gibbs_pre_hook(inputs: Any, bi: int) -> None:
        q, k = stash.pop(bi)
        acc[bi].add(q, k, inputs[0].detach(), routers[bi], top_k)

    handles = []
    for bi, attn in enumerate(attns):
        handles.append(attn.dot.register_forward_hook(lambda _m, i, _o, bi=bi: dot_hook(i, bi)))
        handles.append(attn.gibbs.register_forward_pre_hook(lambda _m, i, bi=bi: gibbs_pre_hook(i, bi)))
    run_hooked(model, loader, batch_indices, handles)
    return [a.summary() for a in acc]


def install_routers_from_state_dict(model: nn.Module, state_dict: Dict[str, Any]) -> None:
    """Create empty :class:`QueryCentroidRouter` modules where ``state_dict`` holds router buffers."""
    for name, m in model.named_modules():
        key = f"{name}.router.assign_scores.coeff" if name else "router.assign_scores.coeff"
        if getattr(m, "router", 0) is None and key in state_dict:
            m.router = QueryCentroidRouter(torch.zeros(state_dict[key].shape))


def _pooled(blocks: Sequence[SupportSummary], values: Sequence[float]) -> float:
    """Row-weighted mean over blocks."""
    return sum(v * b.rows for b, v in zip(blocks, values)) / max(1, sum(b.rows for b in blocks))


@torch.no_grad()
def calibrate_support_routing(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: "SurgeryConfig"
) -> Optional[RoutingCalibration]:
    """Fit query centroids and install the routers; None for ``support_routing="topk"``."""
    if cfg.support_routing not in SUPPORT_ROUTINGS:
        raise ValueError(f"support_routing must be one of {SUPPORT_ROUTINGS}, got {cfg.support_routing!r}")
    if cfg.support_routing == "topk":
        return None
    attns = [a for a in gibbs_attention_modules(model) if hasattr(a, "router")]
    if not attns:
        raise ValueError("support_routing needs surgery attention with the surgery softmax")
    device = get_device()
    cpu_gen = torch.Generator().manual_seed(cfg.routing_seed)
    fit: List[CentroidFit] = []
    routers: List[nn.Module] = []
    with dense_gibbs(attns):
        fit_batches = calibration_batch_indices(
            len(train_loader), cfg.routing_calibration_batches, random=True, generator=cpu_gen
        )
        queries = _collect_queries(model, attns, train_loader, fit_batches, cfg.routing_max_samples, cpu_gen)
        for bi, xq in enumerate(queries):
            dev_gen = torch.Generator(device=device).manual_seed(cfg.routing_seed + 1000 * (bi + 1))
            km = kmeans_per_head(xq.to(device), cfg.routing_clusters, cfg.routing_kmeans_iters, dev_gen)
            routers.append(QueryCentroidRouter(km.centroids).to(device))
            fit.append(CentroidFit(int(xq.shape[1]), km.relative_inertia.tolist(), km.empty_clusters.tolist()))
        cal_batches = calibration_batch_indices(len(train_loader), cfg.gibbs_tail_calibration_batches)
        train_blocks = _support_stats_pass(model, attns, routers, train_loader, cal_batches, cfg.top_k)
        val_batches = calibration_batch_indices(len(val_loader), cfg.routing_eval_batches)
        val_blocks = _support_stats_pass(model, attns, routers, val_loader, val_batches, cfg.top_k)
    for a, r in zip(attns, routers):
        a.router = r
    return RoutingCalibration(
        support_routing="query_centroids",
        routing_clusters=cfg.routing_clusters,
        routing_fit_batches=len(fit_batches),
        routing_fit_by_block=fit,
        routing_train_batches=len(cal_batches),
        routing_val_batches=len(val_batches),
        routing_train_by_block=train_blocks,
        routing_val_by_block=val_blocks,
        routing_val_mass_topk=_pooled(val_blocks, [b.mass_topk for b in val_blocks]),
        routing_val_mass_routed=_pooled(val_blocks, [b.mass_routed for b in val_blocks]),
        routing_val_recall=_pooled(val_blocks, [b.recall for b in val_blocks]),
        routing_train_tail_routed_by_block=[b.tail_routed for b in train_blocks],
    )


def log_support_routing(routing: RoutingCalibration) -> None:
    log_line(
        f"Support routing={routing.support_routing} clusters={routing.routing_clusters} "
        f"fit_batches={routing.routing_fit_batches} val_batches={routing.routing_val_batches}"
    )
    for bi, (b, f) in enumerate(zip(routing.routing_val_by_block, routing.routing_fit_by_block)):
        inertia = sum(f.relative_inertia_by_head) / max(1, len(f.relative_inertia_by_head))
        log_line(
            f"  block {bi}: val mass topk={b.mass_topk:.4f} routed={b.mass_routed:.4f} "
            f"recall={b.recall:.4f} min_head_ratio={b.mass_ratio_min_head:.4f} "
            f"rel_inertia={inertia:.4f}"
        )
    log_line(
        f"Support routing val pooled: mass topk={routing.routing_val_mass_topk:.4f} "
        f"routed={routing.routing_val_mass_routed:.4f} recall={routing.routing_val_recall:.4f}"
    )
