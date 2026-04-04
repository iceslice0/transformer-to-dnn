"""
Discovery, calibration, replacement helpers, validation metrics, and checkpoint export
for the DeiT-Tiny pseudo-hardware surgery pipeline.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default number of PWL knots (log, exp, GELU, Gibbs exp epilogue); cap at 17.
PWL_NUM_KNOTS = 17

# ---------------------------------------------------------------------------
# Selection / routing primitives (explicit ops)
# ---------------------------------------------------------------------------


class AbsOp(nn.Module):
    """Routing: abs(x)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)


class SetSign(nn.Module):
    """setsign(x, y) = sign(x) * |y| with sign(0)=1 (same convention as torch.sign for zero)."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        s = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
        return s * torch.abs(y)


class TuplePack(nn.Module):
    def forward(self, *xs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return xs


class TupleUnpack(nn.Module):
    def forward(self, packed: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        return packed


# ---------------------------------------------------------------------------
# Affine-style explicit combinators
# ---------------------------------------------------------------------------


class ExplicitIdentity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ExplicitNegate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -x


class ExplicitScale(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(float(scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.scale
        if s.device != x.device:
            s = s.to(device=x.device)
        return x * s.to(dtype=x.dtype)


class ExplicitAdd(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


class ExplicitLogPlusEps(nn.Module):
    """``log(x + eps)`` with fixed scalar ``eps`` (LayerNorm-style floor), exact ``torch.log``."""

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.register_buffer("eps", torch.tensor(float(eps)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.eps.to(device=x.device, dtype=x.dtype)
        return torch.log(x + e)


class ExplicitSquare(nn.Module):
    """Exact unary square ``x * x`` (e.g. LayerNorm variance and (a+b)²−(a−b)² identities)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class ExplicitMean(nn.Module):
    """Mean as sum followed by fixed scalar (affine reduction)."""

    def __init__(self, dim: int, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[self.dim]
        s = x.sum(dim=self.dim, keepdim=self.keepdim)
        return s * (1.0 / float(n))


# ---------------------------------------------------------------------------
# Scalar PWL (trainable knots/values) — unary epilogue
# ---------------------------------------------------------------------------


def _pwl_eval(x: torch.Tensor, knots: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation; knots strictly increasing."""
    x_flat = x.reshape(-1)
    k = knots.to(device=x.device, dtype=x.dtype)
    v = values.to(device=x.device, dtype=x.dtype)
    xc = x_flat.clamp(k[0], k[-1])
    idx = torch.searchsorted(k, xc, right=False) - 1
    idx = idx.clamp(0, k.numel() - 2)
    t0, t1 = k[idx], k[idx + 1]
    y0, y1 = v[idx], v[idx + 1]
    w = (xc - t0) / (t1 - t0 + 1e-30)
    y = y0 + w * (y1 - y0)
    return y.reshape_as(x)


class UnaryScalarPWL(nn.Module):
    """Univariate PWL with learnable knot values (knot positions fixed)."""

    def __init__(self, knots: torch.Tensor, init_values: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if knots.ndim != 1 or knots.numel() < 2:
            raise ValueError("knots must be 1D with length >= 2")
        self.register_buffer("knots", knots.clone().detach())
        if init_values is None:
            init_values = torch.zeros_like(knots)
        self.values = nn.Parameter(init_values.clone().float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _pwl_eval(x, self.knots, self.values)


def fit_pwl_to_fn(
    knots: torch.Tensor,
    fn: Callable[[torch.Tensor], torch.Tensor],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Sample fn at knots for initialization."""
    if device is None:
        device = knots.device
    with torch.no_grad():
        y = fn(knots.to(device))
    return y.detach().clone()


class MultiTailPWLEpilogue(nn.Module):
    """
    One logical affine/wide tensor feeds multiple unary PWL tails; each tail is a separate
    submodule (explicit multi-tail epilogue).
    """

    def __init__(self, tails: nn.ModuleDict) -> None:
        super().__init__()
        self.tails = tails

    def forward(self, x: torch.Tensor, which: str) -> torch.Tensor:
        return self.tails[which](x)


# ---------------------------------------------------------------------------
# LayerNorm rewrite: ExplicitLogPlusEps, SetSign, exact exp (no log PWL, no exp PWL)
# ---------------------------------------------------------------------------


class RewrittenLayerNormAbsSign(nn.Module):
    """
    mu = mean(x); u = x - mu; r2 = mean(u*u)
    log_a = log(|u|+eps) - log(r2+eps)/2 via ``ExplicitLogPlusEps`` (exact ``torch.log``)
    a_mag = exp(log_a) — exact ``torch.exp``
    z = setsign(u, a_mag);  y = gamma * z + beta
    """

    def __init__(self, normalized_shape: int, eps: float) -> None:
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        e = float(eps)
        self.log_eps = ExplicitLogPlusEps(e)
        self.mean_u = ExplicitMean(-1, keepdim=True)
        self.mean_r2 = ExplicitMean(-1, keepdim=True)
        self.square = ExplicitSquare()
        self.abs_op = AbsOp()
        self.setsign = SetSign()
        self.neg_half = ExplicitScale(-0.5)

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = self.mean_u(x)
        u = x - mu
        u2 = self.square(u)
        r2 = self.mean_r2(u2)
        au = self.abs_op(u)
        log_num = self.log_eps(au)
        log_den = self.log_eps(r2)
        log_a = log_num + self.neg_half(log_den)
        a_mag = torch.exp(log_a)
        z = self.setsign(u, a_mag)
        return z * self.weight + self.bias


def copy_ln_params_to_rewritten(dst: RewrittenLayerNormAbsSign, src: nn.LayerNorm) -> None:
    """
    Copy ``gamma``/``beta`` from a timm ``LayerNorm`` into :class:`RewrittenLayerNormAbsSign`.

    **Does not** overwrite ``dst.log_eps.eps``: that buffer is set at construction from
    ``eps_ln`` (e.g. run config). Previously we copied ``src.eps`` here, which silently
    discarded a user-configured epsilon and forced the reference value (~1e-5).
    For numerical parity with timm, keep ``eps`` in JSON equal to ``ref.blocks[0].norm1.eps``.
    """
    with torch.no_grad():
        dst.weight.copy_(torch.nan_to_num(src.weight.detach(), nan=1.0, posinf=1.0, neginf=1.0))
        dst.bias.copy_(torch.nan_to_num(src.bias.detach(), nan=0.0, posinf=0.0, neginf=0.0))


# ---------------------------------------------------------------------------
# Pairwise dot via square identity
# ---------------------------------------------------------------------------


class PairwiseDotBySquare(nn.Module):
    """
    scores[b,h,i,j] = (1/sqrt(d)) * sum_l q[b,h,i,l] * k[b,h,j,l]

    **Demonstration / plan.md form (default):** each product ``q[l]*k[l]`` is implemented only
    via the identity ``a*b = ((a+b)²-(a-b)²)/4`` with **explicit** ``plus`` / ``minus`` / square /
    sum — **no** ``torch.matmul`` for QKᵀ. This matches the pseudo-hardware story (fixed
    quarter-scale and squares vs variable activations), not generic var×var GEMM.

    Optional ``allow_matmul_scores=True`` replaces this with fused ``(q/sqrt(d)) @ kᵀ`` for
    speed / debugging only; that path **does** use matrix multiply and is not valid for the
    strict surgery demonstration.
    """

    def __init__(self, head_dim: int, *, allow_matmul_scores: bool = False) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.allow_matmul_scores = allow_matmul_scores
        if allow_matmul_scores:
            self.register_buffer("inv_sqrt_d", torch.tensor(1.0 / math.sqrt(float(head_dim))))
        else:
            scale = 0.25 / math.sqrt(float(head_dim))
            self.dot_scale = ExplicitScale(scale)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        if self.allow_matmul_scores:
            s = self.inv_sqrt_d.to(device=q.device, dtype=q.dtype)
            return (q * s) @ k.transpose(-2, -1)
        qe = q.unsqueeze(3)
        ke = k.unsqueeze(2)
        plus = qe + ke
        minus = qe - ke
        dt = torch.result_type(plus, minus)
        plus = plus.to(dt)
        minus = minus.to(dt)
        diff = plus * plus - minus * minus
        return self.dot_scale(diff.sum(dim=-1))


# ---------------------------------------------------------------------------
# Gibbs Top-K softmax with implicit replicated tail
# ---------------------------------------------------------------------------


class SelectionRoutingTopK(nn.Module):
    def __init__(self, k: int, dim: int = -1) -> None:
        super().__init__()
        self.k = k
        self.dim = dim

    def forward(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k = min(self.k, scores.shape[self.dim])
        vals, idx = torch.topk(scores, k=k, dim=self.dim, largest=True, sorted=True)
        return vals, idx


class GibbsTopKSoftmax(nn.Module):
    """
    Sparse Gibbs Top-K with implicit replicated tail at s_(k).
    Returns per-row: sparse probs on idx, tail mass scalar q_tail, and idx.
    """

    def __init__(self, seq_len: int, top_k: int, attn_drop: float = 0.0) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.attn_drop = attn_drop
        exp_k = torch.linspace(-25.0, 5.0, PWL_NUM_KNOTS)
        self.exp_pwl = UnaryScalarPWL(
            exp_k,
            init_values=fit_pwl_to_fn(exp_k, torch.exp),
        )

    def forward(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        scores: [B, H, Nq, Nk]
        returns probs [B,H,Nq,K], idx [B,H,Nq,K], q_tail [B,H,Nq,1]
        """
        b, h, nq, nk = scores.shape
        k = min(self.top_k, nk)
        scores_stable = scores - scores.max(dim=-1, keepdim=True).values
        vals, idx = torch.topk(scores_stable, k=k, dim=-1, largest=True, sorted=True)
        s_k = vals[..., -1:]
        exp_vals = self.exp_pwl(vals)
        exp_tail = self.exp_pwl(s_k)
        z_tail = exp_vals.sum(dim=-1, keepdim=True) + float(nk - k) * exp_tail
        probs = exp_vals / (z_tail + 1e-30)
        q_tail = float(nk - k) * exp_tail / (z_tail + 1e-30)
        if self.attn_drop > 0.0 and self.training:
            drop = torch.rand_like(probs) > self.attn_drop
            probs = probs * drop.to(probs.dtype) / (1.0 - self.attn_drop + 1e-12)
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-30)
        return probs, idx, q_tail


# ---------------------------------------------------------------------------
# Sparse weighted sum via square identity
# ---------------------------------------------------------------------------


class SparseWeightedSumBySquare(nn.Module):
    """
    y_i[c] = sum_{j in I} p_ij * v_j[c]

    **Default (plan.md):** each product ``p * v`` uses ``((p+v)²-(p-v)²)/4`` with
    :class:`ExplicitSquare` — no dense ``matmul`` for attention-value mixing.

    ``allow_elementwise_mul=True`` uses ``p * v`` then sum (faster; invalid for strict demo).
    """

    def __init__(self, *, allow_elementwise_mul: bool = False) -> None:
        super().__init__()
        self.allow_elementwise_mul = allow_elementwise_mul
        if not allow_elementwise_mul:
            self.square = ExplicitSquare()
            self.q = ExplicitScale(0.25)

    def forward(
        self,
        probs: torch.Tensor,
        idx: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        probs: [B,H,Nq,K]
        idx: [B,H,Nq,K] indices into Nk
        v: [B,H,Nk,D]
        returns [B,H,Nq,D]
        """
        _, _, nq, _ = probs.shape
        _, _, nk, d = v.shape
        idx_e = idx.unsqueeze(-1).expand(-1, -1, -1, -1, d)
        v_h = v.unsqueeze(2).expand(-1, -1, nq, -1, -1)
        v_g = torch.gather(v_h, 3, idx_e)
        if self.allow_elementwise_mul:
            return (probs.unsqueeze(-1) * v_g).sum(dim=3)
        p = probs.unsqueeze(-1)
        plus = p + v_g
        minus = p - v_g
        terms = self.q(self.square(plus) - self.square(minus))
        return terms.sum(dim=3)


# ---------------------------------------------------------------------------
# GELU as unary PWL (MLP nonlinearity)
# ---------------------------------------------------------------------------


class GELUUnaryPWL(nn.Module):
    def __init__(self, knots: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if knots is None:
            knots = torch.linspace(-4.0, 4.0, PWL_NUM_KNOTS)
        ref = F.gelu(knots)
        self.pwl = UnaryScalarPWL(knots, init_values=ref)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pwl(x)


# ---------------------------------------------------------------------------
# Metrics: Jeffreys, KL, LN MSE
# ---------------------------------------------------------------------------


def _kl_safe(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


def jeffreys_divergence_dense(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Symmetric Jeffreys J(p,q) = KL(p||q) + KL(q||p) for full softmax distributions.
    ``p = softmax(teacher_logits / T)``, ``q = softmax(student_logits / T)``. Returns shape [B].
    """
    t = teacher_logits / temperature
    s = student_logits / temperature
    p = F.softmax(t, dim=-1)
    q = F.softmax(s, dim=-1)
    return _kl_safe(p, q) + _kl_safe(q, p)


def jeffreys_distance_sparse_teacher(
    teacher_logits: torch.Tensor,
    vals: torch.Tensor,
    idx: torch.Tensor,
    nk: int,
    k: int,
) -> torch.Tensor:
    """
    teacher_logits: [R, nk]
    vals: top-k stabilized logits (row-wise max subtracted) [R, k]
    idx: [R, k]
    Dense student q: on I, q_i = exp(s_i)/Z_tail; each omitted index gets exp(s_k)/Z_tail.
    """
    t = teacher_logits - teacher_logits.max(dim=-1, keepdim=True).values
    p = F.softmax(t, dim=-1)

    s_k = vals[:, -1:]
    exp_vals = torch.exp(vals)
    exp_tail = torch.exp(s_k)
    z_tail = exp_vals.sum(dim=-1, keepdim=True) + float(nk - k) * exp_tail
    q_on_I = exp_vals / (z_tail + 1e-30)

    r = teacher_logits.shape[0]
    device = teacher_logits.device
    q_tail_each = exp_tail / (z_tail + 1e-30)
    q_dense = q_tail_each.expand(r, nk).clone()
    q_dense.scatter_(1, idx, q_on_I)

    j = _kl_safe(p, q_dense) + _kl_safe(q_dense, p)
    return j


def jeffreys_naive_topk(
    teacher_logits: torch.Tensor,
    vals: torch.Tensor,
    idx: torch.Tensor,
    nk: int,
    k: int,
) -> torch.Tensor:
    """Naive: softmax renormalized only over the top-k logits; zeros elsewhere."""
    t = teacher_logits - teacher_logits.max(dim=-1, keepdim=True).values
    p = F.softmax(t, dim=-1)
    exp_vals = torch.exp(vals)
    z_sub = exp_vals.sum(dim=-1, keepdim=True)
    q_sub = exp_vals / (z_sub + 1e-30)
    q_dense = torch.zeros_like(p)
    q_dense.scatter_(1, idx, q_sub)
    j = _kl_safe(p, q_dense.clamp_min(1e-12)) + _kl_safe(q_dense.clamp_min(1e-12), p)
    return j


def layer_norm_mse(rewrite: nn.Module, reference: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        y0 = reference(x)
        y1 = rewrite(x)
    return F.mse_loss(y1, y0)


# ---------------------------------------------------------------------------
# Checkpoint + metadata
# ---------------------------------------------------------------------------

CALIBRATION_LEGEND_TEXT = (
    "ref_val_acc / ref_val_loss: frozen timm teacher on val (mean CE). "
    "student_pre_ft_val_acc / student_pre_ft_mean_ce: surgery student on val "
    "after transform, before distill. "
    "student_post_distill_* and val_*_post_ft: after Jeffreys distillation "
    "(acc and mean CE / Jeffreys; legacy keys val_acc_post_ft retained)."
)


def build_default_pwl_knots() -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Returns tensor grids and a JSON-safe ``meta`` dict for :class:`SurgeryMeta`.

    **exp_knots** — positions ``linspace(-25, 5, PWL_NUM_KNOTS)``, matching
    :class:`GibbsTopKSoftmax` ``exp_pwl`` (PWL fit to ``exp``). **Used** in forward.

    **log_x_knots** — ``logspace`` grid historically tied to LN log PWL; **not** used in the
    current graph: :class:`RewrittenLayerNormAbsSign` uses :class:`ExplicitLogPlusEps` (exact
    ``torch.log``). Kept for reference only.

    **GELU** knots ``linspace(-4, 4)`` live inside :class:`GELUUnaryPWL` and are not duplicated here.
    """
    n = PWL_NUM_KNOTS
    log_x_knots = torch.logspace(math.log10(1e-8), math.log10(1e3), n)
    exp_knots = torch.linspace(-25.0, 5.0, n)
    meta: Dict[str, Any] = {
        "exp_knots": exp_knots.tolist(),
        "log_x_knots": log_x_knots.tolist(),
        "usage": {
            "exp_knots": (
                "Used: GibbsTopKSoftmax UnaryScalarPWL (same positions as "
                "torch.linspace(-25.0, 5.0, PWL_NUM_KNOTS) in code)."
            ),
            "log_x_knots": (
                "Unused in forward: RewrittenLayerNormAbsSign uses exact log via ExplicitLogPlusEps; "
                "this grid is legacy / reference only."
            ),
            "gelu": (
                "GELUUnaryPWL uses its own knot positions (default linspace(-4, 4)); not listed here."
            ),
        },
    }
    return log_x_knots, exp_knots, meta


@dataclass
class SurgeryMeta:
    patient: str = "DeiT-Tiny"
    dataset: str = "Oxford-IIIT Pet"
    eps: float = 1e-5
    top_k: int = 32
    pwl_knees: Dict[str, Any] = field(default_factory=dict)
    calibration: Dict[str, float] = field(default_factory=dict)
    module_mapping: Dict[str, str] = field(default_factory=dict)
    pet_ref_checkpoint: str = ""
    allow_matmul_scores: bool = False
    allow_elementwise_attn_value_mul: bool = False

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "patient": self.patient,
                    "dataset": self.dataset,
                    "eps": self.eps,
                    "top_k": self.top_k,
                    "pwl_knees": self.pwl_knees,
                    "calibration": self.calibration,
                    "calibration_legend": CALIBRATION_LEGEND_TEXT,
                    "module_mapping": self.module_mapping,
                    "pet_ref_checkpoint": self.pet_ref_checkpoint,
                    "allow_matmul_scores": self.allow_matmul_scores,
                    "allow_elementwise_attn_value_mul": self.allow_elementwise_attn_value_mul,
                },
                f,
                indent=2,
            )


def register_calibration_stats(meta: SurgeryMeta, key: str, value: float) -> None:
    meta.calibration[key] = float(value)
