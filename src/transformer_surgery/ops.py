"""
Discovery, calibration, replacement helpers, validation metrics, and checkpoint export
for the DeiT-Tiny pseudo-hardware surgery pipeline.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default number of PWL knots (log, exp, GELU, Gibbs exp epilogue); cap at 17.
PWL_NUM_KNOTS = 17

# Process-wide dtype for surgery tensor literals and ``.to(dtype=...)`` (set via ``apply_dtype_from_config``).
_SURGERY_DTYPE: torch.dtype = torch.bfloat16


def get_surgery_dtype() -> torch.dtype:
    """Current surgery compute dtype (default ``bfloat16``; set with :func:`set_surgery_dtype`)."""
    return _SURGERY_DTYPE


def set_surgery_dtype(dt: torch.dtype) -> None:
    """Set global surgery dtype (mirrors process device pattern in ``transformer_surgery.pet``)."""
    global _SURGERY_DTYPE
    _SURGERY_DTYPE = dt

# ---------------------------------------------------------------------------
# Op vocabulary: Affine* (fixed coeff einsum, channel scale+bias, fixed matrix mixes), Unary* (maps & reductions),
# MatMul* (contracting ``matmul`` + Hadamard ``a*b`` with two variable tensors). Routing helpers below.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Affine-style explicit combinators
# ---------------------------------------------------------------------------


class UnaryScale(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(float(scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.scale
        if s.device != x.device:
            s = s.to(device=x.device)
        return x * s.to(dtype=x.dtype)


class MatMulHadamard(nn.Module):
    """Elementwise / broadcast product ``a * b``; both operands are tensors (MatMul group, not ``@``)."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b


class MatMul(nn.Module):
    """Contracting product ``torch.matmul(a, b)`` (when ``allow_matmul`` enables batched @)."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)


class AffineScaleBias(nn.Module):
    """Per-channel ``y = x * weight + bias`` (LayerNorm gamma / beta on the last dim)."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias


class UnaryLogPlusEps(nn.Module):
    """``log(x + eps)`` with fixed scalar ``eps`` (LayerNorm-style floor), exact ``torch.log``."""

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.register_buffer("eps", torch.tensor(float(eps)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.eps.to(device=x.device, dtype=x.dtype)
        return torch.log(x + e)


class UnarySquare(nn.Module):
    """Unary square ``x * x`` (e.g. variance and (a+b)²−(a−b)² identities)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class UnaryRsqrtPlusEps(nn.Module):
    """``1/sqrt(x + eps)`` via ``torch.rsqrt`` (e.g. LayerNorm inv-std from variance)."""

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.register_buffer("eps", torch.tensor(float(eps)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.eps.to(device=x.device, dtype=x.dtype)
        return torch.rsqrt(x + e)


class UnaryReciprocalPlusEps(nn.Module):
    """``1 / (x + eps)`` (pair with :class:`MatMulHadamard`, not ``/``)."""

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.register_buffer("eps", torch.tensor(float(eps)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.eps.to(device=x.device, dtype=x.dtype)
        return torch.reciprocal(x + e)


class UnarySum(nn.Module):
    """Sum reduction (unlike :class:`UnaryMean`, no ``1/n`` scaling)."""

    def __init__(self, dim: int, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=self.dim, keepdim=self.keepdim)


class UnaryExp(nn.Module):
    """``torch.exp`` (Gibbs softmax, LN magnitude, etc.)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)


class UnarySqrtExp(nn.Module):
    """``sqrt(exp(x))`` — use in strict LN as ``SqrtExp(2·log|u| - log(r2))`` instead of ``exp(x - ½·y)``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.exp(x))


class AffineFixedMix(nn.Module):
    """
    Fixed 2×2 linear mix on the operand channel:
    ``y[...,p,:] = Σ_q M[p,q] * x[...,q,:]`` via ``torch.einsum`` (default ``M`` builds plus/minus).
    Exposed as a submodule so surgery graphs list an explicit affine node, like LN submodules.
    """

    def __init__(
        self,
        einsum_equation: str,
        matrix: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.einsum_equation = einsum_equation
        if matrix is None:
            matrix = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=get_surgery_dtype())
        self.register_buffer("weight", matrix.clone().detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(device=x.device, dtype=x.dtype)
        return torch.einsum(self.einsum_equation, w, x)


class AffineContract(nn.Module):
    """
    Contract operand / head dims with a fixed coefficient vector (e.g. ``(w,−w)`` for scaled
    ``plus²−minus²``). One graph node for the output affine of the square-identity chain.
    """

    def __init__(self, einsum_equation: str, coeffs: torch.Tensor) -> None:
        super().__init__()
        self.einsum_equation = einsum_equation
        self.register_buffer("coeff", coeffs.clone().detach().float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.coeff.to(device=x.device, dtype=x.dtype)
        return torch.einsum(self.einsum_equation, c, x)


class SquareIdentityOperandChain(nn.Module):
    """
    ``((a+b)²-(a-b)²)/4`` implemented as ``torch.stack((a,b), dim=-2)`` (operand layout only) →
    fixed 2×2 mix → :class:`UnarySquare` → coeff contraction. :class:`PairwiseDotBySquare` and
    :class:`SparseWeightedSumBySquare` share this chain; they differ only in how ``a`` and ``b`` are
    formed (QK broadcast vs gathered ``p``/``v``) and in the ``einsum`` equations / coefficient
    vector (attention includes ``1/√d``, value mix does not).
    """

    def __init__(
        self,
        mix_einsum: str,
        contract_einsum: str,
        coeffs: torch.Tensor,
    ) -> None:
        super().__init__()
        self.operand_mix = AffineFixedMix(mix_einsum)
        self.square = UnarySquare()
        self.out_contract = AffineContract(contract_einsum, coeffs)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack((a, b), dim=-2)
        pm = self.operand_mix(stacked)
        pm = pm.to(torch.result_type(a, b))
        sq = self.square(pm)
        return self.out_contract(sq)


class UnaryMean(nn.Module):
    """Mean as sum followed by fixed scalar ``1/n``."""

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


# ---------------------------------------------------------------------------
# LayerNorm rewrite: relu± stack + affine reductions + UnarySqrtExp (no log PWL)
# ---------------------------------------------------------------------------


class RewrittenLayerNorm(nn.Module):
    """
    mu = mean(x); u = x - mu; r2 = mean(u*u).

    **``allow_matmul=False`` (default, strict):** ``au = stack(relu(u),relu(-u))`` (dim ``-2``);
    ``log_num = log_eps(au)`` (operand ``p``); ``log_den = log_eps(r2)`` broadcast;
    ``t = AffineContract([2,-1])(stack(log_num, log_den))`` (= ``2·log_num - log_den``);
    ``a_mag = UnarySqrtExp(t)`` (= ``sqrt(exp(t))``, same value as ``exp(log_num - ½·log_den)``);
    ``z = out_contract(a_mag)``.

    **``allow_matmul=True`` (debug / fast):** same ``r2``, then ``z = u * inv_std`` with
    ``inv_std = 1/sqrt(r2+eps)`` via :class:`UnaryRsqrtPlusEps` and :class:`MatMulHadamard`, not the
    log-domain path.
    """

    def __init__(self, normalized_shape: int, eps: float, *, allow_matmul: bool = False) -> None:
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        e = float(eps)
        self.allow_matmul = allow_matmul
        self.mean_u = UnaryMean(-1, keepdim=True)
        self.mean_r2 = UnaryMean(-1, keepdim=True)
        self.square = UnarySquare()
        self.u_center_contract = AffineContract(
            "i,...i->...",
            torch.tensor([1.0, -1.0], dtype=get_surgery_dtype()),
        )
        self.affine = AffineScaleBias(normalized_shape)
        if allow_matmul:
            self.inv_sqrt_var = UnaryRsqrtPlusEps(e)
            self.u_mul_invstd = MatMulHadamard()
        else:
            self.out_contract = AffineContract("p,...pc->...c", torch.tensor([1.0, -1.0]))
            self.log_a_contract = AffineContract("q,...qpc->...pc", torch.tensor([2.0, -1.0]))
            self.log_eps = UnaryLogPlusEps(e)
            self.sqrt_exp = UnarySqrtExp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = self.mean_u(x)
        _x, _m = torch.broadcast_tensors(x, mu)
        u = self.u_center_contract(torch.stack((_x, _m), dim=-1))
        u2 = self.square(u)
        r2 = self.mean_r2(u2)
        if self.allow_matmul:
            inv_std = self.inv_sqrt_var(r2)
            z = self.u_mul_invstd(u, inv_std)
        else:
            au = torch.stack((F.relu(u), F.relu(-u)), dim=-2)
            log_num = self.log_eps(au)
            log_den = self.log_eps(r2).unsqueeze(-2).expand_as(log_num)
            lin_log = self.log_a_contract(torch.stack((log_num, log_den), dim=2))
            a_mag = self.sqrt_exp(lin_log)
            z = self.out_contract(a_mag)
        return self.affine(z)


def copy_ln_params_to_rewritten(dst: RewrittenLayerNorm, src: nn.LayerNorm) -> None:
    """
    Copy ``gamma``/``beta`` from a timm ``LayerNorm`` into :class:`RewrittenLayerNorm`.

    **Does not** copy ``src.eps`` into ``dst``: ``dst.log_eps.eps`` is fixed at construction from
    ``eps_ln`` (run config). Weight and bias are copied from the reference checkpoint.
    """
    with torch.no_grad():
        dst.affine.weight.copy_(torch.nan_to_num(src.weight.detach(), nan=1.0, posinf=1.0, neginf=1.0))
        dst.affine.bias.copy_(torch.nan_to_num(src.bias.detach(), nan=0.0, posinf=0.0, neginf=0.0))


# ---------------------------------------------------------------------------
# Pairwise dot via square identity
# ---------------------------------------------------------------------------


class PairwiseDotBySquare(nn.Module):
    """
    scores[b,h,i,j] = (1/sqrt(d)) * sum_l q[b,h,i,l] * k[b,h,j,l]

    **Demonstration / plan.md form (default):** uses :class:`SquareIdentityOperandChain` with
    mix ``pq,...qd→...pd``, contract ``p,...pd→...``, coeffs ``±1/(4√d)``. Operands ``a,b`` are the
    broadcast Q/K grid. **No** ``torch.matmul`` for QKᵀ.

    Optional ``allow_matmul=True`` replaces this with fused ``(q/sqrt(d)) @ kᵀ`` for
    speed / debugging only; that path **does** use matrix multiply and is not valid for the
    strict surgery demonstration.
    """

    def __init__(self, head_dim: int, *, allow_matmul: bool = False) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.allow_matmul = allow_matmul
        if allow_matmul:
            self.q_scale = UnaryScale(1.0 / math.sqrt(float(head_dim)))
            self.qk_matmul = MatMul()
        else:
            w = 0.25 / math.sqrt(float(head_dim))
            self.square_chain = SquareIdentityOperandChain(
                "pq,...qd->...pd",
                "p,...pd->...",
                torch.tensor([w, -w], dtype=get_surgery_dtype()),
            )

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        if self.allow_matmul:
            qs = self.q_scale(q)
            return self.qk_matmul(qs, k.transpose(-2, -1))
        qe = q.unsqueeze(3)
        ke = k.unsqueeze(2)
        nq = q.shape[2]
        nk = k.shape[2]
        qe_b = qe.expand(-1, -1, -1, nk, -1)
        ke_b = ke.expand(-1, -1, nq, -1, -1)
        return self.square_chain(qe_b, ke_b)


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
    Sparse Gibbs Top-K with replicated tail at s_(k).
    Returns per-row: sparse probs on idx, tail mass scalar q_tail, and idx.

    Normalization never uses the ``/`` operator: with ``allow_matmul=True`` use
    :class:`UnaryReciprocalPlusEps` and :class:`MatMulHadamard`; with
    ``allow_matmul=False`` use ``exp(vals - log(z_tail + eps))`` (same math, no division).

    **z_tail**: ``sum_k exp(val_k) + (N_k - K) * exp(s_K)`` via :class:`UnarySum`,
    tail mass as :class:`MatMulHadamard` (``allow_matmul=True``) or :class:`UnaryScale` with
    fixed ``(N_k - K)`` from ``seq_len``/``top_k`` (strict — no Hadamard in graph), then
    ``stack`` (routing) and :class:`AffineContract` ``(1,1)`` on the last dim.

    Only subgraphs for the chosen ``allow_matmul`` mode are registered (no unused children).

    ``eps`` is the same floor as LayerNorm / run config: ``log(z_tail + eps)`` and ``1/(z_tail + eps)``.
    """

    def __init__(
        self,
        seq_len: int,
        top_k: int,
        *,
        eps: float,
        allow_matmul: bool = False,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.allow_matmul = allow_matmul
        e = float(eps)
        self.exp = UnaryExp()
        self.sum_exp_vals = UnarySum(-1, keepdim=True)
        self.z_tail_contract = AffineContract(
            "i,...i->...",
            torch.tensor([1.0, 1.0], dtype=get_surgery_dtype()),
        )
        self.scores_stable_contract = AffineContract(
            "i,...i->...",
            torch.tensor([1.0, -1.0], dtype=get_surgery_dtype()),
        )
        if allow_matmul:
            self.tail_mass_mul = MatMulHadamard()
            self.inv_z = UnaryReciprocalPlusEps(e)
            self.mul_by_inv_z = MatMulHadamard()
        else:
            nk = seq_len
            kk = min(top_k, nk)
            self.tail_mass_mul = UnaryScale(float(max(0, nk - kk)))
            self.log_z = UnaryLogPlusEps(e)
            self.logit_logz_contract = AffineContract(
                "i,...i->...",
                torch.tensor([1.0, 1.0], dtype=get_surgery_dtype()),
            )

    def forward(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        scores: [B, H, Nq, Nk]
        returns probs [B,H,Nq,K], idx [B,H,Nq,K], q_tail [B,H,Nq,1]
        """
        _b, _h, _nq, nk = scores.shape
        k = min(self.top_k, nk)
        tail_coeff = float(nk - k)
        row_max = scores.max(dim=-1, keepdim=True).values
        _s, _r = torch.broadcast_tensors(scores, row_max)
        scores_stable = self.scores_stable_contract(torch.stack((_s, _r), dim=-1))
        vals, idx = torch.topk(scores_stable, k=k, dim=-1, largest=True, sorted=True)
        s_k = vals[..., -1:]
        exp_vals = self.exp(vals)
        exp_tail = self.exp(s_k)
        sum_exp = self.sum_exp_vals(exp_vals)
        if self.allow_matmul:
            tail_term = self.tail_mass_mul(exp_tail, torch.full_like(exp_tail, tail_coeff))
        else:
            tail_term = self.tail_mass_mul(exp_tail)
        _se, _tt = torch.broadcast_tensors(sum_exp, tail_term)
        z_tail = self.z_tail_contract(torch.stack((_se, _tt), dim=-1))

        if self.allow_matmul:
            inv_z = self.inv_z(z_tail)
            probs = self.mul_by_inv_z(exp_vals, inv_z)
            q_tail = self.mul_by_inv_z(tail_term, inv_z)
        else:
            log_z = self.log_z(z_tail)
            neg_log_z = -log_z
            _v, _nlz = torch.broadcast_tensors(vals, neg_log_z)
            logits_norm = self.logit_logz_contract(torch.stack((_v, _nlz), dim=-1))
            probs = self.exp(logits_norm)
            _sk, _nlz2 = torch.broadcast_tensors(s_k, neg_log_z)
            sk_norm = self.logit_logz_contract(torch.stack((_sk, _nlz2), dim=-1))
            q_tail = self.tail_mass_mul(self.exp(sk_norm))

        return probs, idx, q_tail


# ---------------------------------------------------------------------------
# Sparse weighted sum via square identity
# ---------------------------------------------------------------------------


class SparseWeightedSumBySquare(nn.Module):
    """
    y[b,h,nq,d] = sum_{k in top} p[b,h,nq,k] * v[b,h, idx[b,h,nq,k], d]

    **Default (plan.md):** uses :class:`SquareIdentityOperandChain` with mix ``pq,...kqd→...kpd``,
    contract ``p,...kpd→...d``, coeffs ``(¼,−¼)``. Operands ``a,b`` are expanded prob and gathered
    value per top-k slot. No dense ``matmul`` for attention-value mixing.

    ``allow_matmul=True`` uses ``p * v`` then sum (faster; invalid for strict demo).
    """

    def __init__(self, *, allow_matmul: bool = False) -> None:
        super().__init__()
        self.allow_matmul = allow_matmul
        if allow_matmul:
            self.pv_matmul = MatMul()
        else:
            self.square_chain = SquareIdentityOperandChain(
                "pq,...kqd->...kpd",
                "p,...kpd->...d",
                torch.tensor([0.25, -0.25], dtype=get_surgery_dtype()),
            )

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
        if self.allow_matmul:
            # [B,H,Nq,1,K] @ [B,H,Nq,K,D] -> [B,H,Nq,1,D]
            return self.pv_matmul(probs.unsqueeze(-2), v_g).squeeze(-2)
        p = probs.unsqueeze(-1)
        p_b = p.expand(-1, -1, -1, -1, d)
        return self.square_chain(p_b, v_g)


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


def _jeffreys_metric_eps(dt: torch.dtype) -> Tuple[float, float]:
    """``(kl_clamp, denom_add)`` for Jeffreys helpers: scalars must survive ``dt`` (fp16 cannot hold 1e-30)."""
    f = torch.finfo(dt)
    smn = float(f.smallest_normal)
    sms = float(getattr(f, "smallest_subnormal", smn))
    return (max(1e-12, sms), max(1e-30, smn))


def _kl_safe(p: torch.Tensor, q: torch.Tensor, eps: Optional[float] = None) -> torch.Tensor:
    if eps is None:
        eps, _ = _jeffreys_metric_eps(p.dtype)
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
    loss_dtype = torch.promote_types(teacher_logits.dtype, student_logits.dtype)
    if loss_dtype in (torch.float16, torch.bfloat16):
        loss_dtype = torch.float32
    t = teacher_logits.to(dtype=loss_dtype) / temperature
    s = student_logits.to(dtype=loss_dtype) / temperature
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
    _, denom_eps = _jeffreys_metric_eps(teacher_logits.dtype)

    s_k = vals[:, -1:]
    exp_vals = torch.exp(vals)
    exp_tail = torch.exp(s_k)
    z_tail = exp_vals.sum(dim=-1, keepdim=True) + float(nk - k) * exp_tail
    q_on_I = exp_vals / (z_tail + denom_eps)

    r = teacher_logits.shape[0]
    device = teacher_logits.device
    q_tail_each = exp_tail / (z_tail + denom_eps)
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
    kl_eps, denom_eps = _jeffreys_metric_eps(teacher_logits.dtype)
    exp_vals = torch.exp(vals)
    z_sub = exp_vals.sum(dim=-1, keepdim=True)
    q_sub = exp_vals / (z_sub + denom_eps)
    q_dense = torch.zeros_like(p)
    q_dense.scatter_(1, idx, q_sub)
    j = _kl_safe(p, q_dense.clamp_min(kl_eps)) + _kl_safe(q_dense.clamp_min(kl_eps), p)
    return j


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


def build_surgery_pwl_meta() -> Dict[str, Any]:
    """
    JSON-safe summary of **live** PWL in the surgery model. Gibbs / LN use exact ``exp`` /
    ``log`` on buffers — no PWL grids there. Only :class:`GELUUnaryPWL` uses scalar PWL; knot
    positions match its default (values are trainable parameters in the checkpoint).
    """
    k = torch.linspace(-4.0, 4.0, PWL_NUM_KNOTS)
    return {
        "gelu_mlp": {
            "knot_positions": [float(x) for x in k],
            "num_knots": int(PWL_NUM_KNOTS),
            "note": (
                "Default knot x-positions for GELUUnaryPWL; knot values are "
                "``blocks.*.mlp.act.pwl.values`` in the state dict."
            ),
        },
    }


def _forward_output_shape_str(out: Any) -> str:
    if torch.is_tensor(out):
        return str(tuple(out.shape))
    if isinstance(out, (tuple, list)):
        return "(" + ", ".join(_forward_output_shape_str(x) for x in out) + ")"
    return type(out).__name__


def write_model_structure_txt(
    path: str,
    model: nn.Module,
    title: str,
    *,
    example_input: Optional[torch.Tensor] = None,
    default_input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    include_forward_shapes: bool = True,
) -> None:
    """
    Write model repr, parameter counts, ``named_modules`` listing, and (by default) per-module
    forward **output** tensor shapes from one ``eval`` pass with a dummy batch (for ``artifacts/logs`` dumps).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines: List[str] = [
        title,
        "=" * min(80, max(len(title), 40)),
        f"class: {type(model).__name__}",
        f"parameters: total={n_all:,} trainable={n_train:,}",
        "",
        str(model),
        "",
        "--- named_modules (name: class) ---",
        "",
    ]
    for name, mod in model.named_modules():
        lines.append(f"{name if name else '<root>'}: {type(mod).__name__}")

    if include_forward_shapes:
        out_shapes: Dict[str, str] = {}
        shape_err: Optional[str] = None
        x_log: Optional[torch.Tensor] = None
        try:
            try:
                device = next(model.parameters()).device
                dtype = next(model.parameters()).dtype
            except StopIteration:
                device = torch.device("cpu")
                dtype = get_surgery_dtype()
            x = example_input
            if x is None:
                x = torch.zeros(default_input_shape, device=device, dtype=dtype)
            else:
                x = x.to(device=device, dtype=dtype)
            x_log = x

            hooks: List[Any] = []

            def _make_hook(key: str):
                def _hook(_mod: nn.Module, _inp: Any, out: Any) -> None:
                    out_shapes[key] = _forward_output_shape_str(out)

                return _hook

            for name, mod in model.named_modules():
                key = name if name else "<root>"
                hooks.append(mod.register_forward_hook(_make_hook(key)))

            was_training = model.training
            model.eval()
            with torch.no_grad():
                model(x)
            if was_training:
                model.train()
            for h in hooks:
                h.remove()
        except Exception as ex:
            shape_err = repr(ex)

        lines.extend(
            [
                "",
                "--- forward output shapes (one eval batch; dummy input unless example_input set) ---",
            ]
        )
        if shape_err is not None:
            lines.append(f"(forward shape trace failed: {shape_err})")
        elif x_log is not None:
            lines.append(
                f"example_input: {tuple(x_log.shape)}  dtype={x_log.dtype}  device={x_log.device}"
            )
            lines.append("")
            for name, _mod in model.named_modules():
                key = name if name else "<root>"
                label = name if name else "<root>"
                lines.append(f"{label}: {out_shapes.get(key, '—')}")
        else:
            lines.append("(no example batch; shapes skipped)")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


@dataclass
class SurgeryMeta:
    model_key: str = "deit_tiny_pet"
    patient: str = "DeiT-Tiny"
    dataset: str = "Oxford-IIIT Pet"
    eps: float = 1e-5
    top_k: int = 32
    surgery_dtype: str = "bfloat16"
    pwl: Dict[str, Any] = field(default_factory=dict)
    calibration: Dict[str, float] = field(default_factory=dict)
    module_mapping: Dict[str, str] = field(default_factory=dict)
    reference_checkpoint: str = ""
    allow_matmul: bool = False

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_key": self.model_key,
                    "patient": self.patient,
                    "dataset": self.dataset,
                    "eps": self.eps,
                    "top_k": self.top_k,
                    "surgery_dtype": self.surgery_dtype,
                    "pwl": self.pwl,
                    "calibration": self.calibration,
                    "calibration_legend": CALIBRATION_LEGEND_TEXT,
                    "module_mapping": self.module_mapping,
                    "reference_checkpoint": self.reference_checkpoint,
                    "allow_matmul": self.allow_matmul,
                },
                f,
                indent=2,
            )


def register_calibration_stats(meta: SurgeryMeta, key: str, value: float) -> None:
    meta.calibration[key] = float(value)
