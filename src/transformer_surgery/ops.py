"""
Discovery, calibration, replacement helpers, validation metrics, and checkpoint export
for the DeiT-Tiny pseudo-hardware surgery graph.
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
from timm.layers import DropPath as _TimmDropPath

# Default number of PWL knots (log, exp, GELU, Gibbs exp epilogue); cap at 17.
PWL_NUM_KNOTS = 17

# Process-wide dtype for surgery tensor literals and ``.to(dtype=...)`` (set via ``apply_dtype_from_config``).
_SURGERY_DTYPE: torch.dtype = torch.bfloat16


def get_surgery_dtype() -> torch.dtype:
    """Current surgery compute dtype (default ``bfloat16``; set with :func:`set_surgery_dtype`)."""
    return _SURGERY_DTYPE


def set_surgery_dtype(dt: torch.dtype) -> None:
    """Set global surgery dtype (mirrors the process device pattern in ``transformer_surgery.util``)."""
    global _SURGERY_DTYPE
    _SURGERY_DTYPE = dt


# ---------------------------------------------------------------------------
# Routing vocabulary: aliases to torch / F / nn ops that carry no parameters and do no
# arithmetic — pure tensor wiring and discrete selection. Defined here, before the op classes,
# so internal call sites in this module use ``Routing*`` names rather than ``torch.*``/``F.*``
# directly. External callers may import them to make the routing surface explicit.
# ---------------------------------------------------------------------------

RoutingReshape = torch.reshape
RoutingTranspose = torch.transpose
RoutingCat = torch.cat
RoutingStack = torch.stack
RoutingUnsqueeze = torch.unsqueeze
RoutingBroadcastTensors = torch.broadcast_tensors
RoutingTopK = torch.topk
RoutingGather = torch.gather
RoutingMax = torch.max
RoutingFullLike = torch.full_like
RoutingReLU = F.relu
RoutingDropout = nn.Dropout
RoutingDropPath = _TimmDropPath


def RoutingExpand(x: torch.Tensor, *sizes: int) -> torch.Tensor:
    return x.expand(*sizes)


def RoutingExpandAs(x: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return x.expand_as(other)


def RoutingSqueeze(x: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    return x.squeeze() if dim is None else x.squeeze(dim)


# ---------------------------------------------------------------------------
# Op vocabulary: three groups.
#   Affine*  — linear in each operand: parameterized layers, fixed-coeff einsum, per-channel
#              scale+bias, fixed 2x2 mixes, linear unary reductions/scalings, and the variable
#              bilinear ops (AffineMatMul, AffineHadamard) gated by allow_matmul.
#   NL*      — strictly nonlinear scalar maps (square, exp, log+eps, sqrt-exp, rsqrt+eps,
#              reciprocal+eps, scalar PWL, GELU-as-PWL).
#   Routing* — pure tensor wiring and discrete selection (reshape/transpose/cat/stack/expand/
#              squeeze/broadcast_tensors/topk/gather/full_like, F.relu, torch.max, Dropout/DropPath); see the
#              ``Routing*`` aliases above.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Affine-style explicit combinators
# ---------------------------------------------------------------------------


class AffineScale(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(float(scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.scale
        if s.device != x.device:
            s = s.to(device=x.device)
        return x * s.to(dtype=x.dtype)


class AffineHadamard(nn.Module):
    """Elementwise / broadcast product ``a * b``; both operands are tensors (bilinear, no ``@``)."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b


class AffineMatMul(nn.Module):
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


class NLLogPlusEps(nn.Module):
    """``log(x + eps)`` with fixed scalar ``eps`` (LayerNorm-style floor), exact ``torch.log``."""

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.register_buffer("eps", torch.tensor(float(eps)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.eps.to(device=x.device, dtype=x.dtype)
        return torch.log(x + e)


class NLSquare(nn.Module):
    """Unary square ``x * x`` (e.g. variance and (a+b)²−(a−b)² identities)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class NLRsqrtPlusEps(nn.Module):
    """``1/sqrt(x + eps)`` via ``torch.rsqrt`` (e.g. LayerNorm inv-std from variance)."""

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.register_buffer("eps", torch.tensor(float(eps)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.eps.to(device=x.device, dtype=x.dtype)
        return torch.rsqrt(x + e)


class NLReciprocalPlusEps(nn.Module):
    """``1 / (x + eps)`` (pair with :class:`AffineHadamard`, not ``/``)."""

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.register_buffer("eps", torch.tensor(float(eps)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.eps.to(device=x.device, dtype=x.dtype)
        return torch.reciprocal(x + e)


class AffineSum(nn.Module):
    """Sum reduction (unlike :class:`AffineMean`, no ``1/n`` scaling)."""

    def __init__(self, dim: int, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=self.dim, keepdim=self.keepdim)


class NLExp(nn.Module):
    """``torch.exp`` (Gibbs softmax, LN magnitude, etc.)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)


class NLSqrtExp(nn.Module):
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
    fixed 2×2 mix → :class:`NLSquare` → coeff contraction. :class:`PairwiseDotBySquare` and
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
        self.square = NLSquare()
        self.out_contract = AffineContract(contract_einsum, coeffs)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        stacked = RoutingStack((a, b), dim=-2)
        pm = self.operand_mix(stacked)
        pm = pm.to(torch.result_type(a, b))
        sq = self.square(pm)
        return self.out_contract(sq)


class AffineMean(nn.Module):
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


class NLScalarPWL(nn.Module):
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
# LayerNorm rewrite: relu± stack + affine reductions + NLSqrtExp (no log PWL)
# ---------------------------------------------------------------------------


class RewrittenLayerNorm(nn.Module):
    """
    mu = mean(x); u = x - mu; r2 = mean(u*u).

    **``allow_matmul=False`` (default, strict):** ``au = stack(relu(u),relu(-u))`` (dim ``-2``);
    ``log_num = log_eps(au)`` (operand ``p``); ``log_den = log_eps(r2)`` broadcast;
    ``t = AffineContract([2,-1])(stack(log_num, log_den))`` (= ``2·log_num - log_den``);
    ``a_mag = NLSqrtExp(t)`` (= ``sqrt(exp(t))``, same value as ``exp(log_num - ½·log_den)``);
    ``z = out_contract(a_mag)``.

    **``allow_matmul=True`` (debug / fast):** same ``r2``, then ``z = u * inv_std`` with
    ``inv_std = 1/sqrt(r2+eps)`` via :class:`NLRsqrtPlusEps` and :class:`AffineHadamard`, not the
    log-domain path.
    """

    def __init__(self, normalized_shape: int, eps: float, *, allow_matmul: bool = False) -> None:
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        e = float(eps)
        self.allow_matmul = allow_matmul
        self.mean_u = AffineMean(-1, keepdim=True)
        self.mean_r2 = AffineMean(-1, keepdim=True)
        self.square = NLSquare()
        self.u_center_contract = AffineContract(
            "i,...i->...",
            torch.tensor([1.0, -1.0], dtype=get_surgery_dtype()),
        )
        self.affine = AffineScaleBias(normalized_shape)
        if allow_matmul:
            self.inv_sqrt_var = NLRsqrtPlusEps(e)
            self.u_mul_invstd = AffineHadamard()
        else:
            self.out_contract = AffineContract("p,...pc->...c", torch.tensor([1.0, -1.0]))
            self.log_a_contract = AffineContract("q,...qpc->...pc", torch.tensor([2.0, -1.0]))
            self.log_eps = NLLogPlusEps(e)
            self.sqrt_exp = NLSqrtExp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = self.mean_u(x)
        _x, _m = RoutingBroadcastTensors(x, mu)
        u = self.u_center_contract(RoutingStack((_x, _m), dim=-1))
        u2 = self.square(u)
        r2 = self.mean_r2(u2)
        if self.allow_matmul:
            inv_std = self.inv_sqrt_var(r2)
            z = self.u_mul_invstd(u, inv_std)
        else:
            au = RoutingStack((RoutingReLU(u), RoutingReLU(-u)), dim=-2)
            log_num = self.log_eps(au)
            log_den = RoutingExpandAs(RoutingUnsqueeze(self.log_eps(r2), -2), log_num)
            lin_log = self.log_a_contract(RoutingStack((log_num, log_den), dim=2))
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

    **Strict form (default):** uses :class:`SquareIdentityOperandChain` with
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
            self.q_scale = AffineScale(1.0 / math.sqrt(float(head_dim)))
            self.qk_matmul = AffineMatMul()
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
            return self.qk_matmul(qs, RoutingTranspose(k, -2, -1))
        qe = RoutingUnsqueeze(q, 3)
        ke = RoutingUnsqueeze(k, 2)
        nq = q.shape[2]
        nk = k.shape[2]
        qe_b = RoutingExpand(qe, -1, -1, -1, nk, -1)
        ke_b = RoutingExpand(ke, -1, -1, nq, -1, -1)
        return self.square_chain(qe_b, ke_b)


# ---------------------------------------------------------------------------
# Gibbs Top-K softmax with implicit replicated tail
# ---------------------------------------------------------------------------


class GibbsTopKSoftmax(nn.Module):
    """
    Sparse Gibbs Top-K with configurable omitted-tail handling.
    Returns per-row: sparse probs on idx, tail mass scalar q_tail, and idx.

    Normalization never uses the ``/`` operator: with ``allow_matmul=True`` use
    :class:`NLReciprocalPlusEps` and :class:`AffineHadamard`; with
    ``allow_matmul=False`` use ``exp(vals - log(z_tail + eps))`` (same math, no division).

    ``tail_use_prob_eps=False`` keeps the original approximation:
    ``Z_tail = sum_k exp(val_k) + (N-K) * exp(s_K)``.

    ``tail_use_prob_eps=True`` reserves probability mass ``gibbs_tail_prob_eps`` for all omitted
    entries and scales the top-k probabilities by ``1 - gibbs_tail_prob_eps``. The value is an
    ``nn.Parameter`` initialized from config.

    Only normalization subgraphs for the chosen ``allow_matmul`` mode are registered.

    ``eps`` is the same floor as LayerNorm / run config for log/reciprocal normalizers.
    """

    def __init__(
        self,
        seq_len: int,
        top_k: int,
        *,
        eps: float,
        tail_use_prob_eps: bool,
        tail_prob_eps: float,
        allow_matmul: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.top_k = top_k
        self.tail_use_prob_eps = bool(tail_use_prob_eps)
        tail_prob = float(tail_prob_eps)
        if not 0.0 <= tail_prob < 1.0:
            raise ValueError("tail_prob_eps must be in [0, 1)")
        self.gibbs_tail_prob_eps = nn.Parameter(
            torch.tensor(tail_prob, dtype=get_surgery_dtype()),
            requires_grad=self.tail_use_prob_eps,
        )
        self.allow_matmul = allow_matmul
        e = float(eps)
        self.exp = NLExp()
        self.sum_exp_vals = AffineSum(-1, keepdim=True)
        self.scores_stable_contract = AffineContract(
            "i,...i->...",
            torch.tensor([1.0, -1.0], dtype=get_surgery_dtype()),
        )
        if not self.tail_use_prob_eps:
            k_static = min(int(top_k), self.seq_len)
            self.tail_mass_mul = AffineScale(float(max(0, self.seq_len - k_static)))
            self.z_tail_contract = AffineContract(
                "i,...i->...",
                torch.tensor([1.0, 1.0], dtype=get_surgery_dtype()),
            )
        if allow_matmul:
            self.inv_z = NLReciprocalPlusEps(e)
            self.mul_by_inv_z = AffineHadamard()
        else:
            self.log_z = NLLogPlusEps(e)
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
        row_max = RoutingMax(scores, dim=-1, keepdim=True).values
        _s, _r = RoutingBroadcastTensors(scores, row_max)
        scores_stable = self.scores_stable_contract(RoutingStack((_s, _r), dim=-1))
        vals, idx = RoutingTopK(scores_stable, k=k, dim=-1, largest=True, sorted=True)
        s_k = vals[..., -1:]
        exp_vals = self.exp(vals)
        sum_exp = self.sum_exp_vals(exp_vals)

        if not self.tail_use_prob_eps:
            exp_tail = self.exp(s_k)
            if nk > k:
                tail_term = self.tail_mass_mul(exp_tail)
            else:
                tail_term = RoutingFullLike(exp_tail, 0.0)
            _se, _tt = RoutingBroadcastTensors(sum_exp, tail_term)
            normalizer = self.z_tail_contract(RoutingStack((_se, _tt), dim=-1))
        else:
            normalizer = sum_exp

        if self.allow_matmul:
            inv_z = self.inv_z(normalizer)
            top_probs = self.mul_by_inv_z(exp_vals, inv_z)
        else:
            log_z = self.log_z(normalizer)
            neg_log_z = -log_z
            _v, _nlz = RoutingBroadcastTensors(vals, neg_log_z)
            logits_norm = self.logit_logz_contract(RoutingStack((_v, _nlz), dim=-1))
            top_probs = self.exp(logits_norm)

        if not self.tail_use_prob_eps:
            probs = top_probs
            if self.allow_matmul:
                q_tail = self.mul_by_inv_z(tail_term, inv_z)
            else:
                _sk, _nlz2 = RoutingBroadcastTensors(s_k, neg_log_z)
                sk_norm = self.logit_logz_contract(RoutingStack((_sk, _nlz2), dim=-1))
                if nk > k:
                    q_tail = self.tail_mass_mul(self.exp(sk_norm))
                else:
                    q_tail = RoutingFullLike(sk_norm, 0.0)
        elif nk > k:
            tail_prob = self.gibbs_tail_prob_eps.to(device=s_k.device, dtype=s_k.dtype).clamp(0.0, 1.0)
            probs = top_probs * (1.0 - tail_prob)
            q_tail = RoutingExpandAs(tail_prob, s_k)
        else:
            probs = top_probs
            q_tail = RoutingFullLike(s_k, 0.0)

        return probs, idx, q_tail


# ---------------------------------------------------------------------------
# Sparse weighted sum via square identity
# ---------------------------------------------------------------------------


class SparseWeightedSumBySquare(nn.Module):
    """
    y[b,h,nq,d] = sum_{k in top} p[b,h,nq,k] * v[b,h, idx[b,h,nq,k], d]

    **Strict form (default):** uses :class:`SquareIdentityOperandChain` with mix ``pq,...kqd→...kpd``,
    contract ``p,...kpd→...d``, coeffs ``(¼,−¼)``. Operands ``a,b`` are expanded prob and gathered
    value per top-k slot. No dense ``matmul`` for attention-value mixing.

    ``allow_matmul=True`` uses ``p * v`` then sum (faster; invalid for strict demo).
    """

    def __init__(self, *, allow_matmul: bool = False) -> None:
        super().__init__()
        self.allow_matmul = allow_matmul
        if allow_matmul:
            self.pv_matmul = AffineMatMul()
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
        idx_e = RoutingExpand(RoutingUnsqueeze(idx, -1), -1, -1, -1, -1, d)
        v_h = RoutingExpand(RoutingUnsqueeze(v, 2), -1, -1, nq, -1, -1)
        v_g = RoutingGather(v_h, 3, idx_e)
        if self.allow_matmul:
            # [B,H,Nq,1,K] @ [B,H,Nq,K,D] -> [B,H,Nq,1,D]
            return RoutingSqueeze(self.pv_matmul(RoutingUnsqueeze(probs, -2), v_g), -2)
        p = RoutingUnsqueeze(probs, -1)
        p_b = RoutingExpand(p, -1, -1, -1, -1, d)
        return self.square_chain(p_b, v_g)


# ---------------------------------------------------------------------------
# GELU as unary PWL (MLP nonlinearity)
# ---------------------------------------------------------------------------


class NLGELU(nn.Module):
    def __init__(self, knots: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if knots is None:
            knots = torch.linspace(-4.0, 4.0, PWL_NUM_KNOTS)
        ref = F.gelu(knots)
        self.pwl = NLScalarPWL(knots, init_values=ref)

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
    *,
    tail_use_prob_eps: bool,
    tail_prob_eps: float,
) -> torch.Tensor:
    """
    teacher_logits: [R, nk]
    vals: top-k stabilized logits (row-wise max subtracted) [R, k]
    idx: [R, k]
    Dense student q uses either the original k-th-logit replicated tail or a fixed
    omitted-tail probability mass.
    """
    use_prob_eps = bool(tail_use_prob_eps)
    tail_prob = float(tail_prob_eps)
    if not 0.0 <= tail_prob < 1.0:
        raise ValueError("tail_prob_eps must be in [0, 1)")
    t = teacher_logits - teacher_logits.max(dim=-1, keepdim=True).values
    p = F.softmax(t, dim=-1)
    _, denom_eps = _jeffreys_metric_eps(teacher_logits.dtype)

    s_k = vals[:, -1:]
    exp_vals = torch.exp(vals)
    r = teacher_logits.shape[0]
    q_dense = torch.zeros_like(p)
    if not use_prob_eps:
        exp_tail = torch.exp(s_k)
        tail_count = float(nk - k) if nk > k else 0.0
        z_tail = exp_vals.sum(dim=-1, keepdim=True) + tail_count * exp_tail
        q_on_I = exp_vals / (z_tail + denom_eps)
        if nk > k:
            q_tail_each = exp_tail / (z_tail + denom_eps)
            q_dense = RoutingExpand(q_tail_each, r, nk).clone()
    else:
        tail_mass = tail_prob if nk > k else 0.0
        z_top = exp_vals.sum(dim=-1, keepdim=True)
        q_on_I = (1.0 - tail_mass) * exp_vals / (z_top + denom_eps)
        if nk > k:
            q_dense = torch.full_like(p, tail_mass / float(nk - k))
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
    "(acc and mean CE / Jeffreys; legacy keys val_acc_post_ft retained). "
    "gibbs_tail_prob_eps_calibrated_*: observed dense-softmax omitted tail mass for top-k scores; "
    "gibbs_tail_prob_eps_applied_*: values copied into GibbsTopKSoftmax parameters when fixed-tail "
    "probability mode is active."
)


def build_surgery_pwl_meta() -> Dict[str, Any]:
    """
    JSON-safe summary of **live** PWL in the surgery model. Gibbs / LN use exact ``exp`` /
    ``log`` on buffers — no PWL grids there. Only :class:`NLGELU` uses scalar PWL; knot
    positions match its default (values are trainable parameters in the checkpoint).
    """
    k = torch.linspace(-4.0, 4.0, PWL_NUM_KNOTS)
    return {
        "gelu_mlp": {
            "knot_positions": [float(x) for x in k],
            "num_knots": int(PWL_NUM_KNOTS),
            "note": (
                "Default knot x-positions for NLGELU; knot values are "
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
    gibbs_tail_use_prob_eps: bool = False
    gibbs_tail_prob_eps: float = 1e-5
    top_k: int = 32
    surgery_dtype: str = "bfloat16"
    pwl: Dict[str, Any] = field(default_factory=dict)
    calibration: Dict[str, Any] = field(default_factory=dict)
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
                    "gibbs_tail_use_prob_eps": self.gibbs_tail_use_prob_eps,
                    "gibbs_tail_prob_eps": self.gibbs_tail_prob_eps,
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
