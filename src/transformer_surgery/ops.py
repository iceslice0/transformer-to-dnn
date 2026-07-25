"""Forward op vocabulary for the surgery graph."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath as _TimmDropPath

from transformer_surgery.internal.util import get_surgery_dtype


# ---------------------------------------------------------------------------
# Routing vocabulary: aliases to torch / F / nn ops that carry no parameters and do no
# arithmetic - pure tensor wiring and discrete selection. Defined here, before the op classes,
# so internal call sites in this module use ``Routing*`` names rather than ``torch.*``/``F.*``
# directly. External callers may import them to make the routing surface explicit.
# ---------------------------------------------------------------------------

RoutingTranspose = torch.transpose
RoutingCat = torch.cat
RoutingStack = torch.stack
RoutingUnsqueeze = torch.unsqueeze
RoutingBroadcastTensors = torch.broadcast_tensors
RoutingTopK = torch.topk
RoutingGather = torch.gather
RoutingFullLike = torch.full_like
RoutingReLU = F.relu
RoutingDropPath = _TimmDropPath


def RoutingExpand(x: torch.Tensor, *sizes: int) -> torch.Tensor:
    return x.expand(*sizes)


def RoutingExpandAs(x: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return x.expand_as(other)


def RoutingSqueeze(x: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    return x.squeeze() if dim is None else x.squeeze(dim)


# ---------------------------------------------------------------------------
# Op vocabulary: three groups.
#   Affine*  - linear in each operand: parameterized layers, fixed-coeff einsum, per-channel
#              scale+bias, fixed 2x2 mixes, linear unary reductions/scalings, and the variable
#              bilinear ops (AffineMatMul, AffineHadamard) gated by allow_matmul.
#   NL*      - strictly nonlinear scalar maps (square, exp, log+eps, sqrt-exp, rsqrt+eps,
#              reciprocal+eps, GELU).
#   Routing* - pure tensor wiring and discrete selection (reshape/transpose/cat/stack/expand/
#              squeeze/broadcast_tensors/topk/gather/full_like, F.relu, Dropout/DropPath); see the
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
    """Unary square ``x * x``."""

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
    """``sqrt(exp(x))``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.exp(x))


class AffineFixedMix(nn.Module):
    """
    Fixed 2x2 linear mix on the operand channel:
    ``y[...,p,:] = sum_q M[p,q] * x[...,q,:]`` via ``torch.einsum``.
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
    Contract operand / head dims with a fixed coefficient vector.
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
    ``((a+b)^2-(a-b)^2)/4`` as operand stack, fixed 2x2 mix, square, coeff contraction.
    :class:`PairwiseDotBySquare` and
    :class:`SparseWeightedSumBySquare` share this chain; they differ only in how ``a`` and ``b`` are
    formed (QK broadcast vs gathered ``p``/``v``) and in the ``einsum`` equations / coefficient
    vector (attention includes ``1/sqrt(d)``, value mix does not).
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
# LayerNorm rewrite: relu+/- stack + affine reductions + NLSqrtExp
# ---------------------------------------------------------------------------


class RewrittenLayerNorm(nn.Module):
    """
    mu = mean(x); u = x - mu; r2 = mean(u*u).

    **``allow_matmul=False`` (default, strict):** ``au = stack(relu(u),relu(-u))`` (dim ``-2``);
    ``log_num = log_eps(au)`` (operand ``p``); ``log_den = log_eps(r2)`` broadcast;
    ``t = AffineContract([2,-1])(stack(log_num, log_den))``.
    ``a_mag = NLSqrtExp(t)``.
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


# ---------------------------------------------------------------------------
# Pairwise dot via square identity
# ---------------------------------------------------------------------------


class PairwiseDotBySquare(nn.Module):
    """
    scores[b,h,i,j] = (1/sqrt(d)) * sum_l q[b,h,i,l] * k[b,h,j,l]

    **Strict form (default):** uses :class:`SquareIdentityOperandChain` with
    mix ``pq,...qd->...pd``, contract ``p,...pd->...``. Operands are the broadcast Q/K grid.

    Optional ``allow_matmul=True`` replaces this with fused ``(q/sqrt(d)) @ k.T`` for
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
# Gibbs Top-K softmax with fixed omitted-tail probability
# ---------------------------------------------------------------------------


class GibbsTopKSoftmax(nn.Module):
    """
    Sparse Gibbs Top-K with omitted-tail probability mass.
    Returns per-row: sparse probs on idx, tail mass scalar q_tail, and idx.

    Normalization never uses the ``/`` operator: with ``allow_matmul=True`` use
    :class:`NLReciprocalPlusEps` and :class:`AffineHadamard`; with
    ``allow_matmul=False`` use ``exp(vals - log(sum_exp + eps))`` (same math, no division).

    By default ``gibbs_tail_prob_eps`` reserves probability mass for all omitted entries and scales
    the top-k probabilities by ``1 - gibbs_tail_prob_eps``. The value is an ``nn.Parameter``
    initialized from config, then typically overwritten by surgery calibration. The scale is applied
    to top-k probabilities through an explicit ``AffineHadamard`` module rather than a bare tensor
    multiply in ``forward``.

    With ``use_exact_tail_mass=True``, ``q_tail`` is the exact dense-softmax omitted mass computed at
    runtime via the centroid partition trick: ``Z_all = N * mean(exp(scores - row_max))``,
    ``q_tail = 1 - Z_top / Z_all``. Omitted keys are never gathered; the mean over all keys is the
    centroid. The calibrated ``gibbs_tail_prob_eps`` parameter is unused in that mode.

    Only normalization subgraphs for the chosen ``allow_matmul`` / exact-tail mode are registered.

    ``eps`` is the same floor as LayerNorm / run config for log/reciprocal normalizers.
    """

    def __init__(
        self,
        seq_len: int,
        top_k: int,
        *,
        eps: float,
        gibbs_tail_prob_eps: float,
        allow_matmul: bool = False,
        use_exact_tail_mass: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.top_k = top_k
        self.use_exact_tail_mass = bool(use_exact_tail_mass)
        tail_prob = float(gibbs_tail_prob_eps)
        if not 0.0 <= tail_prob < 1.0:
            raise ValueError("gibbs_tail_prob_eps must be in [0, 1)")
        self.gibbs_tail_prob_eps = nn.Parameter(
            torch.tensor(tail_prob, dtype=get_surgery_dtype()),
        )
        self.allow_matmul = allow_matmul
        e = float(eps)
        self.exp = NLExp()
        self.sum_exp_vals = AffineSum(-1, keepdim=True)
        self.top_vals_stable_contract = AffineContract(
            "i,...i->...",
            torch.tensor([1.0, -1.0], dtype=get_surgery_dtype()),
        )
        self.scale_top_probs_by_tail = AffineHadamard()
        if allow_matmul:
            self.inv_z = NLReciprocalPlusEps(e)
            self.mul_by_inv_z = AffineHadamard()
        else:
            self.log_z = NLLogPlusEps(e)
            self.logit_logz_contract = AffineContract(
                "i,...i->...",
                torch.tensor([1.0, 1.0], dtype=get_surgery_dtype()),
            )
        if self.use_exact_tail_mass:
            self.all_vals_stable_contract = AffineContract(
                "i,...i->...",
                torch.tensor([1.0, -1.0], dtype=get_surgery_dtype()),
            )
            self.mean_exp_all = AffineMean(dim=-1, keepdim=True)
            self.scale_z_all = AffineScale(float(self.seq_len))
            self.one_minus_top_mass = AffineContract(
                "i,...i->...",
                torch.tensor([1.0, -1.0], dtype=get_surgery_dtype()),
            )
            if allow_matmul:
                self.inv_z_all = NLReciprocalPlusEps(e)
                self.mul_ztop_inv_zall = AffineHadamard()
            else:
                self.log_z_all = NLLogPlusEps(e)
                self.log_z_top = NLLogPlusEps(e)
                self.top_mass_log_contract = AffineContract(
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
        raw_vals, idx = RoutingTopK(scores, k=k, dim=-1, largest=True, sorted=True)
        row_max = raw_vals[..., :1]
        _v, _r = RoutingBroadcastTensors(raw_vals, row_max)
        vals = self.top_vals_stable_contract(RoutingStack((_v, _r), dim=-1))
        s_k = vals[..., -1:]
        exp_vals = self.exp(vals)
        normalizer = self.sum_exp_vals(exp_vals)

        if self.allow_matmul:
            inv_z = self.inv_z(normalizer)
            top_probs = self.mul_by_inv_z(exp_vals, inv_z)
        else:
            log_z = self.log_z(normalizer)
            neg_log_z = -log_z
            _v, _nlz = RoutingBroadcastTensors(vals, neg_log_z)
            logits_norm = self.logit_logz_contract(RoutingStack((_v, _nlz), dim=-1))
            top_probs = self.exp(logits_norm)

        if nk > k:
            if self.use_exact_tail_mass:
                _s, _rm = RoutingBroadcastTensors(scores, row_max)
                scores_stable = self.all_vals_stable_contract(RoutingStack((_s, _rm), dim=-1))
                exp_all = self.exp(scores_stable)
                z_all = self.scale_z_all(self.mean_exp_all(exp_all))
                if self.allow_matmul:
                    top_mass = self.mul_ztop_inv_zall(normalizer, self.inv_z_all(z_all))
                else:
                    log_z_top = self.log_z_top(normalizer)
                    neg_log_z_all = -self.log_z_all(z_all)
                    _lt, _nlza = RoutingBroadcastTensors(log_z_top, neg_log_z_all)
                    top_mass = self.exp(self.top_mass_log_contract(RoutingStack((_lt, _nlza), dim=-1)))
                ones = RoutingFullLike(top_mass, 1.0)
                q_tail = self.one_minus_top_mass(RoutingStack((ones, top_mass), dim=-1))
                probs = self.scale_top_probs_by_tail(top_probs, top_mass)
            else:
                tail_prob = self.gibbs_tail_prob_eps.to(device=s_k.device, dtype=s_k.dtype).clamp(0.0, 1.0)
                probs = self.scale_top_probs_by_tail(top_probs, 1.0 - tail_prob)
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

    **Strict form (default):** uses :class:`SquareIdentityOperandChain` with mix ``pq,...kqd->...kpd``,
    contract ``p,...kpd->...d``. Operands are expanded prob and gathered
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
# Attention replacement
# ---------------------------------------------------------------------------


class SurgeryAttention(nn.Module):
    """ViT multi-head attention assembled from surgery op modules."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        seq_len: int,
        top_k: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_attention_surgery: bool = True,
        use_surgery_softmax: bool = True,
        allow_matmul: bool = False,
        eps_ln: float = 1e-5,
        gibbs_tail_prob_eps: float = 1e-5,
        use_exact_tail_mass: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_attention_surgery = use_attention_surgery
        self.use_surgery_softmax = use_surgery_softmax
        self.allow_matmul = allow_matmul
        if use_attention_surgery and not use_surgery_softmax and allow_matmul:
            self.matmul = AffineMatMul()
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if use_attention_surgery:
            self.dot = PairwiseDotBySquare(self.head_dim, allow_matmul=allow_matmul)
            if use_surgery_softmax:
                self.gibbs = GibbsTopKSoftmax(
                    seq_len,
                    top_k,
                    eps=eps_ln,
                    gibbs_tail_prob_eps=gibbs_tail_prob_eps,
                    allow_matmul=allow_matmul,
                    use_exact_tail_mass=use_exact_tail_mass,
                )
                self.sparse_mix = SparseWeightedSumBySquare(allow_matmul=allow_matmul)
                k_eff = min(int(top_k), int(seq_len))
                n_dropped = max(int(seq_len) - k_eff, 1)
                inv_dropped = 1.0 / n_dropped
                n_over_dropped = float(seq_len) / n_dropped
                self.mean_v = AffineMean(dim=-2, keepdim=True)
                self.scale_mean_v_by_tail = AffineHadamard()
                self.adj_probs_contract = AffineContract(
                    "i,...i->...",
                    torch.tensor([1.0, -inv_dropped], dtype=get_surgery_dtype()),
                )
                self.attn_tail_sum = AffineContract(
                    "i,...i->...",
                    torch.tensor([1.0, n_over_dropped], dtype=get_surgery_dtype()),
                )
            else:
                self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.register_buffer("attn_scale", torch.tensor(float(self.head_dim) ** -0.5))
            self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if not self.use_attention_surgery:
            s = self.attn_scale.to(device=q.device, dtype=q.dtype)
            qs = q * s
            kt = RoutingTranspose(k, -2, -1)
            attn = qs @ kt
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            attn = attn @ v
        elif self.use_surgery_softmax:
            scores = self.dot(q, k)
            probs, idx, q_tail = self.gibbs(scores)
            _p, _qt = RoutingBroadcastTensors(probs, q_tail)
            adjusted_probs = self.adj_probs_contract(RoutingStack((_p, _qt), dim=-1))
            attn_top = self.sparse_mix(adjusted_probs, idx, v)
            mean_v = self.mean_v(v)
            tail_contrib = self.scale_mean_v_by_tail(mean_v, q_tail)
            _a, _t = RoutingBroadcastTensors(attn_top, tail_contrib)
            attn = self.attn_tail_sum(RoutingStack((_a, _t), dim=-1))
        else:
            scores = self.dot(q, k)
            attn = scores.softmax(dim=-1)
            attn = self.attn_drop(attn)
            if self.allow_matmul:
                attn = self.matmul(attn, v)
            else:
                _, _, nq, nk = attn.shape
                v_b = RoutingExpand(RoutingUnsqueeze(v, 2), -1, -1, nq, nk, -1)
                attn = (RoutingUnsqueeze(attn, -1) * v_b).sum(dim=3)
        attn = RoutingTranspose(attn, 1, 2).reshape(b, n, c)
        attn = self.proj(attn)
        attn = self.proj_drop(attn)
        return attn


# ---------------------------------------------------------------------------
# GELU basis op (MLP nonlinearity)
# ---------------------------------------------------------------------------


class NLGELU(nn.Module):
    """Exact GELU as a nonlinear unary basis op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


# ---------------------------------------------------------------------------
# PTQ forward wrapper
# ---------------------------------------------------------------------------


def ptq_signed_qrange(bits: int) -> Tuple[int, int]:
    qmax = (1 << (int(bits) - 1)) - 1
    qmin = -(1 << (int(bits) - 1))
    return qmin, qmax


def ptq_quantize_proxy(x: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    qmin, qmax = ptq_signed_qrange(bits)
    s = scale.to(device=x.device, dtype=torch.float32)
    return torch.clamp(torch.round(x.to(dtype=torch.float32) / s), qmin, qmax)


class PTQInputQuantizer(nn.Module):
    def __init__(self, activation_bits: int, input_scale: torch.Tensor) -> None:
        super().__init__()
        self.activation_bits = int(activation_bits)
        self._qmin, self._qmax = ptq_signed_qrange(self.activation_bits)
        self.register_buffer("input_scale", input_scale.to(dtype=torch.float32))
        self.register_buffer("input_inv_scale", input_scale.reciprocal().to(dtype=torch.float32))

    def forward(self, *inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(
            torch.clamp(torch.round(x.float() * self.input_inv_scale[i]), self._qmin, self._qmax)
            for i, x in enumerate(inputs)
        )


class CalibratedAffinePTQWrapper(nn.Module):
    """Forward-only PTQ proxy: quantize, accumulate, dequantize."""

    def __init__(
        self,
        *,
        activation_bits: int,
        input_scale: torch.Tensor,
        accumulator: nn.Module,
        out_scale: torch.Tensor,
        out_bias: torch.Tensor,
        skip_out_scale: bool = False,
        out_bcast_shape: Optional[Tuple[int, ...]] = None,
        module_device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.activation_bits = int(activation_bits)
        self.skip_out_scale = bool(skip_out_scale)
        self.quantizer = PTQInputQuantizer(activation_bits, input_scale)
        self.accumulator = accumulator
        self.register_buffer("out_scale", out_scale.to(dtype=torch.float32))
        self.register_buffer("out_bias", out_bias.to(dtype=torch.float32))
        self.out_bcast_shape = out_bcast_shape

        if module_device is not None:
            self.to(device=module_device)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        acc = self.accumulator(*self.quantizer(*inputs))
        if self.out_bcast_shape is not None:
            bias = self.out_bias.view(self.out_bcast_shape)
            out = acc + bias if self.skip_out_scale else self.out_scale.view(self.out_bcast_shape) * acc + bias
        else:
            out = acc + self.out_bias if self.skip_out_scale else self.out_scale * acc + self.out_bias
        return out.to(dtype=inputs[0].dtype)
