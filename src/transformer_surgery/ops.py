"""Forward op vocabulary for the surgery graph."""

from __future__ import annotations

import math
import warnings
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
RoutingScatter = torch.scatter
RoutingFullLike = torch.full_like
RoutingReLU = F.relu
RoutingDropPath = _TimmDropPath


def RoutingExpand(x: torch.Tensor, *sizes: int) -> torch.Tensor:
    return x.expand(*sizes)


def RoutingExpandAs(x: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return x.expand_as(other)


def RoutingSqueeze(x: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    return x.squeeze() if dim is None else x.squeeze(dim)


def RoutingAdd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pure broadcast add used for score masking / residual wiring helpers."""
    return a + b


def build_causal_attn_mask(seq_len: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Additive causal mask ``[1, 1, N, N]``: 0 on allowed keys, ``-inf`` on future keys."""
    n = int(seq_len)
    mask = torch.zeros((1, 1, n, n), dtype=dtype)
    future = torch.triu(torch.ones((n, n), dtype=torch.bool), diagonal=1)
    mask = mask.masked_fill(future, float("-inf"))
    return mask


def build_causal_valid_key_mean_matrix(seq_len: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Lower-triangular mean matrix ``L[i, j] = 1/(i+1)`` for ``j <= i`` (valid-key mean of V)."""
    n = int(seq_len)
    idx = torch.arange(n, dtype=dtype)
    valid = idx + 1.0
    tril = torch.tril(torch.ones((n, n), dtype=dtype))
    return tril / valid.unsqueeze(1)


# ---------------------------------------------------------------------------
# Op vocabulary: three groups.
#   Affine*  - linear in each operand: parameterized layers, fixed-coeff einsum, per-channel
#              scale+bias, fixed 2x2 mixes, linear unary reductions/scalings, and the variable
#              bilinear ops (AffineMatMul, AffineHadamard) gated by allow_matmul (and AffineHadamard
#              alone gated by ``hadamard``).
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


class AffineScaleTensor(nn.Module):
    """Elementwise scale by a fixed buffer tensor (broadcastable), e.g. rotary cos/sin tables."""

    def __init__(self, scale: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("scale", scale.clone().detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.scale
        if s.device != x.device:
            s = s.to(device=x.device)
        return x * s.to(dtype=x.dtype)


class AffineHadamard(nn.Module):
    """Elementwise / broadcast product ``a * b``; both operands are tensors (bilinear, no ``@``)."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b


class AffineSelfHadamard(nn.Module):
    """Square ``x * x`` as one elementwise product of an activation with itself (Hadamard basis, not unary)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class AffineAdd(nn.Module):
    """Elementwise / broadcast sum ``a + b`` (e.g. additive attention masks)."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


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


# In-place evaluation of full-size intermediates (strict graph): on by default, used only when gradients are off.
# PTQ turns it off, since its calibration reads node inputs in forward hooks after the node has run.
_INPLACE_EVAL = [True]


def set_inplace_eval(enabled: bool) -> None:
    _INPLACE_EVAL[0] = bool(enabled)


def _inplace_ok(flag: bool) -> bool:
    return bool(flag) and _INPLACE_EVAL[0] and not torch.is_grad_enabled()


class NLSquare(nn.Module):
    """Unary square ``x * x``; ``inplace=True`` squares a fresh tensor in place during evaluation."""

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = bool(inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _inplace_ok(self.inplace):
            return x.square_()
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


class NLSigmoid(nn.Module):
    """Logistic sigmoid ``1/(1+exp(-x))`` as a unary basis op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


# "sigmoid_rsqrt": squares, sigmoid, inverse square root (the paper's strict graph, default).
# "log_exp": deprecated log-domain LayerNorm and softmax, kept to reload and reproduce earlier runs.
STRICT_UNARY_SETS = ("sigmoid_rsqrt", "log_exp")
GELU_MODES = ("exact", "sigmoid")


def check_strict_unary(strict_unary: str) -> str:
    value = str(strict_unary)
    if value not in STRICT_UNARY_SETS:
        raise ValueError(f"strict_unary must be one of {STRICT_UNARY_SETS}, got {value!r}")
    return value


# Products of two activations in the strict graph.
# "norm" (default, the paper's strict construction): ab = ((a+b)^2 - a^2 - b^2)/2 wherever a^2 or b^2 is shared, so
# it costs one new square per product: the scores ( sum_l (q+k)^2 - |q|^2 - |k|^2 ) / (2 sqrt d), the mix
# ( sum_k (v^2 - (v-p)^2) + sum_k p^2 ) / 2 with the two close squares subtracted per term before the sum over keys,
# LayerNorm ((u+a)^2 - u^2 - a^2)/2 with u^2 from the variance, and the exact tail term. Polarization stays where it
# is cheaper (isolated products: softmax, the GELU approximation).
# "polarization": every product as ((a+b)^2-(a-b)^2)/4, kept to reload and reproduce earlier runs.
STRICT_PRODUCTS = ("norm", "polarization")


def check_strict_product(strict_product: str, *, allow_matmul: bool, hadamard: bool = False) -> str:
    """The product rule of the strict graph; graphs with activation-activation products ignore it."""
    value = str(strict_product)
    if value not in STRICT_PRODUCTS:
        raise ValueError(f"strict_product must be one of {STRICT_PRODUCTS}, got {value!r}")
    return "polarization" if (allow_matmul or hadamard) else value


def check_hadamard(hadamard: bool, *, allow_matmul: bool, strict_unary: str = "sigmoid_rsqrt") -> bool:
    """``hadamard=True`` is the Hadamard graph: the strict graph with every product of two activations, squares
    included, as one elementwise product instead of polarization, and QK^T and PV as products plus sums (no
    matmul). It requires ``allow_matmul=False`` and ``strict_unary="sigmoid_rsqrt"``."""
    value = bool(hadamard)
    if value and (allow_matmul or str(strict_unary) != "sigmoid_rsqrt"):
        raise ValueError("hadamard=True needs allow_matmul=False and strict_unary='sigmoid_rsqrt'")
    return value


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


class AffineCombine(AffineContract):
    """
    Fixed affine combination ``sum_i c_i x_i`` of several inputs, evaluated in input order and broadcast to the
    shape of ``x_0``, with an optional sum of ``x_0`` over ``reduce_first`` before combining and of the result over
    ``reduce_out`` after. It is an :class:`AffineContract` (coefficients in ``coeff``), so PTQ quantizes every input
    and accumulates the combination in one node. ``inplace=True`` writes into a fresh ``x_0`` during evaluation.

    The inputs differ in scale (for example ``v`` and ``p``, or ``u^2`` and ``inv_std^2``), so under PTQ each input is
    quantized on its own grid and rescaled before the sum, as an integer Add with per-input multipliers does
    (``ptq_dequantized_inputs``): the accumulator receives code times scale, not the bare codes.
    """

    ptq_dequantized_inputs = True

    def __init__(
        self,
        coeffs,
        *,
        reduce_first: Optional[int] = None,
        reduce_out: Optional[int] = None,
        inplace: bool = False,
    ) -> None:
        super().__init__("combine", torch.tensor(coeffs, dtype=torch.float32))
        self.reduce_first = reduce_first
        self.reduce_out = reduce_out
        self.inplace = bool(inplace)
        self._coeff_key: Optional[Tuple[int, int]] = None
        self._coeff_vals: list = []

    def _coefficients(self) -> list:
        # Python copies of ``coeff``, refreshed when PTQ swaps in quantized coefficients or the buffer moves.
        c = self.coeff
        key = (c.data_ptr(), c._version)
        if key != self._coeff_key:
            self._coeff_vals = [float(v) for v in c.detach().float().cpu().tolist()]
            self._coeff_key = key
        return self._coeff_vals

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        c = self._coefficients()
        x0 = xs[0] if self.reduce_first is None else xs[0].sum(dim=self.reduce_first)
        if torch.is_grad_enabled():
            y = x0 * c[0]
            for x, ci in zip(xs[1:], c[1:]):
                y = torch.add(y, x, alpha=ci)
        else:
            start = 1
            if _inplace_ok(self.inplace or self.reduce_first is not None):
                y = x0 if c[0] == 1.0 else x0.mul_(c[0])
            elif c[0] == 1.0 and len(xs) > 1:
                # One fused pass for the first two inputs, stored in the dtype of x_0 like the in-place sum: under
                # autocast a plain torch.add would promote a bf16 x_0 plus an fp32 x_1 (a mean or sum) to fp32.
                y = torch.empty(torch.broadcast_shapes(x0.shape, xs[1].shape), dtype=x0.dtype, device=x0.device)
                torch.add(x0, xs[1], alpha=c[1], out=y)
                start = 2
            else:
                y = x0 * c[0]
            for x, ci in zip(xs[start:], c[start:]):
                y.add_(x, alpha=ci)  # y is this node's own tensor, so accumulating in place is safe
        return y if self.reduce_out is None else y.sum(dim=self.reduce_out)


class NormExpansionProduct(nn.Module):
    """
    Product of two broadcast activations ``a * b = ((a+b)^2 - a^2 - b^2) / 2``: the squares of the small operands
    are formed once, so a product of shapes ``[.., 1, D]`` and ``[.., N, 1]`` costs ``N D + N + D`` squares against
    ``2 N D`` by polarization.
    """

    def __init__(self) -> None:
        super().__init__()
        self.add = AffineCombine([1.0, 1.0])
        self.square = NLSquare(inplace=True)
        self.square_a = NLSquare()
        self.square_b = NLSquare()
        self.combine = AffineCombine([0.5, -0.5, -0.5], inplace=True)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_full, b_full = RoutingBroadcastTensors(a, b)
        s = self.square(self.add(a_full, b_full))
        return self.combine(s, self.square_a(a), self.square_b(b))


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


class AffineContractPerTerm(AffineContract):
    """
    :class:`AffineContract` that combines the operand channel per term and then sums over ``sum_dim``: the einsum
    ``p,...kpd->...kd`` followed by a sum over the keys. Polarization in a contraction then subtracts its two squares
    per term before the sum, so no large partial sum of squares is rounded; PTQ still quantizes the stacked squares on
    one grid and accumulates the whole node.
    """

    def __init__(self, einsum_equation: str, coeffs: torch.Tensor, sum_dim: int) -> None:
        super().__init__(einsum_equation, coeffs)
        self.sum_dim = int(sum_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).sum(dim=self.sum_dim)


class ElementwiseProduct(nn.Module):
    """
    Product of two activations ``a * b`` (broadcast). Strict: polarization ``((a+b)^2-(a-b)^2)/4``,
    two squares per entry. ``allow_matmul=True`` or ``hadamard=True``: one :class:`AffineHadamard` node.
    """

    def __init__(self, *, allow_matmul: bool = False, hadamard: bool = False) -> None:
        super().__init__()
        self.allow_matmul = bool(allow_matmul) or bool(hadamard)
        if self.allow_matmul:
            self.hadamard = AffineHadamard()
        else:
            self.square_chain = SquareIdentityOperandChain(
                "pq,...qd->...pd",
                "p,...pd->...d",
                torch.tensor([0.25, -0.25], dtype=get_surgery_dtype()),
            )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a, b = RoutingBroadcastTensors(a, b)
        if self.allow_matmul:
            return self.hadamard(a, b)
        return self.square_chain(a, b)


class ReciprocalByRsqrt(nn.Module):
    """``1/x = (x^{-1/2})^2`` for ``x > 0``: one inverse square root and one square (a self-product if ``hadamard``)."""

    def __init__(self, eps: float = 0.0, *, hadamard: bool = False) -> None:
        super().__init__()
        self.rsqrt = NLRsqrtPlusEps(eps)
        self.square = AffineSelfHadamard() if hadamard else NLSquare()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.square(self.rsqrt(x))


class ExpBySigmoid(nn.Module):
    """
    ``exp(t) = sigmoid(t) / sigmoid(-t)`` with the reciprocal as :class:`ReciprocalByRsqrt` and the product
    by polarization: two sigmoids, one inverse square root and three squares per entry, no exp and no log.
    ``hadamard=True``: the square and the product are elementwise products, so only the two sigmoids and the
    inverse square root are unary.
    """

    def __init__(self, *, hadamard: bool = False) -> None:
        super().__init__()
        self.sigmoid = NLSigmoid()
        self.negate = AffineScale(-1.0)
        self.reciprocal = ReciprocalByRsqrt(0.0, hadamard=hadamard)
        self.product = ElementwiseProduct(allow_matmul=False, hadamard=hadamard)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.product(self.sigmoid(t), self.reciprocal(self.sigmoid(self.negate(t))))


class SigmoidGELU(nn.Module):
    """GELU approximation ``x * sigmoid(1.702 x)`` (Hendrycks and Gimpel); product by polarization if strict."""

    def __init__(self, *, allow_matmul: bool = False, hadamard: bool = False) -> None:
        super().__init__()
        self.scale = AffineScale(1.702)
        self.sigmoid = NLSigmoid()
        self.product = ElementwiseProduct(allow_matmul=allow_matmul, hadamard=hadamard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.product(x, self.sigmoid(self.scale(x)))


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
# LayerNorm rewrite: polarization with the inverse square root (strict), Hadamard (fast)
# ---------------------------------------------------------------------------


class RewrittenLayerNorm(nn.Module):
    """
    mu = mean(x); u = x - mu; r2 = mean(u*u).

    **``allow_matmul=False``, ``strict_unary="sigmoid_rsqrt"`` (default, strict):** ``inv_std = 1/sqrt(r2+eps)``
    via :class:`NLRsqrtPlusEps`, and ``z = u * inv_std`` by polarization (:class:`ElementwiseProduct`):
    no log, no exp and no sign split.

    **``allow_matmul=False``, ``strict_unary="log_exp"`` (deprecated):** ``au = stack(relu(u),relu(-u))``
    (dim ``-2``); ``log_num = log_eps(au)`` (operand ``p``); ``log_den = log_eps(r2)`` broadcast;
    ``t = AffineContract([2,-1])(stack(log_num, log_den))``; ``a_mag = NLSqrtExp(t)``;
    ``z = out_contract(a_mag)``.

    **``allow_matmul=True`` (debug / fast):** same ``r2``, then ``z = u * inv_std`` with
    ``inv_std = 1/sqrt(r2+eps)`` via :class:`NLRsqrtPlusEps` and :class:`AffineHadamard`, not the
    log-domain path.

    **``hadamard=True`` (Hadamard graph):** ``u*u`` and ``u * inv_std`` are elementwise products; the inverse
    square root is the only unary map.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float,
        *,
        allow_matmul: bool = False,
        strict_unary: str = "sigmoid_rsqrt",
        hadamard: bool = False,
        strict_product: str = "norm",
    ) -> None:
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        e = float(eps)
        self.allow_matmul = allow_matmul
        self.strict_unary = check_strict_unary(strict_unary)
        self.hadamard = check_hadamard(hadamard, allow_matmul=allow_matmul, strict_unary=self.strict_unary)
        self.strict_product = check_strict_product(strict_product, allow_matmul=allow_matmul, hadamard=self.hadamard)
        # Norm expansion of u * inv_std: u^2 is already formed for the variance, inv_std^2 is one square per token.
        self.norm_ln = self.strict_product == "norm" and self.strict_unary == "sigmoid_rsqrt"
        self.mean_u = AffineMean(-1, keepdim=True)
        self.mean_r2 = AffineMean(-1, keepdim=True)
        self.square = AffineSelfHadamard() if self.hadamard else NLSquare()
        self.u_center_contract = AffineContract(
            "i,...i->...",
            torch.tensor([1.0, -1.0], dtype=get_surgery_dtype()),
        )
        self.affine = AffineScaleBias(normalized_shape)
        if allow_matmul:
            self.inv_sqrt_var = NLRsqrtPlusEps(e)
            self.u_mul_invstd = AffineHadamard()
        elif self.norm_ln:
            self.inv_sqrt_var = NLRsqrtPlusEps(e)
            self.add_u_invstd = AffineCombine([1.0, 1.0])
            self.square_sum = NLSquare(inplace=True)
            self.square_invstd = NLSquare()
            self.combine_z = AffineCombine([0.5, -0.5, -0.5], inplace=True)
        elif self.strict_unary == "sigmoid_rsqrt":
            self.inv_sqrt_var = NLRsqrtPlusEps(e)
            self.u_mul_invstd = ElementwiseProduct(allow_matmul=False, hadamard=self.hadamard)
        else:
            warnings.warn(
                "strict_unary='log_exp' (log-domain strict graph) is deprecated; use 'sigmoid_rsqrt'",
                FutureWarning,
                stacklevel=2,
            )
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
        if self.norm_ln:
            inv_std = self.inv_sqrt_var(r2)
            _u, _a = RoutingBroadcastTensors(u, inv_std)
            s = self.square_sum(self.add_u_invstd(_u, _a))
            z = self.combine_z(s, u2, self.square_invstd(inv_std))
        elif self.allow_matmul or self.strict_unary == "sigmoid_rsqrt":
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

    ``hadamard=True`` (Hadamard graph): ``(q/sqrt(d)) * k`` on the broadcast grid, one elementwise product
    per term, then a fixed sum over the head dimension. No matmul and no unary map.
    """

    def __init__(
        self, head_dim: int, *, allow_matmul: bool = False, hadamard: bool = False, strict_product: str = "norm"
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.allow_matmul = allow_matmul
        self.hadamard = bool(hadamard) and not allow_matmul
        self.norm_expansion = (
            check_strict_product(strict_product, allow_matmul=allow_matmul, hadamard=self.hadamard) == "norm"
        )
        if self.norm_expansion:
            # S = ( sum_l (q+k)^2 - |q|^2 - |k|^2 ) / (2 sqrt(d)): one square per term, norms once per row.
            w = 0.5 / math.sqrt(float(head_dim))
            self.sum_qk = AffineCombine([1.0, 1.0])
            self.square = NLSquare(inplace=True)
            self.square_q = NLSquare()
            self.square_k = NLSquare()
            self.norm_q = AffineSum(-1, keepdim=True)
            self.norm_k = AffineSum(-1)
            self.combine = AffineCombine([w, -w, -w], reduce_first=-1)
        elif allow_matmul:
            self.q_scale = AffineScale(1.0 / math.sqrt(float(head_dim)))
            self.qk_matmul = AffineMatMul()
        elif self.hadamard:
            self.q_scale = AffineScale(1.0 / math.sqrt(float(head_dim)))
            self.qk_hadamard = AffineHadamard()
            self.reduce = AffineSum(-1)
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
        if self.hadamard:
            # Scale q before the broadcast, so only the product grid is materialized (same values).
            qs_b = RoutingExpand(RoutingUnsqueeze(self.q_scale(q), 3), -1, -1, -1, nk, -1)
            return self.reduce(self.qk_hadamard(qs_b, ke_b))
        if self.norm_expansion:
            sq = self.square(self.sum_qk(qe_b, ke_b))
            nq2 = self.norm_q(self.square_q(q))                                   # [B,H,Nq,1]
            nk2 = RoutingUnsqueeze(self.norm_k(self.square_k(k)), -2)            # [B,H,1,Nk]
            return self.combine(sq, nq2, nk2)
        return self.square_chain(qe_b, ke_b)

    def selected(self, q: torch.Tensor, k: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Scores on a given support only: ``q`` [B,H,Nq,D], ``idx`` [B,H,Nq,K] -> [B,H,Nq,K]."""
        nq = q.shape[2]
        d = k.shape[-1]
        kk = idx.shape[-1]
        idx_e = RoutingExpand(RoutingUnsqueeze(idx, -1), -1, -1, -1, -1, d)
        k_g = RoutingGather(RoutingExpand(RoutingUnsqueeze(k, 2), -1, -1, nq, -1, -1), 3, idx_e)
        if self.allow_matmul:
            qs = RoutingUnsqueeze(self.q_scale(q), -2)
            return RoutingSqueeze(self.qk_matmul(qs, RoutingTranspose(k_g, -2, -1)), -2)
        q_b = RoutingExpand(RoutingUnsqueeze(q, 3), -1, -1, -1, kk, -1)
        if self.hadamard:
            qs_b = RoutingExpand(RoutingUnsqueeze(self.q_scale(q), 3), -1, -1, -1, kk, -1)
            return self.reduce(self.qk_hadamard(qs_b, k_g))
        if self.norm_expansion:
            sq = self.square(self.sum_qk(k_g, q_b))
            nq2 = self.norm_q(self.square_q(q))                                   # [B,H,Nq,1]
            nk2_all = RoutingExpand(RoutingUnsqueeze(self.norm_k(self.square_k(k)), 2), -1, -1, nq, -1)
            return self.combine(sq, nq2, RoutingGather(nk2_all, 3, idx))
        return self.square_chain(q_b, k_g)


# ---------------------------------------------------------------------------
# Gibbs Top-K softmax with fixed omitted-tail probability
# ---------------------------------------------------------------------------


class GibbsTopKSoftmax(nn.Module):
    """
    Sparse Gibbs Top-K with omitted-tail probability mass.
    Returns per-row: sparse probs on idx, tail mass scalar q_tail, and idx.

    Normalization never uses the ``/`` operator: with ``allow_matmul=True`` use
    :class:`NLReciprocalPlusEps` and :class:`AffineHadamard`; strict ``strict_unary="sigmoid_rsqrt"`` uses
    :class:`ExpBySigmoid` and ``1/(Z+eps) = ((Z+eps)^{-1/2})^2`` (:class:`ReciprocalByRsqrt`) with products by
    polarization (no exp, no log); the deprecated strict ``"log_exp"`` uses ``exp(vals - log(sum_exp + eps))``.

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
        strict_unary: str = "sigmoid_rsqrt",
        hadamard: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.top_k = top_k
        self.use_exact_tail_mass = bool(use_exact_tail_mass)
        self.strict_unary = check_strict_unary(strict_unary)
        h = check_hadamard(hadamard, allow_matmul=allow_matmul, strict_unary=self.strict_unary)
        # Strict graph without exp and log: exp(t) = sigmoid(t)/sigmoid(-t), 1/x = rsqrt(x)^2.
        self.sigmoid_rsqrt = (not allow_matmul) and self.strict_unary == "sigmoid_rsqrt"
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
        elif self.sigmoid_rsqrt:
            self.exp = ExpBySigmoid(hadamard=h)
            self.inv_z = ReciprocalByRsqrt(e, hadamard=h)
            self.mul_by_inv_z = ElementwiseProduct(allow_matmul=False, hadamard=h)
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
            elif self.sigmoid_rsqrt:
                self.inv_z_all = ReciprocalByRsqrt(e, hadamard=h)
                self.mul_ztop_inv_zall = ElementwiseProduct(allow_matmul=False, hadamard=h)
                self.scale_top_probs_by_tail = ElementwiseProduct(allow_matmul=False, hadamard=h)
            else:
                self.log_z_all = NLLogPlusEps(e)
                self.log_z_top = NLLogPlusEps(e)
                self.top_mass_log_contract = AffineContract(
                    "i,...i->...",
                    torch.tensor([1.0, 1.0], dtype=get_surgery_dtype()),
                )

    def forward(
        self, scores: torch.Tensor, selection_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        scores: [B, H, Nq, Nk]
        selection_scores: optional [B, H, Nq, Nk] that choose the support (e.g. from
        :class:`QueryCentroidRouter`); ``None`` keeps the top-k of ``scores``.
        returns probs [B,H,Nq,K], idx [B,H,Nq,K], q_tail [B,H,Nq,1]
        """
        _b, _h, _nq, nk = scores.shape
        k = min(self.top_k, nk)
        if selection_scores is None:
            raw_vals, idx = RoutingTopK(scores, k=k, dim=-1, largest=True, sorted=True)
        else:
            _sel_vals, sel_idx = RoutingTopK(selection_scores, k=k, dim=-1, largest=True, sorted=False)
            # Exact scores on the routed support, ordered so the first slot holds the row max.
            raw_vals, order = RoutingTopK(
                RoutingGather(scores, -1, sel_idx), k=k, dim=-1, largest=True, sorted=True
            )
            idx = RoutingGather(sel_idx, -1, order)
        return self._normalize_support(raw_vals, idx, nk, scores)

    def forward_selected(
        self, selected_scores: torch.Tensor, idx: torch.Tensor, nk: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Exact scores read only on a support chosen elsewhere (e.g. :meth:`QueryCentroidRouter.support`).

        selected_scores, idx: [B, H, Nq, K]; nk: number of keys. Calibrated or fixed tail mass only:
        exact mode needs every score for the dense normalizer.
        """
        if self.use_exact_tail_mass:
            raise ValueError("forward_selected needs a calibrated or fixed tail mass, not exact mode")
        kk = selected_scores.shape[-1]
        raw_vals, order = RoutingTopK(selected_scores, k=kk, dim=-1, largest=True, sorted=True)
        return self._normalize_support(raw_vals, RoutingGather(idx, -1, order), int(nk), None)

    def _normalize_support(
        self, raw_vals: torch.Tensor, idx: torch.Tensor, nk: int, scores: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Probabilities on the support from scores sorted in descending order; ``scores`` for exact mode."""
        k = raw_vals.shape[-1]
        row_max = raw_vals[..., :1]
        _v, _r = RoutingBroadcastTensors(raw_vals, row_max)
        vals = self.top_vals_stable_contract(RoutingStack((_v, _r), dim=-1))
        s_k = vals[..., -1:]
        exp_vals = self.exp(vals)
        normalizer = self.sum_exp_vals(exp_vals)

        if self.allow_matmul or self.sigmoid_rsqrt:
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
                if self.allow_matmul or self.sigmoid_rsqrt:
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
# Sparse support routing by fixed query centroids
# ---------------------------------------------------------------------------


class QueryCentroidRouter(nn.Module):
    """
    Selection scores for the sparse support from fixed query centroids ``c[h, m]`` (calibrated offline).

    Query ``i`` goes to its nearest centroid ``m(i) = argmax_m (q_i . c_m - |c_m|^2 / 2)`` and scores key
    ``j`` by ``c_{m(i)} . k_j``. Both contractions have fixed coefficients and the assignment and gather
    are routing, so the router adds no unary evaluation and no activation-activation product.
    :class:`GibbsTopKSoftmax` keeps the top-k keys of these scores and reads exact scores only there.
    """

    def __init__(self, centroids: torch.Tensor) -> None:
        super().__init__()
        c = centroids.detach().float()
        if c.dim() != 3:
            raise ValueError(f"centroids must be [heads, clusters, head_dim], got {tuple(c.shape)}")
        self.num_clusters = int(c.shape[1])
        self.assign_scores = AffineContract("hcd,bhnd->bhnc", c)
        self.key_scores = AffineContract("hcd,bhkd->bhck", c)
        self.register_buffer("assign_bias", (-0.5 * c.pow(2).sum(-1)).view(1, c.shape[0], 1, c.shape[1]))
        self.add_assign_bias = AffineAdd()

    def forward(self, q: torch.Tensor, k: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """q: [B,H,Nq,D], k: [B,H,Nk,D], optional additive mask; returns selection scores [B,H,Nq,Nk]."""
        bias = self.assign_bias.to(device=q.device, dtype=q.dtype)
        assign = self.add_assign_bias(self.assign_scores(q), bias)
        _a, m = RoutingTopK(assign, k=1, dim=-1)
        key_scores = self.key_scores(k)
        m_e = RoutingExpand(m, -1, -1, -1, key_scores.shape[-1])
        selection = RoutingGather(key_scores, 2, m_e)
        return selection if mask is None else RoutingAdd(selection, mask)

    def support(self, q: torch.Tensor, k: torch.Tensor, top_k: int) -> torch.Tensor:
        """Unmasked support ``[B,H,Nq,K]``: top-k keys per centroid, gathered by each query's centroid.

        Equals the top-k of :meth:`forward` row by row without forming an ``Nq x Nk`` matrix.
        """
        bias = self.assign_bias.to(device=q.device, dtype=q.dtype)
        assign = self.add_assign_bias(self.assign_scores(q), bias)
        _a, m = RoutingTopK(assign, k=1, dim=-1)
        _v, idx_c = RoutingTopK(self.key_scores(k), k=int(top_k), dim=-1, largest=True, sorted=False)
        return RoutingGather(idx_c, 2, RoutingExpand(m, -1, -1, -1, int(top_k)))


# ---------------------------------------------------------------------------
# Sparse weighted sum via square identity
# ---------------------------------------------------------------------------


class SparseWeightedSumBySquare(nn.Module):
    """
    y[b,h,nq,d] = sum_{k in top} p[b,h,nq,k] * v[b,h, idx[b,h,nq,k], d]

    **Strict form (default):** uses :class:`SquareIdentityOperandChain` with mix ``pq,...kqd->...kpd``,
    contract ``p,...kpd->...d``. Operands are expanded prob and gathered
    value per top-k slot. No dense ``matmul`` for attention-value mixing.

    ``allow_matmul=True`` uses ``p * v`` then sum (faster; invalid for strict demo). At full support
    (``K == Nk``) ``idx`` is a permutation of the keys: the weights are scattered back to key order and
    mixed by one dense ``[Nq,Nk] @ [Nk,D]`` product, the same sum without a gathered ``[Nq,K,D]`` tensor.
    """

    def __init__(self, *, allow_matmul: bool = False, hadamard: bool = False, strict_product: str = "norm") -> None:
        super().__init__()
        self.allow_matmul = allow_matmul
        # Hadamard graph: ``p * v`` as elementwise products, then a fixed sum over the support.
        self.hadamard = bool(hadamard) and not allow_matmul
        self.norm_expansion = (
            check_strict_product(strict_product, allow_matmul=allow_matmul, hadamard=self.hadamard) == "norm"
        )
        if self.norm_expansion:
            # y = ( sum_k (v^2 - (v-p)^2) + sum_k p^2 ) / 2: v^2 once per key and gathered, p^2 per (query, key), one
            # square of v - p per term. The two close squares are subtracted per term inside the node that sums
            # over keys, so no large partial sum is rounded; the full-size steps run in place during evaluation.
            self.sub_vp = AffineCombine([1.0, -1.0], inplace=True)
            self.square = NLSquare(inplace=True)
            self.square_v = NLSquare()
            self.diff_sum = AffineCombine([1.0, -1.0], reduce_out=-2, inplace=True)
            self.square_p = NLSquare()
            self.sum_p2 = AffineSum(-1, keepdim=True)
            self.combine = AffineCombine([0.5, 0.5])
        elif allow_matmul:
            self.pv_matmul = AffineMatMul()
        elif self.hadamard:
            self.pv_hadamard = AffineHadamard()
            self.reduce = AffineSum(-2)
        else:
            # Polarization, combined per term: (p+v)^2 - (p-v)^2 for each key, then the sum over keys, in one node.
            self.square_chain = SquareIdentityOperandChain(
                "pq,...kqd->...kpd",
                "p,...kpd->...kd",
                torch.tensor([0.25, -0.25], dtype=get_surgery_dtype()),
            )
            self.square_chain.out_contract = AffineContractPerTerm(
                "p,...kpd->...kd", torch.tensor([0.25, -0.25], dtype=get_surgery_dtype()), sum_dim=-2
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
        _, _, nq, kk = probs.shape
        _, _, nk, d = v.shape
        if self.allow_matmul and kk == nk:
            return self.pv_matmul(RoutingScatter(RoutingFullLike(probs, 0.0), -1, idx, probs), v)
        idx_e = RoutingExpand(RoutingUnsqueeze(idx, -1), -1, -1, -1, -1, d)
        v_h = RoutingExpand(RoutingUnsqueeze(v, 2), -1, -1, nq, -1, -1)
        v_g = RoutingGather(v_h, 3, idx_e)
        if self.allow_matmul:
            # [B,H,Nq,1,K] @ [B,H,Nq,K,D] -> [B,H,Nq,1,D]
            return RoutingSqueeze(self.pv_matmul(RoutingUnsqueeze(probs, -2), v_g), -2)
        p = RoutingUnsqueeze(probs, -1)
        p_b = RoutingExpand(p, -1, -1, -1, -1, d)
        if self.hadamard:
            return self.reduce(self.pv_hadamard(p_b, v_g))
        if self.norm_expansion:
            sq = self.square(self.sub_vp(v_g, p_b))
            v2_h = RoutingExpand(RoutingUnsqueeze(self.square_v(v), 2), -1, -1, nq, -1, -1)
            t = self.diff_sum(RoutingGather(v2_h, 3, idx_e), sq)
            del sq
            return self.combine(t, self.sum_p2(self.square_p(probs)))
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
        strict_unary: str = "sigmoid_rsqrt",
        hadamard: bool = False,
        strict_product: str = "norm",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.strict_unary = check_strict_unary(strict_unary)
        self.hadamard = check_hadamard(hadamard, allow_matmul=allow_matmul, strict_unary=self.strict_unary)
        self.strict_product = check_strict_product(strict_product, allow_matmul=allow_matmul, hadamard=self.hadamard)
        self.use_attention_surgery = use_attention_surgery
        self.use_surgery_softmax = use_surgery_softmax
        self.allow_matmul = allow_matmul
        if use_attention_surgery and not use_surgery_softmax and allow_matmul:
            self.matmul = AffineMatMul()
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # Optional support router (e.g. QueryCentroidRouter); None keeps the top-k of exact scores.
        self.router: Optional[nn.Module] = None
        if use_attention_surgery:
            self.dot = PairwiseDotBySquare(
                self.head_dim, allow_matmul=allow_matmul, hadamard=self.hadamard, strict_product=self.strict_product
            )
            if use_surgery_softmax:
                self.gibbs = GibbsTopKSoftmax(
                    seq_len,
                    top_k,
                    eps=eps_ln,
                    gibbs_tail_prob_eps=gibbs_tail_prob_eps,
                    allow_matmul=allow_matmul,
                    use_exact_tail_mass=use_exact_tail_mass,
                    strict_unary=self.strict_unary,
                    hadamard=self.hadamard,
                )
                self.sparse_mix = SparseWeightedSumBySquare(
                    allow_matmul=allow_matmul, hadamard=self.hadamard, strict_product=self.strict_product
                )
                k_eff = min(int(top_k), int(seq_len))
                n_dropped = max(int(seq_len) - k_eff, 1)
                inv_dropped = 1.0 / n_dropped
                n_over_dropped = float(seq_len) / n_dropped
                self.mean_v = AffineMean(dim=-2, keepdim=True)
                if use_exact_tail_mass and not allow_matmul and self.strict_unary == "sigmoid_rsqrt":
                    if self.strict_product == "norm":
                        self.scale_mean_v_by_tail = NormExpansionProduct()
                    else:
                        self.scale_mean_v_by_tail = ElementwiseProduct(allow_matmul=False, hadamard=self.hadamard)
                else:
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
            nk = k.shape[2]
            if self.router is not None and self.gibbs.top_k < nk and not self.gibbs.use_exact_tail_mass:
                # Routed support first, exact scores only on it: no Nq x Nk score matrix.
                support = self.router.support(q, k, self.gibbs.top_k)
                probs, idx, q_tail = self.gibbs.forward_selected(self.dot.selected(q, k, support), support, nk)
            else:
                scores = self.dot(q, k)
                selection = self.router(q, k) if self.router is not None else None
                probs, idx, q_tail = self.gibbs(scores, selection)
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


class SurgeryCausalAttention(nn.Module):
    """
    Causal GPT-NeoX-style attention in the surgery vocabulary.

    Fused QKV uses the NeoX head-interleaved layout ``(B, H, N, 3*D) -> chunk Q/K/V``.
    Partial rotary is applied with fixed cos/sin buffers, then scores are masked with an
    additive causal ``-inf`` mask before Gibbs top-k / softmax.

    Sparse omitted-tail redistribution is query-position aware: for row ``i`` the allowed
    key count is ``i+1``, and exact tail mass is reinjected only over valid keys via a
    fixed lower-triangular mean matrix. Dense runs set ``top_k == seq_len`` so ``q_tail=0``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        seq_len: int,
        top_k: int,
        *,
        rotary_ndims: int,
        rope_theta: float = 10000.0,
        attn_bias: bool = True,
        attn_drop: float = 0.0,
        use_attention_surgery: bool = True,
        use_surgery_softmax: bool = True,
        allow_matmul: bool = False,
        eps_ln: float = 1e-5,
        gibbs_tail_prob_eps: float = 1e-5,
        use_exact_tail_mass: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.seq_len = int(seq_len)
        self.rotary_ndims = int(rotary_ndims)
        if self.rotary_ndims < 0 or self.rotary_ndims > self.head_dim:
            raise ValueError(f"rotary_ndims={rotary_ndims} out of range for head_dim={self.head_dim}")
        if self.rotary_ndims % 2 != 0:
            raise ValueError(f"rotary_ndims must be even, got {self.rotary_ndims}")
        self.use_attention_surgery = bool(use_attention_surgery)
        self.use_surgery_softmax = bool(use_surgery_softmax)
        self.allow_matmul = bool(allow_matmul)
        self.use_exact_tail_mass = bool(use_exact_tail_mass)
        self.top_k = int(top_k)

        self.query_key_value = nn.Linear(dim, dim * 3, bias=attn_bias)
        self.dense = nn.Linear(dim, dim, bias=attn_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # Optional support router (e.g. QueryCentroidRouter); None keeps the top-k of exact scores.
        self.router: Optional[nn.Module] = None

        self.register_buffer("causal_mask", build_causal_attn_mask(self.seq_len), persistent=False)
        self._init_rotary_buffers(rope_theta)

        if use_attention_surgery:
            self.dot = PairwiseDotBySquare(self.head_dim, allow_matmul=allow_matmul)
            # Hookable masked-score node so calibration sees causal logits (not raw QK).
            self.mask_scores = AffineAdd()
            if use_surgery_softmax:
                # Softmax normalizer floor is independent of LayerNorm eps; use 0 to match
                # dense softmax (no additive floor in logZ / reciprocal).
                self.gibbs = GibbsTopKSoftmax(
                    self.seq_len,
                    top_k,
                    eps=0.0,
                    gibbs_tail_prob_eps=gibbs_tail_prob_eps,
                    allow_matmul=allow_matmul,
                    use_exact_tail_mass=use_exact_tail_mass,
                )
                self.sparse_mix = SparseWeightedSumBySquare(allow_matmul=allow_matmul)
                self._init_causal_tail_modules(top_k)
            elif allow_matmul:
                self.matmul = AffineMatMul()
        else:
            self.register_buffer("attn_scale", torch.tensor(float(self.head_dim) ** -0.5))
            if allow_matmul:
                self.matmul = AffineMatMul()

        self.rotate_neg = AffineScale(-1.0)
        self.rope_cos_mul = AffineHadamard()
        self.rope_sin_mul = AffineHadamard()
        self.rope_sum = AffineContract(
            "i,...i->...",
            torch.tensor([1.0, 1.0], dtype=get_surgery_dtype()),
        )

    def _init_rotary_buffers(self, rope_theta: float) -> None:
        n = self.seq_len
        rd = self.rotary_ndims
        if rd == 0:
            self.register_buffer("cos_cached", torch.ones(1, 1, n, 0), persistent=False)
            self.register_buffer("sin_cached", torch.zeros(1, 1, n, 0), persistent=False)
            return
        inv_freq = 1.0 / (
            float(rope_theta)
            ** (torch.arange(0, rd, 2, dtype=torch.float64) / float(rd))
        )
        positions = torch.arange(n, dtype=torch.float64)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=get_surgery_dtype())
        sin = emb.sin().to(dtype=get_surgery_dtype())
        self.register_buffer("cos_cached", cos.view(1, 1, n, rd), persistent=False)
        self.register_buffer("sin_cached", sin.view(1, 1, n, rd), persistent=False)

    def _init_causal_tail_modules(self, top_k: int) -> None:
        n = self.seq_len
        k_eff = min(int(top_k), n)
        valid = torch.arange(1, n + 1, dtype=torch.float32)
        n_dropped = torch.clamp(valid - float(k_eff), min=1.0)
        inv_dropped = 1.0 / n_dropped
        valid_over_dropped = valid / n_dropped
        self.register_buffer("inv_n_dropped", inv_dropped.view(1, 1, n, 1), persistent=False)
        self.register_buffer("valid_over_dropped", valid_over_dropped.view(1, 1, n, 1), persistent=False)
        self.valid_key_mean = AffineContract(
            "nk,bhkd->bhnd",
            build_causal_valid_key_mean_matrix(n, dtype=get_surgery_dtype()),
        )
        self.scale_mean_v_by_tail = AffineHadamard()
        self.scale_q_tail_by_inv_dropped = AffineHadamard()
        self.scale_tail_by_valid_over_dropped = AffineHadamard()
        self.scale_top_probs_by_valid = AffineHadamard()
        self.adj_probs_contract = AffineContract(
            "i,...i->...",
            torch.tensor([1.0, -1.0], dtype=get_surgery_dtype()),
        )
        self.attn_tail_sum = AffineContract(
            "i,...i->...",
            torch.tensor([1.0, 1.0], dtype=get_surgery_dtype()),
        )

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return RoutingCat((self.rotate_neg(x2), x1), dim=-1)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rd = self.rotary_ndims
        if rd == 0:
            return q, k
        cos = self.cos_cached.to(device=q.device, dtype=q.dtype)
        sin = self.sin_cached.to(device=q.device, dtype=q.dtype)
        q_rot, q_pass = q[..., :rd], q[..., rd:]
        k_rot, k_pass = k[..., :rd], k[..., rd:]
        q_embed = self.rope_sum(
            RoutingStack(
                (self.rope_cos_mul(q_rot, cos), self.rope_sin_mul(self._rotate_half(q_rot), sin)),
                dim=-1,
            )
        )
        k_embed = self.rope_sum(
            RoutingStack(
                (self.rope_cos_mul(k_rot, cos), self.rope_sin_mul(self._rotate_half(k_rot), sin)),
                dim=-1,
            )
        )
        return RoutingCat((q_embed, q_pass), dim=-1), RoutingCat((k_embed, k_pass), dim=-1)

    def _split_qkv(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, _ = hidden.shape
        # NeoX layout: (B, N, H, 3*D) -> (B, H, N, 3*D) -> chunk on last dim.
        qkv = self.query_key_value(hidden).view(b, n, self.num_heads, 3 * self.head_dim)
        qkv = qkv.transpose(1, 2)
        return tuple(qkv.chunk(3, dim=-1))  # type: ignore[return-value]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        if n != self.seq_len:
            raise ValueError(f"SurgeryCausalAttention expected seq_len={self.seq_len}, got {n}")
        q, k, v = self._split_qkv(x)
        q, k = self._apply_rotary(q, k)
        mask = self.causal_mask.to(device=q.device, dtype=q.dtype)

        if not self.use_attention_surgery:
            s = self.attn_scale.to(device=q.device, dtype=q.dtype)
            qs = q * s
            kt = RoutingTranspose(k, -2, -1)
            scores = self.matmul(qs, kt) if self.allow_matmul else torch.matmul(qs, kt)
            scores = RoutingAdd(scores, mask)
            attn = scores.softmax(dim=-1)
            attn = self.attn_drop(attn)
            attn = self.matmul(attn, v) if self.allow_matmul else torch.matmul(attn, v)
        elif self.use_surgery_softmax:
            if self.router is not None and self.gibbs.top_k < n and not self.gibbs.use_exact_tail_mass:
                # The causal mask differs per row, so the support is the top-k of the masked selection;
                # exact scores are formed only on it.
                _sv, support = RoutingTopK(
                    self.router(q, k, mask), k=self.gibbs.top_k, dim=-1, largest=True, sorted=False
                )
                mask_sel = RoutingGather(RoutingExpand(mask, b, self.num_heads, n, n), -1, support)
                scores_sel = self.mask_scores(self.dot.selected(q, k, support), mask_sel)
                probs, idx, q_tail = self.gibbs.forward_selected(scores_sel, support, n)
            else:
                scores = self.mask_scores(self.dot(q, k), mask)
                selection = self.router(q, k, mask) if self.router is not None else None
                probs, idx, q_tail = self.gibbs(scores, selection)
            # When valid_keys < top_k, topk may return future (-inf) filler indices with ~0
            # probability. Zero them explicitly so no mass is routed past the causal boundary.
            query_pos = torch.arange(n, device=idx.device).view(1, 1, n, 1)
            valid = (idx <= query_pos).to(dtype=probs.dtype)
            probs = self.scale_top_probs_by_valid(probs, valid)
            inv = self.inv_n_dropped.to(device=probs.device, dtype=probs.dtype)
            vod = self.valid_over_dropped.to(device=probs.device, dtype=probs.dtype)
            q_tail_adj = self.scale_q_tail_by_inv_dropped(q_tail, inv)
            _p, _qta = RoutingBroadcastTensors(probs, q_tail_adj)
            adjusted_probs = self.adj_probs_contract(RoutingStack((_p, _qta), dim=-1))
            adjusted_probs = self.scale_top_probs_by_valid(adjusted_probs, valid)
            attn_top = self.sparse_mix(adjusted_probs, idx, v)
            mean_v = self.valid_key_mean(v)
            tail_scaled = self.scale_mean_v_by_tail(mean_v, q_tail)
            tail_contrib = self.scale_tail_by_valid_over_dropped(tail_scaled, vod)
            _a, _t = RoutingBroadcastTensors(attn_top, tail_contrib)
            attn = self.attn_tail_sum(RoutingStack((_a, _t), dim=-1))
        else:
            scores = self.mask_scores(self.dot(q, k), mask)
            attn = scores.softmax(dim=-1)
            attn = self.attn_drop(attn)
            if self.allow_matmul:
                attn = self.matmul(attn, v)
            else:
                _, _, nq, nk = attn.shape
                v_b = RoutingExpand(RoutingUnsqueeze(v, 2), -1, -1, nq, nk, -1)
                attn = (RoutingUnsqueeze(attn, -1) * v_b).sum(dim=3)

        attn = RoutingTranspose(attn, 1, 2).reshape(b, n, c)
        return self.dense(attn)


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
        q = self.quantizer(*inputs)
        if getattr(self.accumulator, "ptq_dequantized_inputs", False):
            q = tuple(t * self.quantizer.input_scale[i] for i, t in enumerate(q))
        acc = self.accumulator(*q)
        if self.out_bcast_shape is not None:
            bias = self.out_bias.view(self.out_bcast_shape)
            out = acc + bias if self.skip_out_scale else self.out_scale.view(self.out_bcast_shape) * acc + bias
        else:
            out = acc + self.out_bias if self.skip_out_scale else self.out_scale * acc + self.out_bias
        return out.to(dtype=inputs[0].dtype)
