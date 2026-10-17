"""Surgery calibration (Gibbs tail mass, LayerNorm rewrite check) and PTQ calibration (range and dequant fits)."""

from __future__ import annotations

import itertools
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, Iterator, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention as TimmAttention
from torch.utils.data import DataLoader

from transformer_surgery.internal.metrics import (
    jeffreys_distance_sparse_teacher,
    jeffreys_naive_topk,
    logit_stable_rows,
)
from transformer_surgery.internal.util import (
    get_device,
    get_surgery_dtype,
    maybe_surgery_cuda_autocast,
    to_model_input,
)
from transformer_surgery.ops import (
    AffineContract,
    AffineFixedMix,
    AffineHadamard,
    AffineMatMul,
    AffineScale,
    AffineScaleBias,
    PTQ_FP32_EXACT_INT,
    CalibratedAffinePTQWrapper,
    GibbsTopKSoftmax,
    OutputIdentity,
    RewrittenLayerNorm,
    ptq_check_fp32,
    ptq_quantize_proxy,
    ptq_signed_qrange,
)

if TYPE_CHECKING:
    from transformer_surgery.cli.ptq_config import PTQSurgeryConfig
    from transformer_surgery.cli.surgery_config import SurgeryConfig


# ---------------------------------------------------------------------------
# Calibration batches and hooked forward passes
# ---------------------------------------------------------------------------


def calibration_batch_indices(
    total: int, requested: Optional[int], *, random: bool = False, generator: Optional[torch.Generator] = None
) -> List[int]:
    """All ``total`` batches when ``requested`` is None, else the first ``requested`` or a random sorted subset."""
    if requested is None:
        return list(range(total))
    if int(requested) < 1:
        raise ValueError(f"calibration batch count must be >= 1 or null, got {requested!r}")
    keep = min(total, int(requested))
    if random:
        return sorted(torch.randperm(total, generator=generator)[:keep].tolist())
    return list(range(keep))


@torch.no_grad()
def run_hooked(model: nn.Module, loader: Any, batch_indices: Sequence[int], handles: List[Any]) -> int:
    """Run ``model`` on the selected loader batches, then remove the hook ``handles``; returns the batch count."""
    try:
        selected = set(batch_indices)
        if not selected:
            return 0
        model.eval()
        device, dt = get_device(), get_surgery_dtype()
        last = max(selected)
        n_run = 0
        for i, (x, _y) in enumerate(loader):
            if i > last:
                break
            if i in selected:
                with maybe_surgery_cuda_autocast(device, dt):
                    model(to_model_input(x, model))
                n_run += 1
        return n_run
    finally:
        for h in handles:
            h.remove()


def gibbs_attention_modules(model: nn.Module) -> List[nn.Module]:
    """Attention modules with a Gibbs top-k softmax, in module (block) order."""
    return [m for m in model.modules() if isinstance(getattr(m, "gibbs", None), GibbsTopKSoftmax)]


@contextmanager
def dense_gibbs(attns: Sequence[nn.Module]) -> Iterator[None]:
    """Run every Gibbs softmax at full support (``top_k = seq_len``), which wraps the dense reference."""
    saved = [a.gibbs.top_k for a in attns]
    for a in attns:
        a.gibbs.top_k = a.gibbs.seq_len
    try:
        yield
    finally:
        for a, top_k in zip(attns, saved):
            a.gibbs.top_k = top_k


# ---------------------------------------------------------------------------
# Gibbs tail mass and LayerNorm rewrite statistics
# ---------------------------------------------------------------------------

# Source of the scores of a surgery calibration; the Pythia adapter relabels its student calibration.
CalibrationMode = Literal["reference", "surgery_student", "skipped_no_gibbs_attention", "pythia_surgery_student_causal"]

# Upper clamp of an applied tail mass, so the kept top-k mass 1 - q stays positive.
_MAX_TAIL = 1.0 - 1e-7


def clamp_tail(value: float) -> float:
    return max(0.0, min(float(value), _MAX_TAIL))


def sample_topk_scores(
    scores: torch.Tensor, top_k: int, *, max_rows: int = 4096
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Random score rows (float32) with their stabilized top-k values and indices, ``nk`` and the effective ``k``."""
    flat = scores.reshape(-1, scores.shape[-1])
    rows = min(flat.shape[0], max_rows)
    teacher = flat[torch.randperm(flat.shape[0], device=flat.device)[:rows]].float()
    nk = teacher.shape[-1]
    k_top = min(int(top_k), nk)
    vals, idx = torch.topk(logit_stable_rows(teacher), k=k_top, dim=-1, largest=True, sorted=True)
    return teacher, vals, idx, nk, k_top


def topk_tail_mass(teacher: torch.Tensor, idx: torch.Tensor, nk: int, k_top: int) -> torch.Tensor:
    """Dense-softmax mass outside the top-k support ``idx``, per row."""
    if nk <= k_top:
        return torch.zeros(teacher.shape[0], device=teacher.device)
    top_mass = torch.softmax(logit_stable_rows(teacher), dim=-1).gather(1, idx).sum(dim=-1)
    return (1.0 - top_mass).clamp(0.0, 1.0)


@dataclass
class TailCalibration:
    """Gibbs tail mass and LayerNorm rewrite statistics of a surgery calibration (metadata ``calibration`` keys)."""

    calibration_mode: CalibrationMode
    disable_calib_gibbs_tail_prob: bool
    use_exact_tail_mass: bool
    gibbs_tail_calibration_batches_requested: Optional[int]
    gibbs_tail_calibration_batches: int
    gibbs_tail_prob_eps_configured: float
    gibbs_tail_prob_eps_metric: float
    calibration_loader_split: Literal["train"] = "train"
    ln_rewrite_mse_layer0_minibatch: Optional[float] = None
    ln_rewrite_mse_all_norms_mean: Optional[float] = None
    # Dense-softmax mass outside the top-k support, per block and pooled; gathered unless tail calibration is off.
    gibbs_tail_prob_eps_exact_mean_by_block: Optional[List[float]] = None
    gibbs_tail_prob_eps_exact_std_by_block: Optional[List[float]] = None
    gibbs_tail_prob_eps_exact_min_by_block: Optional[List[float]] = None
    gibbs_tail_prob_eps_exact_max_by_block: Optional[List[float]] = None
    gibbs_tail_prob_eps_exact_rows_by_block: Optional[List[int]] = None
    gibbs_tail_prob_eps_exact_mean: Optional[float] = None
    gibbs_tail_prob_eps_exact_std: Optional[float] = None
    gibbs_tail_prob_eps_exact_min: Optional[float] = None
    gibbs_tail_prob_eps_exact_max: Optional[float] = None
    # Calibrated mode (no exact runtime tail, calibration on): the per-block means become the model's tail masses.
    gibbs_tail_prob_eps_calibrated_by_block: Optional[List[float]] = None
    gibbs_tail_prob_eps_calibrated_min_by_block: Optional[List[float]] = None
    gibbs_tail_prob_eps_calibrated_max_by_block: Optional[List[float]] = None
    gibbs_tail_prob_eps_calibration_rows_by_block: Optional[List[int]] = None
    gibbs_tail_prob_eps_calibrated_mean: Optional[float] = None
    # Sparse vs naive top-k Jeffreys on synthetic Gaussian scores of block 0's shape.
    jeffreys_gibbs_mean_synthetic: Optional[float] = None
    jeffreys_naive_mean_synthetic: Optional[float] = None
    jeffreys_improvement_naive_minus_gibbs_synthetic: Optional[float] = None


@dataclass
class _TailBlock:
    """Running sums of the tail mass of one block."""

    total: float = 0.0
    total_sq: float = 0.0
    rows: int = 0
    min: float = float("inf")
    max: float = float("-inf")


class _TailAndLayerNormStats:
    """Per-block omitted-tail mass of the top-k support and the LayerNorm rewrite error over calibration batches."""

    def __init__(self, cfg: "SurgeryConfig") -> None:
        self.cfg = cfg
        self.gather_tail = cfg.use_exact_tail_mass or not cfg.disable_calib_gibbs_tail_prob
        self.blocks: Dict[int, _TailBlock] = {}
        self.block0_nk = 0
        self.block0_k = 0
        self.ln_mse_sum = 0.0
        self.ln_count = 0
        self.ln_layer0: Optional[float] = None

    def scores(self, block: int, scores: torch.Tensor) -> None:
        if block != 0 and not self.gather_tail:
            return
        teacher, _vals, idx, nk, k_top = sample_topk_scores(scores, self.cfg.top_k)
        if block == 0:
            self.block0_nk, self.block0_k = nk, k_top
        if self.gather_tail:
            tail = topk_tail_mass(teacher, idx, nk, k_top)
            acc = self.blocks.setdefault(block, _TailBlock())
            acc.total += float(tail.mean()) * tail.numel()
            acc.total_sq += float(tail.pow(2).sum())
            acc.rows += tail.numel()
            acc.min = min(acc.min, float(tail.min()))
            acc.max = max(acc.max, float(tail.max()))

    def layernorm(self, rewritten: torch.Tensor, reference: torch.Tensor, layer0: bool) -> None:
        mse = float((rewritten - reference).pow(2).mean().item())
        self.ln_mse_sum += mse
        self.ln_count += 1
        if layer0 and self.ln_layer0 is None:
            self.ln_layer0 = mse

    def report(self, batches: int, mode: CalibrationMode) -> TailCalibration:
        cfg = self.cfg
        out = TailCalibration(
            calibration_mode=mode,
            disable_calib_gibbs_tail_prob=cfg.disable_calib_gibbs_tail_prob,
            use_exact_tail_mass=cfg.use_exact_tail_mass,
            gibbs_tail_calibration_batches_requested=cfg.gibbs_tail_calibration_batches,
            gibbs_tail_calibration_batches=batches,
            gibbs_tail_prob_eps_configured=cfg.gibbs_tail_prob_eps,
            gibbs_tail_prob_eps_metric=float(cfg.gibbs_tail_prob_eps),
            ln_rewrite_mse_layer0_minibatch=self.ln_layer0,
            ln_rewrite_mse_all_norms_mean=self.ln_mse_sum / self.ln_count if self.ln_count else None,
        )
        if self.blocks:
            blocks = [self.blocks[b] for b in sorted(self.blocks)]
            s = torch.tensor([b.total for b in blocks], dtype=torch.float64)
            n = torch.tensor([b.rows for b in blocks], dtype=torch.float64)
            sq = torch.tensor([b.total_sq for b in blocks], dtype=torch.float64)
            means_t = s / n
            means = means_t.tolist()
            mins, maxs, rows = [b.min for b in blocks], [b.max for b in blocks], [b.rows for b in blocks]
            pooled_mean = s.sum() / n.sum()
            out.gibbs_tail_prob_eps_exact_mean_by_block = means
            out.gibbs_tail_prob_eps_exact_std_by_block = (sq / n - means_t.square()).clamp_min(0.0).sqrt().tolist()
            out.gibbs_tail_prob_eps_exact_min_by_block = mins
            out.gibbs_tail_prob_eps_exact_max_by_block = maxs
            out.gibbs_tail_prob_eps_exact_rows_by_block = rows
            out.gibbs_tail_prob_eps_exact_mean = float(pooled_mean)
            out.gibbs_tail_prob_eps_exact_std = float((sq.sum() / n.sum() - pooled_mean.square()).clamp_min(0.0).sqrt())
            out.gibbs_tail_prob_eps_exact_min = min(mins)
            out.gibbs_tail_prob_eps_exact_max = max(maxs)
            if not cfg.disable_calib_gibbs_tail_prob and not cfg.use_exact_tail_mass:
                out.gibbs_tail_prob_eps_calibrated_by_block = means
                out.gibbs_tail_prob_eps_calibrated_min_by_block = mins
                out.gibbs_tail_prob_eps_calibrated_max_by_block = maxs
                out.gibbs_tail_prob_eps_calibration_rows_by_block = rows
                out.gibbs_tail_prob_eps_calibrated_mean = float(pooled_mean)
            out.gibbs_tail_prob_eps_metric = float(means[0])
        if self.block0_k > 0:
            teacher = torch.randn(4096, self.block0_nk, device=get_device(), dtype=get_surgery_dtype())
            vals, idx = torch.topk(logit_stable_rows(teacher), k=self.block0_k, dim=-1, largest=True, sorted=True)
            tail_prob = clamp_tail(out.gibbs_tail_prob_eps_metric)
            j_gibbs = jeffreys_distance_sparse_teacher(
                teacher, vals, idx, self.block0_nk, self.block0_k, gibbs_tail_prob_eps=tail_prob
            ).mean()
            j_naive = jeffreys_naive_topk(teacher, vals, idx, self.block0_nk, self.block0_k).mean()
            out.jeffreys_gibbs_mean_synthetic = float(j_gibbs)
            out.jeffreys_naive_mean_synthetic = float(j_naive)
            out.jeffreys_improvement_naive_minus_gibbs_synthetic = float(j_naive - j_gibbs)
        return out


def fused_qkv_attention_qk_scores(attn: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Scaled QK logits ``[B, H, N, N]`` of a fused-``qkv`` attention (``qkv``, ``num_heads``, ``scale``) on ``x``."""
    b, n, c = x.shape
    qkv = attn.qkv(x).reshape(b, n, 3, attn.num_heads, c // attn.num_heads).permute(2, 0, 3, 1, 4)
    return (qkv[0] @ qkv[1].transpose(-2, -1)) * float(attn.scale)


@torch.no_grad()
def calibrate_reference(reference: nn.Module, loader: DataLoader, cfg: "SurgeryConfig") -> TailCalibration:
    """Gibbs tail and LayerNorm rewrite statistics from a stock reference, by hooks on its attention and LayerNorms.

    Scores are read at each ``nn.Softmax`` with 4-D ``[B, H, Nq, Nk]`` input (MambaIR window attention) and from the
    fused ``qkv`` of each timm ``Attention`` (DeiT-Tiny). Score sources are indexed in firing order, which matches the
    depth-first ``GibbsTopKSoftmax`` order of the surgery student, so :func:`apply_gibbs_tail_calibration` aligns
    per-block tail masses. Each ``nn.LayerNorm`` is compared with a :class:`RewrittenLayerNorm` of the same affine.
    """
    acc = _TailAndLayerNormStats(cfg)
    block_of: Dict[int, int] = {}
    rewritten: Dict[int, RewrittenLayerNorm] = {}
    layernorms = [m for m in reference.modules() if isinstance(m, nn.LayerNorm)]

    def on_scores(mod: nn.Module, scores: torch.Tensor) -> None:
        if scores.dim() == 4:
            acc.scores(block_of.setdefault(id(mod), len(block_of)), scores)

    def on_layernorm(mod: nn.LayerNorm, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        h = inputs[0]
        rw = rewritten.get(id(mod))
        if rw is None:
            rw = RewrittenLayerNorm(mod.normalized_shape[0], eps=cfg.eps, allow_matmul=cfg.allow_matmul)
            rw = rewritten[id(mod)] = rw.to(device=h.device, dtype=h.dtype)
            rw.affine.load_state_dict(mod.state_dict())
        acc.layernorm(rw(h), output, mod is layernorms[0])

    handles = []
    for m in reference.modules():
        if isinstance(m, nn.Softmax):
            handles.append(m.register_forward_hook(lambda mod, inputs, _out: on_scores(mod, inputs[0])))
        elif isinstance(m, TimmAttention):
            handles.append(
                m.register_forward_pre_hook(
                    lambda mod, inputs: on_scores(mod, fused_qkv_attention_qk_scores(mod, inputs[0]))
                )
            )
        elif isinstance(m, nn.LayerNorm):
            handles.append(m.register_forward_hook(on_layernorm))
    batches = calibration_batch_indices(len(loader), cfg.gibbs_tail_calibration_batches)
    processed = run_hooked(reference, loader, batches, handles)
    if processed == 0:
        raise ValueError("gibbs tail calibration requires at least one training batch")
    return acc.report(processed, "reference")


@torch.no_grad()
def calibrate_surgery_student(model: nn.Module, loader: DataLoader, cfg: "SurgeryConfig") -> TailCalibration:
    """Gibbs tail and LayerNorm rewrite statistics from the surgery student run at full support (the wrapped reference).

    Scores are read at each attention's ``mask_scores`` (after the causal mask) or ``dot`` output; tail sampling uses
    ``cfg.top_k``. Each :class:`RewrittenLayerNorm` is compared with ``F.layer_norm`` of the same affine. Apply the
    calibrated tail masses with :func:`apply_gibbs_tail_calibration`.
    """
    acc = _TailAndLayerNormStats(cfg)
    attns = gibbs_attention_modules(model)
    if not attns:
        return acc.report(0, "skipped_no_gibbs_attention")
    layernorms = [m for m in model.modules() if isinstance(m, RewrittenLayerNorm)]

    def on_layernorm(mod: RewrittenLayerNorm, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        ref = F.layer_norm(inputs[0], mod.normalized_shape, mod.affine.weight, mod.affine.bias, cfg.eps)
        acc.layernorm(output, ref, mod is layernorms[0])

    handles = [
        getattr(a, "mask_scores", a.dot).register_forward_hook(lambda _m, _i, out, block=block: acc.scores(block, out))
        for block, a in enumerate(attns)
    ]
    handles += [m.register_forward_hook(on_layernorm) for m in layernorms]
    batches = calibration_batch_indices(len(loader), cfg.gibbs_tail_calibration_batches, random=True)
    with dense_gibbs(attns):
        processed = run_hooked(model, loader, batches, handles)
    if processed == 0:
        raise ValueError("gibbs tail calibration requires at least one training batch")
    return acc.report(processed, "surgery_student")


@dataclass
class AppliedTail:
    """Tail masses copied into the Gibbs softmax modules (metadata ``calibration`` keys)."""

    gibbs_tail_prob_eps_applied_by_block: List[float]
    gibbs_tail_prob_eps_applied_mean: float


def apply_gibbs_tail_calibration(model: nn.Module, values: Sequence[float]) -> Optional[AppliedTail]:
    """Copy per-block tail masses into the Gibbs softmax modules; the last value covers extra blocks."""
    gibbs = [m for m in model.modules() if isinstance(m, GibbsTopKSoftmax)]
    if not gibbs or not values:
        return None
    applied = [clamp_tail(values[min(i, len(values) - 1)]) for i in range(len(gibbs))]
    with torch.no_grad():
        for g, value in zip(gibbs, applied):
            g.gibbs_tail_prob_eps.fill_(value)
    return AppliedTail(applied, sum(applied) / len(applied))


# ---------------------------------------------------------------------------
# PTQ calibration
# ---------------------------------------------------------------------------

PTQKind = Literal[
    "linear",
    "conv2d",
    "affine_scale",
    "affine_scale_bias",
    "affine_fixed_mix",
    "affine_contract",
    "matmul",
    "matmul_hadamard",
    "output_identity",
]
# Input-code width group of a node kind (``ptq_activation_bits``).
ActivationGroup = Literal["linear", "affine", "matmul", "output"]
# Output map of a wrapper: Linear/Conv2d scale ``s_in * s_w`` with a fitted bias, or the OLS affine fit.
OutScaleMode = Literal["analytical_s_in_times_s_w", "ols_affine"]

PTQ_LINEAR_KINDS: FrozenSet[PTQKind] = frozenset({"linear", "conv2d"})
PTQ_MATMUL_KINDS: FrozenSet[PTQKind] = frozenset({"matmul", "matmul_hadamard"})

_PTQ_KIND_BY_TYPE: Tuple[Tuple[type, PTQKind], ...] = (
    (nn.Linear, "linear"),
    (nn.Conv2d, "conv2d"),
    (OutputIdentity, "output_identity"),  # before its base class AffineScale
    (AffineScale, "affine_scale"),
    (AffineScaleBias, "affine_scale_bias"),
    (AffineFixedMix, "affine_fixed_mix"),
    (AffineContract, "affine_contract"),
    (AffineMatMul, "matmul"),
    (AffineHadamard, "matmul_hadamard"),
)

# Fixed weight of each accumulator kind; the matmul kinds have none.
_WEIGHT_ATTR: Dict[PTQKind, str] = {
    "linear": "weight",
    "conv2d": "weight",
    "affine_scale": "scale",
    "affine_scale_bias": "weight",
    "affine_fixed_mix": "weight",
    "affine_contract": "coeff",
    "output_identity": "scale",
}


def ptq_module_kind(module: nn.Module) -> Optional[PTQKind]:
    for cls, label in _PTQ_KIND_BY_TYPE:
        if isinstance(module, cls):
            return label
    return None


def ptq_activation_group(kind: PTQKind) -> ActivationGroup:
    if kind in PTQ_LINEAR_KINDS:
        return "linear"
    if kind in PTQ_MATMUL_KINDS:
        return "matmul"
    if kind == "output_identity":
        return "output"
    return "affine"


def ptq_activation_bits(kind: PTQKind, cfg: "PTQSurgeryConfig") -> int:
    """Input-code width of a node kind: ``activation_bits`` for Linear/Conv2d; the matmul group, the compiled
    fixed-affine group and the model output use ``matmul_activation_bits``, ``compiled_affine_input_bits`` and
    ``output_bits`` when set, else ``activation_bits``."""
    group = ptq_activation_group(kind)
    bits = {
        "matmul": cfg.matmul_activation_bits,
        "affine": cfg.compiled_affine_input_bits,
        "output": cfg.output_bits,
    }.get(group)
    return int(cfg.activation_bits if bits is None else bits)


@dataclass
class PTQAffineFitStats:
    """Moments of the least-squares fit ``out ~ scale * acc + bias`` per channel."""

    count: int = 0
    sums: Optional[List[torch.Tensor]] = None  # sum x, sum y, sum x^2, sum xy

    def _add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        # One full-size product at a time: each is summed before the next is formed.
        moments = [x.sum(dim=0), y.sum(dim=0), (x * x).sum(dim=0), (x * y).sum(dim=0)]
        moments = [m.to(dtype=torch.float32, device="cpu") for m in moments]
        self.sums = moments if self.sums is None else [a + b for a, b in zip(self.sums, moments)]
        self.count += int(x.shape[0])

    def update(self, acc: torch.Tensor, out: torch.Tensor, *, channel_axis: Optional[int]) -> None:
        axis = _channel_axis(acc, out, channel_axis)
        if axis is None:
            self._add(acc.reshape(-1, 1).float(), out.reshape(-1, 1).float())
        else:
            self._add(
                acc.movedim(axis, -1).reshape(-1, acc.shape[axis]).float(),
                out.movedim(axis, -1).reshape(-1, out.shape[axis]).float(),
            )

    def fit(self, var_eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.count <= 0 or self.sums is None:
            return torch.tensor(1.0), torch.tensor(0.0)
        sum_x, sum_y, sum_x2, sum_xy = self.sums
        n = float(self.count)
        mx, my = sum_x / n, sum_y / n
        var = sum_x2 / n - mx * mx
        cov = sum_xy / n - mx * my
        if var.numel() == 1 and float(var.abs().item()) < var_eps:
            scale = (sum_xy / n) / (sum_x2 / n).clamp_min(var_eps)
        else:
            scale = torch.where(var.abs() < var_eps, torch.zeros_like(var), cov / var.clamp_min(var_eps))
        return scale.float(), (my - scale * mx).float()


@dataclass
class PTQBiasFitStats:
    """Mean residual ``out - out_scale * acc`` per channel (Linear/Conv2d with an analytical output scale)."""

    count: int = 0
    residual_sum: Optional[torch.Tensor] = None

    def update(
        self, acc: torch.Tensor, out: torch.Tensor, out_scale: torch.Tensor, *, channel_axis: Optional[int]
    ) -> None:
        out_scale = out_scale.to(device=acc.device, dtype=torch.float32)
        acc_fp, out_fp = acc.float(), out.float()
        # The output bias is fitted per channel whenever the output has a channel axis, also for a
        # scalar out_scale (per-tensor weights): the layer bias is a per-channel vector in both cases.
        axis = _channel_axis(acc, out, channel_axis)
        if axis is None:
            residual = out_fp - out_scale.reshape(()) * acc_fp
            total, count = residual.sum().reshape(()), residual.numel()
        else:
            residual = out_fp - _broadcast_axis(out_scale, acc_fp, axis) * acc_fp
            reduce_dims = tuple(i for i in range(residual.ndim) if i != axis)
            total = residual.sum(dim=reduce_dims) if reduce_dims else residual
            count = residual.numel() // residual.shape[axis]
        total = total.to(dtype=torch.float32, device="cpu")
        self.residual_sum = total if self.residual_sum is None else self.residual_sum + total
        self.count += int(count)

    def bias(self) -> torch.Tensor:
        if self.count <= 0 or self.residual_sum is None:
            return torch.tensor(0.0)
        return (self.residual_sum / float(self.count)).float()


def _channel_axis(acc: torch.Tensor, out: torch.Tensor, channel_axis: Optional[int]) -> Optional[int]:
    """Non-negative channel axis shared by ``acc`` and ``out``, or None for a per-tensor fit."""
    if channel_axis is None or acc.ndim == 0:
        return None
    axis = channel_axis if channel_axis >= 0 else acc.ndim + channel_axis
    if 0 <= axis < acc.ndim and axis < out.ndim and acc.shape[axis] == out.shape[axis]:
        return axis
    return None


@dataclass
class PTQNodeStats:
    name: str
    kind: PTQKind
    input_arity: int = 0
    input_max_abs: List[torch.Tensor] = field(default_factory=list)
    output_rank: int = 0
    output_channel_axis: Optional[int] = None
    examples_total: int = 0
    affine_fit: PTQAffineFitStats = field(default_factory=PTQAffineFitStats)
    bias_fit: PTQBiasFitStats = field(default_factory=PTQBiasFitStats)
    # Largest sum_i |q_i| |w_i| over the calibration data, a bound on every partial sum of the integer accumulation;
    # None for nodes whose inputs are rescaled codes (AffineCombine).
    integer_sum_bound: Optional[float] = None

    def observe_range(self, inputs: Sequence[torch.Tensor], output: torch.Tensor) -> None:
        if self.input_arity == 0:
            self.input_arity = len(inputs)
            self.input_max_abs = [torch.tensor(0.0) for _ in inputs]
        if len(inputs) != self.input_arity:
            raise ValueError(f"Input arity changed during PTQ calibration for node {self.name}")
        for i, tensor in enumerate(inputs):
            self.input_max_abs[i] = torch.maximum(self.input_max_abs[i], tensor.detach().float().abs().max().cpu())
        self.output_rank = output.ndim
        self.output_channel_axis = _default_output_channel_axis(self.kind, output)
        self.examples_total += 1 if output.ndim == 0 else int(output.shape[0])


@dataclass
class PTQNodeSetup:
    name: str
    kind: PTQKind
    activation_bits: int
    weight_bits: int
    output_channel_axis: Optional[int]
    output_rank: int
    per_output_channel_weights_config: bool
    per_output_channel_weights_effective: bool
    input_scale: torch.Tensor
    weight_scale: torch.Tensor
    q_weight: Optional[torch.Tensor]

    @property
    def per_channel_output_affine(self) -> bool:
        return self.output_channel_axis is not None


@dataclass
class PTQNodeMeta:
    """Quantization parameters of one wrapped node (PTQ metadata ``ptq_nodes`` entry). Scales, zero points and biases
    hold one value per tensor or per channel (see ``per_output_channel_weights_effective`` and
    ``per_channel_output_affine``)."""

    name: str
    kind: PTQKind
    activation_group: ActivationGroup
    activation_bits: int
    weight_bits: int
    per_output_channel_weights_config: bool
    per_output_channel_weights_effective: bool
    per_channel_output_affine: bool
    output_channel_axis: Optional[int]
    input_scale: List[float]
    input_zero_point: List[float]
    weight_scale: List[float]
    weight_zero_point: List[float]
    out_scale_mode: OutScaleMode
    out_scale: List[float]
    out_scale_baked_into_weight: bool
    out_bias: List[float]
    integer_sum_bound: Optional[float]
    stride: Optional[List[int]]
    padding: Optional[List[int]]
    dilation: Optional[List[int]]
    groups: int
    einsum_equation: Optional[str]


@dataclass
class PTQReloadConfig:
    """What a PTQ checkpoint needs to rebuild one wrapper skeleton before ``load_state_dict``."""

    name: str
    kind: PTQKind
    activation_bits: int
    out_bcast_shape: Optional[List[int]]
    skip_out_scale: bool
    state_shapes: Dict[str, List[int]]


@dataclass
class PTQBuiltWrapper:
    module: CalibratedAffinePTQWrapper
    metadata: PTQNodeMeta
    reload_config: PTQReloadConfig


def _tensor_hook(fn: Callable[[nn.Module, Tuple[torch.Tensor, ...], torch.Tensor], None]) -> Callable[..., None]:
    """Forward hook that passes only the tensor inputs, and only for a tensor output."""

    def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        tensors = tuple(x for x in inputs if torch.is_tensor(x))
        if torch.is_tensor(output) and tensors:
            fn(module, tensors, output)

    return hook


@torch.no_grad()
def gather_ptq_range_moments(
    model: nn.Module, loader: Any, selected: Dict[str, PTQKind], batch_indices: Sequence[int]
) -> Dict[str, PTQNodeStats]:
    """Input ranges and output layout of each selected node over the calibration batches."""
    stats = {name: PTQNodeStats(name=name, kind=kind) for name, kind in selected.items()}

    def observe(name: str) -> Callable[..., None]:
        return _tensor_hook(lambda _m, inp, out: stats[name].observe_range(inp, out))

    handles = [model.get_submodule(n).register_forward_hook(observe(n)) for n in selected]
    run_hooked(model, loader, batch_indices, handles)
    return stats


@torch.no_grad()
def gather_ptq_dequant_moments(
    model: nn.Module,
    loader: Any,
    selected: Dict[str, PTQKind],
    stats: Dict[str, PTQNodeStats],
    setups: Dict[str, PTQNodeSetup],
    batch_indices: Sequence[int],
) -> None:
    """Output dequant fits of each selected node: its quantized accumulation against the float output."""

    def observe(name: str) -> Callable[..., None]:
        return _tensor_hook(lambda m, inp, out: _observe_ptq_dequant(m, stats[name], setups[name], inp, out))

    handles = [model.get_submodule(n).register_forward_hook(observe(n)) for n in selected]
    run_hooked(model, loader, batch_indices, handles)


def _observe_ptq_dequant(
    module: nn.Module,
    stats: PTQNodeStats,
    setup: PTQNodeSetup,
    inputs: Sequence[torch.Tensor],
    output: torch.Tensor,
) -> None:
    with _autocast_off():
        _observe_ptq_dequant_fp32(module, stats, setup, [inp.float() for inp in inputs], output)


@contextmanager
def _autocast_off() -> Iterator[None]:
    with torch.autocast("cuda", enabled=False), torch.autocast("cpu", enabled=False):
        yield


def _observe_ptq_dequant_fp32(
    module: nn.Module,
    stats: PTQNodeStats,
    setup: PTQNodeSetup,
    inputs: Sequence[torch.Tensor],
    output: torch.Tensor,
) -> None:
    """Quantized accumulation of one node in fp32 (autocast off) on the inputs of the float reference, fitted against
    the reference output; the reference itself runs in its surgery dtype."""
    ptq_check_fp32(f"{stats.name} calibration input", *inputs)
    q_inputs = [ptq_quantize_proxy(inp, setup.input_scale[i], setup.activation_bits) for i, inp in enumerate(inputs)]
    q_weight = None if setup.q_weight is None else setup.q_weight.to(device=output.device)
    if getattr(module, "ptq_dequantized_inputs", False):
        # Several inputs on different grids (AffineCombine): rescale each code by its own input scale before the sum.
        q_inputs = [q * setup.input_scale[i].to(device=q.device) for i, q in enumerate(q_inputs)]
    else:
        # Integer codes times integer weight codes: the sum of absolute products bounds every partial sum.
        abs_weight = None if q_weight is None else q_weight.abs()
        bound = float(_accumulator_forward(module, setup.kind, abs_weight, [q.abs() for q in q_inputs]).max())
        stats.integer_sum_bound = max(bound, stats.integer_sum_bound or 0.0)
    acc = _accumulator_forward(module, setup.kind, q_weight, q_inputs)
    ptq_check_fp32(f"{stats.name} calibration accumulation", acc)
    if setup.kind in PTQ_LINEAR_KINDS:
        out_scale = _linear_conv_out_scale(setup.input_scale, setup.weight_scale)
        stats.bias_fit.update(acc, output, out_scale, channel_axis=setup.output_channel_axis)
    else:
        stats.affine_fit.update(acc, output, channel_axis=setup.output_channel_axis)


def build_ptq_node_setup(name: str, module: nn.Module, stats: PTQNodeStats, cfg: "PTQSurgeryConfig") -> PTQNodeSetup:
    kind = ptq_module_kind(module)
    if kind is None:
        raise ValueError(f"Unsupported PTQ module at {name}: {type(module).__name__}")
    act_bits = ptq_activation_bits(kind, cfg)
    weight_axis: Optional[int] = None
    weight_scale = torch.tensor(1.0)
    q_weight: Optional[torch.Tensor] = None
    if kind in _WEIGHT_ATTR:
        weight_fp = getattr(module, _WEIGHT_ATTR[kind]).detach().to(dtype=torch.float32, device="cpu")
        if getattr(module, "ptq_dequantized_inputs", False):
            # AffineCombine: each input is rescaled by its own multiplier before the sum, as in an integer Add with
            # per-input multipliers, so its coefficients are part of those multipliers and stay exact.
            q_weight = weight_fp
        else:
            weight_axis = 0 if kind in PTQ_LINEAR_KINDS and cfg.per_output_channel else None
            weight_scale = _symmetric_scale(weight_fp, cfg.weight_bits, axis=weight_axis)
            q_weight = ptq_quantize_proxy(weight_fp, weight_scale, cfg.weight_bits).cpu()
    return PTQNodeSetup(
        name=name,
        kind=kind,
        activation_bits=act_bits,
        weight_bits=int(cfg.weight_bits),
        output_channel_axis=stats.output_channel_axis,
        output_rank=stats.output_rank,
        per_output_channel_weights_config=bool(cfg.per_output_channel),
        per_output_channel_weights_effective=weight_axis is not None,
        input_scale=_input_scales(stats, act_bits),
        weight_scale=weight_scale.float().cpu(),
        q_weight=q_weight,
    )


def check_ptq_integer_sums(stats: Dict[str, PTQNodeStats]) -> None:
    """Every integer accumulation of the calibration data stays within the integers that fp32 holds exactly."""
    over = {n: s.integer_sum_bound for n, s in stats.items() if (s.integer_sum_bound or 0.0) > PTQ_FP32_EXACT_INT}
    if over:
        raise RuntimeError(f"PTQ integer sums exceed 2^24, the exact fp32 integer range: {over}")


def build_ptq_wrapper(
    module: nn.Module, setup: PTQNodeSetup, stats: PTQNodeStats, *, dequant_var_eps: float
) -> PTQBuiltWrapper:
    q_weight = None if setup.q_weight is None else setup.q_weight.clone()
    if setup.kind in PTQ_LINEAR_KINDS and q_weight is not None:
        # The accumulator keeps the integer weight codes, so it computes an exact integer sum; the output map applies
        # s_in * s_w and the fitted per-channel bias.
        out_scale = _linear_conv_out_scale(setup.input_scale, setup.weight_scale)
        out_bias = stats.bias_fit.bias()
        out_scale_mode: OutScaleMode = "analytical_s_in_times_s_w"
    else:
        out_scale, out_bias = stats.affine_fit.fit(float(dequant_var_eps))
        out_scale_mode = "ols_affine"
    skip_out_scale = False

    # Linear/Conv: the per-channel bias needs its channel axis even when out_scale is a scalar
    # (per-tensor weights). With per-channel weights both vectors give the same shape.
    out_bcast_shape = _out_bcast_shape(setup, out_bias)
    metadata = _wrapper_metadata(module, setup, stats, out_scale_mode, out_scale, out_bias, skip_out_scale)
    device = _module_device(module)
    if q_weight is not None:
        _swap_accumulator_weight(module, setup.kind, q_weight)
    wrapper = CalibratedAffinePTQWrapper(
        activation_bits=setup.activation_bits,
        input_scale=setup.input_scale,
        accumulator=module,
        out_scale=out_scale,
        out_bias=out_bias,
        skip_out_scale=skip_out_scale,
        out_bcast_shape=out_bcast_shape,
    ).to(device=device)
    reload_config = PTQReloadConfig(
        name=setup.name,
        kind=setup.kind,
        activation_bits=setup.activation_bits,
        out_bcast_shape=None if out_bcast_shape is None else list(out_bcast_shape),
        skip_out_scale=skip_out_scale,
        state_shapes={name: list(t.shape) for name, t in wrapper.state_dict().items()},
    )
    return PTQBuiltWrapper(module=wrapper, metadata=metadata, reload_config=reload_config)


def _values(t: torch.Tensor) -> List[float]:
    """Entries of a scalar or per-channel tensor as a flat list."""
    return t.detach().flatten().cpu().tolist()


def _wrapper_metadata(
    module: nn.Module,
    setup: PTQNodeSetup,
    stats: PTQNodeStats,
    out_scale_mode: OutScaleMode,
    out_scale: torch.Tensor,
    out_bias: torch.Tensor,
    skip_out_scale: bool,
) -> PTQNodeMeta:
    conv = module if isinstance(module, nn.Conv2d) else None
    return PTQNodeMeta(
        name=setup.name,
        kind=setup.kind,
        activation_group=ptq_activation_group(setup.kind),
        activation_bits=setup.activation_bits,
        weight_bits=setup.weight_bits,
        per_output_channel_weights_config=setup.per_output_channel_weights_config,
        per_output_channel_weights_effective=setup.per_output_channel_weights_effective,
        per_channel_output_affine=setup.per_channel_output_affine,
        output_channel_axis=setup.output_channel_axis,
        input_scale=_values(setup.input_scale),
        input_zero_point=[0.0] * setup.input_scale.numel(),
        weight_scale=_values(setup.weight_scale),
        weight_zero_point=[0.0] * setup.weight_scale.numel(),
        out_scale_mode=out_scale_mode,
        out_scale=_values(out_scale),
        out_scale_baked_into_weight=skip_out_scale,
        out_bias=_values(out_bias),
        integer_sum_bound=stats.integer_sum_bound,
        stride=None if conv is None else list(conv.stride),
        padding=None if conv is None else list(conv.padding),
        dilation=None if conv is None else list(conv.dilation),
        groups=1 if conv is None else conv.groups,
        einsum_equation=getattr(module, "einsum_equation", None),
    )


def ptq_wrapper_from_reload_config(config: PTQReloadConfig, module: nn.Module) -> CalibratedAffinePTQWrapper:
    """Skeleton wrapper with the saved shapes, ready for ``load_state_dict``."""
    shapes = config.state_shapes
    attr = _WEIGHT_ATTR.get(config.kind)
    if attr is not None:
        _swap_accumulator_weight(module, config.kind, torch.zeros(shapes[f"accumulator.{attr}"]))
    return CalibratedAffinePTQWrapper(
        activation_bits=config.activation_bits,
        input_scale=torch.ones(shapes["quantizer.input_scale"]),
        accumulator=module,
        out_scale=torch.ones(shapes["out_scale"]),
        out_bias=torch.zeros(shapes["out_bias"]),
        skip_out_scale=config.skip_out_scale,
        out_bcast_shape=None if config.out_bcast_shape is None else tuple(config.out_bcast_shape),
    )


def _accumulator_forward(
    module: nn.Module, kind: PTQKind, q_weight: Optional[torch.Tensor], q_inputs: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Accumulation of ``module`` on quantized inputs with its quantized weight; ``forward`` skips the hooks."""
    if kind not in _WEIGHT_ATTR or q_weight is None:
        return module.forward(*q_inputs)
    original = _swap_accumulator_weight(module, kind, q_weight)
    try:
        return module.forward(*q_inputs)
    finally:
        for attr, value in original.items():
            setattr(module, attr, value)


def _swap_accumulator_weight(module: nn.Module, kind: PTQKind, q_weight: torch.Tensor) -> Dict[str, Any]:
    """Install the integer weight (and a zero or dropped bias); returns the replaced attributes."""
    attr = _WEIGHT_ATTR[kind]
    old = getattr(module, attr)
    new = q_weight.reshape_as(old).to(device=old.device, dtype=torch.float32)
    original = {attr: old}
    if isinstance(old, nn.Parameter):
        new = nn.Parameter(new, requires_grad=False)
    setattr(module, attr, new)
    if kind in PTQ_LINEAR_KINDS:
        original["bias"] = module.bias
        module.bias = None
    elif kind == "affine_scale_bias":
        original["bias"] = module.bias
        module.bias = nn.Parameter(torch.zeros_like(module.bias, dtype=torch.float32), requires_grad=False)
    return original


def _out_bcast_shape(setup: PTQNodeSetup, out_scale: torch.Tensor) -> Optional[Tuple[int, ...]]:
    if setup.output_channel_axis is None or setup.output_rank <= 0 or out_scale.ndim == 0:
        return None
    axis = setup.output_channel_axis % setup.output_rank
    shape = [1] * setup.output_rank
    shape[axis] = -1
    return tuple(shape)


def _symmetric_scale(x: torch.Tensor, bits: int, axis: Optional[int] = None) -> torch.Tensor:
    qmax = ptq_signed_qrange(bits)[1]
    if axis is None:
        max_abs = x.abs().max()
    else:
        dims = tuple(i for i in range(x.ndim) if i != axis)
        max_abs = x.abs().amax(dim=dims, keepdim=True) if dims else x.abs()
    return (max_abs / float(max(qmax, 1))).clamp_min(1e-8).float()


def _default_output_channel_axis(kind: PTQKind, out: torch.Tensor) -> Optional[int]:
    if out.ndim == 0:
        return None
    if kind == "conv2d":
        return 1 if out.ndim > 1 else 0
    if kind == "affine_fixed_mix":
        return -2 if out.ndim >= 2 else None
    if kind == "output_identity":
        return None  # one scale for the whole model output
    return -1


def _broadcast_axis(vec: torch.Tensor, ref: torch.Tensor, axis: Optional[int]) -> torch.Tensor:
    if vec.ndim == 0 or axis is None:
        return vec
    shape = [1] * ref.ndim
    shape[axis % ref.ndim] = vec.numel()
    return vec.view(*shape)


def _linear_conv_out_scale(input_scale: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    s_in = input_scale[0].float()
    ws = weight_scale.float()
    return s_in * ws if ws.ndim == 0 else s_in * ws.reshape(-1)


def _input_scales(stats: PTQNodeStats, activation_bits: int) -> torch.Tensor:
    if not stats.input_max_abs:
        raise ValueError(f"No input range statistics collected for node {stats.name}")
    qmax = float(max(ptq_signed_qrange(activation_bits)[1], 1))
    return torch.stack([(max_abs.float() / qmax).clamp_min(1e-8) for max_abs in stats.input_max_abs])


def _module_device(module: nn.Module) -> torch.device:
    tensor = next(itertools.chain(module.parameters(), module.buffers()), None)
    return torch.device("cpu") if tensor is None else tensor.device

