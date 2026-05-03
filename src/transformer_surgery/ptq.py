#!/usr/bin/env python3
"""
Standalone PTQ for a surgery checkpoint.

Flow: pick nodes → register hooks → run calibration batches → remove hooks (ranges, then dequant
fits) → install wrappers → save. Full validation runs without hooks.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_surgery.models.adapters import (
    apply_load_cfg_overrides,
    get_model_adapter,
    load_surgery_student_checkpoint,
    surgery_dtype_from_extra,
)
from transformer_surgery.util import (
    describe_device,
    describe_dtype,
    ensure_mapping,
    get_device,
    metadata_path_for_checkpoint,
    namespace_from_mapping,
    namespace_to_mapping,
    save_model_checkpoint,
    traceable_artifact_path,
    traceable_log_path,
)
from transformer_surgery.ops import (
    AffineContract,
    AffineFixedMix,
    AffineHadamard,
    AffineMatMul,
    AffineScale,
    AffineScaleBias,
    get_surgery_dtype,
    set_surgery_dtype,
    write_model_structure_txt,
)


SUPPORTED_LINEAR_CONV_TYPES = (nn.Linear, nn.Conv2d)
SUPPORTED_AFFINE_TYPES = (
    AffineScale,
    AffineScaleBias,
    AffineFixedMix,
    AffineContract,
)
SUPPORTED_MATMUL_TYPES = (AffineMatMul, AffineHadamard)
MATMUL_KINDS = frozenset({"matmul", "matmul_hadamard"})


_MODULE_KIND_BY_TYPE: Tuple[Tuple[type, str], ...] = (
    (nn.Linear, "linear"),
    (nn.Conv2d, "conv2d"),
    (AffineScale, "affine_scale"),
    (AffineScaleBias, "affine_scale_bias"),
    (AffineFixedMix, "affine_fixed_mix"),
    (AffineContract, "affine_contract"),
    (AffineMatMul, "matmul"),
    (AffineHadamard, "matmul_hadamard"),
)


def _name_matches(name: str, patterns: Sequence[str]) -> bool:
    for pattern in patterns:
        needle = pattern.strip()
        if needle and needle in name:
            return True
    return False


def _module_kind(module: nn.Module) -> Optional[str]:
    for cls, label in _MODULE_KIND_BY_TYPE:
        if isinstance(module, cls):
            return label
    return None


def _module_type_selected(module: nn.Module, cfg: Any) -> bool:
    if cfg.wrap_linear_conv and isinstance(module, SUPPORTED_LINEAR_CONV_TYPES):
        return True
    if cfg.wrap_affine and isinstance(module, SUPPORTED_AFFINE_TYPES):
        return True
    if cfg.wrap_matmul and isinstance(module, SUPPORTED_MATMUL_TYPES):
        return True
    return False


def _activation_group_for_kind(kind: str) -> str:
    return "matmul" if kind in MATMUL_KINDS else "affine"


def _activation_bits_for_kind(kind: str, cfg: Any) -> int:
    default_bits = int(cfg.activation_bits)
    if kind in MATMUL_KINDS:
        if cfg.matmul_activation_bits is not None:
            return int(cfg.matmul_activation_bits)
        return default_bits
    if cfg.affine_activation_bits is not None:
        return int(cfg.affine_activation_bits)
    return default_bits


def _module_device(module: nn.Module) -> torch.device:
    for tensor in list(module.parameters()) + list(module.buffers()):
        return tensor.device
    return get_device()


def _set_module(root: nn.Module, name: str, new_module: nn.Module) -> None:
    parent_name, _, leaf = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, leaf, new_module)


def _leading_examples(t: torch.Tensor) -> int:
    if t.ndim == 0:
        return 1
    return int(t.shape[0])


def _select_examples_by_index(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if t.ndim == 0:
        return t.detach().reshape(1)
    idx_dev = idx.to(device=t.device, dtype=torch.long)
    return t.index_select(0, idx_dev).detach().contiguous()


def _sample_calibration_batch_indices(
    total_batches: int,
    requested_batches: int,
    *,
    generator: Optional[torch.Generator] = None,
) -> List[int]:
    total = int(total_batches)
    requested = int(requested_batches)
    if total < 1:
        return []
    keep = min(total, requested)
    return sorted(torch.randperm(total, generator=generator)[:keep].tolist())


@dataclass
class RunningAffineFitStats:
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
                x = acc.movedim(axis, -1).reshape(-1, acc.shape[axis]).to(dtype=torch.float32)
                y = out.movedim(axis, -1).reshape(-1, out.shape[axis]).to(dtype=torch.float32)
                self._add(x, y)
                return
        self._add(
            acc.reshape(-1, 1).to(dtype=torch.float32),
            out.reshape(-1, 1).to(dtype=torch.float32),
        )

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
            s = (self.sum_xy / denom) / second
            c = my - s * mx
        else:
            s = torch.where(var.abs() < var_eps, torch.zeros_like(var), cov / var.clamp_min(var_eps))
            c = my - s * mx
        return s.to(dtype=torch.float32), c.to(dtype=torch.float32)


@dataclass
class RunningBiasFitStats:
    count: int = 0
    residual_sum: Optional[torch.Tensor] = None

    def _add(self, residual_sum: torch.Tensor, count: int) -> None:
        rs = residual_sum.to(dtype=torch.float32, device="cpu")
        if self.residual_sum is None:
            self.residual_sum = rs
        else:
            self.residual_sum += rs
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
                pred = _broadcast_named_axis(out_scale, acc_fp, axis) * acc_fp
                residual = out_fp - pred
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
class NodeCalibrationData:
    name: str
    kind: str
    input_arity: int = 0
    input_max_abs: List[torch.Tensor] = field(default_factory=list)
    output_rank: int = 0
    output_channel_axis: Optional[int] = None
    batch_indices: List[int] = field(default_factory=list)
    examples_by_batch: Dict[int, int] = field(default_factory=dict)
    dequant_examples_by_batch: Dict[int, int] = field(default_factory=dict)
    stored_examples: int = 0
    dequant_examples: int = 0
    range_calls: int = 0
    dequant_calls: int = 0
    affine_fit: RunningAffineFitStats = field(default_factory=RunningAffineFitStats)
    bias_fit: RunningBiasFitStats = field(default_factory=RunningBiasFitStats)

    def observe_range(self, inputs: Sequence[torch.Tensor], output: torch.Tensor, batch_idx: int) -> None:
        if self.input_arity == 0:
            self.input_arity = len(inputs)
            self.input_max_abs = [torch.tensor(0.0, dtype=torch.float32) for _ in range(self.input_arity)]
        if len(inputs) != self.input_arity:
            raise ValueError(f"Input arity changed during PTQ calibration for node {self.name}")
        self.range_calls += 1
        if batch_idx not in self.examples_by_batch:
            self.batch_indices.append(batch_idx)
        for idx, tensor in enumerate(inputs):
            t = tensor.detach().to(dtype=torch.float32)
            self.input_max_abs[idx] = torch.maximum(self.input_max_abs[idx], t.abs().max().to(device="cpu"))
        self.output_rank = output.ndim
        self.output_channel_axis = _default_output_channel_axis(self.kind, output)

    def mark_examples(self, batch_idx: int, n: int) -> None:
        self.examples_by_batch[batch_idx] = int(self.examples_by_batch.get(batch_idx, 0)) + int(n)
        self.stored_examples += int(n)

    def mark_dequant_examples(self, batch_idx: int, n: int) -> None:
        self.dequant_examples_by_batch[batch_idx] = int(self.dequant_examples_by_batch.get(batch_idx, 0)) + int(n)
        self.dequant_examples += int(n)


@dataclass
class NodeQuantSetup:
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
    input_arity: int
    input_scale: torch.Tensor
    weight_scale: torch.Tensor
    q_weight: Optional[torch.Tensor]
    module_device: torch.device


@dataclass
class _CurrentBatch:
    idx: int = -1


def _hook_tensors_full_batch(
    inputs: Tuple[Any, ...],
    output: Any,
) -> Optional[Tuple[Tuple[torch.Tensor, ...], torch.Tensor, int]]:
    """All stacked examples consistent across tensor inputs/output (lead dim batch)."""
    if not torch.is_tensor(output):
        return None
    tensor_inputs = tuple(x for x in inputs if torch.is_tensor(x))
    if not tensor_inputs:
        return None
    available = min([_leading_examples(output)] + [_leading_examples(x) for x in tensor_inputs])
    if available <= 0:
        return None
    idx = torch.arange(available, device=output.device, dtype=torch.long)
    sampled_inputs = tuple(_select_examples_by_index(x, idx) for x in tensor_inputs)
    sampled_output = _select_examples_by_index(output, idx)
    return sampled_inputs, sampled_output, int(available)


def _remove_forward_hooks(handles: List[Any]) -> None:
    for h in handles:
        h.remove()
    handles.clear()


def _register_ptq_hooks(
    model: nn.Module, selected: Dict[str, str], hook_for: Callable[[str], Callable[..., None]]
) -> List[Any]:
    return [model.get_submodule(n).register_forward_hook(hook_for(n)) for n in selected]


def _range_hook(
    name: str,
    stats: Dict[str, NodeCalibrationData],
    batch_ctx: _CurrentBatch,
    batch_index_set: set[int],
) -> Callable[..., None]:
    def fn(_module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        bi = batch_ctx.idx
        if bi not in batch_index_set:
            return
        slot = stats[name]
        if int(slot.examples_by_batch.get(bi, 0)) > 0:
            return
        sample = _hook_tensors_full_batch(inputs, output)
        if sample is None:
            return
        sampled_inputs, sampled_output, keep = sample
        slot.observe_range(sampled_inputs, sampled_output, bi)
        slot.mark_examples(bi, keep)

    return fn


def _dequant_hook(
    name: str,
    stats: Dict[str, NodeCalibrationData],
    quant_setups: Dict[str, NodeQuantSetup],
    batch_ctx: _CurrentBatch,
    batch_index_set: set[int],
) -> Callable[..., None]:
    def fn(_module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        bi = batch_ctx.idx
        if bi not in batch_index_set:
            return
        slot = stats[name]
        if int(slot.dequant_examples_by_batch.get(bi, 0)) > 0:
            return
        sample = _hook_tensors_full_batch(inputs, output)
        if sample is None:
            return
        sampled_inputs, sampled_output, keep = sample
        setup = quant_setups[name]
        q_inputs = [
            _quantize_proxy(inp, setup.input_scale[i], setup.activation_bits)
            for i, inp in enumerate(sampled_inputs)
        ]
        acc = _simulate_accumulator(
            setup.kind,
            q_inputs,
            q_weight=setup.q_weight.to(device=sampled_output.device) if setup.q_weight is not None else None,
            stride=setup.stride,
            padding=setup.padding,
            dilation=setup.dilation,
            groups=setup.groups,
            einsum_equation=setup.einsum_equation,
        ).to(dtype=torch.float32)
        if setup.kind in ("linear", "conv2d") and setup.q_weight is not None:
            out_scale = _linear_conv_out_scale_from_quant_scales(setup.input_scale, setup.weight_scale)
            slot.bias_fit.update(
                acc,
                sampled_output,
                out_scale,
                channel_axis=setup.output_channel_axis,
                per_channel=setup.per_channel_output_affine,
            )
        else:
            slot.affine_fit.update(
                acc,
                sampled_output,
                channel_axis=setup.output_channel_axis,
                per_channel=setup.per_channel_output_affine,
            )
        slot.dequant_calls += 1
        slot.mark_dequant_examples(bi, keep)

    return fn


def _run_ptq_hook_phase(
    model: nn.Module,
    loader,
    batch_indices: Sequence[int],
    install_hooks: Callable[[_CurrentBatch, set[int]], List[Any]],
) -> None:
    batch_ctx = _CurrentBatch(-1)
    batch_set = {int(i) for i in batch_indices}
    hooks = install_hooks(batch_ctx, batch_set)
    try:
        _forward_ptq_calibration_batches(model, loader, batch_indices, batch_ctx)
    finally:
        _remove_forward_hooks(hooks)


def _input_dtype(model: nn.Module) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return get_surgery_dtype()


def _forward_ptq_calibration_batches(
    model: nn.Module,
    loader,
    batch_indices: Sequence[int],
    batch_ctx: _CurrentBatch,
) -> None:
    batch_index_set = {int(i) for i in batch_indices}
    if not batch_index_set:
        return
    model.eval()
    device = get_device()
    use_cuda = device.type == "cuda"
    input_dtype = _input_dtype(model)
    last = max(batch_index_set)
    for bi, (x, _y) in enumerate(loader):
        if bi > last:
            break
        if bi not in batch_index_set:
            continue
        batch_ctx.idx = bi
        x = x.to(device, dtype=input_dtype, non_blocking=use_cuda)
        model(x)


def gather_ptq_range_moments(
    model: nn.Module,
    loader,
    selected: Dict[str, str],
    batch_indices: Sequence[int],
    *,
    stats: Optional[Dict[str, NodeCalibrationData]] = None,
) -> Dict[str, NodeCalibrationData]:
    """Register range hooks, run calibration batches, remove hooks. Uses full each calibration minibatch."""
    out = stats or {name: NodeCalibrationData(name=name, kind=k) for name, k in selected.items()}

    def install(batch_ctx: _CurrentBatch, batch_set: set[int]) -> List[Any]:
        return _register_ptq_hooks(
            model,
            selected,
            lambda name: _range_hook(name, out, batch_ctx, batch_set),
        )

    _run_ptq_hook_phase(model, loader, batch_indices, install)
    return out


def gather_ptq_dequant_moments(
    model: nn.Module,
    loader,
    selected: Dict[str, str],
    stats: Dict[str, NodeCalibrationData],
    quant_setups: Dict[str, NodeQuantSetup],
    batch_indices: Sequence[int],
) -> None:
    """Register dequant-fit hooks, run calibration batches, remove hooks; mutates ``stats``."""

    def install(batch_ctx: _CurrentBatch, batch_set: set[int]) -> List[Any]:
        return _register_ptq_hooks(
            model,
            selected,
            lambda name: _dequant_hook(name, stats, quant_setups, batch_ctx, batch_set),
        )

    _run_ptq_hook_phase(model, loader, batch_indices, install)


def _signed_qrange(bits: int) -> Tuple[int, int]:
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    return qmin, qmax


def _symmetric_scale(x: torch.Tensor, bits: int, axis: Optional[int] = None) -> torch.Tensor:
    _qmin, qmax = _signed_qrange(bits)
    if axis is None:
        max_abs = x.abs().max()
    else:
        dims = tuple(i for i in range(x.ndim) if i != axis)
        if len(dims) == 0:
            max_abs = x.abs()
        else:
            max_abs = x.abs().amax(dim=dims, keepdim=True)
    return (max_abs / float(max(qmax, 1))).clamp_min(1e-8).to(dtype=torch.float32)


def _quantize_proxy(x: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    qmin, qmax = _signed_qrange(bits)
    scale = scale.to(device=x.device, dtype=torch.float32)
    x_fp = x.to(dtype=torch.float32)
    return torch.clamp(torch.round(x_fp / scale), qmin, qmax)


def _default_output_channel_axis(kind: str, out: torch.Tensor) -> Optional[int]:
    """Layout hint for per-channel affine dequant: conv uses NCHW ``C``, fixed-mix penultimate."""
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


def _extract_weight_tensor(module: nn.Module, kind: str) -> Optional[torch.Tensor]:
    attr = _WEIGHT_ATTR.get(kind)
    if attr is None:
        return None
    return getattr(module, attr).detach().to(dtype=torch.float32, device="cpu")


def _broadcast_last_dim(vec: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if vec.ndim == 0:
        return vec
    return vec.view(*([1] * (ref.ndim - 1)), vec.numel())


def _broadcast_named_axis(vec: torch.Tensor, ref: torch.Tensor, axis: Optional[int]) -> torch.Tensor:
    if vec.ndim == 0 or axis is None:
        return vec
    axis = axis if axis >= 0 else ref.ndim + axis
    shape = [1] * ref.ndim
    shape[axis] = vec.numel()
    return vec.view(*shape)


def _simulate_accumulator(
    kind: str,
    q_inputs: Sequence[torch.Tensor],
    *,
    q_weight: Optional[torch.Tensor] = None,
    stride: Optional[Tuple[int, int]] = None,
    padding: Optional[Tuple[int, int]] = None,
    dilation: Optional[Tuple[int, int]] = None,
    groups: int = 1,
    einsum_equation: Optional[str] = None,
) -> torch.Tensor:
    if kind == "linear":
        return F.linear(q_inputs[0], q_weight, bias=None)
    if kind == "conv2d":
        return F.conv2d(
            q_inputs[0],
            q_weight,
            bias=None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    if kind == "affine_scale":
        return q_inputs[0] * q_weight
    if kind == "affine_scale_bias":
        return q_inputs[0] * _broadcast_last_dim(q_weight.reshape(-1), q_inputs[0])
    if kind in ("affine_fixed_mix", "affine_contract"):
        return torch.einsum(einsum_equation, q_weight, q_inputs[0])
    if kind == "matmul":
        return torch.matmul(q_inputs[0], q_inputs[1])
    if kind == "matmul_hadamard":
        return q_inputs[0] * q_inputs[1]
    raise ValueError(f"Unsupported PTQ kind: {kind}")


def _linear_conv_out_scale_from_quant_scales(
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """y ≈ s_in s_w acc for symmetric input/weight quant (first input scale × weight scales)."""
    s_in = input_scale[0].to(dtype=torch.float32)
    ws = weight_scale.to(dtype=torch.float32)
    if ws.ndim == 0:
        return s_in * ws
    return (s_in * ws.reshape(-1)).to(dtype=torch.float32)


def _calibrated_input_scales(data: NodeCalibrationData, activation_bits: int) -> torch.Tensor:
    """One symmetric scale per input tensor: max |x| over all axes (channels share the quant step for dot-product / conv sums)."""
    if data.input_arity <= 0 or not data.input_max_abs:
        raise ValueError(f"No input range statistics collected for node {data.name}")
    qmax = float(max(_signed_qrange(activation_bits)[1], 1))
    in_scales = [(max_abs.to(dtype=torch.float32) / qmax).clamp_min(1e-8) for max_abs in data.input_max_abs]
    return torch.stack(in_scales).to(dtype=torch.float32)


def _build_quant_setup(name: str, module: nn.Module, data: NodeCalibrationData, cfg: Any) -> NodeQuantSetup:
    kind = _module_kind(module)
    if kind is None:
        raise ValueError(f"Unsupported PTQ module at {name}: {type(module).__name__}")
    activation_bits = _activation_bits_for_kind(kind, cfg)
    weight_bits = int(cfg.weight_bits)
    input_scale = _calibrated_input_scales(data, activation_bits)
    weight_axis_used: Optional[int] = None
    weight_fp = _extract_weight_tensor(module, kind)
    q_weight: Optional[torch.Tensor] = None
    if weight_fp is not None:
        weight_axis_used = 0 if kind in ("linear", "conv2d") and cfg.per_output_channel else None
        weight_scale = _symmetric_scale(weight_fp, weight_bits, axis=weight_axis_used)
        q_weight = _quantize_proxy(weight_fp, weight_scale, weight_bits).to(dtype=torch.float32, device="cpu")
    else:
        weight_scale = torch.tensor(1.0, dtype=torch.float32)
    if isinstance(module, (AffineFixedMix, AffineContract)):
        einsum_equation = module.einsum_equation
    else:
        einsum_equation = None
    return NodeQuantSetup(
        name=name,
        kind=kind,
        activation_bits=activation_bits,
        weight_bits=weight_bits,
        einsum_equation=einsum_equation,
        stride=tuple(module.stride) if isinstance(module, nn.Conv2d) else None,
        padding=tuple(module.padding) if isinstance(module, nn.Conv2d) else None,
        dilation=tuple(module.dilation) if isinstance(module, nn.Conv2d) else None,
        groups=int(module.groups) if isinstance(module, nn.Conv2d) else 1,
        output_channel_axis=data.output_channel_axis,
        output_rank=data.output_rank,
        per_channel_output_affine=data.output_channel_axis is not None,
        per_output_channel_weights_config=bool(cfg.per_output_channel),
        per_output_channel_weights_effective=weight_axis_used is not None,
        input_arity=data.input_arity,
        input_scale=input_scale,
        weight_scale=weight_scale.to(dtype=torch.float32, device="cpu"),
        q_weight=q_weight,
        module_device=_module_device(module),
    )


class CalibratedAffinePTQWrapper(nn.Module):
    def __init__(self, setup: NodeQuantSetup, data: NodeCalibrationData, cfg: Any) -> None:
        super().__init__()
        self.node_name = setup.name
        self.kind = setup.kind
        self.activation_bits = setup.activation_bits
        self.weight_bits = setup.weight_bits
        self.einsum_equation = setup.einsum_equation
        self.stride = setup.stride
        self.padding = setup.padding
        self.dilation = setup.dilation
        self.groups = setup.groups
        self.output_channel_axis = setup.output_channel_axis
        self.per_channel_output_affine = setup.per_channel_output_affine
        self.per_output_channel_weights_config = setup.per_output_channel_weights_config
        self.input_arity = setup.input_arity

        self.register_buffer("input_scale", setup.input_scale.to(dtype=torch.float32))
        self.register_buffer("input_inv_scale", setup.input_scale.reciprocal().to(dtype=torch.float32))
        self.register_buffer("input_zero_point", torch.zeros(self.input_arity, dtype=torch.float32))
        self._act_qmin, self._act_qmax = _signed_qrange(self.activation_bits)

        q_weight = setup.q_weight.clone() if setup.q_weight is not None else None
        self.register_buffer("weight_scale", setup.weight_scale.to(dtype=torch.float32))
        self.register_buffer("weight_zero_point", torch.zeros_like(setup.weight_scale, dtype=torch.float32))
        self.per_output_channel_weights_effective = setup.per_output_channel_weights_effective

        if self.kind in ("linear", "conv2d") and q_weight is not None:
            out_scale = _linear_conv_out_scale_from_quant_scales(self.input_scale, self.weight_scale)
            out_bias = data.bias_fit.bias()
            self.out_scale_mode = "analytical_s_in_times_s_w"
        else:
            out_scale, out_bias = data.affine_fit.fit(float(cfg.dequant_var_eps))
            self.out_scale_mode = "ols_affine"

        if (
            self.per_channel_output_affine
            and self.output_channel_axis is not None
            and setup.output_rank > 0
            and out_scale.ndim > 0
        ):
            axis = self.output_channel_axis if self.output_channel_axis >= 0 else setup.output_rank + self.output_channel_axis
            bshape = [1] * setup.output_rank
            bshape[axis] = -1
            self._out_bcast_shape: Optional[Tuple[int, ...]] = tuple(bshape)
        else:
            self._out_bcast_shape = None

        # For Linear/Conv2d, bake out_scale into q_weight (axis 0 of weight == output channel for both).
        # Saves one multiply over the output tensor each forward.
        self._skip_out_scale = False
        if self.kind in ("linear", "conv2d") and q_weight is not None:
            if out_scale.ndim == 0:
                q_weight = q_weight * out_scale.to(dtype=torch.float32)
            else:
                wshape = [1] * q_weight.ndim
                wshape[0] = -1
                q_weight = q_weight * out_scale.view(*wshape).to(dtype=torch.float32)
            self._skip_out_scale = True
            self._out_scale_value = out_scale.detach().cpu().tolist()
            out_scale_buf = torch.ones((), dtype=torch.float32)
        else:
            self._out_scale_value = out_scale.detach().cpu().tolist()
            out_scale_buf = out_scale.to(dtype=torch.float32)

        if q_weight is not None:
            self.register_buffer("q_weight", q_weight)
        else:
            self.q_weight = None
        self.register_buffer("out_scale", out_scale_buf)
        self.register_buffer("out_bias", out_bias.to(dtype=torch.float32))

        self.to(device=setup.module_device)

    @property
    def activation_group(self) -> str:
        return _activation_group_for_kind(self.kind)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        qmin, qmax = self._act_qmin, self._act_qmax
        inv_scale = self.input_inv_scale
        q_inputs = [
            torch.clamp(torch.round(inp.float() * inv_scale[i]), qmin, qmax)
            for i, inp in enumerate(inputs)
        ]
        acc = _simulate_accumulator(
            self.kind,
            q_inputs,
            q_weight=self.q_weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            einsum_equation=self.einsum_equation,
        )
        bshape = self._out_bcast_shape
        if bshape is not None:
            out_bias_v = self.out_bias.view(bshape)
            if self._skip_out_scale:
                out = acc + out_bias_v
            else:
                out = self.out_scale.view(bshape) * acc + out_bias_v
        else:
            if self._skip_out_scale:
                out = acc + self.out_bias
            else:
                out = self.out_scale * acc + self.out_bias
        return out.to(dtype=inputs[0].dtype)

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.node_name,
            "kind": self.kind,
            "activation_group": self.activation_group,
            "activation_bits": self.activation_bits,
            "weight_bits": self.weight_bits,
            "per_output_channel_weights_config": self.per_output_channel_weights_config,
            "per_output_channel_weights_effective": self.per_output_channel_weights_effective,
            "per_channel_output_affine": self.per_channel_output_affine,
            "output_channel_axis": self.output_channel_axis,
            "input_scale": self.input_scale.detach().cpu().tolist(),
            "input_zero_point": self.input_zero_point.detach().cpu().tolist(),
            "weight_scale": self.weight_scale.detach().cpu().tolist(),
            "weight_zero_point": self.weight_zero_point.detach().cpu().tolist(),
            "out_scale_mode": self.out_scale_mode,
            "out_scale": self._out_scale_value,
            "out_scale_baked_into_weight": self._skip_out_scale,
            "out_bias": self.out_bias.detach().cpu().tolist(),
            "stride": list(self.stride) if self.stride is not None else None,
            "padding": list(self.padding) if self.padding is not None else None,
            "dilation": list(self.dilation) if self.dilation is not None else None,
            "groups": self.groups,
            "einsum_equation": self.einsum_equation,
        }

    def reload_config(self) -> Dict[str, Any]:
        """Plain-attr config + buffer shapes needed to rebuild a skeleton wrapper before load_state_dict."""
        buffer_shapes = {name: list(buf.shape) for name, buf in self.named_buffers(recurse=False)}
        return {
            "name": self.node_name,
            "kind": self.kind,
            "activation_bits": self.activation_bits,
            "weight_bits": self.weight_bits,
            "einsum_equation": self.einsum_equation,
            "stride": list(self.stride) if self.stride is not None else None,
            "padding": list(self.padding) if self.padding is not None else None,
            "dilation": list(self.dilation) if self.dilation is not None else None,
            "groups": self.groups,
            "output_channel_axis": self.output_channel_axis,
            "per_channel_output_affine": self.per_channel_output_affine,
            "per_output_channel_weights_config": self.per_output_channel_weights_config,
            "per_output_channel_weights_effective": self.per_output_channel_weights_effective,
            "input_arity": self.input_arity,
            "out_bcast_shape": list(self._out_bcast_shape) if self._out_bcast_shape is not None else None,
            "skip_out_scale": self._skip_out_scale,
            "out_scale_value": self._out_scale_value,
            "out_scale_mode": self.out_scale_mode,
            "buffer_shapes": buffer_shapes,
        }

    @classmethod
    def from_reload_config(cls, config: Dict[str, Any]) -> "CalibratedAffinePTQWrapper":
        """Build an empty skeleton with correct attrs and buffer shapes; load_state_dict fills values."""
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.node_name = config["name"]
        instance.kind = config["kind"]
        instance.activation_bits = int(config["activation_bits"])
        instance.weight_bits = int(config["weight_bits"])
        instance.einsum_equation = config.get("einsum_equation")
        instance.stride = tuple(config["stride"]) if config.get("stride") is not None else None
        instance.padding = tuple(config["padding"]) if config.get("padding") is not None else None
        instance.dilation = tuple(config["dilation"]) if config.get("dilation") is not None else None
        instance.groups = int(config.get("groups", 1))
        instance.output_channel_axis = config["output_channel_axis"]
        instance.per_channel_output_affine = bool(config["per_channel_output_affine"])
        instance.per_output_channel_weights_config = bool(config["per_output_channel_weights_config"])
        instance.per_output_channel_weights_effective = bool(config["per_output_channel_weights_effective"])
        instance.input_arity = int(config["input_arity"])
        instance._act_qmin, instance._act_qmax = _signed_qrange(instance.activation_bits)
        bshape = config.get("out_bcast_shape")
        instance._out_bcast_shape = tuple(bshape) if bshape is not None else None
        instance._skip_out_scale = bool(config["skip_out_scale"])
        instance._out_scale_value = config.get("out_scale_value")
        instance.out_scale_mode = config["out_scale_mode"]
        for buf_name, shape in config["buffer_shapes"].items():
            instance.register_buffer(buf_name, torch.zeros(shape, dtype=torch.float32))
        if "q_weight" not in config["buffer_shapes"]:
            instance.q_weight = None
        return instance


def _build_node_selection(model: nn.Module, cfg: Any) -> Dict[str, str]:
    selected: Dict[str, str] = {}
    for name, module in model.named_modules():
        if not name:
            continue
        kind = _module_kind(module)
        if kind is None:
            continue
        if _name_matches(name, cfg.exclude_names):
            continue
        if _module_type_selected(module, cfg) or _name_matches(name, cfg.include_names):
            selected[name] = kind
    return selected


def _node_debug_metadata(data: NodeCalibrationData) -> Dict[str, Any]:
    return {
        "stored_examples": data.stored_examples,
        "dequant_examples": data.dequant_examples,
        "range_calls": data.range_calls,
        "dequant_calls": data.dequant_calls,
        "batch_indices": list(data.batch_indices),
        "examples_by_batch": {str(k): int(v) for k, v in sorted(data.examples_by_batch.items())},
        "dequant_examples_by_batch": {
            str(k): int(v) for k, v in sorted(data.dequant_examples_by_batch.items())
        },
    }


@torch.no_grad()
def validate_model(
    model: nn.Module,
    loader,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.eval()
    device = get_device()
    use_cuda = device.type == "cuda"
    loss_sum_t = torch.zeros((), device=device, dtype=torch.float64)
    correct_t = torch.zeros((), device=device, dtype=torch.long)
    n = 0
    input_dtype = _input_dtype(model)

    for _, (x, y) in enumerate(loader):
        x = x.to(device, dtype=input_dtype, non_blocking=use_cuda)
        y = y.to(device, non_blocking=use_cuda)
        logits = model(x)
        loss_sum_t += criterion(logits.float(), y).double() * y.size(0)
        correct_t += (logits.argmax(dim=-1) == y).sum()
        n += y.size(0)
    denom = max(n, 1)
    return correct_t.item() / denom, loss_sum_t.item() / denom


def _build_ptq_model(
    fp_model: nn.Module,
    selected: Dict[str, str],
    calibration_stats: Dict[str, NodeCalibrationData],
    quant_setups: Dict[str, NodeQuantSetup],
    cfg: Any,
) -> Tuple[nn.Module, List[Dict[str, Any]], List[Dict[str, Any]]]:
    out = copy.deepcopy(fp_model)
    node_meta: List[Dict[str, Any]] = []
    reload_configs: List[Dict[str, Any]] = []
    for name in selected:
        cal = calibration_stats[name]
        wrapper = CalibratedAffinePTQWrapper(quant_setups[name], cal, cfg)
        _set_module(out, name, wrapper)
        meta = wrapper.metadata()
        meta["calibration_debug"] = _node_debug_metadata(cal)
        node_meta.append(meta)
        reload_configs.append(wrapper.reload_config())
    return out, node_meta, reload_configs


def load_ptq_checkpoint(path: str, cfg: Any) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a PTQ checkpoint: rebuild the surgery template, install skeleton PTQ modules, then load_state_dict."""
    device = get_device()
    payload = torch.load(path, map_location=device, weights_only=False)
    extra_ns = namespace_from_mapping(dict(payload["extra"]))
    adapter = get_model_adapter(extra_ns.model_key)
    apply_load_cfg_overrides(extra_ns, cfg)
    state_dict = payload["model_state_dict"]
    set_surgery_dtype(surgery_dtype_from_extra(extra_ns))
    extra_ns.surgery_dtype = describe_dtype(get_surgery_dtype())
    model = adapter.build_surgery_model_from_extra(ensure_mapping(extra_ns), cfg).to(
        device=device, dtype=get_surgery_dtype()
    )
    for wc in extra_ns.ptq_wrappers:
        wc_d = namespace_to_mapping(wc)
        skeleton = CalibratedAffinePTQWrapper.from_reload_config(wc_d).to(device=device)
        _set_module(model, wc_d["name"], skeleton)
    model.load_state_dict(state_dict, strict=True)
    return model, ensure_mapping(extra_ns)


def _ptq_summary(
    cfg: Any,
    selected: Dict[str, str],
    fp_acc: float,
    fp_loss: float,
    ptq_acc: float,
    ptq_loss: float,
    node_meta: List[Dict[str, Any]],
    *,
    calibration_batch_indices: Sequence[int],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "fp_val_acc": float(fp_acc),
        "fp_val_loss": float(fp_loss),
        "ptq_val_acc": float(ptq_acc),
        "ptq_val_loss": float(ptq_loss),
        "acc_delta": float(ptq_acc - fp_acc),
        "loss_delta": float(ptq_loss - fp_loss),
    }
    return {
        "source_checkpoint": os.path.abspath(cfg.fp_checkpoint),
        "output_checkpoint": os.path.abspath(cfg.output),
        "selection": {
            "wrap_linear_conv": cfg.wrap_linear_conv,
            "wrap_affine": cfg.wrap_affine,
            "wrap_matmul": cfg.wrap_matmul,
            "include_names": list(cfg.include_names),
            "exclude_names": list(cfg.exclude_names),
            "selected_nodes": list(selected.keys()),
        },
        "quantization": {
            "weight_bits": int(cfg.weight_bits),
            "activation_bits": int(cfg.activation_bits),
            "affine_activation_bits": int(_activation_bits_for_kind("linear", cfg)),
            "matmul_activation_bits": int(_activation_bits_for_kind("matmul", cfg)),
            "per_output_channel": bool(cfg.per_output_channel),
        },
        "calibration": {
            "batches_requested": int(cfg.calibration_batches),
            "batches_sampled": len(calibration_batch_indices),
            "batch_indices": [int(i) for i in calibration_batch_indices],
        },
        "metrics": metrics,
        "ptq_nodes": node_meta,
    }


def run_ptq(
    cfg: Any,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    device = get_device() if device is None else device
    fp_path = os.path.abspath(cfg.fp_checkpoint)
    out_abs = traceable_artifact_path(cfg.output, cfg, "ts-ptq", "", ".pt")
    meta_abs = metadata_path_for_checkpoint(out_abs)
    model_log_abs = traceable_log_path(cfg.log_dir, cfg, "ts-ptq", "model_after_ptq")
    cfg.output = out_abs

    print(f"Using device: {describe_device(device)}", flush=True)
    if cfg.config_json_path:
        print(f"config_json={cfg.config_json_path}", flush=True)

    criterion = nn.CrossEntropyLoss()

    fp_model, fp_extra = load_surgery_student_checkpoint(fp_path, cfg, surgery_dtype=dtype)
    print(f"Surgery dtype (from checkpoint): {describe_dtype(get_surgery_dtype())}", flush=True)
    adapter = get_model_adapter(fp_extra["model_key"])
    _train_loader, val_loader = adapter.build_loaders(cfg)
    print(f"Using model adapter: {adapter.key}", flush=True)
    calibration_batch_indices = _sample_calibration_batch_indices(
        len(val_loader),
        int(cfg.calibration_batches),
    )
    if not calibration_batch_indices:
        raise SystemExit("No validation batches available for PTQ calibration.")
    print(
        f"PTQ calibration batches: requested={int(cfg.calibration_batches)} "
        f"sampled={len(calibration_batch_indices)} indices={calibration_batch_indices}",
        flush=True,
    )
    selected = _build_node_selection(fp_model, cfg)
    if not selected:
        raise SystemExit("No PTQ-wrappable nodes selected by the current config.")

    print(f"Selected {len(selected)} PTQ node(s).", flush=True)
    for name, kind in selected.items():
        print(f"  {name}: {kind}", flush=True)

    fp_acc, fp_loss = validate_model(fp_model, val_loader, criterion)
    print(f"Float model val acc={fp_acc:.4f} loss={fp_loss:.4f}", flush=True)

    stats = gather_ptq_range_moments(
        fp_model,
        val_loader,
        selected,
        calibration_batch_indices,
    )

    missing = [name for name in selected if stats[name].stored_examples <= 0]
    if missing:
        raise SystemExit(f"Calibration range statistics missing for selected node(s): {missing}")

    quant_setups = {
        name: _build_quant_setup(name, fp_model.get_submodule(name), stats[name], cfg)
        for name in selected
    }
    gather_ptq_dequant_moments(
        fp_model,
        val_loader,
        selected,
        stats,
        quant_setups,
        calibration_batch_indices,
    )

    missing_dequant = [
        name
        for name in selected
        if stats[name].bias_fit.count <= 0 and stats[name].affine_fit.count <= 0
    ]
    if missing_dequant:
        raise SystemExit(f"Calibration dequant statistics missing for selected node(s): {missing_dequant}")

    ptq_model, node_meta, reload_configs = _build_ptq_model(
        fp_model, selected, stats, quant_setups, cfg
    )

    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    out_extra = {
        **fp_extra,
        "ptq_meta_path": os.path.basename(meta_abs),
        "ptq_wrappers": reload_configs,
    }
    write_model_structure_txt(model_log_abs, ptq_model, "PTQ Surgery Model")
    save_model_checkpoint(out_abs, ptq_model, extra=out_extra)
    print(f"wrote {model_log_abs}", flush=True)
    print(f"wrote {out_abs}", flush=True)

    del ptq_model
    reloaded_model, _ = load_ptq_checkpoint(out_abs, cfg)
    print("reloaded PTQ checkpoint from disk for validation", flush=True)
    ptq_acc, ptq_loss = validate_model(reloaded_model, val_loader, criterion)

    print(f"PTQ model val acc={ptq_acc:.4f} loss={ptq_loss:.4f}", flush=True)
    print(f"Delta acc={ptq_acc - fp_acc:+.4f} loss={ptq_loss - fp_loss:+.4f}", flush=True)

    meta = _ptq_summary(
        cfg,
        selected,
        fp_acc,
        fp_loss,
        ptq_acc,
        ptq_loss,
        node_meta,
        calibration_batch_indices=calibration_batch_indices,
    )

    with open(meta_abs, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {meta_abs}", flush=True)
