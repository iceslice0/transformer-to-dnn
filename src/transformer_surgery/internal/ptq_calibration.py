"""PTQ calibration, wrapper construction, and reload metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from transformer_surgery.ops import (
    AffineContract,
    AffineFixedMix,
    AffineHadamard,
    AffineMatMul,
    AffineScale,
    AffineScaleBias,
    CalibratedAffinePTQWrapper,
    ptq_quantize_proxy,
    ptq_signed_qrange,
)

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
    if kind == "linear":
        if not isinstance(module, nn.Linear):
            raise TypeError(f"expected nn.Linear, got {type(module).__name__}")
        original = (module.weight, module.bias)
        module.weight = nn.Parameter(q_weight.to(device=module.weight.device, dtype=torch.float32), requires_grad=False)
        module.bias = None
        return original
    if kind == "conv2d":
        if not isinstance(module, nn.Conv2d):
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
