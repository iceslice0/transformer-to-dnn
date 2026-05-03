#!/usr/bin/env python3
"""
Standalone PTQ for a surgery checkpoint.

The script:
1. loads a floated surgery checkpoint,
2. collects one-pass calibration samples from selected affine / matmul leaf nodes,
3. validates the float model,
4. deep-copies and wraps the selected nodes with PTQ wrappers,
5. initializes wrapper quant / dequant parameters from calibration data,
6. validates the wrapped model,
7. saves the wrapped checkpoint and metadata.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_surgery.models.adapters import get_model_adapter, load_surgery_student_checkpoint
from transformer_surgery.util import (
    describe_device,
    describe_dtype,
    get_device,
    metadata_path_for_checkpoint,
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
MATMUL_KINDS = {"matmul", "matmul_hadamard"}


def _name_matches(name: str, patterns: Sequence[str]) -> bool:
    for pattern in patterns:
        needle = pattern.strip()
        if needle and needle in name:
            return True
    return False


def _module_kind(module: nn.Module) -> Optional[str]:
    if isinstance(module, nn.Linear):
        return "linear"
    if isinstance(module, nn.Conv2d):
        return "conv2d"
    if isinstance(module, AffineScale):
        return "affine_scale"
    if isinstance(module, AffineScaleBias):
        return "affine_scale_bias"
    if isinstance(module, AffineFixedMix):
        return "affine_fixed_mix"
    if isinstance(module, AffineContract):
        return "affine_contract"
    if isinstance(module, AffineMatMul):
        return "matmul"
    if isinstance(module, AffineHadamard):
        return "matmul_hadamard"
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
    if kind in MATMUL_KINDS:
        return "matmul"
    return "affine"


def _activation_bits_for_kind(kind: str, cfg: Any) -> int:
    legacy_bits = int(cfg.activation_bits)
    if kind in MATMUL_KINDS:
        if cfg.matmul_activation_bits is not None:
            return int(cfg.matmul_activation_bits)
        return legacy_bits
    if cfg.affine_activation_bits is not None:
        return int(cfg.affine_activation_bits)
    return legacy_bits


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


def _take_examples(t: torch.Tensor, n: int) -> torch.Tensor:
    if t.ndim == 0:
        return t.detach().reshape(1).to(dtype=torch.float32, device="cpu")
    return t[:n].detach().to(dtype=torch.float32, device="cpu").contiguous()


@dataclass
class NodeCalibrationData:
    name: str
    kind: str
    input_samples: List[Tuple[torch.Tensor, ...]] = field(default_factory=list)
    output_samples: List[torch.Tensor] = field(default_factory=list)
    stored_examples: int = 0
    num_calls: int = 0


class CalibrationController:
    def __init__(self, selected: Dict[str, str], max_batches: int, max_examples_per_node: int) -> None:
        self.selected = selected
        self.max_batches = max_batches
        self.max_examples_per_node = max_examples_per_node
        self.current_batch = -1
        self.cache: Dict[str, NodeCalibrationData] = {
            name: NodeCalibrationData(name=name, kind=kind) for name, kind in selected.items()
        }
        self._hooks: List[Any] = []

    def _hook(self, name: str):
        def fn(_module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            if self.current_batch < 0 or self.current_batch >= self.max_batches:
                return
            if not torch.is_tensor(output):
                return
            tensor_inputs = tuple(x for x in inputs if torch.is_tensor(x))
            if not tensor_inputs:
                return
            slot = self.cache[name]
            slot.num_calls += 1
            left = self.max_examples_per_node - slot.stored_examples
            if left <= 0:
                return
            keep = min(left, _leading_examples(output))
            slot.input_samples.append(tuple(_take_examples(x, keep) for x in tensor_inputs))
            slot.output_samples.append(_take_examples(output, keep))
            slot.stored_examples += keep

        return fn

    def register(self, model: nn.Module) -> None:
        for name in self.selected:
            hook = model.get_submodule(name).register_forward_hook(self._hook(name))
            self._hooks.append(hook)

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


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


def _preferred_output_channel_axis(kind: str, out: torch.Tensor) -> Optional[int]:
    if out.ndim == 0:
        return None
    if kind == "conv2d":
        return 1 if out.ndim > 1 else 0
    if kind == "affine_fixed_mix":
        return -2 if out.ndim >= 2 else None
    return -1


def _diagnostic_linear_conv_max_s_global_over_s_pc(
    model: nn.Module,
    selected: Dict[str, str],
    weight_bits: int,
) -> Optional[float]:
    """
    Max over all output channels (all selected Linear/Conv) of ``s_global / s_per_channel``.
    Same ``_symmetric_scale`` as PTQ; ≥1 when some channel uses a tighter scale than the global max.
    """
    chunks: List[torch.Tensor] = []
    for name, kind in selected.items():
        if kind not in ("linear", "conv2d"):
            continue
        mod = model.get_submodule(name)
        w = _extract_weight_tensor(mod, kind)
        if w is None:
            continue
        s_g = _symmetric_scale(w, weight_bits, axis=None)
        s_pc = _symmetric_scale(w, weight_bits, axis=0)
        r = s_g.reshape(()) / s_pc.reshape(-1).clamp_min(1e-8)
        chunks.append(r)
    if not chunks:
        return None
    cat = torch.cat(chunks)
    return float(cat.max().item())


def _weight_quant_axis(kind: str, cfg: Any) -> Optional[int]:
    """
    Symmetric weight quantization axis. ``Linear``/``Conv2d``: axis 0 (per output filter) when
    ``cfg.per_output_channel`` else **None** (single global scale). Affine/coeff: always
    **None** (one scale over the whole tensor). ``AffineMatMul`` / ``AffineHadamard``: no weight tensor.
    """
    if kind in {"linear", "conv2d"}:
        return 0 if cfg.per_output_channel else None
    return None


_WEIGHT_ATTR_BY_KIND: Dict[str, str] = {
    "linear": "weight",
    "conv2d": "weight",
    "affine_scale": "scale",
    "affine_scale_bias": "weight",
    "affine_fixed_mix": "weight",
    "affine_contract": "coeff",
}


def _extract_weight_tensor(module: nn.Module, kind: str) -> Optional[torch.Tensor]:
    attr = _WEIGHT_ATTR_BY_KIND.get(kind)
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
    if kind == "affine_fixed_mix":
        return torch.einsum(einsum_equation, q_weight, q_inputs[0])
    if kind == "affine_contract":
        return torch.einsum(einsum_equation, q_weight, q_inputs[0])
    if kind == "matmul":
        return torch.matmul(q_inputs[0], q_inputs[1])
    if kind == "matmul_hadamard":
        return q_inputs[0] * q_inputs[1]
    raise ValueError(f"Unsupported PTQ kind: {kind}")


def _fit_affine_dequant(
    acc_samples: Sequence[torch.Tensor],
    out_samples: Sequence[torch.Tensor],
    *,
    channel_axis: Optional[int],
    per_output_channel: bool,
    var_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not acc_samples:
        return torch.tensor(1.0), torch.tensor(0.0)
    if not per_output_channel or channel_axis is None:
        x = torch.cat([a.reshape(-1).to(dtype=torch.float32) for a in acc_samples], dim=0)
        y = torch.cat([b.reshape(-1).to(dtype=torch.float32) for b in out_samples], dim=0)
        assert x.numel() == y.numel(), "accumulator / output length mismatch in global affine dequant fit"
        mx = x.mean()
        my = y.mean()
        var = (x * x).mean() - mx * mx
        cov = (x * y).mean() - mx * my
        if float(var.abs().item()) < var_eps:
            # Pooled quantized acc nearly constant; fall back to second-moment OLS through origin.
            denom = (x * x).mean().clamp_min(var_eps)
            s = (x * y).mean() / denom
            c = my - s * mx
            return s.to(dtype=torch.float32), c.to(dtype=torch.float32)
        s = cov / var
        c = my - s * mx
        return s.to(dtype=torch.float32), c.to(dtype=torch.float32)

    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    for acc, out in zip(acc_samples, out_samples):
        axis = channel_axis if channel_axis >= 0 else acc.ndim + channel_axis
        if axis < 0 or axis >= acc.ndim or acc.shape[axis] != out.shape[axis]:
            return _fit_affine_dequant(
                acc_samples, out_samples, channel_axis=None, per_output_channel=False, var_eps=var_eps
            )
        xs.append(acc.movedim(axis, -1).reshape(-1, acc.shape[axis]).to(dtype=torch.float32))
        ys.append(out.movedim(axis, -1).reshape(-1, out.shape[axis]).to(dtype=torch.float32))
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    mx = x.mean(dim=0)
    my = y.mean(dim=0)
    var = (x * x).mean(dim=0) - mx * mx
    cov = (x * y).mean(dim=0) - mx * my
    s = torch.where(var.abs() < var_eps, torch.zeros_like(var), cov / var.clamp_min(var_eps))
    c = my - s * mx
    return s.to(dtype=torch.float32), c.to(dtype=torch.float32)


def _linear_conv_out_scale_from_quant_scales(
    kind: str,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Float output from quantized matmul/conv (symmetric): x ≈ s_in q_x, W ≈ s_w q_w ⇒ y ≈ s_in s_w acc.
    No OLS on slope — only the product of input and weight quantization scales.
    """
    s_in = input_scale[0].to(dtype=torch.float32)
    ws = weight_scale.to(dtype=torch.float32)
    if ws.ndim == 0:
        return s_in * ws
    return (s_in * ws.reshape(-1)).to(dtype=torch.float32)


def _calibrated_out_bias_fixed_scale(
    acc_samples: Sequence[torch.Tensor],
    out_samples: Sequence[torch.Tensor],
    out_scale: torch.Tensor,
    *,
    channel_axis: Optional[int],
    per_channel: bool,
) -> torch.Tensor:
    """Best constant(s) b minimizing mismatch: y ≈ out_scale * acc + b (mean residual per channel or global)."""
    if not acc_samples:
        return torch.tensor(0.0, dtype=torch.float32)
    out_scale = out_scale.to(dtype=torch.float32)
    if not per_channel or channel_axis is None or out_scale.ndim == 0:
        os = out_scale.reshape(())
        diffs: List[torch.Tensor] = []
        for acc, out in zip(acc_samples, out_samples):
            diffs.append((out.to(dtype=torch.float32) - os * acc.to(dtype=torch.float32)).reshape(-1))
        return torch.cat(diffs, dim=0).mean()

    vecs: List[torch.Tensor] = []
    for acc, out in zip(acc_samples, out_samples):
        pred = _broadcast_named_axis(out_scale, acc, channel_axis) * acc.to(dtype=torch.float32)
        diff = out.to(dtype=torch.float32) - pred
        axis = channel_axis if channel_axis >= 0 else acc.ndim + channel_axis
        reduce_dims = tuple(i for i in range(diff.ndim) if i != axis)
        vecs.append(diff.mean(dim=reduce_dims) if reduce_dims else diff)
    return torch.stack(vecs, dim=0).mean(dim=0).to(dtype=torch.float32)


def _tensor_stats(tensors: Sequence[torch.Tensor]) -> Dict[str, float]:
    if not tensors:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    flat = torch.cat([t.reshape(-1).to(dtype=torch.float32) for t in tensors], dim=0)
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
    }


def _calibrated_input_scales(data: NodeCalibrationData, activation_bits: int) -> torch.Tensor:
    """One symmetric scale per input tensor: max |x| over all axes (channels share the quant step for dot-product / conv sums)."""
    qmax = float(max(_signed_qrange(activation_bits)[1], 1))
    in_scales: List[torch.Tensor] = []
    for input_idx in range(len(data.input_samples[0])):
        max_abs = torch.tensor(0.0, dtype=torch.float32)
        for sample in data.input_samples:
            cur = sample[input_idx].abs().max()
            max_abs = torch.maximum(max_abs, cur.to(dtype=torch.float32))
        in_scales.append((max_abs / qmax).clamp_min(1e-8))
    return torch.stack(in_scales).to(dtype=torch.float32)


class CalibratedAffinePTQWrapper(nn.Module):
    def __init__(self, name: str, module: nn.Module, data: NodeCalibrationData, cfg: Any) -> None:
        super().__init__()
        if not data.input_samples or not data.output_samples:
            raise ValueError(f"No calibration samples cached for node {name}")

        self.node_name = name
        self.kind = _module_kind(module)
        if self.kind is None:
            raise ValueError(f"Unsupported PTQ module at {name}: {type(module).__name__}")
        self.activation_group = _activation_group_for_kind(self.kind)
        self.activation_bits = _activation_bits_for_kind(self.kind, cfg)
        self.weight_bits = int(cfg.weight_bits)
        self.einsum_equation = getattr(module, "einsum_equation", None)
        self.stride = tuple(module.stride) if isinstance(module, nn.Conv2d) else None
        self.padding = tuple(module.padding) if isinstance(module, nn.Conv2d) else None
        self.dilation = tuple(module.dilation) if isinstance(module, nn.Conv2d) else None
        self.groups = int(module.groups) if isinstance(module, nn.Conv2d) else 1
        out_example = data.output_samples[0]
        self.output_channel_axis = _preferred_output_channel_axis(self.kind, out_example)
        # Per-channel output *bias* when tensor has an output channel axis (linear/conv); scale is analytical for those kinds.
        self.per_channel_output_affine = self.output_channel_axis is not None
        self.per_output_channel_weights_config = bool(cfg.per_output_channel)
        self.input_arity = len(data.input_samples[0])

        input_scale = _calibrated_input_scales(data, self.activation_bits)
        self.register_buffer("input_scale", input_scale)
        self.register_buffer("input_inv_scale", input_scale.reciprocal().to(dtype=torch.float32))
        self.register_buffer("input_zero_point", torch.zeros(self.input_arity, dtype=torch.float32))
        self._act_qmin, self._act_qmax = _signed_qrange(self.activation_bits)

        weight_axis_used: Optional[int] = None
        weight_fp = _extract_weight_tensor(module, self.kind)
        q_weight: Optional[torch.Tensor] = None
        if weight_fp is not None:
            weight_axis_used = _weight_quant_axis(self.kind, cfg)
            weight_scale = _symmetric_scale(weight_fp, self.weight_bits, axis=weight_axis_used)
            q_weight = _quantize_proxy(weight_fp, weight_scale, self.weight_bits).to(dtype=torch.float32)
            self.register_buffer("weight_scale", weight_scale.to(dtype=torch.float32))
            self.register_buffer("weight_zero_point", torch.zeros_like(weight_scale, dtype=torch.float32))
        else:
            self.register_buffer("weight_scale", torch.tensor(1.0, dtype=torch.float32))
            self.register_buffer("weight_zero_point", torch.tensor(0.0, dtype=torch.float32))
        self.per_output_channel_weights_effective = weight_axis_used is not None

        acc_samples: List[torch.Tensor] = []
        for input_sample in data.input_samples:
            q_inputs = [
                _quantize_proxy(inp, input_scale[i], self.activation_bits)
                for i, inp in enumerate(input_sample)
            ]
            acc = _simulate_accumulator(
                self.kind,
                q_inputs,
                q_weight=q_weight,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                einsum_equation=self.einsum_equation,
            )
            acc_samples.append(acc.to(dtype=torch.float32))
        if self.kind in ("linear", "conv2d") and q_weight is not None:
            out_scale = _linear_conv_out_scale_from_quant_scales(self.kind, input_scale, self.weight_scale)
            out_bias = _calibrated_out_bias_fixed_scale(
                acc_samples,
                data.output_samples,
                out_scale,
                channel_axis=self.output_channel_axis,
                per_channel=self.per_channel_output_affine,
            )
            self.out_scale_mode = "analytical_s_in_times_s_w"
        else:
            out_scale, out_bias = _fit_affine_dequant(
                acc_samples,
                data.output_samples,
                channel_axis=self.output_channel_axis,
                per_output_channel=self.per_channel_output_affine,
                var_eps=float(cfg.dequant_var_eps),
            )
            self.out_scale_mode = "ols_affine"

        out_ref = data.output_samples[0]
        out_rank = out_ref.ndim
        if (
            self.per_channel_output_affine
            and self.output_channel_axis is not None
            and out_rank > 0
            and out_scale.ndim > 0
        ):
            axis = self.output_channel_axis if self.output_channel_axis >= 0 else out_rank + self.output_channel_axis
            bshape = [1] * out_rank
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

        self.to(device=_module_device(module))

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
            "per_output_channel": self.per_output_channel_weights_config,
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
        instance.activation_group = _activation_group_for_kind(instance.kind)
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
    input_stats = []
    n_inputs = len(data.input_samples[0]) if data.input_samples else 0
    for idx in range(n_inputs):
        input_stats.append(_tensor_stats([sample[idx] for sample in data.input_samples]))
    output_stats = _tensor_stats(list(data.output_samples))
    return {
        "stored_examples": data.stored_examples,
        "num_calls": data.num_calls,
        "input_stats": input_stats,
        "output_stats": output_stats,
    }


@torch.no_grad()
def validate_model(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    *,
    calibration: Optional[CalibrationController] = None,
) -> Tuple[float, float]:
    model.eval()
    device = get_device()
    use_cuda = device.type == "cuda"
    loss_sum_t = torch.zeros((), device=device, dtype=torch.float64)
    correct_t = torch.zeros((), device=device, dtype=torch.long)
    n = 0
    try:
        input_dtype = next(model.parameters()).dtype
    except StopIteration:
        input_dtype = get_surgery_dtype()

    for bi, (x, y) in enumerate(loader):
        x = x.to(device, dtype=input_dtype, non_blocking=use_cuda)
        y = y.to(device, non_blocking=use_cuda)
        if calibration is not None:
            calibration.current_batch = bi
        logits = model(x)
        loss_sum_t += criterion(logits.float(), y).double() * y.size(0)
        correct_t += (logits.argmax(dim=-1) == y).sum()
        n += y.size(0)
    denom = max(n, 1)
    return correct_t.item() / denom, loss_sum_t.item() / denom


def _build_wrapped_model(
    fp_model: nn.Module,
    selected: Dict[str, str],
    calibration_cache: Dict[str, NodeCalibrationData],
    cfg: Any,
) -> Tuple[nn.Module, List[Dict[str, Any]], List[Dict[str, Any]]]:
    wrapped = copy.deepcopy(fp_model)
    node_meta: List[Dict[str, Any]] = []
    reload_configs: List[Dict[str, Any]] = []
    for name in selected:
        cal = calibration_cache[name]
        wrapper = CalibratedAffinePTQWrapper(name, wrapped.get_submodule(name), cal, cfg)
        _set_module(wrapped, name, wrapper)
        meta = wrapper.metadata()
        meta["calibration_debug"] = _node_debug_metadata(cal)
        node_meta.append(meta)
        reload_configs.append(wrapper.reload_config())
    return wrapped, node_meta, reload_configs


def load_ptq_wrapped_checkpoint(path: str, cfg: Any) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a PTQ-wrapped checkpoint: rebuild the FP surgery template, install skeleton wrappers, then load_state_dict."""
    device = get_device()
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    extra = dict(payload["extra"])
    adapter = get_model_adapter(extra.get("model_key", getattr(cfg, "model_key", None)))
    if getattr(cfg, "top_k", None) is not None:
        extra["top_k"] = int(cfg.top_k)
    if getattr(cfg, "eps", None) is not None:
        extra["eps_ln"] = float(cfg.eps)
    extra["surgery_dtype"] = describe_dtype(get_surgery_dtype())
    extra.setdefault("model_key", adapter.key)
    extra.setdefault("patient", adapter.patient_name)
    extra.setdefault("dataset", adapter.dataset_name)
    model = adapter.build_surgery_model_from_extra(extra, cfg).to(device=device, dtype=get_surgery_dtype())
    for wc in extra.get("ptq_wrappers", []):
        skeleton = CalibratedAffinePTQWrapper.from_reload_config(wc).to(device=device)
        _set_module(model, wc["name"], skeleton)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, extra


def _ptq_summary(
    cfg: Any,
    selected: Dict[str, str],
    fp_acc: float,
    fp_loss: float,
    ptq_acc: float,
    ptq_loss: float,
    node_meta: List[Dict[str, Any]],
    *,
    linear_conv_max_s_global_over_s_pc: Optional[float] = None,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "fp_val_acc": float(fp_acc),
        "fp_val_loss": float(fp_loss),
        "ptq_val_acc": float(ptq_acc),
        "ptq_val_loss": float(ptq_loss),
        "acc_delta": float(ptq_acc - fp_acc),
        "loss_delta": float(ptq_loss - fp_loss),
    }
    if linear_conv_max_s_global_over_s_pc is not None:
        metrics["linear_conv_max_s_global_over_s_pc"] = linear_conv_max_s_global_over_s_pc
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
            "batches": int(cfg.calibration_batches),
            "examples_per_node": int(cfg.calibration_examples_per_node),
        },
        "metrics": metrics,
        "wrapped_nodes": node_meta,
    }


def run_ptq(
    cfg: Any,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    device = get_device() if device is None else device
    dtype = get_surgery_dtype() if dtype is None else dtype
    fp_path = os.path.abspath(cfg.fp_checkpoint)
    out_abs = traceable_artifact_path(cfg.output, cfg, "ts-ptq", "wrapped", ".pt")
    meta_abs = metadata_path_for_checkpoint(out_abs)
    model_log_abs = traceable_log_path(cfg.log_dir, cfg, "ts-ptq", "model_after_ptq")
    cfg.output = out_abs

    print(f"Using device: {describe_device(device)}", flush=True)
    print(f"Requested surgery dtype: {describe_dtype(dtype)}", flush=True)
    if cfg.config_json_path:
        print(f"config_json={cfg.config_json_path}", flush=True)

    criterion = nn.CrossEntropyLoss()

    fp_model, fp_extra = load_surgery_student_checkpoint(fp_path, cfg)
    adapter = get_model_adapter(fp_extra.get("model_key", getattr(cfg, "model_key", None)))
    _train_loader, val_loader = adapter.build_loaders(cfg)
    print(f"Using model adapter: {adapter.key}", flush=True)
    selected = _build_node_selection(fp_model, cfg)
    if not selected:
        raise SystemExit("No PTQ-wrappable nodes selected by the current config.")

    print(f"Selected {len(selected)} PTQ node(s).", flush=True)
    for name, kind in selected.items():
        print(f"  {name}: {kind}", flush=True)

    controller = CalibrationController(
        selected=selected,
        max_batches=int(cfg.calibration_batches),
        max_examples_per_node=int(cfg.calibration_examples_per_node),
    )
    controller.register(fp_model)
    fp_acc, fp_loss = validate_model(fp_model, val_loader, criterion, calibration=controller)
    controller.close()

    print(f"Float model val acc={fp_acc:.4f} loss={fp_loss:.4f}", flush=True)

    missing = [name for name in selected if not controller.cache[name].output_samples]
    if missing:
        raise SystemExit(f"Calibration samples missing for selected node(s): {missing}")

    wrapped_model, node_meta, reload_configs = _build_wrapped_model(
        fp_model, selected, controller.cache, cfg
    )

    linear_conv_max_sg_over_spc = _diagnostic_linear_conv_max_s_global_over_s_pc(
        fp_model, selected, int(cfg.weight_bits)
    )

    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    out_extra = dict(fp_extra)
    out_extra["ptq_meta_path"] = os.path.basename(meta_abs)
    out_extra["ptq_wrappers"] = reload_configs
    write_model_structure_txt(model_log_abs, wrapped_model, "PTQ-Wrapped Surgery Model")
    save_model_checkpoint(out_abs, wrapped_model, extra=out_extra)
    print(f"wrote {model_log_abs}", flush=True)
    print(f"wrote {out_abs}", flush=True)

    del wrapped_model
    reloaded_model, _ = load_ptq_wrapped_checkpoint(out_abs, cfg)
    print("reloaded wrapped checkpoint from disk for validation", flush=True)
    ptq_acc, ptq_loss = validate_model(reloaded_model, val_loader, criterion)

    print(f"Wrapped model val acc={ptq_acc:.4f} loss={ptq_loss:.4f}", flush=True)
    print(f"Delta acc={ptq_acc - fp_acc:+.4f} loss={ptq_loss - fp_loss:+.4f}", flush=True)
    if linear_conv_max_sg_over_spc is not None:
        print(
            f"Linear/Conv weight_scale: max(s_global/s_per_channel)={linear_conv_max_sg_over_spc:.6g} "
            f"(over all selected conv/linear output channels; same symmetric scales as PTQ)",
            flush=True,
        )

    meta = _ptq_summary(
        cfg,
        selected,
        fp_acc,
        fp_loss,
        ptq_acc,
        ptq_loss,
        node_meta,
        linear_conv_max_s_global_over_s_pc=linear_conv_max_sg_over_spc,
    )

    with open(meta_abs, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {meta_abs}", flush=True)
