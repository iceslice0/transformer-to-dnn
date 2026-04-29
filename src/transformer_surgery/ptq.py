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


def _get_module(root: nn.Module, name: str) -> nn.Module:
    mod = root
    for part in name.split("."):
        mod = mod._modules[part]
    return mod


def _set_module(root: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent._modules[part]
    parent._modules[parts[-1]] = new_module


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
            hook = _get_module(model, name).register_forward_hook(self._hook(name))
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
        mod = _get_module(model, name)
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


def _extract_weight_tensor(module: nn.Module, kind: str) -> Optional[torch.Tensor]:
    if kind == "linear":
        return module.weight.detach().to(dtype=torch.float32, device="cpu")
    if kind == "conv2d":
        return module.weight.detach().to(dtype=torch.float32, device="cpu")
    if kind == "affine_scale":
        return module.scale.detach().to(dtype=torch.float32, device="cpu")
    if kind == "affine_scale_bias":
        return module.weight.detach().to(dtype=torch.float32, device="cpu")
    if kind == "affine_fixed_mix":
        return module.weight.detach().to(dtype=torch.float32, device="cpu")
    if kind == "affine_contract":
        return module.coeff.detach().to(dtype=torch.float32, device="cpu")
    return None


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
        if float(var.abs().item()) < 1e-8:
            # Pooled quantized acc nearly constant; avoid y ≈ constant(my).
            denom = (x * x).mean().clamp_min(1e-12)
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
            return _fit_affine_dequant(acc_samples, out_samples, channel_axis=None, per_output_channel=False)
        xs.append(acc.movedim(axis, -1).reshape(-1, acc.shape[axis]).to(dtype=torch.float32))
        ys.append(out.movedim(axis, -1).reshape(-1, out.shape[axis]).to(dtype=torch.float32))
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    mx = x.mean(dim=0)
    my = y.mean(dim=0)
    var = (x * x).mean(dim=0) - mx * mx
    cov = (x * y).mean(dim=0) - mx * my
    s = torch.where(var.abs() < 1e-8, torch.zeros_like(var), cov / var.clamp_min(1e-8))
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

        self.register_buffer("input_scale", _calibrated_input_scales(data, self.activation_bits))
        self.register_buffer("input_zero_point", torch.zeros(self.input_arity, dtype=torch.float32))

        weight_axis_used: Optional[int] = None
        weight_fp = _extract_weight_tensor(module, self.kind)
        if weight_fp is not None:
            weight_axis_used = _weight_quant_axis(self.kind, cfg)
            weight_scale = _symmetric_scale(weight_fp, self.weight_bits, axis=weight_axis_used)
            q_weight = _quantize_proxy(weight_fp, weight_scale, self.weight_bits).to(dtype=torch.float32)
            self.register_buffer("weight_scale", weight_scale.to(dtype=torch.float32))
            self.register_buffer("weight_zero_point", torch.zeros_like(weight_scale, dtype=torch.float32))
            self.register_buffer("q_weight", q_weight)
        else:
            self.register_buffer("weight_scale", torch.tensor(1.0, dtype=torch.float32))
            self.register_buffer("weight_zero_point", torch.tensor(0.0, dtype=torch.float32))
            self.q_weight = None
        self.per_output_channel_weights_effective = weight_axis_used is not None

        acc_samples: List[torch.Tensor] = []
        for input_sample in data.input_samples:
            fp_inputs = tuple(input_sample)
            q_inputs = [
                _quantize_proxy(inp, self.input_scale[i], self.activation_bits).to(dtype=torch.float32)
                for i, inp in enumerate(fp_inputs)
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
            acc_samples.append(acc.to(dtype=torch.float32))
        if self.kind in ("linear", "conv2d") and self.q_weight is not None:
            out_scale = _linear_conv_out_scale_from_quant_scales(self.kind, self.input_scale, self.weight_scale)
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
            )
            self.out_scale_mode = "ols_affine"
        self.register_buffer("out_scale", out_scale.to(dtype=torch.float32))
        self.register_buffer("out_bias", out_bias.to(dtype=torch.float32))

        self.to(device=_module_device(module))

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        fp_inputs = tuple(inputs)
        q_inputs = [
            _quantize_proxy(inp, self.input_scale[i], self.activation_bits).to(dtype=torch.float32)
            for i, inp in enumerate(fp_inputs)
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
        out_scale = _broadcast_named_axis(self.out_scale, acc, self.output_channel_axis if self.per_channel_output_affine else None)
        out_bias = _broadcast_named_axis(self.out_bias, acc, self.output_channel_axis if self.per_channel_output_affine else None)
        out = out_scale.to(device=acc.device, dtype=torch.float32) * acc + out_bias.to(device=acc.device, dtype=torch.float32)
        return out.to(dtype=fp_inputs[0].dtype)

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
            "out_scale": self.out_scale.detach().cpu().tolist(),
            "out_bias": self.out_bias.detach().cpu().tolist(),
            "stride": list(self.stride) if self.stride is not None else None,
            "padding": list(self.padding) if self.padding is not None else None,
            "dilation": list(self.dilation) if self.dilation is not None else None,
            "groups": self.groups,
            "einsum_equation": self.einsum_equation,
        }


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
    n = 0
    correct = 0
    loss_sum = 0.0
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
        loss_sum += criterion(logits.float(), y).item() * y.size(0)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        n += y.size(0)
    return correct / max(n, 1), loss_sum / max(n, 1)


def _build_wrapped_model(
    fp_model: nn.Module,
    selected: Dict[str, str],
    calibration_cache: Dict[str, NodeCalibrationData],
    cfg: Any,
) -> Tuple[nn.Module, List[Dict[str, Any]]]:
    wrapped = copy.deepcopy(fp_model)
    node_meta: List[Dict[str, Any]] = []
    for name in selected:
        cal = calibration_cache[name]
        wrapper = CalibratedAffinePTQWrapper(name, _get_module(wrapped, name), cal, cfg)
        _set_module(wrapped, name, wrapper)
        meta = wrapper.metadata()
        meta["calibration_debug"] = _node_debug_metadata(cal)
        node_meta.append(meta)
    return wrapped, node_meta


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
            "per_output_channel_weights": bool(cfg.per_output_channel),
            "per_output_channel_linear_conv_weights": bool(cfg.per_output_channel),
            "per_output_channel_note": (
                "When true: per-output-channel weight scales for Linear/Conv2d. When false: global weight "
                "scale for Linear/Conv2d. Affine/coeff: always global weight scale. Ignored for AffineMatMul/AffineHadamard."
            ),
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
    print(f"Loaded checkpoint dtype: {describe_dtype(get_surgery_dtype())}", flush=True)
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

    wrapped_model, node_meta = _build_wrapped_model(fp_model, selected, controller.cache, cfg)
    ptq_acc, ptq_loss = validate_model(wrapped_model, val_loader, criterion)

    linear_conv_max_sg_over_spc = _diagnostic_linear_conv_max_s_global_over_s_pc(
        fp_model, selected, int(cfg.weight_bits)
    )

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

    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(meta_abs) or ".", exist_ok=True)

    out_extra = dict(fp_extra)
    out_extra["ptq"] = {
        "selected_nodes": list(selected.keys()),
        "weight_bits": int(cfg.weight_bits),
        "activation_bits": int(cfg.activation_bits),
        "affine_activation_bits": int(_activation_bits_for_kind("linear", cfg)),
        "matmul_activation_bits": int(_activation_bits_for_kind("matmul", cfg)),
        "per_output_channel_weights": bool(cfg.per_output_channel),
        "per_output_channel": bool(cfg.per_output_channel),
        "calibration_batches": int(cfg.calibration_batches),
        "calibration_examples_per_node": int(cfg.calibration_examples_per_node),
    }
    write_model_structure_txt(model_log_abs, wrapped_model, "PTQ-Wrapped Surgery Model")
    save_model_checkpoint(out_abs, wrapped_model, extra=out_extra)
    with open(meta_abs, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {model_log_abs}", flush=True)
    print(f"wrote {out_abs}", flush=True)
    print(f"wrote {meta_abs}", flush=True)
