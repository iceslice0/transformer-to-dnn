"""Per-module GPU time and peak-memory profile of surgery checkpoints (strict vs fast).

Times the top-level attention, LayerNorm and MLP submodules of every block with CUDA events and
records the extra peak memory each one allocates. Used to check where the strict path spends the
time and memory it adds over the fast path. Routed checkpoints (support_routing=query_centroids) call
``dot.selected``, ``gibbs.forward_selected`` and ``router.support`` as methods, which forward hooks
do not see; these methods are wrapped and timed under the same groups.

``--reference`` adds the Pet reference (timm DeiT-Tiny) with the same groups: its attention runs unfused, with
``q k^T``, softmax and ``attn @ v`` as separate hookable modules (checked equal to timm's unfused attention), and a second
pass times its fused attention kernel (``F.scaled_dot_product_attention``), the path the latency tables use.

Example:
    python scripts/profile_modules.py \
        --checkpoints artifacts/checkpoints/ts_surgery_topk64_strict.pt \
                      artifacts/checkpoints/ts_surgery_topk64_fast.pt \
        --reference artifacts/checkpoints/ts_pretrain_pet_deit_tiny.pt \
        --out artifacts/logs/profile_2026-10-03/topk64.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_surgery.cli.ptq_config import PTQSurgeryConfig
from transformer_surgery.internal.util import (
    get_device,
    get_surgery_dtype,
    maybe_surgery_cuda_autocast,
    set_default_device,
    set_surgery_dtype,
)
from transformer_surgery.models.adapters import get_model_adapter, load_surgery_student_checkpoint

# Methods called outside ``forward`` on the routed sparse path, per group module.
METHODS = {
    ".attn.dot": ("selected",),
    ".attn.gibbs": ("forward_selected",),
    ".attn.router": ("support",),
}

# Module-name suffix -> reported group. Suffixes are matched against direct children of each block.
GROUPS = {
    ".attn.dot": "Score (QK^T)",
    ".attn.sparse_mix": "Mix (PV)",
    ".attn.gibbs": "Softmax (top-k Gibbs)",
    ".attn.router": "Routing (centroid select)",
    ".norm1": "LayerNorm",
    ".norm2": "LayerNorm",
    ".attn.qkv": "Linear (qkv, proj)",
    ".attn.proj": "Linear (qkv, proj)",
    ".mlp": "MLP",
    # Reference attention, split into hookable modules (see ``install_reference_attention``).
    ".attn.score_op": "Score (QK^T)",
    ".attn.softmax_op": "Softmax (top-k Gibbs)",
    ".attn.mix_op": "Mix (PV)",
    ".attn.fused_op": "Attention (fused SDPA)",
}


class _Score(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, scale: float) -> torch.Tensor:
        return (q * scale) @ k.transpose(-2, -1)


class _Softmax(nn.Module):
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return s.softmax(dim=-1)


class _Mix(nn.Module):
    def forward(self, p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return p @ v


class _FusedAttention(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v)


def install_reference_attention(model: nn.Module, fused: bool) -> None:
    """Route every timm attention core through ``score_op``, ``softmax_op``, ``mix_op`` (unfused) or ``fused_op``,
    with the arithmetic of timm's own paths (no mask, eval mode)."""
    for blk in model.blocks:
        attn = blk.attn
        attn.score_op, attn.softmax_op, attn.mix_op, attn.fused_op = _Score(), _Softmax(), _Mix(), _FusedAttention()

        def forward(x, attn_mask=None, is_causal=False, attn=attn):
            b, n, _ = x.shape
            qkv = attn.qkv(x).reshape(b, n, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = attn.q_norm(q), attn.k_norm(k)
            if fused:
                o = attn.fused_op(q, k, v)
            else:
                o = attn.mix_op(attn.softmax_op(attn.score_op(q, k, attn.scale)), v)
            o = o.transpose(1, 2).reshape(b, n, attn.attn_dim)
            return attn.proj_drop(attn.proj(attn.norm(o)))

        attn.forward = forward


def _group_of(name: str) -> str | None:
    for suffix, group in GROUPS.items():
        if name.endswith(suffix) and name.startswith("blocks."):
            return group
    return None


def _methods_of(name: str) -> tuple:
    for suffix, methods in METHODS.items():
        if name.endswith(suffix) and name.startswith("blocks."):
            return methods
    return ()


def profile_checkpoint(path: str, batches: int, warmup: int) -> Dict[str, object]:
    cfg = PTQSurgeryConfig()
    cfg.top_k = None
    cfg.eps = None
    model, extra = load_surgery_student_checkpoint(os.path.abspath(path), cfg)
    model.eval()
    adapter = get_model_adapter(extra["model_key"])
    _, val_loader = adapter.build_loaders(cfg)
    groups, total_ms = _profile(model, val_loader, batches, warmup)
    return {
        "checkpoint": os.path.abspath(path),
        "top_k": extra.get("top_k"),
        "support_routing": extra.get("support_routing", "topk"),
        "strict_product": extra.get("strict_product"),
        "allow_matmul": extra.get("allow_matmul"),
        "hadamard": bool(extra.get("hadamard", False)),
        "surgery_dtype": extra.get("surgery_dtype"),
        "batches": len(total_ms),
        "batch_size": cfg.batch_size,
        "device": torch.cuda.get_device_name(get_device()),
        "forward_ms_per_batch_with_hooks": sum(total_ms) / max(len(total_ms), 1),
        "groups": groups,
    }


def profile_reference(path: str, batches: int, warmup: int) -> Dict[str, object]:
    """The Pet reference in the surgery dtype of the compiled checkpoints (bf16 under autocast): unfused attention for
    the per-group rows, then a second pass that times its fused attention kernel."""
    cfg = PTQSurgeryConfig()
    adapter = get_model_adapter("deit_tiny_pet")
    _, val_loader = adapter.build_loaders(cfg)
    set_surgery_dtype(torch.bfloat16)  # the dtype of the compiled checkpoints
    device, dt = get_device(), get_surgery_dtype()
    model = adapter.load_reference_checkpoint(os.path.abspath(path)).to(device, dt).eval()
    x0 = next(iter(val_loader))[0][:4].to(device, dt)
    with torch.no_grad(), maybe_surgery_cuda_autocast(device, dt):
        for blk in model.blocks:
            blk.attn.fused_attn = False
        timm_unfused = model(x0).float()
        install_reference_attention(model, fused=False)
        ours = model(x0).float()
    if not torch.equal(timm_unfused, ours):
        raise RuntimeError(f"unfused reference attention differs from timm: {(timm_unfused - ours).abs().max().item()}")
    groups, total_ms = _profile(model, val_loader, batches, warmup)
    install_reference_attention(model, fused=True)
    fused_groups, fused_total = _profile(model, val_loader, batches, warmup)
    return {
        "checkpoint": os.path.abspath(path),
        "model": "reference (timm DeiT-Tiny), attention unfused; fused_groups: fused SDPA attention",
        "surgery_dtype": str(dt).replace("torch.", ""),
        "batches": len(total_ms),
        "batch_size": cfg.batch_size,
        "device": torch.cuda.get_device_name(device),
        "forward_ms_per_batch_with_hooks": sum(total_ms) / max(len(total_ms), 1),
        "groups": groups,
        "fused_forward_ms_per_batch_with_hooks": sum(fused_total) / max(len(fused_total), 1),
        "fused_groups": fused_groups,
    }


def _profile(model: nn.Module, val_loader, batches: int, warmup: int):
    """Per-group CUDA time per batch and extra peak memory, from forward hooks on the grouped modules."""
    device = get_device()
    dt = get_surgery_dtype()

    times: Dict[str, List[float]] = defaultdict(list)
    peak_extra: Dict[str, int] = defaultdict(int)
    pending: Dict[str, tuple] = {}
    recording = {"on": False}

    def pre_hook(name: str):
        def fn(_module, _inputs):
            if not recording["on"]:
                return
            torch.cuda.synchronize(device)
            base = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            pending[name] = (start, base)

        return fn

    def post_hook(name: str, group: str):
        def fn(_module, _inputs, _output):
            if not recording["on"] or name not in pending:
                return
            start, base = pending.pop(name)
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize(device)
            times[group].append(start.elapsed_time(end))
            peak_extra[group] = max(peak_extra[group], int(torch.cuda.max_memory_allocated(device)) - base)

        return fn

    def timed_method(name: str, group: str, fn):
        start_fn, end_fn = pre_hook(name), post_hook(name, group)

        def wrapped(*args, **kwargs):
            start_fn(None, args)
            out = fn(*args, **kwargs)
            end_fn(None, args, out)
            return out

        return wrapped

    handles = []
    wrapped_methods = []
    for name, module in model.named_modules():
        group = _group_of(name)
        if group is not None:
            handles.append(module.register_forward_pre_hook(pre_hook(name)))
            handles.append(module.register_forward_hook(post_hook(name, group)))
            for meth in _methods_of(name):
                if hasattr(module, meth):
                    setattr(module, meth, timed_method(f"{name}.{meth}", group, getattr(module, meth)))
                    wrapped_methods.append((module, meth))

    total_ms: List[float] = []
    with torch.no_grad():
        for i, (x, _y) in enumerate(val_loader):
            if i >= warmup + batches:
                break
            x = x.to(device, dtype=next(model.parameters()).dtype)
            recording["on"] = i >= warmup
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with maybe_surgery_cuda_autocast(device, dt):
                model(x)
            end.record()
            torch.cuda.synchronize(device)
            if recording["on"]:
                total_ms.append(start.elapsed_time(end))
    for h in handles:
        h.remove()
    for module, meth in wrapped_methods:
        delattr(module, meth)

    n = max(len(total_ms), 1)
    groups = {
        g: {
            "ms_per_batch": sum(v) / n,
            "peak_extra_mib": peak_extra[g] / 2**20,
        }
        for g, v in times.items()
    }
    return groups, total_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--reference", default=None, help="Pet reference checkpoint to profile with the same groups.")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    set_default_device(torch.device("cuda"))
    results = [profile_checkpoint(p, args.batches, args.warmup) for p in args.checkpoints]
    if args.reference:
        results.append(profile_reference(args.reference, args.batches, args.warmup))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    for r in results:
        print(f"== {os.path.basename(r['checkpoint'])}: forward {r['forward_ms_per_batch_with_hooks']:.2f} ms/batch")
        for g, v in sorted(r["groups"].items(), key=lambda kv: -kv[1]["ms_per_batch"]):
            print(f"   {g:24s} {v['ms_per_batch']:9.3f} ms  peak +{v['peak_extra_mib']:8.1f} MiB")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
