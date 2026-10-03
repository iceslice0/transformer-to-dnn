"""Per-module GPU time and peak-memory profile of surgery checkpoints (strict vs fast).

Times the top-level attention, LayerNorm and MLP submodules of every block with CUDA events and
records the extra peak memory each one allocates. Used to check where the strict path spends the
time and memory it adds over the fast path.

Example:
    python scripts/profile_modules.py \
        --checkpoints artifacts/checkpoints/ts_surgery_topk64_strict.pt \
                      artifacts/checkpoints/ts_surgery_topk64_fast.pt \
        --out artifacts/logs/profile_2026-10-03/topk64.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import torch

from transformer_surgery.cli.ptq_config import PTQSurgeryConfig
from transformer_surgery.internal.util import (
    get_device,
    get_surgery_dtype,
    maybe_surgery_cuda_autocast,
    set_default_device,
)
from transformer_surgery.models.adapters import get_model_adapter, load_surgery_student_checkpoint

# Module-name suffix -> reported group. Suffixes are matched against direct children of each block.
GROUPS = {
    ".attn.dot": "Score (QK^T)",
    ".attn.sparse_mix": "Mix (PV)",
    ".attn.gibbs": "Softmax (top-k Gibbs)",
    ".norm1": "LayerNorm",
    ".norm2": "LayerNorm",
    ".attn.qkv": "Linear (qkv, proj)",
    ".attn.proj": "Linear (qkv, proj)",
    ".mlp": "MLP",
}


def _group_of(name: str) -> str | None:
    for suffix, group in GROUPS.items():
        if name.endswith(suffix) and name.startswith("blocks."):
            return group
    return None


def profile_checkpoint(path: str, batches: int, warmup: int) -> Dict[str, object]:
    cfg = PTQSurgeryConfig()
    cfg.top_k = None
    cfg.eps = None
    model, extra = load_surgery_student_checkpoint(os.path.abspath(path), cfg)
    model.eval()
    adapter = get_model_adapter(extra["model_key"])
    _, val_loader = adapter.build_loaders(cfg)
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

    handles = []
    for name, module in model.named_modules():
        group = _group_of(name)
        if group is not None:
            handles.append(module.register_forward_pre_hook(pre_hook(name)))
            handles.append(module.register_forward_hook(post_hook(name, group)))

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

    n = max(len(total_ms), 1)
    groups = {
        g: {
            "ms_per_batch": sum(v) / n,
            "peak_extra_mib": peak_extra[g] / 2**20,
        }
        for g, v in times.items()
    }
    return {
        "checkpoint": os.path.abspath(path),
        "top_k": extra.get("top_k"),
        "allow_matmul": extra.get("allow_matmul"),
        "surgery_dtype": extra.get("surgery_dtype"),
        "batches": len(total_ms),
        "batch_size": cfg.batch_size,
        "device": torch.cuda.get_device_name(device),
        "forward_ms_per_batch_with_hooks": sum(total_ms) / n,
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    set_default_device(torch.device("cuda"))
    results = [profile_checkpoint(p, args.batches, args.warmup) for p in args.checkpoints]
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
