"""Model-only GPU latency of compiled models.

A whole validation pass is bound by data loading on DeiT-Tiny/Pet (about 85 ms per batch of 32 for JPEG decoding,
against 15 to 76 ms for the eager forward), so pass time does not measure the model. This script times the forward pass
alone: Pet validation batches are moved to the GPU first, each model runs in bf16 under autocast (the surgery dtype),
and ``torch.compile`` fuses the graph, as an NPU toolchain compiles a whole network. Each model is compiled in every mode
of ``--modes`` and timed with CUDA events after a warm-up; the fastest mode is reported, together with the peak GPU
memory of the forward above the loaded weights and batches.

Usage: python scripts/latency_compiled.py OUT_JSON [--modes default reduce-overhead] [--names topk16_fast ...]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import warnings
from typing import Any, Dict, List

import torch

from transformer_surgery.cli.ptq_config import PTQSurgeryConfig
from transformer_surgery.internal.util import maybe_surgery_cuda_autocast, set_default_device, set_surgery_dtype
from transformer_surgery.models.adapters import get_model_adapter, load_surgery_student_checkpoint

REFERENCE = "artifacts/checkpoints/ts_pretrain_pet_deit_tiny.pt"
TOPK = (16, 32, 64, 197)
ROUTED = (16, 32, 64)
# Four graphs with top-k and routed support, the extended graph at every k, the routing benchmark (strict with GELU).
NAMES: List[str] = (
    [f"topk{k}_{g}" for k in TOPK for g in ("fast", "hadamard_sr_gs", "strict_sr_gs", "strict_sr_gs_pol")]
    + [f"topk{k}_{g}_qc32" for k in ROUTED for g in ("fast", "hadamard_sr_gs", "strict_sr_gs", "strict_sr_gs_pol")]
    + [f"topk{k}_fast" for k in (1, 2, 4, 8, 96, 128)]
    + [f"topk{k}_strict_sr{s}" for k in ROUTED for s in ("", "_qc32")]
)


def time_model(model: torch.nn.Module, xs: List[torch.Tensor], modes: List[str], iters: int) -> Dict[str, Any]:
    device = torch.device("cuda")
    out: Dict[str, Any] = {}
    for mode in modes:
        torch._dynamo.reset()
        compiled = torch.compile(model, mode=None if mode == "eager" else mode, disable=mode == "eager")
        with torch.no_grad(), maybe_surgery_cuda_autocast(device, torch.bfloat16):
            t0 = time.perf_counter()
            for x in xs[:3]:
                compiled(x)
            torch.cuda.synchronize()
            compile_s = time.perf_counter() - t0
            base = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            per_batch = []
            for _ in range(iters):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                for x in xs:
                    compiled(x)
                b.record()
                torch.cuda.synchronize()
                per_batch.append(a.elapsed_time(b) / len(xs))
            peak = torch.cuda.max_memory_allocated(device) - base
        out[mode] = {
            "ms_per_batch": statistics.median(per_batch),
            "ms_per_batch_runs": per_batch,
            "peak_mb": peak / 1e6,
            "warmup_compile_s": compile_s,
        }
        del compiled
    best = min(out, key=lambda m: out[m]["ms_per_batch"])
    return {"modes": out, "fastest_mode": best, "ms_per_batch": out[best]["ms_per_batch"], "peak_mb": out[best]["peak_mb"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--modes", nargs="+", default=["default", "reduce-overhead"])
    parser.add_argument("--names", nargs="+", default=NAMES)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    device = torch.device("cuda")
    set_default_device(device)
    torch.backends.cudnn.benchmark = True
    cfg = PTQSurgeryConfig()
    cfg.top_k, cfg.eps = None, None
    adapter = get_model_adapter("deit_tiny_pet")
    _, val = adapter.build_loaders(cfg)
    xs = [x.to(device, torch.bfloat16) for i, (x, _) in enumerate(val) if i < args.batches]
    rows: List[Dict[str, Any]] = []

    def record(name: str, model: torch.nn.Module) -> None:
        if not next(model.parameters()).is_cuda:
            raise RuntimeError("latency runs on the GPU only")
        r = {"name": name, "batch_size": xs[0].shape[0], **time_model(model.eval(), xs, args.modes, args.iters)}
        rows.append(r)
        print(f"{name:32s} {r['ms_per_batch']:8.2f} ms/batch ({r['fastest_mode']}); "
              + ", ".join(f"{m} {v['ms_per_batch']:.2f} ms, compile {v['warmup_compile_s']:.0f} s" for m, v in r["modes"].items()),
              flush=True)

    set_surgery_dtype(torch.bfloat16)
    record("reference", adapter.load_reference_checkpoint(REFERENCE).to(device, torch.bfloat16))
    for name in args.names:
        model, _ = load_surgery_student_checkpoint(f"artifacts/checkpoints/ts_surgery_{name}.pt", cfg)
        record(name, model)
        del model
        torch.cuda.empty_cache()
    ref = rows[0]["ms_per_batch"]
    for r in rows:
        r["latency_vs_ref"] = r["ms_per_batch"] / ref
    with open(args.out, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(device), "modes": args.modes, "rows": rows}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
