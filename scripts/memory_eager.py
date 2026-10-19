"""Peak GPU memory of an eager evaluation pass with one model resident (paper Table 2).

Surgery records the peak of its evaluation pass while the reference model stays on the GPU, so the compiled-model
peaks include the reference weights and the reference peak does not. This script measures every model of Table 2 and
the reference the same way: the GPU holds nothing before the model is loaded, not even the cuBLAS workspace; then the
model alone runs one eager bf16 evaluation pass over the held-out Pet test split with the surgery evaluation function,
and the peak allocated memory of that pass is recorded. Accuracy is recorded as a check against the surgery metadata.

Usage: python scripts/memory_eager.py OUT_JSON [--names topk16_fast ...]
"""

from __future__ import annotations

import argparse
import gc
import json
import warnings
from typing import Any, Dict, List

import torch

from transformer_surgery.cli.ptq_config import PTQSurgeryConfig
from transformer_surgery.internal.util import set_default_device, set_surgery_dtype
from transformer_surgery.models.adapters import get_model_adapter, load_surgery_student_checkpoint

REFERENCE = "artifacts/checkpoints/ts_pretrain_pet_deit_tiny.pt"
GRAPHS = ("fast", "hadamard_sr_gs", "strict_sr_gs", "strict_sr_gs_pol")
NAMES: List[str] = [f"topk{k}_{g}" for k in (16, 32, 64, 197) for g in GRAPHS] + [
    f"topk{k}_{g}_qc32" for k in (16, 32, 64) for g in GRAPHS
]


def model_bytes(model: torch.nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--names", nargs="+", default=NAMES)
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    device = torch.device("cuda")
    set_default_device(device)
    set_surgery_dtype(torch.bfloat16)
    cfg = PTQSurgeryConfig()
    cfg.top_k, cfg.eps = None, None
    adapter = get_model_adapter("deit_tiny_pet")
    _, test_loader = adapter.build_loaders(cfg)
    rows: List[Dict[str, Any]] = []

    def measure(name: str, load) -> None:
        gc.collect()
        torch._C._cuda_clearCublasWorkspaces()  # the cuBLAS workspace counts as allocated memory; every pass re-allocates it
        torch.cuda.empty_cache()
        empty = torch.cuda.memory_allocated(device)
        if empty != 0:
            raise RuntimeError(f"{empty} bytes resident before loading {name}")
        model = load()
        gc.collect()
        torch.cuda.synchronize(device)
        resident = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        acc, loss, _ = adapter.evaluate(model, test_loader, cfg)
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        rows.append({"name": name, "peak_bytes": peak, "resident_bytes": resident, "model_bytes": model_bytes(model),
                     "acc": acc, "loss": loss})
        print(f"{name:32s} peak {peak / 1e6:9.1f} MB, resident {resident / 1e6:7.1f} MB, "
              f"model {model_bytes(model) / 1e6:7.1f} MB, acc {100 * acc:.2f}", flush=True)
        del model

    measure("reference", lambda: adapter.load_reference_checkpoint(REFERENCE).to(device, torch.bfloat16))
    for name in args.names:
        measure(name, lambda: load_surgery_student_checkpoint(f"artifacts/checkpoints/ts_surgery_{name}.pt", cfg)[0])
    ref = rows[0]["peak_bytes"]
    for r in rows:
        r["peak_vs_ref"] = r["peak_bytes"] / ref
    with open(args.out, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(device), "rows": rows}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
