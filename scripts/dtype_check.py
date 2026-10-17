"""bf16 against fp32 at full support (k = N) for every network: the reference and each compiled graph (paper Supplementary).

bf16 is the paper's setting for DeiT-Tiny and MambaIRv2-Light: weights stored in bf16 and arithmetic under CUDA autocast
(Pythia-70M runs in float32 in the paper). fp32 means fp32 weights and fp32 arithmetic with autocast off. Per graph:
  checkpoint bf16 : the saved compiled checkpoint, evaluated in bf16;
  rebuild bf16    : the graph rebuilt from the checkpoint record in bf16 with the bf16 reference weights (checks the
                    rebuild against the checkpoint);
  rebuild fp32    : the graph rebuilt in fp32, so its fixed coefficients are fp32, with the fp32 reference weights.
With all keys kept no calibrated quantity (tail mass, centroids) is active, so the rebuild is the compiled model.
fp32 is IEEE fp32 throughout: cuDNN convolutions do not use TF32 (PyTorch's default for fp32 convolutions), and cuDNN is
deterministic without autotuning. DeiT-Tiny and Pythia-70M numbers repeat exactly; the MambaIRv2-Light kernels are not
deterministic, and two runs differ by up to 0.01 dB (2026-10-17, in bf16 and fp32 alike).

Usage: python scripts/dtype_check.py OUT_JSON
"""

from __future__ import annotations

import json
import sys
import warnings
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from transformer_surgery.cli.surgery_config import SurgeryConfig
from transformer_surgery.internal.util import set_default_device, set_surgery_dtype
from transformer_surgery.models.adapters import (
    build_surgery_student,
    get_model_adapter,
    load_surgery_student_checkpoint,
)

CKPT = "artifacts/checkpoints"
# Network -> [(graph, surgery config, compiled checkpoint at k = N)].
NETWORKS: Dict[str, List[Tuple[str, str, str]]] = {
    "DeiT-Tiny / Pet": [
        ("extended", "configs/surgery/topk197_fast.json", f"{CKPT}/ts_surgery_topk197_fast.pt"),
        ("strict, GELU", "configs/surgery/topk197_strict_sr.json", f"{CKPT}/ts_surgery_topk197_strict_sr.pt"),
        ("strict, sigmoid GELU", "configs/surgery/topk197_strict_sr_gs.json", f"{CKPT}/ts_surgery_topk197_strict_sr_gs.pt"),
        ("Hadamard, GELU", "configs/surgery/topk197_hadamard_sr.json", f"{CKPT}/ts_surgery_topk197_hadamard_sr.pt"),
        (
            "Hadamard, sigmoid GELU",
            "configs/surgery/topk197_hadamard_sr_gs.json",
            f"{CKPT}/ts_surgery_topk197_hadamard_sr_gs.pt",
        ),
    ],
    "DeiT-Tiny / ImageNet-1k": [
        (
            "extended",
            "configs/surgery_imagenet/topk197_fast_imagenet.json",
            f"{CKPT}/ts_surgery_topk197_fast_imagenet.pt",
        ),
    ],
    "MambaIRv2-Light x2 / Set5": [
        ("extended", "configs/surgery/mambair_x2_topk256_fast.json", f"{CKPT}/ts_surgery_mambair_x2_topk256_fast.pt"),
    ],
    "MambaIRv2-Light x4 / Set5": [
        ("extended", "configs/surgery/mambair_x4_topk256_fast.json", f"{CKPT}/ts_surgery_mambair_x4_topk256_fast.pt"),
    ],
    "Pythia-70M / WikiText-2": [
        (
            "extended",
            "configs/surgery/pythia_70m_topk128_fast_tailmass_exact.json",
            f"{CKPT}/ts_surgery_pythia_70m_topk128_fast_tailmass_exact.pt",
        ),
    ],
}
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def evaluate(adapter: Any, model: nn.Module, val_loader: Any, cfg: SurgeryConfig) -> Dict[str, float]:
    if not next(model.parameters()).is_cuda:
        raise RuntimeError("dtype check runs on the GPU only")
    primary, loss, extra = adapter.evaluate(model, val_loader, cfg)
    return {"primary": float(primary), "loss": float(loss), **{k: float(v) for k, v in extra.items()}}


def main() -> None:
    out_path = sys.argv[1]
    warnings.simplefilter("ignore", FutureWarning)
    device = torch.device("cuda")
    set_default_device(device)
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    results: List[Dict[str, Any]] = []
    for network, graphs in NETWORKS.items():
        cfg0 = SurgeryConfig.load(graphs[0][1])
        adapter = get_model_adapter(cfg0.model_key)
        _, val_loader = adapter.build_loaders(cfg0)
        metric = adapter.primary_metric_name()
        refs: Dict[str, nn.Module] = {}
        for name, dtype in DTYPES.items():
            set_surgery_dtype(dtype)
            refs[name] = adapter.load_reference_checkpoint(adapter.reference_checkpoint_path(cfg0)).to(device, dtype)
            row = {"network": network, "metric": metric, "model": "reference", "variant": name}
            row.update(evaluate(adapter, refs[name], val_loader, cfg0))
            results.append(row)
            print(json.dumps(row), flush=True)
        for graph, cfg_path, ckpt in graphs:
            cfg = SurgeryConfig.load(cfg_path)
            extra = torch.load(ckpt, map_location=device, weights_only=False)["extra"]
            model, _ = load_surgery_student_checkpoint(ckpt, cfg, surgery_dtype=torch.bfloat16)
            variants = [("checkpoint bf16", model, {})]
            for name, dtype in DTYPES.items():
                rebuilt, _ = build_surgery_student(extra, cfg, dtype)
                copied = adapter.copy_reference_weights(rebuilt, refs[name])
                missing = [n for n, _ in rebuilt.named_parameters() if n not in copied]
                variants.append((f"rebuild {name}", rebuilt, {"copied": len(copied), "params_not_copied": missing}))
            for variant, m, info in variants:
                set_surgery_dtype(next(m.parameters()).dtype)
                row = {"network": network, "metric": metric, "model": graph, "variant": variant, **info}
                row.update(evaluate(adapter, m, val_loader, cfg))
                results.append(row)
                print(json.dumps(row), flush=True)
                del m
            del variants, model
            torch.cuda.empty_cache()
        del refs
        torch.cuda.empty_cache()
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
