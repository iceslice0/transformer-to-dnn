"""Float64 end-to-end check of exact compilation: compiled DeiT-Tiny vs its reference on Pet validation images.

Surgery checkpoints store weights in the surgery dtype (bf16), so the reference is compared with its weights rounded
to bf16 the same way; otherwise the weight rounding (about 0.07 in the logits) hides the compilation error. The check
sets the compatibility softmax-normalizer epsilon to zero, as in the reported runs; LayerNorm keeps its reference
epsilon.

Example:
    python scripts/check_exact_fp64.py \\
        --checkpoints artifacts/checkpoints/ts_surgery_topk197_fast.pt \\
                      artifacts/checkpoints/ts_surgery_topk197_strict_sr.pt \\
        --batches 2
"""

from __future__ import annotations

import argparse
import os
import warnings

import torch

from transformer_surgery.cli.ptq_config import PTQSurgeryConfig
from transformer_surgery.internal.util import set_default_device, set_surgery_dtype
from transformer_surgery.models.adapters import get_model_adapter, load_surgery_student_checkpoint
from transformer_surgery.ops import GibbsTopKSoftmax


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--reference", default="artifacts/checkpoints/ts_pretrain_pet_deit_tiny.pt")
    parser.add_argument("--batches", type=int, default=2, help="Validation batches of 8 images.")
    args = parser.parse_args()
    warnings.simplefilter("ignore", FutureWarning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_default_device(device)
    cfg = PTQSurgeryConfig()
    cfg.top_k = None
    cfg.eps = None
    cfg.batch_size = 8
    cfg.workers = 0
    adapter = get_model_adapter("deit_tiny_pet")
    ref = adapter.load_reference_checkpoint(args.reference).to(device, torch.float64).eval()
    ref_bf16 = adapter.load_reference_checkpoint(args.reference).to(device, torch.float64).eval()
    with torch.no_grad():
        for p in ref_bf16.parameters():
            p.copy_(p.to(torch.bfloat16).to(torch.float64))
    _, val_loader = adapter.build_loaders(cfg)
    xs = []
    for i, (x, _y) in enumerate(val_loader):
        if i >= args.batches:
            break
        xs.append(x)
    x = torch.cat(xs).to(device, torch.float64)
    with torch.no_grad():
        r32, r = ref(x), ref_bf16(x)
    print(f"images {x.shape[0]}; reference fp32 vs bf16-rounded weights: max|dlogit| {(r32 - r).abs().max().item():.2e}")
    for path in args.checkpoints:
        set_surgery_dtype(torch.float64)
        model, extra = load_surgery_student_checkpoint(os.path.abspath(path), cfg, surgery_dtype=torch.float64)
        model = model.to(device, torch.float64).eval()
        for module in model.modules():
            if isinstance(module, GibbsTopKSoftmax):
                module.set_normalizer_epsilon(0.0)
        with torch.no_grad():
            y = model(x)
        name = os.path.basename(path)
        print(
            f"{name}: top_k={extra.get('top_k')} allow_matmul={extra.get('allow_matmul')} "
            f"strict_unary={extra.get('strict_unary')} LayerNorm eps={extra.get('eps_ln')} softmax eps=0; "
            f"max|dlogit| {(y - r).abs().max().item():.2e}; "
            f"argmax agreement {(y.argmax(-1) == r.argmax(-1)).float().mean().item():.3f}"
        )


if __name__ == "__main__":
    main()
