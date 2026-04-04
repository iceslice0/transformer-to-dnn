#!/usr/bin/env python3
"""
DeiT-Tiny surgery pipeline: load a Pet-pretrained timm checkpoint, transform, calibrate, save pre-finetune weights.

This script does **not** run distillation or any finetuning: no teacher–student loss, no optimizer, no ``backward``.
It only builds the surgery student from timm weights, runs **eval** accuracy/loss for reporting, and saves
``surgery_pre_ft.pt`` / ``surgery_meta.json``. (The output name means “before optional Jeffreys distillation,”
not that this step finetunes.)

Run `pretrain_pet_deit_tiny.py` first for the Pet timm checkpoint. Defaults live in ``conf/surgery_run_config.json``;
CLI overrides optional. Bisect with ``disable_layernorm_replacement``, ``disable_attention_surgery``,
``disable_softmax_replacement``, ``allow_matmul``. Then run `finetune_surgery_deit_tiny.py` (``conf/surgery_distill_config.json``) **separately**
if you want Jeffreys distillation.
"""

from __future__ import annotations

import json
import os
from typing import Dict

import torch
import torch.nn as nn

from deit_tiny_surgery_model import DeiTTinySurgeryModel, freeze_eps_parameters
from pet_reference_utils import (
    PET_NUM_CLASSES,
    SurgeryRunConfig,
    accuracy_and_loss,
    apply_device_from_config,
    build_pet_loaders,
    describe_device,
    get_device,
    load_timm_deit_pet_checkpoint,
    parse_surgery_run_config,
    pre_ft_checkpoint_extra,
    resolve_path_under_script,
    save_deit_checkpoint,
    surgery_meta_for_pre_ft,
)
from surgery_utils import (
    RewrittenLayerNormAbsSign,
    build_default_pwl_knots,
    copy_ln_params_to_rewritten,
    jeffreys_distance_sparse_teacher,
    jeffreys_naive_topk,
)


def _write_model_structure_txt(path: str, model: nn.Module, title: str) -> None:
    """Write ``str(model)``, parameter counts, and ``named_modules`` listing to a UTF-8 text file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines = [
        title,
        "=" * min(80, max(len(title), 40)),
        f"class: {type(model).__name__}",
        f"parameters: total={n_all:,} trainable={n_train:,}",
        "",
        str(model),
        "",
        "--- named_modules (name: class) ---",
        "",
    ]
    for name, mod in model.named_modules():
        lines.append(f"{name if name else '<root>'}: {type(mod).__name__}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


@torch.no_grad()
def calibration_ln_and_softmax(
    ref: nn.Module,
    loader,
    cfg: SurgeryRunConfig,
) -> Dict[str, float]:
    """
    Read-only diagnostics for ``surgery_meta.json``: LN-rewrite MSE vs timm, and Jeffreys **metrics** comparing
    dense vs sparse top-k softmax (not distillation training—no student model, no gradients).
    """
    device = get_device()
    ref.eval()
    stats: Dict[str, float] = {}
    eps = float(cfg.eps)
    top_k = int(cfg.top_k)
    use_cuda = device.type == "cuda"
    batch, _ = next(iter(loader))
    batch = batch.to(device, non_blocking=use_cuda)

    b = batch.shape[0]
    x = ref.patch_embed(batch)
    x = torch.cat((ref.cls_token.expand(b, -1, -1), x), dim=1) + ref.pos_embed
    x = ref.pos_drop(x)
    h0 = x
    y_ref0 = ref.blocks[0].norm1(h0)
    rw0 = RewrittenLayerNormAbsSign(ref.embed_dim, eps=eps, allow_matmul=cfg.allow_matmul).to(device)
    copy_ln_params_to_rewritten(rw0, ref.blocks[0].norm1)
    y_rw0 = rw0(h0)
    stats["ln_rewrite_mse_layer0_minibatch"] = float(torch.mean((y_ref0 - y_rw0).pow(2)).cpu())

    h = x
    mse_acc = 0.0
    n_ln = 0
    for blk in ref.blocks:
        n1 = blk.norm1(h)
        rw = RewrittenLayerNormAbsSign(ref.embed_dim, eps=eps, allow_matmul=cfg.allow_matmul).to(device)
        copy_ln_params_to_rewritten(rw, blk.norm1)
        mse_acc += torch.mean((rw(h) - n1).pow(2)).item()
        n_ln += 1
        h = h + blk.attn(n1)
        n2 = blk.norm2(h)
        rw2 = RewrittenLayerNormAbsSign(ref.embed_dim, eps=eps, allow_matmul=cfg.allow_matmul).to(device)
        copy_ln_params_to_rewritten(rw2, blk.norm2)
        mse_acc += torch.mean((rw2(h) - n2).pow(2)).item()
        n_ln += 1
        h = h + blk.mlp(n2)
    h_pre = h
    h_out = ref.norm(h_pre)
    rwf = RewrittenLayerNormAbsSign(ref.embed_dim, eps=eps, allow_matmul=cfg.allow_matmul).to(device)
    copy_ln_params_to_rewritten(rwf, ref.norm)
    mse_acc += torch.mean((rwf(h_pre) - h_out).pow(2)).item()
    n_ln += 1
    stats["ln_rewrite_mse_all_norms_mean"] = mse_acc / max(n_ln, 1)

    attn = ref.blocks[0].attn
    qkv = attn.qkv(h0).reshape(b, h0.shape[1], 3, attn.num_heads, ref.embed_dim // attn.num_heads).permute(
        2, 0, 3, 1, 4
    )
    q, k = qkv[0], qkv[1]
    scores = (q @ k.transpose(-2, -1)) * float(attn.scale)
    flat = scores.reshape(-1, scores.shape[-1])
    R = min(flat.shape[0], 4096)
    teacher = flat[:R].clone()
    t = teacher - teacher.max(dim=-1, keepdim=True).values
    nk = t.shape[-1]
    k = min(top_k, nk)
    vals, idx = torch.topk(t, k=k, dim=-1, largest=True, sorted=True)
    j_gibbs = jeffreys_distance_sparse_teacher(teacher, vals, idx, nk, k).mean()
    j_naive = jeffreys_naive_topk(teacher, vals, idx, nk, k).mean()
    stats["jeffreys_gibbs_mean_cached"] = float(j_gibbs.cpu())
    stats["jeffreys_naive_mean_cached"] = float(j_naive.cpu())
    stats["jeffreys_improvement_naive_minus_gibbs_cached"] = float((j_naive - j_gibbs).cpu())

    teacher2 = torch.randn(4096, nk, device=device)
    t2 = teacher2 - teacher2.max(dim=-1, keepdim=True).values
    vals2, idx2 = torch.topk(t2, k=k, dim=-1, largest=True, sorted=True)
    j_gibbs2 = jeffreys_distance_sparse_teacher(teacher2, vals2, idx2, nk, k).mean()
    j_naive2 = jeffreys_naive_topk(teacher2, vals2, idx2, nk, k).mean()
    stats["jeffreys_gibbs_mean_synthetic"] = float(j_gibbs2.cpu())
    stats["jeffreys_naive_mean_synthetic"] = float(j_naive2.cpu())
    stats["jeffreys_improvement_naive_minus_gibbs_synthetic"] = float((j_naive2 - j_gibbs2).cpu())

    return stats


def build_module_mapping(cfg: SurgeryRunConfig) -> Dict[str, str]:
    if cfg.disable_layernorm_replacement:
        ln = "nn.LayerNorm"
    elif cfg.allow_matmul:
        ln = "RewrittenLayerNormAbsSign(rsqrt·mul)"
    else:
        ln = "RewrittenLayerNormAbsSign(log/exp)"
    if cfg.disable_attention_surgery:
        attn = "SurgeryAttention(vanilla scaled QK^T softmax @ V)"
    else:
        dot = "PairwiseDotBySquare(QK^T matmul)" if cfg.allow_matmul else "PairwiseDotBySquare(square identity)"
        if cfg.disable_softmax_replacement:
            attn = f"SurgeryAttention({dot}+full_softmax+dense@V)"
        else:
            mix = (
                "SparseWeightedSumBySquare(elementwise p*v)"
                if cfg.allow_matmul
                else "SparseWeightedSumBySquare(square identity)"
            )
            attn = f"SurgeryAttention({dot}+GibbsTopKSoftmax+{mix})"
    m: Dict[str, str] = {}
    for i in range(12):
        m[f"blocks.{i}.norm1"] = ln
        m[f"blocks.{i}.attn"] = attn
        m[f"blocks.{i}.norm2"] = ln
        m[f"blocks.{i}.mlp.act"] = "GELUUnaryPWL"
    m["fc_norm"] = ln
    return m


def main() -> None:
    cfg = parse_surgery_run_config()

    device = apply_device_from_config(cfg)

    pet_ref_path = os.path.abspath(cfg.pet_ref_checkpoint)
    if not os.path.isfile(pet_ref_path):
        raise SystemExit(
            f"Missing pet reference checkpoint: {pet_ref_path}\n"
            "Run first: python pretrain_pet_deit_tiny.py --output ./pet_timm_deit_tiny.pt"
        )

    print(f"Using device: {describe_device(device)}", flush=True)
    if cfg.config_json_path:
        print(f"config_json={cfg.config_json_path}", flush=True)

    _, val_loader = build_pet_loaders(cfg)

    print(f"Loading timm reference from {pet_ref_path} ...", flush=True)
    ref = load_timm_deit_pet_checkpoint(pet_ref_path)
    log_dir = os.path.abspath("logs")
    _write_model_structure_txt(
        os.path.join(log_dir, "model_before_surgery.txt"),
        ref,
        "Timm DeiT-Tiny (Pet reference, before surgery transform)",
    )
    print(f"wrote {os.path.join(log_dir, 'model_before_surgery.txt')}", flush=True)

    criterion = nn.CrossEntropyLoss()
    ref.eval()
    ref_acc, ref_loss = accuracy_and_loss(ref, val_loader, criterion)
    print(f"Reference timm (Pet) val acc={ref_acc:.4f} loss={ref_loss:.4f}")

    cal = calibration_ln_and_softmax(ref, val_loader, cfg)
    print("Calibration:", json.dumps(cal, indent=2))

    print("Building surgery model...", flush=True)
    print(
        f"  disable_layernorm_replacement={cfg.disable_layernorm_replacement}  "
        f"disable_attention_surgery={cfg.disable_attention_surgery}  "
        f"disable_softmax_replacement={cfg.disable_softmax_replacement}  allow_matmul={cfg.allow_matmul}",
        flush=True,
    )
    model = DeiTTinySurgeryModel.from_surgery_run_config(cfg, num_classes=PET_NUM_CLASSES).to(device)
    mapping = model.load_from_timm(ref)
    freeze_eps_parameters(model)
    print(f"Loaded {len(mapping)} tensors from reference checkpoint.")

    _write_model_structure_txt(
        os.path.join(log_dir, "model_after_surgery.txt"),
        model,
        "DeiTTinySurgeryModel (after surgery, pre-finetune checkpoint)",
    )
    print(f"wrote {os.path.join(log_dir, 'model_after_surgery.txt')}", flush=True)

    pre_acc, pre_loss = accuracy_and_loss(model, val_loader, criterion)
    print(f"Post-transform val acc={pre_acc:.4f} loss={pre_loss:.4f}")

    _, _, pwl_meta = build_default_pwl_knots()
    mod_map = build_module_mapping(cfg)
    meta = surgery_meta_for_pre_ft(
        cfg,
        calibration=cal,
        pet_ref_checkpoint_abs=pet_ref_path,
        pwl_knees=pwl_meta,
        module_mapping=mod_map,
    )
    meta.calibration["ref_val_acc"] = float(ref_acc)
    meta.calibration["ref_val_loss"] = float(ref_loss)
    meta.calibration["student_pre_ft_val_acc"] = float(pre_acc)
    meta.calibration["student_pre_ft_mean_ce"] = float(pre_loss)
    meta_path = resolve_path_under_script(cfg.meta_json, __file__)
    meta.to_json(meta_path)

    pre_path = resolve_path_under_script(cfg.pre_ft_checkpoint, __file__)
    save_deit_checkpoint(
        pre_path,
        model,
        extra=pre_ft_checkpoint_extra(cfg, mapping=mapping),
    )
    print(f"Wrote {pre_path} and {meta_path}")
    print("Next: python finetune_surgery_deit_tiny.py", flush=True)


if __name__ == "__main__":
    main()
