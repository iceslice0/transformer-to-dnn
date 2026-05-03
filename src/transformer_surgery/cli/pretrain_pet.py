#!/usr/bin/env python3
"""Train only the Pet classifier head on timm DeiT-Tiny (ImageNet backbone frozen). Saves weights for surgery."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

from transformer_surgery.cli.pretrain_config import (
    PretrainPetConfig,
    parse_pretrain_pet_config,
    pretrain_train_config_record,
)
from transformer_surgery.models.pet import (
    PET_NUM_CLASSES,
    build_pet_loaders,
    create_deit_tiny_pet,
    load_timm_deit_pet_checkpoint,
    train_timm_deit_on_pet,
)
from transformer_surgery.cli.common import apply_device_from_config
from transformer_surgery.util import (
    accuracy_and_loss,
    describe_device,
    set_seed,
    traceable_artifact_path,
)


def log_pretrain_run_banner(cfg: PretrainPetConfig, device: torch.device) -> None:
    print(
        f"device={describe_device(device)} "
        f"epochs={cfg.epochs} lr={cfg.lr} wd={cfg.weight_decay} "
        f"batch={cfg.batch_size} warmup_ep={cfg.warmup_epochs} grad_clip={cfg.grad_clip} "
        f"label_smoothing={cfg.label_smoothing} seed={cfg.seed} "
        f"cosine_eta_min={cfg.cosine_eta_min} gap_th={cfg.gap_th}",
        flush=True,
    )


def log_pretrain_optional_lines(cfg: PretrainPetConfig) -> None:
    if cfg.config_json_path:
        print(f"config_json={cfg.config_json_path}", flush=True)
    if cfg.max_train_batches is not None:
        print(f"max_train_batches={cfg.max_train_batches}", flush=True)


def log_init_from_checkpoint(path: str) -> None:
    print(f"init checkpoint={path}", flush=True)


def log_resume_best_acc(ref: float) -> None:
    print(f"resume best_acc (val_acc in checkpoint)={ref:.4f}", flush=True)


def log_init_imagenet_pretrained() -> None:
    print("init=imagenet_pretrained", flush=True)


def load_pretrain_model(
    cfg: PretrainPetConfig,
    out_abs: str,
    device: torch.device,
) -> tuple[nn.Module, Optional[str], Optional[float]]:
    """
    Load from ``out_abs`` if present, else ImageNet-pretrained backbone. Returns
    ``(model, resume_path, best_acc_from_ckpt_or_None)`` for metadata in the saved checkpoint.
    """
    resume_path = out_abs if os.path.isfile(out_abs) else None
    best_acc_ref: Optional[float] = None
    if resume_path is not None:
        log_init_from_checkpoint(resume_path)
        payload_pre = torch.load(resume_path, map_location="cpu", weights_only=False)
        best_acc_ref = float(payload_pre["val_acc"])
        log_resume_best_acc(best_acc_ref)
        model = load_timm_deit_pet_checkpoint(resume_path)
        model.train()
    else:
        log_init_imagenet_pretrained()
        model = create_deit_tiny_pet(pretrained=True).to(device)

    return model, resume_path, best_acc_ref


def log_saved_checkpoint(path: str, acc: float, loss_v: float) -> None:
    print(f"saved {path} val_acc={acc:.4f} val_loss={loss_v:.4f}", flush=True)


def log_reload_verify(acc: float, acc2: float) -> None:
    if abs(acc2 - acc) > 1e-4:
        print(f"WARNING reload acc mismatch {acc:.6f} vs {acc2:.6f}", flush=True)
    else:
        print(f"reload_ok val_acc={acc2:.4f}", flush=True)


def main() -> None:
    cfg = parse_pretrain_pet_config()

    set_seed(cfg.seed)
    device = apply_device_from_config(cfg)

    out_abs = traceable_artifact_path(cfg.output, cfg, "ts-pretrain-pet", extension=".pt")
    cfg.output = out_abs
    train_loader, val_loader = build_pet_loaders(cfg)

    model, _, best_acc_ref = load_pretrain_model(cfg, out_abs, device)

    log_pretrain_run_banner(cfg, device)
    log_pretrain_optional_lines(cfg)

    acc, loss_v, best_ep, gap_reverts = train_timm_deit_on_pet(
        model,
        train_loader,
        val_loader,
        cfg,
        resume_val_acc=best_acc_ref,
    )

    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    out_cfg = pretrain_train_config_record(
        cfg,
        output_abs=out_abs,
        best_val_epoch=best_ep,
        best_acc_reference=best_acc_ref,
        gap_revert_count=gap_reverts,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": PET_NUM_CLASSES,
            "dataset": "Oxford-IIIT Pet",
            "val_acc": acc,
            "val_loss": loss_v,
            "train_config": out_cfg,
        },
        out_abs,
    )
    log_saved_checkpoint(out_abs, acc, loss_v)

    criterion = nn.CrossEntropyLoss()
    reloaded = load_timm_deit_pet_checkpoint(out_abs)
    acc2, loss2 = accuracy_and_loss(reloaded, val_loader, criterion)
    log_reload_verify(acc, acc2)


if __name__ == "__main__":
    main()
