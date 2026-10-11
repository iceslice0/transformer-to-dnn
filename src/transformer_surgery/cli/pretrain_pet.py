#!/usr/bin/env python3
"""Train only the Pet classifier head on timm DeiT-Tiny (ImageNet backbone frozen). Saves weights for surgery."""

from __future__ import annotations

import os
from dataclasses import asdict

import torch
import torch.nn as nn

from transformer_surgery.cli.common import apply_device_from_config
from transformer_surgery.cli.pretrain_config import parse_pretrain_pet_config
from transformer_surgery.internal.reporting import describe_device, traceable_artifact_path
from transformer_surgery.internal.util import accuracy_and_loss, set_seed
from transformer_surgery.models.pet import (
    PET_NUM_CLASSES,
    build_pet_loaders,
    create_timm_deit_tiny,
    load_timm_deit_checkpoint,
    train_timm_deit_on_pet,
)


def main() -> None:
    cfg = parse_pretrain_pet_config()
    set_seed(cfg.seed)
    device = apply_device_from_config(cfg)
    out_abs = traceable_artifact_path(cfg.output, cfg, "ts-pretrain-pet", extension=".pt")
    cfg.output = out_abs
    train_loader, val_loader = build_pet_loaders(cfg)

    # Resume from an earlier run's output when present, else start from the ImageNet-pretrained backbone.
    resume_acc = None
    if os.path.isfile(out_abs):
        resume_acc = float(torch.load(out_abs, map_location="cpu", weights_only=False)["val_acc"])
        print(f"init checkpoint={out_abs}\nresume best_acc (val_acc in checkpoint)={resume_acc:.4f}", flush=True)
        model = load_timm_deit_checkpoint(out_abs, PET_NUM_CLASSES)
        model.train()
    else:
        print("init=imagenet_pretrained", flush=True)
        model = create_timm_deit_tiny(PET_NUM_CLASSES, pretrained=True).to(device)

    print(f"device={describe_device(device)} config={asdict(cfg)}", flush=True)
    acc, loss_v, best_ep, gap_reverts = train_timm_deit_on_pet(
        model, train_loader, val_loader, cfg, resume_val_acc=resume_acc
    )

    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    train_config = {
        **asdict(cfg),
        "best_val_epoch": best_ep,
        "best_acc_reference": resume_acc,
        "gap_revert_count": gap_reverts,
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": PET_NUM_CLASSES,
            "dataset": "Oxford-IIIT Pet",
            "val_acc": acc,
            "val_loss": loss_v,
            "train_config": train_config,
        },
        out_abs,
    )
    print(f"saved {out_abs} val_acc={acc:.4f} val_loss={loss_v:.4f}", flush=True)

    acc2, _ = accuracy_and_loss(load_timm_deit_checkpoint(out_abs, PET_NUM_CLASSES), val_loader, nn.CrossEntropyLoss())
    if abs(acc2 - acc) > 1e-4:
        print(f"WARNING reload acc mismatch {acc:.6f} vs {acc2:.6f}", flush=True)
    else:
        print(f"reload_ok val_acc={acc2:.4f}", flush=True)


if __name__ == "__main__":
    main()
