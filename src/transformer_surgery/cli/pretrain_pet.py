#!/usr/bin/env python3
"""Train only the Pet classifier head on timm DeiT-Tiny (ImageNet backbone frozen). Saves weights for surgery.

The saved epoch has the highest training-split accuracy; the Pet test split is evaluated once, on the saved weights.
"""

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
    build_pet_selection_loader,
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
    train_loader, test_loader = build_pet_loaders(cfg)
    select_loader = build_pet_selection_loader(cfg)

    # Always from the ImageNet-pretrained backbone: the checkpoint is selected on the training split only, and the
    # test split is evaluated once, on the selected weights.
    print("init=imagenet_pretrained", flush=True)
    model = create_timm_deit_tiny(PET_NUM_CLASSES, pretrained=True).to(device)

    print(f"device={describe_device(device)} config={asdict(cfg)}", flush=True)
    acc, loss_v, selected_ep, select_loss, select_acc = train_timm_deit_on_pet(
        model, train_loader, select_loader, test_loader, cfg
    )

    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    train_config = {
        **asdict(cfg),
        "init": "imagenet_pretrained",
        "selection": "highest accuracy on the trainval split with eval transforms (earlier epoch on ties)",
        "selected_epoch": selected_ep,
        "selection_train_loss": select_loss,
        "selection_train_acc": select_acc,
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
    print(f"saved {out_abs} test acc={acc:.4f} loss={loss_v:.4f} (selected epoch {selected_ep})", flush=True)

    acc2, _ = accuracy_and_loss(load_timm_deit_checkpoint(out_abs, PET_NUM_CLASSES), test_loader, nn.CrossEntropyLoss())
    if abs(acc2 - acc) > 1e-4:
        print(f"WARNING reload acc mismatch {acc:.6f} vs {acc2:.6f}", flush=True)
    else:
        print(f"reload_ok test acc={acc2:.4f}", flush=True)


if __name__ == "__main__":
    main()
