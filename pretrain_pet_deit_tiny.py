#!/usr/bin/env python3
"""Train only the Pet classifier head on timm DeiT-Tiny (ImageNet backbone frozen). Saves weights for surgery."""

from __future__ import annotations

import copy
import os

import torch

from pet_reference_utils import (
    PET_NUM_CLASSES,
    accuracy_and_loss,
    apply_device_from_config,
    build_pet_loaders,
    create_deit_tiny_pet,
    describe_device,
    load_timm_deit_pet_checkpoint,
    parse_pretrain_pet_config,
    pretrain_train_config_record,
    set_seed,
    train_timm_deit_on_pet,
)

_PRETRAIN_RA_MAG = 9


def main() -> None:
    cfg = parse_pretrain_pet_config()

    set_seed(cfg.seed)
    device = apply_device_from_config(cfg)

    out_abs = os.path.abspath(cfg.output)
    train_loader, val_loader = build_pet_loaders(
        cfg.data_dir,
        cfg.batch_size,
        cfg.workers,
        randaugment=True,
        ra_magnitude=_PRETRAIN_RA_MAG,
        random_erasing_prob=0.0,
    )

    resume_path = out_abs if os.path.isfile(out_abs) else None
    best_acc_ref: float | None = None
    if resume_path is not None:
        print(f"init checkpoint={resume_path}", flush=True)
        try:
            payload_pre = torch.load(resume_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload_pre = torch.load(resume_path, map_location="cpu")
        if isinstance(payload_pre, dict) and payload_pre.get("val_acc") is not None:
            best_acc_ref = float(payload_pre["val_acc"])
            print(f"resume best_acc (val_acc in checkpoint)={best_acc_ref:.4f}", flush=True)
        model = load_timm_deit_pet_checkpoint(resume_path)
        model.train()
    else:
        print("init=imagenet_pretrained", flush=True)
        model = create_deit_tiny_pet(pretrained=True).to(device)

    snapshot_before_train = copy.deepcopy(model.state_dict())

    print(
        f"device={describe_device(device)} "
        f"epochs={cfg.epochs} lr={cfg.lr} wd={cfg.weight_decay} "
        f"batch={cfg.batch_size} warmup_ep={cfg.warmup_epochs} grad_clip={cfg.grad_clip} "
        f"label_smoothing={cfg.label_smoothing} seed={cfg.seed} "
        f"cosine_eta_min={cfg.cosine_eta_min} gap_th={cfg.gap_th}",
        flush=True,
    )
    if cfg.config_json_path:
        print(f"config_json={cfg.config_json_path}", flush=True)
    if cfg.max_train_batches is not None:
        print(f"max_train_batches={cfg.max_train_batches}", flush=True)

    acc, loss_v, best_ep = train_timm_deit_on_pet(
        model,
        train_loader,
        val_loader,
        cfg.epochs,
        cfg.lr,
        max_train_batches=cfg.max_train_batches,
        weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs,
        grad_clip=cfg.grad_clip,
        label_smoothing=cfg.label_smoothing,
        cosine_eta_min=cfg.cosine_eta_min,
    )

    criterion = torch.nn.CrossEntropyLoss()
    gap_th = cfg.gap_th
    reverted = False
    if gap_th is not None:
        if resume_path is None:
            print(
                "note: gap_th set but no resume checkpoint — need --output file to exist to define best_acc; skip revert",
                flush=True,
            )
        elif best_acc_ref is None:
            print(
                "note: gap_th set but checkpoint has no val_acc — skip revert",
                flush=True,
            )
        elif float(acc) < float(best_acc_ref) - float(gap_th):
            trained_acc = float(acc)
            floor = float(best_acc_ref) - float(gap_th)
            model.load_state_dict(snapshot_before_train)
            acc, loss_v = accuracy_and_loss(model, val_loader, criterion)
            best_ep = None
            reverted = True
            print(
                f"reverted: trained val_acc={trained_acc:.4f} < best_acc - gap_th = {floor:.4f} "
                f"(best_acc={best_acc_ref:.4f} gap_th={float(gap_th):.4f}) — restored pre-run weights "
                f"(now val_acc={acc:.4f} val_loss={loss_v:.4f})",
                flush=True,
            )

    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    out_cfg = pretrain_train_config_record(
        cfg,
        output_abs=out_abs,
        best_val_epoch=best_ep,
        best_acc_reference=best_acc_ref,
        reverted_below_best_acc_minus_gap=reverted,
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
    print(f"saved {out_abs} val_acc={acc:.4f} val_loss={loss_v:.4f}", flush=True)

    reloaded = load_timm_deit_pet_checkpoint(out_abs)
    acc2, loss2 = accuracy_and_loss(reloaded, val_loader, criterion)
    if abs(acc2 - acc) > 1e-4:
        print(f"WARNING reload acc mismatch {acc:.6f} vs {acc2:.6f}", flush=True)
    else:
        print(f"reload_ok val_acc={acc2:.4f}", flush=True)


if __name__ == "__main__":
    main()
