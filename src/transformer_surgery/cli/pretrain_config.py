"""CLI config for ``ts-pretrain-pet`` / ``python -m transformer_surgery.cli.pretrain_pet``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from transformer_surgery.cli.common import load_dataclass_from_json, parse_cli_config


@dataclass
class PretrainPetConfig:
    """Pet classifier head training on timm DeiT-Tiny; JSON + CLI via :meth:`load`."""

    data_dir: str = "./data"
    output: str = "artifacts/checkpoints/ts_pretrain_pet_deit_tiny.pt"
    epochs: int = 50
    batch_size: int = 128
    workers: int = 2
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    label_smoothing: float = 0.05
    seed: int = 42
    max_train_batches: Optional[int] = None
    device: str = "cuda"
    cosine_eta_min: float = 1e-6
    gap_th: Optional[float] = None
    randaugment: bool = True
    ra_magnitude: int = 9
    random_erasing_prob: float = 0.0
    config_json_path: Optional[str] = None

    @classmethod
    def load(cls, json_path: str, overrides: Optional[Dict[str, Any]] = None) -> "PretrainPetConfig":
        return load_dataclass_from_json(cls, json_path, overrides)


FIELD_HELP_PRETRAIN: Dict[str, str] = {
    "gap_th": (
        "During training: after each epoch, if val acc is more than this far below the best-so-far "
        "val acc, restore model weights to that best checkpoint only (optimizer state unchanged)."
    ),
}

CLI_PRETRAIN_DESCRIPTION = "Pet head training on timm DeiT-Tiny"
CLI_PRETRAIN_CONFIG_DEFAULT = "configs/pretrain/pet_deit_tiny.json"


def pretrain_train_config_record(
    cfg: PretrainPetConfig,
    *,
    output_abs: str,
    best_val_epoch: Optional[int] = None,
    best_acc_reference: Optional[float] = None,
    gap_revert_count: int = 0,
) -> Dict[str, Any]:
    """JSON-serializable ``train_config`` block for the Pet head checkpoint."""
    rec: Dict[str, Any] = {
        "data_dir": cfg.data_dir,
        "output": output_abs,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "warmup_epochs": cfg.warmup_epochs,
        "grad_clip": cfg.grad_clip,
        "label_smoothing": cfg.label_smoothing,
        "seed": cfg.seed,
        "workers": cfg.workers,
        "device": cfg.device,
        "cosine_eta_min": cfg.cosine_eta_min,
    }
    if cfg.config_json_path:
        rec["config_json"] = cfg.config_json_path
    if cfg.max_train_batches is not None:
        rec["max_train_batches"] = cfg.max_train_batches
    if best_val_epoch is not None:
        rec["best_val_epoch"] = best_val_epoch
    if cfg.gap_th is not None:
        rec["gap_th"] = float(cfg.gap_th)
    if best_acc_reference is not None:
        rec["best_acc_reference"] = best_acc_reference
    if gap_revert_count > 0:
        rec["gap_revert_count"] = gap_revert_count
    return rec


def parse_pretrain_pet_config(argv: Optional[Sequence[str]] = None) -> PretrainPetConfig:
    return parse_cli_config(
        PretrainPetConfig,
        description=CLI_PRETRAIN_DESCRIPTION,
        config_default=CLI_PRETRAIN_CONFIG_DEFAULT,
        field_help=FIELD_HELP_PRETRAIN,
        argv=argv,
    )
