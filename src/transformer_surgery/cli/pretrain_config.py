"""CLI config for ``ts-pretrain-pet`` / ``python -m transformer_surgery.cli.pretrain_pet``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from transformer_surgery.cli.common import RunConfig, parse_cli_config


@dataclass
class PretrainPetConfig(RunConfig):
    """Pet classifier head training on timm DeiT-Tiny."""

    output: str = "artifacts/checkpoints/ts_pretrain_pet_deit_tiny.pt"
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    label_smoothing: float = 0.05
    seed: int = 42
    max_train_batches: Optional[int] = None
    cosine_eta_min: float = 1e-6
    gap_th: Optional[float] = None


CLI_PRETRAIN_CONFIG_DEFAULT = "configs/pretrain/pet_deit_tiny.json"


def parse_pretrain_pet_config(argv: Optional[Sequence[str]] = None) -> PretrainPetConfig:
    return parse_cli_config(
        PretrainPetConfig,
        description="Pet head training on timm DeiT-Tiny",
        config_default=CLI_PRETRAIN_CONFIG_DEFAULT,
        field_help={
            "gap_th": (
                "During training: after each epoch, if val acc is more than this far below the best-so-far "
                "val acc, restore model weights to that best checkpoint only (optimizer state unchanged)."
            ),
        },
        argv=argv,
    )
