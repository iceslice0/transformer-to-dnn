"""CLI config for ``ts-distill`` / ``python -m transformer_surgery.cli.distill``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from transformer_surgery.cli.common import AdapterRunConfig, parse_cli_config


@dataclass
class JeffreysDistillConfig(AdapterRunConfig):
    """Mixed hard-label + teacher-matching (CE + Jeffreys for classification) distillation."""

    epochs: int = 2
    lr: float = 5e-4
    weight_decay: float = 0.05
    warmup_epochs: Optional[int] = None
    grad_clip: float = 1.0
    cosine_eta_min: float = 0.0
    temperature: float = 1.0
    distill_weight: float = 0.5
    max_train_batches: Optional[int] = None
    keep_best: bool = True
    base_seed: int = 42
    num_trainings: int = 1
    pre_checkpoint: str = "artifacts/checkpoints/ts_surgery_topk64_fast.pt"
    output: str = "artifacts/checkpoints/ts_distill_64_fast_jeffreys.pt"
    train_progress_interval: int = 10
    top_k: Optional[int] = None
    eps: Optional[float] = None


FIELD_HELP_JEFFREYS: Dict[str, str] = {
    "warmup_epochs": "Linear warmup epochs; JSON null uses min(5, epochs).",
    "distill_weight": "Mixing weight for teacher matching vs hard-label CE.",
    "base_seed": "Base random seed; run i uses base_seed + i.",
    "num_trainings": "Number of independent distillation trainings to run; only the best checkpoint is saved.",
    "top_k": "Override of the checkpoint's top_k.",
    "eps": "Override of the checkpoint's LayerNorm epsilon.",
}

CLI_JEFFREYS_CONFIG_DEFAULT = "configs/distill/64_fast_jeffreys.json"


def parse_distill_config(argv: Optional[Sequence[str]] = None) -> JeffreysDistillConfig:
    return parse_cli_config(
        JeffreysDistillConfig,
        description="Model-adapter CE + Jeffreys teacher matching",
        config_default=CLI_JEFFREYS_CONFIG_DEFAULT,
        field_help=FIELD_HELP_JEFFREYS,
        argv=argv,
    )
