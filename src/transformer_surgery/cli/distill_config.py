"""CLI config for ``ts-distill`` / ``python -m transformer_surgery.cli.distill``."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

from transformer_surgery.cli.surgery_config import FIELD_HELP_SURGERY_RUN
from transformer_surgery.pipeline import DEFAULT_MODEL_KEY, load_dataclass_from_json


@dataclass
class JeffreysDistillConfig:
    """Mixed CE + Jeffreys distillation; JSON + CLI via :meth:`load`."""

    model_key: str = DEFAULT_MODEL_KEY
    data_dir: str = "./data"
    epochs: int = 2
    batch_size: int = 128
    workers: int = 2
    lr: float = 5e-4
    weight_decay: float = 0.05
    warmup_epochs: Optional[int] = None
    grad_clip: float = 1.0
    cosine_eta_min: float = 0.0
    temperature: float = 1.0
    distill_weight: float = 0.5
    max_train_batches: Optional[int] = None
    keep_best: bool = True
    reference_checkpoint: Optional[str] = None
    pre_checkpoint: str = "artifacts/checkpoints/surgery_pre_ft.pt"
    output: str = "artifacts/checkpoints/surgery_post_ft.pt"
    meta_json: Optional[str] = "artifacts/metadata/surgery_meta.json"
    randaugment: bool = True
    ra_magnitude: int = 9
    random_erasing_prob: float = 0.0
    device: str = "cuda"
    surgery_dtype: str = "bfloat16"
    train_progress_interval: int = 10
    val_progress_batches: int = 20
    top_k: Optional[int] = None
    eps: Optional[float] = None
    config_json_path: Optional[str] = None

    @classmethod
    def load(cls, json_path: str, overrides: Optional[Dict[str, Any]] = None) -> "JeffreysDistillConfig":
        cfg = load_dataclass_from_json(cls, json_path, overrides)
        mj = cfg.meta_json
        if mj is not None and isinstance(mj, str) and not mj.strip():
            cfg = replace(cfg, meta_json=None)
        return cfg


FIELD_HELP_JEFFREYS: Dict[str, str] = {
    "model_key": FIELD_HELP_SURGERY_RUN["model_key"],
    "reference_checkpoint": FIELD_HELP_SURGERY_RUN["reference_checkpoint"],
    "meta_json": "Path to surgery_meta.json; empty string disables merge.",
    "distill_weight": "Mixing weight for teacher matching vs hard-label CE.",
    "surgery_dtype": FIELD_HELP_SURGERY_RUN["surgery_dtype"],
}

CLI_JEFFREYS_DESCRIPTION = "Model-adapter CE + Jeffreys teacher matching"
CLI_JEFFREYS_CONFIG_DEFAULT = "configs/distill/jeffreys_default.json"
CLI_JEFFREYS_CONFIG_HELP = "JSON hyperparameters (merged with JeffreysDistillConfig defaults)."
