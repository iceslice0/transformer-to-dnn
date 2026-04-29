"""CLI config for ``ts-run-surgery`` / ``python -m transformer_surgery.cli.run_surgery``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from transformer_surgery.pipeline import DEFAULT_MODEL_KEY, load_dataclass_from_json, parse_cli_config


@dataclass
class SurgeryRunConfig:
    """Surgery transform + calibration run; JSON + CLI via :meth:`load`."""

    model_key: str = DEFAULT_MODEL_KEY
    data_dir: str = "./data"
    batch_size: int = 128
    workers: int = 2
    top_k: int = 32
    eps: float = 1e-5
    reference_checkpoint: Optional[str] = None
    device: str = "cuda"
    surgery_dtype: str = "bfloat16"
    disable_layernorm_replacement: bool = False
    disable_attention_surgery: bool = False
    disable_softmax_replacement: bool = False
    allow_matmul: bool = False
    randaugment: bool = True
    ra_magnitude: int = 9
    random_erasing_prob: float = 0.0
    meta_json: str = "artifacts/metadata/surgery_meta.json"
    pre_ft_checkpoint: str = "artifacts/checkpoints/surgery_pre_ft.pt"
    log_dir: str = "artifacts/logs"
    config_json_path: Optional[str] = None

    @classmethod
    def load(cls, json_path: str, overrides: Optional[Dict[str, Any]] = None) -> "SurgeryRunConfig":
        return load_dataclass_from_json(cls, json_path, overrides)


FIELD_HELP_SURGERY_RUN: Dict[str, str] = {
    "model_key": "Model adapter key. Default: deit_tiny_pet.",
    "reference_checkpoint": "Reference/teacher checkpoint path.",
    "disable_layernorm_replacement": "Debug: use nn.LayerNorm instead of RewrittenLayerNorm.",
    "disable_attention_surgery": "Debug: use dense scaled-dot attention instead of attention surgery modules.",
    "disable_softmax_replacement": "Debug: when attention surgery is on, use full softmax @ V.",
    "allow_matmul": "Debug: use matmul fast paths where supported.",
    "surgery_dtype": "torch dtype name for surgery compute, e.g. bfloat16, float16, or float32.",
    "log_dir": "Directory for model structure dumps.",
}

CLI_SURGERY_RUN_DESCRIPTION = "Model-adapter surgery: reference checkpoint -> surgery student + metadata"
CLI_SURGERY_RUN_CONFIG_DEFAULT = "configs/surgery/topk64_fast.json"


def parse_surgery_run_config(argv: Optional[Sequence[str]] = None) -> SurgeryRunConfig:
    return parse_cli_config(
        SurgeryRunConfig,
        description=CLI_SURGERY_RUN_DESCRIPTION,
        config_default=CLI_SURGERY_RUN_CONFIG_DEFAULT,
        field_help=FIELD_HELP_SURGERY_RUN,
        argv=argv,
    )
