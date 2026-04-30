"""CLI config for ``ts-surgery`` / ``python -m transformer_surgery.cli.surgery``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from transformer_surgery.cli.common import load_dataclass_from_json, parse_cli_config
from transformer_surgery.util import DEFAULT_MODEL_KEY


@dataclass
class SurgeryConfig:
    """Surgery transform + calibration run; JSON + CLI via :meth:`load`."""

    model_key: str = DEFAULT_MODEL_KEY
    data_dir: str = "./data"
    batch_size: int = 128
    workers: int = 2
    top_k: int = 32
    eps: float = 1e-5
    gibbs_tail_use_prob_eps: bool = False
    gibbs_tail_prob_eps: float = 1e-5
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
    pre_ft_checkpoint: str = "artifacts/checkpoints/ts_surgery_topk64_fast.pt"
    log_dir: str = "artifacts/logs"
    config_json_path: Optional[str] = None

    @classmethod
    def load(cls, json_path: str, overrides: Optional[Dict[str, Any]] = None) -> "SurgeryConfig":
        return load_dataclass_from_json(cls, json_path, overrides)


FIELD_HELP_SURGERY: Dict[str, str] = {
    "model_key": "Model adapter key. Default: deit_tiny_pet.",
    "reference_checkpoint": "Reference/teacher checkpoint path.",
    "gibbs_tail_use_prob_eps": "Use fixed omitted-tail probability mass instead of the k-th-logit estimate.",
    "gibbs_tail_prob_eps": "Fixed omitted-tail probability mass used when gibbs_tail_use_prob_eps is true.",
    "disable_layernorm_replacement": "Debug: use nn.LayerNorm instead of RewrittenLayerNorm.",
    "disable_attention_surgery": "Debug: use dense scaled-dot attention instead of attention surgery modules.",
    "disable_softmax_replacement": "Debug: when attention surgery is on, use full softmax @ V.",
    "allow_matmul": "Debug: use matmul fast paths where supported.",
    "surgery_dtype": "torch dtype name for surgery compute, e.g. bfloat16, float16, or float32.",
    "log_dir": "Directory for model structure dumps.",
}

CLI_SURGERY_DESCRIPTION = "Model-adapter surgery: reference checkpoint -> surgery student + metadata"
CLI_SURGERY_CONFIG_DEFAULT = "configs/surgery/topk64_fast.json"


def parse_surgery_config(argv: Optional[Sequence[str]] = None) -> SurgeryConfig:
    return parse_cli_config(
        SurgeryConfig,
        description=CLI_SURGERY_DESCRIPTION,
        config_default=CLI_SURGERY_CONFIG_DEFAULT,
        field_help=FIELD_HELP_SURGERY,
        argv=argv,
    )
