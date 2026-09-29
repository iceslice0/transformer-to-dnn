"""CLI config for ``ts-surgery`` / ``python -m transformer_surgery.cli.surgery``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from transformer_surgery.cli.common import load_dataclass_from_json, parse_cli_config
from transformer_surgery.internal.util import DEFAULT_MODEL_KEY


@dataclass
class SurgeryConfig:
    """Surgery transform + calibration run; JSON + CLI via :meth:`load`."""

    model_key: str = DEFAULT_MODEL_KEY
    data_dir: str = "./data"
    batch_size: int = 128
    workers: int = 2
    # Super-resolution (MambaIR) fields; ignored by classification adapters.
    scale: int = 2
    train_hr: Optional[str] = None
    train_lr: Optional[str] = None
    val_hr: Optional[str] = None
    val_lr: Optional[str] = None
    patch_size: int = 64
    train_repeat: int = 1
    max_train_items: Optional[int] = None
    max_val_items: Optional[int] = None
    sr_download: bool = True
    # Causal LM (Pythia) fields; ignored by vision / SR adapters.
    context_length: int = 128
    hf_model_id: str = "EleutherAI/pythia-70m"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    max_train_tokens: Optional[int] = None
    max_val_tokens: Optional[int] = None
    top_k: int = 32
    eps: float = 1e-5
    gibbs_tail_prob_eps: float = 1e-5
    gibbs_tail_calibration_batches: Optional[int] = 10
    disable_calib_gibbs_tail_prob: bool = False
    use_exact_tail_mass: bool = False
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
    "gibbs_tail_prob_eps": "Initial omitted-tail probability mass; calibration overwrites per-block values.",
    "gibbs_tail_calibration_batches": "Training batches for Gibbs tail mass; JSON null uses the full train loader.",
    "disable_calib_gibbs_tail_prob": "Disable calibration/application of per-block Gibbs omitted-tail probability.",
    "use_exact_tail_mass": (
        "Compute omitted-tail mass at runtime from dense softmax via the centroid partition trick "
        "(N * mean(exp)) instead of using the calibrated/fixed gibbs_tail_prob_eps scalar."
    ),
    "disable_layernorm_replacement": "Debug: use nn.LayerNorm instead of RewrittenLayerNorm.",
    "disable_attention_surgery": "Debug: use dense scaled-dot attention instead of attention surgery modules.",
    "disable_softmax_replacement": "Debug: when attention surgery is on, use full softmax @ V.",
    "allow_matmul": "Debug: use matmul fast paths where supported.",
    "surgery_dtype": "torch dtype name for surgery compute, e.g. bfloat16, float16, or float32.",
    "log_dir": "Directory for model structure dumps.",
    "context_length": "Fixed token window length for causal-LM surgery (Pythia).",
    "hf_model_id": "Hugging Face model id or local path for the causal-LM reference.",
    "dataset_name": "HF datasets name for LM evaluation (default wikitext).",
    "dataset_config": "HF datasets config name (default wikitext-2-raw-v1).",
    "max_train_tokens": "Optional cap on concatenated train tokens before windowing.",
    "max_val_tokens": "Optional cap on concatenated validation tokens before windowing.",
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
