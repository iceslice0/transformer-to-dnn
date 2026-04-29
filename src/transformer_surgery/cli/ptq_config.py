"""CLI config for ``ts-ptq`` / ``python -m transformer_surgery.cli.ptq``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from transformer_surgery.cli.common import load_dataclass_from_json, parse_cli_config
from transformer_surgery.util import DEFAULT_MODEL_KEY


@dataclass
class PTQSurgeryConfig:
    model_key: str = DEFAULT_MODEL_KEY
    data_dir: str = "./data"
    fp_checkpoint: str = "artifacts/checkpoints/ts_distill_64_fast_jeffreys.pt"
    output: str = "artifacts/checkpoints/ts_ptq_64_fast_jeffreys_8bit_wrapped.pt"
    batch_size: int = 32
    workers: int = 2
    randaugment: bool = True
    ra_magnitude: int = 9
    random_erasing_prob: float = 0.0
    device: str = "cuda"
    surgery_dtype: str = "bfloat16"
    calibration_batches: int = 4
    calibration_examples_per_node: int = 4
    wrap_linear_conv: bool = True
    wrap_affine: bool = False
    wrap_matmul: bool = False
    include_names: List[str] = field(default_factory=list)
    exclude_names: List[str] = field(default_factory=list)
    weight_bits: int = 8
    activation_bits: int = 8
    affine_activation_bits: Optional[int] = None
    matmul_activation_bits: Optional[int] = None
    # For ``Linear``/``Conv2d``: **True** = per-output-channel symmetric weight scales (axis 0);
    # **False** = one global scale over the whole weight tensor (often destroys accuracy). Ignored for
    # affine/unary/coeff (always one global weight scale) and for ``MatMul`` (no weights). Output
    # ``out_scale``/``out_bias`` for Linear/Conv do not depend on this flag (analytical + residual bias).
    per_output_channel: bool = True
    top_k: Optional[int] = None
    eps: Optional[float] = None
    log_dir: str = "artifacts/logs"
    config_json_path: Optional[str] = None

    @classmethod
    def load(cls, json_path: str, overrides: Optional[Dict[str, Any]] = None) -> "PTQSurgeryConfig":
        return load_dataclass_from_json(cls, json_path, overrides)


FIELD_HELP_PTQ: Dict[str, str] = {
    "model_key": "Model adapter key. Default: deit_tiny_pet.",
    "fp_checkpoint": "Float checkpoint to wrap with PTQ modules.",
    "output": "PTQ checkpoint output path.",
    "surgery_dtype": "torch dtype name for model load/eval, e.g. bfloat16, float16, or float32.",
    "include_names": "Only wrap nodes whose module name contains one of these substrings.",
    "exclude_names": "Skip nodes whose module name contains one of these substrings.",
    "per_output_channel": "Use per-output-channel weight scales for Linear/Conv2d.",
}

CLI_PTQ_DESCRIPTION = "Standalone PTQ for surgery checkpoints"
CLI_PTQ_CONFIG_DEFAULT = "configs/ptq/64_fast_jeffreys_8bit.json"
CLI_PTQ_CONFIG_HELP = "JSON config for PTQ wrapping and validation."


def parse_ptq_config(argv: Optional[Sequence[str]] = None) -> PTQSurgeryConfig:
    return parse_cli_config(
        PTQSurgeryConfig,
        description=CLI_PTQ_DESCRIPTION,
        config_default=CLI_PTQ_CONFIG_DEFAULT,
        config_help=CLI_PTQ_CONFIG_HELP,
        field_help=FIELD_HELP_PTQ,
        argv=argv,
    )
