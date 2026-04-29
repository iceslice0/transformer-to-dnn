"""CLI config for ``ts-ptq`` / ``python -m transformer_surgery.cli.ptq``."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from transformer_surgery.pipeline import DEFAULT_MODEL_KEY, load_dataclass_from_json


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
    def load(cls, json_path: str) -> "PTQSurgeryConfig":
        return load_dataclass_from_json(cls, json_path, overrides=None)


def parse_ptq_config(argv: Optional[Sequence[str]] = None) -> PTQSurgeryConfig:
    parser = argparse.ArgumentParser(description="Standalone PTQ for surgery checkpoints")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ptq/64_fast_jeffreys_8bit.json",
        help="JSON config for PTQ wrapping and validation.",
    )
    args = parser.parse_args(argv)
    return PTQSurgeryConfig.load(args.config)
