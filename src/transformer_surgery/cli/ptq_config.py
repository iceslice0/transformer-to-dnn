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
    fp_checkpoint: str = "artifacts/checkpoints/surgery_post_ft.pt"
    output: str = "artifacts/checkpoints/surgery_ptq.pt"
    meta_json: str = "artifacts/metadata/surgery_ptq_meta.json"
    batch_size: int = 32
    workers: int = 2
    randaugment: bool = True
    ra_magnitude: int = 9
    random_erasing: float = 0.0
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
    debug_node_stats: bool = True
    quiet: bool = False
    log_dir: str = "artifacts/logs"
    config_json_path: Optional[str] = None
    # Print effective quant policy + per-node table; set True or pass --quant-policy-debug.
    quant_policy_debug: bool = False

    @classmethod
    def load(cls, json_path: str) -> "PTQSurgeryConfig":
        cfg = load_dataclass_from_json(cls, json_path, overrides=None)
        if not cfg.meta_json:
            cfg.meta_json = "artifacts/metadata/surgery_ptq_meta.json"
        return cfg


def parse_ptq_config(argv: Optional[Sequence[str]] = None) -> PTQSurgeryConfig:
    parser = argparse.ArgumentParser(description="Standalone PTQ for surgery checkpoints")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ptq/full_8bit.json",
        help="JSON config for PTQ wrapping and validation.",
    )
    parser.add_argument(
        "--quant-policy-debug",
        action="store_true",
        help="Print per-node weight scale mode (per_output_channel applies to Linear/Conv2d weights only).",
    )
    args = parser.parse_args(argv)
    cfg = PTQSurgeryConfig.load(args.config)
    if args.quant_policy_debug:
        cfg.quant_policy_debug = True
    return cfg
