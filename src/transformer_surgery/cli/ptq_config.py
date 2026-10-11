"""CLI config for ``ts-ptq`` / ``python -m transformer_surgery.cli.ptq``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from transformer_surgery.cli.common import AdapterRunConfig, parse_cli_config


@dataclass
class PTQSurgeryConfig(AdapterRunConfig):
    """Post-training quantization of a surgery checkpoint."""

    batch_size: int = 32
    fp_checkpoint: str = "artifacts/checkpoints/ts_distill_64_fast_jeffreys.pt"
    output: str = "artifacts/checkpoints/ts_ptq_64_fast_jeffreys_8bit.pt"
    wrap_linear_conv: bool = True
    wrap_affine: bool = False
    wrap_matmul: bool = False
    include_names: List[str] = field(default_factory=list)
    exclude_names: List[str] = field(default_factory=list)
    weight_bits: int = 8
    activation_bits: int = 8
    compiled_accumulator_bits: Optional[int] = None
    matmul_activation_bits: Optional[int] = None
    per_output_channel: bool = True
    brent_scaling: bool = False
    brent_sites: List[str] = field(default_factory=lambda: ["mix", "layernorm"])
    calibration_seed: Optional[int] = 0
    dequant_var_eps: float = 1e-8
    top_k: Optional[int] = None
    eps: Optional[float] = None
    log_dir: str = "artifacts/logs"


# A producer's output is its consumer's input: each boundary is quantized once, at the selected consumer, with that
# consumer's width and scale. The wrapper quantizes, accumulates and dequantizes; the next selected consumer applies
# the next grid, after any unary or routing nodes.
FIELD_HELP_PTQ: Dict[str, str] = {
    "fp_checkpoint": "Float checkpoint to wrap with PTQ modules.",
    "output": "PTQ checkpoint output path.",
    "include_names": "Only wrap nodes whose module name contains one of these substrings.",
    "exclude_names": "Skip nodes whose module name contains one of these substrings.",
    "activation_bits": "Input-code width of the Linear/Conv2d nodes of the source model.",
    "compiled_accumulator_bits": (
        "Input-code width of the fixed affine nodes that compilation introduces when it splits LayerNorm, softmax "
        "and attention (AffineScale, AffineScaleBias, AffineFixedMix, AffineContract), the accumulators of the "
        "compiled subgraphs. The strict graph holds squares and sums of squares there, so the paper's strict "
        "configs use 12 bits. Falls back to activation_bits."
    ),
    "matmul_activation_bits": "Input-code width of AffineMatMul / AffineHadamard nodes. Falls back to activation_bits.",
    "per_output_channel": (
        "Linear/Conv2d weights: per-output-channel symmetric scales (true) or one scale for the whole tensor (false, "
        "often destroys accuracy). Fixed affine nodes always use one weight scale; AffineCombine keeps its "
        "coefficients exact, as the per-input multipliers of an integer Add."
    ),
    "brent_scaling": (
        "Fold Brent's operand scaling into the norm-expansion products (strict Mix and LayerNorm): a -> lam a, "
        "b -> b / lam with lam^2 = max|b| / max|a| from a first range pass, rounded to a power of two so the folded "
        "coefficients stay exact in bf16; the ranges are then measured again."
    ),
    "brent_sites": "Product sites folded by brent_scaling: any of 'mix' (value mix) and 'layernorm' (u * inv_std).",
    "calibration_seed": (
        "Seed set before the loaders are built, so PTQ calibration repeats; JSON null leaves it unseeded."
    ),
    "dequant_var_eps": (
        "OLS affine-dequant variance threshold for Affine/MatMul kinds. Channels with var(acc) below this collapse "
        "to slope 0 (output equals the channel mean). Also clamps the OLS denominator."
    ),
    "top_k": "Override of the checkpoint's top_k.",
    "eps": "Override of the checkpoint's LayerNorm epsilon.",
}


def parse_ptq_config(argv: Optional[Sequence[str]] = None) -> PTQSurgeryConfig:
    return parse_cli_config(
        PTQSurgeryConfig,
        description="Standalone PTQ for surgery checkpoints",
        config_default="configs/ptq/64_fast_jeffreys_8bit.json",
        field_help=FIELD_HELP_PTQ,
        argv=argv,
    )
