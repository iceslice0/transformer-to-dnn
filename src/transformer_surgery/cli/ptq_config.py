"""CLI config for ``ts-ptq`` / ``python -m transformer_surgery.cli.ptq``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, get_args

from transformer_surgery.cli.common import AdapterRunConfig, parse_cli_config

# Product sites of the strict graph that Brent's operand scaling folds: the value mix and LayerNorm u * inv_std.
BrentSite = Literal["mix", "layernorm"]


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
    compiled_affine_input_bits: Optional[int] = None
    matmul_activation_bits: Optional[int] = None
    per_output_channel: bool = True
    quantize_output: bool = False
    output_bits: Optional[int] = None
    brent_scaling: bool = False
    brent_sites: List[BrentSite] = field(default_factory=lambda: list(get_args(BrentSite)))
    calibration_seed: Optional[int] = 0
    dequant_var_eps: float = 1e-8
    top_k: Optional[int] = None
    eps: Optional[float] = None
    log_dir: str = "artifacts/logs"


# A producer's output is its consumer's input: each boundary is quantized once, at the selected consumer, with that
# consumer's width and scale, so a consumer's input quantizer is also its producer's output quantizer. The graph between
# two quantizers (one wrapper with its output map, then any unary, routing, AffineSum or AffineMean nodes) is a unit
# with integer inputs and integer outputs. AffineSum and AffineMean add integer codes with fixed scales (or their
# squares), so their results are integers with a fixed scale; they keep their full integer range and are not rounded
# to a specific bit width. Every *_bits field below is the width of input codes; no field sets the width of an
# accumulator or of a sum. Calibration and the quantized model run in fp32 with autocast off, so the integer codes and
# their sums are exact (checked at run time); only the float model is also validated in its surgery dtype.
FIELD_HELP_PTQ: Dict[str, str] = {
    "fp_checkpoint": "Float checkpoint to wrap with PTQ modules.",
    "output": "PTQ checkpoint output path.",
    "include_names": "Only wrap nodes whose module name contains one of these substrings.",
    "exclude_names": "Skip nodes whose module name contains one of these substrings.",
    "activation_bits": "Input-code width of the Linear/Conv2d nodes of the source model.",
    "compiled_affine_input_bits": (
        "Input-code width of the fixed affine nodes (AffineScale, AffineScaleBias, AffineFixedMix, AffineContract "
        "and its subclasses): the nodes that compilation adds when it splits LayerNorm, softmax and attention, and the "
        "residual and position-embedding additions. In the strict graph their inputs include squares and sums of "
        "squares, so the strict configs use 12-bit input codes there. Falls back to activation_bits."
    ),
    "matmul_activation_bits": "Input-code width of AffineMatMul / AffineHadamard nodes. Falls back to activation_bits.",
    "per_output_channel": (
        "Linear/Conv2d weights: per-output-channel symmetric scales (true) or one scale for the whole tensor (false, "
        "often destroys accuracy). Fixed affine nodes always use one weight scale; AffineCombine keeps its "
        "coefficients exact, as the per-input multipliers of an integer Add."
    ),
    "quantize_output": (
        "Route the model output through an identity node (OutputIdentity) added at PTQ time, so its input quantizer, "
        "the output quantizer of the last node, quantizes the model output with output_bits codes and one scale. "
        "The surgery checkpoint is unchanged."
    ),
    "output_bits": "Input-code width of the output identity (quantize_output). Falls back to activation_bits.",
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
