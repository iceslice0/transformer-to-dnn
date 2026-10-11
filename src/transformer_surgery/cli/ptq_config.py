"""CLI config for ``ts-ptq`` / ``python -m transformer_surgery.cli.ptq``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from transformer_surgery.cli.common import load_dataclass_from_json, parse_cli_config
from transformer_surgery.internal.util import DEFAULT_MODEL_KEY


@dataclass
class PTQSurgeryConfig:
    model_key: str = DEFAULT_MODEL_KEY
    data_dir: str = "./data"
    fp_checkpoint: str = "artifacts/checkpoints/ts_distill_64_fast_jeffreys.pt"
    output: str = "artifacts/checkpoints/ts_ptq_64_fast_jeffreys_8bit.pt"
    batch_size: int = 32
    workers: int = 2
    randaugment: bool = True
    ra_magnitude: int = 9
    random_erasing_prob: float = 0.0
    device: str = "cuda"
    wrap_linear_conv: bool = True
    wrap_affine: bool = False
    wrap_matmul: bool = False
    include_names: List[str] = field(default_factory=list)
    exclude_names: List[str] = field(default_factory=list)
    weight_bits: int = 8
    # The paper describes node input/output precision with a single bit width.
    # A producer's output is its consumer's input: quantize that boundary once,
    # at the selected consumer, using the consumer's own width and scale.
    # The wrapper quantizes, accumulates, and dequantizes; the next selected
    # consumer applies the next boundary's grid, after any unary/routing nodes.
    # The input-code width is selected independently for each node group:
    # - activation_bits: Linear / Conv2d inherited from the source model.
    # - compiled_accumulator_bits: the fixed affine nodes that compilation introduces when it splits the
    #   Transformer-specific layers (LayerNorm, softmax, attention): AffineScale / AffineScaleBias /
    #   AffineFixedMix / AffineContract, i.e. the accumulators of the compiled subgraphs. In the strict graph they hold
    #   squares and sums of squares, so the paper's strict configs use 12 for their wider range.
    #   Applies only to selected fixed-affine nodes; falls back to activation_bits when unset.
    #   Does not change weight_bits or the Linear/Conv and MatMul/Hadamard input widths.
    # - matmul_activation_bits: AffineMatMul / AffineHadamard (falls back to activation_bits).
    activation_bits: int = 8
    compiled_accumulator_bits: Optional[int] = None
    matmul_activation_bits: Optional[int] = None
    # Deprecated name of compiled_accumulator_bits, still accepted from older configs.
    affine_activation_bits: Optional[int] = None
    # For ``Linear``/``Conv2d``: **True** = per-output-channel symmetric weight scales (axis 0);
    # **False** = one global scale over the whole weight tensor (often destroys accuracy). Ignored for
    # affine/coeff (always one global weight scale; ``AffineCombine`` keeps its coefficients exact, as the per-input
    # multipliers of an integer Add) and for ``AffineMatMul``/``AffineHadamard`` (no weights). Output
    # ``out_scale``/``out_bias`` for Linear/Conv do not depend on this flag (analytical + residual bias).
    per_output_channel: bool = True
    # Brent's operand scaling of the norm-expansion products (strict Mix and LayerNorm): a -> lam a, b -> b / lam with
    # lam^2 = max|b| / max|a| from the first calibration range pass, rounded to a power of two so that the folded
    # coefficients stay exact in bf16; folded into the AffineCombine coefficients, then the ranges are measured again.
    brent_scaling: bool = False
    # Product sites that brent_scaling folds: "mix" (value mix) and "layernorm" (u * inv_std).
    brent_sites: List[str] = field(default_factory=lambda: ["mix", "layernorm"])
    # Seed of the global RNGs, set before the loaders are built, so the calibration batches and their augmentation
    # repeat run to run. None: unseeded (PTQ runs before 2026-10-11).
    calibration_seed: Optional[int] = 0
    # Variance/eps threshold used by the OLS affine-dequant fit (Affine and MatMul kinds).
    # When per-channel ``var(acc)`` falls below this, that channel's slope collapses to 0
    # (output ≈ channel mean). Also used as the stability ``clamp_min`` for the OLS denominator.
    # Lower → tighter fit but more numerical noise on near-constant channels; higher → more
    # aggressive collapse to mean. Default 1e-8 is safe for fp32 calibration tensors.
    dequant_var_eps: float = 1e-8
    top_k: Optional[int] = None
    eps: Optional[float] = None
    log_dir: str = "artifacts/logs"
    config_json_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.affine_activation_bits is not None:
            if self.compiled_accumulator_bits is not None and self.compiled_accumulator_bits != self.affine_activation_bits:
                raise ValueError("compiled_accumulator_bits and its deprecated alias affine_activation_bits disagree")
            self.compiled_accumulator_bits = self.affine_activation_bits
            self.affine_activation_bits = None

    @classmethod
    def load(cls, json_path: str, overrides: Optional[Dict[str, Any]] = None) -> "PTQSurgeryConfig":
        return load_dataclass_from_json(cls, json_path, overrides)


FIELD_HELP_PTQ: Dict[str, str] = {
    "model_key": "Model adapter key. Default: deit_tiny_pet.",
    "fp_checkpoint": "Float checkpoint to wrap with PTQ modules.",
    "output": "PTQ checkpoint output path.",
    "include_names": "Only wrap nodes whose module name contains one of these substrings.",
    "exclude_names": "Skip nodes whose module name contains one of these substrings.",
    "per_output_channel": "Use per-output-channel weight scales for Linear/Conv2d.",
    "brent_scaling": "Fold Brent's power-of-two operand scaling into the strict Mix and LayerNorm products before PTQ.",
    "brent_sites": "Product sites folded by brent_scaling: any of 'mix' and 'layernorm'.",
    "calibration_seed": "Seed set before the loaders are built, so PTQ calibration repeats; JSON null leaves it unseeded.",
    "compiled_accumulator_bits": (
        "Input-code width of selected compiled fixed-affine nodes (paper strict configs: 12 bits). "
        "Falls back to activation_bits. Weights and Linear/Conv and MatMul/Hadamard input widths have separate "
        "settings. Each boundary is quantized once at the selected consumer's input, using that consumer's "
        "width and scale."
    ),
    "dequant_var_eps": (
        "OLS affine-dequant variance threshold for Affine/MatMul kinds. Channels with "
        "var(acc) below this collapse to slope=0 (output equals the channel mean). "
        "Also clamps the OLS denominator. Default 1e-8."
    ),
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
