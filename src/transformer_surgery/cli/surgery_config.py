"""CLI config for ``ts-surgery`` / ``python -m transformer_surgery.cli.surgery``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Sequence

from transformer_surgery.cli.common import AdapterRunConfig, parse_cli_config

# Products of two activations in the DeiT-Tiny strict graph (see FIELD_HELP_SURGERY).
StrictProduct = Literal["norm", "polarization"]
# MLP activation: exact GELU or x*sigmoid(1.702x).
GeluMode = Literal["exact", "sigmoid"]
# Sparse support: top-k exact scores or the top-k keys of the nearest fixed query centroid.
SupportRouting = Literal["topk", "query_centroids"]


@dataclass
class SurgeryConfig(AdapterRunConfig):
    """Surgery transform + calibration run; also the architecture record of a surgery checkpoint."""

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
    gelu: GeluMode = "exact"
    hadamard: bool = False
    strict_product: StrictProduct = "norm"
    support_routing: SupportRouting = "topk"
    routing_clusters: int = 32
    routing_calibration_batches: Optional[int] = 20
    routing_max_samples: int = 65536
    routing_kmeans_iters: int = 30
    routing_seed: int = 0
    routing_eval_batches: Optional[int] = None
    calibration_seed: Optional[int] = 0
    surgery_dtype: str = "bfloat16"
    disable_layernorm_replacement: bool = False
    disable_attention_surgery: bool = False
    disable_softmax_replacement: bool = False
    allow_matmul: bool = False
    pre_ft_checkpoint: str = "artifacts/checkpoints/ts_surgery_topk64_fast.pt"
    log_dir: str = "artifacts/logs"


FIELD_HELP_SURGERY: Dict[str, str] = {
    "eps": "LayerNorm epsilon; softmax normalization always uses zero additive epsilon.",
    "gibbs_tail_prob_eps": "Initial omitted-tail probability mass; calibration overwrites per-block values.",
    "gibbs_tail_calibration_batches": "Training batches for Gibbs tail mass; JSON null uses the full train loader.",
    "disable_calib_gibbs_tail_prob": "Disable calibration/application of per-block Gibbs omitted-tail probability.",
    "use_exact_tail_mass": (
        "Compute omitted-tail mass at runtime from dense softmax via the centroid partition trick "
        "(N * mean(exp)) instead of using the calibrated/fixed gibbs_tail_prob_eps scalar."
    ),
    "gelu": "MLP activation: 'exact' GELU or 'sigmoid' x*sigmoid(1.702x) (a model change for NPUs without GELU).",
    "strict_product": (
        "DeiT-Tiny strict graph products of two activations: 'norm' (default; ((a+b)^2 - a^2 - b^2)/2 where a "
        "square is shared: QK^T, PV, LayerNorm and the exact tail term; polarization elsewhere) or 'polarization' "
        "((a+b)^2 - (a-b)^2)/4 everywhere (earlier runs)."
    ),
    "hadamard": (
        "DeiT-Tiny Hadamard graph: elementwise products instead of polarization, QK^T and PV as products plus "
        "sums, no matmul; needs allow_matmul=false."
    ),
    "support_routing": (
        "Sparse support selection: 'topk' (top-k exact scores) or 'query_centroids' "
        "(top-k keys of the nearest fixed query centroid; no activation-activation products)."
    ),
    "routing_clusters": "Number of k-means query centroids per head for support_routing=query_centroids.",
    "routing_calibration_batches": "Training batches sampled for the centroid fit; JSON null uses all.",
    "routing_max_samples": "Cap on query vectors per head used by k-means.",
    "routing_kmeans_iters": "Lloyd iterations after k-means++ initialization.",
    "routing_seed": "Seed for batch sampling, query subsampling and k-means++.",
    "routing_eval_batches": "Validation batches for support diagnostics (captured mass, recall); JSON null uses all.",
    "calibration_seed": (
        "Seed set before the loaders are built, so calibration batches and their augmentation repeat; "
        "JSON null leaves the run unseeded."
    ),
    "disable_layernorm_replacement": "Debug: use nn.LayerNorm instead of RewrittenLayerNorm.",
    "disable_attention_surgery": "Debug: use dense scaled-dot attention instead of attention surgery modules.",
    "disable_softmax_replacement": "Debug: when attention surgery is on, use full softmax @ V.",
    "allow_matmul": "Fast graph: matmul and Hadamard products of activations.",
    "surgery_dtype": "torch dtype name for surgery compute, e.g. bfloat16, float16, or float32.",
    "log_dir": "Directory for model structure dumps.",
    "context_length": "Fixed token window length for causal-LM surgery (Pythia).",
    "hf_model_id": "Hugging Face model id or local path for the causal-LM reference.",
    "dataset_name": "HF datasets name for LM evaluation (default wikitext).",
    "dataset_config": "HF datasets config name (default wikitext-2-raw-v1).",
    "max_train_tokens": "Optional cap on concatenated train tokens before windowing.",
    "max_val_tokens": "Optional cap on concatenated validation tokens before windowing.",
}

CLI_SURGERY_CONFIG_DEFAULT = "configs/surgery/topk64_fast.json"


def parse_surgery_config(argv: Optional[Sequence[str]] = None) -> SurgeryConfig:
    return parse_cli_config(
        SurgeryConfig,
        description="Model-adapter surgery: reference checkpoint -> surgery student + metadata",
        config_default=CLI_SURGERY_CONFIG_DEFAULT,
        field_help=FIELD_HELP_SURGERY,
        argv=argv,
    )
