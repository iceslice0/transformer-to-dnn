"""Pythia / GPT-NeoX surgery graph (causal LM)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from transformer_surgery.internal.calibration import copy_ln_params_to_rewritten
from transformer_surgery.internal.util import ensure_mapping, get_surgery_dtype
from transformer_surgery.ops import (
    AffineContract,
    GibbsTopKSoftmax,
    NLGELU,
    RewrittenLayerNorm,
    RoutingStack,
    SurgeryCausalAttention,
)


PYTHIA_70M_DEFAULTS: Dict[str, Any] = {
    "vocab_size": 50304,
    "hidden_size": 512,
    "num_hidden_layers": 6,
    "num_attention_heads": 8,
    "intermediate_size": 2048,
    "layer_norm_eps": 1e-5,
    "rotary_pct": 0.25,
    "rope_theta": 10000.0,
    "attention_bias": True,
    "use_parallel_residual": True,
    "hidden_dropout": 0.0,
    "attention_dropout": 0.0,
}


def arch_dict_from_hf_config(config: Any) -> Dict[str, Any]:
    """Extract the architecture fields needed to rebuild the surgery model offline."""
    rope = getattr(config, "rope_parameters", None) or {}
    if not isinstance(rope, dict):
        rope = dict(rope)
    partial = rope.get("partial_rotary_factor", getattr(config, "rotary_pct", 0.25))
    theta = rope.get("rope_theta", getattr(config, "rotary_emb_base", 10000.0))
    return {
        "vocab_size": int(config.vocab_size),
        "hidden_size": int(config.hidden_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "intermediate_size": int(config.intermediate_size),
        "layer_norm_eps": float(config.layer_norm_eps),
        "rotary_pct": float(partial),
        "rope_theta": float(theta),
        "attention_bias": bool(getattr(config, "attention_bias", True)),
        "use_parallel_residual": bool(getattr(config, "use_parallel_residual", True)),
        "hidden_dropout": float(getattr(config, "hidden_dropout", 0.0)),
        "attention_dropout": float(getattr(config, "attention_dropout", 0.0)),
    }


class SurgeryPythiaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.dense_h_to_4h = nn.Linear(hidden_size, intermediate_size)
        self.dense_4h_to_h = nn.Linear(intermediate_size, hidden_size)
        self.act = NLGELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_4h_to_h(self.act(self.dense_h_to_4h(x)))


class SurgeryPythiaBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        seq_len: int,
        top_k: int,
        rotary_ndims: int,
        rope_theta: float,
        layer_norm_eps: float,
        attention_bias: bool,
        use_parallel_residual: bool,
        use_surgery_layernorm: bool,
        use_attention_surgery: bool,
        use_surgery_softmax: bool,
        allow_matmul: bool,
        gibbs_tail_prob_eps: float,
        use_exact_tail_mass: bool,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_parallel_residual = bool(use_parallel_residual)
        if use_surgery_layernorm:
            self.input_layernorm = RewrittenLayerNorm(hidden_size, eps=layer_norm_eps, allow_matmul=allow_matmul)
            self.post_attention_layernorm = RewrittenLayerNorm(
                hidden_size, eps=layer_norm_eps, allow_matmul=allow_matmul
            )
        else:
            self.input_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention = SurgeryCausalAttention(
            hidden_size,
            num_heads,
            seq_len,
            top_k,
            rotary_ndims=rotary_ndims,
            rope_theta=rope_theta,
            attn_bias=attention_bias,
            attn_drop=attention_dropout,
            use_attention_surgery=use_attention_surgery,
            use_surgery_softmax=use_surgery_softmax,
            allow_matmul=allow_matmul,
            eps_ln=layer_norm_eps,
            gibbs_tail_prob_eps=gibbs_tail_prob_eps,
            use_exact_tail_mass=use_exact_tail_mass,
        )
        self.mlp = SurgeryPythiaMLP(hidden_size, intermediate_size)
        self.post_attention_dropout = nn.Dropout(hidden_dropout)
        self.post_mlp_dropout = nn.Dropout(hidden_dropout)
        if use_parallel_residual:
            self.residual_contract = AffineContract(
                "i,...i->...",
                torch.tensor([1.0, 1.0, 1.0], dtype=get_surgery_dtype()),
            )
        else:
            self.attn_residual_contract = AffineContract(
                "i,...i->...",
                torch.tensor([1.0, 1.0], dtype=get_surgery_dtype()),
            )
            self.mlp_residual_contract = AffineContract(
                "i,...i->...",
                torch.tensor([1.0, 1.0], dtype=get_surgery_dtype()),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.post_attention_dropout(self.attention(self.input_layernorm(x)))
        if self.use_parallel_residual:
            mlp_out = self.post_mlp_dropout(self.mlp(self.post_attention_layernorm(x)))
            return self.residual_contract(RoutingStack((x, attn_out, mlp_out), dim=-1))
        attn_res = self.attn_residual_contract(RoutingStack((x, attn_out), dim=-1))
        mlp_out = self.post_mlp_dropout(self.mlp(self.post_attention_layernorm(attn_res)))
        return self.mlp_residual_contract(RoutingStack((attn_res, mlp_out), dim=-1))


class PythiaSurgeryModel(nn.Module):
    """GPT-NeoX / Pythia compiled into the surgery op vocabulary."""

    def __init__(
        self,
        *,
        vocab_size: int = 50304,
        hidden_size: int = 512,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        seq_len: int = 128,
        top_k: int = 128,
        layer_norm_eps: float = 1e-5,
        rotary_pct: float = 0.25,
        rope_theta: float = 10000.0,
        attention_bias: bool = True,
        use_parallel_residual: bool = True,
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        gibbs_tail_prob_eps: float = 1e-5,
        use_surgery_layernorm: bool = True,
        use_attention_surgery: bool = True,
        use_surgery_softmax: bool = True,
        allow_matmul: bool = True,
        use_exact_tail_mass: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        head_dim = hidden_size // num_attention_heads
        rotary_ndims = int(head_dim * float(rotary_pct))
        if rotary_ndims % 2 != 0:
            rotary_ndims -= 1

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.intermediate_size = int(intermediate_size)
        self.seq_len = int(seq_len)
        self.top_k = int(top_k)
        self.layer_norm_eps = float(layer_norm_eps)
        self.rotary_pct = float(rotary_pct)
        self.rotary_ndims = int(rotary_ndims)
        self.rope_theta = float(rope_theta)
        self.attention_bias = bool(attention_bias)
        self.use_parallel_residual = bool(use_parallel_residual)
        self.gibbs_tail_prob_eps = float(gibbs_tail_prob_eps)
        self.use_surgery_layernorm = bool(use_surgery_layernorm)
        self.use_attention_surgery = bool(use_attention_surgery)
        self.use_surgery_softmax = bool(use_surgery_softmax)
        self.allow_matmul = bool(allow_matmul)
        self.use_exact_tail_mass = bool(use_exact_tail_mass)
        self.eps_ln = self.layer_norm_eps

        self.embed_in = nn.Embedding(self.vocab_size, self.hidden_size)
        self.emb_dropout = nn.Dropout(hidden_dropout)
        self.layers = nn.ModuleList(
            [
                SurgeryPythiaBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_attention_heads,
                    intermediate_size=self.intermediate_size,
                    seq_len=self.seq_len,
                    top_k=self.top_k,
                    rotary_ndims=self.rotary_ndims,
                    rope_theta=self.rope_theta,
                    layer_norm_eps=self.layer_norm_eps,
                    attention_bias=self.attention_bias,
                    use_parallel_residual=self.use_parallel_residual,
                    use_surgery_layernorm=self.use_surgery_layernorm,
                    use_attention_surgery=self.use_attention_surgery,
                    use_surgery_softmax=self.use_surgery_softmax,
                    allow_matmul=self.allow_matmul,
                    gibbs_tail_prob_eps=self.gibbs_tail_prob_eps,
                    use_exact_tail_mass=self.use_exact_tail_mass,
                    attention_dropout=attention_dropout,
                    hidden_dropout=hidden_dropout,
                )
                for _ in range(self.num_hidden_layers)
            ]
        )
        if self.use_surgery_layernorm:
            self.final_layer_norm = RewrittenLayerNorm(
                self.hidden_size, eps=self.layer_norm_eps, allow_matmul=self.allow_matmul
            )
        else:
            self.final_layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def arch_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "layer_norm_eps": self.layer_norm_eps,
            "rotary_pct": self.rotary_pct,
            "rope_theta": self.rope_theta,
            "attention_bias": self.attention_bias,
            "use_parallel_residual": self.use_parallel_residual,
            "seq_len": self.seq_len,
            "context_length": self.seq_len,
        }

    @classmethod
    def from_arch_and_surgery(
        cls,
        arch: Dict[str, Any],
        cfg: Any,
    ) -> "PythiaSurgeryModel":
        return cls(
            vocab_size=int(arch["vocab_size"]),
            hidden_size=int(arch["hidden_size"]),
            num_hidden_layers=int(arch["num_hidden_layers"]),
            num_attention_heads=int(arch["num_attention_heads"]),
            intermediate_size=int(arch["intermediate_size"]),
            seq_len=int(getattr(cfg, "context_length", arch.get("seq_len", arch.get("context_length", 128)))),
            top_k=int(cfg.top_k),
            layer_norm_eps=float(arch.get("layer_norm_eps", getattr(cfg, "eps", 1e-5))),
            rotary_pct=float(arch.get("rotary_pct", 0.25)),
            rope_theta=float(arch.get("rope_theta", 10000.0)),
            attention_bias=bool(arch.get("attention_bias", True)),
            use_parallel_residual=bool(arch.get("use_parallel_residual", True)),
            hidden_dropout=float(arch.get("hidden_dropout", 0.0)),
            attention_dropout=float(arch.get("attention_dropout", 0.0)),
            gibbs_tail_prob_eps=float(cfg.gibbs_tail_prob_eps),
            use_surgery_layernorm=not bool(cfg.disable_layernorm_replacement),
            use_attention_surgery=not bool(cfg.disable_attention_surgery),
            use_surgery_softmax=not bool(cfg.disable_softmax_replacement),
            allow_matmul=bool(cfg.allow_matmul),
            use_exact_tail_mass=bool(getattr(cfg, "use_exact_tail_mass", False)),
        )

    @classmethod
    def from_surgery_config(cls, cfg: Any, *, arch: Optional[Dict[str, Any]] = None) -> "PythiaSurgeryModel":
        base = dict(PYTHIA_70M_DEFAULTS)
        if arch:
            base.update(arch)
        # Prefer cfg.eps for LN floor when provided; keep architecture eps as default.
        if getattr(cfg, "eps", None) is not None:
            base["layer_norm_eps"] = float(cfg.eps)
        return cls.from_arch_and_surgery(base, cfg)

    @classmethod
    def from_pretrained_extra(cls, extra: Any, cfg: Any) -> "PythiaSurgeryModel":
        d = ensure_mapping(extra)
        arch = ensure_mapping(d.get("arch", d))
        # Rebuild surgery flags from checkpoint extra (offline restore).
        class _Cfg:
            pass

        ns = _Cfg()
        ns.top_k = int(d["top_k"])
        ns.gibbs_tail_prob_eps = float(d["gibbs_tail_prob_eps"])
        ns.disable_layernorm_replacement = bool(d["disable_layernorm_replacement"])
        ns.disable_attention_surgery = bool(d["disable_attention_surgery"])
        ns.disable_softmax_replacement = bool(d["disable_softmax_replacement"])
        ns.allow_matmul = bool(d["allow_matmul"])
        ns.use_exact_tail_mass = bool(d.get("use_exact_tail_mass", False))
        ns.context_length = int(d.get("context_length", d.get("seq_len", arch.get("seq_len", 128))))
        ns.eps = float(d.get("eps_ln", arch.get("layer_norm_eps", 1e-5)))
        if getattr(cfg, "top_k", None) is not None:
            ns.top_k = int(cfg.top_k)
        if getattr(cfg, "eps", None) is not None:
            ns.eps = float(cfg.eps)
        return cls.from_arch_and_surgery(arch, ns)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype not in (torch.long, torch.int, torch.int32, torch.int64):
            raise TypeError(f"PythiaSurgeryModel expects integer token ids, got dtype={input_ids.dtype}")
        hidden = self.emb_dropout(self.embed_in(input_ids))
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.final_layer_norm(hidden)
        return self.lm_head(hidden)

    def load_from_reference(self, reference: nn.Module) -> Dict[str, str]:
        """Copy weights from a Hugging Face ``GPTNeoXForCausalLM`` (or compatible module tree)."""
        gpt = getattr(reference, "gpt_neox", reference)
        mapping: Dict[str, str] = {}

        def _copy(dst: nn.Parameter, src: torch.Tensor, dst_name: str, src_name: str) -> None:
            if tuple(dst.shape) != tuple(src.shape):
                raise ValueError(f"shape mismatch {dst_name} {tuple(dst.shape)} vs {src_name} {tuple(src.shape)}")
            dst.data.copy_(src.detach())
            mapping[dst_name] = src_name

        with torch.no_grad():
            _copy(self.embed_in.weight, gpt.embed_in.weight, "embed_in.weight", "gpt_neox.embed_in.weight")
            lm_w = reference.lm_head.weight if hasattr(reference, "lm_head") else gpt.embed_out.weight
            _copy(self.lm_head.weight, lm_w, "lm_head.weight", "lm_head.weight")

            for i, layer in enumerate(self.layers):
                src = gpt.layers[i]
                prefix = f"layers.{i}"
                src_prefix = f"gpt_neox.layers.{i}"

                if isinstance(layer.input_layernorm, RewrittenLayerNorm):
                    copy_ln_params_to_rewritten(layer.input_layernorm, src.input_layernorm)
                    mapping[f"{prefix}.input_layernorm.affine"] = f"{src_prefix}.input_layernorm"
                else:
                    _copy(layer.input_layernorm.weight, src.input_layernorm.weight, f"{prefix}.input_layernorm.weight", f"{src_prefix}.input_layernorm.weight")
                    _copy(layer.input_layernorm.bias, src.input_layernorm.bias, f"{prefix}.input_layernorm.bias", f"{src_prefix}.input_layernorm.bias")

                if isinstance(layer.post_attention_layernorm, RewrittenLayerNorm):
                    copy_ln_params_to_rewritten(layer.post_attention_layernorm, src.post_attention_layernorm)
                    mapping[f"{prefix}.post_attention_layernorm.affine"] = f"{src_prefix}.post_attention_layernorm"
                else:
                    _copy(
                        layer.post_attention_layernorm.weight,
                        src.post_attention_layernorm.weight,
                        f"{prefix}.post_attention_layernorm.weight",
                        f"{src_prefix}.post_attention_layernorm.weight",
                    )
                    _copy(
                        layer.post_attention_layernorm.bias,
                        src.post_attention_layernorm.bias,
                        f"{prefix}.post_attention_layernorm.bias",
                        f"{src_prefix}.post_attention_layernorm.bias",
                    )

                attn_dst = layer.attention
                attn_src = src.attention
                _copy(
                    attn_dst.query_key_value.weight,
                    attn_src.query_key_value.weight,
                    f"{prefix}.attention.query_key_value.weight",
                    f"{src_prefix}.attention.query_key_value.weight",
                )
                if attn_dst.query_key_value.bias is not None:
                    _copy(
                        attn_dst.query_key_value.bias,
                        attn_src.query_key_value.bias,
                        f"{prefix}.attention.query_key_value.bias",
                        f"{src_prefix}.attention.query_key_value.bias",
                    )
                _copy(
                    attn_dst.dense.weight,
                    attn_src.dense.weight,
                    f"{prefix}.attention.dense.weight",
                    f"{src_prefix}.attention.dense.weight",
                )
                if attn_dst.dense.bias is not None:
                    _copy(
                        attn_dst.dense.bias,
                        attn_src.dense.bias,
                        f"{prefix}.attention.dense.bias",
                        f"{src_prefix}.attention.dense.bias",
                    )

                _copy(
                    layer.mlp.dense_h_to_4h.weight,
                    src.mlp.dense_h_to_4h.weight,
                    f"{prefix}.mlp.dense_h_to_4h.weight",
                    f"{src_prefix}.mlp.dense_h_to_4h.weight",
                )
                _copy(
                    layer.mlp.dense_h_to_4h.bias,
                    src.mlp.dense_h_to_4h.bias,
                    f"{prefix}.mlp.dense_h_to_4h.bias",
                    f"{src_prefix}.mlp.dense_h_to_4h.bias",
                )
                _copy(
                    layer.mlp.dense_4h_to_h.weight,
                    src.mlp.dense_4h_to_h.weight,
                    f"{prefix}.mlp.dense_4h_to_h.weight",
                    f"{src_prefix}.mlp.dense_4h_to_h.weight",
                )
                _copy(
                    layer.mlp.dense_4h_to_h.bias,
                    src.mlp.dense_4h_to_h.bias,
                    f"{prefix}.mlp.dense_4h_to_h.bias",
                    f"{src_prefix}.mlp.dense_4h_to_h.bias",
                )

            if isinstance(self.final_layer_norm, RewrittenLayerNorm):
                copy_ln_params_to_rewritten(self.final_layer_norm, gpt.final_layer_norm)
                mapping["final_layer_norm.affine"] = "gpt_neox.final_layer_norm"
            else:
                _copy(self.final_layer_norm.weight, gpt.final_layer_norm.weight, "final_layer_norm.weight", "gpt_neox.final_layer_norm.weight")
                _copy(self.final_layer_norm.bias, gpt.final_layer_norm.bias, "final_layer_norm.bias", "gpt_neox.final_layer_norm.bias")
        return mapping


def freeze_pythia_eps_parameters(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, RewrittenLayerNorm):
            if m.allow_matmul:
                m.inv_sqrt_var.eps.requires_grad = False
            else:
                m.log_eps.eps.requires_grad = False
        if isinstance(m, GibbsTopKSoftmax):
            if m.allow_matmul:
                m.inv_z.eps.requires_grad = False
            else:
                m.log_z.eps.requires_grad = False
            if m.use_exact_tail_mass:
                m.gibbs_tail_prob_eps.requires_grad = False
                if m.allow_matmul:
                    m.inv_z_all.eps.requires_grad = False
                else:
                    m.log_z_all.eps.requires_grad = False
                    m.log_z_top.eps.requires_grad = False
