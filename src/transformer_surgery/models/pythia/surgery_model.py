"""Pythia / GPT-NeoX surgery graph (causal LM)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from transformer_surgery.ops import NLGELU, AffineContract, RoutingStack, SurgeryCausalAttention, surgery_layer_norm

if TYPE_CHECKING:
    from transformer_surgery.cli.surgery_config import SurgeryConfig


@dataclass
class PythiaArch:
    """GPT-NeoX architecture fields that rebuild the surgery model offline (checkpoint record ``arch``)."""

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    rotary_pct: float = 0.25
    rope_theta: float = 10000.0
    attention_bias: bool = True
    use_parallel_residual: bool = True
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0

    @classmethod
    def from_hf_config(cls, config: Any) -> "PythiaArch":
        rope = dict(getattr(config, "rope_parameters", None) or {})
        return cls(
            vocab_size=int(config.vocab_size),
            hidden_size=int(config.hidden_size),
            num_hidden_layers=int(config.num_hidden_layers),
            num_attention_heads=int(config.num_attention_heads),
            intermediate_size=int(config.intermediate_size),
            rotary_pct=float(rope.get("partial_rotary_factor", getattr(config, "rotary_pct", 0.25))),
            rope_theta=float(rope.get("rope_theta", getattr(config, "rotary_emb_base", 10000.0))),
            attention_bias=bool(getattr(config, "attention_bias", True)),
            use_parallel_residual=bool(getattr(config, "use_parallel_residual", True)),
            hidden_dropout=float(getattr(config, "hidden_dropout", 0.0)),
            attention_dropout=float(getattr(config, "attention_dropout", 0.0)),
        )


class SurgeryPythiaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.dense_h_to_4h = nn.Linear(hidden_size, intermediate_size)
        self.dense_4h_to_h = nn.Linear(intermediate_size, hidden_size)
        self.act = NLGELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_4h_to_h(self.act(self.dense_h_to_4h(x)))


class SurgeryPythiaBlock(nn.Module):
    def __init__(self, cfg: "SurgeryConfig", arch: PythiaArch, rotary_ndims: int) -> None:
        super().__init__()
        hidden = arch.hidden_size
        self.use_parallel_residual = arch.use_parallel_residual
        self.input_layernorm = surgery_layer_norm(hidden, cfg)
        self.post_attention_layernorm = surgery_layer_norm(hidden, cfg)
        self.attention = SurgeryCausalAttention(
            hidden,
            arch.num_attention_heads,
            cfg.context_length,
            cfg,
            rotary_ndims=rotary_ndims,
            rope_theta=arch.rope_theta,
            attn_bias=arch.attention_bias,
            attn_drop=arch.attention_dropout,
        )
        self.mlp = SurgeryPythiaMLP(hidden, arch.intermediate_size)
        self.post_attention_dropout = nn.Dropout(arch.hidden_dropout)
        self.post_mlp_dropout = nn.Dropout(arch.hidden_dropout)
        if self.use_parallel_residual:
            self.residual_contract = AffineContract.stacked(1.0, 1.0, 1.0)
        else:
            self.attn_residual_contract = AffineContract.stacked(1.0, 1.0)
            self.mlp_residual_contract = AffineContract.stacked(1.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.post_attention_dropout(self.attention(self.input_layernorm(x)))
        if self.use_parallel_residual:
            mlp_out = self.post_mlp_dropout(self.mlp(self.post_attention_layernorm(x)))
            return self.residual_contract(RoutingStack((x, attn_out, mlp_out), dim=-1))
        attn_res = self.attn_residual_contract(RoutingStack((x, attn_out), dim=-1))
        mlp_out = self.post_mlp_dropout(self.mlp(self.post_attention_layernorm(attn_res)))
        return self.mlp_residual_contract(RoutingStack((attn_res, mlp_out), dim=-1))


class PythiaSurgeryModel(nn.Module):
    """GPT-NeoX / Pythia compiled into the surgery op vocabulary: ``arch`` from the HF config, options from ``cfg``."""

    def __init__(self, cfg: "SurgeryConfig", arch: PythiaArch) -> None:
        super().__init__()
        head_dim = arch.hidden_size // arch.num_attention_heads
        rotary_ndims = int(head_dim * arch.rotary_pct) // 2 * 2
        self.seq_len = cfg.context_length
        self.embed_in = nn.Embedding(arch.vocab_size, arch.hidden_size)
        self.emb_dropout = nn.Dropout(arch.hidden_dropout)
        self.layers = nn.ModuleList(
            [SurgeryPythiaBlock(cfg, arch, rotary_ndims) for _ in range(arch.num_hidden_layers)]
        )
        self.final_layer_norm = surgery_layer_norm(arch.hidden_size, cfg)
        self.lm_head = nn.Linear(arch.hidden_size, arch.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.is_floating_point():
            raise TypeError(f"PythiaSurgeryModel expects integer token ids, got dtype={input_ids.dtype}")
        hidden = self.emb_dropout(self.embed_in(input_ids))
        for layer in self.layers:
            hidden = layer(hidden)
        return self.lm_head(self.final_layer_norm(hidden))
