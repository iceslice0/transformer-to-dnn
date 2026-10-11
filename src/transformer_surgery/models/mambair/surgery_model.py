"""
MambaIRv2 Light SR rewritten in the surgery basis.

Two surgeries are applied to the installed :class:`MambaIRv2Light` (matching the reference
experiment, which only ever touched ``nn.Softmax`` and ``nn.LayerNorm``):

* every ``nn.LayerNorm`` -> :class:`RewrittenLayerNorm` (a clean drop-in, same forward shape);
* every ``WindowAttention`` -> :class:`SurgeryWindowAttention`, whose ``QK^T`` softmax ``@ V``
  is replaced by :class:`GibbsTopKSoftmax` + sparse mix + DeiT-style omitted-tail reinjection
  (the :class:`SparseTailAttention` path shared with :class:`SurgeryAttention`).

Tail mass is controlled exactly as in DeiT via ``gibbs_tail_prob_eps`` (initial), surgery
calibration (``disable_calib_gibbs_tail_prob=false``), or ``use_exact_tail_mass``.

The only *live* softmax in MambaIRv2Light is ``WindowAttention.softmax``; ``AttentiveLayer.softmax``
is defined but never called, so it is left untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_surgery.models.mambair.arch import WindowAttention, build_mambair_lightsr
from transformer_surgery.ops import (
    AffineMean,
    PairwiseDotBySquare,
    RoutingExpand,
    SparseTailAttention,
    graph_options,
    surgery_layer_norm,
)

if TYPE_CHECKING:
    from transformer_surgery.cli.surgery_config import SurgeryConfig


class SurgeryWindowAttention(SparseTailAttention):
    """Drop-in for :class:`WindowAttention` using the surgery attention path.

    Keeps ``proj`` and ``relative_position_bias_table`` under the same names as the reference so
    reference weights copy by name; the sparse ``dot``/``gibbs``/``sparse_mix`` submodules are new.
    """

    def __init__(self, dim: int, window_size: Tuple[int, int], num_heads: int, cfg: "SurgeryConfig") -> None:
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.use_surgery_softmax = not cfg.disable_softmax_replacement
        seq_len = window_size[0] * window_size[1]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        self.proj = nn.Linear(dim, dim)
        # top_k == 0 is the reference's "uniform" attention: every query attends equally to all
        # keys (mask/logits ignored) -> output = mean of V over keys. GibbsTopKSoftmax is sparse
        # and cannot express this, so use an affine key-mean instead.
        self.uniform = self.use_surgery_softmax and cfg.top_k == 0
        hadamard, strict_product = graph_options(cfg)
        self.dot = PairwiseDotBySquare(
            dim // num_heads, allow_matmul=cfg.allow_matmul, hadamard=hadamard, strict_product=strict_product
        )
        if self.uniform:
            self.key_mean = AffineMean(dim=-2, keepdim=True)
        elif self.use_surgery_softmax:
            self._init_sparse_tail(seq_len, cfg)

    def forward(self, qkv: torch.Tensor, rpi: torch.Tensor, mask=None) -> torch.Tensor:
        b_, n, c3 = qkv.shape
        c = c3 // 3
        q, k, v = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        if self.uniform:
            # Uniform attention (top_k == 0): mean over keys, broadcast to every query.
            x = RoutingExpand(self.key_mean(v), -1, -1, n, -1)  # [b_, nH, n, head_dim]
            return self.proj(x.transpose(1, 2).reshape(b_, n, c))

        # PairwiseDotBySquare applies the 1/sqrt(d) scale internally -> pass unscaled q, k.
        attn = self.dot(q, k)  # [b_, nH, n, n]
        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(n, n, -1).permute(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        if self.use_surgery_softmax:
            x = self._sparse_tail_mix(*self.gibbs(attn), v)
        else:
            x = F.softmax(attn, dim=-1) @ v
        return self.proj(x.transpose(1, 2).reshape(b_, n, c))


class MambaIRLightSurgeryModel(nn.Module):
    """MambaIRv2 Light SR with surgery LayerNorm / attention swapped in place."""

    def __init__(self, cfg: "SurgeryConfig") -> None:
        super().__init__()
        self.scale = cfg.scale
        self.net = build_mambair_lightsr(cfg.scale)
        self.seq_len = self.net.window_size**2
        if not cfg.disable_attention_surgery:
            for name, attn in list(self.net.named_modules()):
                if isinstance(attn, WindowAttention):
                    new = SurgeryWindowAttention(attn.dim, attn.window_size, attn.num_heads, cfg)
                    self.net.set_submodule(name, new)
        if not cfg.disable_layernorm_replacement:
            for name, ln in list(self.net.named_modules()):
                if isinstance(ln, nn.LayerNorm):
                    self.net.set_submodule(name, surgery_layer_norm(ln.normalized_shape[0], cfg))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
