"""
MambaIRv2 Light SR rewritten in the surgery basis.

Two surgeries are applied to the installed :class:`MambaIRv2Light` (matching the reference
experiment, which only ever touched ``nn.Softmax`` and ``nn.LayerNorm``):

* every ``nn.LayerNorm`` -> :class:`RewrittenLayerNorm` (a clean drop-in, same forward shape);
* every ``WindowAttention`` -> :class:`SurgeryWindowAttention`, whose ``QK^T`` softmax ``@ V``
  is replaced by :class:`GibbsTopKSoftmax` + sparse mix + DeiT-style omitted-tail reinjection
  (same ``adj_probs`` / ``q_tail * mean(V)`` path as :class:`SurgeryAttention`).

Tail mass is controlled exactly as in DeiT via ``gibbs_tail_prob_eps`` (initial), surgery
calibration (``disable_calib_gibbs_tail_prob=false``), or ``use_exact_tail_mass``.

The only *live* softmax in MambaIRv2Light is ``WindowAttention.softmax``; ``AttentiveLayer.softmax``
is defined but never called, so it is left untouched.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_surgery.internal.calibration import copy_ln_params_to_rewritten
from transformer_surgery.internal.util import ensure_mapping, get_surgery_dtype
from transformer_surgery.models.mambair.arch import (
    MambaIRv2Light,
    WindowAttention,
    build_mambair_lightsr,
)
from transformer_surgery.ops import (
    AffineContract,
    AffineHadamard,
    AffineMean,
    GibbsTopKSoftmax,
    PairwiseDotBySquare,
    RewrittenLayerNorm,
    RoutingBroadcastTensors,
    RoutingExpand,
    RoutingStack,
    SparseWeightedSumBySquare,
)


def _set_module_by_name(root: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace ``root.<dotted.name>`` with ``new_module`` (supports numeric ModuleList indices)."""
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


class SurgeryWindowAttention(nn.Module):
    """Drop-in for :class:`WindowAttention` using the surgery attention path.

    Keeps ``proj`` and ``relative_position_bias_table`` under the same names as the reference so
    reference weights copy by name; the sparse ``dot``/``gibbs``/``mix`` submodules are new.
    """

    def __init__(
        self,
        dim: int,
        window_size,
        num_heads: int,
        *,
        qkv_bias: bool = True,
        top_k: int = 256,
        eps_ln: float = 1e-5,
        gibbs_tail_prob_eps: float = 1e-5,
        use_surgery_softmax: bool = True,
        allow_matmul: bool = False,
        use_exact_tail_mass: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.use_surgery_softmax = use_surgery_softmax
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.seq_len = int(window_size[0] * window_size[1])

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        self.proj = nn.Linear(dim, dim)

        self.top_k = int(top_k)
        # top_k == 0 is the reference's "uniform" attention: every query attends equally to all
        # keys (mask/logits ignored) -> output = mean of V over keys. GibbsTopKSoftmax is sparse
        # and cannot express this, so use an affine key-mean instead.
        self.uniform = use_surgery_softmax and self.top_k == 0
        self.dot = PairwiseDotBySquare(head_dim, allow_matmul=allow_matmul)
        if self.uniform:
            self.key_mean = AffineMean(dim=-2, keepdim=True)
        elif use_surgery_softmax:
            self.gibbs = GibbsTopKSoftmax(
                self.seq_len,
                top_k,
                eps=eps_ln,
                gibbs_tail_prob_eps=gibbs_tail_prob_eps,
                allow_matmul=allow_matmul,
                use_exact_tail_mass=use_exact_tail_mass,
            )
            self.mix = SparseWeightedSumBySquare(allow_matmul=allow_matmul)
            # Same omitted-tail reinjection as SurgeryAttention: without it, sparse mix only
            # applies (1 - q_tail) * top-k mass and drops the reserved tail entirely.
            k_eff = min(int(top_k), int(self.seq_len))
            n_dropped = max(int(self.seq_len) - k_eff, 1)
            inv_dropped = 1.0 / n_dropped
            n_over_dropped = float(self.seq_len) / n_dropped
            self.mean_v = AffineMean(dim=-2, keepdim=True)
            self.scale_mean_v_by_tail = AffineHadamard()
            self.adj_probs_contract = AffineContract(
                "i,...i->...",
                torch.tensor([1.0, -inv_dropped], dtype=get_surgery_dtype()),
            )
            self.attn_tail_sum = AffineContract(
                "i,...i->...",
                torch.tensor([1.0, n_over_dropped], dtype=get_surgery_dtype()),
            )

    def forward(self, qkv: torch.Tensor, rpi: torch.Tensor, mask=None) -> torch.Tensor:
        b_, n, c3 = qkv.shape
        c = c3 // 3
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.uniform:
            # Uniform attention (top_k == 0): mean over keys, broadcast to every query.
            x = RoutingExpand(self.key_mean(v), -1, -1, n, -1)  # [b_, nH, n, head_dim]
            x = x.transpose(1, 2).reshape(b_, n, c)
            return self.proj(x)

        # PairwiseDotBySquare applies the 1/sqrt(d) scale internally -> pass unscaled q, k.
        attn = self.dot(q, k)  # [b_, nH, n, n]

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, n, n
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        if self.use_surgery_softmax:
            probs, idx, q_tail = self.gibbs(attn)
            _p, _qt = RoutingBroadcastTensors(probs, q_tail)
            adjusted_probs = self.adj_probs_contract(RoutingStack((_p, _qt), dim=-1))
            attn_top = self.mix(adjusted_probs, idx, v)
            mean_v = self.mean_v(v)
            tail_contrib = self.scale_mean_v_by_tail(mean_v, q_tail)
            _a, _t = RoutingBroadcastTensors(attn_top, tail_contrib)
            x = self.attn_tail_sum(RoutingStack((_a, _t), dim=-1))
        else:
            attn = F.softmax(attn, dim=-1)
            x = attn @ v

        x = x.transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return x


class MambaIRLightSurgeryModel(nn.Module):
    """MambaIRv2 Light SR with surgery LayerNorm / attention swapped in place."""

    def __init__(
        self,
        scale: int = 2,
        top_k: int = 256,
        eps_ln: float = 1e-5,
        gibbs_tail_prob_eps: float = 1e-5,
        use_surgery_layernorm: bool = True,
        use_attention_surgery: bool = True,
        use_surgery_softmax: bool = True,
        allow_matmul: bool = False,
        use_exact_tail_mass: bool = False,
    ) -> None:
        super().__init__()
        self.scale = int(scale)
        self.top_k = int(top_k)
        self.eps_ln = float(eps_ln)
        self.gibbs_tail_prob_eps = float(gibbs_tail_prob_eps)
        self.use_surgery_layernorm = bool(use_surgery_layernorm)
        self.use_attention_surgery = bool(use_attention_surgery)
        self.use_surgery_softmax = bool(use_surgery_softmax)
        self.allow_matmul = bool(allow_matmul)
        self.use_exact_tail_mass = bool(use_exact_tail_mass)

        self.net = build_mambair_lightsr(scale)
        self.seq_len = int(self.net.window_size) ** 2

        if self.use_attention_surgery:
            self._replace_window_attention()
        if self.use_surgery_layernorm:
            self._replace_layernorm()

    def _replace_window_attention(self) -> None:
        targets = [(n, m) for n, m in self.net.named_modules() if isinstance(m, WindowAttention)]
        for name, attn in targets:
            new = SurgeryWindowAttention(
                attn.dim,
                attn.window_size,
                attn.num_heads,
                qkv_bias=attn.qkv_bias,
                top_k=self.top_k,
                eps_ln=self.eps_ln,
                gibbs_tail_prob_eps=self.gibbs_tail_prob_eps,
                use_surgery_softmax=self.use_surgery_softmax,
                allow_matmul=self.allow_matmul,
                use_exact_tail_mass=self.use_exact_tail_mass,
            )
            _set_module_by_name(self.net, name, new)

    def _replace_layernorm(self) -> None:
        targets = [(n, m) for n, m in self.net.named_modules() if isinstance(m, nn.LayerNorm)]
        for name, ln in targets:
            new = RewrittenLayerNorm(int(ln.normalized_shape[0]), eps=self.eps_ln, allow_matmul=self.allow_matmul)
            _set_module_by_name(self.net, name, new)

    @classmethod
    def from_surgery_config(cls, cfg: Any) -> "MambaIRLightSurgeryModel":
        return cls(
            scale=int(cfg.scale),
            top_k=int(cfg.top_k),
            eps_ln=float(cfg.eps),
            gibbs_tail_prob_eps=float(cfg.gibbs_tail_prob_eps),
            use_surgery_layernorm=not bool(cfg.disable_layernorm_replacement),
            use_attention_surgery=not bool(cfg.disable_attention_surgery),
            use_surgery_softmax=not bool(cfg.disable_softmax_replacement),
            allow_matmul=bool(cfg.allow_matmul),
            use_exact_tail_mass=bool(getattr(cfg, "use_exact_tail_mass", False)),
        )

    @classmethod
    def from_pretrained_extra(cls, ex: Any) -> "MambaIRLightSurgeryModel":
        d = ensure_mapping(ex)
        return cls(
            scale=int(d["scale"]),
            top_k=int(d["top_k"]),
            eps_ln=float(d["eps_ln"]),
            gibbs_tail_prob_eps=float(d["gibbs_tail_prob_eps"]),
            use_surgery_layernorm=not bool(d["disable_layernorm_replacement"]),
            use_attention_surgery=not bool(d["disable_attention_surgery"]),
            use_surgery_softmax=not bool(d["disable_softmax_replacement"]),
            allow_matmul=bool(d["allow_matmul"]),
            use_exact_tail_mass=bool(d.get("use_exact_tail_mass", False)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def load_from_reference(self, ref: MambaIRv2Light) -> Dict[str, str]:
        """Copy matching-name/shape weights from a stock ``MambaIRv2Light`` reference."""
        rd = ref.state_dict()
        sd = self.net.state_dict()
        mapping: Dict[str, str] = {}
        with torch.no_grad():
            for k in sd.keys():
                if k in rd and sd[k].shape == rd[k].shape:
                    sd[k].copy_(rd[k])
                    mapping[k] = k
        self.net.load_state_dict(sd, strict=False)
        # RewrittenLayerNorm affine params live under different names -> copy explicitly.
        with torch.no_grad():
            for name, module in self.net.named_modules():
                if isinstance(module, RewrittenLayerNorm):
                    ref_ln = ref.get_submodule(name)
                    copy_ln_params_to_rewritten(module, ref_ln)
        return mapping
