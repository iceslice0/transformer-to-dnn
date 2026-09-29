"""
DeiT-Tiny rewritten in the pseudo-hardware basis: explicit affine + nonlinear unary ops + routing.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
from transformer_surgery.internal.calibration import copy_ln_params_to_rewritten
from transformer_surgery.ops import (
    AffineContract,
    GibbsTopKSoftmax,
    NLGELU,
    RewrittenLayerNorm,
    RoutingBroadcastTensors,
    RoutingCat,
    RoutingDropPath,
    RoutingStack,
    SurgeryAttention,
)
from transformer_surgery.internal.util import get_surgery_dtype
from transformer_surgery.internal.util import ensure_mapping


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 192) -> None:
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class SurgeryMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = NLGELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SurgeryBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        seq_len: int,
        top_k: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        eps_ln: float = 1e-5,
        use_surgery_layernorm: bool = True,
        use_attention_surgery: bool = True,
        use_surgery_softmax: bool = True,
        allow_matmul: bool = False,
        gibbs_tail_prob_eps: float = 1e-5,
        use_exact_tail_mass: bool = False,
    ) -> None:
        super().__init__()
        if use_surgery_layernorm:
            self.norm1 = RewrittenLayerNorm(dim, eps=eps_ln, allow_matmul=allow_matmul)
            self.norm2 = RewrittenLayerNorm(dim, eps=eps_ln, allow_matmul=allow_matmul)
        else:
            self.norm1 = nn.LayerNorm(dim, eps=eps_ln)
            self.norm2 = nn.LayerNorm(dim, eps=eps_ln)
        self.attn = SurgeryAttention(
            dim,
            num_heads=num_heads,
            seq_len=seq_len,
            top_k=top_k,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_attention_surgery=use_attention_surgery,
            use_surgery_softmax=use_surgery_softmax,
            allow_matmul=allow_matmul,
            eps_ln=eps_ln,
            gibbs_tail_prob_eps=gibbs_tail_prob_eps,
            use_exact_tail_mass=use_exact_tail_mass,
        )
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = SurgeryMlp(in_features=dim, hidden_features=mlp_hidden, drop=drop)
        self.drop_path = RoutingDropPath(drop_path)
        self.residual_contract = AffineContract(
            "i,...i->...",
            torch.tensor([1.0, 1.0], dtype=get_surgery_dtype()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = RoutingBroadcastTensors(x, self.drop_path(self.attn(self.norm1(x))))
        x = self.residual_contract(RoutingStack((a, b), dim=-1))
        a, b = RoutingBroadcastTensors(x, self.drop_path(self.mlp(self.norm2(x))))
        x = self.residual_contract(RoutingStack((a, b), dim=-1))
        return x


class DeiTTinySurgeryModel(nn.Module):
    """
    DeiT-Tiny-compatible ViT with surgery modules. Default shapes match timm `deit_tiny_patch16_224`.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        top_k: int = 32,
        eps_ln: float = 1e-5,
        use_surgery_layernorm: bool = True,
        use_attention_surgery: bool = True,
        use_surgery_softmax: bool = True,
        allow_matmul: bool = False,
        gibbs_tail_prob_eps: float = 1e-5,
        use_exact_tail_mass: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_surgery_layernorm = use_surgery_layernorm
        self.use_attention_surgery = use_attention_surgery
        self.use_surgery_softmax = use_surgery_softmax
        self.use_exact_tail_mass = bool(use_exact_tail_mass)
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.seq_len = num_patches + 1
        self.pos_embed_contract = AffineContract(
            "i,...i->...",
            torch.tensor([1.0, 1.0], dtype=get_surgery_dtype()),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                SurgeryBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    seq_len=self.seq_len,
                    top_k=top_k,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    eps_ln=eps_ln,
                    use_surgery_layernorm=use_surgery_layernorm,
                    use_attention_surgery=use_attention_surgery,
                    use_surgery_softmax=use_surgery_softmax,
                    allow_matmul=allow_matmul,
                    gibbs_tail_prob_eps=gibbs_tail_prob_eps,
                    use_exact_tail_mass=use_exact_tail_mass,
                )
                for i in range(depth)
            ]
        )
        if use_surgery_layernorm:
            self.fc_norm = RewrittenLayerNorm(embed_dim, eps=eps_ln, allow_matmul=allow_matmul)
        else:
            self.fc_norm = nn.LayerNorm(embed_dim, eps=eps_ln)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self._init_weights()
        self.eps_ln = eps_ln
        self.gibbs_tail_prob_eps = gibbs_tail_prob_eps
        self.top_k = top_k

    @classmethod
    def from_surgery_config(cls, cfg: Any, *, num_classes: int) -> "DeiTTinySurgeryModel":
        """Build from a surgery config object."""
        return cls(
            num_classes=num_classes,
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
    def from_pretrained_extra(cls, ex: Any, *, num_classes: int) -> "DeiTTinySurgeryModel":
        """Restore architecture from checkpoint ``extra`` (plain ``dict`` or nested ``SimpleNamespace``)."""
        d = ensure_mapping(ex)
        return cls(
            num_classes=num_classes,
            top_k=int(d["top_k"]),
            eps_ln=float(d["eps_ln"]),
            gibbs_tail_prob_eps=float(d["gibbs_tail_prob_eps"]),
            use_surgery_layernorm=not bool(d["disable_layernorm_replacement"]),
            use_attention_surgery=not bool(d["disable_attention_surgery"]),
            use_surgery_softmax=not bool(d["disable_softmax_replacement"]),
            allow_matmul=bool(d["allow_matmul"]),
            use_exact_tail_mass=bool(d.get("use_exact_tail_mass", False)),
        )

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02) if isinstance(self.head, nn.Linear) else None
        if isinstance(self.head, nn.Linear) and self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = RoutingCat((cls, x), dim=1)
        pe_a, pe_b = RoutingBroadcastTensors(x, self.pos_embed)
        x = self.pos_embed_contract(RoutingStack((pe_a, pe_b), dim=-1))
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.fc_norm(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.head(x)

    def load_from_timm(self, ref: nn.Module) -> Dict[str, str]:
        """Copy compatible weights from a timm `deit_tiny_patch16_224` (or same-shape) model."""
        rd = ref.state_dict()
        sd = self.state_dict()
        mapping: Dict[str, str] = {}
        with torch.no_grad():
            for k in sd.keys():
                if k in rd and sd[k].shape == rd[k].shape:
                    sd[k].copy_(rd[k])
                    mapping[k] = k
        self.load_state_dict(sd, strict=False)
        with torch.no_grad():
            for i, rb in enumerate(ref.blocks):
                if isinstance(self.blocks[i].norm1, RewrittenLayerNorm):
                    copy_ln_params_to_rewritten(self.blocks[i].norm1, rb.norm1)
                    copy_ln_params_to_rewritten(self.blocks[i].norm2, rb.norm2)
                else:
                    self.blocks[i].norm1.weight.copy_(rb.norm1.weight)
                    self.blocks[i].norm1.bias.copy_(rb.norm1.bias)
                    self.blocks[i].norm2.weight.copy_(rb.norm2.weight)
                    self.blocks[i].norm2.bias.copy_(rb.norm2.bias)
            if isinstance(self.fc_norm, RewrittenLayerNorm):
                copy_ln_params_to_rewritten(self.fc_norm, ref.norm)
            else:
                self.fc_norm.weight.copy_(ref.norm.weight)
                self.fc_norm.bias.copy_(ref.norm.bias)
        return mapping


def freeze_eps_parameters(model: DeiTTinySurgeryModel) -> None:
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
