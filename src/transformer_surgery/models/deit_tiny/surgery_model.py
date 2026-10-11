"""
DeiT-Tiny rewritten in the pseudo-hardware basis: explicit affine + nonlinear unary ops + routing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from transformer_surgery.ops import (
    GELU_MODES,
    AffineContract,
    NLGELU,
    RoutingBroadcastTensors,
    RoutingCat,
    RoutingDropPath,
    RoutingStack,
    SigmoidGELU,
    SurgeryAttention,
    surgery_layer_norm,
)

if TYPE_CHECKING:
    from transformer_surgery.cli.surgery_config import SurgeryConfig


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 192) -> None:
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class SurgeryMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, cfg: "SurgeryConfig", drop: float = 0.0) -> None:
        super().__init__()
        if cfg.gelu not in GELU_MODES:
            raise ValueError(f"gelu must be one of {GELU_MODES}, got {cfg.gelu!r}")
        self.fc1 = nn.Linear(in_features, hidden_features)
        # "sigmoid": x * sigmoid(1.702 x), a model change used for NPUs without a GELU unit.
        if cfg.gelu == "exact":
            self.act = NLGELU()
        else:
            self.act = SigmoidGELU(allow_matmul=cfg.allow_matmul, hadamard=cfg.hadamard)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class SurgeryBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        seq_len: int,
        cfg: "SurgeryConfig",
        *,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = surgery_layer_norm(dim, cfg)
        self.norm2 = surgery_layer_norm(dim, cfg)
        self.attn = SurgeryAttention(dim, num_heads, seq_len, cfg, attn_drop=attn_drop, proj_drop=drop)
        self.mlp = SurgeryMlp(dim, int(dim * mlp_ratio), cfg, drop=drop)
        self.drop_path = RoutingDropPath(drop_path)
        self.residual_contract = AffineContract.stacked(1.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = RoutingBroadcastTensors(x, self.drop_path(self.attn(self.norm1(x))))
        x = self.residual_contract(RoutingStack((a, b), dim=-1))
        a, b = RoutingBroadcastTensors(x, self.drop_path(self.mlp(self.norm2(x))))
        return self.residual_contract(RoutingStack((a, b), dim=-1))


class DeiTTinySurgeryModel(nn.Module):
    """
    DeiT-Tiny-compatible ViT with surgery modules configured by a ``SurgeryConfig``. Default shapes match timm
    ``deit_tiny_patch16_224``.
    """

    def __init__(
        self,
        cfg: "SurgeryConfig",
        *,
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
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.seq_len = self.patch_embed.num_patches + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.pos_embed_contract = AffineContract.stacked(1.0, 1.0)
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                SurgeryBlock(
                    embed_dim,
                    num_heads,
                    self.seq_len,
                    cfg,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.fc_norm = surgery_layer_norm(embed_dim, cfg)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if isinstance(self.head, nn.Linear):
            nn.init.trunc_normal_(self.head.weight, std=0.02)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = RoutingCat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        pe_a, pe_b = RoutingBroadcastTensors(x, self.pos_embed)
        x = self.pos_drop(self.pos_embed_contract(RoutingStack((pe_a, pe_b), dim=-1)))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.fc_norm(x)[:, 0])
