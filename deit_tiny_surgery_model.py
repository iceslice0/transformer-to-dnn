"""
DeiT-Tiny rewritten in the pseudo-hardware basis: explicit affine + local PWL epilogues + routing.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from timm.layers import DropPath

from surgery_utils import (
    AffineContract,
    GibbsTopKSoftmax,
    MatMul,
    GELUUnaryPWL,
    PairwiseDotBySquare,
    RewrittenLayerNormAbsSign,
    SparseWeightedSumBySquare,
    copy_ln_params_to_rewritten,
    get_surgery_dtype,
)


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


class SurgeryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        seq_len: int,
        top_k: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_attention_surgery: bool = True,
        use_surgery_softmax: bool = True,
        allow_matmul: bool = False,
        eps_ln: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_attention_surgery = use_attention_surgery
        self.use_surgery_softmax = use_surgery_softmax
        self.allow_matmul = allow_matmul
        if use_attention_surgery and not use_surgery_softmax and allow_matmul:
            self.matmul = MatMul()
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if use_attention_surgery:
            self.dot = PairwiseDotBySquare(self.head_dim, allow_matmul=allow_matmul)
            if use_surgery_softmax:
                self.gibbs = GibbsTopKSoftmax(seq_len, top_k, eps=eps_ln, allow_matmul=allow_matmul)
                self.sparse_mix = SparseWeightedSumBySquare(allow_matmul=allow_matmul)
            else:
                self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.register_buffer("attn_scale", torch.tensor(float(self.head_dim) ** -0.5))
            self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if not self.use_attention_surgery:
            s = self.attn_scale.to(device=q.device, dtype=q.dtype)
            qs = q * s
            kt = k.transpose(-2, -1)
            attn = qs @ kt
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            attn = attn @ v
        elif self.use_surgery_softmax:
            scores = self.dot(q, k)
            probs, idx, _q_tail = self.gibbs(scores)
            attn = self.sparse_mix(probs, idx, v)
        else:
            scores = self.dot(q, k)
            attn = scores.softmax(dim=-1)
            attn = self.attn_drop(attn)
            if self.allow_matmul:
                attn = self.matmul(attn, v)
            else:
                _, _, nq, nk = attn.shape
                v_b = v.unsqueeze(2).expand(-1, -1, nq, nk, -1)
                attn = (attn.unsqueeze(-1) * v_b).sum(dim=3)
        attn = attn.transpose(1, 2).reshape(b, n, c)
        attn = self.proj(attn)
        attn = self.proj_drop(attn)
        return attn


class SurgeryMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = GELUUnaryPWL()
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
    ) -> None:
        super().__init__()
        if use_surgery_layernorm:
            self.norm1 = RewrittenLayerNormAbsSign(dim, eps=eps_ln, allow_matmul=allow_matmul)
            self.norm2 = RewrittenLayerNormAbsSign(dim, eps=eps_ln, allow_matmul=allow_matmul)
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
        )
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = SurgeryMlp(in_features=dim, hidden_features=mlp_hidden, drop=drop)
        self.drop_path = DropPath(drop_path)
        self.residual_contract = AffineContract(
            "i,...i->...",
            torch.tensor([1.0, 1.0], dtype=get_surgery_dtype()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.broadcast_tensors(x, self.drop_path(self.attn(self.norm1(x))))
        x = self.residual_contract(torch.stack((a, b), dim=-1))
        a, b = torch.broadcast_tensors(x, self.drop_path(self.mlp(self.norm2(x))))
        x = self.residual_contract(torch.stack((a, b), dim=-1))
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
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_surgery_layernorm = use_surgery_layernorm
        self.use_attention_surgery = use_attention_surgery
        self.use_surgery_softmax = use_surgery_softmax
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
                )
                for i in range(depth)
            ]
        )
        if use_surgery_layernorm:
            self.fc_norm = RewrittenLayerNormAbsSign(embed_dim, eps=eps_ln, allow_matmul=allow_matmul)
        else:
            self.fc_norm = nn.LayerNorm(embed_dim, eps=eps_ln)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self._init_weights()
        self.eps_ln = eps_ln
        self.top_k = top_k

    @classmethod
    def from_surgery_run_config(cls, cfg: Any, *, num_classes: int) -> "DeiTTinySurgeryModel":
        """Build from :class:`pet_reference_utils.SurgeryRunConfig` (or same fields)."""
        return cls(
            num_classes=num_classes,
            top_k=int(cfg.top_k),
            eps_ln=float(cfg.eps),
            use_surgery_layernorm=not bool(cfg.disable_layernorm_replacement),
            use_attention_surgery=not bool(cfg.disable_attention_surgery),
            use_surgery_softmax=not bool(cfg.disable_softmax_replacement),
            allow_matmul=bool(cfg.allow_matmul),
        )

    @classmethod
    def from_pretrained_extra(cls, ex: Dict[str, Any], *, num_classes: int) -> "DeiTTinySurgeryModel":
        """Restore architecture from a checkpoint ``extra`` dict (e.g. ``surgery_pre_ft.pt``)."""
        return cls(
            num_classes=num_classes,
            top_k=int(ex["top_k"]),
            eps_ln=float(ex["eps_ln"]),
            use_surgery_layernorm=not bool(ex["disable_layernorm_replacement"]),
            use_attention_surgery=not bool(ex["disable_attention_surgery"]),
            use_surgery_softmax=not bool(ex["disable_softmax_replacement"]),
            allow_matmul=bool(ex["allow_matmul"]),
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
        x = torch.cat((cls, x), dim=1)
        pe_a, pe_b = torch.broadcast_tensors(x, self.pos_embed)
        x = self.pos_embed_contract(torch.stack((pe_a, pe_b), dim=-1))
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
                if isinstance(self.blocks[i].norm1, RewrittenLayerNormAbsSign):
                    copy_ln_params_to_rewritten(self.blocks[i].norm1, rb.norm1)
                    copy_ln_params_to_rewritten(self.blocks[i].norm2, rb.norm2)
                else:
                    self.blocks[i].norm1.weight.copy_(rb.norm1.weight)
                    self.blocks[i].norm1.bias.copy_(rb.norm1.bias)
                    self.blocks[i].norm2.weight.copy_(rb.norm2.weight)
                    self.blocks[i].norm2.bias.copy_(rb.norm2.bias)
            if hasattr(ref, "norm"):
                if isinstance(self.fc_norm, RewrittenLayerNormAbsSign):
                    copy_ln_params_to_rewritten(self.fc_norm, ref.norm)
                else:
                    self.fc_norm.weight.copy_(ref.norm.weight)
                    self.fc_norm.bias.copy_(ref.norm.bias)
        return mapping


def freeze_eps_parameters(model: DeiTTinySurgeryModel) -> None:
    for m in model.modules():
        if isinstance(m, RewrittenLayerNormAbsSign):
            if hasattr(m, "log_eps"):
                m.log_eps.eps.requires_grad = False
            if hasattr(m, "inv_sqrt_var"):
                m.inv_sqrt_var.eps.requires_grad = False
        if isinstance(m, GibbsTopKSoftmax):
            if hasattr(m, "log_z"):
                m.log_z.eps.requires_grad = False
            if hasattr(m, "inv_z"):
                m.inv_z.eps.requires_grad = False
