"""
DeiT-Tiny rewritten in the pseudo-hardware basis: explicit affine + nonlinear unary ops + routing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformer_surgery.models.adapters import (
    add_sparse_topk_jeffreys_stats,
    sample_topk_scores,
    topk_tail_mass_stats,
)
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
    copy_ln_params_to_rewritten,
    get_surgery_dtype,
)
from transformer_surgery.util import get_device


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
                    gibbs_tail_prob_eps=gibbs_tail_prob_eps,
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
        )

    @classmethod
    def from_pretrained_extra(cls, ex: Dict[str, Any], *, num_classes: int) -> "DeiTTinySurgeryModel":
        """Restore architecture from a checkpoint ``extra`` dict (e.g. ``surgery.pt``)."""
        return cls(
            num_classes=num_classes,
            top_k=int(ex["top_k"]),
            eps_ln=float(ex["eps_ln"]),
            gibbs_tail_prob_eps=float(ex["gibbs_tail_prob_eps"]),
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
            if hasattr(ref, "norm"):
                if isinstance(self.fc_norm, RewrittenLayerNorm):
                    copy_ln_params_to_rewritten(self.fc_norm, ref.norm)
                else:
                    self.fc_norm.weight.copy_(ref.norm.weight)
                    self.fc_norm.bias.copy_(ref.norm.bias)
        return mapping


def freeze_eps_parameters(model: DeiTTinySurgeryModel) -> None:
    for m in model.modules():
        if isinstance(m, RewrittenLayerNorm):
            if hasattr(m, "log_eps"):
                m.log_eps.eps.requires_grad = False
            if hasattr(m, "inv_sqrt_var"):
                m.inv_sqrt_var.eps.requires_grad = False
        if isinstance(m, GibbsTopKSoftmax):
            if hasattr(m, "log_z"):
                m.log_z.eps.requires_grad = False
            if hasattr(m, "inv_z"):
                m.inv_z.eps.requires_grad = False


def timm_attention_scores(attn: nn.Module, x: torch.Tensor) -> torch.Tensor:
    b, n, c = x.shape
    head_dim = c // int(attn.num_heads)
    qkv = attn.qkv(x).reshape(b, n, 3, attn.num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    return (q @ k.transpose(-2, -1)) * float(attn.scale)


@torch.no_grad()
def calibrate_timm_deit_reference(reference: nn.Module, loader: DataLoader, cfg: Any) -> Dict[str, Any]:
    device = get_device()
    dt = get_surgery_dtype()
    reference.eval()
    stats: Dict[str, Any] = {}
    eps = float(cfg.eps)
    gibbs_tail_prob_eps = float(cfg.gibbs_tail_prob_eps)
    disable_tail_calib = bool(cfg.disable_calib_gibbs_tail_prob)
    calibration_batches = max(1, int(getattr(cfg, "gibbs_tail_calibration_batches", 1)))
    top_k = int(cfg.top_k)
    use_cuda = device.type == "cuda"
    mse_acc = 0.0
    n_ln = 0
    tail_eps_sum_by_block = []
    tail_eps_count_by_block = []
    tail_eps_min_by_block = []
    tail_eps_max_by_block = []
    block0_teachers = []
    block0_vals_list = []
    block0_idx_list = []
    block0_nk = 0
    block0_k_top = 0
    processed_batches = 0

    for batch_idx, (batch, _) in enumerate(loader):
        if batch_idx >= calibration_batches:
            break
        processed_batches += 1
        batch = batch.to(device, dtype=dt, non_blocking=use_cuda)

        b = batch.shape[0]
        x = reference.patch_embed(batch)
        x = torch.cat((reference.cls_token.expand(b, -1, -1), x), dim=1) + reference.pos_embed
        x = reference.pos_drop(x)
        if batch_idx == 0:
            y_ref0 = reference.blocks[0].norm1(x)
            rw0 = RewrittenLayerNorm(reference.embed_dim, eps=eps, allow_matmul=cfg.allow_matmul).to(
                device=device, dtype=dt
            )
            copy_ln_params_to_rewritten(rw0, reference.blocks[0].norm1)
            y_rw0 = rw0(x)
            stats["ln_rewrite_mse_layer0_minibatch"] = float(torch.mean((y_ref0 - y_rw0).pow(2)).cpu())

        h = x
        for block_idx, blk in enumerate(reference.blocks):
            n1 = blk.norm1(h)
            rw = RewrittenLayerNorm(reference.embed_dim, eps=eps, allow_matmul=cfg.allow_matmul).to(
                device=device, dtype=dt
            )
            copy_ln_params_to_rewritten(rw, blk.norm1)
            mse_acc += torch.mean((rw(h) - n1).pow(2)).item()
            n_ln += 1
            if block_idx == 0 or not disable_tail_calib:
                scores = timm_attention_scores(blk.attn, n1)
                teacher, vals, idx, nk, k_top = sample_topk_scores(scores, top_k)
                if not disable_tail_calib:
                    tail_stats = topk_tail_mass_stats(teacher, idx, nk, k_top)
                    count = int(tail_stats["count"])
                    while len(tail_eps_sum_by_block) <= block_idx:
                        tail_eps_sum_by_block.append(0.0)
                        tail_eps_count_by_block.append(0)
                        tail_eps_min_by_block.append(float("inf"))
                        tail_eps_max_by_block.append(float("-inf"))
                    tail_eps_sum_by_block[block_idx] += float(tail_stats["mean"]) * count
                    tail_eps_count_by_block[block_idx] += count
                    tail_eps_min_by_block[block_idx] = min(tail_eps_min_by_block[block_idx], float(tail_stats["min"]))
                    tail_eps_max_by_block[block_idx] = max(tail_eps_max_by_block[block_idx], float(tail_stats["max"]))
                if block_idx == 0:
                    block0_teachers.append(teacher)
                    block0_vals_list.append(vals)
                    block0_idx_list.append(idx)
                    block0_nk = nk
                    block0_k_top = k_top
            h = h + blk.attn(n1)
            n2 = blk.norm2(h)
            rw2 = RewrittenLayerNorm(reference.embed_dim, eps=eps, allow_matmul=cfg.allow_matmul).to(
                device=device, dtype=dt
            )
            copy_ln_params_to_rewritten(rw2, blk.norm2)
            mse_acc += torch.mean((rw2(h) - n2).pow(2)).item()
            n_ln += 1
            h = h + blk.mlp(n2)
        h_pre = h
        h_out = reference.norm(h_pre)
        rwf = RewrittenLayerNorm(reference.embed_dim, eps=eps, allow_matmul=cfg.allow_matmul).to(
            device=device, dtype=dt
        )
        copy_ln_params_to_rewritten(rwf, reference.norm)
        mse_acc += torch.mean((rwf(h_pre) - h_out).pow(2)).item()
        n_ln += 1

    if processed_batches == 0:
        raise ValueError("gibbs tail calibration requires at least one validation batch")
    stats["ln_rewrite_mse_all_norms_mean"] = mse_acc / max(n_ln, 1)

    stats["disable_calib_gibbs_tail_prob"] = disable_tail_calib
    stats["gibbs_tail_calibration_batches_requested"] = calibration_batches
    stats["gibbs_tail_calibration_batches"] = processed_batches
    stats["gibbs_tail_prob_eps_configured"] = gibbs_tail_prob_eps
    if disable_tail_calib:
        metric_tail_prob_eps = gibbs_tail_prob_eps
    else:
        tail_eps_by_block = [
            tail_sum / max(tail_count, 1)
            for tail_sum, tail_count in zip(tail_eps_sum_by_block, tail_eps_count_by_block)
        ]
        stats["gibbs_tail_prob_eps_calibrated_by_block"] = tail_eps_by_block
        stats["gibbs_tail_prob_eps_calibrated_min_by_block"] = tail_eps_min_by_block
        stats["gibbs_tail_prob_eps_calibrated_max_by_block"] = tail_eps_max_by_block
        stats["gibbs_tail_prob_eps_calibration_rows_by_block"] = tail_eps_count_by_block
        stats["gibbs_tail_prob_eps_calibrated_mean"] = float(sum(tail_eps_by_block) / max(len(tail_eps_by_block), 1))
        metric_tail_prob_eps = tail_eps_by_block[0] if tail_eps_by_block else gibbs_tail_prob_eps
    stats["gibbs_tail_prob_eps_metric"] = float(metric_tail_prob_eps)

    if not block0_teachers or not block0_vals_list or not block0_idx_list:
        return stats
    block0_teacher = torch.cat(block0_teachers, dim=0)
    block0_vals = torch.cat(block0_vals_list, dim=0)
    block0_idx = torch.cat(block0_idx_list, dim=0)
    add_sparse_topk_jeffreys_stats(
        stats,
        block0_teacher,
        block0_vals,
        block0_idx,
        block0_nk,
        block0_k_top,
        gibbs_tail_prob_eps=metric_tail_prob_eps,
        prefix="cached",
    )

    teacher2 = torch.randn(4096, block0_nk, device=device, dtype=dt)
    t2 = teacher2 - teacher2.max(dim=-1, keepdim=True).values
    vals2, idx2 = torch.topk(t2, k=block0_k_top, dim=-1, largest=True, sorted=True)
    add_sparse_topk_jeffreys_stats(
        stats,
        teacher2,
        vals2,
        idx2,
        block0_nk,
        block0_k_top,
        gibbs_tail_prob_eps=metric_tail_prob_eps,
        prefix="synthetic",
    )
    return stats
