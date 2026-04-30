"""
Model adapter registry for surgery/distillation/PTQ.

The processing code is model-agnostic; concrete adapters own dataset loaders, checkpoint
construction, surgery-model reconstruction, calibration diagnostics, and teacher->student weight
copy details.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer_surgery.ops import (
    RewrittenLayerNorm,
    SurgeryMeta,
    copy_ln_params_to_rewritten,
    get_surgery_dtype,
    jeffreys_distance_sparse_teacher,
    jeffreys_naive_topk,
)
from transformer_surgery.util import DEFAULT_MODEL_KEY, describe_dtype, get_device


def _is_set(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _model_key_from_config_or_extra(cfg: Any = None, extra: Optional[Mapping[str, Any]] = None) -> str:
    if extra is not None and _is_set(extra.get("model_key")):
        return str(extra["model_key"]).strip()
    if cfg is not None and _is_set(getattr(cfg, "model_key", None)):
        return str(getattr(cfg, "model_key")).strip()
    return DEFAULT_MODEL_KEY


class SurgeryModelAdapter:
    key: str = DEFAULT_MODEL_KEY
    patient_name: str = "unknown"
    dataset_name: str = "unknown"

    def reference_checkpoint_path(self, cfg: Any) -> str:
        path = getattr(cfg, "reference_checkpoint", None)
        if not _is_set(path):
            raise ValueError(f"No reference checkpoint configured for model adapter {self.key!r}")
        return os.path.abspath(str(path))

    def build_loaders(self, cfg: Any) -> Tuple[DataLoader, DataLoader]:
        raise NotImplementedError

    def load_reference_checkpoint(self, path: str) -> nn.Module:
        raise NotImplementedError

    def build_surgery_model(self, cfg: Any) -> nn.Module:
        raise NotImplementedError

    def build_surgery_model_from_extra(self, extra: Dict[str, Any], cfg: Any) -> nn.Module:
        raise NotImplementedError

    def copy_reference_weights(self, student: nn.Module, reference: nn.Module) -> Dict[str, str]:
        raise NotImplementedError

    def freeze_surgery_parameters(self, model: nn.Module) -> None:
        return None

    def calibrate_reference(self, reference: nn.Module, loader: DataLoader, cfg: Any) -> Dict[str, Any]:
        return {}

    def apply_calibration(self, model: nn.Module, calibration: Mapping[str, Any]) -> Dict[str, Any]:
        return {}

    def build_module_mapping(self, cfg: Any, model: Optional[nn.Module] = None) -> Dict[str, str]:
        return {}

    def build_surgery_meta(
        self,
        cfg: Any,
        *,
        calibration: Dict[str, Any],
        reference_checkpoint_abs: str,
        pwl: Dict[str, Any],
        module_mapping: Dict[str, str],
    ) -> SurgeryMeta:
        return SurgeryMeta(
            model_key=self.key,
            patient=self.patient_name,
            dataset=self.dataset_name,
            eps=float(cfg.eps),
            gibbs_tail_use_prob_eps=bool(cfg.gibbs_tail_use_prob_eps),
            gibbs_tail_prob_eps=float(cfg.gibbs_tail_prob_eps),
            top_k=int(cfg.top_k),
            surgery_dtype=str(cfg.surgery_dtype),
            pwl=pwl,
            calibration=dict(calibration),
            module_mapping=module_mapping,
            reference_checkpoint=reference_checkpoint_abs,
            allow_matmul=bool(cfg.allow_matmul),
        )

    def pre_ft_checkpoint_extra(self, cfg: Any, *, mapping: Dict[str, Any], metadata_path: str) -> Dict[str, Any]:
        return {
            "model_key": self.key,
            "patient": self.patient_name,
            "dataset": self.dataset_name,
            "reference_checkpoint": self.reference_checkpoint_path(cfg),
            "meta_ref": os.path.basename(metadata_path),
            "mapping": mapping,
            "top_k": int(cfg.top_k),
            "eps_ln": float(cfg.eps),
            "gibbs_tail_use_prob_eps": bool(cfg.gibbs_tail_use_prob_eps),
            "gibbs_tail_prob_eps": float(cfg.gibbs_tail_prob_eps),
            "config_json": cfg.config_json_path,
            "disable_layernorm_replacement": cfg.disable_layernorm_replacement,
            "disable_attention_surgery": cfg.disable_attention_surgery,
            "disable_softmax_replacement": cfg.disable_softmax_replacement,
            "allow_matmul": cfg.allow_matmul,
            "surgery_dtype": str(cfg.surgery_dtype),
        }


class DeiTTinyPetAdapter(SurgeryModelAdapter):
    key = DEFAULT_MODEL_KEY
    patient_name = "DeiT-Tiny"
    dataset_name = "Oxford-IIIT Pet"

    def build_loaders(self, cfg: Any) -> Tuple[DataLoader, DataLoader]:
        from transformer_surgery.models.pet import build_pet_loaders

        return build_pet_loaders(cfg)

    def load_reference_checkpoint(self, path: str) -> nn.Module:
        from transformer_surgery.models.pet import load_timm_deit_pet_checkpoint

        return load_timm_deit_pet_checkpoint(path)

    def build_surgery_model(self, cfg: Any) -> nn.Module:
        from transformer_surgery.models.deit_tiny import DeiTTinySurgeryModel
        from transformer_surgery.models.pet import PET_NUM_CLASSES

        return DeiTTinySurgeryModel.from_surgery_config(cfg, num_classes=PET_NUM_CLASSES)

    def build_surgery_model_from_extra(self, extra: Dict[str, Any], cfg: Any) -> nn.Module:
        from transformer_surgery.models.deit_tiny import DeiTTinySurgeryModel
        from transformer_surgery.models.pet import PET_NUM_CLASSES

        return DeiTTinySurgeryModel.from_pretrained_extra(extra, num_classes=PET_NUM_CLASSES)

    def copy_reference_weights(self, student: nn.Module, reference: nn.Module) -> Dict[str, str]:
        return student.load_from_timm(reference)

    def freeze_surgery_parameters(self, model: nn.Module) -> None:
        from transformer_surgery.models.deit_tiny import freeze_eps_parameters

        freeze_eps_parameters(model)

    @staticmethod
    def _attention_scores(attn: nn.Module, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        head_dim = c // int(attn.num_heads)
        qkv = attn.qkv(x).reshape(b, n, 3, attn.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        return (q @ k.transpose(-2, -1)) * float(attn.scale)

    @staticmethod
    def _sample_topk_tail_mass(
        scores: torch.Tensor,
        top_k: int,
        *,
        max_rows: int = 4096,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, Dict[str, float]]:
        flat = scores.reshape(-1, scores.shape[-1])
        rows = min(flat.shape[0], max_rows)
        teacher = flat[:rows].float()
        t = teacher - teacher.max(dim=-1, keepdim=True).values
        nk = t.shape[-1]
        k_top = min(int(top_k), nk)
        vals, idx = torch.topk(t, k=k_top, dim=-1, largest=True, sorted=True)
        if nk > k_top:
            dense = torch.softmax(t, dim=-1)
            top_mass = dense.gather(1, idx).sum(dim=-1)
            tail_mass = (1.0 - top_mass).clamp(0.0, 1.0)
        else:
            tail_mass = torch.zeros(rows, device=scores.device, dtype=torch.float32)
        tail_stats = {
            "mean": float(tail_mass.mean().cpu()),
            "min": float(tail_mass.min().cpu()),
            "max": float(tail_mass.max().cpu()),
        }
        return teacher, vals, idx, nk, k_top, tail_stats

    @torch.no_grad()
    def calibrate_reference(self, reference: nn.Module, loader: DataLoader, cfg: Any) -> Dict[str, Any]:
        """
        DeiT-specific diagnostics: layernorm rewrite MSE and dense-vs-top-k Jeffreys metrics.
        The generic surgery stage treats this as opaque adapter metadata.
        """
        device = get_device()
        dt = get_surgery_dtype()
        reference.eval()
        stats: Dict[str, Any] = {}
        eps = float(cfg.eps)
        gibbs_tail_use_prob_eps = bool(cfg.gibbs_tail_use_prob_eps)
        gibbs_tail_prob_eps = float(cfg.gibbs_tail_prob_eps)
        top_k = int(cfg.top_k)
        use_cuda = device.type == "cuda"
        batch, _ = next(iter(loader))
        batch = batch.to(device, dtype=dt, non_blocking=use_cuda)

        b = batch.shape[0]
        x = reference.patch_embed(batch)
        x = torch.cat((reference.cls_token.expand(b, -1, -1), x), dim=1) + reference.pos_embed
        x = reference.pos_drop(x)
        h0 = x
        y_ref0 = reference.blocks[0].norm1(h0)
        rw0 = RewrittenLayerNorm(reference.embed_dim, eps=eps, allow_matmul=cfg.allow_matmul).to(
            device=device, dtype=dt
        )
        copy_ln_params_to_rewritten(rw0, reference.blocks[0].norm1)
        y_rw0 = rw0(h0)
        stats["ln_rewrite_mse_layer0_minibatch"] = float(torch.mean((y_ref0 - y_rw0).pow(2)).cpu())

        h = x
        mse_acc = 0.0
        n_ln = 0
        tail_eps_by_block = []
        tail_eps_min_by_block = []
        tail_eps_max_by_block = []
        block0_teacher: Optional[torch.Tensor] = None
        block0_vals: Optional[torch.Tensor] = None
        block0_idx: Optional[torch.Tensor] = None
        block0_nk = 0
        block0_k_top = 0
        for block_idx, blk in enumerate(reference.blocks):
            n1 = blk.norm1(h)
            rw = RewrittenLayerNorm(reference.embed_dim, eps=eps, allow_matmul=cfg.allow_matmul).to(
                device=device, dtype=dt
            )
            copy_ln_params_to_rewritten(rw, blk.norm1)
            mse_acc += torch.mean((rw(h) - n1).pow(2)).item()
            n_ln += 1
            scores = self._attention_scores(blk.attn, n1)
            teacher, vals, idx, nk, k_top, tail_stats = self._sample_topk_tail_mass(scores, top_k)
            tail_eps_by_block.append(tail_stats["mean"])
            tail_eps_min_by_block.append(tail_stats["min"])
            tail_eps_max_by_block.append(tail_stats["max"])
            if block_idx == 0:
                block0_teacher = teacher
                block0_vals = vals
                block0_idx = idx
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
        stats["ln_rewrite_mse_all_norms_mean"] = mse_acc / max(n_ln, 1)

        stats["gibbs_tail_use_prob_eps"] = gibbs_tail_use_prob_eps
        stats["gibbs_tail_prob_eps_configured"] = gibbs_tail_prob_eps
        stats["gibbs_tail_prob_eps_calibrated_by_block"] = tail_eps_by_block
        stats["gibbs_tail_prob_eps_calibrated_min_by_block"] = tail_eps_min_by_block
        stats["gibbs_tail_prob_eps_calibrated_max_by_block"] = tail_eps_max_by_block
        stats["gibbs_tail_prob_eps_calibrated_mean"] = float(sum(tail_eps_by_block) / max(len(tail_eps_by_block), 1))
        metric_tail_prob_eps = tail_eps_by_block[0] if gibbs_tail_use_prob_eps and tail_eps_by_block else gibbs_tail_prob_eps
        stats["gibbs_tail_prob_eps_metric"] = float(metric_tail_prob_eps)

        if block0_teacher is None or block0_vals is None or block0_idx is None:
            return stats
        j_gibbs = jeffreys_distance_sparse_teacher(
            block0_teacher,
            block0_vals,
            block0_idx,
            block0_nk,
            block0_k_top,
            tail_use_prob_eps=gibbs_tail_use_prob_eps,
            tail_prob_eps=metric_tail_prob_eps,
        ).mean()
        j_naive = jeffreys_naive_topk(block0_teacher, block0_vals, block0_idx, block0_nk, block0_k_top).mean()
        stats["jeffreys_gibbs_mean_cached"] = float(j_gibbs.cpu())
        stats["jeffreys_naive_mean_cached"] = float(j_naive.cpu())
        stats["jeffreys_improvement_naive_minus_gibbs_cached"] = float((j_naive - j_gibbs).cpu())

        teacher2 = torch.randn(4096, block0_nk, device=device, dtype=dt)
        t2 = teacher2 - teacher2.max(dim=-1, keepdim=True).values
        vals2, idx2 = torch.topk(t2, k=block0_k_top, dim=-1, largest=True, sorted=True)
        j_gibbs2 = jeffreys_distance_sparse_teacher(
            teacher2,
            vals2,
            idx2,
            block0_nk,
            block0_k_top,
            tail_use_prob_eps=gibbs_tail_use_prob_eps,
            tail_prob_eps=metric_tail_prob_eps,
        ).mean()
        j_naive2 = jeffreys_naive_topk(teacher2, vals2, idx2, block0_nk, block0_k_top).mean()
        stats["jeffreys_gibbs_mean_synthetic"] = float(j_gibbs2.cpu())
        stats["jeffreys_naive_mean_synthetic"] = float(j_naive2.cpu())
        stats["jeffreys_improvement_naive_minus_gibbs_synthetic"] = float((j_naive2 - j_gibbs2).cpu())
        return stats

    def apply_calibration(self, model: nn.Module, calibration: Mapping[str, Any]) -> Dict[str, Any]:
        if not bool(getattr(model, "gibbs_tail_use_prob_eps", False)):
            return {}
        values = calibration.get("gibbs_tail_prob_eps_calibrated_by_block")
        if not isinstance(values, list) or not values or not hasattr(model, "blocks"):
            return {}
        applied = []
        with torch.no_grad():
            for block_idx, blk in enumerate(model.blocks):
                gibbs = getattr(getattr(blk, "attn", None), "gibbs", None)
                param = getattr(gibbs, "gibbs_tail_prob_eps", None)
                if not isinstance(param, nn.Parameter):
                    continue
                raw_value = float(values[min(block_idx, len(values) - 1)])
                value = max(0.0, min(raw_value, 1.0 - 1e-7))
                param.copy_(torch.tensor(value, device=param.device, dtype=param.dtype))
                applied.append(value)
        if not applied:
            return {}
        model.gibbs_tail_prob_eps = float(sum(applied) / len(applied))
        return {
            "gibbs_tail_prob_eps_applied_by_block": applied,
            "gibbs_tail_prob_eps_applied_mean": model.gibbs_tail_prob_eps,
        }

    def build_module_mapping(self, cfg: Any, model: Optional[nn.Module] = None) -> Dict[str, str]:
        if cfg.disable_layernorm_replacement:
            ln = "nn.LayerNorm"
        elif cfg.allow_matmul:
            ln = "RewrittenLayerNorm(rsqrt.mul)"
        else:
            ln = "RewrittenLayerNorm(log/sqrt_exp)"
        if cfg.disable_attention_surgery:
            attn = "SurgeryAttention(vanilla scaled QK^T softmax @ V)"
        else:
            dot = "PairwiseDotBySquare(QK^T matmul)" if cfg.allow_matmul else "PairwiseDotBySquare(square identity)"
            if cfg.disable_softmax_replacement:
                attn = f"SurgeryAttention({dot}+full_softmax+dense@V)"
            else:
                mix = (
                    "SparseWeightedSumBySquare(elementwise p*v)"
                    if cfg.allow_matmul
                    else "SparseWeightedSumBySquare(square identity)"
                )
                attn = f"SurgeryAttention({dot}+GibbsTopKSoftmax+{mix})"
        mapping: Dict[str, str] = {}
        depth = len(model.blocks) if model is not None and hasattr(model, "blocks") else 12
        for i in range(depth):
            mapping[f"blocks.{i}.norm1"] = ln
            mapping[f"blocks.{i}.attn"] = attn
            mapping[f"blocks.{i}.norm2"] = ln
            mapping[f"blocks.{i}.mlp.act"] = "NLGELU"
        mapping["fc_norm"] = ln
        return mapping


_ADAPTERS: Dict[str, SurgeryModelAdapter] = {}


def register_model_adapter(adapter: SurgeryModelAdapter) -> None:
    _ADAPTERS[adapter.key] = adapter


def get_model_adapter(key_or_cfg: Any = None, *, extra: Optional[Mapping[str, Any]] = None) -> SurgeryModelAdapter:
    key = key_or_cfg if isinstance(key_or_cfg, str) else _model_key_from_config_or_extra(key_or_cfg, extra)
    key = str(key).strip() if _is_set(key) else DEFAULT_MODEL_KEY
    try:
        return _ADAPTERS[key]
    except KeyError as exc:
        known = ", ".join(sorted(_ADAPTERS))
        raise KeyError(f"Unknown model adapter {key!r}. Available adapters: {known}") from exc


def load_surgery_student_checkpoint(
    path: str,
    cfg: Any,
    *,
    adapter: Optional[SurgeryModelAdapter] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    device = get_device()
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    extra = dict(payload["extra"])
    if adapter is None:
        adapter = get_model_adapter(cfg, extra=extra)
    if getattr(cfg, "top_k", None) is not None:
        extra["top_k"] = int(cfg.top_k)
    if getattr(cfg, "eps", None) is not None:
        extra["eps_ln"] = float(cfg.eps)
    if getattr(cfg, "gibbs_tail_use_prob_eps", None) is not None:
        extra["gibbs_tail_use_prob_eps"] = bool(cfg.gibbs_tail_use_prob_eps)
    if getattr(cfg, "gibbs_tail_prob_eps", None) is not None:
        extra["gibbs_tail_prob_eps"] = float(cfg.gibbs_tail_prob_eps)
    runtime_dtype = get_surgery_dtype()
    extra["surgery_dtype"] = describe_dtype(runtime_dtype)
    extra.setdefault("model_key", adapter.key)
    extra.setdefault("patient", adapter.patient_name)
    extra.setdefault("dataset", adapter.dataset_name)
    model = adapter.build_surgery_model_from_extra(extra, cfg).to(device=device, dtype=get_surgery_dtype())
    model.load_state_dict(payload["model_state_dict"], strict=True)
    adapter.freeze_surgery_parameters(model)
    return model, extra


register_model_adapter(DeiTTinyPetAdapter())
