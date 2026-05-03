"""
Model adapter registry for surgery/distillation/PTQ.

The processing code is model-agnostic; concrete adapters own dataset loaders, checkpoint
construction, surgery-model reconstruction, generic calibration plumbing, and simple replacement
metadata.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer_surgery.ops import (
    SurgeryMeta,
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


def sample_topk_scores(
    scores: torch.Tensor,
    top_k: int,
    *,
    max_rows: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    flat = scores.reshape(-1, scores.shape[-1])
    rows = min(flat.shape[0], max_rows)
    teacher = flat[torch.randperm(flat.shape[0], device=flat.device)[:rows]].float()
    t = teacher - teacher.max(dim=-1, keepdim=True).values
    nk = t.shape[-1]
    k_top = min(int(top_k), nk)
    vals, idx = torch.topk(t, k=k_top, dim=-1, largest=True, sorted=True)
    return teacher, vals, idx, nk, k_top


def topk_tail_mass_stats(
    teacher: torch.Tensor,
    idx: torch.Tensor,
    nk: int,
    k_top: int,
) -> Dict[str, Any]:
    t = teacher - teacher.max(dim=-1, keepdim=True).values
    if nk > k_top:
        dense = torch.softmax(t, dim=-1)
        top_mass = dense.gather(1, idx).sum(dim=-1)
        tail_mass = (1.0 - top_mass).clamp(0.0, 1.0)
    else:
        tail_mass = torch.zeros(teacher.shape[0], device=teacher.device, dtype=torch.float32)
    return {
        "mean": float(tail_mass.mean().cpu()),
        "min": float(tail_mass.min().cpu()),
        "max": float(tail_mass.max().cpu()),
        "count": int(tail_mass.numel()),
    }


def add_sparse_topk_jeffreys_stats(
    stats: Dict[str, Any],
    teacher: torch.Tensor,
    vals: torch.Tensor,
    idx: torch.Tensor,
    nk: int,
    k_top: int,
    *,
    gibbs_tail_prob_eps: float,
    prefix: str,
) -> None:
    j_gibbs = jeffreys_distance_sparse_teacher(
        teacher,
        vals,
        idx,
        nk,
        k_top,
        gibbs_tail_prob_eps=gibbs_tail_prob_eps,
    ).mean()
    j_naive = jeffreys_naive_topk(teacher, vals, idx, nk, k_top).mean()
    stats[f"jeffreys_gibbs_mean_{prefix}"] = float(j_gibbs.cpu())
    stats[f"jeffreys_naive_mean_{prefix}"] = float(j_naive.cpu())
    stats[f"jeffreys_improvement_naive_minus_gibbs_{prefix}"] = float((j_naive - j_gibbs).cpu())


def apply_gibbs_tail_calibration(model: nn.Module, calibration: Mapping[str, Any]) -> Dict[str, Any]:
    if bool(calibration.get("disable_calib_gibbs_tail_prob", False)):
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
        return apply_gibbs_tail_calibration(model, calibration)

    def build_module_mapping(self, cfg: Any, model: Optional[nn.Module] = None) -> Dict[str, str]:
        return {}

    def build_surgery_meta(
        self,
        cfg: Any,
        *,
        calibration: Dict[str, Any],
        reference_checkpoint_abs: str,
        module_mapping: Dict[str, str],
    ) -> SurgeryMeta:
        return SurgeryMeta(
            model_key=self.key,
            patient=self.patient_name,
            dataset=self.dataset_name,
            eps=float(cfg.eps),
            gibbs_tail_prob_eps=float(cfg.gibbs_tail_prob_eps),
            disable_calib_gibbs_tail_prob=bool(cfg.disable_calib_gibbs_tail_prob),
            top_k=int(cfg.top_k),
            surgery_dtype=str(cfg.surgery_dtype),
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
            "gibbs_tail_prob_eps": float(cfg.gibbs_tail_prob_eps),
            "gibbs_tail_calibration_batches": int(cfg.gibbs_tail_calibration_batches),
            "disable_calib_gibbs_tail_prob": bool(cfg.disable_calib_gibbs_tail_prob),
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

    def calibrate_reference(self, reference: nn.Module, loader: DataLoader, cfg: Any) -> Dict[str, Any]:
        from transformer_surgery.models.deit_tiny import calibrate_timm_deit_reference

        return calibrate_timm_deit_reference(reference, loader, cfg)

    def apply_calibration(self, model: nn.Module, calibration: Mapping[str, Any]) -> Dict[str, Any]:
        return super().apply_calibration(model, calibration)

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
