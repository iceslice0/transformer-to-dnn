"""
Model adapter registry for surgery/distillation/PTQ.

The processing code is model-agnostic; concrete adapters own dataset loaders, checkpoint
construction, surgery-model reconstruction, generic calibration plumbing, and simple replacement
metadata.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer_surgery.internal.calibration import (
    add_sparse_topk_jeffreys_stats,
    apply_gibbs_tail_calibration,
    sample_topk_scores,
    topk_tail_mass_stats,
)
from transformer_surgery.internal.reporting import CALIBRATION_LEGEND_TEXT, describe_dtype
from transformer_surgery.internal.util import get_surgery_dtype, set_surgery_dtype
from transformer_surgery.internal.util import (
    DEFAULT_MODEL_KEY,
    ensure_mapping,
    get_device,
    namespace_from_mapping,
    torch_dtype_from_name,
)


def _is_set(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


class SurgeryModelAdapter:
    key: str = DEFAULT_MODEL_KEY
    patient_name: str = "unknown"
    dataset_name: str = "unknown"

    def reference_checkpoint_path(self, cfg: Any) -> str:
        path = cfg.reference_checkpoint
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

    def build_surgery_meta_dict(
        self,
        cfg: Any,
        *,
        calibration: Dict[str, Any],
        reference_checkpoint_abs: str,
        module_mapping: Dict[str, str],
    ) -> Dict[str, Any]:
        return {
            **asdict(cfg),
            "model_key": self.key,
            "patient": self.patient_name,
            "dataset": self.dataset_name,
            "calibration": dict(calibration),
            "module_mapping": module_mapping,
            "reference_checkpoint": reference_checkpoint_abs,
            "calibration_legend": CALIBRATION_LEGEND_TEXT,
        }

    def pre_ft_checkpoint_extra(self, cfg: Any, *, mapping: Dict[str, Any], metadata_path: str) -> Dict[str, Any]:
        ex = asdict(cfg)
        ex.update(
            {
                "model_key": self.key,
                "patient": self.patient_name,
                "dataset": self.dataset_name,
                "reference_checkpoint": self.reference_checkpoint_path(cfg),
                "meta_ref": os.path.basename(metadata_path),
                "mapping": mapping,
                "eps_ln": float(cfg.eps),
            }
        )
        return ex


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
        from transformer_surgery.internal.calibration import calibrate_vit_reference

        return calibrate_vit_reference(reference, loader, cfg)

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
        depth = len(model.blocks) if model is not None else 12
        for i in range(depth):
            mapping[f"blocks.{i}.norm1"] = ln
            mapping[f"blocks.{i}.attn"] = attn
            mapping[f"blocks.{i}.norm2"] = ln
            mapping[f"blocks.{i}.mlp.act"] = "NLGELU"
        mapping["fc_norm"] = ln
        return mapping


class DeiTTinyImageNetAdapter(SurgeryModelAdapter):
    key = "deit_tiny_imagenet"
    patient_name = "DeiT-Tiny"
    dataset_name = "ImageNet-1k"

    def build_loaders(self, cfg: Any) -> Tuple[DataLoader, DataLoader]:
        from transformer_surgery.models.imagenet import build_imagenet_loaders

        return build_imagenet_loaders(cfg)

    def load_reference_checkpoint(self, path: str) -> nn.Module:
        from transformer_surgery.models.imagenet import load_timm_deit_imagenet_checkpoint

        return load_timm_deit_imagenet_checkpoint(path)

    def build_surgery_model(self, cfg: Any) -> nn.Module:
        from transformer_surgery.models.deit_tiny import DeiTTinySurgeryModel
        from transformer_surgery.models.imagenet import IMAGENET_NUM_CLASSES

        return DeiTTinySurgeryModel.from_surgery_config(cfg, num_classes=IMAGENET_NUM_CLASSES)

    def build_surgery_model_from_extra(self, extra: Dict[str, Any], cfg: Any) -> nn.Module:
        from transformer_surgery.models.deit_tiny import DeiTTinySurgeryModel
        from transformer_surgery.models.imagenet import IMAGENET_NUM_CLASSES

        return DeiTTinySurgeryModel.from_pretrained_extra(extra, num_classes=IMAGENET_NUM_CLASSES)

    def copy_reference_weights(self, student: nn.Module, reference: nn.Module) -> Dict[str, str]:
        return student.load_from_timm(reference)

    def freeze_surgery_parameters(self, model: nn.Module) -> None:
        from transformer_surgery.models.deit_tiny import freeze_eps_parameters

        freeze_eps_parameters(model)

    def calibrate_reference(self, reference: nn.Module, loader: DataLoader, cfg: Any) -> Dict[str, Any]:
        from transformer_surgery.internal.calibration import calibrate_vit_reference

        return calibrate_vit_reference(reference, loader, cfg)

    def apply_calibration(self, model: nn.Module, calibration: Mapping[str, Any]) -> Dict[str, Any]:
        return super().apply_calibration(model, calibration)

    def build_module_mapping(self, cfg: Any, model: Optional[nn.Module] = None) -> Dict[str, str]:
        return DeiTTinyPetAdapter.build_module_mapping(self, cfg, model)


_ADAPTERS: Dict[str, SurgeryModelAdapter] = {}


def register_model_adapter(adapter: SurgeryModelAdapter) -> None:
    _ADAPTERS[adapter.key] = adapter


def surgery_dtype_from_extra(extra: Any) -> torch.dtype:
    """Parse ``extra['surgery_dtype']`` written at checkpoint save time (e.g. ``float16``)."""
    return torch_dtype_from_name(str(ensure_mapping(extra)["surgery_dtype"]))


def get_model_adapter(model_key: str) -> SurgeryModelAdapter:
    key = str(model_key).strip() if _is_set(model_key) else DEFAULT_MODEL_KEY
    return _ADAPTERS[key]


def apply_load_cfg_overrides(extra_ns: Any, cfg: Any) -> None:
    if cfg.top_k is not None:
        extra_ns.top_k = int(cfg.top_k)
    if cfg.eps is not None:
        extra_ns.eps_ln = float(cfg.eps)


def load_surgery_student_checkpoint(
    path: str,
    cfg: Any,
    *,
    adapter: Optional[SurgeryModelAdapter] = None,
    surgery_dtype: Optional[torch.dtype] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    device = get_device()
    payload = torch.load(path, map_location=device, weights_only=False)
    extra_ns = namespace_from_mapping(dict(payload["extra"]))
    state_dict = payload["model_state_dict"]
    if surgery_dtype is None:
        surgery_dtype = surgery_dtype_from_extra(extra_ns)
    set_surgery_dtype(surgery_dtype)
    if adapter is None:
        adapter = get_model_adapter(extra_ns.model_key)
    apply_load_cfg_overrides(extra_ns, cfg)
    extra_ns.surgery_dtype = describe_dtype(get_surgery_dtype())
    model = adapter.build_surgery_model_from_extra(ensure_mapping(extra_ns), cfg).to(
        device=device, dtype=get_surgery_dtype()
    )
    model.load_state_dict(state_dict, strict=True)
    adapter.freeze_surgery_parameters(model)
    return model, ensure_mapping(extra_ns)


register_model_adapter(DeiTTinyPetAdapter())
register_model_adapter(DeiTTinyImageNetAdapter())
