"""Model adapters for DeiT-Tiny (Oxford-IIIT Pet and ImageNet-1k classification)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import torch.nn as nn
from torch.utils.data import DataLoader

from transformer_surgery.internal.util import DEFAULT_MODEL_KEY
from transformer_surgery.models.adapters import SurgeryModelAdapter


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
        from transformer_surgery.models.deit_tiny.surgery_model import DeiTTinySurgeryModel
        from transformer_surgery.models.pet import PET_NUM_CLASSES

        return DeiTTinySurgeryModel.from_surgery_config(cfg, num_classes=PET_NUM_CLASSES)

    def build_surgery_model_from_extra(self, extra: Dict[str, Any], cfg: Any) -> nn.Module:
        from transformer_surgery.models.deit_tiny.surgery_model import DeiTTinySurgeryModel
        from transformer_surgery.models.pet import PET_NUM_CLASSES

        return DeiTTinySurgeryModel.from_pretrained_extra(extra, num_classes=PET_NUM_CLASSES)

    def copy_reference_weights(self, student: nn.Module, reference: nn.Module) -> Dict[str, str]:
        return student.load_from_timm(reference)

    def freeze_surgery_parameters(self, model: nn.Module) -> None:
        from transformer_surgery.models.deit_tiny.surgery_model import freeze_eps_parameters

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
                tail = "+exact_tail" if bool(getattr(cfg, "use_exact_tail_mass", False)) else ""
                attn = f"SurgeryAttention({dot}+GibbsTopKSoftmax{tail}+{mix})"
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
        from transformer_surgery.models.deit_tiny.surgery_model import DeiTTinySurgeryModel
        from transformer_surgery.models.imagenet import IMAGENET_NUM_CLASSES

        return DeiTTinySurgeryModel.from_surgery_config(cfg, num_classes=IMAGENET_NUM_CLASSES)

    def build_surgery_model_from_extra(self, extra: Dict[str, Any], cfg: Any) -> nn.Module:
        from transformer_surgery.models.deit_tiny.surgery_model import DeiTTinySurgeryModel
        from transformer_surgery.models.imagenet import IMAGENET_NUM_CLASSES

        return DeiTTinySurgeryModel.from_pretrained_extra(extra, num_classes=IMAGENET_NUM_CLASSES)

    def copy_reference_weights(self, student: nn.Module, reference: nn.Module) -> Dict[str, str]:
        return student.load_from_timm(reference)

    def freeze_surgery_parameters(self, model: nn.Module) -> None:
        from transformer_surgery.models.deit_tiny.surgery_model import freeze_eps_parameters

        freeze_eps_parameters(model)

    def calibrate_reference(self, reference: nn.Module, loader: DataLoader, cfg: Any) -> Dict[str, Any]:
        from transformer_surgery.internal.calibration import calibrate_vit_reference

        return calibrate_vit_reference(reference, loader, cfg)

    def apply_calibration(self, model: nn.Module, calibration: Mapping[str, Any]) -> Dict[str, Any]:
        return super().apply_calibration(model, calibration)

    def build_module_mapping(self, cfg: Any, model: Optional[nn.Module] = None) -> Dict[str, str]:
        return DeiTTinyPetAdapter.build_module_mapping(self, cfg, model)
