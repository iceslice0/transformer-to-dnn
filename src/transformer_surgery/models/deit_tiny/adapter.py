"""Model adapters for DeiT-Tiny (Oxford-IIIT Pet and ImageNet-1k classification)."""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Tuple

import torch.nn as nn
from torch.utils.data import DataLoader

from transformer_surgery.cli.surgery_config import SurgeryConfig
from transformer_surgery.internal.util import DEFAULT_MODEL_KEY
from transformer_surgery.models.adapters import SurgeryModelAdapter
from transformer_surgery.models.pet import PET_NUM_CLASSES, build_pet_loaders, load_timm_deit_checkpoint


class DeiTTinyPetAdapter(SurgeryModelAdapter):
    key = DEFAULT_MODEL_KEY
    patient_name = "DeiT-Tiny"
    dataset_name = "Oxford-IIIT Pet"
    num_classes = PET_NUM_CLASSES
    strict_graph_options = True

    def build_loaders(self, cfg: Any) -> Tuple[DataLoader, DataLoader]:
        return build_pet_loaders(cfg)

    def load_reference_checkpoint(self, path: str) -> nn.Module:
        return load_timm_deit_checkpoint(path, self.num_classes)

    def build_surgery_model(self, cfg: SurgeryConfig, extra: Optional[Mapping[str, Any]] = None) -> nn.Module:
        from transformer_surgery.models.deit_tiny.surgery_model import DeiTTinySurgeryModel

        return DeiTTinySurgeryModel(cfg, num_classes=self.num_classes)

    def reference_key(self, key: str) -> str:
        # The student's final norm is timm's ``norm`` (timm's ``fc_norm`` is an identity for token pooling).
        return re.sub(r"^fc_norm\.", "norm.", super().reference_key(key))


class DeiTTinyImageNetAdapter(DeiTTinyPetAdapter):
    key = "deit_tiny_imagenet"
    dataset_name = "ImageNet-1k"
    num_classes = 1000

    def build_loaders(self, cfg: Any) -> Tuple[DataLoader, DataLoader]:
        from transformer_surgery.models.imagenet import build_imagenet_loaders

        return build_imagenet_loaders(cfg)
