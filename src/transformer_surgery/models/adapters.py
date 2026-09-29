"""
Model adapter registry for surgery/distillation/PTQ.

The processing code is model-agnostic; concrete adapters own dataset loaders, checkpoint
construction, surgery-model reconstruction, generic calibration plumbing, and simple replacement
metadata. Adapter implementations live under ``models/<family>/adapter.py``.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformer_surgery.internal.calibration import apply_gibbs_tail_calibration
from transformer_surgery.internal.metrics import jeffreys_divergence_dense
from transformer_surgery.internal.reporting import CALIBRATION_LEGEND_TEXT, describe_dtype
from transformer_surgery.internal.util import accuracy_and_loss, get_surgery_dtype, set_surgery_dtype
from transformer_surgery.internal.util import (
    DEFAULT_MODEL_KEY,
    ensure_mapping,
    get_device,
    maybe_surgery_cuda_autocast,
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

    def calibrate_after_build(self, model: nn.Module, loader: DataLoader, cfg: Any) -> Dict[str, Any]:
        """Optional Gibbs/LN stats gathered on the built surgery student (e.g. causal LM)."""
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

    # -- Task hooks (classification defaults; regression models override) -------------------
    # ``primary`` is a higher-is-better scalar (accuracy for classification, PSNR for SR), so
    # keep-best logic in surgery/distill is task-agnostic.

    def example_model_input(self, cfg: Any = None) -> Optional[torch.Tensor]:
        """Optional representative input for structure dumps; ``None`` uses the image default."""
        return None

    def primary_metric_name(self) -> str:
        return "accuracy"

    def next_stage_hint(self) -> str:
        return "Next: run the distill stage."

    def evaluate(self, model: nn.Module, val_loader: DataLoader, cfg: Any = None) -> Tuple[float, float, Dict[str, float]]:
        """Return ``(primary, loss, extra_metrics)`` for a standalone model on ``val_loader``."""
        acc, loss = accuracy_and_loss(model, val_loader, nn.CrossEntropyLoss())
        return float(acc), float(loss), {}

    @torch.no_grad()
    def eval_student_vs_teacher(
        self, teacher: nn.Module, student: nn.Module, val_loader: DataLoader, *, temperature: float = 1.0
    ) -> Tuple[float, float, float]:
        """Return ``(primary, loss, teacher_match)`` for a student against a teacher."""
        device = get_device()
        teacher.eval()
        student.eval()
        use_cuda = device.type == "cuda"
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        match_sum = torch.zeros((), device=device, dtype=torch.float64)
        correct = torch.zeros((), device=device, dtype=torch.long)
        n = 0
        dt = get_surgery_dtype()
        for x, y in val_loader:
            x = x.to(device, dtype=dt, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)
            with maybe_surgery_cuda_autocast(device, dt):
                t_out = teacher(x)
                s_out = student(x)
            loss_sum += F.cross_entropy(s_out.float(), y, reduction="sum").double()
            match_sum += jeffreys_divergence_dense(t_out, s_out, temperature=temperature).sum().double()
            correct += (s_out.argmax(dim=-1) == y).sum()
            n += y.size(0)
        return correct.item() / n, loss_sum.item() / n, match_sum.item() / n

    def distill_step_losses(
        self, student_out: torch.Tensor, teacher_out: torch.Tensor, target: torch.Tensor, *, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(hard_loss, teacher_match_loss)`` for one training batch."""
        hard = F.cross_entropy(student_out.float(), target, reduction="mean")
        match = jeffreys_divergence_dense(teacher_out, student_out, temperature=temperature).mean()
        return hard, match


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


# Concrete adapters (heavy imports stay lazy inside each adapter's methods).
from transformer_surgery.models.deit_tiny.adapter import (  # noqa: E402
    DeiTTinyImageNetAdapter,
    DeiTTinyPetAdapter,
)
from transformer_surgery.models.mambair.adapter import MambaIRLightSRAdapter  # noqa: E402
from transformer_surgery.models.pythia.adapter import Pythia70MWikiText2Adapter  # noqa: E402

register_model_adapter(DeiTTinyPetAdapter())
register_model_adapter(DeiTTinyImageNetAdapter())
register_model_adapter(MambaIRLightSRAdapter())
register_model_adapter(Pythia70MWikiText2Adapter())
