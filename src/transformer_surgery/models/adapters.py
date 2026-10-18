"""
Model adapter registry for surgery/distillation/PTQ.

The processing code is model-agnostic; concrete adapters own dataset loaders, checkpoint
construction, surgery-model reconstruction, calibration, and task metrics. Adapter implementations
live under ``models/<family>/adapter.py``.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformer_surgery.cli.surgery_config import SurgeryConfig
from transformer_surgery.internal.calibration import (
    AppliedTail,
    TailCalibration,
    apply_gibbs_tail_calibration,
    calibrate_reference,
)
from transformer_surgery.internal.metrics import jeffreys_divergence_dense
from transformer_surgery.internal.reporting import CALIBRATION_LEGEND_TEXT, describe_dtype
from transformer_surgery.internal.routing import install_routers_from_state_dict
from transformer_surgery.internal.util import (
    accuracy_and_loss,
    copy_matching_weights,
    dataclass_from_record,
    get_device,
    get_surgery_dtype,
    maybe_surgery_cuda_autocast,
    set_surgery_dtype,
    torch_dtype_from_name,
)
from transformer_surgery.ops import GibbsTopKSoftmax, NLGELU, RewrittenLayerNorm, SigmoidGELU


class SurgeryModelAdapter:
    key: str = ""
    patient_name: str = "unknown"
    dataset_name: str = "unknown"
    # The DeiT-Tiny graph implements hadamard=True and the norm-expansion strict products.
    strict_graph_options: bool = False

    def reference_checkpoint_path(self, cfg: Any) -> str:
        if not cfg.reference_checkpoint:
            raise ValueError(f"No reference checkpoint configured for model adapter {self.key!r}")
        return os.path.abspath(cfg.reference_checkpoint)

    def build_loaders(self, cfg: Any) -> Tuple[DataLoader, DataLoader]:
        raise NotImplementedError

    def build_selection_loader(self, cfg: Any) -> DataLoader:
        """The training split in evaluation form (no augmentation, fixed order) for checkpoint selection; the
        held-out loader of ``build_loaders`` is evaluated only."""
        raise NotImplementedError(f"No selection loader for model adapter {self.key!r}")

    def load_reference_checkpoint(self, path: str) -> nn.Module:
        raise NotImplementedError

    def build_surgery_model(self, cfg: SurgeryConfig, extra: Optional[Mapping[str, Any]] = None) -> nn.Module:
        """Surgery student for ``cfg``; ``extra`` is the checkpoint record when restoring a saved student."""
        raise NotImplementedError

    def reference_key(self, key: str) -> str:
        """Reference ``state_dict`` name of a student entry (``RewrittenLayerNorm`` keeps its weights in ``affine``)."""
        return key.replace(".affine.", ".")

    def copy_reference_weights(self, student: nn.Module, reference: nn.Module) -> Dict[str, str]:
        return copy_matching_weights(student, reference.state_dict(), self.reference_key)

    def freeze_surgery_parameters(self, model: nn.Module) -> None:
        """Freeze the calibrated tail mass where the exact runtime tail mass replaces it."""
        for m in model.modules():
            if isinstance(m, GibbsTopKSoftmax) and m.use_exact_tail_mass:
                m.gibbs_tail_prob_eps.requires_grad_(False)

    def calibrate_reference(
        self, reference: nn.Module, loader: DataLoader, cfg: SurgeryConfig
    ) -> Optional[TailCalibration]:
        return calibrate_reference(reference, loader, cfg)

    def calibrate_after_build(
        self, model: nn.Module, loader: DataLoader, cfg: SurgeryConfig
    ) -> Optional[TailCalibration]:
        """Gibbs/LN statistics gathered on the built surgery student (e.g. causal LM); replaces the reference ones."""
        return None

    def apply_calibration(self, model: nn.Module, tail: TailCalibration) -> Optional[AppliedTail]:
        return apply_gibbs_tail_calibration(model, tail.gibbs_tail_prob_eps_calibrated_by_block or [])

    def build_module_mapping(self, cfg: SurgeryConfig, model: nn.Module) -> Dict[str, str]:
        """Replacement chosen for each LayerNorm, attention and MLP activation of the surgery student."""
        if cfg.allow_matmul:
            graph = "fast"
        elif cfg.hadamard:
            graph = "hadamard"
        else:
            graph = f"strict, {cfg.strict_product} products"
        ln = "nn.LayerNorm" if cfg.disable_layernorm_replacement else f"RewrittenLayerNorm({graph})"
        if cfg.disable_attention_surgery:
            attn = "dense scaled QK^T softmax @ V"
        elif cfg.disable_softmax_replacement:
            attn = f"PairwiseDotBySquare({graph}) + dense softmax @ V"
        else:
            tail = "calibrated"
            if cfg.use_exact_tail_mass:
                tail = "exact"
            elif cfg.disable_calib_gibbs_tail_prob:
                tail = "fixed"
            attn = f"PairwiseDotBySquare + GibbsTopKSoftmax({tail} tail) + SparseWeightedSumBySquare ({graph})"
        mapping: Dict[str, str] = {}
        for name, m in model.named_modules():
            if isinstance(m, (RewrittenLayerNorm, nn.LayerNorm)):
                mapping[name] = ln
            elif hasattr(m, "use_surgery_softmax"):
                mapping[name] = "AffineMean uniform key-mean @ V" if getattr(m, "uniform", False) else attn
            elif isinstance(m, (NLGELU, SigmoidGELU)):
                mapping[name] = type(m).__name__
        return mapping

    def _run_record(self, cfg: SurgeryConfig) -> Dict[str, Any]:
        return {
            **asdict(cfg),
            "model_key": self.key,
            "patient": self.patient_name,
            "dataset": self.dataset_name,
            "reference_checkpoint": self.reference_checkpoint_path(cfg),
        }

    def build_surgery_meta_dict(
        self, cfg: SurgeryConfig, *, calibration: Dict[str, Any], module_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        return {
            **self._run_record(cfg),
            "calibration": dict(calibration),
            "module_mapping": module_mapping,
            "calibration_legend": CALIBRATION_LEGEND_TEXT,
        }

    def pre_ft_checkpoint_extra(
        self, cfg: SurgeryConfig, *, mapping: Dict[str, str], metadata_path: str
    ) -> Dict[str, Any]:
        return {**self._run_record(cfg), "meta_ref": os.path.basename(metadata_path), "mapping": mapping}

    # -- Task hooks (classification defaults; regression models override) -------------------
    # ``primary`` is a higher-is-better scalar (accuracy for classification, PSNR for SR), so
    # keep-best logic in surgery/distill is task-agnostic.

    def example_model_input(self, cfg: SurgeryConfig) -> Optional[torch.Tensor]:
        """Representative input for structure dumps; ``None`` uses the image default."""
        return None

    def primary_metric_name(self) -> str:
        return "accuracy"

    def next_stage_hint(self) -> str:
        return "Next: run the distill stage."

    def evaluate(self, model: nn.Module, val_loader: DataLoader, cfg: Any) -> Tuple[float, float, Dict[str, float]]:
        """Return ``(primary, loss, extra_metrics)`` for a standalone model on ``val_loader``."""
        acc, loss = accuracy_and_loss(model, val_loader, nn.CrossEntropyLoss())
        return acc, loss, {}

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


def get_model_adapter(model_key: str) -> SurgeryModelAdapter:
    return _ADAPTERS[model_key]


def build_surgery_student(
    extra: Mapping[str, Any], cfg: Any, surgery_dtype: Optional[torch.dtype] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Surgery student architecture of a checkpoint record ``extra``; ``cfg.top_k`` / ``cfg.eps`` override it when set.

    Sets the surgery dtype to ``surgery_dtype`` or the checkpoint's. Returns the model (weights not loaded) and the
    record with the overrides applied.
    """
    extra = dict(extra)
    if cfg.top_k is not None:
        extra["top_k"] = int(cfg.top_k)
    if cfg.eps is not None:
        extra["eps"] = float(cfg.eps)
    set_surgery_dtype(torch_dtype_from_name(extra["surgery_dtype"]) if surgery_dtype is None else surgery_dtype)
    extra["surgery_dtype"] = describe_dtype(get_surgery_dtype())
    surgery_cfg = dataclass_from_record(SurgeryConfig, extra)
    model = get_model_adapter(extra["model_key"]).build_surgery_model(surgery_cfg, extra)
    return model.to(device=get_device(), dtype=get_surgery_dtype()), extra


def load_surgery_student_checkpoint(
    path: str, cfg: Any, *, surgery_dtype: Optional[torch.dtype] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Rebuild a surgery student from its checkpoint (see :func:`build_surgery_student`) and load its weights."""
    payload = torch.load(path, map_location=get_device(), weights_only=False)
    model, extra = build_surgery_student(payload["extra"], cfg, surgery_dtype)
    install_routers_from_state_dict(model, payload["model_state_dict"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    get_model_adapter(extra["model_key"]).freeze_surgery_parameters(model)
    return model, extra


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
