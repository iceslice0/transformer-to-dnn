"""Model adapter for MambaIRv2 Light SR (super-resolution regression).

Overrides the task hooks (L1 loss, Y-channel PSNR/SSIM, teacher-L1 matching) so the shared
``ts-surgery`` / ``ts-distill`` stages run the regression pipeline unchanged. All heavy imports
(the installed MambaIR ``basicsr`` arch, which needs ``mamba-ssm`` + CUDA) are lazy so this
adapter can be *registered* on any machine.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformer_surgery.internal.util import get_device, get_surgery_dtype, maybe_surgery_cuda_autocast
from transformer_surgery.models.adapters import SurgeryModelAdapter


def _infer_scale_from_state(state: Dict[str, torch.Tensor]) -> int:
    """Recover the upscale factor from a stock MambaIRv2Light checkpoint (``pixelshuffledirect``)."""
    w = state.get("upsample.0.weight")
    if w is None:
        raise ValueError("Cannot infer SR scale: 'upsample.0.weight' missing from checkpoint")
    out_ch = int(w.shape[0])  # scale**2 * num_out_ch (num_out_ch == 3)
    scale = int(round(math.sqrt(out_ch / 3.0)))
    if scale * scale * 3 != out_ch:
        raise ValueError(f"Cannot infer SR scale from upsample out_channels={out_ch}")
    return scale


def _crop_border(model: nn.Module, cfg: Any = None) -> int:
    if cfg is not None and getattr(cfg, "scale", None):
        return int(cfg.scale)
    return int(getattr(model, "scale", None) or getattr(model, "upscale", 1))


class MambaIRLightSRAdapter(SurgeryModelAdapter):
    key = "mambair_lightsr"
    patient_name = "MambaIRv2-Light"
    dataset_name = "DIV2K/Set5"

    # -- construction / IO -----------------------------------------------------------------

    def build_loaders(self, cfg: Any) -> Tuple[DataLoader, DataLoader]:
        from transformer_surgery.models.mambair.loaders import build_mambair_loaders

        return build_mambair_loaders(cfg)

    def load_reference_checkpoint(self, path: str) -> nn.Module:
        from transformer_surgery.models.mambair.arch import build_mambair_lightsr

        device = get_device()
        path = os.path.abspath(path)
        payload = torch.load(path, map_location=device, weights_only=False)
        state = None
        for key in ("params_ema", "params", "state_dict"):
            if isinstance(payload, dict) and key in payload:
                state = payload[key]
                break
        if state is None:
            state = payload
        scale = _infer_scale_from_state(state)
        model = build_mambair_lightsr(scale).to(device)
        model.load_state_dict(state, strict=False)
        model.eval()
        return model

    def build_surgery_model(self, cfg: Any) -> nn.Module:
        from transformer_surgery.models.mambair.surgery_model import MambaIRLightSurgeryModel

        return MambaIRLightSurgeryModel.from_surgery_config(cfg)

    def build_surgery_model_from_extra(self, extra: Dict[str, Any], cfg: Any) -> nn.Module:
        from transformer_surgery.models.mambair.surgery_model import MambaIRLightSurgeryModel

        return MambaIRLightSurgeryModel.from_pretrained_extra(extra)

    def copy_reference_weights(self, student: nn.Module, reference: nn.Module) -> Dict[str, str]:
        return student.load_from_reference(reference)

    def freeze_surgery_parameters(self, model: nn.Module) -> None:
        from transformer_surgery.models.deit_tiny.surgery_model import freeze_eps_parameters

        freeze_eps_parameters(model)

    def calibrate_reference(self, reference: nn.Module, loader: DataLoader, cfg: Any) -> Dict[str, Any]:
        from transformer_surgery.internal.calibration import calibrate_softmax_layernorm_reference

        return calibrate_softmax_layernorm_reference(reference, loader, cfg)

    def build_module_mapping(self, cfg: Any, model: Optional[nn.Module] = None) -> Dict[str, str]:
        if cfg.disable_layernorm_replacement:
            ln = "nn.LayerNorm"
        elif cfg.allow_matmul:
            ln = "RewrittenLayerNorm(rsqrt.mul)"
        else:
            ln = "RewrittenLayerNorm(log/sqrt_exp)"
        if cfg.disable_attention_surgery:
            attn = "WindowAttention(vanilla scaled QK^T softmax @ V)"
        else:
            dot = "PairwiseDotBySquare(QK^T matmul)" if cfg.allow_matmul else "PairwiseDotBySquare(square identity)"
            if cfg.disable_softmax_replacement:
                attn = f"SurgeryWindowAttention({dot}+full_softmax+dense@V)"
            else:
                mix = (
                    "SparseWeightedSumBySquare(elementwise p*v)"
                    if cfg.allow_matmul
                    else "SparseWeightedSumBySquare(square identity)"
                )
                if bool(getattr(cfg, "use_exact_tail_mass", False)):
                    tail = "+exact_tail@V"
                elif bool(getattr(cfg, "disable_calib_gibbs_tail_prob", False)):
                    tail = "+initial_tail@V"
                else:
                    tail = "+calibrated_tail@V"
                attn = f"SurgeryWindowAttention({dot}+GibbsTopKSoftmax{tail}+{mix})"
        mapping: Dict[str, str] = {}
        if model is None:
            return {"WindowAttention.*": attn, "LayerNorm.*": ln}
        from transformer_surgery.models.mambair.surgery_model import SurgeryWindowAttention
        from transformer_surgery.ops import RewrittenLayerNorm

        uniform_attn = "SurgeryWindowAttention(AffineMean uniform key-mean @ V)"
        net = getattr(model, "net", model)
        for name, module in net.named_modules():
            if isinstance(module, SurgeryWindowAttention):
                mapping[name] = uniform_attn if getattr(module, "uniform", False) else attn
            elif isinstance(module, RewrittenLayerNorm):
                mapping[name] = ln
        return mapping

    # -- SR task hooks (override classification defaults) -----------------------------------

    @torch.no_grad()
    def evaluate(self, model: nn.Module, val_loader: DataLoader, cfg: Any = None) -> Tuple[float, float, Dict[str, float]]:
        from transformer_surgery.internal.sr_metrics import align_spatial, calc_psnr, calc_ssim

        device = get_device()
        model.eval()
        dt = get_surgery_dtype()
        use_cuda = device.type == "cuda"
        crop = _crop_border(model, cfg)
        psnr_sum = l1_sum = ssim_sum = 0.0
        n = 0
        for lq, hr in val_loader:
            lq = lq.to(device, dtype=dt, non_blocking=use_cuda)
            hr = hr.to(device, non_blocking=use_cuda)
            with maybe_surgery_cuda_autocast(device, dt):
                pred = model(lq)
            pred, hr_a = align_spatial(pred.float(), hr.float())
            bs = hr_a.shape[0]
            psnr_sum += float(calc_psnr(pred, hr_a, crop_border=crop).item()) * bs
            ssim_sum += float(calc_ssim(pred, hr_a, crop_border=crop).item()) * bs
            l1_sum += float(F.l1_loss(pred, hr_a).item()) * bs
            n += bs
        return psnr_sum / n, l1_sum / n, {"ssim": ssim_sum / n}

    @torch.no_grad()
    def eval_student_vs_teacher(
        self, teacher: nn.Module, student: nn.Module, val_loader: DataLoader, *, temperature: float = 1.0
    ) -> Tuple[float, float, float]:
        from transformer_surgery.internal.sr_metrics import align_spatial, calc_psnr

        device = get_device()
        teacher.eval()
        student.eval()
        dt = get_surgery_dtype()
        use_cuda = device.type == "cuda"
        crop = _crop_border(student)
        psnr_sum = l1_sum = match_sum = 0.0
        n = 0
        for lq, hr in val_loader:
            lq = lq.to(device, dtype=dt, non_blocking=use_cuda)
            hr = hr.to(device, non_blocking=use_cuda)
            with maybe_surgery_cuda_autocast(device, dt):
                s = student(lq).float()
                t = teacher(lq).float()
            s, hr_a = align_spatial(s, hr.float())
            t, _ = align_spatial(t, hr.float())
            bs = hr_a.shape[0]
            psnr_sum += float(calc_psnr(s, hr_a, crop_border=crop).item()) * bs
            l1_sum += float(F.l1_loss(s, hr_a).item()) * bs
            match_sum += float(F.l1_loss(s, t).item()) * bs
            n += bs
        return psnr_sum / n, l1_sum / n, match_sum / n

    def distill_step_losses(
        self, student_out: torch.Tensor, teacher_out: torch.Tensor, target: torch.Tensor, *, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from transformer_surgery.internal.sr_metrics import align_spatial

        s, hr = align_spatial(student_out, target.to(student_out.dtype))
        _, t = align_spatial(student_out, teacher_out)
        t = t.detach()
        hard = F.l1_loss(s, hr)
        match = F.l1_loss(s, t)
        return hard, match
