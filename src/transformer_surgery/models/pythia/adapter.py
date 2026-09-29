"""Model adapter for EleutherAI Pythia-70M on WikiText-2 (surgery-only causal LM)."""

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


def _require_llm_deps() -> None:
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Pythia support requires optional deps. Install with: pip install -e '.[llm]'"
        ) from exc


class Pythia70MWikiText2Adapter(SurgeryModelAdapter):
    key = "pythia_70m_wikitext2"
    patient_name = "Pythia-70M"
    dataset_name = "WikiText-2"

    def reference_checkpoint_path(self, cfg: Any) -> str:
        path = getattr(cfg, "reference_checkpoint", None)
        if path is not None and str(path).strip():
            raw = str(path).strip()
            # Local checkpoints get abspath; Hugging Face hub ids stay as-is.
            if os.path.isdir(raw) or os.path.isfile(raw) or os.path.isdir(os.path.abspath(raw)) or os.path.isfile(os.path.abspath(raw)):
                return os.path.abspath(raw)
            return raw
        return str(getattr(cfg, "hf_model_id", "EleutherAI/pythia-70m"))

    def build_loaders(self, cfg: Any) -> Tuple[DataLoader, DataLoader]:
        from transformer_surgery.models.pythia.loaders import build_wikitext2_loaders

        return build_wikitext2_loaders(cfg)

    def load_reference_checkpoint(self, path: str) -> nn.Module:
        _require_llm_deps()
        from transformers import GPTNeoXForCausalLM

        device = get_device()
        model_id = str(path)
        if os.path.isdir(model_id) or os.path.isfile(model_id):
            model_id = os.path.abspath(model_id)
        model = GPTNeoXForCausalLM.from_pretrained(model_id, attn_implementation="eager")
        model.to(device)
        model.eval()
        return model

    def _arch_from_reference_or_cfg(self, cfg: Any, reference: Optional[nn.Module] = None) -> Dict[str, Any]:
        from transformer_surgery.models.pythia.surgery_model import (
            PYTHIA_70M_DEFAULTS,
            arch_dict_from_hf_config,
        )

        if reference is not None and hasattr(reference, "config"):
            return arch_dict_from_hf_config(reference.config)
        hf_id = str(getattr(cfg, "hf_model_id", "EleutherAI/pythia-70m"))
        try:
            _require_llm_deps()
            from transformers import AutoConfig

            return arch_dict_from_hf_config(AutoConfig.from_pretrained(hf_id))
        except Exception:
            return dict(PYTHIA_70M_DEFAULTS)

    def build_surgery_model(self, cfg: Any) -> nn.Module:
        from transformer_surgery.models.pythia.surgery_model import PythiaSurgeryModel

        arch = self._arch_from_reference_or_cfg(cfg)
        return PythiaSurgeryModel.from_surgery_config(cfg, arch=arch)

    def build_surgery_model_from_extra(self, extra: Dict[str, Any], cfg: Any) -> nn.Module:
        from transformer_surgery.models.pythia.surgery_model import PythiaSurgeryModel

        return PythiaSurgeryModel.from_pretrained_extra(extra, cfg)

    def copy_reference_weights(self, student: nn.Module, reference: nn.Module) -> Dict[str, str]:
        return student.load_from_reference(reference)

    def freeze_surgery_parameters(self, model: nn.Module) -> None:
        from transformer_surgery.models.pythia.surgery_model import freeze_pythia_eps_parameters

        freeze_pythia_eps_parameters(model)

    def calibrate_reference(self, reference: nn.Module, loader: DataLoader, cfg: Any) -> Dict[str, Any]:
        """
        Causal LM: scalar per-block ``gibbs_tail_prob_eps`` is incorrect under causal masks
        (rows have different valid key counts), so this adapter always disables scalar tail
        application. Exact omitted-tail diagnostics are gathered on the built student via
        :meth:`calibrate_after_build` (shared ``calibrate_surgery_student`` path).
        """
        use_exact = bool(getattr(cfg, "use_exact_tail_mass", False))
        return {
            "disable_calib_gibbs_tail_prob": True,
            "use_exact_tail_mass": use_exact,
            "gibbs_tail_prob_eps": float(cfg.gibbs_tail_prob_eps),
            "gibbs_tail_prob_eps_configured": float(cfg.gibbs_tail_prob_eps),
            "calibration_mode": "pythia_causal_pending_student",
            "calibration_loader_split": "train",
        }

    def calibrate_after_build(self, model: nn.Module, loader: DataLoader, cfg: Any) -> Dict[str, Any]:
        from transformer_surgery.internal.calibration import calibrate_surgery_student

        stats = calibrate_surgery_student(model, loader, cfg)
        # Never apply calibrated scalar tails on causal rows; keep exact-runtime / dense only.
        stats["disable_calib_gibbs_tail_prob"] = True
        stats["calibration_mode"] = "pythia_surgery_student_causal"
        stats["calibration_loader_split"] = "train"
        return stats

    def apply_calibration(self, model: nn.Module, calibration: Dict[str, Any]) -> Dict[str, Any]:
        # Causal rows need exact runtime tail mass (or dense top_k=seq_len); never copy scalars.
        return {}

    def build_module_mapping(self, cfg: Any, model: Optional[nn.Module] = None) -> Dict[str, str]:
        if cfg.disable_layernorm_replacement:
            ln = "nn.LayerNorm"
        elif cfg.allow_matmul:
            ln = "RewrittenLayerNorm(rsqrt.mul)"
        else:
            ln = "RewrittenLayerNorm(log/sqrt_exp)"
        if cfg.disable_attention_surgery:
            attn = "SurgeryCausalAttention(vanilla causal scaled QK^T softmax @ V)"
        else:
            dot = "PairwiseDotBySquare(QK^T matmul)" if cfg.allow_matmul else "PairwiseDotBySquare(square identity)"
            if cfg.disable_softmax_replacement:
                attn = f"SurgeryCausalAttention({dot}+causal_mask+full_softmax+dense@V)"
            else:
                mix = (
                    "SparseWeightedSumBySquare(elementwise p*v)"
                    if cfg.allow_matmul
                    else "SparseWeightedSumBySquare(square identity)"
                )
                if bool(getattr(cfg, "use_exact_tail_mass", False)):
                    tail = "+exact_causal_tail@V"
                else:
                    tail = "+dense_or_zero_tail@V"
                attn = f"SurgeryCausalAttention({dot}+causal_mask+GibbsTopKSoftmax{tail}+{mix})"
        mapping: Dict[str, str] = {"*.attention": attn, "*.layernorm": ln}
        if model is None:
            return mapping
        from transformer_surgery.ops import RewrittenLayerNorm, SurgeryCausalAttention

        for name, module in model.named_modules():
            if isinstance(module, SurgeryCausalAttention):
                mapping[name] = attn
            elif isinstance(module, RewrittenLayerNorm):
                mapping[name] = ln
        return mapping

    def pre_ft_checkpoint_extra(self, cfg: Any, *, mapping: Dict[str, Any], metadata_path: str) -> Dict[str, Any]:
        ex = super().pre_ft_checkpoint_extra(cfg, mapping=mapping, metadata_path=metadata_path)
        arch = self._arch_from_reference_or_cfg(cfg)
        arch["seq_len"] = int(cfg.context_length)
        arch["context_length"] = int(cfg.context_length)
        ex["arch"] = arch
        ex["context_length"] = int(cfg.context_length)
        ex["hf_model_id"] = str(getattr(cfg, "hf_model_id", "EleutherAI/pythia-70m"))
        ex["use_exact_tail_mass"] = bool(getattr(cfg, "use_exact_tail_mass", False))
        ex["disable_layernorm_replacement"] = bool(cfg.disable_layernorm_replacement)
        ex["disable_attention_surgery"] = bool(cfg.disable_attention_surgery)
        ex["disable_softmax_replacement"] = bool(cfg.disable_softmax_replacement)
        ex["allow_matmul"] = bool(cfg.allow_matmul)
        ex["top_k"] = int(cfg.top_k)
        ex["gibbs_tail_prob_eps"] = float(cfg.gibbs_tail_prob_eps)
        return ex

    def example_model_input(self, cfg: Any = None) -> torch.Tensor:
        ctx = int(getattr(cfg, "context_length", 128)) if cfg is not None else 128
        return torch.zeros((1, ctx), dtype=torch.long)

    def primary_metric_name(self) -> str:
        return "neg_nll"

    def next_stage_hint(self) -> str:
        return "Surgery-only causal LM complete (no distill stage)."

    @torch.no_grad()
    def evaluate(self, model: nn.Module, val_loader: DataLoader, cfg: Any = None) -> Tuple[float, float, Dict[str, float]]:
        device = get_device()
        model.eval()
        dt = get_surgery_dtype()
        use_cuda = device.type == "cuda"
        nll_sum = torch.zeros((), device=device, dtype=torch.float64)
        n_tokens = 0
        for x, y in val_loader:
            x = x.to(device, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)
            with maybe_surgery_cuda_autocast(device, dt):
                if hasattr(model, "gpt_neox"):
                    logits = model(input_ids=x, use_cache=False).logits
                else:
                    logits = model(x)
            # logits/labels are next-token aligned windows of equal length.
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                y.reshape(-1),
                reduction="sum",
            )
            nll_sum += loss.double()
            n_tokens += int(y.numel())
        nll = float(nll_sum.item() / max(n_tokens, 1))
        ppl = float(math.exp(min(nll, 100.0)))
        # primary is higher-is-better for the shared surgery driver.
        return -nll, nll, {"nll": nll, "perplexity": ppl, "tokens": float(n_tokens)}
