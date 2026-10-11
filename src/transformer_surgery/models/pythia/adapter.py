"""Model adapter for EleutherAI Pythia-70M on WikiText-2 (surgery-only causal LM)."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformer_surgery.cli.surgery_config import SurgeryConfig
from transformer_surgery.internal.util import get_device, get_surgery_dtype, maybe_surgery_cuda_autocast
from transformer_surgery.models.adapters import SurgeryModelAdapter


class Pythia70MWikiText2Adapter(SurgeryModelAdapter):
    key = "pythia_70m_wikitext2"
    patient_name = "Pythia-70M"
    dataset_name = "WikiText-2"

    def reference_checkpoint_path(self, cfg: Any) -> str:
        """Local checkpoint (absolute path) or Hugging Face hub id; defaults to ``cfg.hf_model_id``."""
        path = (cfg.reference_checkpoint or cfg.hf_model_id).strip()
        return os.path.abspath(path) if os.path.exists(path) else path

    def build_loaders(self, cfg: Any) -> Tuple[DataLoader, DataLoader]:
        from transformer_surgery.models.pythia.loaders import build_wikitext2_loaders

        return build_wikitext2_loaders(cfg)

    def load_reference_checkpoint(self, path: str) -> nn.Module:
        from transformer_surgery.models.pythia.loaders import require_llm_deps

        require_llm_deps()
        from transformers import GPTNeoXForCausalLM

        return GPTNeoXForCausalLM.from_pretrained(path, attn_implementation="eager").to(get_device()).eval()

    def _arch(self, cfg: SurgeryConfig) -> Dict[str, Any]:
        from transformers import AutoConfig

        from transformer_surgery.models.pythia.surgery_model import arch_dict_from_hf_config

        return arch_dict_from_hf_config(AutoConfig.from_pretrained(cfg.hf_model_id))

    def build_surgery_model(self, cfg: SurgeryConfig, extra: Optional[Mapping[str, Any]] = None) -> nn.Module:
        from transformer_surgery.models.pythia.surgery_model import PythiaSurgeryModel

        return PythiaSurgeryModel(cfg, extra["arch"] if extra else self._arch(cfg))

    def reference_key(self, key: str) -> str:
        key = super().reference_key(key)
        return key if key.startswith("lm_head.") else f"gpt_neox.{key}"

    def calibrate_reference(self, reference: nn.Module, loader: DataLoader, cfg: SurgeryConfig) -> Dict[str, Any]:
        """
        Causal LM: scalar per-block ``gibbs_tail_prob_eps`` is incorrect under causal masks
        (rows have different valid key counts), so this adapter always disables scalar tail
        application. Exact omitted-tail diagnostics are gathered on the built student via
        :meth:`calibrate_after_build`.
        """
        return {
            "disable_calib_gibbs_tail_prob": True,
            "use_exact_tail_mass": cfg.use_exact_tail_mass,
            "gibbs_tail_prob_eps": cfg.gibbs_tail_prob_eps,
            "gibbs_tail_prob_eps_configured": cfg.gibbs_tail_prob_eps,
            "calibration_mode": "pythia_causal_pending_student",
            "calibration_loader_split": "train",
        }

    def calibrate_after_build(self, model: nn.Module, loader: DataLoader, cfg: SurgeryConfig) -> Dict[str, Any]:
        from transformer_surgery.internal.calibration import calibrate_surgery_student

        stats = calibrate_surgery_student(model, loader, cfg)
        # Never apply calibrated scalar tails on causal rows; keep exact-runtime / dense only.
        stats["disable_calib_gibbs_tail_prob"] = True
        stats["calibration_mode"] = "pythia_surgery_student_causal"
        return stats

    def apply_calibration(self, model: nn.Module, calibration: Mapping[str, Any]) -> Dict[str, Any]:
        # Causal rows need exact runtime tail mass (or dense top_k=seq_len); never copy scalars.
        return {}

    def pre_ft_checkpoint_extra(
        self, cfg: SurgeryConfig, *, mapping: Dict[str, str], metadata_path: str
    ) -> Dict[str, Any]:
        extra = super().pre_ft_checkpoint_extra(cfg, mapping=mapping, metadata_path=metadata_path)
        return {**extra, "arch": self._arch(cfg)}

    def example_model_input(self, cfg: SurgeryConfig) -> torch.Tensor:
        return torch.zeros((1, cfg.context_length), dtype=torch.long)

    def primary_metric_name(self) -> str:
        return "neg_nll"

    def next_stage_hint(self) -> str:
        return "Surgery-only causal LM complete (no distill stage)."

    @torch.no_grad()
    def evaluate(self, model: nn.Module, val_loader: DataLoader, cfg: Any) -> Tuple[float, float, Dict[str, float]]:
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
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum")
            nll_sum += loss.double()
            n_tokens += int(y.numel())
        nll = float(nll_sum.item() / max(n_tokens, 1))
        ppl = float(math.exp(min(nll, 100.0)))
        # primary is higher-is-better for the shared surgery driver.
        return -nll, nll, {"nll": nll, "perplexity": ppl, "tokens": float(n_tokens)}
