"""Jeffreys divergences: dense for distillation, sparse top-k (Gibbs tail and naive) for calibration diagnostics."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def logit_stable_rows(logits: torch.Tensor) -> torch.Tensor:
    """Row-wise max-subtracted logits (stable softmax input)."""
    return logits - logits.max(dim=-1, keepdim=True).values


def _metric_eps(dt: torch.dtype) -> Tuple[float, float]:
    """Probability floor of the KL terms and the normalizer offset for ``dt``."""
    tiny = torch.finfo(dt).smallest_normal
    return max(1e-12, tiny), max(1e-30, tiny)


def _kl_safe(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    eps, _ = _metric_eps(p.dtype)
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


def jeffreys_divergence_dense(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    loss_dtype = torch.promote_types(teacher_logits.dtype, student_logits.dtype)
    if loss_dtype in (torch.float16, torch.bfloat16):
        loss_dtype = torch.float32
    p = F.softmax(teacher_logits.to(dtype=loss_dtype) / temperature, dim=-1)
    q = F.softmax(student_logits.to(dtype=loss_dtype) / temperature, dim=-1)
    return _kl_safe(p, q) + _kl_safe(q, p)


def jeffreys_distance_sparse_teacher(
    teacher_logits: torch.Tensor,
    vals: torch.Tensor,
    idx: torch.Tensor,
    nk: int,
    k: int,
    *,
    gibbs_tail_prob_eps: float,
) -> torch.Tensor:
    """Jeffreys divergence of the dense softmax and the Gibbs top-k distribution with tail mass spread over the rest."""
    tail_prob = float(gibbs_tail_prob_eps)
    if not 0.0 <= tail_prob < 1.0:
        raise ValueError("gibbs_tail_prob_eps must be in [0, 1)")
    p = F.softmax(logit_stable_rows(teacher_logits), dim=-1)
    _, denom_eps = _metric_eps(teacher_logits.dtype)
    exp_vals = torch.exp(vals)
    tail_mass = tail_prob if nk > k else 0.0
    q_on_i = (1.0 - tail_mass) * exp_vals / (exp_vals.sum(dim=-1, keepdim=True) + denom_eps)
    q_dense = torch.full_like(p, tail_mass / float(nk - k)) if nk > k else torch.zeros_like(p)
    q_dense.scatter_(1, idx, q_on_i)
    return _kl_safe(p, q_dense) + _kl_safe(q_dense, p)


def jeffreys_naive_topk(
    teacher_logits: torch.Tensor,
    vals: torch.Tensor,
    idx: torch.Tensor,
    nk: int,
    k: int,
) -> torch.Tensor:
    """Jeffreys divergence of the dense softmax and the top-k softmax renormalized without a tail."""
    p = F.softmax(logit_stable_rows(teacher_logits), dim=-1)
    kl_eps, denom_eps = _metric_eps(teacher_logits.dtype)
    exp_vals = torch.exp(vals)
    q_dense = torch.zeros_like(p)
    q_dense.scatter_(1, idx, exp_vals / (exp_vals.sum(dim=-1, keepdim=True) + denom_eps))
    return _kl_safe(p, q_dense.clamp_min(kl_eps)) + _kl_safe(q_dense.clamp_min(kl_eps), p)
