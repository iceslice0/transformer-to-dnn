"""Loss and diagnostic metrics used by surgery/distillation processes."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _jeffreys_metric_eps(dt: torch.dtype) -> Tuple[float, float]:
    f = torch.finfo(dt)
    smn = float(f.smallest_normal)
    sms = float(getattr(f, "smallest_subnormal", smn))
    return (max(1e-12, sms), max(1e-30, smn))


def _kl_safe(p: torch.Tensor, q: torch.Tensor, eps: Optional[float] = None) -> torch.Tensor:
    if eps is None:
        eps, _ = _jeffreys_metric_eps(p.dtype)
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
    t = teacher_logits.to(dtype=loss_dtype) / temperature
    s = student_logits.to(dtype=loss_dtype) / temperature
    p = F.softmax(t, dim=-1)
    q = F.softmax(s, dim=-1)
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
    tail_prob = float(gibbs_tail_prob_eps)
    if not 0.0 <= tail_prob < 1.0:
        raise ValueError("gibbs_tail_prob_eps must be in [0, 1)")
    t = teacher_logits - teacher_logits.max(dim=-1, keepdim=True).values
    p = F.softmax(t, dim=-1)
    _, denom_eps = _jeffreys_metric_eps(teacher_logits.dtype)

    exp_vals = torch.exp(vals)
    q_dense = torch.zeros_like(p)
    tail_mass = tail_prob if nk > k else 0.0
    z_top = exp_vals.sum(dim=-1, keepdim=True)
    q_on_i = (1.0 - tail_mass) * exp_vals / (z_top + denom_eps)
    if nk > k:
        q_dense = torch.full_like(p, tail_mass / float(nk - k))
    q_dense.scatter_(1, idx, q_on_i)

    return _kl_safe(p, q_dense) + _kl_safe(q_dense, p)


def jeffreys_naive_topk(
    teacher_logits: torch.Tensor,
    vals: torch.Tensor,
    idx: torch.Tensor,
    nk: int,
    k: int,
) -> torch.Tensor:
    t = teacher_logits - teacher_logits.max(dim=-1, keepdim=True).values
    p = F.softmax(t, dim=-1)
    kl_eps, denom_eps = _jeffreys_metric_eps(teacher_logits.dtype)
    exp_vals = torch.exp(vals)
    z_sub = exp_vals.sum(dim=-1, keepdim=True)
    q_sub = exp_vals / (z_sub + denom_eps)
    q_dense = torch.zeros_like(p)
    q_dense.scatter_(1, idx, q_sub)
    return _kl_safe(p, q_dense.clamp_min(kl_eps)) + _kl_safe(q_dense.clamp_min(kl_eps), p)
