"""Node-level check of Brent's operand scaling under PTQ-style integer quantization.

Captures the float64 inputs of the strict Score (q, k), value mix (probs, idx, v) and block LayerNorms (x) of
DeiT-Tiny on Pet, and evaluates each product site of the norm-expansion construction the way PTQ treats it:
every input of a compiled affine node is quantized on its own symmetric max-abs grid of b bits (ranges from the
float values on 8 calibration images, clipped on 16 others), accumulation is exact, the fixed coefficients are
exact per-input multipliers, and a least-squares affine map is fitted to the site output on the calibration images.

  Score      q k^T = 1/2 [ sum_d (lam q + k/lam)^2 - lam^2 |q|^2 - |k|^2 / lam^2 ]
  Mix        sum_k p v = 1/2 [ sum_k ( v^2/lam^2 - (v/lam - lam p)^2 ) + lam^2 sum_k p^2 ]
  LayerNorm  u a = 1/2 [ (lam u + a/lam)^2 - lam^2 u^2 - a^2 / lam^2 ],  a = 1/sqrt(var + eps)

lam = 1 is the strict graph; with Brent's scaling lam^2 = max|b| / max|a| (b = k, v, a for the three
sites) from the same calibration ranges, one scalar per site and block; brent_pow2 rounds lam to a power of two, as
the PTQ option brent_scaling does. Reports ||y_hat - y|| / ||y|| on the 16
evaluation images, averaged over blocks, for b in {8, 10, 12}.

Usage: python scripts/ptq_balance_check.py   (uses ts_surgery_topk{16,197}_strict_sr_gs.pt)
"""

from __future__ import annotations

import json
import math
import os
import warnings

import torch

from transformer_surgery.cli.ptq_config import PTQSurgeryConfig
from transformer_surgery.internal.util import set_default_device, set_surgery_dtype
from transformer_surgery.models.adapters import get_model_adapter, load_surgery_student_checkpoint

BITS = (8, 10, 12)
CHUNK = 4  # images per evaluation chunk (the Score grid of 4 images is 238 MB in float64)


class Quant:
    """Max-abs ranges per quantization point (calibration), then symmetric fake quantization with b bits."""

    def __init__(self) -> None:
        self.max: dict = {}
        self.bits = None  # None: calibration mode, values pass unchanged

    def __call__(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if self.bits is None:
            self.max[name] = max(self.max.get(name, 0.0), x.abs().max().item())
            return x
        qmax = 2 ** (self.bits - 1) - 1
        s = self.max[name] / qmax
        return x if s == 0.0 else torch.clamp(torch.round(x / s), -qmax, qmax) * s


def gather(v: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    nq, d = idx.shape[2], v.shape[-1]
    return v.unsqueeze(2).expand(-1, -1, nq, -1, -1).gather(3, idx.unsqueeze(-1).expand(*idx.shape, d))


def score(q, k, lam, Q):
    s = lam * Q("q", q).unsqueeze(3) + Q("k", k).unsqueeze(2) / lam                 # sum_qk
    t = Q("s2", s * s).sum(-1)                                                       # combine input 0, summed over d
    nq = Q("nq", Q("q2", q * q).sum(-1))                                             # norm_q -> combine input 1
    nk = Q("nk", Q("k2", k * k).sum(-1))                                             # norm_k -> combine input 2
    return 0.5 * (t - lam**2 * nq.unsqueeze(-1) - nk.unsqueeze(-2) / lam**2)


def score_ref(q, k):
    return q @ k.transpose(-2, -1)


def mix(p, idx, v, lam, Q):
    vg, pb = gather(v, idx), p.unsqueeze(-1)
    dvp = Q("v", vg) / lam - lam * Q("p", pb)                                        # sub_vp
    t = Q("v2", vg * vg) / lam**2 - Q("dvp2", dvp * dvp)                             # diff_sum inputs
    t = Q("dsum", t.sum(-2))                                                         # diff_sum -> combine input 0
    sp = Q("sp2", Q("p2", p * p).sum(-1, keepdim=True))                              # sum_p2 -> combine input 1
    return 0.5 * (t + lam**2 * sp)


def mix_ref(p, idx, v):
    return (p.unsqueeze(-1) * gather(v, idx)).sum(-2)


def layernorm(x, eps, lam, Q):
    xq = Q("x", x)                                                                   # mean_u, u_center_contract
    u = xq - xq.mean(-1, keepdim=True)
    u2 = u * u
    a = torch.rsqrt(Q("u2", u2).mean(-1, keepdim=True) + eps).expand_as(u)          # mean_r2 -> rsqrt
    s = lam * Q("u", u) + Q("a", a) / lam                                            # add_u_invstd
    return 0.5 * (Q("s2", s * s) - lam**2 * Q("u2", u2) - Q("a2", a * a) / lam**2)  # combine_z


def layernorm_ref(x, eps):
    u = x - x.mean(-1, keepdim=True)
    return u * torch.rsqrt((u * u).mean(-1, keepdim=True) + eps)


def chunks(args):
    n = args[0].shape[0]
    return [tuple(a[i:i + CHUNK] for a in args) for i in range(0, n, CHUNK)]


def site_errors(fn, ref, fit_args, ev_args, lam_pair, extra=()):
    """Relative error per bit width for lam = 1 and for Brent's lam; lam_pair(Q) gives lam from float ranges."""
    out = {}
    q0 = Quant()
    for a in chunks(fit_args):
        fn(*a, *extra, 1.0, q0)
    lam_b = lam_pair(q0.max)
    for label, lam in (("plain", 1.0), ("brent", lam_b), ("brent_pow2", 2.0 ** round(math.log2(lam_b)))):
        Q = Quant()
        for a in chunks(fit_args):
            fn(*a, *extra, lam, Q)
        for b in BITS:
            Q.bits = b
            xs, ys = [], []
            for a in chunks(fit_args):
                xs.append(fn(*a, *extra, lam, Q).flatten()); ys.append(ref(*a, *extra).flatten())
            xh, y = torch.cat(xs), torch.cat(ys)
            A = torch.stack([xh, torch.ones_like(xh)], 1)
            alpha, beta = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.flatten().tolist()  # PTQ output fit
            num = den = 0.0
            for a in chunks(ev_args):
                y = ref(*a, *extra)
                num += ((alpha * fn(*a, *extra, lam, Q) + beta - y) ** 2).sum().item()
                den += (y**2).sum().item()
            out[f"{label}_{b}"] = math.sqrt(num / den)
            Q.bits = None
        out[f"{label}_lam"] = lam
    # Exactness of the emulation without quantization: both lam reproduce the reference.
    for lam in (1.0, lam_pair(q0.max)):
        a = chunks(ev_args)[0]
        y = ref(*a, *extra)
        assert ((fn(*a, *extra, lam, Quant()) - y).norm() / y.norm()).item() < 1e-10
    return out


def main() -> None:
    warnings.simplefilter("ignore", FutureWarning)
    dev = torch.device("cuda")
    set_default_device(dev)
    cfg = PTQSurgeryConfig()
    cfg.top_k, cfg.eps, cfg.batch_size, cfg.workers = None, None, 8, 0
    _, val = get_model_adapter("deit_tiny_pet").build_loaders(cfg)
    it = iter(val)
    x_fit = next(it)[0].to(dev, torch.float64)
    x_eval = torch.cat([next(it)[0], next(it)[0]]).to(dev, torch.float64)
    result = {"note": "generated by scripts/ptq_balance_check.py", "bits": list(BITS), "sites": {}}
    for k in (16, 197):
        set_surgery_dtype(torch.float64)
        model, _ = load_surgery_student_checkpoint(f"artifacts/checkpoints/ts_surgery_topk{k}_strict_sr_gs.pt", cfg)
        model = model.to(dev, torch.float64).eval()

        def capture(x):
            caps = {"dot": [], "mix": [], "ln": []}
            hs = []
            for blk in model.blocks:
                hs.append(blk.attn.dot.register_forward_pre_hook(lambda m, a: caps["dot"].append(a)))
                hs.append(blk.attn.sparse_mix.register_forward_pre_hook(lambda m, a: caps["mix"].append(a)))
                for ln in (blk.norm1, blk.norm2):
                    hs.append(ln.register_forward_pre_hook(lambda m, a, ln=ln: caps["ln"].append((a[0], ln.inv_sqrt_var.eps.item()))))
            with torch.no_grad():
                model(x)
            for h in hs:
                h.remove()
            return caps

        fit, ev = capture(x_fit), capture(x_eval)
        sites = {"mix": (mix, mix_ref, lambda m: math.sqrt(m["v"] / m["p"]))}
        if k == 16:  # Score and LayerNorm do not depend on k
            sites["score"] = (score, score_ref, lambda m: math.sqrt(m["k"] / m["q"]))
            sites["layernorm"] = (layernorm, layernorm_ref, lambda m: math.sqrt(m["a"] / m["u"]))
        with torch.no_grad():
            for name, (fn, ref, lam_pair) in sites.items():
                rows = []
                key = {"score": "dot", "mix": "mix", "layernorm": "ln"}[name]
                for fa, ea in zip(fit[key], ev[key]):
                    if name == "layernorm":
                        rows.append(site_errors(fn, ref, (fa[0],), (ea[0],), lam_pair, extra=(fa[1],)))
                    else:
                        rows.append(site_errors(fn, ref, tuple(fa), tuple(ea), lam_pair))
                avg = {c: sum(r[c] for r in rows) / len(rows) for c in rows[0]}
                label = f"{name}_k{k}" if name == "mix" else name
                result["sites"][label] = {"mean": avg, "by_block": rows}
                print(f"{label:10s} lam^2 median {sorted(r['brent_lam'] ** 2 for r in rows)[len(rows) // 2]:8.3g}  "
                      + "  ".join(f"{b}b: {100 * avg[f'plain_{b}']:.3f}% -> {100 * avg[f'brent_{b}']:.3f}% "
                                  f"(pow2 {100 * avg[f'brent_pow2_{b}']:.3f}%)" for b in BITS),
                      flush=True)
    os.makedirs("artifacts/metadata", exist_ok=True)
    with open("artifacts/metadata/ptq_balance_check.json", "w") as f:
        json.dump(result, f, indent=1)
    print("wrote artifacts/metadata/ptq_balance_check.json")


if __name__ == "__main__":
    main()
