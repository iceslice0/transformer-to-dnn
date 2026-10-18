"""Check every measured number in the paper against run metadata.

Parses the result tables (Tables 2-6, S1-S4) and the Figure 2 coordinates of the paper sources, recomputes each
value from ``artifacts/metadata`` and the profile JSON, and checks the numeric claims of the prose (ranges,
ratios, comparisons). Exit status 1 if anything differs.

Example:
    python scripts/check_paper_numbers.py --sections ../../tex/nips2026/<paper>/sections
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys


FIXES: list = []  # (file, raw row text, cell index, token index, expected value) with --fix


def _fmt_like(old: str, value: float) -> str:
    """``value`` with the decimals and sign style of the numeric token ``old``."""
    dec = len(old.split(".")[1]) if "." in old else 0
    txt = f"{value:.{dec}f}"
    if txt.lstrip("-").strip("0.") == "":  # rounds to zero: no sign
        txt = txt.lstrip("-")
        return txt
    if old[0] in "+-" and not txt.startswith("-"):
        txt = "+" + txt
    return txt


def apply_fixes(PAPER: str) -> int:
    """Write the expected values of FIXES into the table rows of the paper sources."""
    by_row: dict = {}
    for f, raw, i, t, v in FIXES:
        by_row.setdefault((f, raw), []).append((i, t, v))
    for (f, raw), fx in by_row.items():
        cells = raw.split("&")
        for i, t, v in fx:
            toks = list(re.finditer(r"[-+]?\d+\.?\d*", cells[i]))
            m = toks[t]
            cells[i] = cells[i][:m.start()] + _fmt_like(m.group(0), v) + cells[i][m.end():]
        path = f"{PAPER}/{f}"; text = open(path).read()
        assert text.count(raw) == 1, (f, raw[:60])
        open(path, "w").write(text.replace(raw, "&".join(cells)))
    return len(FIXES)


def check_tables(PAPER: str, REPO: str, M: str, fix: bool = False) -> int:
    src = {f: open(f"{PAPER}/{f}").read() for f in ("results_experiments.tex", "supplementary.tex")}
    def J(n): return json.load(open(f"{M}/{n}.json"))
    def S(n): return J(f"ts_surgery_{n}")["calibration"]
    def D(k, l): return J(f"ts_distill_{k}_fast_{l}")["calibration"]
    origin: dict = {}
    def table_rows(label):
        f = next(f for f, t in src.items() if f"\\label{{{label}}}" in t); tex = src[f]
        i = tex.index(f"\\label{{{label}}}"); j = tex.index("\\end{tabular}", i)
        body = tex[tex.index("\\midrule", i):j]
        rows = []
        for raw in body.split("\\\\"):
            line = raw.replace("\\midrule", "").strip()
            if not line or line.startswith("\\bottomrule"): continue
            cells = [c.strip() for c in line.split("&")]
            origin[id(cells)] = (f, raw.split("\\midrule")[-1])
            rows.append(cells)
        return rows
    def num(c):
        c = c.replace("$", "").replace("\\times", "").replace("\\,MB", "").replace("MB", "").strip()
        m = re.findall(r"[-+]?\d+\.?\d*", c); return [float(x) for x in m]
    bad = 0; checked = 0
    def cmp(where, got, exp, tol, at=None):
        nonlocal bad, checked
        checked += 1
        if exp is None and got is None: return
        if exp is None or got is None or abs(got - exp) > tol:
            bad += 1; print(f"MISMATCH {where}: paper {got} expected {exp}")
            if fix and at is not None and exp is not None:
                cells, i, t = at; f, raw = origin[id(cells)]; FIXES.append((f, raw, i, t, exp))
    r2 = lambda x: round(x, 2)
    sel = lambda d: 100 * d["student_post_distill_val_acc"]  # the saved run: highest training-split primary metric
    # Table 2
    support = "topk"
    for cells in table_rows("tab:strict-fast"):
        if cells[0].startswith("\\multicolumn"):
            support = "routed" if "Routed" in cells[0] else "topk"
            continue
        k = int(cells[0])
        suffixes = (
            ("fast", "hadamard_sr_gs", "strict_sr_gs", "strict_sr_gs_pol")
            if support == "topk"
            else ("fast_qc32", "hadamard_sr_gs_qc32", "strict_sr_gs_qc32", "strict_sr_gs_pol_qc32")
        )
        for off, suffix in zip((1, 4, 7, 10), suffixes):
            n = f"topk{k}_{suffix}"
            if cells[off] == "--": cmp(f"T2 k={k} {n}", None, None if not os.path.exists(f"{M}/ts_surgery_{n}.json") else 1, 0); continue
            s = S(n)
            cmp(f"T2 k={k} {n} acc", num(cells[off])[0], 100*s["student_pre_ft_val_acc"], 0.005, (cells, off, 0))
            cmp(f"T2 k={k} {n} lat", num(cells[off+1])[0], s["student_pre_ft_val_relative_slowdown_vs_ref"], 0.005, (cells, off+1, 0))
            mr = s["student_pre_ft_val_peak_gpu_mem_bytes"] / s["ref_val_peak_gpu_mem_bytes"]
            cmp(f"T2 k={k} {n} mem", num(cells[off+2])[0], mr, 0.005 if mr < 10 else 0.05, (cells, off+2, 0))
    # Table 3
    for cells in table_rows("tab:ptq"):
        k = int(cells[0]); g = cells[2]
        mc, mt = ("strict_sr_gs", "strict_sr_gs_pertensor") if g == "strict" else ("fast_jeffreys", "fast_jeffreys_pertensor")
        c = J(f"ts_ptq_{k}_{mc}_8bit"); t = J(f"ts_ptq_{k}_{mt}_8bit")
        cmp(f"T3 k={k} {g} nodes", float(cells[3]), len(c["ptq_nodes"]), 0, (cells, 3, 0))
        cmp(f"T3 k={k} {g} PV", float(cells[1]), round(k/197, 2), 0.005, (cells, 1, 0))
        a_raw = 100*c["metrics"]["fp_val_acc"]; a = r2(a_raw); cmp(f"T3 k={k} {g} bf16", float(cells[4]), a, 0.005, (cells, 4, 0))
        cmp(f"T3 k={k} {g} bf16 = pertensor bf16", r2(100*t["metrics"]["fp_val_acc"]), a, 0.005)
        b_raw = 100*c["metrics"]["ptq_val_acc"]; bt_raw = 100*t["metrics"]["ptq_val_acc"]
        b = r2(b_raw); bt = r2(bt_raw)
        cmp(f"T3 k={k} {g} ptq", float(cells[5]), b, 0.005, (cells, 5, 0)); cmp(f"T3 k={k} {g} d", num(cells[6])[0], r2(b_raw-a_raw), 0.005, (cells, 6, 0))
        cmp(f"T3 k={k} {g} ptq-t", float(cells[7]), bt, 0.005, (cells, 7, 0)); cmp(f"T3 k={k} {g} d-t", num(cells[8])[0], r2(bt_raw-a_raw), 0.005, (cells, 8, 0))
        if g == "strict": cmp(f"T3 k={k} bf16 = surgery gs acc", a, r2(100*S(f"topk{k}_strict_sr_gs")["student_pre_ft_val_acc"]), 0.005)
        else: cmp(f"T3 k={k} bf16 = saved Jeffreys run", a, r2(sel(D(k, "jeffreys"))), 0.005)
    # Table 4
    for cells in table_rows("tab:deit"):
        k = int(cells[0]); s = S(f"topk{k}_fast"); d = D(k, "jeffreys"); r = d["student_post_distill_runs"][0]
        cmp(f"T4 k={k} PV", float(cells[1]), round(k/197, 2), 0.005, (cells, 1, 0))
        cmp(f"T4 k={k} acc", float(cells[2]), 100*s["student_pre_ft_val_acc"], 0.005, (cells, 2, 0))
        dj = float(cells[3]); e = r["baseline_val_jeffreys_mean"]; cmp(f"T4 k={k} DJ", dj, e, max(0.005, 0.05*e) if e > 0.01 else 0.0005*max(1, e/0.001), (cells, 3, 0))
        mean, sd = num(cells[4]); cmp(f"T4 k={k} dist mean", mean, 100*d["student_post_distill_val_acc_mean"], 0.005, (cells, 4, 0)); cmp(f"T4 k={k} dist sd", sd, 100*d["student_post_distill_val_acc_std"], 0.005, (cells, 4, 1))
        cmp(f"T4 k={k} lat", num(cells[5])[0], s["student_pre_ft_val_relative_slowdown_vs_ref"], 0.005, (cells, 5, 0))
        im = S(f"topk{k}_fast_imagenet"); cmp(f"T4 k={k} imagenet", float(cells[7]), 100*im["student_pre_ft_val_acc"], 0.005, (cells, 7, 0))
    # Table 5
    mb = {n: {r["k"]: r for r in json.load(open(f"{M}/mambair_{n}_set5_by_k.json"))["rows"]} for n in ("x4_tailmass0", "x4", "x4_exact", "x2")}
    for cells in table_rows("tab:mambair"):
        k = int(cells[0]); i = 1
        for n in ("x4_tailmass0", "x4", "x4_exact", "x2"):
            r = mb[n].get(k) or (mb["x4"][k] if (n == "x4_tailmass0" and k == 256) else None)
            for key in ("pre_psnr", "post_psnr"):
                got = None if cells[i] == "--" else float(cells[i]); exp = r.get(key) if r else None
                cmp(f"T5 k={k} {n} {key}", got, exp, 0.005, (cells, i, 0)); i += 1
    # Table 6 Pythia
    for cells in table_rows("tab:pythia"):
        k = int(cells[0])
        for off, n in ((1, f"pythia_70m_topk{k}_fast_tailmass_exact"), (4, f"pythia_70m_topk{k}_fast_tailmass0")):
            p = f"{M}/ts_surgery_{n}.json"
            if not os.path.exists(p): print("missing", n); bad += 1; continue
            s = S(n); nll = s["student_pre_ft_val_loss"]; ref = s["ref_val_loss"]
            cmp(f"T6 k={k} {n} nll", float(cells[off]), nll, 0.00005, (cells, off, 0)); cmp(f"T6 k={k} {n} dnll", float(cells[off+1]), nll-ref, 0.00006, (cells, off+1, 0))
            import math; cmp(f"T6 k={k} {n} ppl", float(cells[off+2]), math.exp(nll), 0.005, (cells, off+2, 0))
        tau = S(f"pythia_70m_topk{k}_fast_tailmass_exact").get("gibbs_tail_prob_eps_exact_mean")
        cmp(f"T6 k={k} tau", float(cells[7]), tau, 0.0005, (cells, 7, 0))
    # S1
    for cells in table_rows("tab:s-tail"):
        k = int(cells[0])
        for i, n in enumerate((f"topk{k}_fast_tailmass0", f"topk{k}_fast_tailmass1e3", f"topk{k}_fast", f"topk{k}_fast_tailmass_exact"), 1):
            cmp(f"S1 k={k} {n}", float(cells[i]), 100*S(n)["student_pre_ft_val_acc"], 0.005, (cells, i, 0))
        cmp(f"S1 k={k} saved J", float(cells[5]), sel(D(k, "jeffreys")), 0.005, (cells, 5, 0))
    # S2
    for cells in table_rows("tab:s-q"):
        k = int(cells[0]); a = S(f"topk{k}_fast"); e = S(f"topk{k}_fast_tailmass_exact")
        cmp(f"S2 k={k} cal", float(cells[1]), a["gibbs_tail_prob_eps_calibrated_mean"], 0.00005, (cells, 1, 0))
        v = num(cells[2])
        if k == 197: cmp("S2 197", v[0], 0.0, 0); continue
        sb = e["gibbs_tail_prob_eps_exact_std_by_block"]
        cmp(f"S2 k={k} mean", v[0], e["gibbs_tail_prob_eps_exact_mean"], 0.00005, (cells, 2, 0)); cmp(f"S2 k={k} sd", v[1], sum(sb)/len(sb), 0.00005, (cells, 2, 1))
        cmp(f"S2 k={k} min", v[2], e["gibbs_tail_prob_eps_exact_min"], 0.00005, (cells, 2, 2)); cmp(f"S2 k={k} max", v[3], e["gibbs_tail_prob_eps_exact_max"], 0.00005, (cells, 2, 3))
    # S3
    for cells in table_rows("tab:s-distill"):
        k = int(cells[0]); cmp(f"S3 k={k} compiled", float(cells[1]), 100*S(f"topk{k}_fast")["student_pre_ft_val_acc"], 0.005, (cells, 1, 0))
        for off, l in ((2, "ce"), (4, "jeffreys")):
            d = D(k, l); cmp(f"S3 k={k} {l} saved", float(cells[off]), sel(d), 0.005, (cells, off, 0))
            mean, sd = num(cells[off+1]); cmp(f"S3 k={k} {l} mean", mean, 100*d["student_post_distill_val_acc_mean"], 0.005, (cells, off+1, 0)); cmp(f"S3 k={k} {l} sd", sd, 100*d["student_post_distill_val_acc_std"], 0.005, (cells, off+1, 1))
    # S4 profile
    R = json.load(open(f"{REPO}/artifacts/logs/profile_gs_ref_2026-10-18/profile.json"))
    by = {os.path.basename(r["checkpoint"]).replace("ts_surgery_", "").replace(".pt", ""): r["groups"] for r in R}
    ref_rec = next(r for r in R if os.path.basename(r["checkpoint"]).startswith("ts_pretrain_pet"))
    by["reference"] = ref_rec["groups"]
    gm = {"$\\mathrm{Score}$ ($QK^\\top$)": "Score (QK^T)", "$\\mathrm{Mix}$ ($PV$)": "Mix (PV)", "LayerNorm": "LayerNorm", "Softmax": "Softmax (top-k Gibbs)", "Centroid selection": "Routing (centroid select)", "Linear (qkv, proj)": "Linear (qkv, proj)", "MLP": "MLP"}
    cols = ("topk16_strict_sr_gs", "topk16_strict_sr_gs_qc32", "topk16_hadamard_sr_gs", "topk16_hadamard_sr_gs_qc32", "topk16_fast", "reference")
    for cells in table_rows("tab:s-profile"):
        g = gm[cells[0]]
        for i, m in enumerate(cols):
            v = by[m].get(g)
            cmp(f"S4 {g} {m} ms", None if cells[1+i] == "--" else float(cells[1+i]), v["ms_per_batch"] if v else None, 0.05, (cells, 1+i, 0))
            cmp(f"S4 {g} {m} MiB", None if cells[7+i] == "--" else float(cells[7+i]), v["peak_extra_mib"] if v else None, 0.5, (cells, 7+i, 0))
    # Table S9: bf16 against fp32 with all keys (make dtype-check)
    dt = {(r["network"], r["model"], r["variant"]): r for r in json.load(open(f"{REPO}/artifacts/logs/dtype_check_2026-10-18/dtype_check.json"))}
    net_of = {"DeiT-Tiny, Pet": ("DeiT-Tiny / Pet", 100, 2), "DeiT-Tiny, ImageNet-1k": ("DeiT-Tiny / ImageNet-1k", 100, 2),
              "MambaIRv2-Light $\\times2$, Set5": ("MambaIRv2-Light x2 / Set5", 1, 3),
              "MambaIRv2-Light $\\times4$, Set5": ("MambaIRv2-Light x4 / Set5", 1, 3),
              "Pythia-70M, WikiText-2": ("Pythia-70M / WikiText-2", 1, 3)}
    net = None
    for cells in table_rows("tab:s-dtype"):
        if cells[0]: net = net_of[cells[0]]
        name, scale, nd = net
        model = cells[1]
        val = lambda r: scale * r.get("nll", r["primary"])
        if model == "reference":
            b, f = val(dt[(name, model, "bf16")]), val(dt[(name, model, "fp32")])
        else:
            b, f = val(dt[(name, model, "checkpoint bf16")]), val(dt[(name, model, "rebuild fp32")])
        tol = 0.5 * 10 ** -nd + 1e-9
        cmp(f"S9 {name} {model} bf16", float(cells[2]), round(b, nd), tol, (cells, 2, 0))
        cmp(f"S9 {name} {model} fp32", float(cells[3]), round(f, nd), tol, (cells, 3, 0))
        cmp(f"S9 {name} {model} diff", num(cells[4])[0], round(b - f, nd), tol, (cells, 4, 0))
    print(f"tables: checked {checked} cells, {bad} mismatches")
    return bad


FIG_FIXES: list = []  # (old coordinate text, new) with --fix


def check_figure(tex: str, M: str) -> int:
    def S(n): return json.load(open(f"{M}/ts_surgery_{n}.json"))["calibration"]
    mb = {n: {r["k"]: r for r in json.load(open(f"{M}/mambair_{n}_set5_by_k.json"))["rows"]} for n in ("x4_tailmass0", "x4", "x4_exact")}
    ref = S("pythia_70m_topk128_fast_tailmass_exact")["ref_val_loss"]
    bad = points = 0
    for idx, (style, coords) in enumerate(re.findall(r"\\addplot\[(\w+)\] coordinates \{([^}]*)\}", tex)):
        for c in re.findall(r"\(([^)]*)\)", coords):
            k, v = c.split(","); k, v = int(float(k)), float(v); points += 1
            if idx < 3:
                n = f"topk{k}_fast" + {"dropstyle": "_tailmass0", "calstyle": "", "exactstyle": "_tailmass_exact"}[style]
                e, tol = 100 * S(n)["student_pre_ft_val_acc"], 0.005
            elif idx < 5:
                n = f"pythia_70m_topk{k}_fast_" + ("tailmass0" if style == "dropstyle" else "tailmass_exact")
                e, tol = S(n)["student_pre_ft_val_loss"] - ref, 0.00006
            else:
                e, tol = mb[{"dropstyle": "x4_tailmass0", "calstyle": "x4", "exactstyle": "x4_exact"}[style]][k]["pre_psnr"], 0.0005
            if abs(v - e) > tol:
                bad += 1; print(f"MISMATCH figure plot {idx} {style} k={k}: paper {v} expected {e}")
                FIG_FIXES.append((c, f"{k},{_fmt_like(c.split(',')[1], e)}"))
    print(f"figure: checked {points} points, {bad} mismatches")
    return bad


def check_text(P: str, M: str) -> int:
    T = {f: open(f"{P}/{f}.tex").read() for f in ("results_experiments", "supplementary", "methods", "introduction", "abstract", "discussion")}
    ALL = "\n".join(T.values())
    def S(n): return json.load(open(f"{M}/ts_surgery_{n}.json"))["calibration"]
    def D(k, l): return json.load(open(f"{M}/ts_distill_{k}_fast_{l}.json"))["calibration"]
    acc = lambda n: 100*S(n)["student_pre_ft_val_acc"]
    lat = lambda n: S(n)["student_pre_ft_val_relative_slowdown_vs_ref"]
    mem = lambda n: S(n)["student_pre_ft_val_peak_gpu_mem_bytes"]/1e6
    KS = [1, 2, 4, 8, 16, 32, 64, 96, 128, 197]
    bad = 0
    def claim(desc, ok, detail=""):
        nonlocal bad
        if not ok: bad += 1
        print(("OK  " if ok else "BAD ") + desc + (f"  [{detail}]" if detail else ""))
    def has(s, where=ALL): return s in where
    # models match
    claim("Pet 91.36 vs 91.36", has("$91.36\\%$ against $91.36\\%$") and f"{acc('topk197_fast'):.2f}" == "91.36" and f"{100*S('topk197_fast')['ref_val_acc']:.2f}" == "91.36")
    claim("ImageNet 72.01 vs 72.02", has("$72.01\\%$ against $72.02\\%$") and f"{acc('topk197_fast_imagenet'):.2f}" == "72.01" and f"{100*S('topk197_fast_imagenet')['ref_val_acc']:.2f}" == "72.02")
    mb4 = {r["k"]: r for r in json.load(open(f"{M}/mambair_x4_set5_by_k.json"))["rows"]}
    mbe = {r["k"]: r for r in json.load(open(f"{M}/mambair_x4_exact_set5_by_k.json"))["rows"]}
    mbd = {r["k"]: r for r in json.load(open(f"{M}/mambair_x4_tailmass0_set5_by_k.json"))["rows"]}
    claim("Set5 32.51 vs 32.51", f"{mb4[256]['pre_psnr']:.2f}" == "32.51" and f"{mb4[256]['ref_psnr']:.2f}" == "32.51", f"{mb4[256]['pre_psnr']:.3f}/{mb4[256]['ref_psnr']:.3f}")
    py = S("pythia_70m_topk128_fast_tailmass_exact"); claim("Pythia NLL 4.5496 vs 4.5496", f"{py['student_pre_ft_val_loss']:.4f}" == "4.5496" and f"{py['ref_val_loss']:.4f}" == "4.5496")
    gap = max(abs(acc(f"topk{k}_strict_sr") - acc(f"topk{k}_fast")) for k in (16, 32, 64, 197)); claim("strict within 1.6 of extended (Table 2)", gap <= 1.6, f"{gap:.2f}")
    le = [lat(f"topk{k}_fast") for k in KS] + [lat(f"topk{k}_fast_qc32") for k in (16, 32, 64)]; claim("extended latency 0.97..1.03, both supports", f"{min(le):.2f}" == "0.97" and f"{max(le):.2f}" == "1.03" and has("$0.97$ to $1.03$ times the reference latency in every row"), f"{min(le):.3f}..{max(le):.3f}")
    lh = [lat(f"topk{k}_hadamard_sr_gs{s}") for k in (16, 32, 64) for s in ("", "_qc32")]; claim("intro: Hadamard 1.01..1.03x for k<=64, both supports", f"{min(lh):.2f}" == "1.01" and f"{max(lh):.2f}" == "1.03" and has("$1.01$ to $1.03$ times the reference latency for $k\\le64$"), f"{min(lh):.3f}..{max(lh):.3f}")
    ls = [lat(f"topk{k}_strict_sr_gs") for k in (16, 32, 64, 197)] + [lat(f"topk{k}_strict_sr_gs_qc32") for k in (16, 32, 64)]
    claim("strict 1.02..2.55x (intro), 1.0..2.6 (results)", f"{min(ls):.2f}" == "1.02" and f"{max(ls):.2f}" == "2.55" and has("runs at $1.02$ to $2.55$ times the reference latency") and has("is $1.0$ to $2.6$ times slower"), f"{min(ls):.3f}..{max(ls):.3f}")
    claim("2.3x memory at top-64", f"{mem('topk64_strict_sr_gs')/mem('topk64_fast'):.1f}" == "2.3" and has("at top-$64$ it uses $2.3$ times the peak memory"))
    claim("strict 90.73 / ext 91.36 all keys", f"{acc('topk197_strict_sr_gs'):.2f}" == "90.73" and f"{acc('topk197_fast'):.2f}" == "91.36")
    g = [acc(f"topk{k}_strict_sr") - acc(f"topk{k}_strict_sr_gs") for k in (16, 32, 64, 197)]; claim("GELU costs strict 0.3..2.3", f"{min(g):.1f}" == "0.3" and f"{max(g):.1f}" == "2.3" and has("strict graph $0.3$ to $2.3$ points"), ", ".join(f"{x:.2f}" for x in g))
    # PTQ ranges
    def pt(k, m): x = json.load(open(f"{M}/ts_ptq_{k}_{m}_8bit.json"))["metrics"]; return 100*(x["ptq_val_acc"] - x["fp_val_acc"])
    # PTQ (2026-10-17 rerun: fp32 simulation, bf16 reference, 8-bit logits)
    for m, lo, hi, text in (
        ("strict_sr_gs", "-1.47", "-0.05", "strict graph by $-1.47$ to $-0.05$ points with per-channel"),
        ("strict_sr_gs_pertensor", "-1.06", "-0.14", "and by $-1.06$ to $-0.14$ with one scale per weight tensor"),
        ("strict_sr", "-1.44", "-0.08", "by $-1.44$ to $-0.08$ points with per-channel weight scales"),
        ("strict_sr_pertensor", "-0.90", "+0.19", "and by $-0.90$ to $+0.19$ points with per-tensor scales"),
    ):
        d = [pt(k, m) for k in (16, 32, 64, 197)]
        claim(f"PTQ {m} {lo}..{hi}", f"{min(d):+.2f}" == lo and f"{max(d):+.2f}" == hi and has(text), ", ".join(f"{x:+.2f}" for x in d))
    d = [pt(k, m) for k in (16, 32, 64, 197) for m in ("fast_jeffreys", "fast_jeffreys_pertensor")]
    claim("PTQ extended -2.5..-3.9", f"{max(d):.1f}" == "-2.5" and f"{min(d):.1f}" == "-3.9" and has("changes accuracy by $-2.5$ to $-3.9$ points"), f"{min(d):.2f}..{max(d):.2f}")
    ptq_runs = [json.load(open(f"{M}/ts_ptq_{k}_{m}_8bit.json")) for k in (16, 32, 64, 197)
                for m in ("strict_sr_gs", "strict_sr_gs_pertensor", "fast_jeffreys", "fast_jeffreys_pertensor", "strict_sr", "strict_sr_pertensor")]
    oq = [100*(r["metrics"]["ptq_val_acc"] - r["metrics"]["ptq_val_acc_unquantized_output"]) for r in ptq_runs]
    claim("8-bit logits change accuracy by -0.16..+0.30", f"{min(oq):+.2f}" == "-0.16" and f"{max(oq):+.2f}" == "+0.30" and has("Quantizing the logits changes accuracy by $-0.16$ to $+0.30$ points"), f"{min(oq):+.2f}..{max(oq):+.2f}")
    bound = max(n["integer_sum_bound"] or 0 for r in ptq_runs for n in r["ptq_nodes"])
    claim("integer sums below 4.5e6 < 2^24", bound < 4.5e6 and has("stays below $4.5\\times10^{6}$"), f"{bound:.4g}")
    claim("all PTQ runs fp32, 8-bit logits", all(r["quantization"]["simulation_dtype"] == "float32" and r["quantization"]["quantize_output"] and r["quantization"]["output_bits"] == 8 for r in ptq_runs))
    # tail mass text
    dj = lambda k: D(k, "jeffreys")["student_post_distill_runs"][0]["baseline_val_jeffreys_mean"]
    claim("DJ 0.98 k=8, 0.043 k=64, 2e-4 k=197", f"{dj(8):.2f}" == "0.98" and f"{dj(64):.3f}" == "0.043" and f"{dj(197):.0e}" == "2e-04" and has("falls from $0.98$ at $k=8$ to $0.043$ at $k=64$"), f"{dj(8):.4f} {dj(64):.5f} {dj(197):.2e}")
    d64 = D(64, "jeffreys"); claim("top-64 distilled 90.43±0.18", f"{100*d64['student_post_distill_val_acc_mean']:.2f}±{100*d64['student_post_distill_val_acc_std']:.2f}" == "90.43±0.18" and has("$90.43\\pm0.18\\%$"))
    claim("ImageNet top-64 within 1.19", f"{100*S('topk64_fast_imagenet')['ref_val_acc'] - acc('topk64_fast_imagenet'):.2f}" == "1.19")
    # MambaIR
    claim("MambaIR exact pre highest for every 0<k<256", all(mbe[k]["pre_psnr"] >= max(mb4[k]["pre_psnr"], (mbd.get(k) or {}).get("pre_psnr") or -1) for k in (0, 1, 2, 4, 16, 64)))
    claim("MambaIR exact top-64 within 0.03 dB of ref", abs(mbe[64]["ref_psnr"] - mbe[64]["pre_psnr"]) <= 0.035, f"{mbe[64]['ref_psnr'] - mbe[64]['pre_psnr']:.4f}")
    dd = [abs(mbe[k]["post_psnr"] - mbd[k]["post_psnr"]) for k in (4, 16, 64)]; claim("MambaIR post exact vs drop within 0.06 for k>=4", max(dd) <= 0.065, ", ".join(f"{x:.3f}" for x in dd))
    claim("MambaIR exact at k<=2 beats drop at 2k", mbe[1]["pre_psnr"] > mbd[2]["pre_psnr"] and mbe[2]["pre_psnr"] > mbd[4]["pre_psnr"])
    claim("MambaIR calibrated below drop for k>=16", all(mb4[k]["pre_psnr"] < mbd[k]["pre_psnr"] for k in (16, 64)))
    c16 = S("mambair_x4_topk16_fast"); sb = c16["gibbs_tail_prob_eps_exact_std_by_block"]
    claim("MambaIR k=16 sd ~0.2 at mean 0.57", f"{c16['gibbs_tail_prob_eps_exact_mean']:.2f}" == "0.57" and abs(sum(sb)/len(sb) - 0.2) < 0.025, f"mean {c16['gibbs_tail_prob_eps_exact_mean']:.3f} sd {sum(sb)/len(sb):.3f}")
    # DeiT exact vs drop at 2k
    ex = {k: acc(f"topk{k}_fast_tailmass_exact") for k in KS}; dr = {k: acc(f"topk{k}_fast_tailmass0") for k in (1, 2, 4, 8, 16, 32, 64)}; ca = {k: acc(f"topk{k}_fast") for k in KS}
    claim("DeiT exact at k beats drop at 2k, k=1..32", all(ex[k] > dr[2*k] for k in (1, 2, 4, 8, 16, 32)))
    claim("DeiT calibrated at k beats drop at 2k, k<=16", all(ca[k] > dr[2*k] for k in (1, 2, 4, 8, 16)))
    claim("calibrated k=32 within 0.2 of drop 64", abs(ca[32] - dr[64]) <= 0.2, f"{dr[64]-ca[32]:.2f}")
    claim("k=8 cal vs drop 73.8 (79.50 vs 5.70)", f"{ca[8]-dr[8]:.1f}" == "73.8" and f"{ca[8]:.2f}" == "79.50" and f"{dr[8]:.2f}" == "5.70" and has("calibrated mode gains $73.8$ points"))
    claim("calibrated beats exact for k<=4, not k=8 (15.73 vs 8.69 at k=1)", all(ca[k] > ex[k] for k in (1, 2, 4)) and ca[8] < ex[8] and f"{ca[1]:.2f}" == "15.73" and f"{ex[1]:.2f}" == "8.69" and has("for $k\\le4$ calibrated mode beats exact mode"))
    # Pythia text
    pn = lambda k, m: S(f"pythia_70m_topk{k}_fast_{m}")["student_pre_ft_val_loss"]
    ref = py["ref_val_loss"]
    claim("Pythia exact within 0.01 down to k=16, drop 0.04 at 16", all(pn(k, "tailmass_exact") - ref < 0.01 for k in (16, 32, 64, 128)) and f"{pn(16, 'tailmass0') - ref:.2f}" == "0.04")
    claim("Pythia exact at k beats drop at 2k for k<=8", all(pn(k, "tailmass_exact") < pn(2*k, "tailmass0") for k in (1, 2, 4, 8)))
    claim("Pythia k=16,32 exact within 0.002 of drop at 2k", all(abs(pn(k, "tailmass_exact") - pn(2*k, "tailmass0")) <= 0.002 for k in (16, 32)), ", ".join(f"{pn(k,'tailmass_exact') - pn(2*k,'tailmass0'):+.4f}" for k in (16, 32)))
    # clustered
    pw = lambda n: S(n)["student_pointwise_per_input"]
    rm = lambda n: S(n)["student_pre_ft_val_peak_gpu_mem_bytes"]/S(n)["ref_val_peak_gpu_mem_bytes"]
    claim("routed k=16: 2.1x vs 7.9x reference memory, 78.22 vs 84.08, latency 1.2x", f"{lat('topk16_strict_sr_gs_qc32'):.1f}" == "1.2" and f"{rm('topk16_strict_sr_gs_qc32'):.1f}" == "2.1" and f"{rm('topk16_strict_sr_gs'):.1f}" == "7.9" and has("uses $2.1\\times$ instead of $7.9\\times$ the reference memory") and f"{acc('topk16_strict_sr_gs_qc32'):.2f}" == "78.22" and f"{acc('topk16_strict_sr_gs'):.2f}" == "84.08" and has("at $78.22\\%$ against $84.08\\%$ accuracy"))
    claim("unary 2.4e7 vs 1.1e8", f"{pw('topk16_strict_sr_gs_qc32'):.1e}" == "2.4e+07" and f"{pw('topk16_strict_sr_gs'):.1e}" == "1.1e+08")
    claim("k=32 routed 85.96 >= exact top-16 84.08; 37% unary; 42% memory", acc("topk32_strict_sr_gs_qc32") >= acc("topk16_strict_sr_gs") and round(pw("topk32_strict_sr_gs_qc32")/pw("topk16_strict_sr_gs"), 2) == 0.37 and round(mem("topk32_strict_sr_gs_qc32")/mem("topk16_strict_sr_gs"), 2) == 0.42, f"unary ratio {pw('topk32_strict_sr_gs_qc32')/pw('topk16_strict_sr_gs'):.3f} mem ratio {mem('topk32_strict_sr_gs_qc32')/mem('topk16_strict_sr_gs'):.3f}")
    cut = 1 - mem("topk16_hadamard_sr_gs_qc32")/mem("topk16_hadamard_sr_gs"); claim("routing cuts Hadamard memory by 73% at k=16", f"{100*cut:.0f}" == "73" and has("memory by $73\\%$ at $k=16$"), f"{100*cut:.1f}%")
    print(f"text: {bad} claims failed")
    return bad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", required=True, help="Paper sections directory (results_experiments.tex, ...).")
    parser.add_argument("--repo", default=".", help="Experiment repository root.")
    parser.add_argument("--fix", action="store_true", help="Write the expected table cells and figure points into the paper.")
    args = parser.parse_args()
    repo = os.path.abspath(args.repo)
    meta = os.path.join(repo, "artifacts", "metadata")
    sections = os.path.abspath(args.sections)
    bad = check_tables(sections, repo, meta, fix=args.fix)
    fig_path = os.path.join(sections, "results_experiments.tex")
    bad += check_figure(open(fig_path).read(), meta)
    if args.fix:
        n_cells = apply_fixes(sections)
        text = open(fig_path).read()
        for old, new in FIG_FIXES:
            assert text.count(f"({old})") >= 1, old
            text = text.replace(f"({old})", f"({new})", 1)
        open(fig_path, "w").write(text)
        print(f"fixed {n_cells} table cells and {len(FIG_FIXES)} figure points; text claims are not fixed")
    bad += check_text(sections, meta)
    print("all paper numbers match" if bad == 0 else f"{bad} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
