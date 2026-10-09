"""Summarize query-centroid support routing against exact top-k from per-run surgery metadata.

Reads ``ts_surgery_<run>.json`` for the routed runs (``*_qc{C}*``) and for the matching exact top-k
runs, prints Markdown tables, and writes ``artifacts/metadata/routing_{deit_pet,pythia_70m}_by_k.json``.
Support statistics (captured dense mass, recall of the exact top-k keys) are validation numbers
measured on the dense student before the router is installed, so they do not depend on the tail mode.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

META = "artifacts/metadata"
DEIT_KS = [8, 16, 32, 64]
DEIT_CLUSTERS = [8, 32, 128]
PYTHIA_KS = [8, 16, 32]
PYTHIA_CLUSTERS = [8, 32, 128]


def _load(name: str) -> Optional[Dict[str, Any]]:
    path = f"{META}/ts_surgery_{name}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["calibration"]


def _support(cal: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if cal is None:
        return {}
    blocks = cal["routing_val_by_block"]
    return {
        "mass_topk": cal["routing_val_mass_topk"],
        "mass_routed": cal["routing_val_mass_routed"],
        "recall": cal["routing_val_recall"],
        "mass_ratio_by_block": [b["mass_routed"] / b["mass_topk"] for b in blocks],
        "recall_by_block": [b["recall"] for b in blocks],
        "min_head_ratio_by_block": [b["mass_ratio_min_head"] for b in blocks],
        "relative_inertia_by_block": [
            sum(f["relative_inertia_by_head"]) / len(f["relative_inertia_by_head"]) for f in cal["routing_fit_by_block"]
        ],
    }


def _acc(cal: Optional[Dict[str, Any]]) -> Optional[float]:
    return None if cal is None else 100.0 * float(cal["student_pre_ft_val_acc"])


def _ppl(cal: Optional[Dict[str, Any]]) -> Optional[float]:
    return None if cal is None else math.exp(float(cal["student_pre_ft_val_loss"]))


def _fmt(v: Optional[float], digits: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def deit() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for k in DEIT_KS:
        routed = {m: _load(f"topk{k}_fast_qc32{s}") for m, s in (("cal", ""), ("exact", "_tailmass_exact"), ("drop", "_tailmass0"))}
        topk = {m: _load(f"topk{k}_fast{s}") for m, s in (("cal", ""), ("exact", "_tailmass_exact"), ("drop", "_tailmass0"))}
        sup = _support(routed["cal"] or routed["exact"] or routed["drop"])
        rows.append(
            {
                "k": k,
                "clusters": 32,
                **sup,
                "acc_topk": {m: _acc(c) for m, c in topk.items()},
                "acc_routed": {m: _acc(c) for m, c in routed.items()},
            }
        )
    clusters = []
    for c in DEIT_CLUSTERS:
        cal = _load(f"topk16_fast_qc{c}")
        clusters.append({"k": 16, "clusters": c, **_support(cal), "acc_routed_cal": _acc(cal)})
    ref = _load("topk16_fast_qc32")
    return {"model": "DeiT-Tiny", "dataset": "Oxford-IIIT Pet", "ref_acc": _acc({"student_pre_ft_val_acc": ref["ref_val_acc"]}) if ref else None, "rows": rows, "clusters": clusters}


def pythia() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for k in PYTHIA_KS:
        routed = {m: _load(f"pythia_70m_topk{k}_fast_qc32{s}") for m, s in (("exact", "_tailmass_exact"), ("drop", "_tailmass0"))}
        topk = {m: _load(f"pythia_70m_topk{k}_fast{s}") for m, s in (("exact", "_tailmass_exact"), ("drop", "_tailmass0"))}
        sup = _support(routed["exact"] or routed["drop"])
        rows.append(
            {
                "k": k,
                "clusters": 32,
                **sup,
                "ppl_topk": {m: _ppl(c) for m, c in topk.items()},
                "ppl_routed": {m: _ppl(c) for m, c in routed.items()},
            }
        )
    clusters = []
    for c in PYTHIA_CLUSTERS:
        cal = _load(f"pythia_70m_topk16_fast_qc{c}_tailmass_exact")
        clusters.append({"k": 16, "clusters": c, **_support(cal), "ppl_routed_exact": _ppl(cal)})
    ref = _load("pythia_70m_topk16_fast_qc32_tailmass_exact")
    return {"model": "Pythia-70M", "dataset": "WikiText-2", "ref_ppl": math.exp(ref["ref_val_loss"]) if ref else None, "rows": rows, "clusters": clusters}


def _print_deit(s: Dict[str, Any]) -> None:
    print(f"\nDeiT-Tiny / Pet (reference {_fmt(s['ref_acc'])}%), 32 query centroids per head\n")
    print("| k | mass top-k | mass routed | ratio | recall | top-k cal / exact / drop | routed cal / exact / drop |")
    print("|---:|---:|---:|---:|---:|---|---|")
    for r in s["rows"]:
        if "mass_topk" not in r:
            continue
        t, q = r["acc_topk"], r["acc_routed"]
        print(
            f"| {r['k']} | {r['mass_topk']:.3f} | {r['mass_routed']:.3f} | {r['mass_routed'] / r['mass_topk']:.3f} | "
            f"{r['recall']:.3f} | {_fmt(t['cal'])} / {_fmt(t['exact'])} / {_fmt(t['drop'])} | "
            f"{_fmt(q['cal'])} / {_fmt(q['exact'])} / {_fmt(q['drop'])} |"
        )
    print("\n| clusters (k=16) | mass routed | ratio | recall | routed cal acc |")
    print("|---:|---:|---:|---:|---:|")
    for r in s["clusters"]:
        if "mass_topk" in r:
            print(f"| {r['clusters']} | {r['mass_routed']:.3f} | {r['mass_routed'] / r['mass_topk']:.3f} | {r['recall']:.3f} | {_fmt(r['acc_routed_cal'])} |")


def _print_pythia(s: Dict[str, Any]) -> None:
    print(f"\nPythia-70M / WikiText-2 (reference PPL {_fmt(s['ref_ppl'])}), 32 query centroids per head\n")
    print("| k | mass top-k | mass routed | ratio | recall | top-k PPL exact / drop | routed PPL exact / drop |")
    print("|---:|---:|---:|---:|---:|---|---|")
    for r in s["rows"]:
        if "mass_topk" not in r:
            continue
        t, q = r["ppl_topk"], r["ppl_routed"]
        print(
            f"| {r['k']} | {r['mass_topk']:.3f} | {r['mass_routed']:.3f} | {r['mass_routed'] / r['mass_topk']:.3f} | "
            f"{r['recall']:.3f} | {_fmt(t['exact'])} / {_fmt(t['drop'])} | {_fmt(q['exact'])} / {_fmt(q['drop'])} |"
        )
    print("\n| clusters (k=16) | mass routed | ratio | recall | routed exact PPL |")
    print("|---:|---:|---:|---:|---:|")
    for r in s["clusters"]:
        if "mass_topk" in r:
            print(f"| {r['clusters']} | {r['mass_routed']:.3f} | {r['mass_routed'] / r['mass_topk']:.3f} | {r['recall']:.3f} | {_fmt(r['ppl_routed_exact'])} |")


def _print_blocks(title: str, sup: Dict[str, Any]) -> None:
    if not sup:
        return
    print(f"\n{title}\n")
    print("| block | mass ratio | worst head ratio | recall | relative inertia |")
    print("|---:|---:|---:|---:|---:|")
    for i, (a, b, c, d) in enumerate(
        zip(sup["mass_ratio_by_block"], sup["min_head_ratio_by_block"], sup["recall_by_block"], sup["relative_inertia_by_block"])
    ):
        print(f"| {i} | {a:.3f} | {b:.3f} | {c:.3f} | {d:.3f} |")


def main() -> None:
    note = "generated by scripts/summarize_routing.py from per-run surgery metadata"
    for name, s, printer in (("deit_pet", deit(), _print_deit), ("pythia_70m", pythia(), _print_pythia)):
        s["note"] = note
        out = f"{META}/routing_{name}_by_k.json"
        with open(out, "w") as f:
            json.dump(s, f, indent=1)
        printer(s)
        print(f"\nwrote {out}")
    _print_blocks("DeiT-Tiny per block, k=16, 32 centroids", _support(_load("topk16_fast_qc32")))
    _print_blocks("Pythia-70M per block, k=16, 32 centroids", _support(_load("pythia_70m_topk16_fast_qc32_tailmass_exact")))


if __name__ == "__main__":
    main()
