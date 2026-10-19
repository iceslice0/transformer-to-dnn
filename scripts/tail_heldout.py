"""Calibrated tail mass against the exact tail mass on held-out data.

The calibrated ``q`` of a block is the mean omitted tail mass of the reference DeiT-Tiny over the Pet training split
(``calibrate_reference`` in surgery). For each ``k`` this script repeats that calibration with the surgery config and
seed, checks that it reproduces the calibrated mean saved in the surgery metadata, and gathers the same per-query
statistics on the Pet validation split (the official test split), which no calibration or selection reads.

The validation pass runs before the training pass, as the reference evaluation does in surgery, so both draw the same
random state.

Usage: python scripts/tail_heldout.py OUT_JSON [--ks 1 2 4 ...]
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict
from typing import Any, Dict, List

from transformer_surgery.cli.common import apply_device_from_config, apply_dtype_from_config
from transformer_surgery.cli.surgery_config import parse_surgery_config
from transformer_surgery.internal.calibration import calibrate_reference
from transformer_surgery.internal.util import get_device, get_surgery_dtype, set_seed
from transformer_surgery.models.adapters import get_model_adapter

KS = (1, 2, 4, 8, 16, 32, 64, 96, 128, 197)
KEYS = (
    "gibbs_tail_calibration_batches",
    "gibbs_tail_prob_eps_exact_mean_by_block",
    "gibbs_tail_prob_eps_exact_std_by_block",
    "gibbs_tail_prob_eps_exact_min_by_block",
    "gibbs_tail_prob_eps_exact_max_by_block",
    "gibbs_tail_prob_eps_exact_rows_by_block",
    "gibbs_tail_prob_eps_exact_mean",
    "gibbs_tail_prob_eps_exact_std",
    "gibbs_tail_prob_eps_exact_min",
    "gibbs_tail_prob_eps_exact_max",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--ks", nargs="+", type=int, default=list(KS))
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    rows: List[Dict[str, Any]] = []
    for k in args.ks:
        cfg = parse_surgery_config(["--config", f"configs/surgery/topk{k}_fast.json"])
        apply_device_from_config(cfg)
        apply_dtype_from_config(cfg)
        if get_device().type != "cuda":
            raise RuntimeError("tail statistics run on the GPU only")
        adapter = get_model_adapter(cfg.model_key)
        set_seed(cfg.calibration_seed)
        train_loader, val_loader = adapter.build_loaders(cfg)
        ref = adapter.load_reference_checkpoint(adapter.reference_checkpoint_path(cfg))
        ref = ref.to(device=get_device(), dtype=get_surgery_dtype())
        val = asdict(calibrate_reference(ref, val_loader, cfg))
        train = asdict(calibrate_reference(ref, train_loader, cfg))
        saved = json.load(open(f"artifacts/metadata/ts_surgery_topk{k}_fast.json"))["calibration"]
        row = {
            "k": k,
            "calibrated_mean_saved": saved.get("gibbs_tail_prob_eps_calibrated_mean"),
            "calibrated_by_block_saved": saved.get("gibbs_tail_prob_eps_calibrated_by_block"),
            "train": {key: train[key] for key in KEYS},
            "val": {key: val[key] for key in KEYS},
        }
        rows.append(row)
        print(
            f"k={k:3d} calibrated q saved {row['calibrated_mean_saved']} recomputed {train['gibbs_tail_prob_eps_exact_mean']}"
            f" | validation tau {val['gibbs_tail_prob_eps_exact_mean']} +- {val['gibbs_tail_prob_eps_exact_std']}"
            f" ({val['gibbs_tail_prob_eps_exact_min']}, {val['gibbs_tail_prob_eps_exact_max']})"
            f" batches train {train['gibbs_tail_calibration_batches']} val {val['gibbs_tail_calibration_batches']}",
            flush=True,
        )
        del ref
    with open(args.out, "w") as f:
        json.dump({"split_train": "Pet trainval", "split_val": "Pet test (validation)", "rows": rows}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
