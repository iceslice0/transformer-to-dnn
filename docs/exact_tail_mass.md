# Exact Tail-Mass vs Calibrated (Pet Fast)

> Numbers in this file predate the 2026-10-17 leak-fix rerun (new Pet reference, 91.36% instead of 91.52%; checkpoint
> selection on the training split only, `artifacts/logs/leakfix_2026-10-17/`). Current values: `artifacts/metadata/` and
> the paper tables (`scripts/check_paper_numbers.py`).

Pre-finetune val accuracy and omitted-tail mass after surgery (`make a` calibrated fast vs `make e` exact), reference
LayerNorm eps 1e-6 (rerun 2026-10-09, logs in `artifacts/logs/eps_rerun_2026-10-09/`).
Accuracies are `student_pre_ft_val_acc`. Calibrated mean \(q\) is `gibbs_tail_prob_eps_calibrated_mean`.
Exact \(q\) stats come from the exact runs: the mean, the within-block std averaged over blocks
(`gibbs_tail_prob_eps_exact_std_by_block`, the paper's Table S2), min and max. The last column is the std pooled
over all blocks (`gibbs_tail_prob_eps_exact_std`; since commit 1c48bf1 this field is pooled, earlier runs stored
the within-block average under the same name).

| k | calib acc | exact acc | calib mean q | exact mean q ± within-block std (min, max) | pooled std |
|---:|---:|---:|---:|---|---:|
| 1 | 14.75% | 7.77% | 0.7823 | 0.7824 ± 0.1660 (0.0005316, 0.9938) | 0.1969 |
| 2 | 42.16% | 20.09% | 0.7138 | 0.7137 ± 0.1795 (9.227e-05, 0.9869) | 0.2175 |
| 4 | 66.61% | 57.35% | 0.6294 | 0.6302 ± 0.1896 (0.0001156, 0.9749) | 0.2320 |
| 8 | 79.99% | 81.33% | 0.5282 | 0.5274 ± 0.1924 (0.0001246, 0.9507) | 0.2332 |
| 16 | 85.75% | 88.99% | 0.4118 | 0.4116 ± 0.1796 (3.815e-06, 0.9019) | 0.2133 |
| 32 | 88.99% | 90.90% | 0.2828 | 0.2826 ± 0.1493 (0, 0.8089) | 0.1740 |
| 64 | 90.81% | 91.55% | 0.1513 | 0.1510 ± 0.0982 (0, 0.6369) | 0.1139 |
| 96 | 90.95% | 91.66% | 0.0810 | 0.0813 ± 0.0606 (0, 0.4695) | 0.0705 |
| 128 | 91.41% | 91.66% | 0.0394 | 0.0395 ± 0.0328 (0, 0.3170) | 0.0385 |
| 197 | 91.74% | 91.74% | 0.0000 | 0.0000 ± 0.0000 (0, 0.0000) | 0.0000 |

## Held-out tail mass (2026-10-19)

`make tail-heldout` (`scripts/tail_heldout.py`; log `artifacts/logs/tail_heldout_2026-10-19/`) repeats the surgery
calibration of each `topk{k}_fast` config (seed 0, full Pet training split; recomputed calibrated means match the saved
ones within 3e-5) and gathers the same exact-tail statistics on the Pet validation split, which no calibration or
selection reads. Paper Supplementary Table S2 compares the calibrated q (training) with the validation tau: the means
differ by at most 0.012 (k=1) and by at most 0.001 for k>=16; validation per-query s.d. within a block is 0.15 to 0.19
for k<=16.
