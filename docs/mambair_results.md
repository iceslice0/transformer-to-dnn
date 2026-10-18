# MambaIRv2-Light Results (Set5, DIV2K fine-tuning)

> Numbers in this file predate the 2026-10-17/18 leak-fix reruns (fine-tunes now keep the epoch with the highest
> training PSNR; `artifacts/logs/leakfix_accsel_2026-10-18/`). Current values: `artifacts/metadata/mambair_*_set5_by_k.json`
> and the paper tables (`scripts/check_paper_numbers.py`).

Reference LayerNorm eps 1e-5 (rerun 2026-10-09/10, logs in `artifacts/logs/eps_rerun_2026-10-09/`).
Y-channel PSNR (dB) on Set5 before (pre = surgery) and after (post = 3 epochs of L1 fine-tuning on DIV2K; since
2026-10-18 the kept epoch has the highest training PSNR on one centered patch per DIV2K training pair, before that the
best Set5 epoch).
Generated from per-run metadata by `scripts/summarize_mambair.py` (summaries in `artifacts/metadata/mambair_x*_set5_by_k.json`;
pre-2026-10-03 summaries, which predate the 2026-07-27 reruns, are kept in `artifacts/metadata/mambair_by_k_2026-07-27/`).
Tail drop (q=0) sweep: `make mambair-x4-tailmass0`, logs in `artifacts/logs/mambair_x4_tailmass0_2026-10-03/`.
At k=256 (full support of the 16x16 window) the extended-graph mix scatters the weights back to key order and uses
one dense product (`SparseWeightedSumBySquare`, `K == Nk`), the same sum as the gathered form. The gathered
`[Nq,K,D]` value tensor peaks at 7.0 GiB when fine-tuning at batch 2 (about 28 GiB at batch 8, more than a 16 GB GPU);
the dense form peaks at 3.2 GiB at batch 2 and 12.7 GiB at batch 8. Changed 2026-10-10; the k=256 surgery checkpoints were rebuilt with it.

| k | x4 drop pre | x4 drop post | x4 calibrated pre | x4 calibrated post | x4 exact pre | x4 exact post | x2 calibrated pre | x2 calibrated post |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | — | — | 30.22 | 31.10 | 30.22 | 31.10 | 35.98 | 37.24 |
| 1 | 22.11 | 30.73 | 30.45 | 31.32 | 30.95 | 31.74 | 36.17 | 37.37 |
| 2 | 26.05 | 31.61 | 30.62 | 31.55 | 31.36 | 31.96 | 36.35 | 37.51 |
| 4 | 29.66 | 32.04 | 30.85 | 31.82 | 31.71 | 32.10 | 36.50 | 37.63 |
| 16 | 31.99 | 32.39 | 31.55 | 32.24 | 32.19 | 32.34 | 37.05 | 37.91 |
| 64 | 32.43 | 32.47 | 32.27 | 32.45 | 32.48 | 32.51 | 37.90 | 38.14 |
| 256 | — | — | 32.51 | 32.51 | 32.50 | 32.51 | 38.23 | 38.23 |
