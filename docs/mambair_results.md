# MambaIRv2-Light Results (Set5, DIV2K fine-tuning)

Y-channel PSNR (dB) on Set5 before (pre = surgery) and after (post = 3 epochs of L1 fine-tuning on DIV2K, best epoch).
Generated from per-run metadata by `scripts/summarize_mambair.py` (summaries in `artifacts/metadata/mambair_x*_set5_by_k.json`;
pre-2026-10-03 summaries, which predate the 2026-07-27 reruns, are kept in `artifacts/metadata/mambair_by_k_2026-07-27/`).
Tail drop (q=0) sweep: `make mambair-x4-tailmass0`, logs in `artifacts/logs/mambair_x4_tailmass0_2026-10-03/`.

| k | x4 drop pre | x4 drop post | x4 calibrated pre | x4 calibrated post | x4 exact pre | x4 exact post | x2 calibrated pre | x2 calibrated post |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0   | —     | —     | 30.23 | 31.10 | 30.22 | 31.10 | 35.97 | 37.24 |
| 1   | 22.05 | 30.73 | 30.42 | 31.31 | 30.95 | 31.73 | 36.18 | 37.38 |
| 2   | 26.04 | 31.69 | 30.65 | 31.58 | 31.35 | 31.96 | 36.31 | 37.48 |
| 4   | 29.64 | 32.06 | 30.88 | 31.84 | 31.71 | 32.10 | 36.49 | 37.63 |
| 16  | 31.99 | 32.38 | 31.56 | 32.25 | 32.19 | 32.34 | 37.09 | 37.93 |
| 64  | 32.42 | 32.47 | 32.30 | 32.46 | 32.49 | 32.51 | 37.89 | 38.13 |
| 256 | —     | —     | 32.51 | 32.51 | 32.51 | 32.51 | 38.24 | 38.23 |
