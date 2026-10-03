# Per-Module Profile: Strict vs Fast (Pet, pre-distill surgery checkpoints)

GPU time and extra peak memory per top-level submodule of every block, measured with CUDA events
and `torch.cuda.max_memory_allocated` in forward hooks (`scripts/profile_modules.py`), bf16 autocast,
20 validation batches of 32 after 3 warm-up batches, NVIDIA RTX 4070 Ti SUPER. Times are summed over
the 12 blocks per batch; peak memory is the largest extra allocation of one module call.
Raw output: `artifacts/logs/profile_2026-10-03/profile_topk16_32_64.json`.

- k=16 added=253.6 share_score_mix=90.8 fwd_ratio=3.62
- k=32 added=265.6 share_score_mix=93.5 fwd_ratio=3.44
- k=64 added=296.2 share_score_mix=93.6 fwd_ratio=3.60

| k | module | strict ms | extended ms | strict peak MiB | extended peak MiB |
|---|---|---:|---:|---:|---:|
| 16 | Score (QK^T) | 213.1 | 4.0 | 2751 | 14 |
| 16 | Mix (PV) | 24.5 | 3.5 | 265 | 39 |
| 16 | Softmax (top-k Gibbs) | 16.6 | 10.8 | 9 | 7 |
| 16 | LayerNorm | 36.9 | 18.1 | 60 | 19 |
| 16 | Linear (qkv, proj) | 3.8 | 4.6 | 7 | 9 |
| 16 | MLP | 3.8 | 4.2 | 19 | 19 |
| 32 | Score (QK^T) | 213.1 | 3.4 | 2751 | 14 |
| 32 | Mix (PV) | 46.3 | 7.6 | 524 | 76 |
| 32 | Softmax (top-k Gibbs) | 15.4 | 10.6 | 18 | 14 |
| 32 | LayerNorm | 32.9 | 18.4 | 60 | 19 |
| 32 | Linear (qkv, proj) | 3.4 | 4.4 | 7 | 9 |
| 32 | MLP | 3.6 | 4.7 | 18 | 19 |
| 64 | Score (QK^T) | 213.2 | 4.1 | 2751 | 14 |
| 64 | Mix (PV) | 81.3 | 13.0 | 1041 | 150 |
| 64 | Softmax (top-k Gibbs) | 15.4 | 10.3 | 35 | 28 |
| 64 | LayerNorm | 32.9 | 17.7 | 60 | 19 |
| 64 | Linear (qkv, proj) | 3.7 | 4.2 | 7 | 9 |
| 64 | MLP | 3.6 | 4.4 | 19 | 18 |
