# Per-Module Profile: Strict vs Fast, Top-k vs Routed (Pet, pre-distill surgery checkpoints)

GPU time and working memory per top-level submodule of every block, measured with CUDA events and
`torch.cuda.max_memory_allocated` in forward hooks (`scripts/profile_modules.py`), bf16 autocast, 20 validation
batches of 32 after 3 warm-up batches, NVIDIA RTX 4070 Ti SUPER. Times are summed over the 12 blocks per batch.
Working memory is the peak memory a module call allocates above what is in use when it starts; the table reports
the largest value over calls. Routed checkpoints (32 query centroids) call `dot.selected`, `gibbs.forward_selected`
and `router.support` as methods; the script wraps and times them under the Score, Softmax and Routing groups.
Strict = the strict graph (no exp, no log), reference LayerNorm eps 1e-6.
Command: `artifacts/logs/profile_2026-10-10/command.txt`; raw output: `artifacts/logs/profile_2026-10-10/profile_topk16_32_64.json`.

- k=16: module totals strict 322.7, strict routed 126.3, extended 54.1, extended routed 63.6 ms/batch; strict adds 268.6 ms over extended, 88.7% in Score and Mix
- k=32: module totals strict 339.9, strict routed 161.0, extended 50.1, extended routed 67.8 ms/batch; strict adds 289.7 ms over extended, 89.3% in Score and Mix
- k=64: module totals strict 378.4, strict routed 237.5, extended 67.0, extended routed 84.1 ms/batch; strict adds 311.3 ms over extended, 91.9% in Score and Mix

| k | module | strict ms | strict routed ms | extended ms | extended routed ms | strict MiB | strict routed MiB | extended MiB | extended routed MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | Score (QK^T) | 222.2 | 22.4 | 5.0 | 6.0 | 2751 | 261 | 14 | 42 |
| 16 | Mix (PV) | 25.6 | 26.3 | 4.4 | 5.8 | 265 | 265 | 39 | 39 |
| 16 | Softmax (top-k Gibbs) | 32.8 | 27.4 | 12.9 | 11.2 | 14 | 17 | 7 | 10 |
| 16 | Routing (centroid select) | — | 8.5 | — | 9.4 | — | 4 | — | 4 |
| 16 | LayerNorm | 34.0 | 33.3 | 21.3 | 20.7 | 40 | 39 | 19 | 19 |
| 16 | Linear (qkv, proj) | 3.9 | 4.1 | 4.9 | 4.9 | 7 | 7 | 9 | 9 |
| 16 | MLP | 4.1 | 4.3 | 5.5 | 5.6 | 19 | 18 | 19 | 18 |
| 32 | Score (QK^T) | 222.4 | 40.8 | 3.7 | 9.7 | 2751 | 521 | 14 | 80 |
| 32 | Mix (PV) | 47.8 | 48.0 | 7.9 | 9.3 | 524 | 524 | 76 | 76 |
| 32 | Softmax (top-k Gibbs) | 30.4 | 26.2 | 10.7 | 11.7 | 28 | 33 | 14 | 19 |
| 32 | Routing (centroid select) | — | 7.8 | — | 8.8 | — | 7 | — | 7 |
| 32 | LayerNorm | 31.7 | 30.8 | 19.2 | 19.1 | 40 | 39 | 19 | 19 |
| 32 | Linear (qkv, proj) | 3.7 | 3.7 | 4.3 | 4.3 | 7 | 7 | 9 | 9 |
| 32 | MLP | 3.8 | 3.6 | 4.3 | 4.9 | 18 | 18 | 19 | 18 |
| 64 | Score (QK^T) | 221.1 | 79.9 | 4.1 | 16.9 | 2751 | 1041 | 14 | 155 |
| 64 | Mix (PV) | 84.7 | 83.8 | 15.7 | 16.6 | 1041 | 1041 | 150 | 150 |
| 64 | Softmax (top-k Gibbs) | 31.6 | 27.8 | 13.9 | 11.9 | 55 | 65 | 28 | 37 |
| 64 | Routing (centroid select) | — | 7.7 | — | 8.7 | — | 12 | — | 12 |
| 64 | LayerNorm | 32.7 | 30.7 | 22.5 | 20.4 | 39 | 39 | 19 | 19 |
| 64 | Linear (qkv, proj) | 4.3 | 3.8 | 5.0 | 4.6 | 7 | 7 | 9 | 9 |
| 64 | MLP | 3.9 | 3.7 | 5.7 | 4.9 | 18 | 18 | 18 | 18 |

## Reference columns (2026-10-18)

`make profile-gs` now also profiles the Pet reference (`scripts/profile_modules.py --reference`; log
`artifacts/logs/profile_gs_ref_2026-10-18/`). Its attention runs unfused, with `q k^T`, softmax and `attn @ v` routed
through hookable modules (`install_reference_attention`, checked bit-identical to timm's own unfused path), so it fills
the Score, Softmax and Mix rows of the profile; a second pass times the fused kernel
(`F.scaled_dot_product_attention`, the path of the latency tables): 4.1 ms and 2 MiB per batch at k=16, against 9.0 ms for
the unfused score, softmax and mix. Reference module total: 25.9 ms per batch, against 52.2 ms for the extended and
117.0 ms for the strict graph in the same session.

## Compiled model-only latency (2026-10-18)

`make latency-compiled` (`scripts/latency_compiled.py`; log `artifacts/logs/latency_compiled_2026-10-18/`) times the
forward pass alone. Ten Pet validation batches of 32 are stored on the GPU first, so data loading is
excluded. Each model runs in bf16 under `torch.compile` in the default, reduce-overhead and max-autotune modes; the
fastest mode counts, as the median of five timed runs over the ten batches. The compiled reference (fused attention)
takes 2.20 ms per batch; at k=16 the extended, Hadamard and strict graphs take 7.21, 9.18 and 11.08 ms. The per-module
profile above runs eagerly and synchronizes around every module, so its totals are not forward latencies. Peak memory
stays the eager whole-pass peak from the surgery metadata. `scripts/check_paper_numbers.py` reads `latency.json`.
