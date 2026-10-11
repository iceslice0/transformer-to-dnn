# Query-Centroid Support Routing (Pet DeiT-Tiny, Pythia-70M)

Exact top-k needs all \(N^2\) scores before it selects \(k\) keys per query, so a strict graph pays
squares on the full score matrix even when it mixes only \(k\) values. Query-centroid routing
selects the support before any score is formed:

- per block and head, k-means (k-means++ then 30 Lloyd iterations; since 2026-10-11 the centroid update is a one-hot
  contraction, not CUDA `scatter_add_`, so a run with `calibration_seed` repeats bitwise) fits \(C\) query centroids
  \(c_m\) on calibration queries (train split, `routing_calibration_batches` batches, at most
  `routing_max_samples` queries per head; after rotary for Pythia);
- query \(i\) goes to its nearest centroid \(m(i)=\arg\max_m (q_i\cdot c_m-\|c_m\|^2/2)\);
- its support is the top-k keys of \(c_{m(i)}\cdot k_j\) (causal mask applied for Pythia);
- exact scores are read on that support only, and the tail mode (calibrated, exact, drop) is unchanged.

With fixed centroids both contractions are fixed affine maps (`AffineContract`) and the assignment and
selection are routing (`RoutingTopK`, `RoutingGather`), so the router adds no unary evaluation and no
activation-activation product (`QueryCentroidRouter` in `ops.py`). With the norm expansion (default
`strict_product="norm"`) the strict cost of the score and mixing transforms becomes \(h(2Nkd+Nk+3Nd)\)
unary evaluations instead of \(h(N^2d+Nkd+Nk+3Nd)\) with exact top-k (paper Proposition `prop:routed`).

Calibrated mode uses the mean routed tail mass \(1-\sum_{j\in S}\pi_{ij}\) of the dense student on the
same train batches as the top-k calibration (`gibbs_tail_calibration_batches`). Support statistics are
validation numbers on the dense student (all validation batches):

- **mass**: dense attention mass inside the support, rows with more than \(k\) valid keys;
- **ratio**: routed mass / exact top-k mass;
- **recall**: fraction of the exact top-k keys inside the routed support;
- **relative inertia**: k-means inertia / mean \(\|q\|^2\) (0 = queries sit on their centroids).

With a calibrated or zero tail mass the routed graph forms only the selected scores
(`QueryCentroidRouter.support` + `PairwiseDotBySquare.selected` + `GibbsTopKSoftmax.forward_selected`):
each centroid keeps its top-k keys, each query gathers the list of its centroid, and the norm expansion
(strict) or a batched product (fast) runs on the gathered keys only. No `N x N` matrix is formed for ViT-style
attention; under a causal mask the selection is taken per row from the masked `N x N` centroid scores
(affine + routing, no products). Exact tail mass needs the dense normalizer and keeps dense scores.
The sparse path equals the dense routed path exactly in the strict graph and to 1e-16 in the fast graph
(unit test `test_routed_support_selection`).

Routed checkpoints reload: `load_surgery_student_checkpoint` re-creates the routers from their saved
buffers (`install_routers_from_state_dict`).

Every surgery run records `student_pointwise_per_input` (unary evaluations per input, by op type) from
forward hooks on the `NL*` nodes. At full
support the strict DeiT-Tiny count with exact GELU is 196,498,438 per image (`topk197_strict_sr`, norm
expansion) = 12 x 16,368,533 (paper Table 1 block total) + 76,042 (final LayerNorm).

## DeiT-Tiny / Pet

Reference 91.52%, reference LayerNorm eps 1e-6 (rerun 2026-10-09). Pre-finetune validation accuracy; top-k columns are the existing `make a`, `make e`
and `make b` runs (`topk{k}_fast`, `_tailmass_exact`, `_tailmass0`).

| k | mass top-k | mass routed | ratio | recall | top-k cal / exact / drop | routed cal / exact / drop |
|---:|---:|---:|---:|---:|---|---|
| 8 | 0.470 | 0.392 | 0.834 | 0.590 | 79.99 / 81.33 / 5.18 | 60.83 / 50.89 / 42.44 |
| 16 | 0.587 | 0.515 | 0.877 | 0.645 | 85.75 / 88.99 / 38.62 | 80.81 / 80.51 / 59.01 |
| 32 | 0.717 | 0.656 | 0.915 | 0.715 | 88.99 / 90.90 / 82.12 | 86.94 / 88.36 / 81.08 |
| 64 | 0.849 | 0.807 | 0.951 | 0.795 | 90.81 / 91.55 / 89.13 | 89.81 / 90.87 / 89.13 |

Centroid count at k=16, calibrated mode:

| clusters | mass routed | ratio | recall | routed cal acc |
|---:|---:|---:|---:|---:|
| 8 | 0.448 | 0.763 | 0.559 | 56.45 |
| 32 | 0.515 | 0.877 | 0.645 | 80.81 |
| 128 | 0.539 | 0.917 | 0.702 | 83.40 |

Per block, k=16, 32 centroids:

| block | ratio | worst head ratio | recall | relative inertia |
|---:|---:|---:|---:|---:|
| 0 | 0.794 | 0.711 | 0.586 | 0.267 |
| 1 | 0.815 | 0.705 | 0.538 | 0.345 |
| 2 | 0.905 | 0.780 | 0.586 | 0.355 |
| 3 | 0.817 | 0.662 | 0.579 | 0.526 |
| 4 | 0.749 | 0.683 | 0.522 | 0.540 |
| 5 | 0.902 | 0.861 | 0.611 | 0.452 |
| 6 | 0.852 | 0.812 | 0.612 | 0.458 |
| 7 | 0.881 | 0.870 | 0.627 | 0.439 |
| 8 | 0.914 | 0.887 | 0.673 | 0.410 |
| 9 | 0.939 | 0.914 | 0.702 | 0.360 |
| 10 | 0.971 | 0.965 | 0.809 | 0.259 |
| 11 | 0.992 | 0.989 | 0.900 | 0.195 |

## Strict graph with routed support (DeiT-Tiny / Pet, calibrated mode, 32 centroids)

Strict graph without exp and log (configs `topk{k}_strict_sr_qc32.json`, `make
strict-sr`), with the norm expansion (`strict_product="norm"`, final rerun 2026-10-10 23:07, `run_final.sh`; the polarization runs are in
`artifacts/backup_polarization_2026-10-10/`), each timed against the bf16 reference in its own run; strict top-k rows
are the `topk{k}_strict_sr` runs of the same target. Unary evaluations per image from the hook count.

| k | top-k acc | routed acc | top-k latency | routed latency | top-k memory | routed memory | top-k unary/img | routed unary/img |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 85.96 | 79.12 | 1.03x | 1.21x | 523 MiB | 141 MiB | 1.03e+08 | 2.06e+07 |
| 32 | 89.13 | 86.29 | 1.12x | 1.14x | 523 MiB | 221 MiB | 1.11e+08 | 3.62e+07 |
| 64 | 90.68 | 89.42 | 1.34x | 1.16x | 523 MiB | 380 MiB | 1.28e+08 | 6.73e+07 |

The routed unary count per image equals `12 (h(2Nkd + 9Nk + 3Nd + 2N) + 8N d_model + 4N) + 2N d_model + 2N`
exactly at every k (paper Table 1 and Proposition `prop:routed`): 20,638,114 / 36,183,778 / 67,275,106 at
k = 16 / 32 / 64. With the sigmoid GELU (`*_strict_sr_gs_qc32`) the routed graph reaches 78.47 / 86.18 / 88.96%. Two runs of one config differ by up
to 0.5 points with top-k and 1.5 points routed: q and the centroids are calibrated on unseeded augmented train batches.

Same-session bench (`make routing-strict-bench`, `scripts/bench_strict_routing.py`, one warm-up pass per
graph, bf16 reference; log `artifacts/logs/norm_construction_2026-10-10/run_final.log`, 2026-10-11 01:49): strict
top-k 1.17x / 1.34x / 1.67x, strict routed 1.21x / 1.21x / 1.29x at k = 16 / 32 / 64
(`artifacts/metadata/routing_strict_sr_bench.json`). Fast (extended) graph: top-k 1.00x / 1.03x / 1.01x, routed
1.01x / 1.02x / 1.00x, peak memory 101 / 144 / 230 MB against 97 / 142 / 233 MB, same unary work
(`artifacts/metadata/routing_fast_bench.json`). The extended graph keeps scores on the MAC array; at N=197 the
score contractions are a small part of a block, so routing changes neither latency nor memory there.
Per-module profile with the routed checkpoints: `docs/profile.md`.

## Pythia-70M / WikiText-2

Reference perplexity 94.59, 252,544 validation tokens, context 128. Top-k columns are the existing
`make pythia` and `make pythia-tailmass0` runs.

| k | mass top-k | mass routed | ratio | recall | top-k PPL exact / drop | routed PPL exact / drop |
|---:|---:|---:|---:|---:|---|---|
| 8 | 0.868 | 0.727 | 0.837 | 0.675 | 97.73 / 107.89 | 151.73 / 183.95 |
| 16 | 0.929 | 0.844 | 0.909 | 0.741 | 95.43 / 98.35 | 117.59 / 128.99 |
| 32 | 0.969 | 0.928 | 0.958 | 0.810 | 94.75 / 95.27 | 101.28 / 103.79 |

Centroid count at k=16, exact mode:

| clusters | mass routed | ratio | recall | routed exact PPL |
|---:|---:|---:|---:|---:|
| 8 | 0.828 | 0.892 | 0.722 | 135.38 |
| 32 | 0.844 | 0.909 | 0.741 | 117.59 |
| 128 | 0.862 | 0.928 | 0.765 | 106.77 |

Per block, k=16, 32 centroids:

| block | ratio | worst head ratio | recall | relative inertia |
|---:|---:|---:|---:|---:|
| 0 | 0.908 | 0.787 | 0.813 | 0.313 |
| 1 | 0.932 | 0.875 | 0.766 | 0.321 |
| 2 | 0.948 | 0.800 | 0.801 | 0.263 |
| 3 | 0.912 | 0.778 | 0.704 | 0.079 |
| 4 | 0.876 | 0.746 | 0.672 | 0.005 |
| 5 | 0.884 | 0.795 | 0.690 | 0.003 |

## Observations

- The routed support holds 83 to 96% of the exact top-k mass on both models, and the share grows with k
  and with the number of centroids.
- DeiT-Tiny: at k=64 routing is within 1.0 point of exact top-k in every tail mode, at k=32 within
  2.6 points. Routed support with the tail-mass term beats exact top-k with tail drop for every k >= 16.
- DeiT-Tiny depth: the last blocks route almost exactly (ratio 0.97 and 0.99, recall 0.81 and 0.90),
  where queries are most clustered (relative inertia 0.26 and 0.20).
- Pythia-70M: perplexity rises 7% at k=32 and 23% at k=16 against exact top-k. Its last two blocks have
  near-collapsed queries (relative inertia 0.005) and still lose 12% of the mass, so position (rotary,
  causal mask) separates keys that one centroid per query cannot.
- With tail drop, DeiT routed support beats exact top-k at k=8 and k=16 (42.44 vs 5.18, 59.01 vs 38.62).

## Reproduce

```bash
make routing          # 14 DeiT runs (k=8..64 x cal/exact/drop at 32 centroids, k=16 at 8 and 128)
make pythia-routing   # 8 Pythia runs (k=8..32 x exact/drop at 32 centroids, k=16 at 8 and 128)
make strict-sr       # strict graph without exp/log: top-k and routed (k=16, 32, 64), exact and sigmoid GELU
make routing-strict  # strict graph with polarization products, routed support
make routing-strict-bench  # same-session cost bench, strict and fast graphs
python scripts/summarize_routing.py   # tables above; writes artifacts/metadata/routing_*_by_k.json
```

Configs: `configs/surgery/topk{k}_fast_qc{C}[_tailmass_exact|_tailmass0].json` and
`configs/surgery/pythia_70m_topk{k}_fast_qc{C}_tailmass{_exact,0}.json`.
Logs: `artifacts/logs/routing_2026-10-09/` (first runs, old eps) and `artifacts/logs/eps_rerun_2026-10-09/`
(DeiT reruns with the reference LayerNorm eps; Pythia runs are unaffected).
