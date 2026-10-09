# Query-Centroid Support Routing (Pet DeiT-Tiny, Pythia-70M)

Exact top-k needs all \(N^2\) scores before it selects \(k\) keys per query, so a strict graph pays
polarization on the full score matrix even when it mixes only \(k\) values. Query-centroid routing
selects the support before any score is formed:

- per block and head, k-means (k-means++ then 30 Lloyd iterations) fits \(C\) query centroids
  \(c_m\) on calibration queries (train split, `routing_calibration_batches` batches, at most
  `routing_max_samples` queries per head; after rotary for Pythia);
- query \(i\) goes to its nearest centroid \(m(i)=\arg\max_m (q_i\cdot c_m-\|c_m\|^2/2)\);
- its support is the top-k keys of \(c_{m(i)}\cdot k_j\) (causal mask applied for Pythia);
- exact scores are read on that support only, and the tail mode (calibrated, exact, drop) is unchanged.

With fixed centroids both contractions are fixed affine maps (`AffineContract`) and the assignment and
selection are routing (`RoutingTopK`, `RoutingGather`), so the router adds no unary evaluation and no
activation-activation product (`QueryCentroidRouter` in `ops.py`). The strict cost of the score and
mixing transforms becomes \(2hNkd\) each instead of \(2hN^2d\) and \(2hNkd\).

Calibrated mode uses the mean routed tail mass \(1-\sum_{j\in S}\pi_{ij}\) of the dense student on the
same train batches as the top-k calibration (`gibbs_tail_calibration_batches`). Support statistics are
validation numbers on the dense student (all validation batches):

- **mass**: dense attention mass inside the support, rows with more than \(k\) valid keys;
- **ratio**: routed mass / exact top-k mass;
- **recall**: fraction of the exact top-k keys inside the routed support;
- **relative inertia**: k-means inertia / mean \(\|q\|^2\) (0 = queries sit on their centroids).

With a calibrated or zero tail mass the routed graph forms only the selected scores
(`QueryCentroidRouter.support` + `PairwiseDotBySquare.selected` + `GibbsTopKSoftmax.forward_selected`):
each centroid keeps its top-k keys, each query gathers the list of its centroid, and polarization (strict)
or a batched product (fast) runs on the gathered keys only. No `N x N` matrix is formed for ViT-style
attention; under a causal mask the selection is taken per row from the masked `N x N` centroid scores
(affine + routing, no products). Exact tail mass needs the dense normalizer and keeps dense scores.
The sparse path equals the dense routed path exactly in the strict graph and to 1e-16 in the fast graph
(unit test `test_routed_support_selection`).

Routed checkpoints reload: `load_surgery_student_checkpoint` re-creates the routers from their saved
buffers (`install_routers_from_state_dict`).

Every surgery run records `student_pointwise_per_input` (unary evaluations per input, by op type) from
forward hooks on the `NL*` nodes plus the two ReLUs per entry of the strict LayerNorm. At full support the
strict DeiT-Tiny count is 368,904,761 per image = 12 x 30,719,983 (paper Table 2 block total) + 264,965
(final LayerNorm).

## DeiT-Tiny / Pet

Reference 91.52%. Pre-finetune validation accuracy; top-k columns are the existing `make a`, `make e`
and `make b` runs (`topk{k}_fast`, `_tailmass_exact`, `_tailmass0`).

| k | mass top-k | mass routed | ratio | recall | top-k cal / exact / drop | routed cal / exact / drop |
|---:|---:|---:|---:|---:|---|---|
| 8 | 0.469 | 0.391 | 0.833 | 0.589 | 79.29 / 81.38 / 5.45 | 61.19 / 48.51 / 16.60 |
| 16 | 0.587 | 0.515 | 0.878 | 0.647 | 85.45 / 88.83 / 39.19 | 80.76 / 78.52 / 66.97 |
| 32 | 0.716 | 0.656 | 0.915 | 0.716 | 88.85 / 90.90 / 82.23 | 87.54 / 89.10 / 82.67 |
| 64 | 0.848 | 0.807 | 0.951 | 0.795 | 90.84 / 91.44 / 89.10 | 89.97 / 91.11 / 88.91 |

Centroid count at k=16, calibrated mode:

| clusters | mass routed | ratio | recall | routed cal acc |
|---:|---:|---:|---:|---:|
| 8 | 0.448 | 0.763 | 0.560 | 56.17 |
| 32 | 0.515 | 0.878 | 0.647 | 80.76 |
| 128 | 0.538 | 0.917 | 0.702 | 82.94 |

Per block, k=16, 32 centroids:

| block | ratio | worst head ratio | recall | relative inertia |
|---:|---:|---:|---:|---:|
| 0 | 0.798 | 0.720 | 0.589 | 0.271 |
| 1 | 0.815 | 0.706 | 0.541 | 0.347 |
| 2 | 0.908 | 0.783 | 0.589 | 0.353 |
| 3 | 0.817 | 0.665 | 0.579 | 0.525 |
| 4 | 0.755 | 0.693 | 0.528 | 0.537 |
| 5 | 0.901 | 0.860 | 0.609 | 0.451 |
| 6 | 0.852 | 0.813 | 0.612 | 0.456 |
| 7 | 0.882 | 0.870 | 0.629 | 0.435 |
| 8 | 0.914 | 0.888 | 0.672 | 0.404 |
| 9 | 0.940 | 0.915 | 0.704 | 0.356 |
| 10 | 0.971 | 0.965 | 0.809 | 0.256 |
| 11 | 0.992 | 0.989 | 0.899 | 0.192 |

## Strict graph with routed support (DeiT-Tiny / Pet, calibrated mode, 32 centroids)

Pipeline runs (`make routing-strict`), each timed against the bf16 reference in its own run; strict top-k
rows are the existing `make c` runs. Unary evaluations per image from the hook count.

| k | top-k acc | routed acc | top-k latency | routed latency | top-k memory | routed memory | top-k unary/img | routed unary/img |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 85.09 | 78.30 | 3.01x | 0.99x | 2805 MiB | 327 MiB | 2.02e8 | 3.77e7 |
| 32 | 88.63 | 85.88 | 3.20x | 1.35x | 2805 MiB | 591 MiB | 2.17e8 | 6.70e7 |
| 64 | 90.27 | 89.23 | 3.58x | 2.21x | 2805 MiB | 1121 MiB | 2.46e8 | 1.26e8 |

The routed squares per image equal `12 (4hNkd + 2N d_model) + N d_model` exactly at every k (score and
mixing polarization plus LayerNorm second moments).

Same-session bench (`make routing-strict-bench`, `scripts/bench_strict_routing.py`, one warm-up pass per
graph, bf16 reference): strict top-k 3.19x / 3.50x / 4.04x, strict routed 1.12x / 1.33x / 2.29x at
k = 16 / 32 / 64 (`artifacts/metadata/routing_strict_bench.json`). Fast (extended) graph: top-k 1.03x /
1.05x / 1.04x, routed 1.03x / 1.06x / 1.07x, peak memory 96 / 137 / 219 MiB against 92 / 135 / 222 MiB,
same unary work (`artifacts/metadata/routing_fast_bench.json`). The extended graph keeps scores on the
MAC array; at N=197 the score contractions are a small part of a block, so routing changes neither
latency nor memory there.

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
- DeiT-Tiny: at k=64 routing is within 0.9 points of exact top-k in every tail mode, at k=32 within
  1.8 points. Routed support with the tail-mass term beats exact top-k with tail drop for every k >= 16.
- DeiT-Tiny depth: the last blocks route almost exactly (ratio 0.97 and 0.99, recall 0.81 and 0.90),
  where queries are most clustered (relative inertia 0.26 and 0.19).
- Pythia-70M: perplexity rises 7% at k=32 and 23% at k=16 against exact top-k. Its last two blocks have
  near-collapsed queries (relative inertia 0.005) and still lose 12% of the mass, so position (rotary,
  causal mask) separates keys that one centroid per query cannot.
- With tail drop, DeiT routed support beats exact top-k at k=8 and k=16 (16.60 vs 5.45, 66.97 vs 39.19).

## Reproduce

```bash
make routing          # 14 DeiT runs (k=8..64 x cal/exact/drop at 32 centroids, k=16 at 8 and 128)
make pythia-routing   # 8 Pythia runs (k=8..32 x exact/drop at 32 centroids, k=16 at 8 and 128)
make routing-strict  # 3 strict DeiT runs with routed support (k=16, 32, 64)
make routing-strict-bench  # same-session cost bench, strict and fast graphs
python scripts/summarize_routing.py   # tables above; writes artifacts/metadata/routing_*_by_k.json
```

Configs: `configs/surgery/topk{k}_fast_qc{C}[_tailmass_exact|_tailmass0].json` and
`configs/surgery/pythia_70m_topk{k}_fast_qc{C}_tailmass{_exact,0}.json`.
Logs: `artifacts/logs/routing_2026-10-09/` (one log per config, `run_all.sh`, `run_all.out`).
