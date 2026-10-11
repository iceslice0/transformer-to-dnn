# PTQ

PTQ is a post-training quantization stage for an already distilled surgery checkpoint. It loads the
float surgery student through the model adapter, calibrates selected affine/matmul nodes on
validation data, replaces those nodes with calibrated PTQ wrappers, validates the wrapped model, and
saves a traceable checkpoint plus metadata.

PTQ is included to test whether the surgery compile is genuine. The failure hypothesis is that a
cosmetic rewrite would still rely on dynamic high-precision floating-point arithmetic to carry the
model semantics; in that case fixed-scale PTQ wrappers should break accuracy. The PTQ pass leaves
routing and nonlinear scalar ops explicit, but replaces selected affine, bilinear, and matmul-like
numerical work with ordinary per-node calibrated integer proxies. If the wrapped model keeps
accuracy, those arithmetic paths do not need special dynamic floating-point behavior; their scale
can be replaced by calibrated constants. That is the hardware-facing point of the method: a
transformer should look like a DNN over a compact primitive vocabulary that can map to homogeneous
dedicated NPU hardware, instead of depending on a heterogeneous CPU/NPU or GPU implementation with
special-purpose floating-point kernels.

The procedural runner lives in [src/transformer_surgery/ptq.py](../src/transformer_surgery/ptq.py).
The explicit forward wrapper lives in [src/transformer_surgery/ops.py](../src/transformer_surgery/ops.py)
as `CalibratedAffinePTQWrapper`. It owns a `PTQInputQuantizer` and reuses the module being wrapped
as its `accumulator` submodule (`nn.Linear`, `nn.Conv2d`, `AffineMatMul`, `AffineHadamard`, and so
on), so the PTQ checkpoint structure stays human-readable. Calibration stats, wrapper construction, and reload metadata live
in [src/transformer_surgery/internal/calibration.py](../src/transformer_surgery/internal/calibration.py). The
CLI wrapper is `python -m transformer_surgery.cli.ptq` or `ts-ptq`.

## Inputs

The CLI config is defined in
[src/transformer_surgery/cli/ptq_config.py](../src/transformer_surgery/cli/ptq_config.py). The main
fields are:

- `fp_checkpoint`: float surgery/distill checkpoint to wrap.
- `model_key`: adapter key, default `deit_tiny_pet`.
- dataset/loader fields shared with the adapter, such as `data_dir`, `batch_size`, `workers`,
  `randaugment`, `ra_magnitude`, `random_erasing_prob`.
- runtime fields: `device`, `log_dir`. Surgery compute dtype is taken from the float checkpoint
  (``extra["surgery_dtype"]`` on the checkpoint).
- calibration uses the full training loader (aggregated streaming statistics; no sampled-batch cache).
- selection fields: `wrap_linear_conv`, `wrap_affine`, `wrap_matmul`, `include_names`,
  `exclude_names`.
- quantization fields: `weight_bits`, `activation_bits`, `compiled_accumulator_bits`,
  `matmul_activation_bits`, `per_output_channel`, `dequant_var_eps`.
  Bit widths of the integer codes entering each quantized node, in three groups: `activation_bits` for the
  `Linear`/`Conv2d` layers inherited from the source model; `compiled_accumulator_bits` for the compiled
  accumulators, the fixed affine nodes (`AffineScale`, `AffineScaleBias`, `AffineFixedMix`, `AffineContract` and its
  subclasses) that compilation introduces when it splits LayerNorm, softmax and attention; and
  `matmul_activation_bits` for `AffineMatMul`/`AffineHadamard`. The last two fall back to `activation_bits` when
  unset. In the strict graph the compiled accumulators hold squares and sums of squares, whose range exceeds 8 bits,
  so the strict configs (`configs/ptq/*_strict*.json`) set `compiled_accumulator_bits=12`; every other config sets 8,
  where it equals `activation_bits` and has no effect. The proxy accumulates every node exactly (fp32 on integer
  codes), so none of these fields limits an accumulator's width. `affine_activation_bits` is the deprecated name and
  is still accepted. `AffineCombine` nodes keep their coefficients exact (since 2026-10-11): each input is rescaled by
  its own multiplier before the sum, as in an integer Add with per-input multipliers. The strict baseline coefficients
  ([1, 1], [1, -1], [0.5, 0.5], [0.5, -0.5, -0.5]) were exact on the one-scale 8-bit grid before as well.
- `calibration_seed` (default 0): seed of the global RNGs before the loaders are built, so the calibration batches
  and their augmentation repeat; `null` is unseeded, as were all PTQ runs before 2026-10-11.
- `brent_scaling` (default false): Brent's operand scaling of the strict Mix and LayerNorm products. After the first
  range pass, `lam^2 = max|v| / max|p|` (inputs of `sub_vp`) and `lam^2 = max|a| / max|u|` (inputs of
  `add_u_invstd`), rounded to a power of two, are folded into the `AffineCombine` coefficients
  (`ptq.apply_brent_scaling`); the float graph computes the same function, and the ranges are measured again.
- optional architecture overrides: `top_k`, `eps`.
- `output`: desired checkpoint directory/name. The final filename is normalized from the tool and
  config name.

PTQ does not train the model and does not run QAT. It validates the float model, runs two
forward-only calibration passes (ranges, then dequant moments) on the full training loader,
constructs wrappers, reloads the saved wrapped checkpoint, and evaluates that checkpoint.

## Node Selection

`_build_node_selection` walks `model.named_modules()` and selects supported modules by type and
substring filters. `exclude_names` wins first. A node is selected when either its type is enabled by
the wrap flags or its name contains one of the `include_names` substrings.

Supported node kinds:

- `nn.Linear`, `nn.Conv2d` when `wrap_linear_conv=true`.
- `AffineScale`, `AffineScaleBias`, `AffineFixedMix`, `AffineContract` when `wrap_affine=true`.
- `AffineMatMul`, `AffineHadamard` when `wrap_matmul=true`.

Routing ops, nonlinear unary ops, top-k/gather wiring, dropout, and other non-affine modules are
not PTQ-wrapped.

## Calibration

Forward hooks on selected nodes accumulate statistics from full calibration minibatches. PTQ first
runs full validation (no hooks), then runs hooks over the entire training loader twice: ranges,
then accumulator/output moments for the dequant fit. Each pass is streaming (no activation cache
across batches).

Per node:

- tensor input ranges (every operand for matmul-like nodes);
- running residual or OLS moments for dequantization;
- aggregated example counts.

The float validation pass reports `fp_val_acc` and `fp_val_loss`. If any selected node receives no
statistics on the calibration passes, PTQ stops before writing a wrapped checkpoint.

## Quantization

Each selected module is replaced in a deep copy of the float model with
`CalibratedAffinePTQWrapper`.

Activations use signed symmetric max-abs quantization with one scale per input tensor:

```
scale = max(abs(x)) / qmax
zero_point = 0
```

Weights use the same signed symmetric estimator. For `Linear` and `Conv2d`,
`per_output_channel=true` uses one weight scale per output channel/filter along axis 0. When
`per_output_channel=false`, one global scale is used. Affine coefficient tensors always use one
global scale. `AffineMatMul` and `AffineHadamard` have no stored weight tensor.

The wrapper forward path quantizes inputs with proxy integer tensors, runs its explicit accumulator
submodule, then dequantizes back to the incoming float dtype.

## Dequant Fit

For `Linear` and `Conv2d`, the output scale is analytical:

```
out_scale = input_scale * weight_scale
```

Only output bias is calibrated as the mean residual between the float teacher output and the
scaled integer accumulator. The scale is baked into `q_weight` for these nodes, so the forward path
does not need a separate output multiply.

For surgery affine and matmul kinds, PTQ fits:

```
y_hat = out_scale * accumulator + out_bias
```

The fit is ordinary least squares, per output channel when the output shape exposes a channel axis,
otherwise global. `dequant_var_eps` is the variance threshold and denominator floor for this fit.
Channels with accumulator variance below this threshold collapse to slope 0 and use the teacher
channel mean as the output.

## Outputs

The CLI writes traceable artifacts based on the active config name:

- `artifacts/checkpoints/ts_ptq_<config>.pt`: PTQ checkpoint (same basename stem as config `output` intent).
- `artifacts/metadata/ts_ptq_<config>.json`: metadata path derived from the checkpoint basename.
- `artifacts/logs/ts_ptq_<config>_model_after_ptq.txt`: model structure log.

The checkpoint `extra` keeps the source checkpoint metadata plus:

- `ptq_meta_path`: metadata JSON basename.
- `ptq_wrappers`: wrapper skeleton configs needed to reload the PTQ checkpoint.

The metadata JSON records:

- source and output checkpoint paths.
- selection flags and selected node names.
- quantization bit widths and per-output-channel setting.
- calibration batch/example settings, sampled batch indices, and per-node examples by batch.
- float and PTQ validation accuracy/loss plus deltas.
- per-node quant/dequant parameters and aggregated calibration debug stats.

## Reload

`load_ptq_checkpoint` rebuilds the surgery template through the recorded adapter,
installs skeleton `CalibratedAffinePTQWrapper` modules from `ptq_wrappers`, and then loads the
checkpoint state dict strictly. The PTQ run validates this reloaded checkpoint, so saved artifacts
are checked through the same path downstream code uses.

## Results (Pet, 2026-10-10)

Weights W8; activations A8, except `compiled_accumulator_bits=12` on the compiled accumulators of the strict graph
(`configs/ptq/*_strict_sr*.json`). Strict rows quantize the surgery graph (`strict_unary=sigmoid_rsqrt`, no exp and no
log, norm expansion `strict_product="norm"`; final run 2026-10-11, `artifacts/logs/norm_construction_2026-10-10/run_final.log`)
before distillation; "sigmoid GELU" rows use `gelu=sigmoid` (x*sigmoid(1.702x)), the graph with only squares,
sigmoid and rsqrt that a CNN-class NPU runs (paper Table 3). Extended (fast) rows quantize the Jeffreys-distilled graph.
Per-tensor rows keep the output bias per channel when the weight scale is a scalar, as in standard per-tensor PTQ.
Reference LayerNorm eps (rerun 2026-10-09); logs in `artifacts/logs/eps_rerun_2026-10-09/`. Results of the
log-domain strict graph with the old eps are in git history and in `artifacts/backup_eps_2026-10-09/`.

| k | graph | nodes | bf16 | PTQ per-channel | Δ | PTQ per-tensor | Δ |
|---:|---|---:|---:|---:|---:|---:|---:|
| 16 | strict | 343 | 85.96 | 85.12 | -0.84 | 85.12 | -0.84 |
| 32 | strict | 343 | 89.13 | 88.69 | -0.44 | 88.66 | -0.46 |
| 64 | strict | 343 | 90.68 | 90.27 | -0.41 | 90.32 | -0.35 |
| 197 | strict | 331 | 91.25 | 90.76 | -0.49 | 91.28 | +0.03 |
| 16 | strict, sigmoid GELU | 379 | 83.97 | 83.35 | -0.63 | 82.09 | -1.88 |
| 32 | strict, sigmoid GELU | 379 | 87.60 | 87.57 | -0.03 | 87.63 | +0.03 |
| 64 | strict, sigmoid GELU | 379 | 89.62 | 88.99 | -0.63 | 89.59 | -0.03 |
| 197 | strict, sigmoid GELU | 367 | 90.54 | 89.86 | -0.68 | 90.13 | -0.41 |
| 16 | extended | 246 | 88.58 | 85.50 | -3.08 | 84.57 | -4.01 |
| 32 | extended | 246 | 90.19 | 86.73 | -3.46 | 87.19 | -3.00 |
| 64 | extended | 246 | 91.17 | 87.52 | -3.65 | 88.14 | -3.03 |
| 197 | extended | 234 | 91.74 | 88.83 | -2.91 | 88.20 | -3.54 |

Every surgery run records `student_pointwise_per_input` (unary evaluations per input, by op type) from
forward hooks on the `NL*` nodes (plus two ReLUs per entry for the log-domain strict LayerNorm). At full support the
`sigmoid_rsqrt` strict DeiT-Tiny count with the norm expansion is 196,498,438 per image = 12 x 16,368,533 (paper Table 1
block total) + 76,042 (final LayerNorm). Surgery calibration (tail mass, centroids) uses unseeded augmented train batches,
so two surgery runs of one config differ by up to 0.5 points (top-k) and 1.5 points (routed).

## Brent scaling, node-level check (2026-10-11)

`make ptq-balance-check` (`scripts/ptq_balance_check.py`, log `artifacts/logs/ptq_balance_2026-10-11/`, data
`artifacts/metadata/ptq_balance_check.json`) emulates PTQ on the float64 inputs of the strict Score, Mix and
LayerNorm of `topk{16,197}_strict_sr_gs` (8 calibration and 16 evaluation Pet images): every input of a compiled
affine node on its own symmetric max-abs grid of b bits, exact accumulation, exact coefficients, least-squares output
fit. Brent's scaling a -> lam a, b -> b/lam with lam^2 = max|b|/max|a|, one scalar per site and block. Relative error
of the site output, mean over blocks:

| site | median lam^2 | 8 bits plain / lam | 10 bits plain / lam | 12 bits plain / lam |
|---|---:|---:|---:|---:|
| Score | 0.97 | 4.25% / 4.23% | 1.14% / 1.14% | 0.41% / 0.41% |
| Mix, k=16 | 8.45 | 43.7% / 11.0% | 14.2% / 3.09% | 3.88% / 0.88% |
| Mix, k=197 | 5.73 | 72.4% / 37.0% | 27.8% / 9.97% | 8.15% / 2.86% |
| LayerNorm | 0.20 | 22.4% / 11.0% | 6.87% / 2.90% | 1.81% / 0.73% |

q and k are already balanced, so the Score gains nothing. In Mix and LayerNorm lam cuts the error 2 to 4.5 times at
every width, worth one to two bits. A model-level PTQ run must keep the combine coefficients exact (per-input
multipliers): the current PTQ quantizes `affine_contract` coefficients with one 8-bit scale, and [1/lam, -lam] with
lam^2 = 8.5 would lose about 3% on the smaller coefficient.

## Brent scaling, model level (2026-10-11)

`make ptq-brent` (lam on Mix and LayerNorm) and `make ptq-brent-mix` (`brent_sites=["mix"]`), configs
`configs/ptq/{16,64,197}_strict_sr_gs_acc{12,10}{,_brent,_brentmix}{,_pertensor}_8bit.json`, logs
`artifacts/logs/ptq_brent_2026-10-11/`. Strict graph with sigmoid GELU, seeded PTQ calibration (`calibration_seed=0`),
exact `AffineCombine` coefficients, power-of-two lam from max-abs ranges; one run per cell. PTQ accuracy change
(points) from the bf16 float model (83.97 / 89.62 / 90.54 at k = 16 / 64 / 197), PTQ loss in parentheses:

| k | weights | bits | no lam | lam: Mix + LayerNorm | lam: Mix only |
|---:|---|---:|---:|---:|---:|
| 16 | per-channel | 12 | -1.06 (0.564) | -1.42 (0.566) | -0.68 (0.549) |
| 64 | per-channel | 12 | -0.65 (0.361) | -0.38 (0.366) | -0.22 (0.365) |
| 197 | per-channel | 12 | -0.90 (0.336) | +0.05 (0.326) | -0.44 (0.329) |
| 16 | per-tensor | 12 | -0.76 (0.565) | -0.87 (0.564) | -0.35 (0.546) |
| 64 | per-tensor | 12 | -0.19 (0.362) | -0.38 (0.364) | -0.16 (0.364) |
| 197 | per-tensor | 12 | -0.49 (0.337) | +0.16 (0.327) | -0.76 (0.329) |
| 16 | per-channel | 10 | -4.61 (0.759) | -2.15 (0.631) | -2.15 (0.682) |
| 64 | per-channel | 10 | -4.39 (0.545) | -1.64 (0.441) | -2.73 (0.509) |
| 197 | per-channel | 10 | -8.12 (0.632) | -1.55 (0.395) | -3.52 (0.481) |
| 16 | per-tensor | 10 | -3.90 (0.758) | -1.53 (0.627) | -1.47 (0.680) |
| 64 | per-tensor | 10 | -5.10 (0.548) | -1.20 (0.443) | -2.67 (0.504) |
| 197 | per-tensor | 10 | -7.85 (0.617) | -1.80 (0.399) | -3.08 (0.490) |

Folding lam changes the bf16 float model before PTQ: Mix + LayerNorm 83.18 / 89.45 / 90.30, Mix only 83.81 / 89.62 /
90.46. The bf16 `AffineCombine` rounds each partial sum to bf16, and the LayerNorm lam (0.25 to 0.5) leaves a large
a^2/lam^2 term in the partial sum. With 10-bit compiled accumulators lam on both sites cuts the PTQ loss from 3.9-8.1
to 1.2-2.2 points; with 12 bits the means are -0.68 (no lam), -0.47 (both), -0.44 (Mix only).
