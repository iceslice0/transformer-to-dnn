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
- quantization fields: `weight_bits`, `activation_bits`, `compiled_affine_input_bits`,
  `matmul_activation_bits`, `per_output_channel`, `dequant_var_eps`.
  Bit widths of the integer codes entering each quantized node, in three groups: `activation_bits` for the
  `Linear`/`Conv2d` layers inherited from the source model; `compiled_affine_input_bits` for the fixed affine nodes
  (`AffineScale`, `AffineScaleBias`, `AffineFixedMix`, `AffineContract` and its subclasses): the nodes that
  compilation adds when it splits LayerNorm, softmax and attention, and the residual and position-embedding
  additions; and `matmul_activation_bits` for `AffineMatMul`/`AffineHadamard`.
  The last two fall back to `activation_bits` when unset. Every width is an input-code width; no field sets the
  width of an accumulator. The strict configs use `compiled_affine_input_bits=12`, since the inputs of these
  nodes include squares and sums of squares; bit-width sweep configs also test other widths. This override affects only selected compiled
  fixed-affine nodes. It does not change weight precision or the Linear/Conv and MatMul/Hadamard input widths.
  Quantization is applied once per boundary, at the consuming node's input. `AffineCombine` nodes keep their coefficients exact (since 2026-10-11): each input is rescaled by
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

Routing ops, nonlinear unary ops, plain sums and means (`AffineSum`, `AffineMean`), top-k/gather wiring, dropout,
and other non-affine modules have no quantizer of their own; they sit inside the integer units described below.
A plain sum or mean adds integer codes with fixed scales (or their squares), so its result is an integer with a fixed
scale. It keeps its full integer range and is not rounded to a specific bit width; the next quantizer sets the width.

## Calibration

Forward hooks on selected nodes accumulate statistics from full calibration minibatches. PTQ first
runs full validation (no hooks), then runs hooks over the entire training loader twice: ranges,
then accumulator/output moments for the dequant fit. Each pass is streaming (no activation cache
across batches).

Per node:

- tensor input ranges (every operand for matmul-like nodes);
- running residual or OLS moments for dequantization;
- aggregated example counts.

The float validation pass reports `fp_val_acc` and `fp_val_loss`. The float model is the reference: it keeps the
checkpoint's surgery dtype (bf16 under autocast, as in every other experiment) through both calibration passes, so the
ranges come from its activations and the output maps are fitted against its outputs. Inside each calibration hook the
node's quantized accumulation runs in fp32 with autocast off. Only the copy that becomes the PTQ model moves to fp32. If
any selected node receives no statistics on the calibration passes, PTQ stops before writing a wrapped checkpoint.

### fp32 simulation and its checks (since 2026-10-17)

Before 2026-10-17 the PTQ runs used the checkpoint's bf16 autocast. Autocast runs `einsum`, `linear`, `matmul` and
`conv` in bf16, which holds integers exactly only up to 256: it rounded 12-bit input codes inside the fixed affine
nodes, and every wrapper cast its output back to bf16 before the next quantizer. The quantized model and the quantized
accumulations of calibration now run in fp32, which holds every integer up to 2^24, and this is checked at run time
(`ops.ptq_check_fp32`); the float reference stays bf16:

- every quantizer input and every accumulation is a float32 tensor, with CPU and CUDA autocast off;
- every floating parameter and buffer of the PTQ model is float32 (`ptq.to_fp32_simulation`);
- the calibration pass records, per node with integer inputs, the largest `sum_i |q_i| |w_i|` over the data
  (`integer_sum_bound` in the metadata), which bounds every partial sum of the integer accumulation; PTQ stops if it
  exceeds 2^24 (`check_ptq_integer_sums`). `AffineCombine` rescales its codes before the sum (an integer Add with
  per-input multipliers) and has no bound.

## Model output

With `quantize_output=true`, PTQ routes the model output through an identity node `ptq_output` (`OutputIdentity`,
added by a forward hook at PTQ time, so the surgery checkpoint is unchanged). Its input quantizer is the output
quantizer of the last node: the model output (the logits of DeiT-Tiny) becomes `output_bits` codes (default
`activation_bits`) with one symmetric scale, and the identity's output map returns them as a fixed affine map of these
codes. The run also validates the reloaded PTQ model with the identity bypassed and records it as
`ptq_val_acc_unquantized_output`, so the cost of quantizing the output is measured in the same run. Without it, the last
node's output map applied to its integer sum is the model output, as in the PTQ protocols that quantize the weights
and inputs of every linear layer (QDrop, PTQ4ViT, RepQ-ViT).

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

The wrapper forward path quantizes inputs to integer codes held in float32 tensors, runs its explicit accumulator
submodule on them, and applies its output map; everything stays in fp32.

### Meaning of the 12-bit setting

The fixed affine nodes of the strict graph take 12-bit inputs:
`compiled_affine_input_bits=12` sets the input grid of those nodes. Quantization sits at each selected node's input:

```text
integer input codes -> integer products and sums -> output map (scale, bias)
      -> optional unary/routing/sum/mean nodes -> quantize to the next selected node's integer codes
```

A consumer's input quantizer is also its producer's output quantizer, so the graph between two quantizers is a unit
with integer inputs and integer outputs. Without an intervening unary node, the output map and the next quantizer
form the requantization of integer inference (a fixed multiplier and offset).

In the strict configs, selected compiled fixed-affine nodes use 12-bit input codes;
Linear/Conv and MatMul/Hadamard nodes use their separately configured widths, normally 8 bits.
Weights keep `weight_bits`, normally 8. A consumer's width is not inherited from its producer.
At a boundary between quantized nodes, the producer's output is the consumer's input. The consumer
quantizes that boundary once, on its own grid; a second quantizer at the producer's output would
duplicate the boundary operation. When unary or routing nodes intervene, the next selected node's
input quantizer follows those operations. The simulation holds
integer codes in float32 tensors and evaluates each unit in fp32, where integer sums up to 2^24 are exact.
See `ptq_activation_bits` for group selection and `CalibratedAffinePTQWrapper.forward`
for quantizer placement.

## Dequant Fit

For `Linear` and `Conv2d`, the output scale is analytical:

```
out_scale = input_scale * weight_scale
```

Only output bias is calibrated as the mean residual between the float teacher output and the
scaled integer accumulator. The accumulator keeps the integer weight codes, so it computes an exact integer sum, and
the output map applies `out_scale` and the bias. Checkpoints written before 2026-10-17 fold `out_scale` into
`q_weight` (`skip_out_scale=true` in `ptq_wrappers`, `out_scale_baked_into_weight` in the metadata); they still load.

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
- `ptq_output_identity`: whether the run added the output identity (`quantize_output`).
- `ptq_simulation_dtype`: `float32`.

The metadata JSON records:

- source and output checkpoint paths.
- selection flags and selected node names.
- quantization bit widths and per-output-channel setting.
- calibration settings (train loader, seed, Brent scaling and its folded `lam` per site).
- float reference validation accuracy/loss (`fp_*`, surgery dtype), PTQ accuracy/loss (with `quantize_output`, also
  with the output identity bypassed), and deltas against the reference.
- per-node quant/dequant parameters (input and weight scales, output scale and bias) and `integer_sum_bound`.

## Reload

`load_ptq_checkpoint` rebuilds the surgery template through the recorded adapter in its surgery dtype,
adds the output identity when `ptq_output_identity` is set, installs skeleton `CalibratedAffinePTQWrapper` modules from
`ptq_wrappers`, moves the model to fp32, and then loads the checkpoint state dict strictly. The PTQ run validates this reloaded checkpoint, so saved artifacts
are checked through the same path downstream code uses.

## Results (Pet, 2026-10-18: fp32 simulation, bf16 reference, 8-bit logits, new Pet reference)

Rerun 2026-10-18 in the leak-fix batch (`artifacts/logs/leakfix_2026-10-17/`): the Pet reference and the distilled
checkpoints are selected on the training split only (`docs/distill.md`, Checkpoint selection). The extended rows were
rerun on 2026-10-18 after the switch to training-accuracy selection (`artifacts/logs/leakfix_accsel_2026-10-18/`).

All 24 configs of this batch (`make ptq-paper`) set `quantize_output: true` (8-bit logits, `output_bits` default). The float reference is the bf16
model (calibration under autocast); the PTQ model runs in fp32 with the checks above. Largest integer sum bound over all
runs: 4.39e6, against 2^24 = 1.68e7.
(the batch with the wrong fp32 reference is kept as `run_fp32_reference_INVALID.log`); the previous artifacts are in
`artifacts/backup_ptq_fp32_2026-10-17/`. "out f" is the PTQ model with the output identity bypassed (logits not quantized);
quantizing the logits moves accuracy by -0.16 to +0.30 points. The extended k=197 rows were rerun after the k=197 surgery and distill
reruns of the same day (`artifacts/logs/sync_fullsupport_2026-10-17/`), which use the dense full-support value mix. Strict with exact GELU matches no NPU
configuration and is a side comparison only.

| k | graph | nodes | bf16 | PTQ per-channel | Δ | out f | PTQ per-tensor | Δ | out f |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | strict, sigmoid GELU | 380 | 84.08 | 82.61 | -1.47 | 82.72 | 83.02 | -1.06 | 83.10 |
| 32 | strict, sigmoid GELU | 380 | 87.98 | 87.30 | -0.68 | 87.08 | 87.27 | -0.71 | 87.11 |
| 64 | strict, sigmoid GELU | 380 | 89.78 | 89.72 | -0.05 | 89.78 | 89.64 | -0.14 | 89.62 |
| 197 | strict, sigmoid GELU | 368 | 90.73 | 90.27 | -0.46 | 90.35 | 89.75 | -0.98 | 89.89 |
| 16 | extended | 247 | 88.58 | 86.13 | -2.45 | 86.05 | 84.90 | -3.68 | 85.06 |
| 32 | extended | 247 | 90.19 | 86.26 | -3.92 | 86.32 | 86.54 | -3.65 | 86.62 |
| 64 | extended | 247 | 90.76 | 87.44 | -3.33 | 87.44 | 87.16 | -3.60 | 87.11 |
| 197 | extended | 235 | 91.36 | 88.74 | -2.62 | 88.69 | 88.50 | -2.86 | 88.53 |
| 16 | strict, GELU (no NPU configuration) | 344 | 86.37 | 84.93 | -1.44 | 84.63 | 85.47 | -0.90 | 85.42 |
| 32 | strict, GELU (no NPU configuration) | 344 | 89.07 | 88.31 | -0.76 | 88.25 | 88.39 | -0.68 | 88.36 |
| 64 | strict, GELU (no NPU configuration) | 344 | 90.35 | 89.86 | -0.49 | 89.89 | 90.41 | +0.05 | 90.24 |
| 197 | strict, GELU (no NPU configuration) | 332 | 91.06 | 90.98 | -0.08 | 91.06 | 91.25 | +0.19 | 91.28 |

## Results (Pet, 2026-10-10, bf16 simulation; superseded by the fp32 rerun of 2026-10-17)

Weights W8; input codes A8, except `compiled_affine_input_bits=12` on the compiled fixed affine nodes of the strict graph
(`configs/ptq/*_strict_sr*.json`). Strict rows quantize the surgery graph (no exp and no
log, norm expansion `strict_product="norm"`; final run 2026-10-11, `artifacts/logs/norm_construction_2026-10-10/run_final.log`)
before distillation; "sigmoid GELU" rows use `gelu=sigmoid` (x*sigmoid(1.702x)), the graph with only squares,
sigmoid and rsqrt that a CNN-class NPU runs. Extended (fast) rows quantize the Jeffreys-distilled graph.
Per-tensor rows keep the output bias per channel when the weight scale is a scalar, as in standard per-tensor PTQ.
Reference LayerNorm eps (rerun 2026-10-09); logs in `artifacts/logs/eps_rerun_2026-10-09/`.

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
forward hooks on the `NL*` nodes. At full support the
strict DeiT-Tiny count with the norm expansion is 196,498,438 per image = 12 x 16,368,533 (per
block) + 76,042 (final LayerNorm). Surgery calibration (tail mass, centroids) uses unseeded augmented train batches,
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
a^2/lam^2 term in the partial sum. With 10-bit input codes on the compiled affine nodes lam on both sites cuts the PTQ loss from 3.9-8.1
to 1.2-2.2 points; with 12 bits the means are -0.68 (no lam), -0.47 (both), -0.44 (Mix only).
