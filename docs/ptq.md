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
- calibration fields: `calibration_batches` (each sampled minibatch contributes all examples per node).
- selection fields: `wrap_linear_conv`, `wrap_affine`, `wrap_matmul`, `include_names`,
  `exclude_names`.
- quantization fields: `weight_bits`, `activation_bits`, `affine_activation_bits`,
  `matmul_activation_bits`, `per_output_channel`, `dequant_var_eps`.
- optional architecture overrides: `top_k`, `eps`.
- `output`: desired checkpoint directory/name. The final filename is normalized from the tool and
  config name.

PTQ does not train the model and does not run QAT. It validates the float model, runs two
forward-only calibration passes (ranges, then dequant moments) on the sampled batches, constructs
wrappers, reloads the saved wrapped checkpoint, and evaluates that checkpoint.

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
runs full validation (no hooks), samples up to `calibration_batches` validation indices with
`torch.randperm`, then runs hooks over those minibatches twice: ranges, then accumulator/output
moments for the dequant fit. Each pass is streaming (no activation cache across batches).

Per node:

- tensor input ranges (every operand for matmul-like nodes);
- running residual or OLS moments for dequantization;
- batch indices and example counts (for metadata).

The float validation pass reports `fp_val_acc` and `fp_val_loss`. If any selected node receives no
statistics on the sampled batches, PTQ stops before writing a wrapped checkpoint.

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
