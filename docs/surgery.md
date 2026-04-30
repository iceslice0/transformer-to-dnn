# Surgery

Surgery is the first stage of the pipeline. It loads a pretrained classifier through a model
adapter, builds a surgery student in a restricted op vocabulary (affine + local unary + selection
/routing), copies reference weights into the student, runs a calibration pass, and saves the
pre-finetune checkpoint with metadata and model-structure logs.

The core code lives in [src/transformer_surgery/surgery.py](../src/transformer_surgery/surgery.py)
and the op vocabulary in [src/transformer_surgery/ops.py](../src/transformer_surgery/ops.py). The CLI
wrapper is `python -m transformer_surgery.cli.surgery` or `ts-surgery`.

## Inputs

- `reference_checkpoint`: pretrained teacher checkpoint, resolved through the model adapter.
- `model_key`: adapter key, default `deit_tiny_pet`.
- dataset/loader fields shared with the adapter, such as `data_dir`, `batch_size`, `workers`,
  `randaugment`, `ra_magnitude`, `random_erasing_prob`.
- surgery shape/numerics fields: `top_k`, `eps`, `gibbs_tail_prob_eps`,
  `disable_calib_gibbs_tail_prob`, `surgery_dtype`.
- transform flags: `disable_layernorm_replacement`, `disable_attention_surgery`,
  `disable_softmax_replacement`, `allow_matmul`.
- output paths: `pre_ft_checkpoint`, `log_dir`.

Surgery code stays model-agnostic. It uses `get_model_adapter` so dataset
loaders, reference loading, surgery-model construction, weight copy, and calibration diagnostics
stay inside the adapter layer. Default adapter:

- patient: DeiT-Tiny from timm, fine-tuned on Oxford-IIIT Pet.
- dataset: Oxford-IIIT Pet classification (37 classes).
- surgery model: `DeiTTinySurgeryModel` in
  [src/transformer_surgery/models/deit_tiny.py](../src/transformer_surgery/models/deit_tiny.py).

Extra models or datasets are added by registering a new adapter, not by branching the surgery
driver.

## Op Vocabulary

The surgery student is built only from explicit `nn.Module` classes in three groups. All names
match exports from `transformer_surgery.ops`.

### Affine

Everything linear, including parameterized layers, fixed-coefficient contractions, and linear
unary reductions/scalings:

- `nn.Linear`, `nn.Conv2d` (patch embed) — standard parameterized layers.
- `AffineScaleBias` — per-channel `y = x * weight + bias`; carries LayerNorm `gamma`/`beta`.
- `AffineContract(einsum_equation, coeffs)` — fixed coefficient contraction over operand or head
  dims via `torch.einsum` on a registered `coeff` buffer. One graph node per logical affine.
- `AffineFixedMix(einsum_equation, matrix)` — fixed 2×2 mix on the operand axis (default builds
  plus/minus combinations for the square identity).
- Linear unary reductions and scalings: `AffineMean`, `AffineSum`, `AffineScale`.
- Variable bilinear ops (`allow_matmul` only): `AffineMatMul` (`torch.matmul(a, b)`) and
  `AffineHadamard` (elementwise `a * b`). Both are linear in each operand separately and so
  belong to the affine group; they are gated by `allow_matmul` and not registered on the
  strict path.

Binary mixes (residual, positional embedding, centering, Gibbs intermediates) have no separate
"add module". Callers route operands with `torch.broadcast_tensors` followed by
`torch.stack(..., dim=-1)` and feed the result into `AffineContract("i,...i->...", [w0, w1])`.
The graph shows an `AffineContract` child; the stack is plain tensor wiring in `forward`.

### Nonlinear Unary

Strictly nonlinear scalar maps:

- Exact unaries: `NLSquare`, `NLExp`, `NLLogPlusEps`, `NLSqrtExp`,
  `NLRsqrtPlusEps`, `NLReciprocalPlusEps`.
- Trainable PWL: `NLScalarPWL` (learnable knot values, fixed knot positions);
  `NLGELU` wraps it with default knots on `linspace(-4, 4, PWL_NUM_KNOTS)` initialized to
  `F.gelu(knots)`.

`NLLogPlusEps`, `NLRsqrtPlusEps`, and `NLReciprocalPlusEps` carry a fixed `eps` buffer
set from the run config and frozen at training time by `freeze_eps_parameters`.

### Routing

Pure tensor wiring and discrete selection — no arithmetic, no parameters:

- Shape and layout: `reshape`/`view`, `transpose`, `expand`, `cat`, `stack`,
  `broadcast_tensors`, `unsqueeze`.
- Discrete selection: `torch.topk`, `torch.gather`, `torch.max` for row max (Gibbs
  stabilization), `F.relu` (sign split in the LN strict path), `Dropout`/`DropPath`.

`F.relu` and `torch.max` are classified as routing — they choose between operands or zero
rather than computing a smooth nonlinearity, so they live with `topk` and `gather`, not with
`NLSquare`/`NLExp`.

The routing surface is also re-exported from `transformer_surgery.ops` under `Routing*`
aliases (`RoutingTranspose`, `RoutingCat`, `RoutingStack`, `RoutingUnsqueeze`, `RoutingExpand`,
`RoutingExpandAs`, `RoutingSqueeze`, `RoutingBroadcastTensors`, `RoutingTopK`, `RoutingGather`,
`RoutingMax`, `RoutingFullLike`, `RoutingReLU`, `RoutingDropPath`) so the routing vocabulary is
discoverable alongside the `Affine*` and `NL*` classes. The op implementations use these aliases
for routing inside `ops.py`; the aliases are names for tensor wiring, not extra graph modules.

## Transform 1: LayerNorm Rewrite

`RewrittenLayerNorm` (in `transformer_surgery.ops`) replaces `nn.LayerNorm`.

Common prefix:

- `mu = AffineMean(-1, keepdim=True)(x)`.
- `u = AffineContract("i,...i->...", [1, -1])(stack(x, mu, dim=-1))` (centering, no add module).
- `r2 = AffineMean(-1, keepdim=True)(NLSquare()(u))`.

Strict path (`allow_matmul=False`, default):

- `au = stack(F.relu(u), F.relu(-u), dim=-2)` — magnitude split, no `abs`.
- `log_num = NLLogPlusEps(eps)(au)`.
- `log_den = NLLogPlusEps(eps)(r2)` broadcast to match `log_num`.
- `t = AffineContract("q,...qpc->...pc", [2, -1])(stack(log_num, log_den, dim=2))`.
- `a_mag = NLSqrtExp()(t)` (= `sqrt(exp(t))`, equivalent to `exp(log_num - ½·log_den)`).
- `z = AffineContract("p,...pc->...c", [1, -1])(a_mag)`.

Fast path (`allow_matmul=True`):

- `inv_std = NLRsqrtPlusEps(eps)(r2)`.
- `z = AffineHadamard()(u, inv_std)`.

Final affine: `AffineScaleBias(normalized_shape)` carries `gamma`/`beta`. Weights are copied
from the timm `nn.LayerNorm` by `copy_ln_params_to_rewritten`. `eps` is fixed at construction
from `cfg.eps` and frozen by `freeze_eps_parameters` (it is not copied from `src.eps`).

## Transform 2: Variable Matrix-Multiplication Replacement in Attention

Attention is implemented in `SurgeryAttention` (in
[src/transformer_surgery/models/deit_tiny.py](../src/transformer_surgery/models/deit_tiny.py)).

In strict mode (`allow_matmul=False`) there is no variable `matmul` for QKᵀ score computation
or for sparse value mixing PV. Both use the identity `a*b = ((a+b)² - (a-b)²)/4` through
`SquareIdentityOperandChain`:

```
stack((a, b), dim=-2)
    -> AffineFixedMix(plus/minus 2x2)
    -> NLSquare()
    -> AffineContract(coeffs)
```

Two attention modules use the chain with different einsum equations and coefficients:

- `PairwiseDotBySquare(head_dim)` — Q/K grid → scores. Mix `pq,...qd->...pd`,
  contract `p,...pd->...`, coeffs `±1/(4√d)`. Operands are broadcast Q/K.
- `SparseWeightedSumBySquare()` — sparse probs and gathered values → output. Mix
  `pq,...kqd->...kpd`, contract `p,...kpd->...d`, coeffs `(¼, −¼)`. Operands are expanded probs
  and `torch.gather`-ed values per top-k slot.

When `allow_matmul=True`, `PairwiseDotBySquare` falls back to `(q/√d) @ kᵀ` via `AffineMatMul`, and
`SparseWeightedSumBySquare` falls back to `probs.unsqueeze(-2) @ v_g` via `AffineMatMul`. Only the
modules for the chosen mode are registered.

## Transform 3: Softmax Replacement

`GibbsTopKSoftmax(seq_len, top_k, eps, gibbs_tail_prob_eps, allow_matmul)` replaces dense softmax. It
returns sparse probabilities on the top-k indices, the index tensor, and a tail-mass scalar
`q_tail`.

Score stabilization, tail-sum aggregation, and logit normalization are exposed as
`AffineContract` submodules so the graph lists explicit affine nodes:

- `scores_stable_contract` (`[1, -1]`) implements `scores - row_max` over a stacked operand.
- `gibbs_tail_prob_eps` is a scalar module parameter that reserves omitted-tail probability mass.
- `scale_top_probs_by_tail` (`AffineHadamard`) applies `1 - gibbs_tail_prob_eps` to the top-k
  probabilities.
- `logit_logz_contract` (`[1, 1]`) implements `vals + (-log_z)`.

Tail term:

- The total omitted-tail probability is the `gibbs_tail_prob_eps` parameter when `N > K`.
- When `N <= K`, `q_tail` is a routed zero tensor from `RoutingFullLike`.

Normalization:

- Strict: `top_probs = exp(vals - log(sum_exp + eps))` via `NLLogPlusEps` + `AffineContract`
  + `NLExp`, then `probs = AffineHadamard(top_probs, 1 - gibbs_tail_prob_eps)`. No `/`
  operator.
- Fast: `top_probs = AffineHadamard()(exp(vals), NLReciprocalPlusEps(eps)(sum_exp))`, then
  `probs = AffineHadamard(top_probs, 1 - gibbs_tail_prob_eps)`.

`eps` is the normalizer floor used in `log(sum_exp + eps)` / `1/(sum_exp + eps)`.
`gibbs_tail_prob_eps` is an `nn.Parameter` initialized from config, saved in the checkpoint state
dict, overwritten by calibration, and clamped to `[0, 1]` in `forward` before applying it as
probability mass.

## Transform Flags

The CLI config (in
[src/transformer_surgery/cli/surgery_config.py](../src/transformer_surgery/cli/surgery_config.py))
exposes four debug flags that selectively bypass the rewrites:

- `disable_layernorm_replacement` — keep `nn.LayerNorm` instead of `RewrittenLayerNorm`.
- `disable_attention_surgery` — keep dense scaled-dot QKᵀ + softmax + dense `@V`.
- `disable_softmax_replacement` — keep `PairwiseDotBySquare` for QKᵀ but use full softmax then
  dense `@V` (or a Hadamard expansion when `allow_matmul=True`).
- `allow_matmul` — switch the strict subgraphs to the matmul/Hadamard fast paths in
  `RewrittenLayerNorm`, `PairwiseDotBySquare`, `SparseWeightedSumBySquare`, and
  `GibbsTopKSoftmax`.

`build_module_mapping` records the resulting per-block module choices (`norm1`, `attn`,
`norm2`, `mlp.act`, `fc_norm`) into the surgery metadata.

## Calibration

`adapter.calibrate_reference` runs once on a single validation minibatch and reports adapter
diagnostics. For `deit_tiny_pet` these are:

- `ln_rewrite_mse_layer0_minibatch` — MSE of `RewrittenLayerNorm` against the reference LN at
  block 0 on the cached batch.
- `ln_rewrite_mse_all_norms_mean` — mean MSE across all `norm1`/`norm2` and the final `norm`
  while replaying the residual stream of the reference model.
- `jeffreys_gibbs_mean_cached` and `jeffreys_naive_mean_cached` — mean Jeffreys divergence of
  the Gibbs Top-K and naive Top-K approximations against the dense softmax on cached attention
  scores from block 0. Gibbs uses the block-0 calibrated `gibbs_tail_prob_eps` value for this
  metric.
- `jeffreys_improvement_naive_minus_gibbs_cached` — signed gap between naive and Gibbs Top-K under
  the active tail behavior.
- `gibbs_tail_prob_eps_calibrated_by_block` — per-block estimates of the true omitted dense-softmax
  probability mass outside the top-k set on the calibration minibatch.
- `gibbs_tail_prob_eps_calibrated_mean` plus per-block min/max variants — summary statistics for
  those omitted-tail estimates. The surgery CLI prints the by-block values.
- `gibbs_tail_prob_eps_applied_by_block` — values copied into each `GibbsTopKSoftmax` scalar
  parameter.
- `disable_calib_gibbs_tail_prob` — when true, the by-block tail estimate and parameter copy are
  skipped; metrics use the configured `gibbs_tail_prob_eps`.
- Synthetic `*_synthetic` variants on a Gaussian score tensor of the same shape, for
  cross-checking.

The surgery driver then runs full-validation passes on both the reference (`ref_val_acc`,
`ref_val_loss`) and the freshly built surgery student (`student_pre_ft_val_acc`,
`student_pre_ft_mean_ce`) and adds them to the calibration block.

## Outputs

The CLI writes:

- `artifacts/checkpoints/ts_surgery_<config>.pt` — pre-finetune surgery checkpoint
  (`save_model_checkpoint` with adapter-supplied `extra`).
- `artifacts/metadata/ts_surgery_<config>.json` — `SurgeryMeta` JSON with `model_key`,
  `patient`, `dataset`, `eps`, `top_k`, `surgery_dtype`, `pwl`, `calibration`, `module_mapping`,
  `reference_checkpoint`, `allow_matmul`, `gibbs_tail_prob_eps`, and
  `disable_calib_gibbs_tail_prob`.
- `artifacts/logs/ts_surgery_<config>_model_before_surgery.txt` and `_model_after_surgery.txt`
  — `write_model_structure_txt` dumps with `repr(model)`, parameter counts, the
  `named_modules` listing, and per-module forward output tensor shapes from one `eval` pass on
  a dummy batch.

Checkpoint `extra` records the adapter key, reference checkpoint, metadata basename, mapping,
top_k, eps_ln, the applied-mean `gibbs_tail_prob_eps`, config JSON path, all transform flags, and
the surgery dtype, so the surgery student can be reconstructed by `build_surgery_model_from_extra`
without the original config. Per-block calibrated `gibbs_tail_prob_eps` values live in the
checkpoint state dict, checkpoint `extra`, and the metadata calibration block.
When `disable_calib_gibbs_tail_prob` is true, the configured scalar remains in the checkpoint state
dict and no per-block applied values are written.

## Validation Expectations

The surgery stage is expected to satisfy:

1. It loads the reference and builds the student through the adapter and does not depend on a
   concrete model module.
2. With strict flags (no `disable_*`, `allow_matmul=False`), the student contains no
   `nn.LayerNorm` in surgery-configured blocks, no dense softmax in attention, and no variable
   `torch.matmul` or `AffineHadamard` in attention score or value paths.
3. Binary mixes (residual, positional embedding, centering, Gibbs intermediates) appear as
   `AffineContract` after `stack`, not inside opaque add modules.
4. Calibration reports LN MSE, calibrated `gibbs_tail_prob_eps` values, and Gibbs-vs-naive Jeffreys
   metrics for the fixed probability tail behavior.
5. `eps` is a fixed normalizer floor from config. `gibbs_tail_prob_eps` is a scalar module
   parameter and does not multiply `exp(s_K)`.
6. The post-transform student validates reasonably close to the reference under the chosen
   flags.
7. The CLI writes the checkpoint, metadata JSON, and before/after structure logs for the run
   config.
