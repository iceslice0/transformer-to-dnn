Goal

Create a pure Python + PyTorch surgery pipeline that takes pretrained DeiT-Tiny and rewrites it into the pseudo-hardware basis

  { affine + local multi-tail PWL epilogue, selection/routing ops }

with the following properties:

1. The transformed model is tangible and reusable:
   - explicit Python model class
   - pre-finetune checkpoint
   - post-finetune checkpoint

2. The transformed model is close to the original by construction, not merely structurally similar.

3. The transformed graph exposes all affine computation explicitly so that QAT can later quantize only affine nodes.

4. The transform must include:
   - exact or near-exact LayerNorm conversion
   - explicit variable matrix multiplications in attention using only affine + square/PWL + affine reductions
   - Gibbs TopK softmax replacement with implicit replicated tail normalization

5. Fine-tune all model parameters except eps for a few iterations on Oxford-IIIT Pet and save final checkpoint.

6. The final artifacts must be directly usable as input to QAT later.

Input

Nothing.

Output

1. deit_tiny_surgery_model.py
   The transformed model class with explicit node types.

2. surgery_utils.py
   Utilities for discovery, calibration, replacement, validation, and checkpoint export.

3. run_deit_tiny_surgery.py
   End-to-end script:
   - load model + dataset
   - validate original
   - transform model
   - validate transformed
   - save transformed model + checkpoint
   - fine-tune
   - save final checkpoint

4. surgery_pre_ft.pt
   Checkpoint after transform, before fine-tuning.

5. surgery_post_ft.pt
   Checkpoint after short fine-tuning.

6. surgery_meta.json
   Metadata:
   - patient = DeiT-Tiny
   - dataset = Oxford-IIIT Pet
   - eps
   - top_k
   - PWL knees
   - calibration settings
   - module mapping

Patient and dataset

Patient:
- DeiT-Tiny from timm, pretrained

Dataset:
- Oxford-IIIT Pet classification
- standard train/validation split
- standard DeiT preprocessing

Pseudo-hardware graph grammar

The transformed graph may contain only:

A. Affine nodes
Definition:
- any map x -> A x + b

This includes:
- Linear
- Conv / 1x1 Conv if needed
- residual add
- sum
- mean
- avgpool
- scalar multiply
- negation
- identity
- channelwise affine gamma/beta application
- fixed selection matrices when indices are static
- fixed scaling such as 1/sqrt(d)
- explicit projection adapters introduced by the surgery

B. Local multi-tail PWL epilogue nodes
Definition:
- one affine node produces a widened accumulator
- a local epilogue applies one or more unary PWL maps to that same accumulator
- each tail emits a quantized-style output tensor
- tails are local to the same affine output
- no external accumulator fan-out beyond the local epilogue

C. Selection/routing ops
Allowed selection/routing ops are exactly:
- reshape / view
- flatten / unflatten
- transpose / permute
- split / chunk
- concat
- copy
- broadcast / expand
- pack / unpack tuples
- static slice / static index select
- dynamic top-k support selection
- comparator / compare-and-select
- mux / demux / swap
- gather / scatter by selected support
- max / min as selection/routing primitives
- argmax / top-k indices
- routing of associated values/tuples together with selected scores
- abs
- setsign(x, y) = sign(x) * y

Important:
- max belongs to selection/routing ops
- abs belongs to selection/routing ops
- setsign belongs to selection/routing ops
- no hidden arithmetic is allowed inside selection/routing ops except comparison/select/sign/routing semantics
- all arithmetic must be explicit in affine nodes or local PWL epilogues

Global transformation rule

The transformed model must have no implicit:
- LayerNorm
- dense softmax
- torch.matmul for attention scores
- torch.matmul for attention-value mixing
- hidden affine reductions buried inside library calls

Everything must be lowered into explicit:
- affine nodes
- local multi-tail PWL epilogues
- selection/routing ops

Strict hardware convention

1. Only requantized outputs are allowed to fan out globally.
2. Wide accumulator fan-out through memory is forbidden.
3. If one affine output must feed several unary transforms, this must happen only inside one local multi-tail PWL epilogue.
4. All branching outside local epilogues must occur on quantized-style outputs.

Specific conversion instructions

1. LayerNorm conversion

Replace each LayerNorm with the following exact or near-exact graph.

Given input x, define:

  mu = mean(x, dim=-1, keepdim=True)
  u  = x - mu
  r2 = mean(u * u, dim=-1, keepdim=True)

Then use the efficient abs + setsign normalization law:

  a = exp( log(abs(u) + eps) - 0.5 * log(r2 + eps) )
  z = setsign(u, a)
  y = beta + gamma * z

Interpretation:
- a is the magnitude of the normalized output
- setsign(u, a) restores the sign of u
- this is algebraically equivalent to:
    z = u / sqrt(r2 + eps)
  up to the chosen eps convention

Requirements:
- eps is explicit and fixed
- gamma and beta are explicit affine parameters copied from the original LayerNorm
- mean and r2 must be explicit affine/reduction nodes
- abs is explicit and classified as selection/routing
- log and exp must be explicit unary PWL tails
- setsign is explicit and classified as selection/routing
- no torch.nn.LayerNorm remains in the transformed model

Validation requirement:
- on a held-out minibatch, transformed LN outputs must match original LN outputs very closely before any fine-tuning
- log per-layer LN rewrite MSE

2. Variable matrix multiplication in attention

No torch.matmul / einsum may remain in transformed attention for:
- score computation Q K^T
- value mixing P V

Replace variable products using the exact square identity:

  a * b = ((a + b)^2 - (a - b)^2) / 4

2.1 Attention score computation

Original score:
  s_ij = ( q_i^T k_j ) / sqrt(d)

Convert each scalar product term q_i[l] * k_j[l] by:

  q_i[l] * k_j[l]
    = ( (q_i[l] + k_j[l])^2 - (q_i[l] - k_j[l])^2 ) / 4

Operational form:
- use selection/routing ops to align q_i and k_j for all query-key pairs
- create explicit affine plus/minus adapters:
    plus  = q + k
    minus = q - k
- apply square as unary PWL
- sum across l explicitly as affine reduction
- apply fixed scale 1/sqrt(d) as affine scaling

Required module:
- PairwiseDotBySquare

This module must be explicit in the transformed graph.
No fallback to torch.matmul is allowed inside transformed attention.

2.2 Attention-value mixing

Original output:
  y_i[c] = sum_j p_ij * v_j[c]

Convert each scalar product p_ij * v_j[c] by the same identity:

  p_ij * v_j[c]
    = ( (p_ij + v_j[c])^2 - (p_ij - v_j[c])^2 ) / 4

Operational form:
- selection/routing ops align each selected probability with the corresponding value component
- explicit affine plus/minus adapters
- square PWL
- affine reduction over j

Required module:
- SparseWeightedSumBySquare

Important:
- after transform there must be no variable matrix multiplication left in the transformed attention
- all bilinear interaction must be explicit through:
    plus/minus affine
    square PWL
    affine reduction

3. Softmax conversion to Gibbs TopK_Softmax with implicit replicated tail

Do not use naive hard top-k softmax.
Do not use dense copied-tail outputs.
Use sparse Gibbs TopK softmax with implicit replicated tail normalization.

For each attention score row s in R^N:

3.1 Support selection
- select support I = TopK(s, k)
- sort retained logits descending:
    s_(1) >= s_(2) >= ... >= s_(k)

3.2 Tail assumption
Assume all omitted logits equal the worst retained logit s_(k).

Do not materialize omitted logits individually.

3.3 Implicit replicated-tail partition function
Define

  Z_tail
    = sum_{j in I} exp(s_j) + (N - k) * exp(s_(k))

3.4 Explicit retained probabilities
For i in I define

  q_i = exp(s_i) / Z_tail

These are the only per-index probabilities to materialize explicitly.

3.5 Tail mass
Define one scalar tail confidence

  q_tail = (N - k) * exp(s_(k)) / Z_tail

This represents the total probability mass of all omitted logits.

Do not materialize probabilities for omitted indices individually.

3.6 Why this is the correct smoothing
- sparsity is preserved because only top-k outputs are stored explicitly
- confidence is not overestimated because omitted logits still contribute to normalization
- the amount of smoothing is automatically controlled by the top-1 to top-k gap:
    Delta_k = s_(1) - s_(k)
- no extra tau/eps smoothing maps are needed

3.7 Jeffreys calibration objective
Use the original full softmax teacher distribution p_full(s).

For each cached score row, compare:
- naive top-k softmax
- GibbsTopKSoftmax with implicit replicated tail

Measure Jeffreys distance:

  J(p, q) = KL(p || q) + KL(q || p)

where q is interpreted as:
- explicit q_i on retained indices
- dense tail completion with omitted logits all equal to s_(k) for evaluation only

Calibration objective:
- verify that GibbsTopKSoftmax materially improves over naive top-k softmax in Jeffreys distance to teacher softmax
- no learned smoothing map is required beyond this construction

3.8 Inference-time block behavior
At inference and in the transformed model, for each row:
- compute I = TopK(s, k)
- compute Z_tail
- compute explicit retained probabilities q_i for i in I
- compute one scalar q_tail
- keep sparse outputs only:
    (indices I, retained confidences q_i, tail mass q_tail)

This is still sparse because:
- outputs beyond top-k are pruned
- only their total confidence contributes to normalization

3.9 Attention execution with sparse outputs
Default execution rule:
- use only the retained probabilities q_i and retained values v_i in the sparse weighted sum
- q_tail is retained as confidence metadata but not expanded into dense outputs

So default attention output is

  y_i[c] = sum_{j in I} q_j * v_j[c]

where q_j already includes the denominator correction from omitted logits.

Required module:
- GibbsTopKSoftmax

Validation requirement:
- on cached attention rows, report average Jeffreys distance between:
  full softmax
  naive top-k softmax
  GibbsTopKSoftmax with implicit replicated tail
- report improvement over naive top-k softmax

4. Explicit affine/reduction combinators

All of the following must be made explicit as affine nodes:
- residual add
- token mean
- channel mean
- average pooling
- sum
- scalar multiply
- fixed scaling such as 1/sqrt(d)
- identity and negation adapters used in square-product decomposition
- gamma/beta affine output of LN
- any hidden affine reduction inside surgery modules

No implicit arithmetic should remain hidden inside black-box library modules.

5. Local multi-tail PWL epilogues

A local multi-tail PWL epilogue is allowed only as follows:
- one affine output accumulator may feed multiple unary PWL tails
- all tails are local to that affine node
- all tail outputs are materialized as separate quantized-style tensors
- no external accumulator fan-out beyond the local epilogue

This is needed for:
- LayerNorm:
  one centered affine output may feed square and log-compatible branches
- attention bilinear decomposition:
  one affine adapter may feed square tails
- softmax partition logic:
  one retained score block may feed exp paths and selection/routing metadata as needed

The script must keep local multi-tail epilogues explicit in code.

Required transform modules to implement

1. ExplicitMean
2. ExplicitAdd
3. ExplicitScale
4. ExplicitNegate
5. ExplicitIdentity
6. PairwiseDotBySquare
7. SparseWeightedSumBySquare
8. RewrittenLayerNormAbsSign
9. GibbsTopKSoftmax
10. MultiTailPWLEpilogue
11. SelectionRoutingTopK
12. SetSign
13. AbsOp
14. TuplePack / TupleUnpack if helpful

Required surgery script behavior

The main script must:

1. Instantiate DeiT-Tiny
2. Instantiate Oxford-IIIT Pet
3. Validate the original model
4. Build caches from the original model:
   - LN inputs/outputs
   - attention score rows
   - full softmax rows
5. Evaluate and log:
   - LN exact rewrite sanity
   - Jeffreys distance of naive top-k softmax
   - Jeffreys distance of GibbsTopKSoftmax with implicit replicated tail
6. Transform the model into the target basis
7. Validate immediately after transform:
   - classification accuracy
   - loss
   - optional logit MSE against original
8. Save transformed model class reference and pre-finetune checkpoint
9. Fine-tune all parameters except eps
10. Save final checkpoint
11. Print final validation report

Fine-tuning rule

Fine-tune all trainable parameters, including:
- affine weights
- affine biases
- PWL parameters
- gamma/beta

Do NOT fine-tune:
- eps

Keep eps fixed throughout.

Artifacts to save

1. transformed model Python code
2. pre-finetune checkpoint
3. post-finetune checkpoint
4. surgery metadata
5. optional calibration cache stats:
   - LN rewrite MSE
   - average Jeffreys for naive top-k softmax
   - average Jeffreys for GibbsTopKSoftmax
   - chosen top-k
   - chosen number of PWL knees
   - chosen eps

Acceptance criteria

The transformed model is acceptable only if:

1. Immediate post-transform validation is reasonably close to the original by construction.
2. LayerNorm rewrite is numerically very close to original LN on sampled activations.
3. GibbsTopKSoftmax materially improves over naive top-k softmax in Jeffreys distance to teacher softmax.
4. The transformed graph contains only:
   - affine nodes
   - local multi-tail PWL epilogues
   - selection/routing ops
5. No torch LayerNorm, dense softmax, or variable torch.matmul remains in transformed attention.
6. Both pre-finetune and post-finetune checkpoints are saved.

Final instruction to agent

Implement the transform completely and explicitly.
Do not leave hidden arithmetic inside black-box modules.
Do not use naive top-k softmax.
Do not use approximate affine-only LN surrogates.
Do not keep variable matrix multiplication in attention.
Do not use the old LN positive/negative split.
Do not introduce extra tau/eps smoothing maps for top-k softmax.

Use:
- abs
- setsign
- exact or near-exact LN rewrite
- GibbsTopKSoftmax with implicit replicated tail normalization
- explicit affine nodes
- local multi-tail PWL epilogues
- selection/routing ops only

The transformed model must be a real, saved PyTorch model in the target basis, ready for later QAT quantization of affine nodes.
