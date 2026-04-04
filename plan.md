
What this project is trying to do

Take a pretrained DeiT-Tiny model and rewrite it into a graph that uses
only three kinds of things:

1. affine nodes
2. local multi-tail PWL epilogues
3. selection/routing ops

The rewritten model should stay close to the original model immediately
after surgery, should be saved as a real PyTorch model and checkpoint,
and should later be ready for QAT quantization of affine nodes.

What the output of the work should be

You should produce:

- a transformed DeiT-Tiny model definition in Python
- a script that performs the surgery
- a checkpoint after surgery, before fine-tuning
- a checkpoint after short fine-tuning
- a metadata file describing the surgery choices and validation results

Use exactly one patient and one dataset

Patient:
- DeiT-Tiny from timm, pretrained

Dataset:
- Oxford-IIIT Pet classification

Do not add extra models or datasets at this stage.

Allowed graph basis after surgery

After the rewrite, the model may contain only:

A. Affine nodes
These are all explicit x -> A x + b type blocks.

This includes:
- Linear layers
- Conv / 1x1 conv if needed
- residual adds
- sums
- means
- average pooling
- scalar multiply
- negation
- identity
- fixed scaling such as 1/sqrt(d)
- gamma/beta affine output of LayerNorm
- any explicit add/subtract adapters introduced by the rewrite

B. Local multi-tail PWL epilogues
These are local unary nonlinear blocks attached to one affine output.

Rules:
- one affine node may produce one widened accumulator
- that accumulator may feed several local unary PWL tails
- those tails are local to the same affine output
- no global fan-out of accumulator values through memory is allowed

C. Selection/routing ops
These are non-arithmetic routing/control operations.

Allowed selection/routing ops:
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
- max / min
- argmax / top-k indices
- routing scores and associated values together
- abs
- setsign(x, y) = sign(x) * y

Important convention:
- max is a selection/routing op
- abs is a selection/routing op
- setsign is a selection/routing op
- arithmetic must not be hidden inside selection/routing ops

What must be removed from the original model

After transformation, there should be no implicit:
- LayerNorm
- dense softmax
- torch.matmul for attention score computation
- torch.matmul for attention-value mixing
- hidden affine reductions buried inside black-box calls

Everything important must be explicit in the graph.

Main transform 1: LayerNorm rewrite

Each LayerNorm should be replaced by this explicit computation:

Given x:

- mu = mean(x, dim=-1, keepdim=True)
- u = x - mu
- r2 = mean(u * u, dim=-1, keepdim=True)

Then:

- a = exp( log(abs(u) + eps) - 0.5 * log(r2 + eps) )
- z = setsign(u, a)
- y = beta + gamma * z

Interpretation:
- abs(u) gives the magnitude input
- log/exp produce the normalized magnitude
- setsign restores the sign of u
- this is equivalent to u / sqrt(r2 + eps), up to eps convention

Why this version is preferred:
- more efficient than the earlier positive/negative split
- only one exp branch
- fewer branches
- cleaner pseudo-hardware form
- easier later QAT integration

Requirements:
- eps must be explicit and fixed
- gamma and beta copied from original LayerNorm
- mean and r2 explicit as affine/reduction nodes
- abs explicit
- log and exp explicit as unary PWL tails
- setsign explicit
- no torch.nn.LayerNorm left in transformed model

Validation:
- compare rewritten LN output with original LN output on held-out minibatches
- log per-layer LN MSE

Main transform 2: Replace variable matrix multiplication in attention

There should be no matmul or einsum left in transformed attention for:
- score computation Q K^T
- value mixing P V

Use the exact identity:

a * b = ((a + b)^2 - (a - b)^2) / 4

2A. Attention score computation

Original:
- s_ij = q_i^T k_j / sqrt(d)

Replace each product q_i[l] * k_j[l] using the square identity.

Operationally:
- align q_i and k_j with routing ops
- build plus = q + k
- build minus = q - k
- apply square PWL
- subtract and scale by 1/4
- sum over feature dimension explicitly
- scale by 1/sqrt(d)

Create a module:
- PairwiseDotBySquare

2B. Attention-value mixing

Original:
- y_i[c] = sum_j p_ij * v_j[c]

Replace each product p_ij * v_j[c] using the same square identity.

Operationally:
- align each retained probability with the matching value component
- build plus and minus
- square
- subtract and scale
- affine reduce over selected keys

Create a module:
- SparseWeightedSumBySquare

Main transform 3: Replace softmax by Gibbs TopK Softmax

Do not use naive hard top-k softmax.
Do not use unnecessary learned smoothing maps.

Use this simpler geometric construction.

For each score row s:

1. Select support
- I = TopK(s, k)
- retained logits sorted:
  s_(1) >= s_(2) >= ... >= s_(k)

2. Tail assumption
Assume all omitted logits equal the worst retained logit s_(k).

Do not materialize omitted logits individually.

3. Partition function
- Z_tail = sum_{j in I} exp(s_j) + (N - k) * exp(s_(k))

4. Retained probabilities
For i in I:
- q_i = exp(s_i) / Z_tail

5. Tail mass
- q_tail = (N - k) * exp(s_(k)) / Z_tail

Interpretation:
- only top-k probabilities are stored explicitly
- omitted logits still contribute through the denominator
- this preserves sparsity but prevents overconfident normalization

Attention execution rule:
- use only retained probabilities q_i and retained values v_i for the sparse weighted sum
- keep q_tail only as metadata / denominator correction
- do not materialize dense tail outputs

Create a module:
- GibbsTopKSoftmax

Validation:
- compare full softmax vs naive top-k softmax vs GibbsTopKSoftmax
- use Jeffreys distance:
  J(p, q) = KL(p || q) + KL(q || p)
- verify GibbsTopKSoftmax improves over naive top-k softmax

Explicit affine combinators that must be exposed

Make these explicit:
- residual add
- token mean
- channel mean
- average pooling
- sum
- scalar multiply
- identity
- negation
- fixed scaling
- gamma/beta affine output
- plus/minus adapters used in square-product decomposition

Nothing affine should remain hidden in black-box library calls.

Use of local multi-tail PWL epilogues

Use local multi-tail PWL epilogues only where needed.

Examples:
- LayerNorm: one affine output may feed square/log-related unary tails
- attention bilinear decomposition: plus/minus adapters may feed square tails
- softmax: retained score path may feed local exp-related tails

No external accumulator fan-out is allowed.

Modules that should exist after implementation

Please implement at least these:

- ExplicitMean
- ExplicitAdd
- ExplicitScale
- ExplicitNegate
- ExplicitIdentity
- PairwiseDotBySquare
- SparseWeightedSumBySquare
- RewrittenLayerNormAbsSign
- GibbsTopKSoftmax
- MultiTailPWLEpilogue
- SelectionRoutingTopK
- SetSign
- AbsOp
- TuplePack / TupleUnpack if useful

What the main script should do

The main script should:

1. Instantiate DeiT-Tiny
2. Instantiate Oxford-IIIT Pet dataset and dataloaders
3. Validate the original model
4. Build caches from the original model:
   - LayerNorm inputs/outputs
   - attention score rows
   - full softmax rows
5. Validate the transforms before application:
   - LayerNorm rewrite sanity
   - Jeffreys distance for naive top-k softmax
   - Jeffreys distance for GibbsTopKSoftmax
6. Transform the model into the target basis
7. Validate immediately after transform:
   - classification accuracy
   - loss
   - optional logit MSE vs original model
8. Save transformed model and pre-finetune checkpoint
9. Fine-tune all trainable parameters except eps
10. Validate again
11. Save final checkpoint and metadata

Fine-tuning rule

Trainable:
- affine weights
- affine biases
- PWL parameters
- gamma/beta

Not trainable:
- eps

Keep eps fixed throughout.

Artifacts to save

Save:
- transformed model Python code
- pre-finetune checkpoint
- post-finetune checkpoint
- surgery metadata
- optional calibration stats:
  - per-layer LN MSE
  - Jeffreys distance for naive top-k softmax
  - Jeffreys distance for GibbsTopKSoftmax
  - top-k used
  - PWL knees used
  - eps used

Acceptance criteria

The surgery is acceptable only if:

1. Immediate post-transform validation is reasonably close to the original.
2. LayerNorm rewrite is numerically very close to the original LN.
3. GibbsTopKSoftmax is better than naive top-k softmax in Jeffreys distance.
4. The transformed graph contains only:
   - affine nodes
   - local multi-tail PWL epilogues
   - selection/routing ops
5. No torch LayerNorm remains.
6. No dense softmax remains.
7. No variable torch.matmul remains in transformed attention.
8. Both pre-finetune and post-finetune checkpoints are saved.

Final instruction

Implement the transform completely and explicitly.

Do not:
- leave hidden arithmetic in black-box modules
- use naive top-k softmax
- use affine-only LN surrogates
- keep variable matrix multiplication in attention
- use the old positive/negative LN split
- add unnecessary smoothing-map complexity

Do:
- use abs
- use setsign
- use the exact or near-exact LN rewrite
- use GibbsTopKSoftmax with implicit replicated tail normalization
- make all affine nodes explicit
- keep local multi-tail PWL epilogues explicit
- isolate selection/routing ops cleanly

The final transformed model must be a real saved PyTorch model in the target basis and ready for later QAT quantization of affine nodes.

