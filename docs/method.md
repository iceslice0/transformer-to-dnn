
What this project is trying to do

Take a pretrained DeiT-Tiny model and rewrite it into a graph that uses
only three kinds of things:

1. affine nodes (including fixed-coefficient ``einsum`` contracts and standard ``Linear`` / ``Conv``)
2. local unary PWL epilogues (and exact unary ops like ``log``, ``exp``, ``sqrt`` where used)
3. selection/routing ops

The rewritten model should stay close to the original model immediately
after surgery, should be saved as a real PyTorch model and checkpoint,
and should later be ready for quantization of affine nodes (PTQ tooling
exists separately; QAT may follow).

What the output of the work should be

You should produce:

- a transformed DeiT-Tiny model definition in Python (``src/transformer_surgery/model.py``)
- scripts that perform surgery, optional distillation fine-tuning, and optional PTQ
- a checkpoint after surgery, before fine-tuning
- a checkpoint after short fine-tuning (optional pipeline step)
- metadata describing the surgery choices and validation results

Use exactly one patient and one dataset

Patient:
- DeiT-Tiny from timm, pretrained

Dataset:
- Oxford-IIIT Pet classification

Do not add extra models or datasets at this stage.

Allowed graph basis after surgery

After the rewrite, the strict graph uses explicit modules in ``src/transformer_surgery/ops.py`` and
``src/transformer_surgery/model.py``. Conceptually:

A. Affine and fixed-linear nodes

- **Standard layers:** ``nn.Linear``, ``nn.Conv2d`` (patch embed).
- **Per-channel scale+bias:** ``AffineScaleBias`` (LayerNorm ``gamma``/``beta`` in the rewrite).
- **Fixed coefficient contraction:** ``AffineContract(einsum_equation, coeff)`` — one ``nn.Module``
  per logical affine; ``torch.einsum`` over a registered ``coeff`` buffer.
- **Fixed 2×2 mixes on an operand axis:** ``AffineFixedMix`` (used inside the square-identity chain
  to form plus/minus combinations before ``UnarySquare``).
- **Unary reductions / scalings:** ``UnaryMean``, ``UnarySum``, ``UnaryScale``, etc., implemented
  as small ``nn.Module``s so they appear in ``named_modules``.

**Binary mixes (residual, pos-embed, centering, Gibbs intermediates):** there is *no* separate
“add module”. Callers use **routing only** — ``torch.broadcast_tensors`` then
``torch.stack(..., dim=-1)`` — and pass the result into ``AffineContract("i,...i->...", [w0, w1])``.
The graph shows an ``AffineContract`` child; the stack is plain tensor wiring in ``forward``.

B. Local unary nonlinearities

- **Scalar PWL:** ``UnaryScalarPWL``, ``GELUUnaryPWL`` (trainable knot values where applicable).
- **Exact unaries used in strict paths:** ``UnaryLogPlusEps``, ``UnaryExp``, ``UnarySqrtExp``,
  ``UnarySquare``, ``UnaryRsqrtPlusEps``, ``UnaryReciprocalPlusEps``, etc., as documented on each class.

Rules (conceptual):

- one affine or fixed-linear node may feed one or more unary tails locally
- no unspecified global fan-out of “accumulator buses” beyond what the explicit module tree shows

C. Selection/routing ops

Non-arithmetic wiring and discrete choices: ``reshape``/view, transpose, expand, concat,
``torch.topk``, ``F.relu``, ``torch.max`` for row max (Gibbs stabilization), gather-style indexing
for sparse attention, ``Dropout``/``DropPath``, etc.

**Convention:** anything that is pure tensor routing or discrete selection stays *outside* the
``AffineContract`` / ``Linear`` nodes; arithmetic uses the explicit modules above.

**Debug / fast path:** run configs may set ``allow_matmul=True`` to use ``MatMul`` (``@``) or
``MatMulHadamard`` (elementwise ``*``) where implemented (e.g. QK dot, attn×V, LN inverse-std path).
That path is **not** the strict demonstration graph; strict mode keeps the explicit decompositions.

What must be removed from the original model

After transformation (in **strict** surgery mode), there should be no implicit:

- ``torch.nn.LayerNorm`` in blocks that use ``RewrittenLayerNorm``
- dense softmax over full key length when ``use_surgery_softmax`` is enabled
- variable ``torch.matmul`` for attention **score** computation or for **sparse value mixing**
  when ``allow_matmul=False``

Everything important must be explicit in the module tree. Optional flags may re-enable library
``LayerNorm``, dense softmax, or matmul for debugging (see run configs).

Main transform 1: LayerNorm rewrite — ``RewrittenLayerNorm``

Implemented in ``transformer_surgery.ops.RewrittenLayerNorm``.

Given ``x``:

- ``mu = mean(x, dim=-1, keepdim=True)`` via ``UnaryMean``
- ``u = x - mu`` via **routing** + ``AffineContract("i,...i->...", [1, -1])`` (submodule ``u_center_contract``)
- ``r2 = mean(u*u, dim=-1, keepdim=True)`` via ``UnarySquare`` and ``UnaryMean``

**Strict path (``allow_matmul=False``):** magnitude in log domain using a **relu± split**, not a
single ``abs`` tensor op:

- ``au = stack(relu(u), relu(-u), dim=-2)``; ``log_num = log(au + eps)`` via ``UnaryLogPlusEps``
- ``log_den = log(r2 + eps)`` broadcast; combine with ``AffineContract`` on stacked logs
  (coeffs ``[2, -1]``) then ``UnarySqrtExp`` for the magnitude factor
- final channel mix with another ``AffineContract`` and ``AffineScaleBias`` for ``gamma``/``beta``

**Fast path (``allow_matmul=True``):** ``inv_std = rsqrt(r2 + eps)``, then ``u * inv_std`` via
``UnaryRsqrtPlusEps`` and ``MatMulHadamard`` (not the log-domain chain).

Requirements:

- ``eps`` is explicit (buffers / constructor args); not trained in fine-tuning (see ``freeze_eps_parameters``)
- ``gamma``/``beta`` copied from timm LayerNorm where applicable (``copy_ln_params_to_rewritten``)

Validation (see ``python -m transformer_surgery.cli.run_surgery`` / ``transformer_surgery.pet``): compare rewritten LN to
reference LN on minibatches; metrics go into surgery metadata.

Main transform 2: Replace variable matrix multiplication in attention (strict mode)

There should be no variable ``matmul`` for:

- score computation ``Q K^T`` (when ``allow_matmul=False``)
- sparse value mixing ``P V`` (when ``allow_matmul=False``)

Use the identity ``a * b = ((a+b)^2 - (a-b)^2) / 4``.

**Implementation:** ``SquareIdentityOperandChain`` — operands ``stack((a,b), dim=-2)`` → ``AffineFixedMix``
→ ``UnarySquare`` → ``AffineContract`` with coeffs ``±1/(4√d)`` (QK) or ``±1/4`` (sparse mix), with
equations chosen per use case.

Modules:

- ``PairwiseDotBySquare`` — QK grid → scores (or ``MatMul`` when ``allow_matmul=True``)
- ``SparseWeightedSumBySquare`` — sparse probs + gathered values → output (or Hadamard path when allowed)

Main transform 3: Replace softmax by Gibbs Top-K — ``GibbsTopKSoftmax``

Same geometric construction as before (top-k support, tail mass at ``s_(k)``, partition function
``Z_tail``, sparse probs, tail mass ``q_tail``). **Implementation detail:** score stabilization
``scores - row_max``, tail sum ``sum_exp + (N-k)*exp(s_K)``, and logit normalization use
**``broadcast_tensors`` + ``stack`` + ``AffineContract``** submodules (e.g. ``scores_stable_contract``,
``z_tail_contract``, ``logit_logz_contract``) so the graph lists ``AffineContract`` nodes rather than
opaque binary ops.

Normalization avoids raw ``/`` in the strict path (``exp(vals - log(z_tail+eps))`` style); with
``allow_matmul=True``, reciprocal + ``MatMulHadamard`` may be used.

Module: ``GibbsTopKSoftmax`` (selection uses ``torch.topk`` in ``forward``). A separate
``SelectionRoutingTopK`` helper exists in ``transformer_surgery.ops`` but is not required by the current Gibbs path.

Validation: Jeffreys divergence metrics vs dense / naive top-k (see ``transformer_surgery.ops`` / run scripts).

Concrete building blocks (current code)

These are the main exported concepts; names match ``transformer_surgery.ops`` / ``transformer_surgery.model``:

- **Fixed affine:** ``AffineContract``, ``AffineFixedMix``, ``AffineScaleBias``
- **Unary:** ``UnaryMean``, ``UnarySum``, ``UnaryScale``, ``UnarySquare``, ``UnaryExp``, ``UnaryLogPlusEps``,
  ``UnarySqrtExp``, ``UnaryRsqrtPlusEps``, ``UnaryReciprocalPlusEps``, …
- **Chains:** ``SquareIdentityOperandChain``, ``PairwiseDotBySquare``, ``SparseWeightedSumBySquare``
- **LayerNorm replacement:** ``RewrittenLayerNorm``
- **Attention softmax:** ``GibbsTopKSoftmax``
- **GELU:** ``GELUUnaryPWL``
- **Optional matmul wrappers:** ``MatMul``, ``MatMulHadamard``
- **Model:** ``DeiTTinySurgeryModel`` with ``residual_contract`` / ``pos_embed_contract`` as
  ``AffineContract("i,...i->...", (1,1))`` plus stack routing in ``forward``

There are **no** separate classes named ``ExplicitAdd``, ``ExplicitMean``, ``SetSign``, ``AbsOp``,
``MultiTailPWLEpilogue`` — those ideas are expressed with the modules above.

What the main scripts do (reference)

- ``python -m transformer_surgery.cli.run_surgery``: load Pet timm checkpoint, build surgery student, eval, write
  ``surgery_pre_ft.pt``, ``surgery_meta.json``, and ``artifacts/logs/model_{before,after}_surgery.txt``
  (includes forward **output shape** traces via ``write_model_structure_txt``).
- ``python -m transformer_surgery.cli.distill``: Jeffreys distillation / fine-tuning from config.
- ``python -m transformer_surgery.cli.ptq``: optional post-training quantization; writes ``artifacts/logs/model_after_ptq.txt``.

Fine-tuning rule

Trainable:
- affine weights and biases where ``requires_grad`` is set
- PWL parameters (knot values)
- ``gamma``/``beta`` in ``AffineScaleBias``

Not trainable:
- fixed ``eps`` buffers/parameters frozen by ``freeze_eps_parameters`` (and similar fixed scalars
  per config)

Artifacts to save

Save:
- transformed model Python code
- pre-finetune checkpoint
- post-finetune checkpoint (when run)
- surgery / PTQ metadata JSON
- optional calibration stats (LN MSE, Jeffreys metrics, top-k, GELU PWL knot positions in ``pwl``, eps)
- text dumps under ``artifacts/logs/`` with model structure and per-layer forward output shapes

Acceptance criteria

The surgery is acceptable only if:

1. Immediate post-transform validation is reasonably close to the original (under chosen flags).
2. LayerNorm rewrite matches the reference LN closely on held-out data (see MSE metrics).
3. Gibbs Top-K improves over naive top-k in Jeffreys distance where measured.
4. The transformed **strict** graph exposes explicit affine and unary modules as above; binary
   mixes appear as ``AffineContract`` after ``stack``, not hidden inside custom “add” modules.
5. No ``torch.nn.LayerNorm`` remains in blocks that are configured for surgery LN.
6. No dense softmax over full keys when surgery softmax is enabled.
7. No variable ``torch.matmul`` in attention score or sparse value paths when ``allow_matmul=False``.
8. Checkpoints and metadata are produced for the chosen pipeline steps.

Final instruction

Implement the transform completely and explicitly in the **current** abstraction: routing
(``broadcast``, ``stack``, indexing) is separate; fixed linear maps are ``AffineContract`` /
``AffineFixedMix`` / ``Linear`` / ``Conv``; unary nonlinearities are explicit submodule classes.

Do not:
- hide affine structure inside opaque binary “add/sub” modules (use ``AffineContract`` + routing)
- use naive hard top-k softmax when Gibbs replacement is enabled
- rely on undocumented black-box shortcuts in strict mode

Do:
- use ``RewrittenLayerNorm`` strict or fast path as configured
- use Gibbs Top-K with implicit replicated tail normalization
- keep ``allow_matmul`` as an explicit escape hatch for speed/debug, distinct from strict demos
- save real PyTorch checkpoints and human-readable ``artifacts/logs/*.txt`` structure dumps

The final transformed model must be a real saved PyTorch model in this basis and suitable for
later quantization of affine nodes (see PTQ script and ``ptq_plan.md``).
