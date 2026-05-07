# Transformer Surgery

A transformer is just a DNN. This repo demonstrates that operationally: it compiles a trained transformer (timm DeiT-Tiny on Oxford-IIIT Pet) into an explicit graph built from a small, fixed vocabulary of standard neural-network ops, and shows the compiled model retains the reference accuracy after a brief distillation pass and tolerates 8-bit PTQ like any conventional DNN.

There is no architectural magic in attention, LayerNorm, or softmax. Each is a particular composition of three op classes:

- **affine** - `nn.Linear`, `nn.Conv2d`, fixed-coefficient contractions, scale/bias, mean/sum reductions.
- **nonlinear unary** - pointwise scalar maps: `exp`, `log(x+eps)`, `sqrt(exp(x))`, `1/(x+eps)`, `1/sqrt(x+eps)`, `square`, GELU.
- **routing** - pure tensor wiring and discrete selection: `reshape`, `transpose`, `cat`, `stack`, `expand`, `broadcast_tensors`, `topk`, `gather`, `relu`.

A transformer block is then a wiring diagram over this vocabulary. The repo produces two compilations:

- **strict** (`allow_matmul=False`): no variable matrix multiplication anywhere. QK^T scores, sparse value mixing, and the LayerNorm scale-and-shift all expand into the three classes above using the algebraic identity `a*b = ((a+b)^2 - (a-b)^2)/4` and an exact log-domain rewrite of LayerNorm. The compiled model contains only fixed-coefficient affine ops plus the listed nonlinearities and routing. This mode is inherently inefficient: replacing each variable product with complex computation path creates more intermediate tensors, memory traffic, and elementwise work than native matmul/Hadamard kernels. It is the proof-oriented compile target, not the practical runtime path.
- **fast** (`allow_matmul=True`): the same vocabulary plus variable `matmul` / Hadamard. This mode exists because the strict expansion is expensive; it keeps the surgery structure and Gibbs top-k behavior while using efficient tensor kernels for variable bilinear work. It is the practical baseline for calibration, distillation, and PTQ runs.

Dense softmax is replaced everywhere with a **top-k Gibbs softmax** - sparse normalization over the top-k scores plus a calibrated tail-mass parameter to reduce memory access.

PTQ is part of the evidence for the compile, not just an extra benchmark. The failure hypothesis is
simple: if the surgery graph only works because its arithmetic paths depend on dynamic
high-precision floating-point behavior, then replacing selected affine and bilinear work with
fixed-scale 8-bit wrappers should break accuracy. If accuracy is retained, those paths are behaving
like ordinary DNN computations whose floating-point scale can be replaced by calibrated constants,
with discontinuous behavior kept explicit as routing. That DNN-centric view matters for hardware:
the goal is a homogeneous dedicated NPU implementation built from a small repeated primitive set,
not a heterogeneous CPU/NPU or GPU stack where special floating-point kernels carry the model
semantics.

The takeaway: transformers are not a separate species of model. They are standard DNNs with a specific choice of nonlinearities and routing ops. The compiled graph is faithful enough that short distillation recovers reference accuracy, and PTQ confirms the structure quantizes without surprises.

## Stage Flow

Four stages, each taking a JSON config and writing traceable artifacts under `artifacts/`:

```bash
python -m transformer_surgery.cli.pretrain_pet   # 1. train the DeiT-Tiny reference on Pet
python -m transformer_surgery.cli.surgery        # 2. compile reference -> surgery student (no training)
python -m transformer_surgery.cli.distill        # 3. CE + Jeffreys distillation against the reference
python -m transformer_surgery.cli.ptq            # 4. 8-bit PTQ on selected affine / matmul nodes
```

Equivalent console scripts after `pip install -e .`: `ts-pretrain-pet`, `ts-surgery`, `ts-distill`, `ts-ptq`.

`pretrain_pet` is the only model-specific stage. `surgery`, `distill`, and `ptq` go through a model adapter (`model_key`, default `deit_tiny_pet`); adding a new transformer means registering an adapter, not branching the stage code. Two adapters ship in the repo: `deit_tiny_pet` (DeiT-Tiny on Oxford-IIIT Pet) and `imagenet` ([src/transformer_surgery/models/imagenet.py](src/transformer_surgery/models/imagenet.py)).

A top-level `Makefile` orchestrates the full experiment grid (top-k sweeps, tail-mass ablations, strict vs fast, PTQ variants, ImageNet). Use `make help` to list targets, or `make a` / `make a_imagenet` / `make all` to run a sweep; targets call into the same CLIs and write to `artifacts/`.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The dataset is expected under `data/oxford-iiit-pet/`. If missing, the torchvision loader downloads it once into `data/`.

## Configs

Default configs under `configs/`:

- `configs/pretrain/pet_deit_tiny.json` - Pet classifier-head training for the timm reference.
- `configs/surgery/topk64_strict.json` - **strict** compile (DNN ops only), top-k 64.
- `configs/surgery/topk64_fast.json` - **fast** compile (DNN ops + matmul), top-k 64.
- `configs/distill/64_fast_jeffreys.json` - CE + Jeffreys distillation.
- `configs/ptq/*.json` - PTQ variants (full, linear-only, no-matmul, per-tensor).

Pass `--config path/to/config.json`; CLI flags override JSON fields where supported. For adapter-based stages, `reference_checkpoint` is the teacher/reference path. Generated artifacts use `<tool>_<config>[_stage]` stems (e.g. `ts_surgery_topk64_fast.pt`) so every checkpoint is traceable to the CLI and config that produced it. Metadata JSON sits under `artifacts/metadata/` keyed by checkpoint basename.

## Artifacts

Generated outputs are git-ignored:

- `artifacts/checkpoints/` - checkpoints
- `artifacts/metadata/` - metadata JSON
- `artifacts/logs/` - model structure dumps for surgery and PTQ

Default end-to-end artifacts: `ts_pretrain_pet_deit_tiny.pt`, `ts_surgery_topk64_fast.pt`, `ts_distill_64_fast_jeffreys.pt`, `ts_ptq_64_fast_jeffreys_8bit.pt`, plus matching metadata where the stage writes it and surgery/PTQ logs.

## Method Notes

- [docs/surgery.md](docs/surgery.md) - the strict / fast op vocabulary, LayerNorm rewrite, attention rewrite, Gibbs top-k softmax, calibration, and metadata schema.
- [docs/distill.md](docs/distill.md) - CE + Jeffreys distillation loop and metric definitions.
- [docs/ptq.md](docs/ptq.md) - single-pass calibration-driven post-training quantization.
