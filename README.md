# Transformer Surgery

License: [MIT](LICENSE).

A transformer is just a DNN. This repo demonstrates that operationally: it compiles a trained transformer into an explicit graph built from a small, fixed vocabulary of standard neural-network ops. Primary patients are timm DeiT-Tiny (Oxford-IIIT Pet / ImageNet), MambaIRv2 Light SR, and a surgery-only causal LM (Pythia-70M on WikiText-2). Vision students retain reference accuracy after a brief distillation pass and tolerate 8-bit PTQ like any conventional DNN; the LLM patient shows the same compile applies at full support without noticeable degradation (float32).

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

`pretrain_pet` is the only model-specific stage. `surgery`, `distill`, and `ptq` go through a model adapter (`model_key`, default `deit_tiny_pet`); adding a new transformer means registering an adapter, not branching the stage code. Adapters that ship in the repo: `deit_tiny_pet` (DeiT-Tiny on Oxford-IIIT Pet), `imagenet` ([src/transformer_surgery/models/imagenet.py](src/transformer_surgery/models/imagenet.py)), `mambair_lightsr` (MambaIRv2 Light SR; see Install below), and `pythia_70m_wikitext2` (Pythia-70M on WikiText-2, surgery-only).

A top-level `Makefile` orchestrates the full experiment grid (top-k sweeps, tail-mass ablations, strict vs fast, PTQ variants, ImageNet, MambaIR, Pythia). Use `make help` to list targets, or `make a` / `make a_imagenet` / `make mambair` / `make pythia` / `make all` to run a sweep; targets call into the same CLIs and write to `artifacts/`.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The Pet dataset is expected under `data/oxford-iiit-pet/`. If missing, the torchvision loader downloads it once into `data/`.

### MambaIR (optional)

The `mambair_lightsr` adapter compiles MambaIRv2 Light SR (window attention + LayerNorm surgery on DIV2K/Set5). The network definition is the **csguoh/MambaIR** tree already under `mambal/softmax/MambaIR` (imported as `basicsr` — not the PyPI package). `mamba-ssm` is **not** installed from PyPI; use a GitHub release wheel that matches your Python / torch / CUDA / CXX11 ABI.

```bash
# SR Python deps (no mamba-ssm from PyPI)
pip install -e '.[mambair]'

# Fused selective-scan CUDA kernel from state-spaces/mamba releases (required for usable speed).
# List wheels: https://github.com/state-spaces/mamba/releases/tag/v2.3.1
# Example that works with torch 2.11+cu130, Python 3.12, cxx11 ABI True on linux x86_64:
pip install --no-deps \
  'https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'

# Verify:
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; import selective_scan_cuda; print('ok')"

# Released Light-SR weights used by the default configs (already present if you keep mambal/):
#   mambal/softmax/mambairv2_lightSR_x2.pth
#   mambal/softmax/mambairv2_lightSR_x4.pth
```

If no prebuilt wheel matches, you need a CUDA *toolkit* equal to `torch.version.cuda` and:

```bash
MAMBA_FORCE_BUILD=TRUE MAMBA_KEEP_CUDA_BUILD=TRUE \
  pip install --no-cache-dir --no-deps --no-build-isolation \
  'git+https://github.com/state-spaces/mamba.git@v2.3.1'
```

Then `make mambair-data` (DIV2K + Set5 x2/x4 LR) and `make mambair` / `make mambair-x4` (surgery + distill). Configs: `configs/surgery/mambair_x{2,4}_*.json`, `configs/distill/mambair_x{2,4}_*.json`. Reference pipeline: [mambal/softmax/README.md](mambal/softmax/README.md).

### Pythia-70M (optional, surgery-only)

Causal-LM patient using Hugging Face `EleutherAI/pythia-70m` and WikiText-2 (`wikitext-2-raw-v1`) at a fixed non-overlapping `context_length` (default 128). Surgery-only: no distill or PTQ for this adapter. Use `surgery_dtype=float32`; a full bfloat16 weight cast drops accuracy sharply on this patient.

```bash
pip install -e '.[llm]'

# Full-context baseline (top_k == context_length):
python -m transformer_surgery.cli.surgery --config configs/surgery/pythia_70m_topk128_fast_tailmass_exact.json

# Sparse causal with exact per-query tail mass:
python -m transformer_surgery.cli.surgery --config configs/surgery/pythia_70m_topk32_fast_tailmass_exact.json

# Or via Make: make pythia-smoke | make pythia | make pythia-tailmass0
```

Evaluation reports token NLL and perplexity (primary metric is `-NLL`). Scalar per-block `gibbs_tail_prob_eps` is disabled: causal rows have different valid key counts. Exact-tail configs set `use_exact_tail_mass=true`; `*_tailmass0` configs drop omitted mass (ablation). Exact omitted-tail stats (`gibbs_tail_prob_eps_exact_*`) are written into surgery metadata by the shared student calibration path. Sweep table: [docs/pythia_results.md](docs/pythia_results.md).

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
- [docs/exact_tail_mass.md](docs/exact_tail_mass.md) - exact runtime omitted-tail mass.
- [docs/pythia_results.md](docs/pythia_results.md) - Pythia-70M WikiText-2 surgery sweep (exact vs tail0).
- [docs/distill.md](docs/distill.md) - CE + Jeffreys distillation loop and metric definitions.
- [docs/ptq.md](docs/ptq.md) - single-pass calibration-driven post-training quantization.
