# TI TDA4VM (C7x DSP with Matrix Multiply Accelerator)

Automotive vision SoC (Jacinto 7). Toolchain: TIDL (edgeai-tidl-tools); unsupported operators run on the Arm
Cortex-A cores.

**Source.** edgeai-tidl-tools `docs/operators.md` at commit `7cdef46` (github.com/TexasInstruments/edgeai-tidl-tools),
accessed 2026-10-09. Files in `sources/`: `operators.md`, `topk.md`, `vision_transformers.md`.

**On the accelerator** (ONNX operators, with the constraints that matter here):

| Operator | Constraint |
|---|---|
| Mul / Div (TIDL_EltWiseLayer) | two inputs, same or broadcastable dimensions |
| Gemm / MatMul (TIDL_InnerProductLayer) | MatMul accepts two variable inputs; signed and unsigned inputs cannot be mixed on TDA4VM |
| Sigmoid, Tanh | up to 6 non-singleton dimensions |
| Exp, Log, Sqrt | up to 6 non-singleton dimensions |
| Pow | exponent a constant of size 1 |
| Softmax | along width or height |
| LayerNormalization | across the width axis |
| GELU | fused pattern (Erf only inside GELU) |
| Gather, ReduceMax (height axis), ReduceSum | |
| TopK | sorted, K constant, width or height axis |
| Attention | non-causal; no past key/value |

**Strict graph: whole.** Square as Mul(x, x), Sigmoid, inverse square root as Pow(x, -0.5) (or Sqrt and Div).

**Extended graph: whole.** MatMul with two variable inputs, Mul, Exp, Div, Sqrt or Pow, GELU, native Softmax and
LayerNorm.

**TopK.** On the accelerator: the only surveyed device that runs the top-k selection itself.
