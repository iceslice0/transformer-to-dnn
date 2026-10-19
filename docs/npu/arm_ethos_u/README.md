# Arm Ethos-U55/U65 and Ethos-U85

Microcontroller-class NPUs. Toolchain: the Vela compiler (TFLite input), regor backend.

**Source.** Vela at commit `dd3f874` (gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela), accessed
2026-10-03. Files in `sources/`: `SUPPORTED_OPS.md` (per-operator NPU placement and constraints),
`tflite_supported_operators_u55.cpp`, `tflite_supported_operators_u85.cpp`,
`ethos_u55_register_cs_generator.cpp` (U55 MatMul lowering), `ethos_u85.cpp` (U85 MatMul on the MAC array).

**On the NPU** (relevant TFLite operators; anything not listed or violating constraints runs on the CPU):

| Operator | U55/U65 | U85 |
|---|---|---|
| MUL (two variable inputs, broadcast) | yes | yes |
| LOGISTIC (sigmoid), TANH | yes | yes |
| RSQRT, EXP, LOG, GELU | yes | yes |
| SOFTMAX | yes | yes |
| BATCH_MATMUL (two variable inputs) | yes, lowered | yes, MAC array |
| REDUCE_MAX, SUM, MEAN | yes | yes |
| DIV | no | yes |
| GATHER | no | yes |
| TOPK_V2 | no | no |

U55 lowering (`InsertMatMulCommand`): the matrix product of two activations runs one output column at a time as
an elementwise MUL into an int32 scratch tensor followed by a reduction. U85 maps MatMul onto its convolution
(MAC) engine.

**Strict graph: whole.** Square as MUL(x, x), sigmoid, inverse square root; GELU native.

**Extended graph: whole.** BATCH_MATMUL, MUL, EXP, RSQRT, GELU and native SOFTMAX. On U55 the activation
MatMul takes the Hadamard route (elementwise product and sum), which writes the products to scratch memory.

**TopK.** Host on both. U85 runs GATHER, so a support chosen on the host can be gathered on the NPU.
