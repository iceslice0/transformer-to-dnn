# VeriSilicon VIP NPUs

NPU IP licensed into many application SoCs. Toolchain: TensorFlow Lite VX delegate (TIM-VX).

**Source.** tflite-vx-delegate at commit `b76047d` (github.com/VeriSilicon/tflite-vx-delegate), accessed
2026-10-03. Files in `sources/`: `op_status.md` (per-operator support) and `op_map.cc` (operator mapping).

**On the NPU** (relevant operators, `op_status.md` "yes" and mapped in `op_map.cc`): Mul, Logistic, Tanh, Exp,
Log, Sqrt, Rsqrt, Pow, Div, Softmax, Gelu (mapped), BatchMatmul, Gather, GatherNd.
**Not supported:** TopkV2, LogSoftmax.

**Strict graph: whole.** Square as Mul(x, x), Logistic, Rsqrt.

**Extended graph: whole.** BatchMatmul, Mul, Exp, Div, Rsqrt, Gelu, native Softmax.

**TopK.** Host (TopkV2 not supported); Gather runs on the NPU.
