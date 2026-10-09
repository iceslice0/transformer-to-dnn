# Google Coral Edge TPU

Edge accelerator for TFLite models. Toolchain: Edge TPU Compiler (closed).

**Source.** "TensorFlow models on the Edge TPU" (coral.ai/docs/edgetpu/models-intro), accessed 2026-05-05 and
2026-10-09. File in `sources/`: `models-intro.html` (supported-operations table and compiler behavior).

**On the TPU** (relevant operations): Logistic, Tanh, ReLU, Rsqrt (compiler version 14 or later), Mul,
Squared-difference, Softmax (1-D input only, at most 16,000 elements), ReduceMax and ReduceMin, Mean,
FullyConnected (weights constant). Not listed: Exp, Log, Sqrt, Div, Pow, BatchMatmul, Gather, TopK, GELU.

**Strict graph: whole.** Square as Mul(x, x), Logistic, Rsqrt.

**Extended graph: host for the activation MatMul and the row-wise softmax.**

**Compiler limit.** "Currently, the Edge TPU compiler cannot partition the model more than once": the first
unsupported operation and everything after it run on the CPU. The strict graph is unaffected; the extended graph
runs on the CPU from the first attention score on.
