# Arm Ethos-N78

Application-processor NPU IP. Toolchain: Ethos-N driver stack support library (via Arm NN).

**Source.** ethos-n-driver-stack at commit `6834c82` (github.com/ARM-software/ethos-n-driver-stack), accessed
2026-10-03. Files in `sources/`: `SUPPORTED.md` and `SupportQueries.cpp` (`IsMultiplicationSupported` and the
other support queries).

**On the NPU** (relevant operations): ReLU, Leaky ReLU, Sigmoid, Tanh; Multiplication of two tensors with batch 1
and equal shapes (the code notes that stretching, i.e. broadcasting, is no longer supported); multiplication by
a 1 x 1 x 1 x C constant or a scalar through Arm NN rewrites. No exponential, logarithm, inverse square root,
softmax or matrix multiplication between activations.

**Strict graph: host for the inverse square root.** Squares as Multiplication(x, x) and the sigmoid run on the NPU.

**Extended graph: host** for the activation MatMul, softmax and the LayerNorm inverse square root.

**TopK.** Host.
