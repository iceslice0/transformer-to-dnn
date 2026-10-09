# NVIDIA Jetson Xavier / Orin DLA

Deep Learning Accelerator in Jetson Xavier and Jetson Orin modules (based on the NVDLA architecture).
Toolchain: TensorRT (closed); unsupported layers run on the GPU.

**Source.** TensorRT documentation "DLA Supported Layers and Restrictions", accessed 2026-10-03 and 2026-10-09.
File in `sources/`: `dla-layer-restrictions.html`.

**On the DLA** (relevant layers):

- Activation: ReLU, Sigmoid, TanH, Clipped ReLU, Leaky ReLU (FP16, INT8).
- Unary: Abs, Sin, Cos, Atan (Sin, Cos, Atan with INT8 input).
- ElementWise: Sum, Sub, Product, Max, Min, Div, Pow, Equal, Greater, Less. Pow: one input an FP32 constant with
  one value, the other an INT8 tensor. Div: a constant operand with one value and an INT8 tensor.
- Softmax: Orin only.
- Reduce: MAX only.
- MatrixMultiply: not listed.

**Strict graph: whole, in INT8.** Square as Product(x, x), Sigmoid, inverse square root as Pow(x, -0.5) on an INT8
input, reciprocal as Div(1, x) on an INT8 input.

**Extended graph: host for the activation MatMul.** Softmax runs on the DLA on Orin; the inverse square root as an
INT8 power; GELU only through ReLU or `x * sigmoid(1.702 x)`.

**Note.** The INT8-only power, division, sine, cosine and arctangent match a lookup table on 8-bit inputs; the
NVDLA hardware it is based on has a programmable table (see `../nvdla_ip/README.md`). TensorRT does not document
the implementation.
