# Open-source NVDLA IP

Open hardware design released by NVIDIA in 2017 (Verilog RTL, SystemC model, compiler, Linux drivers); the
Jetson DLA is based on this architecture. Toolchain: the open NVDLA software stack (compiler and runtime).

**Sources.** NVDLA software stack at commit `79538ba` (github.com/nvdla/sw, 2019), accessed 2026-10-03; NVDLA
hardware architectural specification, LUT programming (nvdla.org/hw/v1/ias/lut-programming.html), accessed
2026-10-09. Files in `sources/`: `IType.h` (compiler layer and operation types), `SoftMaxOp.cpp`
(`CPUSoftMaxOpNode`, emulated on the CPU), `lut-programming.html`.

**Compiler layer types:** Convolution, Activation (ReLU, Sigmoid, Tanh), Pooling (min, max, average), LRN, Scale,
Softmax (CPU), Deconvolution, Concatenation, Elementwise (sum, product, min, max of two tensors), Slice.
No inverse square root, no power, no matrix multiplication between activations.

**Hardware.** The SDP and CDP units contain a programmable lookup table: two tables of 65 and 257 entries of 16
bits, linear or exponential indexing, linear interpolation inside and linear extrapolation outside the range.
The specification states that software can program it for any function; the compiler fills it only for
sigmoid and tanh (and LRN).

**Strict graph: host for the inverse square root** (LayerNorm and the reciprocal).

**Extended graph: host** for the activation MatMul, softmax and the LayerNorm inverse square root.

Both limits are compiler limits: the lookup table could hold the inverse square root.
