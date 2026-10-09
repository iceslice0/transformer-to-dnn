# Rockchip RK3588 NPU

Application-processor NPU (RKNPU2). Toolchain: RKNN-Toolkit2; operators outside the NPU list run on the CPU.

**Source.** RKNN Compiler Support Operator List v2.3.2 (dated 2025-03-31) and `RKNNToolKit2_OP_Support-2.3.2.md`,
airockchip/rknn-toolkit2 at commit `59a913d`, accessed 2026-10-09. The PDF has a per-platform section of NPU
operators (RK3588: section 3.2) and a separate CPU-operator section.

**On the NPU** (RK3588 section 3.2, relevant operators):

| Operator | Constraint |
|---|---|
| Mul | both inputs tensors, any broadcast (int8, fp16) |
| Div | both inputs tensors (fp16) |
| Pow | exponent one of 0, 1, 2, 3, 0.5, -0.5 (int8, fp16) |
| Sigmoid, Tanh, exGelu | |
| Softmax, exSoftmax13, exSoftmaxMask | |
| exMatMul | A [b,k,1,n] and B [b,k,1,m] both tensors, k up to 8192 |
| exNorm (LayerNorm), Gather, Max, Where | |
| exSDPAttention | fused scaled dot-product attention |

**On the CPU only:** Exp (CPU-operator section). **Not supported:** TopK, Reciprocal.

**Strict graph: whole.** Square as Mul(x, x) or Pow(x, 2), Sigmoid, inverse square root as Pow(x, -0.5).

**Extended graph: whole.** exMatMul, Mul, Div, Pow(x, -0.5), exGelu, native Softmax and exNorm; the exponential
is needed only inside softmax.

**TopK.** Not supported by the toolchain; selection stays on the host.
