# Renesas R-Car V4H / V4M (CNN-IP with DSP)

Automotive ADAS SoCs (R-Car Gen4). The deep-learning subsystem has a CNN accelerator (CNN-IP) and DSPs on chip.
Toolchain: the R-Car Hybrid Compiler (HyCo), which partitions a model between CNN-IP and DSP.

**Source.** renesas-rcar-HybridCompiler at commit `e73e7d9` (github.com/renesas-rcar/renesas-rcar-HybridCompiler),
accessed 2026-10-09. Files in `sources/`: `03_supported_devices.adoc` (V4H version 2, V4M version 1),
`02_supported_operators.adoc` (matrix: frontend, CNN-IP int8/int16, DSP int8/fp32) and per-operator pages.

**Support matrix** (relevant ONNX operators):

| Operator | CNN-IP int8 | CNN-IP int16 | DSP int8 | DSP fp32 |
|---|---|---|---|---|
| Mul | yes | yes | yes | yes |
| Relu, Sigmoid | yes | yes | yes | yes |
| Tanh | yes | yes | yes | no |
| ReduceMean | yes | no | yes | yes |
| MatMul | no | no | yes | yes |
| Div, Exp, Log, Pow | no | no | yes | yes |
| Sqrt | no | no | yes | no |
| Softmax, LayerNormalization | no | no | yes | yes |
| Gelu | no | no | yes | no |
| ReduceMax, ReduceSum | no | no | yes | yes |
| Gather, TopK | no | no | yes | yes |

**Strict graph: CNN-IP plus DSP.** The CNN-IP runs squares (Mul) and the sigmoid; the inverse square root
(Pow or Sqrt) runs on the DSP. Nothing goes to the host CPU.

**Extended graph: CNN-IP plus DSP.** Activation MatMul, softmax, LayerNorm and GELU run on the DSP. The CNN-IP has
Mul and ReduceMean, so the Hadamard route of Proposition 1 (product, then mean times N) would also fit it, but
HyCo places MatMul on the DSP.

**TopK.** DSP.
