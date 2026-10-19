# NPU survey: can the compiled graph run entirely on the accelerator?

For each edge accelerator the toolchain's own operator list (code at a
pinned commit, or vendor documentation at an access date) decides which operations of the dense compiled
graphs run on the accelerator. The verdicts below are toolchain placements, not compiler runs on our
exported graphs.

What each graph needs:

- **Strict graph** (`allow_matmul=false`): fixed affine maps, routing, the square, the sigmoid and the
  inverse square root (the reciprocal is `rsqrt(x)^2`, the exponential is `sigmoid(t)/sigmoid(-t)`), plus GELU.
- **Hadamard graph** (`hadamard=true`): the operations of the strict graph, with the elementwise product between
  two activations (of equal shape after the same broadcast routing the strict graph uses) instead of an
  activation and itself. Its placement is therefore that of the strict graph. Broadcast limits in the toolchains:
  the Jetson DLA broadcasts only NC11 or N111 operands; the Ethos-N78 multiplies only equal shapes.
- **Extended graph**: fixed affine maps, routing, the exponential, the reciprocal, the inverse square root,
  GELU, activation-activation MatMul and the elementwise (Hadamard) product.

Rules used for the verdicts:

- A square runs as the elementwise product `x * x`, which every accelerator below supports.
- Softmax counts as on the accelerator when it runs as a native row-wise operator, or when the exponential and
  the reciprocal both run there.
- Where an accelerator lacks GELU, it is replaced by ReLU (fine-tuned) or by `x * sigmoid(1.702 x)`.
  Accelerators that run the extended graph implement GELU.

| Accelerator | Summary | Strict graph | Hadamard graph | Extended graph | TopK |
|---|---|---|---|---|---|
| Arm Ethos-U55/U65 | [arm_ethos_u](arm_ethos_u/README.md) | Whole | Whole | Whole (MatMul as elementwise product and sum) | host |
| Arm Ethos-U85 | [arm_ethos_u](arm_ethos_u/README.md) | Whole | Whole | Whole | host |
| VeriSilicon VIP | [verisilicon_vip](verisilicon_vip/README.md) | Whole | Whole | Whole | host |
| TI TDA4VM (C7x with MMA) | [ti_tda4vm](ti_tda4vm/README.md) | Whole | Whole | Whole | accelerator |
| Rockchip RK3588 | [rockchip_rk3588](rockchip_rk3588/README.md) | Whole | Whole | Whole | host (not supported) |
| Renesas R-Car V4H/V4M (CNN-IP with DSP) | [renesas_rcar_v4h](renesas_rcar_v4h/README.md) | DSP: inverse square root | DSP: inverse square root | DSP: act-act MatMul, softmax, LayerNorm, GELU | DSP |
| NVIDIA Jetson Xavier/Orin DLA | [nvidia_jetson_dla](nvidia_jetson_dla/README.md) | Whole (INT8) | Whole (INT8) | Host: act-act MatMul | host |
| Google Edge TPU | [google_edge_tpu](google_edge_tpu/README.md) | Whole | Whole | Host: act-act MatMul, row-wise softmax | host |
| Open-source NVDLA IP | [nvdla_ip](nvdla_ip/README.md) | Host: inverse square root | Host: inverse square root | Host: act-act MatMul, softmax, LayerNorm inverse square root | host |
| Arm Ethos-N78 | [arm_ethos_n78](arm_ethos_n78/README.md) | Host: inverse square root | Host: inverse square root | Host: act-act MatMul, softmax, LayerNorm inverse square root | host |

On R-Car the listed operations run on the on-chip DSP, not on the host CPU.

Toolchain limits, not hardware limits:

- The Edge TPU compiler splits a model only once: the first unsupported operation and everything after it run on
  the CPU. The strict graph is unaffected; the extended graph moves to the CPU from the first attention score.
- NVIDIA: the NVDLA IP that the Jetson DLA is based on has a programmable two-table lookup unit with linear
  interpolation (65 and 257 entries) in its activation and normalization units, which can hold every function of
  both unary sets. The open NVDLA compiler fills it only for sigmoid and tanh; TensorRT also exposes INT8 power,
  division, sine, cosine and arctangent, all restricted to 8-bit inputs as a 256-entry table would be.

## Files

Each `<ip>/README.md` summarizes one accelerator with its source, version and the operator facts behind the
verdicts. `fetch_sources.sh` downloads the third-party sources at the pinned versions into `<ip>/sources/`.
Those copies carry mixed licenses (Apache-2.0, MIT, BSD, NVIDIA Open NVDLA License, vendor web documentation),
so `.gitignore` excludes them from the repository; run the script to rebuild them.
