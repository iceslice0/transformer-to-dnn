#!/usr/bin/env bash
# Download the third-party sources behind the NPU survey at the pinned versions.
# Files land in docs/npu/<ip>/sources/ and are git-ignored (mixed third-party licenses); the summaries
# in docs/npu/<ip>/README.md cite the same URLs and commits.
set -euo pipefail
cd "$(dirname "$0")"
get() {  # get <ip> <url> [<file name>]
  local dir="$1/sources" url="$2" name="${3:-$(basename "${2%%\?*}")}"
  mkdir -p "$dir"
  curl -fsSL --compressed "$url" -o "$dir/$name"
  echo "  $dir/$name"
}
VELA=https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/raw/dd3f874
get arm_ethos_u "$VELA/SUPPORTED_OPS.md"
get arm_ethos_u "$VELA/ethosu/regor/tflite/tflite_supported_operators_u55.cpp"
get arm_ethos_u "$VELA/ethosu/regor/tflite/tflite_supported_operators_u85.cpp"
get arm_ethos_u "$VELA/ethosu/regor/architecture/ethosu55/ethos_u55_register_cs_generator.cpp"
get arm_ethos_u "$VELA/ethosu/regor/architecture/ethosu85/ethos_u85.cpp"
VX=https://raw.githubusercontent.com/VeriSilicon/tflite-vx-delegate/b76047d
get verisilicon_vip "$VX/op_status.md"
get verisilicon_vip "$VX/op_map.cc"
ETHN=https://raw.githubusercontent.com/ARM-software/ethos-n-driver-stack/6834c82
get arm_ethos_n78 "$ETHN/SUPPORTED.md"
get arm_ethos_n78 "$ETHN/driver/support_library/src/SupportQueries.cpp"
get nvidia_jetson_dla "https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html"
NVSW=https://raw.githubusercontent.com/nvdla/sw/79538ba
get nvdla_ip "$NVSW/umd/core/include/nvdla/IType.h"
get nvdla_ip "$NVSW/umd/core/src/compiler/engine-ast/SoftMaxOp.cpp"
get nvdla_ip "https://nvdla.org/hw/v1/ias/lut-programming.html"
get google_edge_tpu "https://gweb-coral-full.uc.r.appspot.com/docs/edgetpu/models-intro/" models-intro.html
TIDL=https://raw.githubusercontent.com/TexasInstruments/edgeai-tidl-tools/7cdef46/docs
get ti_tda4vm "$TIDL/operators.md"
get ti_tda4vm "$TIDL/topk.md"
get ti_tda4vm "$TIDL/vision_transformers.md"
RKNN=https://raw.githubusercontent.com/airockchip/rknn-toolkit2/59a913d/doc
get rockchip_rk3588 "$RKNN/05_RKNN_Compiler_Support_Operator_List_V2.3.2.pdf"
get rockchip_rk3588 "$RKNN/RKNNToolKit2_OP_Support-2.3.2.md"
HYCO=https://raw.githubusercontent.com/renesas-rcar/renesas-rcar-HybridCompiler/e73e7d9
get renesas_rcar_v4h "$HYCO/02_ArchitectureOverview/03_supported_devices.adoc"
get renesas_rcar_v4h "$HYCO/05_Operations/02_supported_operators.adoc"
for op in Mul MatMul Div Pow Sqrt Exp Sigmoid Softmax Gelu LayerNormalization TopK; do
  get renesas_rcar_v4h "$HYCO/05_Operations/operator/$op.adoc"
done
echo "done"
