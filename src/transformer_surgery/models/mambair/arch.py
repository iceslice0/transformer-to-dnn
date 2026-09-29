"""MambaIRv2 Light SR architecture — imports from the local MambaIR ``basicsr`` fork.

``MambaIRv2Light`` / ``WindowAttention`` come from ``mambal/softmax/MambaIR`` (csguoh/MambaIR),
not PyPI ``basicsr``. That tree is put on ``sys.path`` if needed; ``mamba-ssm`` must already
provide the fused selective-scan CUDA kernel (GitHub release wheel — see README).
"""

from __future__ import annotations

import sys
from pathlib import Path

_MAMBAIR_ROOT = Path(__file__).resolve().parents[4] / "mambal" / "softmax" / "MambaIR"
if _MAMBAIR_ROOT.is_dir() and str(_MAMBAIR_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAMBAIR_ROOT))

try:
    from basicsr.archs.mambairv2light_arch import MambaIRv2Light, WindowAttention
except ImportError as exc:  # pragma: no cover - optional MambaIR path
    raise ImportError(
        "MambaIRv2Light requires the MambaIR basicsr fork under mambal/softmax/MambaIR "
        "(not PyPI basicsr) plus mamba-ssm with the CUDA selective-scan kernel. "
        "See README 'MambaIR (optional)'."
    ) from exc

# Hyperparameters matching the released ``mambairv2_lightSR_x{2,4}.pth`` weights
# (from mambal/softmax/softmax_surgery.py build_model). These differ from the class defaults
# (depths (6,6,6,6) / mlp_ratio 2.0), so keep them explicit.
MAMBAIR_LIGHTSR_PRESET = dict(
    img_size=64,
    embed_dim=48,
    d_state=8,
    depths=[5, 5, 5, 5],
    num_heads=[4, 4, 4, 4],
    window_size=16,
    inner_rank=32,
    num_tokens=64,
    convffn_kernel_size=5,
    img_range=1.0,
    mlp_ratio=1.0,
    upsampler="pixelshuffledirect",
    resi_connection="1conv",
)


def build_mambair_lightsr(scale: int) -> MambaIRv2Light:
    """Build MambaIRv2 Light SR at the given upscale factor (weights match the released .pth)."""
    return MambaIRv2Light(upscale=int(scale), **MAMBAIR_LIGHTSR_PRESET)


__all__ = [
    "MambaIRv2Light",
    "WindowAttention",
    "MAMBAIR_LIGHTSR_PRESET",
    "build_mambair_lightsr",
]
