"""MambaIRv2 Light SR model, surgery model, and adapter.

Only the adapter is imported eagerly here (it has no heavy imports), so the adapter can be
registered on any machine. The architecture / surgery model pull in the installed MambaIR
``basicsr`` package (and ``mamba-ssm`` + CUDA to build); import them from their submodules
directly when needed:

    from transformer_surgery.models.mambair.arch import build_mambair_lightsr
    from transformer_surgery.models.mambair.surgery_model import MambaIRLightSurgeryModel
"""

from transformer_surgery.models.mambair.adapter import MambaIRLightSRAdapter

__all__ = ["MambaIRLightSRAdapter"]
