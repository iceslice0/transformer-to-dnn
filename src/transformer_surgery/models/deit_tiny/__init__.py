"""DeiT-Tiny surgery model and adapters.

Only the adapters are imported eagerly here (lightweight), so they can be registered on any
machine. Import the surgery model from its submodule when needed:

    from transformer_surgery.models.deit_tiny.surgery_model import DeiTTinySurgeryModel
"""

from transformer_surgery.models.deit_tiny.adapter import DeiTTinyImageNetAdapter, DeiTTinyPetAdapter

__all__ = ["DeiTTinyImageNetAdapter", "DeiTTinyPetAdapter"]
