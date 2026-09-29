"""Pythia / GPT-NeoX causal-LM surgery patient.

Only the adapter is imported eagerly (lightweight registration). Import the surgery model
from its submodule when needed:

    from transformer_surgery.models.pythia.surgery_model import PythiaSurgeryModel
"""

from transformer_surgery.models.pythia.adapter import Pythia70MWikiText2Adapter

__all__ = ["Pythia70MWikiText2Adapter"]
