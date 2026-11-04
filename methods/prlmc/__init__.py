"""PRLM-C: shared residual memory fusion on a frozen MemoryLLM reader.

Layering, bottom-up:

    layout.py          slot partition of the backbone's memory pool (A | S | R)
    subspace.py        working-memory basis, feature novelty, anchor projection
    capacity.py        occupancy, effective/energy rank, the compact trigger
    importance.py      per-slot importance weights
    consolidate.py     carrier selection and the geometric compactors
    probes.py          differentiable memory-pool injection used by the objective
    manifold_graft.py  the V10 operator: manifold-preserving slot grafting

Nothing here mutates the backbone's weights. The shipped MemoryLLM checkpoint is
used frozen; only the shared merge operator carries trainable parameters.
"""

from methods.prlmc.config import PRLMCConfig
from methods.prlmc.layout import ProtectedResidualMemory, SlotLayout
from methods.prlmc.consolidate import FixedBasisCarrierResidualCompactor
from methods.prlmc.manifold_graft import ManifoldGraftedSlotMerge

__all__ = [
    "PRLMCConfig",
    "ProtectedResidualMemory",
    "SlotLayout",
    "FixedBasisCarrierResidualCompactor",
    "ManifoldGraftedSlotMerge",
]
