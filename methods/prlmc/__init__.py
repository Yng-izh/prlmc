"""PRLM-C: Protected Residual Latent Memory with Capacity-aware Consolidation.

This package implements the design in the companion note
(残差记忆设计_方案整理.md). The entry point used by the benchmark is
methods.prlmc.agent; everything else is a mechanism the adapter composes.

Layering, bottom-up:

    layout.py       slot partition of the backbone's memory pool (A | S | R)
    subspace.py     working-memory basis, feature novelty, Anchor projection
    capacity.py     occupancy, effective/energy rank, the compact trigger
    importance.py   per-slot importance weights
    consolidate.py  geometric and learned compactors
    probes.py       functional probes and the compact objective
    budget.py       byte/token/latency accounting
    metrics.py      CSR and VR (project-defined, not official benchmark metrics)
    agent.py        benchmark adapter, ablation modes B5..Ours

Nothing in here mutates the backbone's weights. The shipped MemoryLLM
checkpoint is used frozen; only the memory pool is managed.
"""

from methods.prlmc.config import PRLMCConfig
from methods.prlmc.layout import ProtectedResidualMemory, SlotLayout

__all__ = ["PRLMCConfig", "ProtectedResidualMemory", "SlotLayout"]
