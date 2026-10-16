"""Cost accounting for PRLM-C.

Section 5.2 requires the budget axes to be reported separately and never
conflated: latent token count, total persistent-state bytes, the model's actual
context length, and inference cost. Section 3.4 adds the rule that basis,
statistics, probe caches, and compaction scratch are *not* free metadata and must
be counted.

The most important thing this module does is refuse to let a rank reduction be
recorded as a byte saving. Truncated SVD produces a low-rank K x d matrix; it
does not shrink the pool, because the pool is a trained parameter of fixed shape
that the forward pass reads in full. `BudgetReport` therefore reports the
physical bytes and the addressable tokens as separate fields, and
`byte_saving_ratio` is explicitly 0 for a token-only compaction.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


def tensor_bytes(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


@dataclass
class BudgetReport:
    """A snapshot of every cost axis, kept deliberately unaggregated."""

    latent_tokens_used: int = 0
    latent_tokens_budget: int = 0
    anchor_tokens: int = 0
    consolidated_tokens: int = 0
    residual_tokens: int = 0

    pool_bytes: int = 0
    auxiliary_bytes: int = 0
    anchor_state_bytes: int = 0
    projection_bytes: int = 0

    context_tokens: int = 0
    compactions: int = 0
    evicted_blocks: int = 0
    skipped_chunks: int = 0

    write_seconds: float = 0.0
    query_seconds: float = 0.0
    read_seconds: float = 0.0

    peak_inference_tokens: int = 0
    rank_reduction: float = 0.0

    extra: Dict[str, float] = field(default_factory=dict)

    @property
    def persistent_bytes(self) -> int:
        """Pool plus every auxiliary structure the note refuses to treat as free."""
        return self.pool_bytes + self.auxiliary_bytes

    @property
    def byte_saving_ratio(self) -> float:
        """Fraction of persistent bytes a token compaction actually removes.

        Zero by construction for token-only compaction: the pool parameter keeps
        its shape, so compaction changes what is addressable, not what is
        allocated. Reporting anything else would repeat the error section 3.4
        warns about.
        """
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latent_tokens_used": self.latent_tokens_used,
            "latent_tokens_budget": self.latent_tokens_budget,
            "anchor_tokens": self.anchor_tokens,
            "consolidated_tokens": self.consolidated_tokens,
            "residual_tokens": self.residual_tokens,
            "persistent_bytes": self.persistent_bytes,
            "pool_bytes": self.pool_bytes,
            "auxiliary_bytes": self.auxiliary_bytes,
            "anchor_state_bytes": self.anchor_state_bytes,
            "projection_bytes": self.projection_bytes,
            "byte_saving_ratio": self.byte_saving_ratio,
            "context_tokens": self.context_tokens,
            "compactions": self.compactions,
            "evicted_blocks": self.evicted_blocks,
            "skipped_chunks": self.skipped_chunks,
            "write_seconds": self.write_seconds,
            "query_seconds": self.query_seconds,
            "read_seconds": self.read_seconds,
            "peak_inference_tokens": self.peak_inference_tokens,
            "rank_reduction": self.rank_reduction,
            **self.extra,
        }


class BudgetTracker:
    """Accumulate cost across one episode and emit a BudgetReport on demand."""

    def __init__(self, manager, context_max_length: Optional[int] = None):
        self.manager = manager
        self.context_max_length = context_max_length
        self.write_seconds = 0.0
        self.query_seconds = 0.0
        self.read_seconds = 0.0
        self.peak_inference_tokens = 0
        self.marks: List[Dict[str, Any]] = []

    def record_write(self, seconds: float, tokens: int) -> None:
        self.write_seconds += seconds
        self.peak_inference_tokens = max(self.peak_inference_tokens, tokens)

    def record_query(self, seconds: float, tokens: int) -> None:
        self.query_seconds += seconds
        self.peak_inference_tokens = max(self.peak_inference_tokens, tokens)

    def record_read(self, seconds: float) -> None:
        self.read_seconds += seconds

    def mark(self, label: str) -> None:
        """Record a labelled checkpoint so a compaction's cost is attributable."""
        self.marks.append({"label": label, "used_slots": self.manager.used})

    def snapshot(self, anchor_tokens: int = 0) -> BudgetReport:
        manager = self.manager
        total_budget = manager.layout.working_capacity
        # Peak occupancy is tracked by the manager on every write rather than
        # recomputed from `manager.history`: the stats file drains that buffer,
        # so reading it here would report only the writes since the last drain.
        rank_reduction = 0.0
        peak = int(getattr(manager, "peak_used", 0))
        if peak > 0:
            rank_reduction = 1.0 - (manager.used / float(peak))

        return BudgetReport(
            latent_tokens_used=manager.used,
            latent_tokens_budget=total_budget,
            anchor_tokens=manager.layout.k_anchor if manager.anchor_slots is not None else 0,
            consolidated_tokens=manager.n_consolidated,
            residual_tokens=max(0, manager.used - manager.n_consolidated),
            pool_bytes=manager.memory_nbytes(),
            auxiliary_bytes=manager.auxiliary_nbytes(),
            anchor_state_bytes=tensor_bytes(manager.anchor_slots),
            projection_bytes=tensor_bytes(manager.projection),
            context_tokens=self.context_max_length or 0,
            compactions=manager.compaction_count,
            evicted_blocks=manager.evicted_blocks,
            skipped_chunks=manager.skipped_chunks,
            write_seconds=self.write_seconds,
            query_seconds=self.query_seconds,
            read_seconds=self.read_seconds,
            peak_inference_tokens=self.peak_inference_tokens,
            rank_reduction=rank_reduction,
            extra={"anchor_tokens_recorded": float(anchor_tokens)},
        )
