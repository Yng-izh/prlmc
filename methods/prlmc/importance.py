"""Per-slot importance weights.

Section 3.4 lists the candidate factors: goal relevance, reuse count, recency,
confidence, surprise, and explicit user emphasis. It also requires that the first
version use weights a benchmark can produce reproducibly, rather than depending
on an unvalidated "importance model", so every factor here is computed from
quantities the run already has.

The important structural point from section 7 (risk 2) is that *novelty is not
importance*: a rare prohibition can be low-frequency and critical, while a
high-novelty observation can be noise. Novelty therefore enters only as one
weakly weighted factor, and the goal factor carries the largest default weight.
"""

from typing import Dict, Optional

import torch

DEFAULT_WEIGHTS: Dict[str, float] = {
    "goal": 1.0,
    "reuse": 0.5,
    "recency": 0.5,
    "confidence": 0.5,
    "surprise": 0.5,
    "emphasis": 1.0,
}

# Cheap, corpus-independent cue that a span states a constraint rather than a
# fact. It is deliberately narrow: the note asks for programmatically checkable
# emphasis, not a classifier.
EMPHASIS_CUES = (
    "must", "must not", "do not", "don't", "never", "forbidden", "prohibited",
    "required", "require", "shall", "only", "禁止", "不得", "必须", "务必",
)


class ImportanceTracker:
    """Accumulate per-slot importance signals and combine them into one score.

    The tracker owns the reuse counter because reuse is the only factor that
    must survive across updates; everything else is recomputed from the current
    state. Slot identity is (layer-agnostic) slot index, since the manager keeps
    the same layout in every layer.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        merged = dict(DEFAULT_WEIGHTS)
        if weights:
            merged.update({k: float(v) for k, v in weights.items()})
        self.weights = merged
        self._reuse: Optional[torch.Tensor] = None

    def _ensure(self, n_slots: int, device) -> None:
        if self._reuse is None or self._reuse.numel() < n_slots:
            grown = torch.zeros(n_slots, device=device, dtype=torch.float32)
            if self._reuse is not None:
                grown[: self._reuse.numel()] = self._reuse
            self._reuse = grown

    def mark_reuse(self, indices: torch.Tensor, n_slots: int) -> None:
        """Increment the reuse counter for slots that were read or rewritten."""
        if indices is None or indices.numel() == 0:
            return
        self._ensure(n_slots, indices.device)
        self._reuse[indices.long()] += 1.0

    def score(
        self,
        slots: torch.Tensor,
        goal_direction: Optional[torch.Tensor] = None,
        recency: Optional[torch.Tensor] = None,
        surprise: Optional[torch.Tensor] = None,
        emphasis: Optional[torch.Tensor] = None,
        n_slots: Optional[int] = None,
    ) -> torch.Tensor:
        """Combine the available factors into a [K] importance vector in [0, 1].

        Factors that are not supplied are dropped and the remaining weights are
        renormalised, so a partially observed state still yields a usable score
        rather than a silently biased one.
        """
        k = slots.shape[0]
        if k == 0:
            return slots.new_zeros(0)

        device = slots.device
        terms = []
        weights = []

        if goal_direction is not None and goal_direction.numel() > 0:
            direction = goal_direction.to(device=device, dtype=torch.float32).flatten()
            direction = direction / (direction.norm() + 1e-8)
            rows = slots.detach().to(torch.float32)
            rows = rows / (rows.norm(dim=-1, keepdim=True) + 1e-8)
            similarity = (rows @ direction).clamp(-1.0, 1.0)
            terms.append(((similarity + 1.0) / 2.0))
            weights.append(self.weights.get("goal", 1.0))

        if self._reuse is not None and self._reuse.numel() > 0:
            reuse = self._reuse[:k].to(device=device, dtype=torch.float32)
            terms.append(reuse / (reuse.max() + 1e-8))
            weights.append(self.weights.get("reuse", 0.5))

        if recency is not None and recency.numel() == k:
            value = recency.to(device=device, dtype=torch.float32)
            terms.append(value / (value.max() + 1e-8))
            weights.append(self.weights.get("recency", 0.5))

        if surprise is not None and surprise.numel() == k:
            value = surprise.to(device=device, dtype=torch.float32).abs()
            terms.append(value / (value.max() + 1e-8))
            weights.append(self.weights.get("surprise", 0.5))

        if emphasis is not None and emphasis.numel() == k:
            terms.append(emphasis.to(device=device, dtype=torch.float32).clamp(0.0, 1.0))
            weights.append(self.weights.get("emphasis", 1.0))

        # Confidence has no per-slot source in this benchmark; a constant factor
        # would only rescale the score, so it is recorded but not applied. The
        # weight stays in the config so a future confidence head can use it.
        terms.append(torch.ones(k, device=device, dtype=torch.float32))
        weights.append(0.0)

        stacked = torch.stack(terms, dim=0)
        weight_vector = torch.tensor(weights, device=device, dtype=torch.float32)
        if float(weight_vector.sum()) <= 0.0:
            return torch.ones(k, device=device, dtype=torch.float32)

        combined = (stacked * weight_vector.unsqueeze(1)).sum(dim=0) / weight_vector.sum()
        return combined.clamp(0.0, 1.0)

    def advance_recency(self) -> None:
        """Decay the reuse counter so recency does not become a pure popularity count."""
        if self._reuse is not None:
            self._reuse *= 0.99


def emphasis_from_text(text: str) -> float:
    """Return 1.0 when a span states a constraint, else 0.0.

    Exposed separately from the tracker so the same rule can be applied while
    building Anchor text and while scoring candidate slots.
    """
    lowered = text.lower()
    return 1.0 if any(cue in lowered for cue in EMPHASIS_CUES) else 0.0
