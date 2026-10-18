"""Training-free slot-level consolidation (v6 Plan A).

Replaces the 19->9 blind carrier-residual merge with a slot-level, zero-parameter
operator:

(a) every native slot is scored with Dispersion/Leverage signals -- the slot's
    orthogonal distance to its own block centroid (dispersion) and to the
    block's first principal direction (leverage). E7a-sim (docs/01:9) showed
    block-level cosine similarity is anisotropic (0.9976-0.9991) and carries no
    functional signal, so cosine is deliberately NOT used to rank importance;
    it is only used as a conservative hard gate below.
(b) the top `target` slots by score are kept verbatim (100% native vectors,
    chronological order preserved). A dropped slot whose cosine to its nearest
    same-neighbourhood kept slot exceeds `merge_tau` is convex-combined into
    that kept slot with normalised significance weights; everything else is
    dropped outright.
(c) no trainable parameter, no low-rank projection: every output row is either
    a verbatim native slot or a convex combination of native slots (weights
    sum to one), so the output stays inside the native span.

Step-0 contract: when `target >= slots` the operator returns the window
untouched, exactly like Selection.
"""
from typing import Optional

import torch

from .consolidate import _EPS, _unit, _SELECT_LAYER

# Conservative hard gate for convex merging. E7a-sim measured reader
# representation cosine between *different* contexts at 0.9976-0.9991, so the
# bar for "same neighbourhood, safe to average" must sit above that band.
DEFAULT_MERGE_TAU = 0.99


def _slot_scores(source: torch.Tensor, ref_summary: torch.Tensor, block: int) -> torch.Tensor:
    """Per-slot Dispersion/Leverage score in the reference layer.

    Dispersion: orthogonal distance from the slot to its block centroid.
    Leverage:   distance from the slot to the block's first principal direction
                (residual after removing the top-1 component of the centred
                block). Both are max-normalised and averaged.
    """
    layers, slots, hidden = source.shape
    b = slots // block
    x = ref_summary.view(b, block, hidden)
    centroid = x.mean(dim=1, keepdim=True)
    centred = x - centroid
    dispersion = centred.norm(dim=-1)

    leverage = torch.zeros_like(dispersion)
    for i in range(b):
        blk = centred[i]
        # Top-1 principal direction of the centred block; a zero-centred block
        # (single slot or identical rows) has no principal direction.
        if torch.linalg.matrix_rank(blk) >= 1 and blk.abs().sum() > 0:
            _, _, vt = torch.linalg.svd(blk, full_matrices=False)
            v = vt[0]
            projected = (blk @ v).unsqueeze(-1) * v
            leverage[i] = (blk - projected).norm(dim=-1)
    score = 0.5 * _unit(dispersion.reshape(-1)) + 0.5 * _unit(leverage.reshape(-1))
    return score


def _nearest_kept(summary: torch.Tensor, kept: torch.Tensor, dropped: torch.Tensor):
    """Most similar kept slot for every dropped slot, plus the cosine."""
    unit = summary / (summary.norm(dim=-1, keepdim=True) + _EPS)
    sim = unit[dropped] @ unit[kept].T  # [n_dropped, n_kept]
    best = sim.argmax(dim=-1)
    return best, sim.gather(1, best.unsqueeze(1)).squeeze(1)


def consolidate_slots(
    source: torch.Tensor,
    target: int,
    block: int,
    weights: Optional[torch.Tensor] = None,
    merge_tau: float = DEFAULT_MERGE_TAU,
    source_allow_mask=None,
    reference_layer: int = _SELECT_LAYER,
):
    """Slot-level consolidation. Returns (output [L, target, d], kept_indices, merge_report).

    Kept slots are chosen by Dispersion/Leverage score, returned in chronological
    order and stay 100% native. Dropped slots merge into the most similar kept
    slot of the same or an adjacent block only when their cosine exceeds
    `merge_tau`, as a significance-weighted convex combination.
    """
    layers, slots, hidden = source.shape
    if slots % block or target % block:
        raise ValueError("window and target must be multiples of the native block")
    if not 0 < target <= slots:
        raise ValueError("target must be positive and no larger than the window")
    if target == slots:
        return source.contiguous(), torch.arange(slots), dict(merged_slots=[])
    if target > slots - (slots // block):
        # Leave at least one block of slack so a degenerate keep-everything
        # call cannot silently bypass the compression contract.
        raise ValueError("target leaves less than one block dropped; tighten the budget")

    ref = min(max(reference_layer, 0), layers - 1)
    summary = source[ref].detach().to(torch.float32)
    score = _slot_scores(source, summary, block)

    allow = torch.ones(slots, dtype=torch.bool, device=source.device)
    if source_allow_mask is not None:
        allow = torch.as_tensor(source_allow_mask, device=source.device)
        if allow.dtype != torch.bool or allow.shape != (slots,) or not allow.any():
            raise ValueError("boolean source mask must be per-slot, bool and non-empty")

    # Chronological candidate order with score as the sort key: top-`target`
    # among allowed slots, then sorted so the output keeps temporal order.
    order = torch.argsort(score, descending=True, stable=True)
    allowed_order = order[allow[order]]
    if allowed_order.numel() < target:
        raise ValueError("source mask leaves fewer than `target` allowed slots")
    kept = allowed_order[:target].sort().values

    all_idx = torch.arange(slots, device=source.device)
    dropped = all_idx[allow & ~torch.isin(all_idx, kept)]
    if dropped.numel() == 0:
        return source.index_select(1, kept).contiguous(), kept, dict(merged_slots=[])

    best, cos = _nearest_kept(summary, kept, dropped)
    blocks = all_idx // block
    kept_blocks = kept // block
    # Same-neighbourhood constraint: the merge target block must be the
    # dropped slot's own block or an adjacent one.
    near = (kept_blocks[best] - blocks[dropped]).abs() <= 1
    mergeable = near & (cos > merge_tau)

    if weights is not None and weights.numel() == slots:
        significance = weights.detach().to(torch.float32).reshape(-1).clamp_min(0.)
    else:
        significance = score.clamp_min(0.)
    if float(significance.sum()) <= 0:
        significance = torch.ones_like(significance)

    out = source.index_select(1, kept).to(torch.float32).clone()
    # Assign every mergeable dropped slot to its most similar kept slot.
    folded = {}
    for local in torch.nonzero(mergeable, as_tuple=False).flatten().tolist():
        folded.setdefault(int(best[local]), []).append(int(dropped[local]))
    # Fold merges per kept slot: normalised weighted average over
    # {kept slot} + assigned dropped slots, a convex combination (sums to one).
    for k_local, sources in folded.items():
        k = int(kept[k_local])
        wk = float(significance[k])
        num = wk * out[:, k_local]
        den = wk
        for j in sources:
            num = num + float(significance[j]) * source[:, j].to(torch.float32)
            den += float(significance[j])
        out[:, k_local] = num / den

    report = dict(
        merged_slots=[
            dict(into=int(kept[k_local]), sources=sources,
                 weights=[float(significance[j]) for j in sources])
            for k_local, sources in folded.items()
        ],
        merge_tau=float(merge_tau),
        kept_slots=kept.detach().cpu().tolist(),
    )
    return out.to(source.dtype).contiguous(), kept, report


class SlotConsolidateMerge(torch.nn.Module):
    """Checkpoint-compatible shell around `consolidate_slots`.

    Holds zero parameters so `--arm Consolidate` runs through the standard
    MergeStream contract without any learned module and without touching the
    live training protocol.
    """

    architecture = "slot_consolidate_v6"

    def __init__(self, num_layers: int, hidden: int, block: int,
                 merge_tau: float = DEFAULT_MERGE_TAU):
        super().__init__()
        if block <= 0 or num_layers <= 0 or hidden <= 0:
            raise ValueError("positive sizes required")
        self.num_layers = int(num_layers)
        self.hidden = int(hidden)
        self.block = int(block)
        self.merge_tau = float(merge_tau)
        self.last_fusion = dict()

    @property
    def reference_layer(self):
        return _SELECT_LAYER

    def forward(self, source, weights=None, target=None, block=None,
                goal_direction=None, select_weights=None, carrier_indices=None,
                source_allow_mask=None):
        if goal_direction is not None or select_weights is not None:
            # The score is deliberately query-independent and cosine-free;
            # extraneous routing inputs are rejected rather than ignored.
            raise ValueError("SlotConsolidateMerge takes no goal/routing input")
        if target is None:
            raise ValueError("SlotConsolidateMerge needs an explicit target width")
        layers, slots, hidden = source.shape
        block = self.block if block is None else int(block)
        out, kept, report = consolidate_slots(
            source, target=int(target), block=block, weights=weights,
            merge_tau=self.merge_tau, source_allow_mask=source_allow_mask)
        self.last_fusion = report
        return out
