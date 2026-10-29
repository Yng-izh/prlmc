"""Protected residual latent memory over the backbone's memory pool.

Section 3.1 of the note specifies the state

    M_t = [A, S_t, R_t],   K_A + K_S + K_R <= K_max

with A a read-only Anchor holding the initial goal and hard constraints, S the
consolidated working memory, and R the fast residual. This module owns that
partition and every mutation of the pool.

Why the pool is managed by hand rather than through the backbone's own update
-----------------------------------------------------------------------------
The shipped `update_memory_with_delta_memory` appends each injection and then
calls `drop_memory`, which evicts `num_tokens` slots chosen by `torch.randperm`
with no seed. That is correct for MemoryLLM as published but it is exactly the
behaviour PRLM-C must replace: eviction is unweighted, unmonitored, and it will
happily discard an Anchor slot. So injection is run with `update_memory=False`
to obtain the delta block without touching the pool, and placement, protection,
eviction and consolidation happen here.

Three facts about the backbone that fix the layout
-------------------------------------------------
1. Injection conditions on the *trailing* `num_tokens` slots of the pool
   (`cat_memory_and_hiddens`, `is_injection=True`, reads `self.memory[idx]
   [-num_tokens:]`), and `update_memory_with_delta_memory` appends the new delta
   at the end. The newest block must therefore occupy the final `num_tokens`
   slots. Every write shifts the existing distinct content left by one block and
   appends the new block at the very end.
2. `inject_memory` returns `delta_memory` of shape [batch, L, num_tokens, d];
   batch is dropped here so a block is [L, n, d].
3. The forward pass reads the entire pool at query time. The pool can therefore
   never be left with holes: free space is tiled with copies of occupied
   content, which is in-distribution, because the backbone's own first-write path
   tiles a short delta across the whole pool.

Two consequences worth stating plainly
-------------------------------------
* The Anchor is not part of the injection prefix, so it does not condition
  writes directly. Its influence on writes comes from the subspace projection
  (section 3.2), which is off by default.
* Capacity is physical and fixed; compaction reduces the *addressable* count.
  The pool is a trained parameter of shape [L, K_max, d] and the forward reads
  all of it, so bytes do not shrink when K' < K. `used` is the logical token
  count and `budget.snapshot()` reports the two axes separately, as section 3.4
  and risk 5 require.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from methods.prlmc.capacity import (
    CapacitySnapshot,
    CompactTrigger,
    effective_rank,
    energy_rank,
    occupancy_ratio,
)
from methods.prlmc.config import PRLMCConfig
from methods.prlmc.consolidate import (
    CarrierResidualCompactor,
    FixedBasisCarrierResidualCompactor,
    _SELECT_LAYER,
    _select_block_indices,
    build_compactor,
    recent_block_compact,
    select_block_compact,
)
from methods.prlmc.importance import ImportanceTracker
from methods.prlmc.subspace import (
    AnchorProjection,
    NoveltyEMA,
    apply_projection,
    build_projection,
    estimate_basis,
    feature_novelty,
)

MIN_INJECT_TOKENS = 16

# Row cap for the per-write decompositions. A full SVD of [12800, 4096] costs
# seconds; the capacity and basis signals are coarse by design, so a bounded
# subsample gives the same decisions far more cheaply. Section 3.3 treats rank as
# a redundancy indicator, not a precise quantity.
_SPECTRUM_MAX_ROWS = 2048
_BASIS_MAX_ROWS = 1024

# Slot-copy granularity for the left shift and the free-space tiling. Bounds the
# transient buffer to this many slots per layer.
_COPY_CHUNK = 512

# The layer every geometric quantity is computed in: the basis, the rank
# spectrum, the goal direction, the novelty candidate and the block addresses.
# One fixed layer rather than an average across layers, because a basis fitted to
# layer 0 and a candidate averaged over all 32 layers live in different spaces and
# their cosine has no stable meaning. Keeping the choice explicit in one constant
# is what makes it checkable.
#
# It is 16, the same semantic mid layer the merge routes in
# (`TRAIN_V4` variants use `reference_layer = 16`) and the same layer the carrier
# selector scores blocks in (`consolidate._SELECT_LAYER`). Layer 0 features are
# dominated by shallow detokenisation, so a carrier set, a goal direction and a
# routing read taken from different layers describe different manifolds -- the
# reason an in-domain gain could vanish on another dataset.
_REFERENCE_LAYER = 16


class SlotLayout:
    """Index arithmetic for the A | working partition.

    The working region is the suffix [k_anchor, total). Within it, the distinct
    slots are the trailing `used`, ordered oldest-first, so the newest block is
    always last. The prefix [k_anchor, total - used) is filler holding tiled
    copies of occupied content.
    """

    def __init__(self, k_anchor: int, total: int, block: int, k_max: Optional[int] = None):
        if not 0 <= k_anchor < total:
            raise ValueError(f"k_anchor must be in [0, {total}), got {k_anchor}")
        self.k_anchor = int(k_anchor)
        self.total = int(total)
        self.block = int(block)
        # `k_max` is the addressable working-slot budget: how many distinct slots
        # may occupy the region outside the Anchor. `None` means "whatever the
        # Anchor leaves", which is the pre-E6c behaviour.
        #
        # It cannot be expressed by narrowing the pool: the fork derives
        # `cache_position` from the config constants, so a shorter tensor would
        # desynchronise the rotary offsets. The pool keeps its physical width and
        # the budget alone shrinks, which grows the *masked* filler and leaves the
        # Anchor, the block grid, and every offset untouched.
        self.k_max = None if k_max is None else int(k_max)
        if self.k_max is not None:
            if self.k_max <= 0:
                raise ValueError("k_max must be positive, got %d" % self.k_max)
            # A write places one whole block at a time, so the budget must admit
            # at least one. Rounded *down* to whole blocks so the effective budget
            # is never looser than requested.
            self.k_max = (self.k_max // self.block) * self.block
            if self.k_max < self.block:
                raise ValueError(
                    "k_max leaves less than one %d-slot block of capacity" % self.block
                )

    @property
    def working_capacity(self) -> int:
        budget = self.total - self.k_anchor
        if self.k_max is not None:
            budget = min(budget, self.k_max)
        return budget

    def anchor_slice(self) -> slice:
        return slice(0, self.k_anchor)

    def working_bounds(self, used: int) -> Tuple[int, int]:
        """Half-open [start, end) of the distinct working suffix."""
        start = self.total - used
        return start, self.total

    def consolidated_bounds(self, used: int, n_consolidated: int) -> Tuple[int, int]:
        """The consolidated region sits at the front of the distinct suffix.

        Appends shift the whole distinct region left without reordering it, so
        this invariant survives repeated writes.
        """
        start, _ = self.working_bounds(used)
        return start, start + n_consolidated

    def residual_bounds(self, used: int, n_consolidated: int) -> Tuple[int, int]:
        start, end = self.working_bounds(used)
        return start + n_consolidated, end

    def free_bounds(self, used: int) -> Tuple[int, int]:
        return self.k_anchor, self.total - used

    def to_dict(self) -> Dict[str, Optional[int]]:
        return {
            "k_anchor": self.k_anchor,
            "total": self.total,
            "working_capacity": self.working_capacity,
            "k_max": self.k_max,
        }


def summarise_block(delta: torch.Tensor, layer: int = 0) -> torch.Tensor:
    """Reduce a [L, n, d] delta block to one [d] candidate representation.

    Used only for the novelty proxy and for block addressing. Averaging over
    tokens is coarse by construction; the note calls rho a proxy and requires it
    to allocate capacity rather than judge value, so a coarse summary is
    acceptable here and a precise one would not change the decisions it is
    allowed to drive.

    `layer` fixes which layer the summary comes from, and it defaults to the
    layer `_refresh_basis` and `_spectrum` already use. Averaging across layers
    as well (the previous behaviour) mixed two different things: the candidate
    then lived in a layer-averaged space while the basis it was scored against
    was estimated from layer 0 alone. The cosine and the novelty therefore had no
    stable geometric meaning, which is what made a three-decimal threshold like
    0.885 meaningless.
    """
    if delta.dim() == 4:
        delta = delta[0]
    if delta.dim() == 3:
        delta = delta[layer]
    return delta.mean(dim=0)


def fit_width(block: torch.Tensor, width: int) -> torch.Tensor:
    """Resize a [L, n, d] block to exactly `width` slots along dim 1.

    In-place addressed updates add a delta into an existing block, and the two
    widths need not agree: an injection block is at most `num_tokens` wide but a
    chunk shorter than that yields a narrower delta. Truncation and tiling keep
    the update defined instead of raising on a shape mismatch mid-episode.
    """
    n = block.shape[1]
    if n == width:
        return block
    if n > width:
        return block[:, :width, :]
    reps = (width + n - 1) // n
    return block.repeat(1, reps, 1)[:, :width, :]


class ProtectedResidualMemory:
    """Owns the pool partition, the write path, and capacity-triggered compaction."""

    # v9 write-once lock defaults; the tensor itself is created lazily by
    # `_fused_flags`, which keeps every existing construction path intact.
    slot_fused = None
    archive_fraction = .5
    archive_max_blocks = 3

    def __init__(self, model, tokenizer, config: PRLMCConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = config

        memory = model.memory.data if hasattr(model.memory, "data") else model.memory
        self.num_layers = int(memory.shape[0])
        self.total = int(memory.shape[1])
        self.hidden = int(memory.shape[2])
        self.block = int(getattr(model, "num_tokens", 256))
        self.dtype = memory.dtype

        k_anchor = config.k_anchor
        if k_anchor % self.block != 0:
            # Anchor width must be a whole number of injection blocks, otherwise
            # the anchor write would straddle a block boundary.
            k_anchor = max(self.block, (k_anchor // self.block) * self.block)
        # `config.k_max` tightens the *working* budget without touching the pool:
        # see `SlotLayout` for why the physical width can never shrink. This is
        # the only place the layout is built, so one value here governs placement,
        # the trigger, the active mask, and `budget.snapshot()` alike.
        self.layout = SlotLayout(k_anchor, self.total, self.block, config.k_max)

        self.used = 0
        self.n_consolidated = 0
        # Width of the most recently appended block. On the warm path this is
        # num_tokens, but a chunk shorter than num_tokens yields a narrower block
        # (the delta is the trailing slice of the concatenated sequence), so the
        # newest block cannot be assumed to be `self.block` wide.
        self.newest_width = self.block
        self.anchor_slots: Optional[torch.Tensor] = None
        self.goal_direction: Optional[torch.Tensor] = None

        self.novelty = NoveltyEMA(config.novelty_ema_beta)
        self.importance = ImportanceTracker(config.importance_weights)
        self.trigger = CompactTrigger(config)
        self.compactor = build_compactor(config, self.hidden)
        # Native block width for the selectors. The manager owns the true width
        # (`model.num_tokens`); the config carries a copy so that
        # `build_compactor` can bind it without importing the model.
        self.compactor_block = self.block  # native block width, for the selectors
        self.select_weights = dict(config.select_weights or {})

        self.basis: Optional[torch.Tensor] = None
        self.basis_mean: Optional[torch.Tensor] = None
        self.projection = AnchorProjection.empty(device=memory.device, dtype=memory.dtype)

        self.last_snapshot = CapacitySnapshot(budget_slots=self.layout.working_capacity)
        self.history: List[Dict[str, Any]] = []
        # Generational bookkeeping (v7 C1/C2). `_block_ages[j]` is the age of
        # physical block `j`: the number of compaction rounds its content has
        # survived, so 0 is the young generation. The valid region is the
        # trailing `ceil(used / block)` entries -- validity is derived from
        # `used`, never stored, so no operation has to clear the free region.
        #
        # Diagnostic state, not pool state (D-06): it must never enter
        # `state_dict`, a determinism digest or the checkpoint contract, because
        # the same pool must hash the same with or without age tracking. It *is*
        # carried by `snapshot`/`restore`, so a rolled-back compaction rolls the
        # ages back with the slots (R-07).
        self._block_ages = torch.zeros(
            max(1, self.total // self.block), dtype=torch.long, device=memory.device
        )
        # Append/merge counters. `note_append` bumps `age_append_events`, which
        # lets a caller that does not own the write path (the streaming
        # evaluation layer) settle appends exactly once instead of double
        # counting them.
        self.age_append_events = 0
        self.age_merge_events = 0
        self.compaction_count = 0
        self.evicted_blocks = 0
        self.skipped_chunks = 0
        self.write_count = 0
        self.peak_used = 0
        # True when the most recent write requested more slots than the remaining
        # budget and `_place` had to compact or evict before it could be placed.
        self.made_room_for_write = False
        self.anchor_info: Dict[str, Any] = {}

        # Phase 2: mark the free region inactive for the reader. The pool keeps
        # its fixed width -- only the attention bias changes -- because the
        # rotary offsets and the prefix length both assume a fixed memory
        # length. See `_sync_active_mask`.
        self.active_memory_mask = bool(getattr(config, "active_memory_mask", False))
        # Phase 2 interim fallback: the filler may only repeat uncompressed
        # native blocks, never compactor output. Repeated native tokens still
        # perturb the attention probability mass, so this is a stopgap -- the
        # active mask is what actually removes the free region from the read.
        self.native_filler_only = bool(getattr(config, "native_filler_only", False))
        self._active_mask_used: Optional[int] = None
        self._active_mask_count = 0

        self._carrier: Optional[CarrierResidualCompactor] = None
        self.compactor_errors = 0
        # A trained checkpoint is validated here rather than on the first
        # compaction. `compact_with_report` wraps the operator call in a broad
        # `except` so one bad forward cannot kill a long run, and a load failure
        # raised inside that block is swallowed the same way: the write frees
        # nothing, `_place` falls through to `_evict_oldest`, and the carrier arm
        # degenerates into the eviction arm -- precisely the comparison the
        # three-arm run exists to make. Loading at construction turns a missing
        # file, a foreign state_dict and a shape mismatch into a startup error.
        if getattr(config, "carrier_ckpt", None):
            self._carrier_compactor()

    # -- helpers ------------------------------------------------------------

    @property
    def _memory(self) -> torch.Tensor:
        """The pool storage, addressed without autograd bookkeeping."""
        tensor = self.model.memory
        return tensor.data if hasattr(tensor, "data") else tensor

    def _distinct(self) -> torch.Tensor:
        """[L, used, d] view of the distinct suffix."""
        start, end = self.layout.working_bounds(self.used)
        return self._memory[:, start:end, :]

    # -- generational ages (v7 C1/C2) ---------------------------------------

    @property
    def num_blocks(self) -> int:
        """Physical blocks in the pool."""
        return int(self._block_ages.numel())

    def _valid_age_span(self) -> Tuple[int, int]:
        """Half-open physical block range of the distinct suffix.

        Block `j` covers slots `[j * block, (j + 1) * block)`, and the pool width
        is a whole number of blocks, so `start_slot // block` *is* the physical
        block number. `used` is block-aligned on every streaming path; a partial
        trailing block (a short chunk) is rounded up so its content still counts.
        """
        total_blocks = self.num_blocks
        if self.used <= 0:
            return total_blocks, total_blocks
        first = (self.total - self.used) // self.block
        count = -(-self.used // self.block)  # ceil
        first = max(0, min(first, total_blocks))
        return first, min(total_blocks, first + count)

    def window_block_ages(self, start_slot: int, window_slots: int) -> torch.Tensor:
        """Ages of the `window_slots // block` blocks starting at `start_slot`.

        This is what the block selector consumes: one age per native block of
        the window it is about to compress, in temporal order. Slots outside the
        pool are clamped; a window reaching into the free region yields zeros,
        which the selector reads as "young" rather than as a negative age.
        """
        block = self.block
        first = max(0, int(start_slot) // block)
        count = max(1, int(window_slots) // block)
        ages = self._block_ages
        out = ages[first : first + count]
        if out.numel() < count:
            out = torch.cat(
                (out, torch.zeros(count - out.numel(), dtype=ages.dtype, device=ages.device))
            )
        return out

    def note_append(self, appended: int = 1, dropped: int = 0) -> None:
        """Record `appended` new trailing blocks (and `dropped` evicted ones).

        Appending shifts every resident block one physical block left and lands
        the new content in the last block, so the ages roll left by `appended`
        and the last `appended` entries are reset to 0. That is exactly
        `torch.roll(ages, -appended)` followed by zeroing the tail -- the
        `-1` direction is the physical append direction (`new[j] == old[j + 1]`).

        `dropped` does not move the survivors: eviction only advances the
        validity boundary, which `_valid_age_span` derives from `used`. It is
        carried for auditing and for the callers that reason about it.
        """
        appended = int(appended)
        dropped = int(dropped)
        if appended < 0 or dropped < 0:
            raise ValueError("block age deltas must be non-negative")
        if appended == 0:
            return
        appended = min(appended, self.num_blocks)
        self._block_ages = torch.roll(self._block_ages, -appended)
        self._block_ages[-appended:] = 0
        self.age_append_events += 1

    # -- v9 write-once lock (M4) --------------------------------------------
    #
    # `slot_fused` marks physical slots that already accepted a high-amplitude
    # bridge write. They are read-only for every later compaction, and a block
    # mostly made of them becomes an archive: pinned into the carrier set rather
    # than sampled by the 19->9 selector. `slot_fused` is not a buffer (it is
    # structural state, not a learned tensor) but `snapshot`/`restore` do carry
    # it, so an episode reset or a rolled-back compaction cannot leak a lock
    # from one history into the next.

    def _fused_flags(self) -> torch.Tensor:
        if self.slot_fused is None or self.slot_fused.numel() != self.total:
            self.slot_fused = torch.zeros(self.total, dtype=torch.bool,
                                          device=self._memory.device)
        return self.slot_fused

    def fused_window(self, start: int, window: int):
        """Per-slot freeze mask and archive-pinned blocks of one window."""
        block = self.block
        frozen = self._fused_flags()[start:start + window].clone()
        per_block = frozen.view(-1, block).float().mean(-1)
        pinned = torch.nonzero(per_block >= self.archive_fraction).flatten().tolist()
        return frozen, [int(i) for i in pinned if i < window // block][:self.archive_max_blocks]

    def mark_fused_slots(self, offsets, base: int = 0) -> int:
        """Freeze the candidate slots that just received a bridge write."""
        if not offsets:
            return 0
        flags = self._fused_flags()
        index = torch.as_tensor([base + int(o) for o in offsets],
                                device=flags.device, dtype=torch.long)
        index = index[(index >= 0) & (index < self.total)]
        flags[index] = True
        return int(index.numel())

    def remap_fused(self, start, window, target, new_start, selected, used_before) -> None:
        """Move the freeze flags with the content they belong to.

        Compaction removes `window - target` slots, so everything moves
        right-to-left. The surviving carriers inherit the flags of the blocks
        they were built from, the uncompressed tail keeps its own, and the
        vacated front of the pool is zeroed.
        """
        block = self.block
        flags = self._fused_flags()
        window_flags = flags[start:start + window].clone()
        # The resident region is [start, total), so the tail runs to `total`;
        # slicing to the *length* `used_before` dropped every tail flag.
        tail = flags[start + window:self.total].clone()
        fresh = torch.zeros_like(flags)
        if selected is not None:
            kept = window_flags.view(-1, block)[torch.as_tensor(selected, device=flags.device)]
            fresh[new_start:new_start + target] = kept.reshape(-1)
        if tail.numel():
            fresh[new_start + target:new_start + target + tail.numel()] = tail
        flags.copy_(fresh)

    def note_merge(
        self,
        selected_blocks,
        window_blocks: int,
        target_blocks: Optional[int] = None,
        tail_blocks: Optional[int] = None,
        used_before: Optional[int] = None,
    ) -> None:
        """Settle the ages of a window compaction (carrier age + 1, tail kept).

        Args:
            selected_blocks: window-relative block indices of the surviving
                carriers, in output order (ascending, as the selector returns).
            window_blocks: window width in native blocks.
            target_blocks: output width in native blocks; defaults to
                `len(selected_blocks)`.
            tail_blocks: uncompressed blocks after the window; defaults to the
                resident count implied by `used_before`.
            used_before: `used` before the compaction; defaults to `self.used`.

        A carrier survives one round, so its output block is one round older
        (D-04). Blocks appended after the window are untouched and keep their
        age; everything else in the pool is free space and resets to 0.
        """
        block = self.block
        total_blocks = self.num_blocks
        selected = [int(i) for i in torch.as_tensor(selected_blocks).flatten().tolist()]
        target_blocks = len(selected) if target_blocks is None else int(target_blocks)
        used_before = int(self.used if used_before is None else used_before)
        if tail_blocks is None:
            tail_blocks = max(0, used_before // block - int(window_blocks))
        tail_blocks = int(tail_blocks)
        if target_blocks < 0 or int(window_blocks) < 0 or tail_blocks < 0:
            raise ValueError("block counts must be non-negative")

        used_after = (target_blocks + tail_blocks) * block
        first_old = (self.total - used_before) // block
        first_new = (self.total - used_after) // block
        source = self._block_ages
        out = torch.zeros_like(source)
        for j, index in enumerate(selected[:target_blocks]):
            old = first_old + index
            if 0 <= first_new + j < total_blocks and 0 <= old < total_blocks:
                out[first_new + j] = source[old] + 1
        for t in range(tail_blocks):
            old = first_old + int(window_blocks) + t
            if 0 <= first_new + target_blocks + t < total_blocks and 0 <= old < total_blocks:
                out[first_new + target_blocks + t] = source[old]
        self._block_ages = out
        self.age_merge_events += 1

    def note_block_move(self, block_index: int) -> None:
        """Record that physical block `block_index` moved to the tail.

        `_merge_into_block` closes the gap left by the addressed block and
        rewrites it at the trailing position. The content is the same content
        refreshed in place, so its age is preserved (D-04) and the blocks after
        it shift one position left.
        """
        index = int(block_index)
        if index < 0 or index >= self.num_blocks:
            return
        ages = self._block_ages
        moved = ages[index]
        self._block_ages = torch.cat((ages[:index], ages[index + 1 :], moved.view(1)))

    def reset_block_ages(self) -> None:
        """Forget every age. Used when the pool is reset to its initial state."""
        self._block_ages.zero_()

    def block_age_summary(self) -> Dict[str, Any]:
        """Diagnostic histogram of the resident ages (never part of the hash)."""
        first, end = self._valid_age_span()
        resident = self._block_ages[first:end]
        if resident.numel() == 0:
            return {"block_age_histogram": {}, "block_age_max": 0,
                    "old_generation_slots": 0, "block_age_events": 0}
        ages, counts = torch.unique(resident, return_counts=True)
        histogram = {int(a): int(c) for a, c in zip(ages.tolist(), counts.tolist())}
        old_blocks = int((resident > 0).sum().item())
        return {
            "block_age_histogram": histogram,
            "block_age_max": int(resident.max().item()),
            "old_generation_slots": old_blocks * self.block,
            "block_age_events": int(self.age_append_events + self.age_merge_events),
        }

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        """Return the pool to an unwritten state. Does not re-anchor."""
        with torch.no_grad():
            self._memory.zero_()
        self.model.initialized.zero_()
        self.used = 0
        self.n_consolidated = 0
        self.newest_width = self.block
        self.anchor_slots = None
        self.goal_direction = None
        self.anchor_info = {}
        self.novelty.reset()
        self.trigger.reset()
        self.basis = None
        self.basis_mean = None
        self.projection = AnchorProjection.empty(device=self._memory.device, dtype=self.dtype)
        self.history = []
        self.compaction_count = 0
        self.evicted_blocks = 0
        self.skipped_chunks = 0
        self.write_count = 0
        self.peak_used = 0
        self.made_room_for_write = False
        # `compactor_errors` is deliberately not reset: it counts operator
        # failures for the whole run, and a reset would hide a persistent one.
        self.last_snapshot = CapacitySnapshot(budget_slots=self.layout.working_capacity)
        self.importance = ImportanceTracker(self.cfg.importance_weights)
        self._active_mask_used = None
        self._block_ages.zero_()
        self._sync_active_mask()

    def snapshot(self) -> Dict[str, Any]:
        """Capture mutable state so a candidate compaction can be tried and undone.

        Every field `compact()` touches is captured, not just the pool. Restoring
        only the pool would leave the compaction counter, the basis and the
        history log describing a compaction that was rolled back, which would
        corrupt the run statistics section 5.2 asks to be auditable.
        """
        return {
            "memory": self._memory.clone(),
            "initialized": self.model.initialized.clone(),
            "used": self.used,
            "n_consolidated": self.n_consolidated,
            "newest_width": self.newest_width,
            "compaction_count": self.compaction_count,
            "evicted_blocks": self.evicted_blocks,
            "skipped_chunks": self.skipped_chunks,
            "write_count": self.write_count,
            "history_len": len(self.history),
            "last_snapshot": self.last_snapshot,
            "basis": None if self.basis is None else self.basis.clone(),
            "basis_mean": None if self.basis_mean is None else self.basis_mean.clone(),
            "novelty": self.novelty.value,
            "novelty_count": self.novelty.count,
            "trigger_consecutive": self.trigger.consecutive,
            "trigger_armed": self.trigger.armed,
            "trigger_updates": self.trigger.update_count,
            "block_ages": self._block_ages.clone(),
            "slot_fused": None if self.slot_fused is None else self.slot_fused.clone(),
        }

    def restore(self, state: Dict[str, Any]) -> None:
        with torch.no_grad():
            self._memory.copy_(state["memory"])
            self.model.initialized.copy_(state["initialized"])
        self.used = state["used"]
        self.n_consolidated = state["n_consolidated"]
        self.newest_width = state["newest_width"]
        self.compaction_count = state["compaction_count"]
        self.evicted_blocks = state["evicted_blocks"]
        self.skipped_chunks = state["skipped_chunks"]
        self.write_count = state["write_count"]
        del self.history[state["history_len"]:]
        self.last_snapshot = state["last_snapshot"]
        self.basis = state["basis"]
        self.basis_mean = state["basis_mean"]
        self.novelty.value = state["novelty"]
        self.novelty.count = state["novelty_count"]
        self.trigger.consecutive = state["trigger_consecutive"]
        self.trigger.armed = state["trigger_armed"]
        self.trigger.update_count = state["trigger_updates"]
        # The pool came back from the snapshot, so the ages must too: a rollback
        # that restored the slots but kept the post-compaction ages would
        # describe a generation layout that no longer exists (R-07).
        self._block_ages.copy_(state["block_ages"])
        # The lock is pool state too: a restored pool must not inherit a lock
        # that belongs to a history the rollback discarded (and an episode reset
        # must start unlocked).
        saved_fused = state.get("slot_fused")
        self.slot_fused = None if saved_fused is None else saved_fused.clone()
        # `used` came back from the snapshot, so the published mask describes a
        # rolled-back pool until this runs. The cache key is dropped rather than
        # compared, because the restore may return to a width the mask already
        # held before the candidate compaction was tried.
        self._active_mask_used = None
        self._sync_active_mask()

    # -- anchor -------------------------------------------------------------

    def initialize_anchor(self, text: str, max_tokens: int = 512) -> Dict[str, Any]:
        """Encode the initial goal and hard constraints into the read-only region.

        Called once per episode. `anchor_source=none` skips this, which is the
        no-Anchor ablation section 5.3 requires.

        The Anchor is encoded *before* `initialized` is set, so the forward runs
        its cold path and does not condition the goal on uninitialised memory.
        All later writes run with `initialized=1` and therefore condition on the
        trailing block, matching the published baseline.
        """
        self.reset()
        self.model.initialized.zero_()

        if self.cfg.anchor_source == "none" or not text:
            self.anchor_info = {"anchor_tokens": 0, "anchor_slots": 0, "reason": "disabled"}
            self._sync_active_mask()
            return self.anchor_info

        block = self._encode(text, max_tokens)
        if block is None:
            # Too short to inject safely. The caller still has to write this chunk
            # through the residual path so it is not silently lost.
            self.anchor_info = {"anchor_tokens": 0, "anchor_slots": 0, "anchor_skipped": True}
            self._sync_active_mask()
            return self.anchor_info

        n = block.shape[1]
        target = self.layout.k_anchor
        reps = (target + n - 1) // n
        tiled = block.repeat(1, reps, 1)[:, :target, :].contiguous()

        with torch.no_grad():
            self._memory[:, self.layout.anchor_slice(), :] = tiled.to(self.dtype)

        self.anchor_slots = tiled.detach().clone()
        self.goal_direction = summarise_block(tiled, layer=min(_REFERENCE_LAYER, tiled.shape[0] - 1)).detach()
        self.model.initialized.fill_(1)

        self._tile_free_region()
        # Only decompose the Anchor when a projection will actually consume it.
        # With `projection: none` this was a wasted SVD of [k_anchor, d] whose
        # result build_projection immediately discarded.
        anchor_basis = None
        if self.cfg.projection != "none":
            anchor_basis = self._anchor_basis()
        self.projection = build_projection(
            anchor_basis=anchor_basis,
            lambdas=self.cfg.projection_lambda_vector(self.hidden, self._memory.device, torch.float32),
            mode=self.cfg.projection,
            d=self.hidden,
            device=self._memory.device,
            dtype=self.dtype,
        )

        self.anchor_info = {
            "anchor_tokens": int(n),
            "anchor_slots": target,
            "anchor_repeats": int(reps),
            "projection": self.cfg.projection,
            "anchor_basis_computed": self.cfg.projection != "none",
            "anchor_basis_rank": 0 if anchor_basis is None else int(anchor_basis.shape[1]),
        }
        return self.anchor_info

    def _anchor_basis(self) -> Optional[torch.Tensor]:
        if self.anchor_slots is None:
            return None
        basis, _, _ = estimate_basis(
            self.anchor_slots[0],
            max_rank=min(self.cfg.projection_rank, self.layout.k_anchor),
            energy=self.cfg.basis_energy,
            subtract_mean=True,
        )
        return basis

    def _encode(self, text: str, max_tokens: int) -> Optional[torch.Tensor]:
        """Run one injection forward and return the [L, n, d] delta, pool untouched.

        `update_memory=False` is what keeps the pool pristine: the backbone still
        computes the delta, but the caller decides where it goes.
        """
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) < MIN_INJECT_TOKENS:
            return None
        ids = ids[:max_tokens]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self._memory.device)
        with torch.no_grad():
            delta = self.model.inject_memory(input_ids, update_memory=False)
        # inject_memory returns [batch, L, n, d]; drop the batch axis.
        return delta.detach()[0]

    # -- write path ---------------------------------------------------------

    def write(self, text: str, max_tokens: int = 512) -> Dict[str, Any]:
        """One write: novelty -> allocate-or-address -> gate -> place -> observe.

        Section 3.2 gives the novelty proxy three bands, and only the middle one
        allocates:

            rho low        -> the candidate lies inside the subspace the working
                              memory already spans, so it updates an addressed
                              block and consumes no new slots.
            rho high + room-> allocate new residual slots.
            rho high + full-> allocate and let the capacity trigger consolidate.

        The previous version ignored the first band entirely: every write was
        appended, so `used` tracked the number of writes rather than the amount of
        independent content. That is what produced two anti-correlated curves --
        occupancy rising on every write while novelty fell as the subspace grew --
        and made `tau_k AND tau_n` structurally unreachable.

        Returns a record for the run log. `dropped` is set when the chunk is
        below the 16-token injection floor, in which case the pool is untouched.
        """
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        raw_tokens = len(ids)
        if raw_tokens < MIN_INJECT_TOKENS:
            self.skipped_chunks += 1
            return {"dropped": True, "raw_tokens": raw_tokens, "reason": "below_min_inject"}

        truncated = raw_tokens > max_tokens
        self.made_room_for_write = False
        started = time.time()
        block = self._encode(text, max_tokens)
        encode_seconds = time.time() - started
        if block is None:
            self.skipped_chunks += 1
            return {"dropped": True, "raw_tokens": raw_tokens, "reason": "encode_failed"}

        candidate = summarise_block(block, layer=min(_REFERENCE_LAYER, block.shape[0] - 1))
        # The basis is estimated from centred activations, so the candidate has
        # to be centred with the same mean before it is scored. Without this the
        # novelty carried a constant offset that had nothing to do with the
        # subspace, and the threshold had no stable meaning across episodes.
        scored = candidate - self.basis_mean if self.basis_mean is not None else candidate
        novelty = feature_novelty(scored, self.basis, eps=self.cfg.novelty_eps)
        novelty_value = self.novelty.update(novelty)

        delta = block
        gate_value = 1.0
        if self.cfg.mode != "full_rewrite":
            delta = apply_projection(delta, self.projection)
            gate = self._gate(novelty)
            gate_value = float(gate)
            delta = delta * gate

        # Both routes into an existing block require an admission decision: the
        # candidate must be one the current memory can already explain, and the
        # block it would join must actually match it. Novelty alone only says the
        # representation is inside the spanned subspace; the address check says
        # *which* stored content it is a correction of, which is the distinction
        # the note draws between a fact update and a new fact.
        addressed = self.cfg.write_mode == "additive"
        allocated = True
        target = None
        if addressed:
            target, similarity = self._address(candidate)
            # The *instantaneous* rho, not the EMA: the note defines the
            # allocation band on this write's candidate. Feeding the EMA here
            # would let one unusual write decide the placement of the next
            # several, and would silently couple `tau_alloc` to
            # `novelty_ema_beta`. The EMA is a trigger signal only.
            explained = float(novelty) < self.cfg.tau_alloc
            matched = similarity >= self.cfg.address_threshold
            if target is not None and (explained or matched):
                allocated = False

        # Capacity is decided *before* placement, not after it.
        #
        # The previous ordering ran `_place` first and asked the trigger
        # afterwards. `_place` frees room itself whenever the block would not
        # fit, and a write can only exceed capacity by needing room, so by the
        # time the trigger looked at the snapshot the over-capacity condition
        # had always been resolved. Every compaction in the smoke run was
        # therefore `reason: headroom` and the configured trigger fired 0 times
        # in 82 writes -- it was unreachable by construction, not by tuning.
        #
        # Asking first gives the trigger the state it is defined on (the pool
        # about to be over capacity) and demotes `_place`'s own compaction to the
        # fallback it is meant to be: only reached if the trigger chose not to act
        # and the block still does not fit.
        requested = int(block.shape[1]) if allocated else 0
        pre = self._snapshot(
            allocated_new_slots=requested,
            # The note's immediate branch: the write would exceed the working
            # capacity, so it must compact now rather than wait for a threshold.
            # An addressed update consumes nothing and can never be over budget.
            over_budget=bool(
                requested > 0
                and self.used + requested > self.layout.working_capacity
            ),
        )
        compacted, report, trigger_reason = self._maybe_compact(pre)

        evicted = self._place(delta, allocated=allocated, target=target)
        # Tracked here rather than derived from `history`, because the drains
        # that feed the stats file empty that buffer and would make the peak read
        # as whatever happened to survive the last drain.
        self.peak_used = max(self.peak_used, int(self.used))
        # The pool is only ever read through this entry point, and `initialized`
        # must be 1 for injection to condition on the trailing block.
        self.model.initialized.fill_(1)

        self._refresh_basis()
        # Re-observed after placement: the record has to describe the state the
        # next write will see, which is the post-placement pool. The spectrum is
        # suppressed here because this snapshot drives no decision; it only fills
        # the reported rank fields, which the pre-placement snapshot already set
        # from the same pool minus one block.
        self._snapshot(
            allocated_new_slots=requested, over_budget=False, carry=pre
        )

        record = {
            "dropped": False,
            "raw_tokens": raw_tokens,
            "tokens": int(block.shape[1]),
            "truncated": truncated,
            "novelty": float(novelty_value),
            "gate": gate_value,
            "encode_seconds": encode_seconds,
            "allocated": allocated,
            "addressed_block": target,
            "made_room_for_write": bool(self.made_room_for_write),
            "over_budget_request": bool(pre.over_budget),
            "trigger_reason": trigger_reason,
            "used_slots": self.used,
            "n_consolidated": self.n_consolidated,
            "evicted_blocks": evicted,
            "compacted": compacted,
            **self.last_snapshot.to_dict(),
        }
        if report is not None:
            record["compaction"] = report
        self.history.append(record)
        self.write_count += 1
        return record

    def drain_history(self) -> List[Dict[str, Any]]:
        """Return the pending per-write records and clear the buffer.

        The calibration procedure in section 5 reads the novelty, occupancy and
        rank signals of *every* write, so they have to reach the stats file
        rather than accumulating only in memory until the process exits. Draining
        keeps each stats line small and bounds memory on a long episode.
        """
        drained = self.history
        self.history = []
        return drained

    def _gate(self, novelty: torch.Tensor) -> torch.Tensor:
        """Scalar write gate in [0, 1], broadcast over slots and features.

        `novelty` is used directly as g_t in section 3.2: a candidate already
        inside the working subspace is written more softly. This is the only thing
        the novelty proxy is allowed to do; per risk 2 it never decides whether a
        fact correction is accepted.
        """
        if self.cfg.mode in ("full_rewrite", "anchor_rewrite"):
            return torch.ones((), device=self._memory.device, dtype=torch.float32)
        if self.cfg.gate_mode == "ones":
            return torch.ones((), device=self._memory.device, dtype=torch.float32)
        if self.cfg.gate_mode == "novelty":
            value = novelty.reshape(-1).mean() if isinstance(novelty, torch.Tensor) else novelty
            return torch.as_tensor(value, device=self._memory.device, dtype=torch.float32).clamp(0.0, 1.0)
        if self.cfg.gate_mode == "learned":
            raise NotImplementedError(
                "gate_mode=learned needs a trained writer; the note makes the "
                "geometric baseline the first version (section 3.4)"
            )
        raise ValueError(f"unknown gate_mode: {self.cfg.gate_mode}")

    def _place(self, delta: torch.Tensor, allocated: bool = True, target: Optional[int] = None) -> int:
        """Place one block, returning the number of blocks lost to eviction.

        Three write regimes, matching the control arms:

        * `full_rewrite` / `anchor_rewrite` (B5, B6) discard the previous working
          state and keep only the current block. That is what "rewrite the whole
          state every step" means, and it is the baseline PRLM-C must beat.
        * `write_mode: append` (the ablation) always allocates a new block.
        * `write_mode: additive` (the design) allocates when the candidate is
          novel and otherwise merges it into an addressed block, which consumes
          no new slots.

        Appending shifts existing content left by one block so the new block lands
        at the very end; injection conditions on the trailing block, and the
        backbone's own update appends there too. An addressed update is moved to
        the same trailing position for the same reason, and the addressed block's
        old slot range is closed up, so `used` is unchanged.
        """
        if self.cfg.mode in ("full_rewrite", "anchor_rewrite"):
            need = delta.shape[1]
            with torch.no_grad():
                self._memory[:, self.total - need : self.total, :] = delta.to(self.dtype)
            self.used = need
            self.newest_width = need
            self.n_consolidated = 0
            # These arms discard the previous working state by definition, so
            # nothing resident is old: the pool is one fresh block.
            self.reset_block_ages()
            self._tile_free_region()
            # Nothing is counted as evicted: these arms discard the previous
            # working state by definition, so a per-write eviction count would
            # report the arm's whole premise as a failure of compaction.
            return 0

        if not allocated:
            if target is None:
                raise ValueError("an addressed write requires a target block")
            self._merge_into_block(target, delta)
            return 0

        need = delta.shape[1]
        if need <= 0:
            return 0
        if need > self.layout.working_capacity:
            raise ValueError(
                "a single block of %d slots does not fit the working capacity of %d"
                % (need, self.layout.working_capacity)
            )

        evicted = 0
        while self.used + need > self.layout.working_capacity:
            freed = self.compact(reason="headroom")
            if freed <= 0:
                self._evict_oldest(need)
                evicted += 1
            # A compaction happened, so the hysteresis latch must be cleared here
            # too. The trigger is asked before placement, so a run that reaches
            # this loop is one the trigger declined to act on; leaving `armed` set
            # would carry an armed latch across a compaction the trigger did not
            # authorize, and the exit thresholds would then apply to a pool the
            # latch no longer describes.
            self.trigger.on_compacted()
            # Recorded so a run can be audited for *why* the fallback ran: it
            # only sets when placement had to free room after the trigger had
            # already been asked and declined.
            self.made_room_for_write = True

        # Ages are settled against the state the append starts from: the loop
        # above may already have compacted (which settles its own carriers) or
        # evicted (which only moves the validity boundary).
        before_append = self.used
        self._shift_left(need)
        with torch.no_grad():
            self._memory[:, self.total - need : self.total, :] = delta.to(self.dtype)
        self.used += need
        self.newest_width = need
        # Ages follow the same physical move the slots just made. `_shift_left`
        # and `_evict_oldest` deliberately do not touch them: the append is the
        # only operation that shifts content, and settling it here counts it
        # exactly once even when the loop above compacted or evicted first.
        appended = self.used // self.block - before_append // self.block
        if appended:
            self.note_append(appended=appended, dropped=evicted)
        # The append path moves the active/free boundary, so the mask has to
        # follow it. `_tile_free_region` is not called here -- appending leaves
        # the existing filler in place -- so the sync is explicit.
        self._sync_active_mask()
        return evicted

    def _merge_into_block(self, target: int, delta: torch.Tensor) -> None:
        """Fold `delta` into block `target` and move the result to the tail.

        `used` is unchanged: this write consumed no new capacity. The merged
        block lands in the trailing slot range because injection conditions on
        the trailing block, and a merge that left the newest content in the
        middle would make the next injection read a stale block. Closing the gap
        left by the old position is what keeps the distinct suffix contiguous.
        """
        used = self.used
        block = self.block
        width = min(block, used - target * block)
        if width <= 0:
            return

        start = self.total - used
        position = start + target * block

        with torch.no_grad():
            existing = self._memory[:, position : position + width, :].detach().to(torch.float32)
            addition = fit_width(delta, width).detach().to(torch.float32)
            merged = (existing + addition).to(self.dtype)

            # Close the gap: [position + width, end) moves left by `width`. Done
            # front-to-back in bounded chunks because the regions overlap.
            end = self.total
            done = 0
            while position + width + done < end:
                take = min(_COPY_CHUNK, end - position - width - done)
                chunk = self._memory[
                    :, position + width + done : position + width + done + take, :
                ].clone()
                self._memory[:, position + done : position + done + take, :] = chunk
                done += take

            self._memory[:, self.total - width : self.total, :] = merged

        if position < start + self.n_consolidated:
            # The block was part of the consolidated region and has just been
            # rewritten, so its content is no longer verified-stable: it leaves
            # S and rejoins R at the tail.
            self.n_consolidated = max(0, self.n_consolidated - width)
        self.newest_width = width
        # The merged block moved to the tail and everything after it shifted one
        # block left. Its age travels with it and does not advance: this is the
        # same content refreshed in place, not a new generation (D-04).
        self.note_block_move(position // block)

    def _shift_left(self, need: int) -> None:
        """Move the distinct suffix left by `need` slots, preserving order.

        Done in front-to-back chunks because the destination overlaps the source.
        A single whole-region copy would be an overlapping device memcpy, whose
        behaviour on CUDA is not defined, and a whole-region clone would cost
        several gigabytes at the default pool width.
        """
        if need <= 0 or self.used <= 0:
            return
        start = self.total - self.used
        with torch.no_grad():
            done = 0
            while done < self.used:
                width = min(_COPY_CHUNK, self.used - done)
                source = self._memory[:, start + done : start + done + width, :].clone()
                self._memory[:, start - need + done : start - need + done + width, :] = source
                done += width

    def _evict_oldest(self, n: int) -> None:
        """Drop the n oldest working slots.

        The distinct suffix keeps its trailing anchor, so removing the oldest
        slots needs no move: the survivors are already at [total - used + n,
        total). Only the count and the consolidated prefix change.
        """
        n = min(n, self.used)
        if n <= 0:
            return
        self.used -= n
        self.n_consolidated = max(0, self.n_consolidated - n)
        self.evicted_blocks += 1
        self._tile_free_region()

    def _sync_active_mask(self) -> None:
        """Publish the active-slot mask the reader should attend to.

        Phase 2. The pool keeps its fixed width; only the attention bias changes.
        Active slots are the Anchor prefix [0, k_anchor) and the distinct suffix
        [total - used, total). Everything between them -- the free region -- is
        marked inactive, so the filler that `_tile_free_region` leaves there is
        never read as evidence.

        Cached on `used`: the mask is a pure function of it, so a write that does
        not move the boundary costs nothing. A run with `active_memory_mask:
        false`, or one whose modeling fork is unpatched, leaves the model's
        all-true buffer in place and therefore reads the pool exactly as before.
        """
        if not self.active_memory_mask:
            return
        buffer = getattr(self.model, "memory_active_mask", None)
        if buffer is None:
            return
        if self._active_mask_used == self.used:
            return

        total = self.layout.total
        k_anchor = self.layout.k_anchor
        mask = torch.zeros(total, dtype=torch.bool, device=buffer.device)
        mask[:k_anchor] = True
        if self.used > 0:
            mask[total - self.used :] = True
        buffer.copy_(mask)
        self._active_mask_used = self.used
        self._active_mask_count += 1

    def _tile_free_region(self) -> None:
        """Fill [k_anchor, total - used) with tiled copies of occupied content.

        The forward pass reads the whole pool, so free space must not be left as
        zeros: the backbone's own first-write path tiles a short delta across the
        entire pool, which makes repetition in-distribution. Only `used` counts
        against the budget; see section 3.4 on not equating physical width with
        addressable tokens.

        Written in source-width chunks so no allocation larger than the source is
        ever materialised.
        """
        self._sync_active_mask()
        free_start, free_end = self.layout.free_bounds(self.used)
        width = free_end - free_start
        if width <= 0:
            return

        if self.used > 0:
            source = self._distinct()
            if self.native_filler_only:
                # Phase 2 interim fallback. The consolidator writes synthetic
                # tokens at the *front* of the distinct suffix, so restricting
                # the filler to the residual tail keeps compactor output out of
                # the free region. When nothing native is left the Anchor is the
                # only uncompressed source; if that is gone too the existing
                # filler is left untouched rather than overwritten with synthetic
                # tokens. Repeated native tokens still change the attention
                # probability mass, so this is a stopgap only -- the active mask
                # is what actually removes the free region from the read.
                start, end = self.layout.working_bounds(self.used)
                native = self._memory[:, start + self.n_consolidated : end, :]
                if native.shape[1] > 0:
                    source = native
                elif self.anchor_slots is not None:
                    source = self.anchor_slots
                else:
                    return
        elif self.anchor_slots is not None:
            source = self.anchor_slots
        else:
            return

        source_width = source.shape[1]
        if source_width <= 0:
            return

        with torch.no_grad():
            written = 0
            while written < width:
                take = min(source_width, width - written)
                self._memory[:, free_start + written : free_start + written + take, :] = source[
                    :, :take, :
                ]
                written += take

    # -- capacity -----------------------------------------------------------

    def _refresh_basis(self) -> None:
        """Re-estimate the working basis U_t from current memory activations.

        The note is explicit that this must run on memory activations, never on
        an orthonormal basis, whose singular values are all 1 (section 3.3).
        """
        if self.used < self.block:
            self.basis = None
            self.basis_mean = None
            return
        activations = self._distinct()[self._reference_layer_index()]
        basis, mean, _ = estimate_basis(
            activations,
            max_rank=min(self.cfg.basis_max_rank, self.used),
            energy=self.cfg.basis_energy,
            subtract_mean=True,
            max_rows=_BASIS_MAX_ROWS,
        )
        self.basis = basis
        self.basis_mean = mean

    def _reference_layer_index(self) -> int:
        """Semantic reference layer, clamped for readers with fewer layers."""
        return min(_REFERENCE_LAYER, self._memory.shape[0] - 1)

    def _address(self, candidate: torch.Tensor) -> Tuple[Optional[int], float]:
        """Return the existing block whose content is closest to `candidate`.

        The note's write path needs a way to say *which* stored content a new
        candidate is a correction of, because "inside the spanned subspace" only
        says the candidate is not new, not what it updates. Blocks are the
        addressing unit: a block is what one injection produced and what the
        reader consumes, so a slot-level address would be finer than anything the
        memory can act on.

        Returns `(block_index, cosine)`, or `(None, 0.0)` when nothing has been
        written yet or the pool holds no whole block.
        """
        used = self.used
        n_blocks = used // self.block
        if n_blocks <= 0:
            return None, 0.0

        reference = candidate.detach().to(torch.float32)
        reference = reference / (reference.norm() + 1e-8)

        with torch.no_grad():
            region = self._distinct()[self._reference_layer_index()][: n_blocks * self.block]
            summaries = region.detach().to(torch.float32).reshape(n_blocks, self.block, -1)
            summaries = summaries.mean(dim=1)
            summaries = summaries / (summaries.norm(dim=-1, keepdim=True) + 1e-8)
            similarity = summaries @ reference

        best = int(torch.argmax(similarity).item())
        return best, float(similarity[best].item())

    def _snapshot(
        self,
        over_budget: bool = False,
        allocated_new_slots: int = 0,
        carry: Optional[CapacitySnapshot] = None,
    ) -> CapacitySnapshot:
        """Observe the capacity signals and store them as `last_snapshot`.

        Split out from the trigger decision so `write()` can ask the trigger
        *before* placing and still record the post-placement state afterwards.
        The two orderings need the same signal computation against different pool
        states, and duplicating the computation is what would let them drift.

        The rank spectrum is only computed when `occupancy_novelty_rank` is the
        trigger and the occupancy condition already holds: a spectrum is never
        needed to decide *not* to compact, and computing it on every write is what
        would make this method dominate the run time. `carry` passes a previously
        computed snapshot whose rank fields should be reused, which is how the
        post-placement record avoids paying for the same SVD a second time.
        """
        occupancy = occupancy_ratio(self.used, self.layout.working_capacity)
        eff = 0.0
        eta = 0
        rank_budget = 0
        needs_spectrum = (
            carry is None
            and self.cfg.trigger == "occupancy_novelty_rank"
            and occupancy >= self.cfg.tau_k
        )
        if carry is not None:
            # Reuse the rank fields the pre-placement snapshot already measured.
            # The pool moved by one block, which is far below the resolution of a
            # rank estimate over thousands of rows, and a second SVD per write is
            # the single most expensive thing this path could do needlessly.
            eff = carry.effective_rank
            eta = carry.energy_rank
            rank_budget = carry.rank_max
        elif needs_spectrum:
            sigma = self._spectrum()
            if sigma is not None:
                eff = effective_rank(sigma)
                eta = energy_rank(sigma, self.cfg.rank_eta)
                rank_budget = min(self.cfg.rank_max, max(1, self.used))

        snapshot = CapacitySnapshot(
            used_slots=self.used,
            budget_slots=self.layout.working_capacity,
            occupancy=occupancy,
            effective_rank=eff,
            energy_rank=eta,
            rank_max=rank_budget,
            novelty_ema=float(self.novelty.value or 0.0),
            allocated_new_slots=allocated_new_slots,
            over_budget=bool(over_budget),
        )
        self.last_snapshot = snapshot
        return snapshot

    def _maybe_compact(self, snapshot: CapacitySnapshot):
        """Feed one snapshot to the trigger and compact if it says so.

        Returns `(compacted, report, reason)`. `reason` is `None` when nothing
        happened, otherwise the reason string recorded on the compaction report,
        so a run can be audited for *which* branch fired rather than only whether
        the pool shrank.
        """
        if self.cfg.mode in ("full_rewrite", "anchor_rewrite"):
            return False, None, None
        if self.cfg.compact_mode == "none":
            return False, None, None

        if not self.trigger.observe(snapshot):
            return False, None, None

        reason = "over_budget" if snapshot.over_budget else "capacity_trigger"
        report = self.compact_with_report(reason=reason)
        self.trigger.on_compacted()
        return report["freed_slots"] > 0, report, reason

    def _spectrum(self, max_rows: int = _SPECTRUM_MAX_ROWS) -> Optional[torch.Tensor]:
        if self.used < self.block:
            return None
        activations = self._distinct()[self._reference_layer_index()].detach().to(torch.float32)
        activations = self._subsample(activations, max_rows)
        activations = activations - activations.mean(dim=0, keepdim=True)
        try:
            return torch.linalg.svdvals(activations)
        except Exception:
            return None

    @staticmethod
    def _subsample(rows: torch.Tensor, max_rows: int) -> torch.Tensor:
        """Evenly spaced row subsample, preserving order. No-op below the cap."""
        n = rows.shape[0]
        if max_rows <= 0 or n <= max_rows:
            return rows
        step = n / float(max_rows)
        index = torch.arange(max_rows, device=rows.device, dtype=torch.float64) * step
        return rows[index.long().clamp_(0, n - 1)]

    # -- consolidation ------------------------------------------------------

    def compact(self, reason: str = "manual", ratio: Optional[float] = None) -> int:
        """Consolidate the oldest working window and return the slots freed."""
        return self.compact_with_report(reason=reason, ratio=ratio)["freed_slots"]

    def compact_with_report(self, reason: str = "manual", ratio: Optional[float] = None) -> Dict[str, Any]:
        """Consolidate the oldest working window.

        The newest block is never inside the window: injection conditions on the
        trailing block, and merging it would change what the next injection sees.
        The Anchor region is never touched at all.

        The compacted result stays at the front of the distinct suffix, so the
        A | S | R ordering of section 3.1 is preserved across repeated writes.
        """
        empty = {"freed_slots": 0, "reason_skipped": "disabled"}
        if self.cfg.compact_mode == "none":
            return {**empty, "reason_skipped": "compact_mode=none"}
        if self.cfg.mode in ("full_rewrite", "anchor_rewrite"):
            return {**empty, "reason_skipped": "no_working_memory"}
        if self.compactor is None:
            return {**empty, "reason_skipped": "no_compactor"}

        ratio = self.cfg.compact_ratio if ratio is None else ratio
        block = self.block
        used = self.used
        # The newest block is never inside the window: injection conditions on the
        # trailing block, and merging it would change what the next injection
        # sees. Its width is the width actually appended, not necessarily `block`.
        protect = max(1, self.newest_width)
        if used < protect + block:
            return {**empty, "reason_skipped": "below_min_compact_size"}

        window = min(used - protect, max(self.cfg.compact_min_slots, used // 2))
        window = max(block, (window // block) * block)
        # Clamp after rounding: window must never reach into the protected block.
        window = min(window, used - protect)
        target = max(block, int(window * ratio))
        target = max(block, (target // block) * block)
        if target >= window:
            return {**empty, "reason_skipped": "target_not_smaller"}

        # The native block selectors emit whole blocks only. `window` and `target`
        # are already rounded to `block` above, but the final clamp against
        # `used - protect` can unround `window` when the newest appended block is
        # narrower than `block`. Refuse explicitly rather than letting the operator
        # raise inside the try: a skipped compaction is visible in the report, an
        # exception is swallowed into `reason_skipped` and loses the reason.
        if self.cfg.compact_mode in ("recent_blocks", "select_blocks", "carrier_residual"):
            if window % block != 0 or target % block != 0:
                return {
                    **empty,
                    "reason_skipped": "block_misaligned:%d/%d" % (window % block, target % block),
                }

        start = self.total - used
        source = self._memory[:, start : start + window, :].detach().to(torch.float32)
        weights = self._window_importance(start, window)
        # Generational ages of exactly this window, in temporal order. They feed
        # the selector's quota and are settled below from the blocks it kept.
        ages = self.window_block_ages(start, window)
        selected_out: Dict[str, Any] = {}

        try:
            compacted = self._run_compactor(
                source, weights, target, block_ages=ages, selected_out=selected_out
            )
        except Exception as error:  # noqa: BLE001 - a failed compaction must not kill the run
            # Counted as well as reported: a compaction that raised freed zero
            # slots, so `_place` evicts instead, and a run whose operator fails
            # every time is an eviction run wearing a carrier label. The count
            # reaches `stats()` and therefore every stats line.
            self.compactor_errors += 1
            return {**empty, "reason_skipped": f"compactor_error: {error!r}"}

        if compacted.shape[1] != target:
            return {
                **empty,
                "reason_skipped": "compactor_width_mismatch:%d!=%d" % (compacted.shape[1], target),
            }


        remainder = self._memory[:, start + window : self.total, :].clone()
        # The merged content must land flush against the trailing end, otherwise
        # the "distinct suffix is the trailing `used`" invariant breaks and the
        # injection prefix would read filler instead of the newest block. The
        # destination is `new_start`, not `start`: compaction removed
        # window - target slots, so everything moves right-to-left by that much.
        new_used = used - window + target
        new_start = self.total - new_used
        with torch.no_grad():
            self._memory[:, new_start : new_start + target, :] = compacted.to(self.dtype)
            if remainder.shape[1] > 0:
                self._memory[
                    :, new_start + target : new_start + target + remainder.shape[1], :
                ] = remainder
        self.used = new_used
        self.n_consolidated = target
        self.compaction_count += 1
        # Surviving carriers are one generation older; the uncompressed tail
        # keeps its age; the rest of the pool resets. Skipped when the operator
        # cannot name its carriers (the SVD baselines): synthesised output has no
        # provenance, so there is no age to inherit.
        selected = selected_out.get("index")
        if selected is not None:
            self.note_merge(
                selected,
                window // block,
                target_blocks=target // block,
                used_before=used,
            )
        self._tile_free_region()
        self._refresh_basis()
        # Surviving slots gain a reuse credit; this is the only reuse signal the
        # pool has, since nothing addresses individual slots.
        self.importance.mark_reuse(
            torch.arange(self.used, device=self._memory.device), self.used
        )

        age_stats = self.block_age_summary()
        report = {
            "freed_slots": window - target,
            "reason": reason,
            "window_slots": window,
            "target_slots": target,
            "used_before": used,
            "used_after": self.used,
            "shots": self.compaction_count,
            "weighted": weights is not None,
            "operator": self.cfg.compact_mode,
            # Generational diagnostics (C2): what survived and how old it is.
            "selected_blocks": None if selected is None else selected.tolist(),
            "block_age_max": age_stats["block_age_max"],
            "block_age_histogram": age_stats["block_age_histogram"],
        }
        self.history.append({"event": "compact", **report})
        return report

    def _run_compactor(
        self,
        source: torch.Tensor,
        weights,
        target: int,
        block_ages: Optional[torch.Tensor] = None,
        selected_out: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Invoke the configured compaction operator.

        The block selectors need two things the geometric operators do not: the
        native block width, and the goal direction that `select_block_compact`
        scores blocks against. Both are threaded in here rather than being added
        to the shared `(source, weights, target)` signature, so the SVD operators
        keep their existing interface and stay untouched.

        `block_ages` (one age per window block) drives the generational quota,
        and `selected_out["index"]` receives the window-relative indices of the
        blocks that survived, so the caller can settle the output ages without
        scoring the window a second time. The SVD operators leave it unset.
        """
        mode = self.cfg.compact_mode
        if mode == "select_blocks":
            index = _select_block_indices(
                source,
                target,
                block=self.compactor_block,
                weights=weights,
                goal_direction=self.goal_direction,
                select_weights=self.select_weights,
                block_ages=block_ages,
            )
            if selected_out is not None:
                selected_out["index"] = index
            return select_block_compact(
                source,
                target,
                block=self.compactor_block,
                weights=weights,
                goal_direction=self.goal_direction,
                select_weights=self.select_weights,
                block_ages=block_ages,
                index=index,
            )
        if mode == "recent_blocks":
            blocks = source.shape[1] // self.compactor_block
            kept = target // self.compactor_block
            if selected_out is not None and 0 < kept < blocks:
                selected_out["index"] = torch.arange(
                    blocks - kept, blocks, device=source.device, dtype=torch.long
                )
            return recent_block_compact(source, target, block=self.compactor_block)
        if mode == "carrier_residual":
            compactor = self._carrier_compactor()
            output = compactor(
                source,
                weights,
                target=target,
                block=self.compactor_block,
                goal_direction=self.goal_direction,
                select_weights=self.select_weights,
                block_ages=block_ages,
            )
            if selected_out is not None:
                selected_out["index"] = compactor.last_carrier_indices
            return output
        return self.compactor(source, weights, target)

    def _carrier_compactor(self):
        """Build the phase-3 compactor, then reuse it.

        The module is built on first use so that configs which never select
        `carrier_residual` do not pay for it. When a checkpoint is configured the
        first use is `__init__`, deliberately: see the comment there. The module
        lives on the model's device and dtype-agnostic in fp32; the forward casts
        back to the pool dtype at the end.
        """
        existing = self._carrier
        if existing is not None:
            return existing

        basis_mode = getattr(self.cfg, "basis_mode", "none")
        ckpt = getattr(self.cfg, "basis_ckpt", None) or getattr(self.cfg, "carrier_ckpt", None)
        if basis_mode == "fixed_qr":
            if not ckpt:
                raise RuntimeError(
                    "basis_mode=fixed_qr requires basis_ckpt or carrier_ckpt "
                    "from the shared train warm-up"
                )
            warmup = CarrierResidualCompactor(
                num_layers=self.num_layers,
                hidden=self.hidden,
                block=self.compactor_block,
                rank=int(getattr(self.cfg, "carrier_rank", 32)),
                gamma_max=float(getattr(self.cfg, "carrier_gamma_max", 0.1)),
            ).to(device=self._memory.device, dtype=torch.float32)
            self._load_carrier_state(warmup, ckpt)
            carrier = FixedBasisCarrierResidualCompactor.from_warmup(warmup)
            carrier.trained = True
            carrier.basis_source = ckpt
        else:
            carrier = CarrierResidualCompactor(
                num_layers=self.num_layers,
                hidden=self.hidden,
                block=self.compactor_block,
                rank=int(getattr(self.cfg, "carrier_rank", 32)),
                gamma_max=float(getattr(self.cfg, "carrier_gamma_max", 0.1)),
            ).to(device=self._memory.device, dtype=torch.float32)
            if ckpt:
                self._load_carrier_state(carrier, ckpt)
                carrier.trained = True  # noqa: attribute set for parity with LearnedCompactor
            else:
                # No checkpoint: this is the identity-at-init control.
                print(
                    "[prlmc] carrier_residual: no carrier_ckpt, running at "
                    "initialisation (equivalent to select_blocks)"
                )

        carrier.eval()
        self._carrier = carrier
        return carrier

    def _load_carrier_state(self, carrier: "CarrierResidualCompactor", ckpt: str) -> None:
        """Load a carrier checkpoint or raise a message that names the file.

        Every failure mode here is a configuration error, not a runtime one, and
        the caller wraps the operator call in a broad `except`. Raising is
        therefore the only way these reach a reader: left to `compact_with_report`
        they become a `compactor_error` string in a per-write record, the write
        frees nothing, and the arm quietly evicts instead.
        """
        try:
            raw = torch.load(ckpt, map_location=self._memory.device)
        except Exception as error:
            raise RuntimeError(
                "carrier_ckpt %r could not be read (%s: %s); the carrier arm would "
                "otherwise evict instead of compact" % (ckpt, type(error).__name__, error)
            ) from error

        state = raw
        if isinstance(raw, dict) and "state_dict" in raw:
            state = raw["state_dict"]
        if not isinstance(state, dict):
            raise RuntimeError(
                "carrier_ckpt %r holds %s, not a state_dict; expected the mapping "
                "written by the training script under a 'state_dict' key"
                % (ckpt, type(raw).__name__)
            )

        target = carrier.state_dict()
        missing = sorted(set(target) - set(state))
        unexpected = sorted(set(state) - set(target))
        if missing or unexpected:
            raise RuntimeError(
                "carrier_ckpt %r does not match this carrier: missing %s, unexpected %s. "
                "A checkpoint from a different rank or block width cannot be fused."
                % (ckpt, missing or "none", unexpected or "none")
            )
        for name, value in state.items():
            if not torch.is_tensor(value):
                raise RuntimeError(
                    "carrier_ckpt %r entry %r is %s, not a tensor"
                    % (ckpt, name, type(value).__name__)
                )
            if tuple(value.shape) != tuple(target[name].shape):
                raise RuntimeError(
                    "carrier_ckpt %r entry %r has shape %s, expected %s; check "
                    "carrier_rank and the training template"
                    % (ckpt, name, tuple(value.shape), tuple(target[name].shape))
                )

        carrier.load_state_dict(state, strict=True)
        # `gamma_max` is a plain attribute, not a parameter, so it is absent from
        # the state_dict the trainer writes and cannot be restored from the file.
        # The effective gate is `gamma_max * sigmoid(gamma_logit)`, so a YAML
        # value other than the training one rescales every learned residual
        # silently. It is printed instead, which is all that is verifiable: the
        # file does not record the value it was trained under.
        print(
            "[prlmc] carrier_residual: gamma_max %.6g x sigmoid(%.6g) = effective gamma %.6g"
            % (
                carrier.gamma_max,
                float(carrier.gamma_logit.detach()),
                float(carrier.gamma.detach()),
            )
        )
        print(f"[prlmc] carrier_residual: loaded {ckpt}")

    def _window_importance(self, start: int, window: int) -> Optional[torch.Tensor]:
        """Importance weights for the sliding window, or None when unused.

        Consumed by the importance-weighted SVD baseline and by the block
        selector. `recent_blocks` deliberately does not ask for them: it must stay
        a pure-recency control, and a control that reads importance is not one.
        """
        if self.cfg.compact_mode not in (
            "weighted_svd_tokens",
            "select_blocks",
            # The carrier picker is the same scorer as `select_blocks`, so it
            # needs the same importance weights. Without them it would fall back
            # to the dispersion stand-in and the two operators would no longer
            # share a carrier set.
            "carrier_residual",
        ):
            return None

        # Same layer the carrier selector scores blocks in and the same layer
        # `goal_direction` was summarised from, so the importance weights and the
        # selector never disagree about which layer defines a block.
        assert _SELECT_LAYER == _REFERENCE_LAYER
        slots = self._memory[self._reference_layer_index(), start : start + window, :]
        recency = torch.arange(window, device=slots.device, dtype=torch.float32)
        region_mean = slots.detach().to(torch.float32).mean(dim=0, keepdim=True)
        surprise = (slots.detach().to(torch.float32) - region_mean).norm(dim=-1)
        return self.importance.score(
            slots,
            goal_direction=self.goal_direction,
            recency=recency,
            surprise=surprise,
        )

    # -- diagnostics --------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "mode": self.cfg.mode,
            "mode_code": self.cfg.mode_code,
            "layout": self.layout.to_dict(),
            "used_slots": self.used,
            "n_consolidated": self.n_consolidated,
            "residual_slots": max(0, self.used - self.n_consolidated),
            "compaction_count": self.compaction_count,
            "compactor_errors": self.compactor_errors,
            "evicted_blocks": self.evicted_blocks,
            "skipped_chunks": self.skipped_chunks,
            "writes": self.write_count,
            "peak_used": self.peak_used,
            **self.last_snapshot.to_dict(),
            "novelty_ema": float(self.novelty.value or 0.0),
            "basis_rank": 0 if self.basis is None else int(self.basis.shape[1]),
            "projection": self.cfg.projection,
            "write_mode": self.cfg.write_mode,
            "compact_mode": self.cfg.compact_mode,
            "active_memory_mask": self.active_memory_mask,
            "active_mask_slots": int(self.layout.k_anchor + self.used)
            if self.active_memory_mask
            else int(self.layout.total),
            "active_mask_updates": self._active_mask_count,
            "native_filler_only": self.native_filler_only,
            "anchor": self.anchor_info,
            # Generational diagnostics (v7 C1). Reported, never hashed: the same
            # pool must produce the same digest with or without age tracking.
            **self.block_age_summary(),
        }

    def memory_nbytes(self) -> int:
        tensor = self.model.memory
        return int(tensor.numel() * tensor.element_size())

    def auxiliary_nbytes(self) -> int:
        """Bytes held by basis, statistics, and compaction scratch."""
        total = 0
        for tensor in (self.basis, self.basis_mean, self.anchor_slots, self.projection):
            if tensor is not None and tensor.numel() > 0:
                total += int(tensor.numel() * tensor.element_size())
        if self.importance._reuse is not None:
            total += int(self.importance._reuse.numel() * self.importance._reuse.element_size())
        return total
