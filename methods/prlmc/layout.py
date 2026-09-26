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
from methods.prlmc.consolidate import build_compactor
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


class SlotLayout:
    """Index arithmetic for the A | working partition.

    The working region is the suffix [k_anchor, total). Within it, the distinct
    slots are the trailing `used`, ordered oldest-first, so the newest block is
    always last. The prefix [k_anchor, total - used) is filler holding tiled
    copies of occupied content.
    """

    def __init__(self, k_anchor: int, total: int):
        if not 0 <= k_anchor < total:
            raise ValueError(f"k_anchor must be in [0, {total}), got {k_anchor}")
        self.k_anchor = int(k_anchor)
        self.total = int(total)

    @property
    def working_capacity(self) -> int:
        return self.total - self.k_anchor

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

    def to_dict(self) -> Dict[str, int]:
        return {
            "k_anchor": self.k_anchor,
            "total": self.total,
            "working_capacity": self.working_capacity,
        }


def summarise_block(delta: torch.Tensor) -> torch.Tensor:
    """Reduce a [L, n, d] delta block to one [d] candidate representation.

    Used only for the novelty proxy. Averaging over layers and tokens is coarse
    by construction; the note calls rho a proxy and requires it to allocate
    capacity rather than judge value, so a coarse summary is acceptable here and
    a precise one would not change the decisions it is allowed to drive.
    """
    if delta.dim() == 4:
        delta = delta[0]
    return delta.mean(dim=1).mean(dim=0)


class ProtectedResidualMemory:
    """Owns the pool partition, the write path, and capacity-triggered compaction."""

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
        self.layout = SlotLayout(k_anchor, self.total)

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

        self.basis: Optional[torch.Tensor] = None
        self.basis_mean: Optional[torch.Tensor] = None
        self.projection = AnchorProjection.empty(device=memory.device, dtype=memory.dtype)

        self.last_snapshot = CapacitySnapshot(budget_slots=self.layout.working_capacity)
        self.history: List[Dict[str, Any]] = []
        self.compaction_count = 0
        self.evicted_blocks = 0
        self.skipped_chunks = 0
        self.write_count = 0
        self.peak_used = 0
        self._headroom_handled = False
        self.anchor_info: Dict[str, Any] = {}

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
        self._headroom_handled = False
        self.last_snapshot = CapacitySnapshot(budget_slots=self.layout.working_capacity)
        self.importance = ImportanceTracker(self.cfg.importance_weights)

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
            return self.anchor_info

        block = self._encode(text, max_tokens)
        if block is None:
            # Too short to inject safely. The caller still has to write this chunk
            # through the residual path so it is not silently lost.
            self.anchor_info = {"anchor_tokens": 0, "anchor_slots": 0, "anchor_skipped": True}
            return self.anchor_info

        n = block.shape[1]
        target = self.layout.k_anchor
        reps = (target + n - 1) // n
        tiled = block.repeat(1, reps, 1)[:, :target, :].contiguous()

        with torch.no_grad():
            self._memory[:, self.layout.anchor_slice(), :] = tiled.to(self.dtype)

        self.anchor_slots = tiled.detach().clone()
        self.goal_direction = summarise_block(tiled).detach()
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
        """One write: novelty -> gate -> project -> place -> observe capacity.

        Returns a record for the run log. `dropped` is set when the chunk is
        below the 16-token injection floor, in which case the pool is untouched.
        """
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        raw_tokens = len(ids)
        if raw_tokens < MIN_INJECT_TOKENS:
            self.skipped_chunks += 1
            return {"dropped": True, "raw_tokens": raw_tokens, "reason": "below_min_inject"}

        truncated = raw_tokens > max_tokens
        self._headroom_handled = False
        started = time.time()
        block = self._encode(text, max_tokens)
        encode_seconds = time.time() - started
        if block is None:
            self.skipped_chunks += 1
            return {"dropped": True, "raw_tokens": raw_tokens, "reason": "encode_failed"}

        candidate = summarise_block(block)
        novelty = feature_novelty(candidate, self.basis, eps=self.cfg.novelty_eps)
        novelty_value = self.novelty.update(novelty)

        delta = block
        gate_value = 1.0
        if self.cfg.mode != "full_rewrite":
            delta = apply_projection(delta, self.projection)
            gate = self._gate(novelty)
            gate_value = float(gate)
            delta = delta * gate

        evicted = self._place(delta)
        # Tracked here rather than derived from `history`, because the drains
        # that feed the stats file empty that buffer and would make the peak read
        # as whatever happened to survive the last drain.
        self.peak_used = max(self.peak_used, int(self.used))
        # The pool is only ever read through this entry point, and `initialized`
        # must be 1 for injection to condition on the trailing block.
        self.model.initialized.fill_(1)

        self._refresh_basis()
        compacted, report = self._observe_capacity()

        record = {
            "dropped": False,
            "raw_tokens": raw_tokens,
            "tokens": int(block.shape[1]),
            "truncated": truncated,
            "novelty": float(novelty_value),
            "gate": gate_value,
            "encode_seconds": encode_seconds,
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

    def _place(self, delta: torch.Tensor) -> int:
        """Place one block, returning the number of blocks lost to eviction.

        Two write regimes, matching the control arms:

        * `full_rewrite` / `anchor_rewrite` (B5, B6) discard the previous working
          state and keep only the current block. That is what "rewrite the whole
          state every step" means, and it is the baseline PRLM-C must beat.
        * `ours` (B7, B8, Ours) appends, keeping the working state, and relies on
          compaction to make room.

        Appending shifts existing content left by one block so the new block lands
        at the very end; injection conditions on the trailing block, and the
        backbone's own update appends there too.
        """
        need = delta.shape[1]
        if need <= 0:
            return 0
        if need > self.layout.working_capacity:
            raise ValueError(
                "a single block of %d slots does not fit the working capacity of %d"
                % (need, self.layout.working_capacity)
            )

        if self.cfg.mode in ("full_rewrite", "anchor_rewrite"):
            with torch.no_grad():
                self._memory[:, self.total - need : self.total, :] = delta.to(self.dtype)
            self.used = need
            self.newest_width = need
            self.n_consolidated = 0
            self._tile_free_region()
            # Nothing is counted as evicted: these arms discard the previous
            # working state by definition, so a per-write eviction count would
            # report the arm's whole premise as a failure of compaction.
            return 0

        evicted = 0
        while self.used + need > self.layout.working_capacity:
            freed = self.compact(reason="headroom")
            if freed <= 0:
                self._evict_oldest(need)
                evicted += 1
        # This write made its own room, so the capacity branch in
        # `_observe_capacity` must not fire a second compaction for the same
        # write: after placement `used == working_capacity` exactly, which the
        # `>=` test in `over_budget` would otherwise read as over budget.
        self._headroom_handled = True

        self._shift_left(need)
        with torch.no_grad():
            self._memory[:, self.total - need : self.total, :] = delta.to(self.dtype)
        self.used += need
        self.newest_width = need
        return evicted

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
        free_start, free_end = self.layout.free_bounds(self.used)
        width = free_end - free_start
        if width <= 0:
            return

        if self.used > 0:
            source = self._distinct()
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
        activations = self._distinct()[0]
        basis, mean, _ = estimate_basis(
            activations,
            max_rank=min(self.cfg.basis_max_rank, self.used),
            energy=self.cfg.basis_energy,
            subtract_mean=True,
            max_rows=_BASIS_MAX_ROWS,
        )
        self.basis = basis
        self.basis_mean = mean

    def _observe_capacity(self):
        """Compute the capacity signals and apply the trigger rule.

        The rank spectrum is only computed when the occupancy condition is
        already satisfied, or when the trigger is periodic. A spectrum is never
        needed to decide *not* to compact, and computing it on every write is what
        would make this method dominate the run time.
        """
        occupancy = occupancy_ratio(self.used, self.layout.working_capacity)
        eff = 0.0
        eta = 0
        rank_budget = 0
        needs_spectrum = self.cfg.trigger == "occupancy_novelty_rank" and (
            occupancy >= self.cfg.tau_k
        )
        if needs_spectrum:
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
            # `>=` rather than `>`: the pool terminates exactly at capacity, so a
            # strict comparison is never true and the immediate branch below is
            # dead. The flag suppresses the branch when _place already made room
            # for this write, which would otherwise compact twice per write.
            over_budget=(
                self.used >= self.layout.working_capacity
                and not self._headroom_handled
            ),
        )
        self.last_snapshot = snapshot

        if self.cfg.mode in ("full_rewrite", "anchor_rewrite"):
            return False, None
        if self.cfg.compact_mode == "none":
            return False, None

        if self.trigger.observe(snapshot):
            report = self.compact_with_report(reason="capacity_trigger")
            self.trigger.on_compacted()
            return report["freed_slots"] > 0, report
        return False, None

    def _spectrum(self, max_rows: int = _SPECTRUM_MAX_ROWS) -> Optional[torch.Tensor]:
        if self.used < self.block:
            return None
        activations = self._distinct()[0].detach().to(torch.float32)
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

        start = self.total - used
        source = self._memory[:, start : start + window, :].detach().to(torch.float32)
        weights = self._window_importance(start, window)

        try:
            compacted = self.compactor(source, weights, target)
        except Exception as error:  # noqa: BLE001 - a failed compaction must not kill the run
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
        self._tile_free_region()
        self._refresh_basis()
        # Surviving slots gain a reuse credit; this is the only reuse signal the
        # pool has, since nothing addresses individual slots.
        self.importance.mark_reuse(
            torch.arange(self.used, device=self._memory.device), self.used
        )

        report = {
            "freed_slots": window - target,
            "reason": reason,
            "window_slots": window,
            "target_slots": target,
            "used_before": used,
            "used_after": self.used,
            "shots": self.compaction_count,
            "weighted": weights is not None,
        }
        self.history.append({"event": "compact", **report})
        return report

    def _window_importance(self, start: int, window: int) -> Optional[torch.Tensor]:
        """Importance weights for the sliding window, or None for the plain SVD baseline."""
        if self.cfg.compact_mode != "weighted_svd_tokens":
            return None

        slots = self._memory[0, start : start + window, :]
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
            "anchor": self.anchor_info,
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
