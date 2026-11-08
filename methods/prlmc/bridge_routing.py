"""V12: a cross-block bridging route and a dynamically allocated 2,304-slot budget.

What this file replaces
-----------------------
Round 6 (V11) built the window out of three modules and only one of them survived
its own controls. Folding same-block redundancy was worth -4 questions against plain
eviction (57 vs 61) and failed the donor test, and the full-pool calibration rewrote
the same values the fold had already rewritten. What DID survive is the part that
was never isolated: reader-aware selection of NATIVE slots, one selection shared by
all 32 layers, chronological output, 2,304 slots.

So this round deletes the fold and the calibration and changes only the two things
V11 could not vary -- who gets the capacity, and how the capacity is priced:

    V11 (fixed quota)   9 carriers x 180 core  + 9 x 76 graft        = 2,304
                        every carrier gives up 76 slots no matter what
                        any one dropped block contributes 0..76, decided by rank
                        inside a pool of 2,560 -- a fixed split, not a decision

    V12 (this file)     9 carriers x 140 protected + 1,044 contested = 2,304
                        the contested 1,044 are chosen GLOBALLY, so a dropped
                        block that holds the evidence can outbid a carrier that
                        does not, and the sharing between dropped blocks is an
                        outcome rather than a parameter

1,044 is not a round number: it is exactly what the nine carriers release,
9 x (256 - 140). The budget is therefore capacity-neutral at 2,304 with the same
arithmetic as every round before it -- every dropped-block slot that enters costs
exactly one carrier slot -- and nothing is added, folded or rescaled. Every output
slot is a byte-for-byte copy of a slot that was written, which the assembler checks
with `torch.equal` rather than asserts in prose.

What is held fixed
------------------
The 19-block write, the nine carriers (`gr.select_carriers`, one selection shared by
every arm), the 2,304-slot budget, the frozen reader and writer, the chronological
output order, and the "same slot index in all 32 layers" transport. Only the ranking
and the allocation differ between the two arms, so an accuracy difference is
attributable to them by construction -- content-causality is still NOT settled by
this design (there is no donor arm this round), and the report says so.

Two routers, one assembler
--------------------------
`FixedQuotaRouter`   arm B: V11's geometry, re-expressed over native slots only
`BridgeRouter`       arm C: protected quota + global competition, ranked by reader
                     utility AND by a cross-block bridging score

`_assemble` orders the window chronologically, builds one row of metadata per slot,
and checks provenance in one place for both routers, so the arms cannot drift apart
in anything except the two mechanisms under test. `read` returns the one reader
reading both of them use: the projections cost the same whether one router or two
ask for them.
"""
import math

import torch
import torch.nn.functional as F

from methods.prlmc.reader_utility import SCHEMA as UTILITY_SCHEMA, mmr_select

SCHEMA = "bridge_routing/1"
# Arm B, V11's quota, unchanged: 180 + 76 = 256 per carrier block.
DEFAULT_CORE_PER_BLOCK = 180
DEFAULT_GRAFT_PER_BLOCK = 76
# Arm C: 140 protected per carrier (1,260) and 1,044 contested. 2,304 - 1,260.
DEFAULT_PROTECTED_PER_BLOCK = 140
DEFAULT_SEEDS_PER_BLOCK = 8
DEFAULT_MMR_PENALTY = 0.5
DEFAULT_NOVELTY_WEIGHT = 0.5


def _normalise(scores):
    """Min-max onto [0, 1]; a constant vector maps to zeros rather than to NaN."""
    span = float(scores.max() - scores.min())
    if not math.isfinite(span) or span <= 0.0:
        return torch.zeros_like(scores)
    return (scores - scores.min()) / span


def _max_cosine(candidate_key, reference_key, chunk=512):
    """`max` cosine from each candidate row to a reference set; both sides unit rows."""
    best = torch.full((int(candidate_key.shape[0]),), -1.0,
                      dtype=candidate_key.dtype, device=candidate_key.device)
    for start in range(0, int(reference_key.shape[0]), chunk):
        block = candidate_key @ reference_key[start:start + chunk].t()
        best = torch.maximum(best, block.max(dim=-1).values)
    return best.clamp(-1.0, 1.0)


def _correlation(left, right):
    """Pearson r over two 1-D score vectors; None when either side is constant."""
    a = left.float() - left.float().mean()
    b = right.float() - right.float().mean()
    denominator = float(a.norm() * b.norm())
    if denominator <= 1e-12:
        return None
    return round(float((a * b).sum()) / denominator, 4)


def memory_salience(source, block_size, key):
    """`disc_leverage`, re-expressed over the written pool: nothing from outside enters.

    V10 ranks its graft candidates with

        ||route|| * (0.6 + dispersion) * (0.5 + relevance_to_kept) * novelty

    (`manifold_graft.py`), a score of a slot against the pool itself. Here the route
    representation is the reader's own multi-layer key of the slot (`SlotReading.key`,
    unit rows, one row per window position) and the reference set is the OTHER carrier
    blocks, so the same four factors answer the question this router needs:

        norm         mean L2 length of the slot across the reader's utility layers
        dispersion   1 - cos(slot, its own block's centroid): how unlike the block's
                     average the slot is
        relevance    max cos to another carrier block's centroid, clamped to
                     [0.2, 0.82] exactly as V10 clamps it
        novelty      1 - 5 * max(0, raw_relevance - 0.85), floored at 0.1

    No question of any kind enters: `key` comes from `ReaderScorer.reading(..., probes=
    None)`, which builds keys and values from the stored vectors alone.
    """
    n_slots = int(source.shape[1])
    block = int(block_size)
    if n_slots % block:
        raise ValueError(f"{n_slots} slots do not fill {block}-slot blocks")
    blocks = n_slots // block
    unit = key.float()
    if int(unit.shape[0]) != n_slots:
        raise ValueError("one key row per slot")
    norm = source.float().norm(dim=-1).mean(dim=0)
    centroids = torch.cat([
        F.normalize(unit[index * block:(index + 1) * block].mean(dim=0, keepdim=True),
                    dim=-1)
        for index in range(blocks)], dim=0)
    similarity = unit @ centroids.t()
    salience = torch.empty(n_slots, dtype=torch.float32, device=unit.device)
    for index in range(blocks):
        lo, hi = index * block, (index + 1) * block
        own = similarity[lo:hi, index].clamp(-1.0, 1.0)
        others = similarity[lo:hi].clone()
        others[:, index] = -2.0
        raw = others.max(dim=-1).values
        relevance = raw.clamp(0.2, 0.82)
        novelty = (1.0 - (raw - 0.85).clamp_min(0.0) * 5.0).clamp_min(0.1)
        salience[lo:hi] = norm[lo:hi] * (0.6 + (1.0 - own)) * (0.5 + relevance) * novelty
    return salience


class Router:
    """Shared machinery: one reading, one novelty discount, one assembler."""

    def __init__(self, scorer, probes=None, mmr_penalty=DEFAULT_MMR_PENALTY,
                 novelty_weight=DEFAULT_NOVELTY_WEIGHT):
        self.scorer = scorer
        # `probes=None` is the probe-free path: the router may use the reader's key
        # and value projections but never a question. `BridgeRouter` runs this way.
        self.probes = list(probes) if probes else None
        self.mmr_penalty = float(mmr_penalty)
        self.novelty_weight = float(novelty_weight)

    # -- shared pieces ------------------------------------------------------
    def read(self, source):
        """One reader reading of the whole window, shared by every router."""
        positions = torch.arange(int(source.shape[1]), device=source.device)
        return self.scorer.reading(source, positions, self.probes)

    def _score(self, utility, novelty, bridge=None, bridge_weight=0.0):
        """Utility plus an optional bridging term, discounted for duplication.

        Both terms are min-max normalised over the candidate pool before they meet,
        so neither can win on scale, and the product form `(1 - w + w * novelty)`
        keeps the [0, 1] range: a candidate that duplicates a kept slot is worth at
        least `1 - w` of its raw score, never zero and never more than its raw score.
        """
        combined = _normalise(utility)
        if bridge is not None:
            weight = float(bridge_weight)
            combined = (1.0 - weight) * combined + weight * _normalise(bridge)
        weight = float(self.novelty_weight)
        return combined * (1.0 - weight + weight * novelty)

    def _offsets(self, blocks, block_size, device):
        blocks = torch.as_tensor([int(block) for block in blocks], dtype=torch.long,
                                 device=device)
        return (blocks.unsqueeze(1) * int(block_size)
                + torch.arange(int(block_size), device=device))

    def _split(self, source, carrier_blocks, block_size):
        window_slots = int(source.shape[1])
        blocks = window_slots // int(block_size)
        if window_slots % int(block_size):
            raise ValueError(f"{window_slots} slots do not fill "
                             f"{block_size}-slot blocks")
        carriers = [int(block) for block in carrier_blocks]
        if len(set(carriers)) != len(carriers):
            raise ValueError("a carrier block is listed twice")
        if not carriers or min(carriers) < 0 or max(carriers) >= blocks:
            raise ValueError(f"carrier set {carriers} outside a {blocks}-block window")
        dropped = [block for block in range(blocks) if block not in set(carriers)]
        return blocks, carriers, dropped

    def _assemble(self, source, slots, origin, meta, block_size):
        """Chronological order, provenance checked against the written window.

        The verbatim check is the whole native-transport claim: whatever the routers
        chose, every installed slot has to be bit-identical to the slot it claims to
        be. It is checked here, once, for both arms.
        """
        if origin.numel() != int(slots.shape[1]):
            raise ValueError("one source position per output slot")
        if origin.numel() != int(torch.unique(origin).numel()):
            raise ValueError("two output slots claim the same source position")
        order = torch.argsort(origin, stable=True)
        window = slots.index_select(1, order)
        source_position = origin.index_select(0, order)
        copied = source.index_select(1, source_position)
        if not torch.equal(window, copied):
            raise ValueError("an output slot is not a verbatim copy of its source: "
                             "the transport stopped being native")
        merged = dict(source_position=source_position,
                      source_block=torch.div(source_position, int(block_size),
                                             rounding_mode="floor"))
        merged.update({name: value.index_select(0, order) for name, value in meta.items()})
        return window, merged


class FixedQuotaRouter(Router):
    """Arm B: V11's geometry over native slots -- 180 core + 76 graft per carrier.

    This is the arm the new router has to beat. It is the V11 EVICTION window
    (`V11Evict`, 61/158) minus the calibration module, rebuilt here instead of loaded
    from a checkpoint, because the calibration is exactly what this round removes:
    the comparison is then "same quota, same reader-aware ranking, same native
    transport, no trained parameters anywhere", and the only difference to arm C is
    the routing rule itself.

    Core: per carrier block, the top `core_per_block` slots by reader utility, kept
    verbatim and in source order. Graft: `graft_per_block` x 9 slots drawn from the
    ten dropped blocks by utility x novelty, de-duplicated with MMR.
    """

    def __init__(self, scorer, probes, core_per_block=DEFAULT_CORE_PER_BLOCK,
                 graft_per_block=DEFAULT_GRAFT_PER_BLOCK, mmr_penalty=DEFAULT_MMR_PENALTY,
                 novelty_weight=DEFAULT_NOVELTY_WEIGHT):
        super().__init__(scorer, probes, mmr_penalty, novelty_weight)
        if int(core_per_block) <= 0 or int(graft_per_block) < 0:
            raise ValueError("the fixed quota must be non-negative")
        self.core_per_block = int(core_per_block)
        self.graft_per_block = int(graft_per_block)

    def build(self, source, carrier_blocks, block_size=256, reading=None):
        """`source` `[layers, window_slots, D]` -> window, per-slot metadata, report."""
        reading = self.read(source) if reading is None else reading
        slots, per_layer = reading
        device = source.device
        blocks, carriers, dropped = self._split(source, carrier_blocks, block_size)
        if self.core_per_block > int(block_size):
            raise ValueError(f"a {self.core_per_block}-slot core does not fit a "
                             f"{block_size}-slot block")
        if self.core_per_block + self.graft_per_block != int(block_size):
            raise ValueError(f"{self.core_per_block} core + {self.graft_per_block} graft "
                             f"does not fill a {block_size}-slot block")

        core_slots, core_origin, core_block, per_block = [], [], [], []
        for carrier in carriers:
            positions = self._offsets([carrier], block_size, device)[0]
            local = slots.utility.index_select(0, positions)
            keep = torch.argsort(local, descending=True)[: self.core_per_block]
            keep, _ = torch.sort(keep)
            chosen = positions.index_select(0, keep)
            core_slots.append(source.index_select(1, chosen))
            core_origin.append(chosen)
            core_block.append(torch.full((int(chosen.numel()),), carrier,
                                         dtype=torch.long, device=device))
            per_block.append(dict(block=carrier, kept=int(chosen.numel()),
                                  released=int(block_size) - int(chosen.numel())))
        core_slots = torch.cat(core_slots, dim=1)
        core_origin = torch.cat(core_origin, dim=0)
        core_block = torch.cat(core_block, dim=0)

        dropped_positions = self._offsets(dropped, block_size, device).reshape(-1)
        wanted = len(carriers) * self.graft_per_block
        if wanted > int(dropped_positions.numel()):
            raise ValueError(f"{wanted} graft slots from {dropped_positions.numel()} "
                             f"candidates")
        novelty = 1.0 - _max_cosine(slots.key.index_select(0, dropped_positions),
                                    slots.key.index_select(0, core_origin))
        score = self._score(slots.utility.index_select(0, dropped_positions), novelty)
        picked = mmr_select(score, slots.key.index_select(0, dropped_positions), wanted,
                            penalty=self.mmr_penalty)
        picked, _ = torch.sort(picked)
        graft_origin = dropped_positions.index_select(0, picked)

        window, meta = self._assemble(
            source, torch.cat([core_slots, source.index_select(1, graft_origin)], dim=1),
            torch.cat([core_origin, graft_origin], dim=0),
            dict(slot_type=torch.cat([
                torch.zeros(int(core_origin.numel()), dtype=torch.long, device=device),
                torch.ones(int(graft_origin.numel()), dtype=torch.long, device=device)]),
                protected=torch.cat([
                    torch.ones(int(core_origin.numel()), dtype=torch.bool, device=device),
                    torch.zeros(int(graft_origin.numel()), dtype=torch.bool,
                                device=device)]),
                utility=slots.utility.index_select(
                    0, torch.cat([core_origin, graft_origin], dim=0)),
                bridge=torch.zeros(int(core_origin.numel()) + int(graft_origin.numel()),
                                   device=device)),
            block_size=block_size)
        per_dropped = [int((meta["source_block"][meta["slot_type"] == 1] == block).sum())
                       for block in dropped]
        per_carrier = [int(meta["protected"][meta["source_block"] == block].sum())
                       for block in carriers]
        telemetry = dict(
            schema=SCHEMA, policy="fixed_quota", routing="reader utility x novelty, MMR",
            carrier_blocks=carriers, dropped_blocks=dropped,
            core_per_block=self.core_per_block, graft_per_block=self.graft_per_block,
            protected_per_block=self.core_per_block,
            protected_total=int(core_origin.numel()),
            contested_total=int(graft_origin.numel()),
            total=int(window.shape[1]),
            admitted_from_dropped=int(graft_origin.numel()), admitted_from_carriers=0,
            admitted_per_dropped_block=per_dropped,
            admitted_per_carrier_block=[int(block_size - count)
                                        for count in per_carrier],
            core_per_carrier_block=per_carrier,
            empty_dropped_blocks=int(sum(1 for count in per_dropped if count == 0)),
            min_per_dropped_block=int(min(per_dropped)) if per_dropped else 0,
            max_per_dropped_block=int(max(per_dropped)) if per_dropped else 0,
            bridge_available=False, bridge_weight=0.0, seeds_per_block=0, seeds_total=0,
            mmr_penalty=self.mmr_penalty, novelty_weight=self.novelty_weight,
            novelty_mean=float(novelty.mean()), novelty_min=float(novelty.min()),
            utility_mean=float(slots.utility.mean()), utility_per_layer=per_layer,
            per_block=per_block)
        return window, meta, telemetry


class BridgeRouter(Router):
    """Arm C: 140 protected per carrier, and 1,044 contested by the whole window.

    NOTHING OUTSIDE THE POOL DECIDES ANYTHING HERE. The first cut of this router
    seeded the bridging queries with the slots that scored highest against a bank of
    16 fixed musique questions, which put an external question back into the one arm
    that exists to remove it. All three decisions are now self-generated:

    1. **Floor.** Each carrier block keeps its top `protected_per_block` slots by
       `memory_salience` -- a function of the written vectors and the reader's own
       key projections only.
    2. **Seeds.** The bridging queries are the carrier's top `seeds_per_block` slots by
       the same score. They are therefore part of the protected set, which is the point:
       the question the bridging mass answers is "what does what I am keeping point at".
    3. **Contested.** The remaining `9 x (256 - 140) = 1,044` slots are filled by every
       unselected carrier slot AND every dropped-block slot, ranked together by the
       bridging mass alone, discounted for duplication and de-duplicated with MMR.

    The bridging mass is still a proxy -- the deployed read path never queries with a
    memory slot -- and it is now a proxy with no question in it, which is the version
    the design asked for.
    """

    def __init__(self, scorer, protected_per_block=DEFAULT_PROTECTED_PER_BLOCK,
                 seeds_per_block=DEFAULT_SEEDS_PER_BLOCK,
                 mmr_penalty=DEFAULT_MMR_PENALTY,
                 novelty_weight=DEFAULT_NOVELTY_WEIGHT):
        super().__init__(scorer, None, mmr_penalty, novelty_weight)
        if int(protected_per_block) <= 0:
            raise ValueError("the protected quota must be positive")
        if int(seeds_per_block) <= 0:
            raise ValueError("the bridge needs at least one seed per carrier")
        self.protected_per_block = int(protected_per_block)
        self.seeds_per_block = int(seeds_per_block)

    def build(self, source, carrier_blocks, block_size=256, reading=None):
        reading = self.read(source) if reading is None else reading
        slots, per_layer = reading
        if slots.utility is not None:
            raise ValueError("the bridge router must run on a probe-free reading")
        device = source.device
        blocks, carriers, dropped = self._split(source, carrier_blocks, block_size)
        if self.protected_per_block > int(block_size):
            raise ValueError(f"a {self.protected_per_block}-slot floor does not fit a "
                             f"{block_size}-slot block")
        salience = memory_salience(source, block_size, slots.key)

        # -- 1. the floor, and the seeds that do the asking -------------------
        protected_slots, protected_origin, per_block = [], [], []
        released, seeds = [], []
        for carrier in carriers:
            positions = self._offsets([carrier], block_size, device)[0]
            local = salience.index_select(0, positions)
            ranked = torch.argsort(local, descending=True)
            keep, _ = torch.sort(ranked[: self.protected_per_block])
            chosen = positions.index_select(0, keep)
            protected_slots.append(source.index_select(1, chosen))
            protected_origin.append(chosen)
            seeds.append(positions.index_select(0, ranked[: self.seeds_per_block]))
            keep_mask = torch.zeros(int(block_size), dtype=torch.bool, device=device)
            keep_mask[keep] = True
            released.append(positions[~keep_mask])
            per_block.append(dict(block=carrier, protected=int(chosen.numel()),
                                  released=int((~keep_mask).sum()),
                                  salience_mean=float(local.mean()),
                                  salience_max=float(local.max())))
        protected_slots = torch.cat(protected_slots, dim=1)
        protected_origin = torch.cat(protected_origin, dim=0)
        seeds = torch.cat(seeds, dim=0)

        # -- 2. the contested pool: released carrier slots + every dropped slot --
        dropped_positions = self._offsets(dropped, block_size, device).reshape(-1)
        candidates, _ = torch.sort(torch.cat(released + [dropped_positions], dim=0))
        wanted = len(carriers) * (int(block_size) - self.protected_per_block)
        if wanted > int(candidates.numel()):
            raise ValueError(f"{wanted} contested slots from {candidates.numel()} "
                             f"candidates")

        # -- 3. the ranking: bridging mass only, discounted for duplication ----
        bridge, bridge_per_layer = self.scorer.affinity(
            source.index_select(1, candidates), candidates,
            source.index_select(1, seeds), seeds)
        candidate_salience = salience.index_select(0, candidates)
        novelty = 1.0 - _max_cosine(slots.key.index_select(0, candidates),
                                    slots.key.index_select(0, protected_origin))
        weight = float(self.novelty_weight)
        score = _normalise(bridge) * (1.0 - weight + weight * novelty)
        picked = mmr_select(score, slots.key.index_select(0, candidates), wanted,
                            penalty=self.mmr_penalty)
        picked, _ = torch.sort(picked)
        admitted_origin = candidates.index_select(0, picked)

        window, meta = self._assemble(
            source, torch.cat([protected_slots, source.index_select(1, admitted_origin)],
                              dim=1),
            torch.cat([protected_origin, admitted_origin], dim=0),
            dict(slot_type=torch.cat([
                torch.zeros(int(protected_origin.numel()), dtype=torch.long,
                            device=device),
                self._is_dropped(admitted_origin, dropped, block_size)]),
                protected=torch.cat([
                    torch.ones(int(protected_origin.numel()), dtype=torch.bool,
                               device=device),
                    torch.zeros(int(admitted_origin.numel()), dtype=torch.bool,
                                device=device)]),
                utility=salience.index_select(
                    0, torch.cat([protected_origin, admitted_origin], dim=0)),
                bridge=torch.cat([
                    torch.full((int(protected_origin.numel()),), float("nan"),
                               device=device),
                    bridge.index_select(0, picked)])),
            block_size=block_size)
        per_dropped = [int((meta["source_block"][meta["slot_type"] == 1] == block).sum())
                       for block in dropped]
        per_carrier_admitted = [int(
            (meta["source_block"][~meta["protected"]] == block).sum()) for block in carriers]
        admitted_from_dropped = int((meta["slot_type"] == 1).sum())
        telemetry = dict(
            schema=SCHEMA, policy="bridge",
            routing="protected floor + global competition, ranked by cross-block "
                    "bridging mass (probe-free: salience floor, memory-only seeds)",
            carrier_blocks=carriers, dropped_blocks=dropped,
            protected_per_block=self.protected_per_block,
            protected_total=int(protected_origin.numel()),
            contested_total=int(admitted_origin.numel()),
            total=int(window.shape[1]),
            admitted_from_dropped=admitted_from_dropped,
            admitted_from_carriers=int(admitted_origin.numel()) - admitted_from_dropped,
            admitted_share_from_dropped=round(
                100.0 * admitted_from_dropped / max(int(admitted_origin.numel()), 1), 2),
            admitted_per_dropped_block=per_dropped,
            admitted_per_carrier_block=per_carrier_admitted,
            empty_dropped_blocks=int(sum(1 for count in per_dropped if count == 0)),
            min_per_dropped_block=int(min(per_dropped)) if per_dropped else 0,
            max_per_dropped_block=int(max(per_dropped)) if per_dropped else 0,
            probe_free=True, external_questions=0,
            seeds_per_block=self.seeds_per_block, seeds_total=int(seeds.numel()),
            salience_mean=float(salience.mean()), salience_max=float(salience.max()),
            salience_protected_share=round(100.0 * float(
                salience.index_select(0, protected_origin).sum()
                / salience.sum().clamp_min(1e-12)), 2),
            bridge_candidate_mean=float(bridge.mean()),
            bridge_candidate_std=float(bridge.std()),
            bridge_admitted_mean=float(bridge.index_select(0, picked).mean()),
            bridge_rejected_mean=self._rejected_mean(bridge, picked),
            bridge_salience_correlation=_correlation(bridge, candidate_salience),
            bridge_per_layer=bridge_per_layer,
            mmr_penalty=self.mmr_penalty, novelty_weight=self.novelty_weight,
            novelty_mean=float(novelty.mean()), novelty_min=float(novelty.min()),
            read_per_layer=per_layer,
            per_block=per_block)
        return window, meta, telemetry

    @staticmethod
    def _is_dropped(origin, dropped, block_size):
        """1 for an admitted slot that came from a dropped block, else 0."""
        dropped_blocks = torch.as_tensor([int(block) for block in dropped],
                                         dtype=torch.long, device=origin.device)
        block_of = torch.div(origin, int(block_size), rounding_mode="floor")
        return torch.isin(block_of, dropped_blocks).to(torch.long)

    @staticmethod
    def _rejected_mean(bridge, picked):
        mask = torch.ones(int(bridge.numel()), dtype=torch.bool, device=bridge.device)
        mask[picked] = False
        if not bool(mask.any()):
            return float("nan")
        return float(bridge[mask].mean())
