"""V13 base window: V11Evict's geometry, and nothing else.

The mechanism this round tests sits in the CALIBRATION, not in the selection, so the
window has to be held exactly at the structure that already works:

    carrier blocks  9 x 256  ->  keep the top 180 by reader utility, verbatim   1,620
    dropped blocks 10 x 256  ->  reader utility x novelty + MMR                  684
                                                                             2,304 slots

That is the `V11Evict` window of experiment 7 (61/158), reproduced here over native
slots with no folding and no trained parameters. `JointConsolidation` keeps its V11
name so the drivers that were written for it keep their shape; the only policy it
accepts is eviction, because the fold was measured to be worth -4 questions.

One addition, needed by the calibration this round: the metadata carries the reader's
own scoring object, because the cross-block context has to be computed from the
window the reader will actually read. `cross_block_calibration` reads it from there
instead of every call site having to thread it through.

V17 adds two window-budget knobs, both CLOSED by default. The window is a control
variable in this project -- every arm of the calibration is compared on a window that
was built the same way -- so a new knob is only allowed to change where the reader's
slots are spent, never how many there are:

    quota_mode="density"   the same 1,620 core slots, split 180/180/... no longer:
                           a block whose material is redundant gets fewer slots and
                           a block of distinct material gets more. The per-block
                           quota is bounded by [min_core, max_core] and summed with an
                           integer largest-remainder pass, so the total is EXACTLY
                           len(carriers) * core_per_block (1,620 for nine carriers).
    absorb_cos>0           the slots a carrier block is about to lose are, when they
                           are near-duplicates (cosine > absorb_cos) of a slot that
                           block keeps, folded into that kept slot through
                           `core_fusion.rms_preserve` instead of being dropped. Bounded
                           by `absorb_alpha_max`, recorded per slot, and confined to
                           same-block pairs -- this is not a cross-block fusion arm.

With `quota_mode="fixed"` and `absorb_cos=0.0` this builder is the V13 one: the same
nine quotas of 180, no absorb pass, and every output slot still has to be a verbatim
copy of its source position.
"""
import math

import torch

from methods.prlmc.reader_utility import SCHEMA as UTILITY_SCHEMA, mmr_select

SCHEMA = "evict_window/1"
# 180 x 9 = 1,620 core, 76 x 9 = 684 graft, 2,304 in total. V11's quota, unchanged.
QUOTA = dict(core_per_block=180, graft_per_block=76)
# `fixed` spends `core_per_block` on every carrier; `density` spends the same total
# unevenly (see `_capped_quota`).
QUOTA_MODES = ("fixed", "density")


def _max_cosine(candidate_key, reference_key, chunk=512):
    """`max` cosine from each candidate row to a reference set; both sides unit rows."""
    best = torch.full((int(candidate_key.shape[0]),), -1.0,
                      dtype=candidate_key.dtype, device=candidate_key.device)
    for start in range(0, int(reference_key.shape[0]), chunk):
        block = candidate_key @ reference_key[start:start + chunk].t()
        best = torch.maximum(best, block.max(dim=-1).values)
    return best.clamp(-1.0, 1.0)


def _block_density(key, block_positions, sample, chunk=256):
    """How redundant one carrier block is, on `sample` evenly spaced slots.

    Density is the mean `1 - max cosine` from a sampled slot to the OTHER slots of the
    same block: a block of near-copies scores near 0 (keeping 180 of it buys 180
    distinct things only if the 180 are distinct), a block of distinct material scores
    near 1. Keys are already unit rows out of `ReaderScorer.reading`, so the dot
    product is the cosine.

    The sample is a fixed, evenly spaced index set, not a random draw: the quota must
    be a deterministic function of the window, otherwise two builds of "the same"
    window differ and every single-variable comparison in V17 loses its control.
    """
    n = int(block_positions.numel())
    if n < 2:
        return 0.0
    count = min(int(sample), n)
    picks = torch.linspace(0, n - 1, count, device=block_positions.device).round().long()
    picks = torch.unique(picks, sorted=True)
    probe = key.index_select(0, block_positions.index_select(0, picks))
    pool = key.index_select(0, block_positions)
    parts = []
    for start in range(0, n, chunk):
        parts.append(probe @ pool[start:start + chunk].t())
    similarity = torch.cat(parts, dim=1)
    # A slot is not its own neighbour: the diagonal would pin every max cosine at 1
    # and report every block as perfectly redundant.
    similarity[torch.arange(int(picks.numel()), device=block_positions.device), picks] = -2.0
    best = similarity.max(dim=-1).values.clamp(-1.0, 1.0)
    return float((1.0 - best).mean())


def _capped_quota(weights, total, low, high):
    """Split `total` core slots over the blocks: `low` each, the rest by weight, capped.

    Integer largest-remainder, because the point of a per-block quota is that the
    reader's budget does not move: rounding each block's float share independently
    would hand back 1,619 or 1,621 slots and silently change the experiment instead of
    only the allocation. Blocks whose proportional share would break `high` are pinned
    there and the remainder is re-shared among the rest, so the caps are respected
    without throwing the proportions away.
    """
    n = len(weights)
    if n == 0:
        raise ValueError("a quota needs at least one block")
    low, high, total = int(low), int(high), int(total)
    if high < low:
        raise ValueError(f"max_core {high} is below min_core {low}")
    held = [high - low] * n
    extra = total - low * n
    if extra < 0 or extra > sum(held):
        raise ValueError(f"{total} core slots cannot be split over {n} blocks "
                         f"within [{low}, {high}]")
    allocated = [0] * n
    remaining = extra
    active = [index for index in range(n) if held[index] > 0]
    while remaining > 0 and active:
        mass = [max(float(weights[index]), 0.0) for index in active]
        denom = sum(mass)
        if denom <= 0.0:                       # all-equal blocks: split evenly
            mass = [1.0] * len(active)
            denom = float(len(active))
        ideal = {index: remaining * mass[position] / denom
                 for position, index in enumerate(active)}
        pinned = [index for index in active if ideal[index] > held[index]]
        if pinned:
            for index in pinned:
                allocated[index] += held[index]
                remaining -= held[index]
                held[index] = 0
            pinned = set(pinned)
            active = [index for index in active if index not in pinned]
            continue
        floors = {index: int(math.floor(ideal[index])) for index in active}
        for index in active:
            allocated[index] += floors[index]
        leftover = remaining - sum(floors.values())
        # Largest remainder first, ties broken by block order so the result is stable.
        ranked = sorted(active, key=lambda index: (-(ideal[index] - floors[index]), index))
        for index in ranked:
            if leftover <= 0:
                break
            if floors[index] < held[index]:
                allocated[index] += 1
                leftover -= 1
        if leftover > 0:
            raise ValueError("largest-remainder quota left slots unassigned")
        remaining = 0
    if remaining > 0:
        raise ValueError(f"{total} core slots do not fit the per-block caps")
    return [low + value for value in allocated]


class JointConsolidation:
    """Builds the V11Evict window. Holds no parameters: selection only."""

    def __init__(self, scorer, probes, core_per_block=QUOTA["core_per_block"],
                 graft_per_block=QUOTA["graft_per_block"], policy="evict",
                 mmr_penalty=0.5, novelty_weight=0.5, merge_kwargs=None,
                 quota_mode="fixed", min_core=140, max_core=256, density_sample=64,
                 absorb_cos=0.0, absorb_alpha_max=0.5):
        if policy != "evict":
            raise ValueError(f"V13 runs the eviction policy only; got {policy!r}")
        if core_per_block + graft_per_block != 256:
            raise ValueError("core + graft per block must fill one 256-slot block")
        if quota_mode not in QUOTA_MODES:
            raise ValueError(f"quota_mode is one of {QUOTA_MODES}; got {quota_mode!r}")
        self.scorer = scorer
        self.probes = list(probes)
        if not self.probes:
            raise ValueError("the reader-utility score needs at least one probe")
        self.core_per_block = int(core_per_block)
        self.graft_per_block = int(graft_per_block)
        self.policy = "evict"
        self.mmr_penalty = float(mmr_penalty)
        self.novelty_weight = float(novelty_weight)
        self.merge_kwargs = dict(merge_kwargs or {})
        self.quota_mode = "fixed" if quota_mode == "fixed" else "density"
        self.min_core = int(min_core)
        self.max_core = int(max_core)
        if self.min_core < 1:
            raise ValueError("min_core must keep at least one slot of a carrier block")
        if self.max_core < self.min_core:
            raise ValueError(f"max_core {self.max_core} is below min_core {self.min_core}")
        self.density_sample = int(density_sample)
        if self.density_sample < 1:
            raise ValueError("density_sample must sample at least one slot per block")
        self.absorb_cos = float(absorb_cos)
        if not 0.0 <= self.absorb_cos <= 1.0:
            raise ValueError("absorb_cos is a cosine threshold and lives in [0, 1]")
        self.absorb_alpha_max = float(absorb_alpha_max)
        if self.absorb_alpha_max < 0.0:
            raise ValueError("absorb_alpha_max cannot be negative")

    # -- the two V17 knobs --------------------------------------------------
    def _density_quotas(self, reading, offsets, block_size):
        """Per-block core quotas from inside-block redundancy; the total never moves."""
        blocks = int(offsets.shape[0])
        total = blocks * self.core_per_block
        if self.max_core > int(block_size):
            raise ValueError(f"max_core {self.max_core} exceeds the {block_size}-slot block")
        density = [_block_density(reading.key, offsets[index], self.density_sample)
                   for index in range(blocks)]
        quotas = _capped_quota(density, total, self.min_core, self.max_core)
        return quotas, density

    def _absorb(self, core_window, kept_per_block, offsets, source, reading):
        """Fold each block's about-to-be-evicted slots into the kept slot they duplicate.

        `absorb_cos` is the only reason a core slot is allowed to stop being a verbatim
        copy of its source. A slot the block was going to delete is merged into a kept
        slot of the SAME block when the two are near-duplicates, so the eviction stops
        being a deletion at all and no cross-block claim is made. The merge goes
        through `core_fusion.rms_preserve` (add, then restore the destination slot's
        RMS), which bounds it by construction: an absorbed slot cannot grow the
        window's norm and quietly buy its block more attention.

        `alpha` is per slot and never above `absorb_alpha_max`; it is the reported
        quantity because a reader of the telemetry has to be able to tell a window
        whose slots were merged at 0.05 from one merged at the ceiling.

        Returns the new core window, a mask of the slots that changed, and a report.
        """
        absorbed = torch.zeros(int(core_window.shape[1]), dtype=torch.bool,
                               device=core_window.device)
        report = dict(absorbed_pairs=0, mean_alpha=0.0, mean_cos=0.0,
                      per_block=[0] * len(kept_per_block))
        if self.absorb_cos <= 0.0:
            return core_window, absorbed, report
        # Imported here, not at module scope: `core_fusion` drags in the V9 geometry
        # description, and the fixed-quota window -- the default, contract-carrying
        # path -- must not gain an import dependency it never uses.
        from methods.prlmc.core_fusion import rms_preserve

        alpha_max = float(self.absorb_alpha_max)
        alphas, cosines = [], []
        start = 0
        for index, kept in enumerate(kept_per_block):
            count = int(kept.numel())
            first = start
            start += count
            block_positions = offsets[index]
            if count == 0 or int(block_positions.numel()) <= count:
                continue
            # Local indices of the kept slots, so "evicted" is a mask over the block
            # instead of a set difference over positions.
            taken = torch.zeros(int(block_positions.numel()), dtype=torch.bool,
                                device=block_positions.device)
            taken[kept - block_positions[0]] = True
            evicted = block_positions[~taken]
            if int(evicted.numel()) == 0:
                continue
            similarity = (reading.key.index_select(0, evicted)
                          @ reading.key.index_select(0, kept).t())
            best, winner = similarity.max(dim=-1)
            picked = torch.nonzero(best > self.absorb_cos, as_tuple=False).flatten()
            for row in picked.tolist():
                cosine = min(float(best[row]), 1.0)
                alpha = min(alpha_max, cosine * alpha_max)
                target = first + int(winner[row])
                index_slot = torch.as_tensor([target], dtype=torch.long,
                                             device=core_window.device)
                base = core_window.index_select(1, index_slot)
                doomed = source.index_select(1, evicted[row:row + 1])
                core_window = core_window.index_copy(
                    1, index_slot, rms_preserve(base, base + alpha * doomed))
                absorbed[target] = True
                alphas.append(alpha)
                cosines.append(cosine)
                report["per_block"][index] += 1
        if alphas:
            report["absorbed_pairs"] = len(alphas)
            report["mean_alpha"] = sum(alphas) / len(alphas)
            report["mean_cos"] = sum(cosines) / len(cosines)
        return core_window, absorbed, report

    # -- the build ----------------------------------------------------------
    def build(self, source, carrier_blocks, block_size=256, window_blocks=None):
        """`source` `[layers, window_slots, D]` -> window, per-slot metadata, report."""
        window_slots = int(source.shape[1])
        blocks = window_slots // int(block_size)
        if window_slots % int(block_size):
            raise ValueError(f"{window_slots} slots do not fill {block_size}-slot blocks")
        carriers = [int(block) for block in carrier_blocks]
        if len(set(carriers)) != len(carriers):
            raise ValueError("a carrier block is listed twice")
        dropped = [block for block in range(blocks) if block not in set(carriers)]
        device = source.device
        positions = torch.arange(window_slots, device=device)
        reading, per_layer = self.scorer.reading(source, positions, self.probes)
        offsets = (torch.as_tensor(carriers, dtype=torch.long, device=device).unsqueeze(1)
                   * int(block_size) + torch.arange(int(block_size), device=device))

        if self.quota_mode == "density":
            quotas, density = self._density_quotas(reading, offsets, int(block_size))
        else:
            quotas, density = [self.core_per_block] * len(carriers), None

        core_pieces, core_origin, per_block, kept_per_block = [], [], [], []
        for index, carrier in enumerate(carriers):
            local = reading.utility.index_select(0, offsets[index])
            keep = torch.argsort(local, descending=True)[: int(quotas[index])]
            keep, _ = torch.sort(keep)
            chosen = offsets[index].index_select(0, keep)
            core_pieces.append(source.index_select(1, chosen))
            core_origin.append(chosen)
            kept_per_block.append(chosen)
            per_block.append(dict(block=carrier, slots_in=int(block_size),
                                  quota=int(quotas[index]), kept=int(chosen.numel()),
                                  evicted=int(block_size) - int(chosen.numel())))
        core_window = torch.cat(core_pieces, dim=1)
        core_origin = torch.cat(core_origin, dim=0)
        core_window, absorbed, absorb_report = self._absorb(
            core_window, kept_per_block, offsets, source, reading)

        dropped_positions = (torch.as_tensor(dropped, dtype=torch.long, device=device)
                             .unsqueeze(1) * int(block_size)
                             + torch.arange(int(block_size), device=device)).reshape(-1)
        wanted = len(carriers) * self.graft_per_block
        if wanted > int(dropped_positions.numel()):
            raise ValueError(f"{wanted} graft slots from {dropped_positions.numel()} "
                             f"candidates")
        # The novelty term is measured on the reader's keys of the SOURCE slots: the
        # absorb pass is a value-space merge that leaves the keys the reader will
        # project untouched, so key-space novelty of a candidate stays what it was.
        novelty = 1.0 - _max_cosine(reading.key.index_select(0, dropped_positions),
                                    reading.key.index_select(0, core_origin))
        utility = reading.utility.index_select(0, dropped_positions)
        span = float(utility.max() - utility.min())
        combined = (utility - utility.min()) / max(span, 1e-12)
        weight = float(self.novelty_weight)
        score = combined * (1.0 - weight + weight * novelty)
        picked = mmr_select(score, reading.key.index_select(0, dropped_positions),
                            wanted, penalty=self.mmr_penalty)
        picked, _ = torch.sort(picked)
        graft_origin = dropped_positions.index_select(0, picked)

        origin = torch.cat([core_origin, graft_origin], dim=0)
        slots = torch.cat([core_window, source.index_select(1, graft_origin)], dim=1)
        if origin.numel() != int(slots.shape[1]):
            raise ValueError("every output slot needs exactly one source position")
        if origin.numel() != int(torch.unique(origin).numel()):
            raise ValueError("two output slots claim the same source position")
        order = torch.argsort(origin, stable=True)
        window = slots.index_select(1, order)
        source_position = origin.index_select(0, order)
        # Graft slots are always verbatim; a core slot is unless `absorb_cos` merged
        # something into it, which is exactly what the relaxed assertion allows.
        verbatim = torch.cat([~absorbed,
                              torch.ones(int(graft_origin.numel()), dtype=torch.bool,
                                         device=device)]).index_select(0, order)
        # `index_select` wants longs: a bool mask works as a Python-side index but not in
        # the CUDA kernel, which is where `--block-merge-cos 0.75` first crashed.
        keep_index = torch.nonzero(verbatim, as_tuple=False).flatten()
        if not torch.equal(window.index_select(1, keep_index),
                           source.index_select(1, source_position[verbatim])):
            raise ValueError("a slot that was not absorbed is not a verbatim copy "
                             "of its source")
        slot_type = torch.cat([torch.zeros(int(core_origin.numel()), dtype=torch.long,
                                           device=device),
                               torch.ones(int(graft_origin.numel()), dtype=torch.long,
                                          device=device)]).index_select(0, order)
        output_position = torch.arange(int(window.shape[1]), device=device)
        meta = dict(source_position=source_position,
                    source_block=torch.div(source_position, int(block_size),
                                           rounding_mode="floor"),
                    slot_type=slot_type,
                    cluster_size=torch.ones(int(window.shape[1]), dtype=torch.long,
                                            device=device),
                    delta_position=(output_position - source_position).float()
                    / float(window_slots),
                    utility=reading.utility.index_select(0, source_position),
                    # position -> block of the SOURCE pool (length = pool size, not
                    # window size): diagnostics on the evicted pool need the block a
                    # source position belongs to, and `source_block` above only covers
                    # the 2,304 positions that survived into the window.
                    pool_block=torch.div(positions, int(block_size),
                                         rounding_mode="floor"),
                    # the calibration computes its cross-block context from the window
                    # the reader will actually read, so it needs the reader's scorer
                    scorer=self.scorer)
        graft_offsets = torch.nonzero(slot_type == 1, as_tuple=False).flatten()
        per_dropped = [int((meta["source_block"][slot_type == 1] == block).sum())
                       for block in dropped]
        absorbed_per_block = [int(count) for count in absorb_report["per_block"]]
        telemetry = dict(
            schema=SCHEMA, policy="evict", carrier_blocks=carriers,
            dropped_blocks=dropped, core_per_block=self.core_per_block,
            graft_per_block=self.graft_per_block, core_total=int(core_origin.numel()),
            graft_total=int(graft_origin.numel()), total=int(window.shape[1]),
            graft_share=round(100.0 * int(graft_origin.numel()) / int(window.shape[1]), 2),
            graft_output_offsets=[int(x) for x in graft_offsets.tolist()],
            graft_per_dropped_block=per_dropped,
            empty_dropped_blocks=int(sum(1 for count in per_dropped if count == 0)),
            merge=dict(policy="evict", blocks=len(carriers),
                       core_per_block=self.core_per_block,
                       core_total=int(core_origin.numel()),
                       slots_in=int(core_origin.numel()),
                       merged_pairs=int(absorb_report["absorbed_pairs"]),
                       evicted=int(core_origin.numel()) - int(
                           len(carriers) * self.core_per_block) + int(
                               len(carriers) * (int(block_size) - self.core_per_block)),
                       splits=0,
                       blocks_with_no_merge=int(sum(1 for count in absorbed_per_block
                                                    if count == 0)),
                       blocks_where_quota_came_from_eviction=len(carriers),
                       mean_cluster_size=1.0, per_block=per_block),
            merge_schema=SCHEMA, utility_schema=UTILITY_SCHEMA,
            mmr_penalty=self.mmr_penalty, novelty_weight=self.novelty_weight,
            novelty_mean=float(novelty.mean()), novelty_min=float(novelty.min()),
            utility_mean=float(reading.utility.mean()), utility_per_layer=per_layer,
            cluster_size_core_mean=1.0, cluster_size_graft=1,
            # V17 window budget: same total, and the record of how it was spent.
            quota_mode=self.quota_mode,
            per_block_quota=[int(value) for value in quotas],
            quota_total=int(sum(quotas)),
            density=None if density is None else [float(value) for value in density],
            min_core=self.min_core, max_core=self.max_core,
            density_sample=self.density_sample,
            absorb_cos=self.absorb_cos, absorb_alpha_max=self.absorb_alpha_max,
            absorbed_pairs=int(absorb_report["absorbed_pairs"]),
            absorbed_mean_alpha=float(absorb_report["mean_alpha"]),
            absorbed_mean_cos=float(absorb_report["mean_cos"]),
            absorbed_per_block=absorbed_per_block)
        return window, meta, telemetry
