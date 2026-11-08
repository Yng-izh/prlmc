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
"""
import torch

from methods.prlmc.reader_utility import SCHEMA as UTILITY_SCHEMA, mmr_select

SCHEMA = "evict_window/1"
# 180 x 9 = 1,620 core, 76 x 9 = 684 graft, 2,304 in total. V11's quota, unchanged.
QUOTA = dict(core_per_block=180, graft_per_block=76)


def _max_cosine(candidate_key, reference_key, chunk=512):
    """`max` cosine from each candidate row to a reference set; both sides unit rows."""
    best = torch.full((int(candidate_key.shape[0]),), -1.0,
                      dtype=candidate_key.dtype, device=candidate_key.device)
    for start in range(0, int(reference_key.shape[0]), chunk):
        block = candidate_key @ reference_key[start:start + chunk].t()
        best = torch.maximum(best, block.max(dim=-1).values)
    return best.clamp(-1.0, 1.0)


class JointConsolidation:
    """Builds the V11Evict window. Holds no parameters: selection only."""

    def __init__(self, scorer, probes, core_per_block=QUOTA["core_per_block"],
                 graft_per_block=QUOTA["graft_per_block"], policy="evict",
                 mmr_penalty=0.5, novelty_weight=0.5, merge_kwargs=None):
        if policy != "evict":
            raise ValueError(f"V13 runs the eviction policy only; got {policy!r}")
        if core_per_block + graft_per_block != 256:
            raise ValueError("core + graft per block must fill one 256-slot block")
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

        core_pieces, core_origin, per_block = [], [], []
        for index, carrier in enumerate(carriers):
            local = reading.utility.index_select(0, offsets[index])
            keep = torch.argsort(local, descending=True)[: self.core_per_block]
            keep, _ = torch.sort(keep)
            chosen = offsets[index].index_select(0, keep)
            core_pieces.append(source.index_select(1, chosen))
            core_origin.append(chosen)
            per_block.append(dict(block=carrier, slots_in=int(block_size),
                                  quota=self.core_per_block, kept=int(chosen.numel()),
                                  evicted=int(block_size) - int(chosen.numel())))
        core_window = torch.cat(core_pieces, dim=1)
        core_origin = torch.cat(core_origin, dim=0)

        dropped_positions = (torch.as_tensor(dropped, dtype=torch.long, device=device)
                             .unsqueeze(1) * int(block_size)
                             + torch.arange(int(block_size), device=device)).reshape(-1)
        wanted = len(carriers) * self.graft_per_block
        if wanted > int(dropped_positions.numel()):
            raise ValueError(f"{wanted} graft slots from {dropped_positions.numel()} "
                             f"candidates")
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
        if not torch.equal(window, source.index_select(1, source_position)):
            raise ValueError("an output slot is not a verbatim copy of its source")
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
                    # the calibration computes its cross-block context from the window
                    # the reader will actually read, so it needs the reader's scorer
                    scorer=self.scorer)
        graft_offsets = torch.nonzero(slot_type == 1, as_tuple=False).flatten()
        per_dropped = [int((meta["source_block"][slot_type == 1] == block).sum())
                       for block in dropped]
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
                       merged_pairs=0, evicted=int(core_origin.numel()) - int(
                           len(carriers) * self.core_per_block) + int(
                               len(carriers) * (int(block_size) - self.core_per_block)),
                       splits=0, blocks_with_no_merge=len(carriers),
                       blocks_where_quota_came_from_eviction=len(carriers),
                       mean_cluster_size=1.0, per_block=per_block),
            merge_schema=SCHEMA, utility_schema=UTILITY_SCHEMA,
            mmr_penalty=self.mmr_penalty, novelty_weight=self.novelty_weight,
            novelty_mean=float(novelty.mean()), novelty_min=float(novelty.min()),
            utility_mean=float(reading.utility.mean()), utility_per_layer=per_layer,
            cluster_size_core_mean=1.0, cluster_size_graft=1)
        return window, meta, telemetry
