"""Round 6: the V11 window -- reader-aware selection, folding, and full-pool slots.

Three modules compose here, and this file is the only place that knows the
composition:

    module 1  `reader_utility`        which slots the reader can actually read
    module 2  `homogeneous_merge`     fold same-block redundancy to pay for them
    module 3  `anchor_residual.FullPoolResidual`   calibrate all 2,304 slots

    carrier blocks   9 x 256 = 2,304 -> fold to 9 x 180 = 1,620 core
    dropped blocks  10 x 256 = 2,560 -> reader-utility + MMR -> 684 graft
                                                            2,304 slots, unchanged

Why this is a joint window and not three arms
---------------------------------------------
Each of the three was, or could be, a separate experiment, and each has an obvious
way to look like a failure on its own: selection without folding has no space to
give the dropped blocks (round 5: more space made it worse), folding without
calibration hands the reader merged vectors it has never been calibrated for, and
calibration without either has nothing new to calibrate. The composition is also
where the capacity actually comes from -- 684 graft slots exist only because the
carriers gave up 684 -- so testing them separately would test each mechanism at a
budget it does not run at.

What is held fixed
------------------
The 19-block window, the nine carriers (`gr.select_carriers`, one selection shared
by every arm), the 2,304-slot budget, the frozen reader, the chronological output
order, and the "same slot index in all 32 layers" transport: the fold assignment
and the graft selection are computed ONCE, from multi-layer scores, and applied at
every layer. `source_positions` is emitted in output order and is what every check
downstream uses to prove provenance.

Two policies
------------
`policy="merge"`  fold same-block redundancy (module 2, arm B)
`policy="evict"`  keep the top-180 by reader utility, verbatim (arm D)

Arm D exists only to attribute a V11 gain: if B and D come out the same, the extra
graft slots and the calibration did the work and the folding did not, and the paper
would have to say so.
"""
import torch

from methods.prlmc.homogeneous_merge import SCHEMA as MERGE_SCHEMA, merge_carriers
from methods.prlmc.reader_utility import SCHEMA as UTILITY_SCHEMA, mmr_select

SCHEMA = "joint_consolidation/1"
# 180 core x 9 carriers = 1,620; 76 graft x 9 = 684; 1,620 + 684 = 2,304.
QUOTA = dict(core_per_block=180, graft_per_block=76)


class JointConsolidation:
    """Builds the V11 window from a written memory pool. Holds no parameters."""

    def __init__(self, scorer, probes, core_per_block=QUOTA["core_per_block"],
                 graft_per_block=QUOTA["graft_per_block"], policy="merge",
                 mmr_penalty=0.5, novelty_weight=0.5, merge_kwargs=None):
        if policy not in ("merge", "evict"):
            raise ValueError(f"unknown policy {policy!r}")
        if core_per_block + graft_per_block != 256:
            raise ValueError("core + graft per block must fill one 256-slot block")
        self.scorer = scorer
        self.probes = list(probes)
        if not self.probes:
            raise ValueError("the reader-utility score needs at least one probe")
        self.core_per_block = int(core_per_block)
        self.graft_per_block = int(graft_per_block)
        self.policy = policy
        self.mmr_penalty = float(mmr_penalty)
        self.novelty_weight = float(novelty_weight)
        self.merge_kwargs = dict(merge_kwargs or {})

    def _graft_scores(self, reading, dropped_positions, core_positions):
        """Reader utility, discounted by how little the core already covers.

        "Novelty" here is measurable and local: `1 - max cosine` between the
        candidate's key representation and the keys of the slots the core kept. A
        candidate that duplicates a kept slot is worth less even if the reader likes
        it, which is what the multiplier in `[1 - w, 1 + w]` encodes. Both terms are
        normalised over the candidates before they meet, so neither can win on scale.
        """
        utility = reading.utility.index_select(0, dropped_positions)
        span = float(utility.max() - utility.min())
        normalised = (utility - utility.min()) / max(span, 1e-12)
        candidate_key = reading.key.index_select(0, dropped_positions)
        core_key = reading.key.index_select(0, core_positions)
        best = torch.zeros(candidate_key.shape[0], device=candidate_key.device)
        for start in range(0, core_key.shape[0], 512):
            best = torch.maximum(best, (candidate_key @ core_key[start:start + 512].t()).max(dim=-1).values)
        weight = float(self.novelty_weight)
        novelty = (1.0 - best.clamp(0.0, 1.0))
        combined = normalised * (1.0 - weight + weight * novelty)
        return combined, dict(normalised_utility=normalised, novelty=novelty)

    def build(self, source, carrier_blocks, block_size=256, window_blocks=None):
        """`source` `[layers, window_slots, D]` -> window, per-slot metadata, report."""
        layers, window_slots, _ = source.shape
        if window_slots % block_size:
            raise ValueError(f"{window_slots} slots do not fill {block_size}-slot blocks")
        blocks = window_slots // block_size
        carriers = [int(block) for block in carrier_blocks]
        carrier_set = set(carriers)
        if len(carrier_set) != len(carriers):
            raise ValueError("a carrier block is listed twice")
        dropped = [block for block in range(blocks) if block not in carrier_set]
        device = source.device
        positions = torch.arange(window_slots, device=device)
        reading, per_layer = self.scorer.reading(source, positions, self.probes)

        carrier_offsets = torch.as_tensor(carriers, dtype=torch.long,
                                          device=device).unsqueeze(1) * block_size
        carrier_offsets = carrier_offsets + torch.arange(block_size, device=device)
        core_window, cluster_size, core_origin, merge_summary = merge_carriers(
            source, carrier_offsets, reading.key, reading.value, reading.utility,
            self.core_per_block, policy=self.policy, **self.merge_kwargs)
        core_total = int(core_window.shape[1])

        dropped_offsets = torch.as_tensor(dropped, dtype=torch.long,
                                          device=device).unsqueeze(1) * block_size
        dropped_offsets = dropped_offsets + torch.arange(block_size, device=device)
        dropped_positions = dropped_offsets.reshape(-1)
        wanted = len(carriers) * self.graft_per_block
        if wanted > dropped_positions.numel():
            raise ValueError(f"{wanted} graft slots from {dropped_positions.numel()} "
                             f"candidates")
        score, novelty = self._graft_scores(reading, dropped_positions, core_origin)
        picked_local = mmr_select(score, reading.key.index_select(0, dropped_positions),
                                  wanted, penalty=self.mmr_penalty)
        picked_local, _ = torch.sort(picked_local)
        graft_positions = dropped_positions.index_select(0, picked_local)
        graft_window = source.index_select(1, graft_positions)
        graft_cluster = torch.ones(wanted, dtype=torch.long, device=device)

        all_slots = torch.cat([core_window, graft_window], dim=1)
        all_origin = torch.cat([core_origin, graft_positions], dim=0)
        all_cluster = torch.cat([cluster_size, graft_cluster], dim=0)
        all_type = torch.cat([torch.zeros(core_total, dtype=torch.long, device=device),
                              torch.ones(wanted, dtype=torch.long, device=device)], dim=0)
        if all_slots.shape[1] != core_total + wanted:
            raise ValueError("the assembled window does not match its parts")
        if int(all_origin.numel()) != all_slots.shape[1]:
            raise ValueError("every output slot needs exactly one source position")
        if all_origin.numel() != int(torch.unique(all_origin).numel()):
            raise ValueError("two output slots claim the same source position")

        # Chronological: the same rule as the V10 operator, over window positions.
        order = torch.argsort(all_origin, stable=True)
        window = all_slots.index_select(1, order)
        source_position = all_origin.index_select(0, order)
        slot_type = all_type.index_select(0, order)
        slot_cluster = all_cluster.index_select(0, order)
        output_position = torch.arange(int(window.shape[1]), device=device)
        delta_position = ((output_position - source_position).float()
                          / float(window_slots))

        meta = dict(
            source_position=source_position,
            source_block=torch.div(source_position, block_size, rounding_mode="floor"),
            slot_type=slot_type, cluster_size=slot_cluster,
            delta_position=delta_position,
            utility=reading.utility.index_select(0, source_position))
        graft_offsets = torch.nonzero(slot_type == 1, as_tuple=False).flatten()
        per_block_graft = [int((meta["source_block"][graft_offsets] == block).sum())
                           for block in dropped]
        telemetry = dict(
            schema=SCHEMA, policy=self.policy,
            carrier_blocks=carriers, dropped_blocks=dropped,
            core_per_block=self.core_per_block,
            graft_per_block=self.graft_per_block,
            core_total=core_total, graft_total=int(wanted),
            total=int(window.shape[1]),
            graft_share=round(100.0 * wanted / int(window.shape[1]), 2),
            graft_output_offsets=[int(x) for x in graft_offsets.tolist()],
            graft_per_dropped_block=per_block_graft,
            empty_dropped_blocks=int(sum(1 for count in per_block_graft if count == 0)),
            merge=merge_summary, merge_schema=MERGE_SCHEMA, utility_schema=UTILITY_SCHEMA,
            mmr_penalty=self.mmr_penalty, novelty_weight=self.novelty_weight,
            novelty_mean=float(novelty["novelty"].mean()),
            novelty_min=float(novelty["novelty"].min()),
            utility_mean=float(reading.utility.mean()),
            utility_per_layer=per_layer,
            cluster_size_core_mean=float(cluster_size.float().mean()),
            cluster_size_graft=1)
        return window, meta, telemetry


def conditions(meta, window_slots):
    """The three conditioning features module 3 is given, `[slots, 3]`.

    `delta_position`  where the slot came from vs where it now sits, scaled by the
                      window so it lives in (-1, 1) for every geometry
    `log_cluster`     `log |C_i|`: 0 for a slot that stands for itself, > 0 for a
                      folded slot. This is how the calibration can tell a verbatim
                      vector from a blend without being told which arm it is in.
    `is_graft`        1 for a dropped-block slot, 0 for a carrier slot.
    """
    cluster = meta["cluster_size"].float()
    return torch.stack([
        meta["delta_position"].float().clamp(-1.0, 1.0),
        (torch.log(cluster) / 1.5).clamp(0.0, 1.0),
        meta["slot_type"].float(),
    ], dim=-1).to(torch.float32)


def calibrate(window, meta, module, layer_chunk=8):
    """Module 3 over ALL 2,304 slots: bounded residual, RMS restored per layer.

    Layer-chunked for the same reason as everywhere else in this project: the
    module is shared across layers, so the chunk size cannot change the result; it
    only bounds the float32 working set that has to stay alive for the backward.
    """
    from methods.prlmc.core_fusion import apply_delta

    cond = conditions(meta, window.shape[1])
    if cond.shape[0] != window.shape[1]:
        raise ValueError("one condition row per slot")
    pieces = []
    delta_sq = slot_sq = 0.0
    per_type = {0: [0.0, 0.0], 1: [0.0, 0.0]}
    for start in range(0, window.shape[0], layer_chunk):
        stop = min(start + layer_chunk, window.shape[0])
        chunk = window[start:stop]
        delta = module(chunk, cond)
        calibrated = apply_delta(chunk, delta)
        pieces.append(calibrated)
        weight = delta.detach().float().square()
        delta_sq += float(weight.sum())
        slot_sq += float(chunk.detach().float().square().sum())
        for kind in (0, 1):
            mask = (meta["slot_type"] == kind)
            if bool(mask.any()):
                per_type[kind][0] += float(weight[:, mask].sum())
                per_type[kind][1] += float(chunk.detach().float()[:, mask].square().sum())
        del delta
    info = dict(
        slots=int(window.shape[1]),
        core_slots=int((meta["slot_type"] == 0).sum()),
        graft_slots=int((meta["slot_type"] == 1).sum()),
        folded_slots=int((meta["cluster_size"] > 1).sum()),
        relative_delta=(delta_sq / max(slot_sq, 1e-24)) ** 0.5,
        core_relative_delta=(per_type[0][0] / max(per_type[0][1], 1e-24)) ** 0.5,
        graft_relative_delta=(per_type[1][0] / max(per_type[1][1], 1e-24)) ** 0.5,
        gate=float(torch.sigmoid(module.gate_logit.detach().float()))
        if hasattr(module, "gate_logit") else None)
    return torch.cat(pieces, dim=0), info
