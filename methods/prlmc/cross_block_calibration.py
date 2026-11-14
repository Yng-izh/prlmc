"""V13/V14: cross-block conditioned calibration of every slot in the 2,304-slot window.

What is new relative to V11Evict
--------------------------------
V11's calibration saw each slot alone. Its input was the slot's own vector plus a
three-number condition (position drift, cluster size, slot type), so it could only ask
"how should this vector change to be easier to read WHERE IT NOW SITS". It could not
ask the question the mechanism needs: *what does this slot relate to in the rest of the
memory?* A graft slot that carries a person's birth city and a core slot that carries
the person's identity are useless apart and useful together, and nothing in V11 let the
residual see that.

V13 adds exactly one capability: before a slot is modified, the calibrator is shown a
SMALL, SPARSE set of slots from OTHER blocks that it is most associated with.

    association   s_ij = <key_i, key_j> over the reader's own multi-layer keys
                  (unit rows, computed on the window the reader will read)
    sparsity      same-block neighbours are masked out (the mechanism is cross-block),
                  the top `k = 4` cross-block neighbours survive, weights are a softmax
                  over those k only
    context       c_i = sum_j w_ij * value_j   -- the neighbours' CONTENT, not an average
                  of the slot with them. Nothing is blended into the slot; c_i is only
                  an input to the residual, and the slot itself is moved by the same
                  bounded low-rank map V11 used.

That last sentence is the difference between this and the merges that failed: the
vector is never replaced by a mixture, so the frozen reader never receives a slot that
is a compromise between two facts.

Asymmetric constraint, soft on both sides
-----------------------------------------
Experiment 4 killed the key channel (50/158) and experiment 5 showed the content mostly
travels through it, so locking keys is not an option either. The constraint here is
therefore a pair of different CEILINGS, not a lock:

    core (carrier) slots   |delta| <= max_ratio_core  x rms      (0.15: protect them)
    graft slots            |delta| <= max_ratio_graft x rms      (0.50: let them move)

and the trainer's residual regulariser weights the two halves differently
(`core_weight = 2.0`, `graft_weight = 0.5`). Both are soft: a core slot can still move
up to 15% of its own RMS, which is more than the 0 change a hard key lock allowed.

V14: where the neighbours come from
-----------------------------------
V13 asks every slot for its k most associated cross-block slots and does not care what
kind they are. That is a gap, and it is measurable: the report records, per row,
`neighbour_type_cross_fraction` -- the share of the selected neighbours whose core/graft
kind differs from the query's. The 684 graft slots are the only slots that carry content
the carriers lost, so a carrier whose top-4 neighbours are all carriers is calibrated
against retained knowledge only, and the new knowledge it has to be read alongside never
reaches its context.

`mode="complement"` therefore builds the neighbour set from BOTH sides of that divide:

    k_same  = 2 slots of the query's own kind
    k_cross = 2 slots from the other side (a carrier sees graft slots, graft sees carriers)
    same-block is still masked, and the total is still k = 4

The four weights are one softmax PER GROUP, with the group masses split evenly
(k_same/k and k_cross/k). Even splitting is deliberate rather than a detail: a single
softmax over the four would let the own-kind similarities, which are systematically the
higher ones, starve the complement side towards zero weight, and the arm would silently
degenerate into "top-2 of my own kind". `cross_weight_mass_mean` in the context report
says whether that happened. Whether the complement side then BUYS anything is the arm's
question, not this file's: V14 must beat V13 (67/158) by a margin worth the change, and
must still lose to its own donor arm.

Nothing else changes between V13 and V14 -- same window, same budget, same 1,620 / 684
split, same rank, ceilings, regulariser, loss, updates and cross learning rate. The only
variable is which slots the context is built from.

What is NOT claimed here
------------------------
The sparse association is a proxy for "these two slots belong to the same fact". It is
measured in the reader's key space, which experiment 5 showed is the channel that
carries content AND the channel that breaks existing answers; whether conditioning on
it helps is the thing the arm has to show, not something this file may assume.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from methods.prlmc.core_fusion import apply_delta
from methods.prlmc.reader_utility import UTILITY_LAYERS

SCHEMA = "cross_block_residual/1"
DEFAULT_KV_DIM = 5 * 8 * 128  # 5 utility layers x 8 kv heads x 128
# "topk"       V13: the k most associated cross-block slots, kind-blind
# "complement" V14: k_same of the query's own kind + k_cross from the other kind
NEIGHBOUR_MODES = ("topk", "complement")
# HOW the cross-block neighbours are picked, independently of how they are weighted:
#   topk    the k most similar (V13's design)
#   random  k uniformly random legal cross-block slots -- same count, same value pool,
#           same projection, so a drop cannot be blamed on the branch being bigger
#   bottom  the k LEAST similar legal cross-block slots -- the mirror image of topk
NEIGHBOUR_SELECTION = ("topk", "random", "bottom")


def conditions(meta, window_slots):
    """The three conditioning features V11 already used, `[slots, 3]`.

    `delta_position`  where the slot came from vs where it now sits
    `log_cluster`     log |C_i| -- always 0 here (nothing is folded any more), kept so
                      a checkpoint trained with folding stays loadable
    `is_graft`        1 for a dropped-block slot, 0 for a carrier slot
    """
    cluster = meta["cluster_size"].float()
    return torch.stack([
        meta["delta_position"].float().clamp(-1.0, 1.0),
        (torch.log(cluster) / 1.5).clamp(0.0, 1.0),
        meta["slot_type"].float(),
    ], dim=-1).to(torch.float32)


def _neighbour_groups(mode, k, k_same, k_cross, slots):
    """`[(kind, how many)]` and the per-group weight masses; `kind` is None for topk."""
    if mode not in NEIGHBOUR_MODES:
        raise ValueError(f"unknown neighbour mode {mode!r}: expected one of "
                         f"{NEIGHBOUR_MODES}")
    if mode == "topk":
        wanted = [(None, int(k))]
    else:
        if int(k_same) < 0 or int(k_cross) < 0:
            raise ValueError("k_same and k_cross must not be negative")
        if int(k_same) + int(k_cross) != int(k):
            raise ValueError(f"complement neighbours read k = k_same + k_cross, but "
                             f"{k_same} + {k_cross} != {k}")
        wanted = [("same", int(k_same)), ("cross", int(k_cross))]
    wanted = [(kind, count) for kind, count in wanted if count > 0]
    if not wanted:
        raise ValueError("the neighbour set is empty: k must be positive")
    limited = [(kind, min(count, slots - 1)) for kind, count in wanted]
    if any(count <= 0 for _, count in limited):
        raise ValueError("the window is too short for a cross-block neighbour set")
    total = sum(count for _, count in limited)
    return limited, [count / total for _, count in limited]


def _span_smooth(values, block, mid=0.5, side=0.25):
    """Local span smoothing along the slot axis, never across a source-block border.

    `v~_j = mid*v_j + side*v_{j-1} + side*v_{j+1}`, where a neighbour that falls in a
    DIFFERENT source block is dropped and its weight redistributed to the centre, so the
    row always sums to one. Zero parameters.
    """
    if values.shape[0] < 2:
        return values
    same = block[1:] == block[:-1]
    zero = torch.zeros(1, dtype=torch.bool, device=values.device)
    prev_ok = torch.cat([zero, same]).to(values.dtype)
    next_ok = torch.cat([same, zero]).to(values.dtype)
    prev = torch.cat([torch.zeros_like(values[:1]), values[:-1]])
    nxt = torch.cat([values[1:], torch.zeros_like(values[:1])])
    numerator = (mid * values
                 + side * prev * prev_ok.unsqueeze(-1)
                 + side * nxt * next_ok.unsqueeze(-1))
    denominator = mid + side * prev_ok + side * next_ok
    return numerator / denominator.unsqueeze(-1)


@torch.no_grad()
def cross_block_context(window, meta, k=4, chunk=256, return_report=False,
                        mode="topk", k_same=2, k_cross=2, selection="topk", seed=0,
                        span_value=False, context_diffusion=False, csls=False,
                        sim_floor=0.0):
    """Sparse cross-block neighbours of every slot, `[slots, key_dim]` float32.

    Keys and values come from the reader's own projections on the window it will read
    (`ReaderScorer.reading(..., probes=None)`), so the association is measured in the
    space the reader uses, at the positions the slots will actually occupy.

    `mode="topk"` (V13): the k most associated slots from other blocks, whatever kind
    they are. `mode="complement"` (V14): k_same slots of the query's own core/graft kind
    plus k_cross from the other side, so every slot is calibrated against both retained
    knowledge and the new knowledge it has to be read alongside. Each group is softmaxed
    separately and the group masses are equal, so the complement side cannot be starved
    by the systematically higher own-kind similarities.

    `selection` picks WHICH k cross-block slots become the neighbours, independently of
    how they are weighted: `topk` the most similar (V13), `random` k uniformly random
    legal cross-block slots, `bottom` the k least similar. The last two are the
    attribution test for "the gain comes from finding the RIGHT neighbours": the count,
    the value pool, the projection and the window are identical, only the choice changes.

    Three optional, parameter-free refinements of the SAME machinery:

    `span_value`       before aggregation, smooth each neighbour's VALUE over its own
                       micro-span (v~ = 0.5 v_j + 0.25 v_{j-1} + 0.25 v_{j+1}, same block
                       only). A neighbour slot matched through a bridging entity usually
                       sits next to the predicate/object slots that carry the answer, and
                       those neighbours' keys do not look like the query -- so the top-4
                       can never reach them directly. This pulls them in through the
                       one neighbour that did match.
    `context_diffusion` after aggregation, smooth the CONTEXT over same-block neighbours
                       (c = 0.70 c_i + 0.15 c_{i-1} + 0.15 c_{i+1}), so the slots around
                       a bridging entity share the cross-block alignment it established.
    `csls`             before selection, subtract the bidirectional background
                       (`S_hat = S - 0.5 (mean_i_cross + mean_j_cross)`). Hub slots that
                       are similar to everything lose their Top-4 seat; a mutual, unique
                       match survives. The means are taken over LEGAL cross-block slots
                       only, so the -2.0 mask never enters the statistic.

    `sim_floor` (V17) is an ABSOLUTE lower bound a candidate must reach to be allowed
    into the Top-k. Below it, a neighbour is not a neighbour: it is the best of a bad
    set, and averaging it into `c_i` dilutes the row with unrelated content at full
    softmax weight. Below-floor candidates are pushed to `-inf` before the sort, so they
    can only be selected when the row has nothing better and then contribute exactly
    zero weight. A row whose ENTIRE legal pool is below the floor keeps its unfloored
    scores -- otherwise it would pick `-inf` slots and the softmax would be NaN, which
    is the one failure this fallback exists to prevent; the report counts how often that
    happened. The floor is only defined for `selection="topk"`: it chooses among
    candidates, it does not reorder a random or bottom draw, and accepting it there
    while ignoring it would look exactly like a floor that fires and does nothing.

    Nothing is averaged into the slot: the returned context is an extra input to the
    residual. Same-block neighbours are masked, so every context row is built from
    other blocks by construction -- that mask is what makes this cross-block at all.
    """
    if selection not in NEIGHBOUR_SELECTION:
        raise ValueError(f"unknown neighbour selection {selection!r}: expected one of "
                         f"{NEIGHBOUR_SELECTION}")
    sim_floor = float(sim_floor)
    if not math.isfinite(sim_floor):
        raise ValueError("the neighbour similarity floor must be finite")
    if sim_floor > 0.0:
        if selection != "topk":
            raise ValueError(f"the similarity floor selects among candidates and only "
                             f"applies to selection='topk', not {selection!r}")
        if csls:
            # The floor is an absolute similarity; csls re-centres the similarity per
            # row, so the same number would mean a different thing per row.
            raise ValueError("the similarity floor is absolute and csls re-centres the "
                             "similarity: enabling both would threshold different "
                             "quantities under one name")
    groups, masses = _neighbour_groups(mode, k, k_same, k_cross, int(window.shape[1]))
    scorer = meta.get("scorer")
    if scorer is None:
        raise ValueError("the metadata carries no reader scorer: the cross-block context "
                         "cannot be computed")
    slots = int(window.shape[1])
    reading, _ = scorer.reading(window, torch.arange(slots, device=window.device), None)
    key, value = reading.key.float(), reading.value.float()
    block = meta["source_block"]
    kind = (meta["slot_type"] > 0.5).to(torch.bool)  # True carries new content (graft)
    rows, used_similarity, distinct = [], [], []
    similarity_out, weight_cross, graft_neighbour, query_graft = [], [], [], []
    values = _span_smooth(value, block, 0.5, 0.25) if span_value else value
    csls_report = None
    floor_below = 0.0      # legal candidates that failed the floor, over all rows
    floor_legal = 0.0      # legal candidates, the denominator of the fraction above
    floor_fallback = 0     # rows whose whole legal pool was below the floor

    def similarity_chunks():
        for start in range(0, slots, chunk):
            stop = min(start + chunk, slots)
            raw = key[start:stop] @ key.t()
            same = block[start:stop].unsqueeze(1) == block.unsqueeze(0)
            yield start, stop, raw.masked_fill(same, -2.0), same

    if csls:
        # Two passes, but the similarity of a 2,304-slot window is 21 MiB -- keeping it
        # is cheaper than recomputing, and the column statistic must see every row.
        materialised = list(similarity_chunks())
        col_sum = torch.zeros(slots, device=window.device)
        col_count = torch.zeros(slots, device=window.device)
        for _, _, similarity, same in materialised:
            legal = ~same
            col_sum += torch.where(legal, similarity,
                                   torch.zeros_like(similarity)).sum(dim=0)
            col_count += legal.sum(dim=0)
        col_mean = col_sum / col_count.clamp_min(1.0)
        source = iter(materialised)
        csls_report = dict(column_mean_mean=float(col_mean.mean()),
                           column_mean_std=float(col_mean.std()))
    else:
        source = similarity_chunks()
    for start, stop, similarity, same in source:
        row_mean = None
        if csls:
            legal = similarity > -1.5
            row_mean = (torch.where(legal, similarity, torch.zeros_like(similarity))
                        .sum(dim=-1) / legal.sum(dim=-1).clamp_min(1.0))
            similarity = (similarity - 0.5 * (row_mean.unsqueeze(-1)
                                              + col_mean.unsqueeze(0)))
            similarity = similarity.masked_fill(~legal, -2.0)
            csls_report.setdefault("row_mean_mean", [])
            csls_report["row_mean_mean"].append(float(row_mean.mean()))
        same_kind = kind[start:stop].unsqueeze(1) == kind.unsqueeze(0)
        picks, scales = [], []
        for (name, count), mass in zip(groups, masses):
            masked = similarity
            if name == "same":
                masked = similarity.masked_fill(~same_kind, -2.0)
            elif name == "cross":
                masked = similarity.masked_fill(same_kind, -2.0)
            if sim_floor > 0.0:
                # `-2.0` is the same-block/kind mask and never a real similarity, so
                # legality is read off it with the threshold the other selections use.
                legal = masked > -1.5
                below = legal & (masked < sim_floor)
                cleaned = masked.masked_fill(below, float("-inf"))
                survives = (cleaned > float("-inf")).any(dim=-1, keepdim=True)
                masked = torch.where(survives, cleaned, masked)
                floor_below += float(below.sum())
                floor_legal += float(legal.sum())
                floor_fallback += int((~survives).sum())
            if selection == "topk":
                index_k = masked.topk(count, dim=-1).indices
            elif selection == "bottom":
                # `masked` carries -2.0 on same-block slots, which the smallest-first
                # sort would happily return; the pool is rebuilt with +inf there.
                index_k = (masked.masked_fill(masked <= -1.5, float("inf"))
                           .topk(count, dim=-1, largest=False).indices)
            else:
                generator = torch.Generator(device="cpu").manual_seed(
                    int(seed) * 1000 + int(start) + count)
                draw = torch.rand(masked.shape, generator=generator).to(masked.device)
                index_k = draw.masked_fill(masked <= -1.5, -1.0).topk(count, dim=-1).indices
            # The weights always come from the TRUE similarity of the slots actually
            # chosen, so a random neighbour is weighted like the weak neighbour it is.
            picked_values = masked.gather(1, index_k)
            picks.append(index_k)
            scales.append(torch.softmax(picked_values, dim=-1) * mass)
            if sim_floor > 0.0:
                # A floored seat carries -inf and weight 0. Reporting its raw value
                # would turn the "how similar were the neighbours" statistic into -inf,
                # so the mean is taken over the seats that actually contributed.
                finite = torch.isfinite(picked_values)
                used_similarity.append(
                    torch.where(finite, picked_values, torch.zeros_like(picked_values))
                    .sum(dim=-1) / finite.sum(dim=-1).clamp_min(1))
            else:
                used_similarity.append(picked_values.mean(dim=-1))
        index = torch.cat(picks, dim=-1)                # [rows, K]
        weights = torch.cat(scales, dim=-1)             # [rows, K], one softmax per group
        gathered = values.index_select(0, index.reshape(-1)).reshape(
            stop - start, index.shape[-1], -1)
        rows.append((gathered * weights.unsqueeze(-1)).sum(dim=1))
        distinct.append(block.index_select(0, index.reshape(-1))
                        .reshape(stop - start, index.shape[-1]).float().std(dim=-1))
        # Composition of the neighbour set, which is the number V13 never reported:
        # how much of the context comes from the other side of the core/graft divide.
        neighbour_graft = kind.index_select(0, index.reshape(-1)).reshape(index.shape)
        graft_neighbour.append(neighbour_graft.float().mean(dim=-1))
        query_graft.append(kind[start:stop])
        cross = (neighbour_graft != kind[start:stop].unsqueeze(1)).float()
        similarity_out.append(cross.mean(dim=-1))
        weight_cross.append((weights * cross).sum(dim=-1))
        del similarity, same, same_kind, picks, scales, masked, index, weights, gathered
    context = torch.cat(rows, dim=0)
    context_raw_rms = float(context.square().mean().sqrt())
    if context_diffusion:
        context = _span_smooth(context, block, 0.70, 0.15)
    if not return_report:
        return context
    used = torch.cat(used_similarity, dim=0)
    cross_each = torch.cat(similarity_out, dim=0)
    cross_mass = torch.cat(weight_cross, dim=0)
    graft_neighbour = torch.cat(graft_neighbour, dim=0)
    is_graft = torch.cat(query_graft, dim=0)

    def _mean(values):
        return float(values.mean()) if int(values.numel()) else None

    if isinstance(csls_report, dict) and isinstance(csls_report.get("row_mean_mean"), list):
        csls_report["row_mean_mean"] = float(sum(csls_report["row_mean_mean"])
                                             / max(len(csls_report["row_mean_mean"]), 1))
    report = dict(mode=mode, selection=selection, seed=int(seed),
                  span_value=bool(span_value), context_diffusion=bool(context_diffusion),
                  csls=bool(csls), csls_statistics=csls_report,
                  context_rms_before_diffusion=context_raw_rms,
                  k=sum(count for _, count in groups), key_dim=int(key.shape[-1]),
                  k_same=groups[0][1] if mode == "complement" else None,
                  k_cross=groups[-1][1] if mode == "complement" else None,
                  neighbour_similarity_mean=_mean(used),
                  neighbour_similarity_min=float(used.min()),
                  neighbour_block_spread_mean=float(torch.cat(distinct, dim=0).mean()),
                  context_rms=float(context.square().mean().sqrt()),
                  neighbour_graft_fraction=_mean(graft_neighbour),
                  neighbour_type_cross_fraction=_mean(cross_each),
                  core_to_graft_neighbour_fraction=_mean(cross_each[~is_graft]),
                  graft_to_core_neighbour_fraction=_mean(cross_each[is_graft]),
                  cross_weight_mass_mean=_mean(cross_mass),
                  own_kind_weight_mass_mean=_mean(1.0 - cross_mass),
                  # The distribution `sim_floor` is a point on. A floor chosen without
                  # it is a guess: with a mean of 0.66 a floor of 0.30 removes almost
                  # nothing, and the arm would look like "the filter does not help"
                  # when in fact it never filtered. Quantiles over the SELECTED
                  # (top-k) similarities, which is the population a floor can act on.
                  neighbour_similarity_quantiles=dict(
                      p01=float(torch.quantile(used, 0.01)),
                      p10=float(torch.quantile(used, 0.10)),
                      p50=float(torch.quantile(used, 0.50)),
                      p90=float(torch.quantile(used, 0.90))))
    if sim_floor > 0.0:
        # Added exactly when the floor is on, so a V13 report is the same dict it was:
        # a `below_floor_fraction` of 0.0 at a floor of 0.0 would read as "the floor was
        # checked and nothing fell through", which is a different claim.
        report["sim_floor"] = sim_floor
        report["below_floor_fraction"] = floor_below / max(floor_legal, 1.0)
        report["floor_fallback_rows"] = int(floor_fallback)
    return context, report


def evicted_positions(meta, source_slots):
    """Source positions that did NOT make it into the 2,304-slot window."""
    kept = meta["source_position"].to(torch.long)
    mask = torch.ones(int(source_slots), dtype=torch.bool, device=kept.device)
    mask[kept] = False
    return torch.nonzero(mask, as_tuple=False).flatten()


@torch.no_grad()
def evicted_pool_context(window, meta, source, k=4, chunk=256, return_report=False):
    """Second cross-block path: the neighbours of each kept slot inside the EVICTED pool.

    The window keeps 2,304 of the 4,864 source slots, so 2,560 are dropped -- 76 per
    carrier block and the ~188 unselected slots of every dropped block. MMR keeps the
    FIRST slot of a repeated span, which is why the predicate/object slots immediately
    next to a kept one are routinely in the pool that was thrown away. This path reaches
    them with the same statistic the in-pool path uses (Reader 5-layer key cosine, top-4,
    their Reader values), with one deliberate difference: **no same-block mask**. The
    slots we want are usually the kept slot's own block neighbours, and masking them out
    is exactly what lost them.

    Cost: two extra reader readings (the window for the query side, the evicted pool for
    the candidate side). The in-pool context is untouched -- it is still V13's.
    """
    scorer = meta.get("scorer")
    if scorer is None:
        raise ValueError("the metadata carries no reader scorer")
    slots = int(window.shape[1])
    positions = evicted_positions(meta, int(source.shape[1]))
    if int(positions.numel()) < 2:
        raise ValueError("the evicted pool is empty: nothing to recover")
    evicted = source.index_select(1, positions)
    query_reading, _ = scorer.reading(window, torch.arange(slots, device=window.device),
                                      None)
    pool_reading, _ = scorer.reading(evicted, torch.arange(evicted.shape[1],
                                                          device=evicted.device), None)
    query_key = query_reading.key.float()
    pool_key = pool_reading.key.float()
    pool_value = pool_reading.value.float()
    if int(query_key.shape[-1]) != int(pool_key.shape[-1]):
        raise ValueError("the kept and evicted readings have different widths")
    keep = min(int(k), int(evicted.shape[1]) - 1)
    rows, similarity_rows = [], []
    for start in range(0, slots, chunk):
        stop = min(start + chunk, slots)
        similarity = query_key[start:stop] @ pool_key.t()
        best = similarity.topk(keep, dim=-1)
        weights = torch.softmax(best.values, dim=-1)
        gathered = pool_value.index_select(0, best.indices.reshape(-1)).reshape(
            stop - start, keep, -1)
        rows.append((gathered * weights.unsqueeze(-1)).sum(dim=1))
        similarity_rows.append(best.values.mean(dim=-1))
        del similarity, best, gathered, weights
    context = torch.cat(rows, dim=0)
    if not return_report:
        return context
    report = dict(k=int(keep), pool_slots=int(positions.numel()),
                  neighbour_similarity_mean=float(torch.cat(similarity_rows).mean()),
                  context_rms=float(context.square().mean().sqrt()),
                  note="unmasked pool: the point of this path is its own-block neighbours. "
                       "The block identity of a pool slot is NOT reported: `source` is not "
                       "ordered by block (select_carriers concatenates the carriers "
                       "first), so the position -> block map would have to come from the "
                       "window builder, not from arithmetic on source_position.")
    return context, report


@torch.no_grad()
def slot_relevance(window, meta, k=4, chunk=256):
    """Mean cosine similarity of every slot to its own cross-block Top-4 neighbours.

    Exactly the statistic the calibration uses to pick those neighbours (same keys, same
    same-block mask, same top-k), returned per slot so an inference-time cap can be made
    proportional to it. No parameters, no extra state.
    """
    scorer = meta.get("scorer")
    if scorer is None:
        raise ValueError("the metadata carries no reader scorer")
    slots = int(window.shape[1])
    reading, _ = scorer.reading(window, torch.arange(slots, device=window.device), None)
    key = reading.key.float()
    block = meta["source_block"]
    keep = min(int(k), slots - 1)
    out = []
    for start in range(0, slots, chunk):
        stop = min(start + chunk, slots)
        similarity = key[start:stop] @ key.t()
        same = block[start:stop].unsqueeze(1) == block.unsqueeze(0)
        similarity = similarity.masked_fill(same, -2.0)
        out.append(similarity.topk(keep, dim=-1).values.mean(dim=-1))
        del similarity, same
    return torch.cat(out, dim=0)


def relevance_graft_caps(relevance, slot_type, low=0.25, high=0.50):
    """`cap_i = low + (high - low) * rank_norm(s_i)` for the graft slots, `[slots]`.

    Rank-proportional, not value-proportional: the ranking is what says "this graft slot
    is a real bridge" versus "this one came along to fill the quota", and it is scale
    free. Carrier slots carry the carrier ceiling and are passed through untouched.
    """
    caps = torch.full_like(relevance, float(high))
    graft = slot_type > 0.5
    count = int(graft.sum())
    if count == 0:
        return caps
    values = relevance[graft].float()
    order = values.argsort().argsort().float()
    normalised = (order + 0.5) / float(count)
    caps[graft] = (float(low) + (float(high) - float(low)) * normalised).to(caps.dtype)
    return caps


class CrossBlockResidual(nn.Module):
    """Bounded low-rank residual for every slot, conditioned on its cross-block context.

    V11's `FullPoolResidual` plus one term: `cross_proj` maps the sparse neighbour
    context into the bottleneck and is zero-initialised, so at update 0 the module is
    EXACTLY V11's calibrator and the identity on the core half. That equality is a
    construction guarantee, not an empirical claim: at step 0 nothing has moved.

    The two halves keep separate output maps and separate residual ceilings, which is
    the soft version of "protect the carriers, free the graft slots".
    """

    def __init__(self, hidden_size=4096, rank=64, max_ratio=0.5, gate_init=0.0,
                 cond_size=3, cond_rank=32, use_cross=True, k_neighbors=4,
                 neighbor_mode="topk", k_same=2, k_cross=2,
                 kv_dim=DEFAULT_KV_DIM, max_ratio_core=0.15, core_weight=2.0,
                 graft_weight=0.5, span_value=False, context_diffusion=False, csls=False,
                 use_evicted=False, sim_floor=0.0):
        super().__init__()
        if rank <= 0 or hidden_size <= 0:
            raise ValueError("rank and hidden_size must be positive")
        if max_ratio_core > max_ratio:
            raise ValueError("the core ceiling must not exceed the graft ceiling")
        if neighbor_mode not in NEIGHBOUR_MODES:
            raise ValueError(f"unknown neighbour mode {neighbor_mode!r}: expected one of "
                             f"{NEIGHBOUR_MODES}")
        if neighbor_mode == "complement" and int(k_same) + int(k_cross) != int(k_neighbors):
            raise ValueError(f"complement neighbours read k = k_same + k_cross, but "
                             f"{k_same} + {k_cross} != {k_neighbors}")
        sim_floor = float(sim_floor)
        if not math.isfinite(sim_floor):
            raise ValueError("the neighbour similarity floor must be finite")
        if sim_floor > 0.0 and not bool(use_cross):
            # Without the cross path there is no neighbour rule for the floor to act on,
            # and a checkpoint that records one would look like it had been applied.
            raise ValueError("sim_floor needs use_cross=True: there is no neighbour rule "
                             "to floor otherwise")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.max_ratio = float(max_ratio)
        self.max_ratio_core = float(max_ratio_core)
        self.cond_size = int(cond_size)
        self.cond_rank = int(cond_rank)
        self.use_cross = bool(use_cross)
        self.k_neighbors = int(k_neighbors)
        self.neighbor_mode = str(neighbor_mode)
        self.k_same = int(k_same)
        self.k_cross = int(k_cross)
        self.sim_floor = sim_floor
        # Parameter-free refinements of the neighbour aggregation: recorded in the
        # checkpoint config so an evaluation rebuilds the same context rule.
        # Inference-time override of the graft ceiling, set by `set_graft_cap`; None
        # keeps the scalar `max_ratio` the module was built with.
        self.graft_cap = None
        self.span_value = bool(span_value)
        self.context_diffusion = bool(context_diffusion)
        self.csls = bool(csls)
        self.use_evicted = bool(use_evicted)
        self.kv_dim = int(kv_dim)
        self.core_weight = float(core_weight)
        self.graft_weight = float(graft_weight)
        self.down = nn.Parameter(torch.empty(self.rank, self.hidden_size))
        self.up_graft = nn.Parameter(torch.zeros(self.hidden_size, self.rank))
        self.up_core = nn.Parameter(torch.zeros(self.hidden_size, self.rank))
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        self.type_gate_logit = nn.Parameter(torch.zeros(()))
        # `cond_down` random and `cond_up` zero: the film is exactly 0 at init, and both
        # halves still have a gradient route out. All-zero deadlocks (measured, round 6).
        self.cond_down = nn.Parameter(torch.empty(self.cond_rank, self.cond_size))
        self.cond_up = nn.Parameter(torch.zeros(2 * self.rank, self.cond_rank))
        nn.init.orthogonal_(self.down)
        nn.init.orthogonal_(self.cond_down, gain=0.1)
        if self.use_cross:
            self.cross_proj = nn.Linear(self.kv_dim, self.rank, bias=False)
            nn.init.zeros_(self.cross_proj.weight)
        if self.use_evicted:
            # Second cross-block path, same width and same zero init: at update 0 the
            # module is still exactly V11's calibrator.
            self.evict_proj = nn.Linear(self.kv_dim, self.rank, bias=False)
            nn.init.zeros_(self.evict_proj.weight)
        # Per-layer weights for the residual regulariser (V17). Non-persistent on
        # purpose: this is not part of the calibrator's numerics, and a PERSISTENT buffer
        # would make every existing V13 checkpoint fail a strict load for a key it was
        # never trained with. All ones is the neutral value, so the accumulation in
        # `calibrate` stays bit-identical to the unweighted sum until a caller writes
        # something else here.
        self.register_buffer("layer_norm_weights",
                             torch.ones(len(UTILITY_LAYERS), dtype=torch.float32),
                             persistent=False)

    def config(self):
        """Constructor kwargs, so a reference copy cannot drift from the trained one."""
        return dict(hidden_size=self.hidden_size, rank=self.rank,
                    max_ratio=self.max_ratio, cond_size=self.cond_size,
                    cond_rank=self.cond_rank, use_cross=self.use_cross,
                    k_neighbors=self.k_neighbors, neighbor_mode=self.neighbor_mode,
                    k_same=self.k_same, k_cross=self.k_cross, kv_dim=self.kv_dim,
                    max_ratio_core=self.max_ratio_core, core_weight=self.core_weight,
                    graft_weight=self.graft_weight, span_value=self.span_value,
                    context_diffusion=self.context_diffusion, csls=self.csls,
                    use_evicted=self.use_evicted, sim_floor=self.sim_floor)

    def evicted(self, window, meta, source, return_report=False):
        """This module's evicted-pool rule -- the second relation path, if enabled."""
        if not self.use_evicted:
            if return_report:
                return None, None
            return None
        return evicted_pool_context(window, meta, source, k=self.k_neighbors,
                                    return_report=return_report)

    def neighbours(self, window, meta, return_report=False):
        """This module's own neighbour rule -- the one place that knows it.

        Callers must not build the context themselves: a V13 checkpoint and a V14
        checkpoint differ in nothing but this rule, so a call site that hard-codes
        `k_neighbors` or the mode would silently evaluate the wrong arm.
        """
        if not self.use_cross:
            if return_report:
                return None, None
            return None
        return cross_block_context(
            window, meta, k=self.k_neighbors, return_report=return_report,
            mode=self.neighbor_mode, k_same=self.k_same, k_cross=self.k_cross,
            span_value=self.span_value, context_diffusion=self.context_diffusion,
            csls=self.csls, sim_floor=self.sim_floor)

    def set_graft_cap(self, cap):
        """Override the graft residual ceiling at inference time; no parameter changes.

        `cap` is a scalar (uniform cap, e.g. 0.38 x rms) or one value per slot (the
        relevance-proportional cap). The carrier ceiling is untouched.
        """
        if cap is None:
            self.graft_cap = None
            return self
        values = torch.as_tensor(cap, dtype=torch.float32).reshape(-1)
        if not torch.isfinite(values).all():
            raise ValueError("the graft cap must be finite")
        if bool((values <= 0).any()):
            # A non-positive ceiling would flip the sign of the residual instead of
            # shrinking it, which is never what this override is for.
            raise ValueError("the graft cap must be strictly positive")
        # Length is checked in `forward`, where the slot count is known: one value means
        # "uniform cap", anything else must match the window exactly.
        self.graft_cap = values
        return self

    def anchor_view(self):
        """`AnchorResidual` carrying this module's graft parameters.

        At initialisation the cross projection is zero and the film is zero, so the
        graft half has to reproduce AnchorOnly exactly; the trainer checks that by
        rebuilding the plain anchor from these parameters and applying it to the same
        slots. Without this view the check cannot distinguish "the module loaded" from
        "the module stayed at zero".
        """
        from methods.prlmc.anchor_residual import AnchorResidual

        anchor = AnchorResidual(hidden_size=self.hidden_size, rank=self.rank,
                               max_ratio=self.max_ratio, gate_init=0.0)
        with torch.no_grad():
            anchor.down.copy_(self.down.detach())
            anchor.up.copy_(self.up_graft.detach())
            anchor.gate_logit.copy_(self.gate_logit.detach())
        return anchor

    # -- initialisation from the finished AnchorOnly run --------------------
    @torch.no_grad()
    def load_anchor_only(self, state_dict, config=None):
        """Start the graft half where AnchorOnly left it; the core half at identity."""
        for name in ("down", "gate_logit"):
            if name not in state_dict:
                raise ValueError(f"the anchor checkpoint has no {name}")
            getattr(self, name).copy_(state_dict[name].float())
        if "up" not in state_dict:
            raise ValueError("the anchor checkpoint has no up")
        self.up_graft.copy_(state_dict["up"].float())
        if config is not None:
            if int(config.get("rank", -1)) != self.rank:
                raise ValueError(f"anchor rank {config.get('rank')} != {self.rank}")
            if int(config.get("hidden_size", -1)) != self.hidden_size:
                raise ValueError(f"anchor hidden {config.get('hidden_size')} != "
                                 f"{self.hidden_size}")
        if float(self.up_core.abs().max()) != 0.0:
            raise ValueError("the core half must start at zero")
        return self

    # -- the residual -------------------------------------------------------
    def forward(self, slots, cond, context=None):
        """`[layers, N, D]` slots, `[N, 3]` conditions, `[N, kv]` context -> delta."""
        if slots.shape[-1] != self.hidden_size:
            raise ValueError(f"expected hidden {self.hidden_size}, got {slots.shape[-1]}")
        if cond.shape != (slots.shape[1], self.cond_size):
            raise ValueError(f"expected conditions {tuple((slots.shape[1], self.cond_size))},"
                             f" got {tuple(cond.shape)}")
        in_pool, evicted = (context if isinstance(context, tuple)
                           else (context, None))
        if self.use_cross:
            if in_pool is None:
                raise ValueError("this module is cross-block conditioned: it needs a "
                                 "context, and got None")
            if in_pool.shape[0] != slots.shape[1]:
                raise ValueError("one context row per slot")
        if self.use_evicted:
            if evicted is None:
                raise ValueError("this module carries the evicted-pool path: pass "
                                 "(in_pool_context, evicted_context)")
            if evicted.shape[0] != slots.shape[1]:
                raise ValueError("one evicted-context row per slot")
        dtype = slots.dtype
        slots_count = int(slots.shape[1])
        x = slots.float()
        bottleneck = F.gelu(F.linear(x, self.down.float()))
        film = F.linear(F.gelu(F.linear(cond.float(), self.cond_down.float())),
                        self.cond_up.float())
        scale = torch.tanh(film[:, : self.rank])
        shift = torch.tanh(film[:, self.rank:])
        bottleneck = bottleneck * (1.0 + scale.unsqueeze(0)) + shift.unsqueeze(0)
        if self.use_cross:
            # Zero-initialised: the first update sees V11's calibrator exactly.
            bottleneck = bottleneck + self.cross_proj(
                in_pool.float().to(bottleneck.device)).unsqueeze(0)
        if self.use_evicted:
            # The dropped 2,560 slots, through their own zero-initialised map.
            bottleneck = bottleneck + self.evict_proj(
                evicted.float().to(bottleneck.device)).unsqueeze(0)
        rms = x.square().mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        graft = F.linear(bottleneck, self.up_graft.float())
        core = F.linear(bottleneck, self.up_core.float())
        mask = cond[:, 2].float().view(1, -1, 1)
        raw = mask * graft + (1.0 - mask) * core
        if self.graft_cap is None:
            graft_ceiling = self.max_ratio
        else:
            cap = self.graft_cap.to(x.device)
            if cap.numel() == 1:
                graft_ceiling = cap
            else:
                if cap.numel() != slots_count:
                    raise ValueError(f"the graft cap has {cap.numel()} values for "
                                     f"{slots_count} slots")
                graft_ceiling = cap.view(1, -1, 1)
        ceiling = mask * graft_ceiling + (1.0 - mask) * self.max_ratio_core
        gate = torch.sigmoid(self.gate_logit.float()
                             + mask * self.type_gate_logit.float()) * ceiling
        return (gate * rms * torch.tanh(raw / rms)).to(dtype)

    def regulariser_weight(self, cond):
        """Per-slot weight for the residual penalty: carriers held tighter than graft."""
        graft = cond[:, 2] > 0.5
        return torch.where(graft, torch.full_like(cond[:, 2], self.graft_weight),
                           torch.full_like(cond[:, 2], self.core_weight))


def _apply(chunk, delta):
    """RMS-preserving update -- `core_fusion.apply_delta`, the one V11 used.

    A plain `chunk + delta` moves the slot's RMS by up to the residual ceiling
    (15% / 50%), and the frozen reader expects the native norm range: the first
    evaluation run failed its own RMS check at 4.55e-2 for exactly this reason.
    """
    return apply_delta(chunk, delta.to(chunk.dtype))


@torch.no_grad()
def calibrate(window, meta, module, layer_chunk=8, context=None, return_report=False,
              source=None):
    """Module over ALL 2,304 slots: bounded residual, RMS restored, context supplied.

    Layer-chunked because the module is shared across layers: the chunk size cannot
    change the result, it only bounds the float32 working set.

    Two report quantities were added for V17, and neither moves a number V13 reports:

    `per_layer_relative_delta`  `sqrt(sum delta^2 / sum slot^2)` per layer. The global
                                figure cannot say whether a residual is spread evenly or
                                carried by one loud layer, and every layer-weighted
                                decision downstream (the regulariser weights, an
                                inference-time per-layer cap) needs the split to exist.
    `regulariser`               present only when the module carries
                                `layer_norm_weights`: the per-layer residual energy
                                accumulated with those weights instead of summed flat.
                                V13's own `relative_delta` stays the UNWEIGHTED ratio on
                                purpose -- it is the number arms are compared on, and it
                                must mean the same thing in every arm.
    """
    cond = conditions(meta, window.shape[1])
    layers = int(window.shape[0])
    layer_weights = getattr(module, "layer_norm_weights", None)
    if layer_weights is not None:
        layer_weights = layer_weights.detach().float().reshape(-1)
        if int(layer_weights.numel()) != layers:
            raise ValueError(f"the module carries {layer_weights.numel()} layer weights "
                             f"for a {layers}-layer window: one weight per layer")
        if not bool(torch.isfinite(layer_weights).all()):
            raise ValueError("layer_norm_weights must be finite")
    context_report = None
    evicted_report = None
    if module.use_cross and context is None:
        context, context_report = module.neighbours(window, meta, return_report=True)
        if getattr(module, "use_evicted", False):
            if source is None:
                raise ValueError("this checkpoint carries the evicted-pool path: the "
                                 "calibration needs the source pool it was evicted from")
            evicted, evicted_report = module.evicted(window, meta, source,
                                                    return_report=True)
            context = (context, evicted)
            del evicted
    elif getattr(module, "use_evicted", False) and not isinstance(context, tuple):
        raise ValueError("the evicted-pool module needs (in_pool, evicted) contexts")
    pieces = []
    delta_sq = slot_sq = 0.0
    per_type = {0: [0.0, 0.0], 1: [0.0, 0.0]}
    layer_delta_sq = [0.0] * layers
    layer_slot_sq = [0.0] * layers
    weighted_delta_sq = 0.0
    for start in range(0, window.shape[0], layer_chunk):
        stop = min(start + layer_chunk, window.shape[0])
        chunk = window[start:stop]
        delta = (module(chunk, cond, None if context is None else context,
                        torch.arange(start, stop, device=window.device))
                 if getattr(module, "uses_layers", False)
                 else module(chunk, cond, None if context is None else context))
        calibrated = _apply(chunk, delta)
        pieces.append(calibrated)
        weight = delta.detach().float().square()
        # The global sums keep their original expressions: recomputing them as sums of
        # per-layer sums would change the last bits of a float32 reduction, and V13's
        # numbers are the reference every arm is measured against.
        delta_sq += float(weight.sum())
        slot_sq += float(chunk.detach().float().square().sum())
        layer_delta = weight.sum(dim=(1, 2))
        layer_slot = chunk.detach().float().square().sum(dim=(1, 2))
        for offset in range(int(chunk.shape[0])):
            layer_delta_sq[start + offset] += float(layer_delta[offset])
            layer_slot_sq[start + offset] += float(layer_slot[offset])
        if layer_weights is not None:
            weighted_delta_sq += float((layer_delta * layer_weights[start:stop]).sum())
        for kind in (0, 1):
            mask = (meta["slot_type"] == kind)
            if bool(mask.any()):
                per_type[kind][0] += float(weight[:, mask].sum())
                per_type[kind][1] += float(chunk.detach().float()[:, mask].square().sum())
        del delta
    per_layer = [(layer_delta_sq[index] / max(layer_slot_sq[index], 1e-24)) ** 0.5
                 for index in range(layers)]
    info = dict(
        slots=int(window.shape[1]),
        core_slots=int((meta["slot_type"] == 0).sum()),
        graft_slots=int((meta["slot_type"] == 1).sum()),
        folded_slots=0,
        relative_delta=(delta_sq / max(slot_sq, 1e-24)) ** 0.5,
        per_layer_relative_delta=per_layer,
        core_relative_delta=(per_type[0][0] / max(per_type[0][1], 1e-24)) ** 0.5,
        graft_relative_delta=(per_type[1][0] / max(per_type[1][1], 1e-24)) ** 0.5,
        use_cross=bool(module.use_cross),
        neighbor_mode=(module.neighbor_mode if module.use_cross else None),
        k_neighbors=(module.k_neighbors if module.use_cross else None),
        cross_proj_norm=float(module.cross_proj.weight.detach().float().norm())
        if module.use_cross else None,
        gate=float(torch.sigmoid(module.gate_logit.detach().float())))
    if context_report is not None:
        info["cross_context"] = context_report
    if evicted_report is not None:
        info["evicted_context"] = evicted_report
    if layer_weights is not None:
        # Mean over layers, so a module whose weights are all 1 reports the same mean
        # residual energy the unweighted arms report; the weights decide WHICH layer's
        # energy counts, not how much of it there is.
        info["layer_norm_weights"] = [float(value) for value in layer_weights]
        info["regulariser"] = float(weighted_delta_sq / max(layers, 1)
                                    / max(slot_sq, 1e-24))
    if return_report:
        return torch.cat(pieces, dim=0), info, context
    return torch.cat(pieces, dim=0), info


@torch.no_grad()
def permute_context(context, meta, seed=0):
    """Shuffle the ALREADY COMPUTED context rows inside (source_block, slot_type) groups.

    This is the attribution test for V13's correspondence, not a second mechanism: the
    module, the window and the self-calibration branch are untouched, and every row that
    moves keeps the distribution it came from -- same source block, same core/graft kind,
    so the batch of contexts a slot can draw from has the same size, the same neighbours,
    the same RMS and the same block structure as the original. What is destroyed is only
    WHICH slot receives WHICH row.

    Permuting the finished context is the point: drawing random top-4 neighbours instead
    would also change the marginal distribution (unrelated slots, other blocks, another
    similarity range), and then a drop would not be attributable to the correspondence.

    Groups with a single member cannot move and are left in place; the returned report
    counts how many slots actually received a different row, because a permutation that
    moved nothing is a bug that looks exactly like "the correspondence does not matter".
    """
    slots = int(context.shape[0])
    block = meta["source_block"].to(torch.long)
    if int(block.numel()) != slots:
        raise ValueError("one source block per context row")
    kind = (meta["slot_type"] > 0.5).to(torch.long)
    group = block * 2 + kind
    generator = torch.Generator().manual_seed(int(seed))
    permuted = context.clone()
    mobile = 0
    for value in torch.unique(group).tolist():
        index = torch.nonzero(group == int(value), as_tuple=False).flatten()
        count = int(index.numel())
        if count < 2:
            continue
        order = torch.randperm(count, generator=generator)
        target = index.to(context.device)
        permuted[target] = context.index_select(0, target[order.to(context.device)])
        mobile += count
    changed = int((permuted != context).any(dim=-1).sum())
    report = dict(seed=int(seed), groups=int(torch.unique(group).numel()),
                  slots_in_multi_slot_groups=mobile, slots_moved=changed,
                  moved_fraction=changed / max(slots, 1),
                  context_rms=float(permuted.square().mean().sqrt()),
                  note="rows rearranged inside (source_block, slot_type); the set of "
                       "contexts a slot can receive is unchanged, only the pairing is")
    return permuted, report


def load_cross(path, hidden_size, rank=None, max_ratio=None):
    """Rebuild a `CrossBlockResidual` from its checkpoint, refusing a mismatch."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA + "/checkpoint":
        raise ValueError(f"{path} is not a {SCHEMA} checkpoint")
    config = dict(payload["config"])
    if int(config.get("hidden_size", -1)) != int(hidden_size):
        raise ValueError(f"{path} was built for hidden {config.get('hidden_size')}, "
                         f"the reader is {hidden_size}")
    if rank is not None and int(config.get("rank", -1)) != int(rank):
        raise ValueError(f"{path} has rank {config.get('rank')}, expected {rank}")
    if max_ratio is not None and float(config.get("max_ratio", -1)) != float(max_ratio):
        raise ValueError(f"{path} has max_ratio {config.get('max_ratio')}")
    module = CrossBlockResidual(**config)
    module.load_state_dict(payload["state_dict"], strict=True)
    return module, config
