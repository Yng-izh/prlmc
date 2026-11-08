"""Round 6, module 2: buy graft slots with same-block redundancy instead of eviction.

Nine carrier blocks are kept whole: 9 x 256 = 2,304 slots, and the ten dropped
blocks get 360 of them. Round 5 showed that giving the dropped blocks more of the
same budget does not help as long as the space comes out of the carrier blocks by
plain eviction -- the slots evicted at the margin are ones the reader was using.
This module tries the other way of paying for it: fold away slots that say the same
thing as another slot in the SAME carrier block, and spend the freed space on the
dropped blocks.

    256 slots per carrier -> 180 kept  (76 released per block, 684 in total)
    =>  1620 core + 684 graft = 2,304 slots, the same budget as before

A pair is only folded when the reader itself cannot tell the two apart: the
similarity is measured on the reader's own keys AND values at the utility layers,
and a pair has to clear both thresholds. Keys alone would merge two slots that
produce the same attention weights but write different values -- the reader would
then be reading a blend of two different facts.

What "the reader cannot tell them apart" is checked against
-----------------------------------------------------------
Nothing about this module is allowed to assume that every block has 76 foldable
pairs. It reports, per block, how many clusters it actually formed and how many
slots it had to evict anyway, and the operator refuses to claim a merge that did
not happen: the returned `merged_pairs` is `slots_in - clusters`, counted, not
estimated. If the released capacity turns out to come from eviction rather than
from folding, the report says so, and an accuracy number that failed to move can be
attributed to the mechanism instead of to the idea.

Blending
--------
A folded slot keeps the representative's direction and moves toward the mean of its
cluster by a bounded weight, and the representive's per-layer RMS is restored
afterwards. Same reason as everywhere else in this project: the frozen reader
expects the native norm range, so a merge may change a slot's direction but not its
scale. Merged slots are no longer verbatim native vectors -- that is the whole
point of the module, and it is recorded per slot so the calibration can condition
on it (`log|C_i|`).
"""
import torch
import torch.nn.functional as F

SCHEMA = "homogeneous_merge/1"


def rms_preserve(base, blended):
    """Blend `blended` toward `base`'s per-layer RMS, keeping `base`'s direction."""
    orig = base.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    new = blended.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    return (blended.float() * (orig / new)).to(base.dtype)


def merge_block(slots, key, value, utility, quota, key_threshold=0.72,
                value_threshold=0.60, max_cluster=3, blend_weight=0.25,
                max_blend=0.5):
    """Fold one block's slots from `slots.shape[1]` down to `quota`.

    `slots`  `[all_layers, N, D]` this block's slots, in block order
    `key`    `[N, Dk]` unit rows, the reader's multi-layer key representation
    `value`  `[N, Dv]` unit rows, the reader's multi-layer value representation
    `utility` `[N]` the reader-utility score, which decides cluster representatives

    Returns `(keep_local, cluster_size, merged, report)`:
      `keep_local`   `[quota]` local indices of the blocks' slots, in the order the
                     representatives were formed (highest utility first, then the
                     splits). The caller restores chronological order.
      `cluster_size` `[quota]` how many native slots each kept slot stands for
                     (1 = an unfolded slot).
      `merged`       `[all_layers, quota, D]` the slot vectors to emit.
    """
    n = int(slots.shape[1])
    if quota <= 0 or quota > n:
        raise ValueError(f"quota {quota} outside 1..{n}")
    if key.shape[0] != n or value.shape[0] != n or utility.numel() != n:
        raise ValueError("key, value and utility disagree with the slot count")

    key_sim = key @ key.t()
    value_sim = value @ value.t()
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=key.device)
    eligible = ((key_sim > key_threshold) & (value_sim > value_threshold)
                & off_diagonal)

    order = torch.argsort(utility, descending=True)
    assigned = torch.zeros(n, dtype=torch.bool, device=key.device)
    clusters = []
    merged_pairs = 0
    pair_key, pair_value = [], []
    for position in order.tolist():
        if bool(assigned[position]):
            continue
        assigned[position] = True
        members = [position]
        if max_cluster > 1:
            neighbour = eligible[position] & ~assigned
            if bool(neighbour.any()):
                index = torch.nonzero(neighbour, as_tuple=False).flatten()
                rank = key_sim[position][index] + value_sim[position][index]
                keep = min(int(max_cluster) - 1, int(index.numel()))
                pick = index[torch.argsort(rank, descending=True)[:keep]]
                assigned[pick] = True
                members.extend(int(x) for x in pick.tolist())
                pair_key.append(key_sim[position][pick])
                pair_value.append(value_sim[position][pick])
                merged_pairs += int(pick.numel())
        clusters.append(members)

    evicted = 0
    splits = 0
    if len(clusters) > quota:
        # The clusters are in utility order, so the tail is the cheapest to drop.
        dropped = clusters[quota:]
        evicted = int(sum(len(cluster) for cluster in dropped))
        clusters = clusters[:quota]
    elif len(clusters) < quota:
        # Folding overshot: hand members back as independent slots, largest
        # clusters first, until the quota is exactly filled. No slot is invented.
        clusters = [list(cluster) for cluster in clusters]
        while len(clusters) < quota:
            donor = max(range(len(clusters)), key=lambda i: len(clusters[i]))
            if len(clusters[donor]) < 2:
                break
            spare = clusters[donor].pop()
            clusters.append([spare])
            splits += 1
        if len(clusters) != quota:
            raise ValueError(f"block holds {n} slots; {len(clusters)} clusters cannot "
                             f"fill a quota of {quota}")

    keep_local = torch.tensor([cluster[0] for cluster in clusters], dtype=torch.long,
                              device=slots.device)
    cluster_size = torch.tensor([len(cluster) for cluster in clusters],
                                dtype=torch.long, device=slots.device)
    pieces = [slots[:, keep_local, :]]
    if bool((cluster_size > 1).any()):
        gathered = slots.float().index_select(1, keep_local)
        blended = gathered.clone()
        for position, cluster in enumerate(clusters):
            if len(cluster) < 2:
                continue
            weight = min(float(blend_weight) * (len(cluster) - 1), float(max_blend))
            mean = slots.float().index_select(
                1, torch.tensor(cluster, dtype=torch.long, device=slots.device)).mean(dim=1)
            mixed = (1.0 - weight) * gathered[:, position] + weight * mean
            blended[:, position] = rms_preserve(gathered[:, position], mixed).float()
        pieces = [blended.to(slots.dtype)]
        del gathered, blended
    merged = pieces[0]

    report = dict(
        slots_in=n, quota=int(quota), kept=int(keep_local.numel()),
        clusters=len(clusters), merged_pairs=merged_pairs, evicted=evicted,
        splits=splits,
        merged_pairs_expected=n - quota,
        mean_cluster_size=round(n / max(len(clusters), 1), 3),
        mean_key_similarity=(round(float(torch.cat(pair_key).mean()), 4)
                             if pair_key else None),
        min_key_similarity=(round(float(torch.cat(pair_key).min()), 4)
                            if pair_key else None),
        mean_value_similarity=(round(float(torch.cat(pair_value).mean()), 4)
                               if pair_value else None),
        min_value_similarity=(round(float(torch.cat(pair_value).min()), 4)
                              if pair_value else None),
        key_threshold=float(key_threshold), value_threshold=float(value_threshold),
        blend_weight=float(blend_weight), max_cluster=int(max_cluster))
    return keep_local, cluster_size, merged, report


def evict_block(slots, utility, quota):
    """The control: keep the top-`quota` slots verbatim, exactly like round 5.

    Same capacity, same release, no folding. This is what arm D of the joint
    experiment builds, and its only job is to attribute a V11 gain to the merge
    rather than to the extra graft slots.
    """
    n = int(slots.shape[1])
    if quota <= 0 or quota > n:
        raise ValueError(f"quota {quota} outside 1..{n}")
    keep = torch.argsort(utility, descending=True)[:quota]
    keep, _ = torch.sort(keep)
    merged = slots.index_select(1, keep)
    cluster_size = torch.ones(quota, dtype=torch.long, device=slots.device)
    report = dict(slots_in=n, quota=int(quota), kept=int(keep.numel()), clusters=int(quota),
                  merged_pairs=0, evicted=n - int(quota), splits=0,
                  merged_pairs_expected=0, mean_cluster_size=1.0,
                  mean_key_similarity=None, min_key_similarity=None,
                  mean_value_similarity=None, min_value_similarity=None,
                  key_threshold=None, value_threshold=None, blend_weight=None,
                  max_cluster=None)
    return keep, cluster_size, merged, report


def merge_carriers(slots, carrier_offsets, key, value, utility, quota,
                   policy="merge", **kwargs):
    """Fold every carrier block down to `quota`, keeping one order for all layers.

    `carrier_offsets` `[blocks, block_size]` the window positions of each carrier;
    `key`, `value`, `utility` are indexed by WINDOW position. The cluster assignment
    is computed once, from the multi-layer representations, and applied at every
    layer -- the operator's "same index in all 32 layers" property is preserved by
    construction, not re-derived per layer.
    """
    if policy not in ("merge", "evict"):
        raise ValueError(f"unknown policy {policy!r}")
    blocks = int(carrier_offsets.shape[0])
    pieces, sizes, reports, origins = [], [], [], []
    for block in range(blocks):
        positions = carrier_offsets[block]
        local_slots = slots.index_select(1, positions)
        local_key = key.index_select(0, positions)
        local_value = value.index_select(0, positions)
        local_utility = utility.index_select(0, positions)
        if policy == "evict":
            keep, cluster_size, merged, report = evict_block(local_slots, local_utility,
                                                             quota)
        else:
            keep, cluster_size, merged, report = merge_block(
                local_slots, local_key, local_value, local_utility, quota, **kwargs)
        report["block"] = block
        reports.append(report)
        pieces.append(merged)
        sizes.append(cluster_size)
        origins.append(positions.index_select(0, keep))
    window = torch.cat(pieces, dim=1)
    cluster_size = torch.cat(sizes, dim=0)
    origin = torch.cat(origins, dim=0)
    summary = dict(
        policy=policy, blocks=blocks, core_per_block=int(quota),
        core_total=int(window.shape[1]),
        slots_in=int(sum(report["slots_in"] for report in reports)),
        merged_pairs=int(sum(report["merged_pairs"] for report in reports)),
        evicted=int(sum(report["evicted"] for report in reports)),
        splits=int(sum(report["splits"] for report in reports)),
        blocks_with_no_merge=int(sum(1 for report in reports
                                     if report["merged_pairs"] == 0)),
        min_merged_pairs_per_block=int(min(report["merged_pairs"] for report in reports)),
        max_merged_pairs_per_block=int(max(report["merged_pairs"] for report in reports)),
        blocks_where_quota_came_from_eviction=int(
            sum(1 for report in reports if report["evicted"] > 0)),
        per_block=reports)
    return window, cluster_size, origin, summary
