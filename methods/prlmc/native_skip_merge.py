"""D with corrected supervision, optionally one full-width source skip path.

The native path reuses D's routing, slot gates, layer grouping and global cap.
It is a source difference, not a claim of lossless copying or new memory slots.

v5 keeps the same claim but makes the passthrough able to carry rare exact
tokens: it averages only the two sharpest routed slots and scales by the routed
confidence, and it carries one signed gain per layer group instead of one global
scalar (shallow groups address, deep groups read out). Both branches are
zero-initialised, so step 0 stays bit-identical to Selection.

v7 bounds the branch instead of renormalising it: the gain passes through
`tanh` (so it can at most replace the carrier) and the combined residual is
`carrier_rms * tanh(combined / carrier_rms)`, matching the adapted path.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .gated_slot_merge import (
    GraftedSlotMemoryMerge, SupervisedDSlotMergeV7, _soft_bound,
)
from .consolidate import _ngram_smooth, select_bridge_aware


class SupervisedDSlotMerge(GraftedSlotMemoryMerge):
    architecture = "grouped_gated_sparse_slot_v5"

    def _sparse_attention(self, query, keys, *extra):
        probabilities, indices, top1 = super()._sparse_attention(query, keys, *extra)
        if hasattr(self, "_coverage_mass"):
            with torch.no_grad():
                bins=(indices.detach()//self._coverage_block).flatten()
                self._coverage_mass += torch.bincount(bins,weights=probabilities.detach().flatten(),
                                                     minlength=self._coverage_mass.numel())
                self._coverage_queries += query.shape[0]
        return probabilities, indices, top1

    def forward(self, source, *args, **kwargs):
        block=int(kwargs.get("block") or self.block)
        self._coverage_block=block
        self._coverage_mass=torch.zeros(source.shape[1]//block,device=source.device)
        self._coverage_queries=0
        output=super().forward(source,*args,**kwargs)
        source_indices=self.last_fusion.get("fusion_source_indices",[])
        mass=(self._coverage_mass/max(self._coverage_queries,1)).detach().cpu().tolist()
        self.last_fusion["per_discarded_block_attention_mass"]={str(b):mass[i] for i,b in enumerate(source_indices)}
        del self._coverage_mass
        return output


class NativeSkipSlotMerge(SupervisedDSlotMerge):
    architecture = "grouped_gated_sparse_native_v5"

    def __init__(self, *args, top_sharp=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_sharp = int(top_sharp)
        if self.top_sharp < 1:
            raise ValueError("positive sharp-slot count required")
        # Zero starts at exactly the same function as supervised D: at
        # native_gain=0 the merge is bit-identical to Selection. One signed gain
        # per layer group, because shallow groups address (Key path) and deep
        # groups integrate and read out (Value path): a single shared scalar
        # forces both regimes to move together and is why the passthrough used
        # to help only inside the training domain.
        self.native_gain = nn.Parameter(torch.zeros(self.num_groups))

    def group_gain(self, group):
        return self.native_gain[int(group)]

    def _group_delta(self, probabilities, indices, source_layer, carrier_layer,
                     down_weight, up_weight, group_gain=None):
        adapted = super()._group_delta(probabilities, indices, source_layer,
                                      carrier_layer, down_weight, up_weight)
        source = source_layer.float().flatten(0, 1)
        carrier = carrier_layer.float().flatten(0, 1)
        sharp = min(self.top_sharp, indices.shape[1])
        # Sharp path: renormalised over the two best slots only. A 4096-D
        # average over eight slots blurs exactly the rare entity/number tokens
        # the low-rank path cannot reconstruct; the sum over the sharp slots is
        # also the routed confidence, which keeps the branch quiet on slots that
        # found nothing.
        weights = probabilities[:, :sharp]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        confidence = probabilities[:, :sharp].sum(-1, keepdim=True)
        chunks = []
        # Do not materialize [2304,k,4096] at once: on a 32-GiB V100 this is the
        # difference between running and OOMing, and a 40-GiB A100 shared with
        # another tenant has the same headroom problem.
        for start in range(0, indices.shape[0], self.query_chunk_size):
            ix = indices[start:start+self.query_chunk_size, :sharp]
            p = weights[start:start+self.query_chunk_size]
            chunks.append((source[ix] * p.unsqueeze(-1)).sum(1))
        # Native branch: a full-width source difference, already on the carrier's
        # scale, so no intermediate renormalisation is applied. The single
        # normalisation below maps the sum back to carrier scale and the global
        # cap bounds what is actually stored.
        native = (torch.cat(chunks) - carrier) * confidence
        gain = self.native_gain if group_gain is None else group_gain
        # v7 C5: the native branch is bounded the same way the adapted branch is.
        # `tanh(gain)` keeps the branch gain inside (-1, 1) -- it can at most
        # replace the carrier, never multiply it -- and the combined residual is
        # soft-bounded instead of being renormalised onto carrier RMS, so a
        # zero-initialised `native_gain` contributes exactly nothing.
        combined = adapted + torch.tanh(gain) * native
        return _soft_bound(combined, carrier)

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        self.last_fusion.update(native_gain=[float(g.detach()) for g in self.native_gain],
            native_path="per-layer-group tanh-bounded signed gain on a top-%d sharpened full-width source "
                        "average minus carrier, routed-confidence scaled, soft-bounded by carrier RMS"
                        % self.top_sharp, sharp_slots=self.top_sharp)
        return output


class NativeSkipSlotMergeV7(SupervisedDSlotMergeV7):
    """v7 native branch: same routing/gates as v7, plus a bounded source skip.

    `combined = adapted + tanh(native_gain) * native`, then the same
    `carrier_rms * tanh(combined / carrier_rms)` bound the adapted path uses.
    `native_gain=0` (the initialisation) makes `tanh(0)=0` exactly, so step 0 is
    still bit-identical to Selection.
    """

    architecture = "grouped_gated_sparse_native_v7"

    def __init__(self, *args, top_sharp=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_sharp = int(top_sharp)
        if self.top_sharp < 1:
            raise ValueError("positive sharp-slot count required")
        self.native_gain = nn.Parameter(torch.zeros(self.num_groups))

    def group_gain(self, group):
        return self.native_gain[int(group)]

    def _group_delta(self, probabilities, indices, source_layer, carrier_layer,
                     down_weight, up_weight, group_gain=None):
        adapted = super()._group_delta(probabilities, indices, source_layer,
                                       carrier_layer, down_weight, up_weight)
        source = source_layer.float().flatten(0, 1)
        carrier = carrier_layer.float().flatten(0, 1)
        sharp = min(self.top_sharp, indices.shape[1])
        # Sharp path unchanged: renormalised over the top slots only, and the
        # sum over them is the routed confidence.
        weights = probabilities[:, :sharp]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        confidence = probabilities[:, :sharp].sum(-1, keepdim=True)
        chunks = []
        for start in range(0, indices.shape[0], self.query_chunk_size):
            ix = indices[start:start+self.query_chunk_size, :sharp]
            p = weights[start:start+self.query_chunk_size]
            chunks.append((source[ix] * p.unsqueeze(-1)).sum(1))
        native = (torch.cat(chunks) - carrier) * confidence
        gain = self.native_gain if group_gain is None else group_gain
        combined = adapted + torch.tanh(gain) * native
        return _soft_bound(combined, carrier)

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        self.last_fusion.update(native_gain=[float(g.detach()) for g in self.native_gain],
            native_path="per-layer-group tanh-bounded signed gain on a top-%d sharpened full-width source "
                        "average minus carrier, routed-confidence scaled, soft-bounded by carrier RMS"
                        % self.top_sharp, sharp_slots=self.top_sharp)
        return output


def _orthogonalise(delta, entity, eps=1e-6):
    """Gram-Schmidt: drop the component of `delta` parallel to `entity`.

    Both are [slots, hidden]. Afterwards the low-rank branch can only add
    relation information in the subspace orthogonal to the entity direction the
    native branch writes, so it cannot rotate a rare token into a sibling token
    on its way to the deep layers.
    """
    denom = entity.square().sum(-1, keepdim=True).clamp_min(eps)
    return delta - ((delta * entity).sum(-1, keepdim=True) / denom) * entity


class NativeSkipSlotMergeV9(SupervisedDSlotMergeV7):
    """Dnative v9: structural protection + sharpened entity skip (M1 + M2).

    M1  In-block slot partitioning. Every retained 256-slot carrier block is
        split by layer-16 significance into a protected core (`bridge_fraction`
        of the slots become bridges, default 15.6% = 40 of 256) and dedicated
        bridge slots. `alpha` is hard-masked to exactly 0 on the core, so the
        frozen reader sees the writer's original numbers there bit for bit, and
        the bridge cap (`bridge_gamma_max=.60`) replaces the uniform `.20`. With
        the global RMS budget at .08 and 84.4% of slots contributing nothing,
        `_cap` leaves the bridges at amplitude ~.08/sqrt(.156) ~ .20, i.e. six
        times the old uniform ~.033: readable by the reader, invisible to the
        core. There is no gradient conflict left to trade off, because no slot
        is asked to be both a protected entity and a fusion receiver.
    M2  Decoupled branches. The native entity branch routes top-1 at
        temperature .1 over the full 4096-D slots (no rank-32 bottleneck on rare
        exact tokens); the low-rank branch writes only its Gram-Schmidt
        orthogonal relation component; and every layer group owns its own native
        gain and write scale, so shallow groups address relations while deep
        groups pass entity values unchanged.
    """

    architecture = "grouped_gated_sparse_native_v9"

    def __init__(self, *args, top_sharp=1, route_temperature_sharp=.1,
                 bridge_fraction=.156, bridge_gamma_max=.60,
                 significance_affinity_weight=.5, native_init=0.,
                 ngram_kernel=(.25, .5, .25), bridge_blocks=3, write_threshold=.5,
                 **kwargs):
        kwargs.setdefault("global_rms_budget", .08)
        super().__init__(*args, **kwargs)
        self.top_sharp = max(1, int(top_sharp))
        self.route_temperature_sharp = float(route_temperature_sharp)
        self.bridge_fraction = float(bridge_fraction)
        self.significance_affinity_weight = float(significance_affinity_weight)
        self.ngram_kernel = tuple(float(v) for v in ngram_kernel)
        self.bridge_blocks = int(bridge_blocks)
        self.write_threshold = float(write_threshold)
        if not 0. < self.bridge_fraction < 1.:
            raise ValueError("bridge fraction must be a strict interior ratio")
        if not self.gamma_max <= float(bridge_gamma_max) < 1.:
            raise ValueError("bridge cap must sit between the core cap and one")
        if self.route_temperature_sharp <= 0:
            raise ValueError("positive sharpening temperature required")
        self.register_buffer("bridge_cap", torch.tensor(float(bridge_gamma_max)))
        # One native gain and one write scale per layer group (8 scalars each):
        # shallow groups address, deep groups protect. Two separate vectors, not
        # one shared scalar, so the two roles cannot cancel in the sum.
        self.native_gain = nn.Parameter(torch.full((self.num_groups,), float(native_init)))
        self.group_scale = nn.Parameter(torch.ones(self.num_groups))
        self._bridge_mask = None
        self._frozen_carriers = None
        self.pinned_blocks = None
        self.bridge_report = {}

    # ---- M1: in-block partition -------------------------------------------
    def _gate_inputs(self, q_nat, probabilities, top1):
        gate_in, rel, raw = super()._gate_inputs(q_nat, probabilities, top1)
        self._bridge_mask = self._partition(raw)
        return gate_in, rel, raw

    def _partition(self, raw):
        """Hard, parameter-free split: the least significant slots bridge.

        `raw[..., 0]` is the top-1 native affinity (the attention-norm proxy)
        and `raw[..., 2]` the cosine redundancy with the block centroid, both
        per block. Significance is z-scored inside the block, so the split is
        invariant to the overall scale of a block and deterministic in the
        slot order the reader produced.
        """
        blocks, width, _ = raw.shape
        affinity, redundancy = raw[..., 0], raw[..., 2]

        def z(x):
            return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)

        significance = (self.significance_affinity_weight * z(affinity)
                        - (1. - self.significance_affinity_weight) * z(redundancy))
        count = max(1, min(width - 1, int(round(width * self.bridge_fraction))))
        bridge = torch.zeros(blocks, width, dtype=torch.bool, device=raw.device)
        bridge.scatter_(-1, significance.argsort(dim=-1)[..., :count], True)
        return bridge.reshape(-1)

    def _write_mask(self, like):
        if self._bridge_mask is None:
            if self._frozen_carriers is not None:
                # Failing open here would let a frozen slot be written again and
                # the M4 assertion would only notice several forwards later.
                raise RuntimeError("frozen carriers without a bridge partition")
            return torch.ones(like.shape[0], dtype=like.dtype, device=like.device)
        mask = self._bridge_mask
        if mask.numel() != like.shape[0]:
            raise RuntimeError("bridge partition does not match the gate rows")
        if self._frozen_carriers is not None:
            # M4: a bridge slot that already took a high-amplitude write is
            # read-only from now on, so repeated compactions cannot stack edits.
            mask = mask & ~self._frozen_carriers.to(mask.device)
        return mask.to(like.dtype)

    def _slot_alpha(self, gate_logits, rel, reject=None):
        alpha = super()._slot_alpha(gate_logits, rel, reject)
        # Re-scale the sealed [0, gamma_max] gate onto the bridge cap and apply
        # the hard structural mask: zero on the protected core by construction.
        scale = self.bridge_cap.to(alpha.dtype) / max(float(self.gamma_max), 1e-8)
        alpha = torch.clamp(alpha * scale * self._write_mask(gate_logits).unsqueeze(-1),
                            0., float(self.bridge_cap))
        self._last_alpha = alpha.detach()
        return alpha

    # ---- M2: decoupled branches -------------------------------------------
    def group_gain(self, group):
        g = int(group)
        return dict(native=self.native_gain[g], scale=self.group_scale[g])

    def _group_delta(self, probabilities, indices, source_layer, carrier_layer,
                     down_weight, up_weight, group_gain=None):
        carrier = carrier_layer.float().flatten(0, 1)
        flat = source_layer.float().flatten(0, 1)
        # Relation branch: Top-8 context through the rank-32 bottleneck.
        values = F.linear(flat, down_weight)
        context = (values[indices] * probabilities.unsqueeze(-1)).sum(1)
        adapted = F.linear(F.gelu(context - F.linear(carrier, down_weight)), up_weight)
        # Entity branch: temperature-sharpened top-1, full width, no bottleneck.
        sharp = min(self.top_sharp, indices.shape[1])
        logits = probabilities[:, :sharp].float().clamp_min(1e-12).log()
        weights = torch.softmax(logits / self.route_temperature_sharp, dim=-1)
        chunks = []
        for start in range(0, indices.shape[0], self.query_chunk_size):
            ix = indices[start:start + self.query_chunk_size, :sharp]
            p = weights[start:start + self.query_chunk_size]
            chunks.append((flat[ix] * p.unsqueeze(-1)).sum(1))
        entity = torch.cat(chunks) - carrier
        # A dict always arrives from `group_gain`; the None branch is defensive
        # and must stay scalar, never broadcast the group vector.
        gain = 0. if group_gain is None else group_gain["native"]
        scale = 1. if group_gain is None else group_gain["scale"]
        delta = (_orthogonalise(adapted, entity) + torch.tanh(gain) * entity) * scale
        return _soft_bound(delta, carrier)

    # ---- M3: 3-gram routing keys ------------------------------------------
    def _route_keys(self, k_nat):
        if len(self.ngram_kernel) < 2:
            return k_nat
        return _ngram_smooth(k_nat, self.block, self.ngram_kernel)

    # ---- forward: selection + write-once mask -----------------------------
    def forward(self, source, weights=None, target=None, block=None, goal_direction=None,
                select_weights=None, carrier_indices=None, source_allow_mask=None,
                frozen_slots=None, pinned_blocks=None):
        self._bridge_mask = None
        self._frozen_carriers = None
        self.bridge_report = {}
        # The write record belongs to exactly one forward. Drop the previous
        # call's entries, otherwise a forward that writes nothing republishes
        # the offsets of an earlier call and the M4 assertion then compares
        # them against a different selection.
        fusion = getattr(self, "last_fusion", None)
        if isinstance(fusion, dict):
            for stale in ("bridge_slot_offsets", "bridge_slot_fraction",
                          "written_slot_count", "bridge_slots", "bridge_cap",
                          "route_kernel", "group_scale", "native_gain"):
                fusion.pop(stale, None)
        block = int(block or self.block)
        target = int(target or 0)
        index = carrier_indices
        if index is None and target and target < source.shape[1]:
            picked = select_bridge_aware(
                source, target, block, weights, goal_direction, select_weights,
                pinned=pinned_blocks, bridge_blocks=self.bridge_blocks)
            index = picked["index"]
            self.bridge_report = {k: v for k, v in picked.items() if k != "index"}
        if frozen_slots is not None and index is not None:
            frozen = torch.as_tensor(frozen_slots, device=source.device).reshape(-1, block)
            ix = torch.as_tensor(index, device=source.device, dtype=torch.long)
            self._frozen_carriers = frozen.index_select(0, ix).reshape(-1)
        output = super().forward(source, weights=weights, target=target, block=block,
                                 goal_direction=goal_direction, select_weights=select_weights,
                                 carrier_indices=index, source_allow_mask=source_allow_mask)
        alpha = getattr(self, "_last_alpha", None)
        if alpha is not None and self.last_fusion.get("fusion_source_indices"):
            # The slot axis is dim 0 (one gate row per carrier slot, the same
            # flattening `install` marks); every remaining axis is a per-slot
            # sample. Reduce over the samples so an offset is a slot index, and
            # validate the tiling first: a bare nonzero() on the 2-D record
            # interleaves (slot, sample) pairs into bogus offsets, which froze
            # arbitrary slots and tripped the M4 assertion at the single ->
            # recurrent phase change.
            slot_count = int(self._bridge_mask.numel()) if self._bridge_mask is not None else 0
            written = alpha.detach() >= self.write_threshold
            if slot_count <= 0 or written.shape[0] != slot_count or written.numel() % slot_count:
                raise ValueError("bridge write record does not tile the slot axis")
            written_slots = written.reshape(slot_count, -1).any(dim=1)
            self.last_fusion.update(
                bridge_slots=int(self._bridge_mask.sum()) if self._bridge_mask is not None else 0,
                bridge_cap=float(self.bridge_cap),
                bridge_slot_offsets=torch.nonzero(written_slots).flatten().cpu().tolist(),
                bridge_slot_fraction=float(written_slots.float().mean()),
                written_slot_count=int(written_slots.sum()),
                route_kernel=list(self.ngram_kernel),
                group_scale=[float(s.detach()) for s in self.group_scale],
                native_gain=[float(g.detach()) for g in self.native_gain],
                structural_core="alpha hard-masked to 0 on the protected core",
                **self.bridge_report)
        return output
