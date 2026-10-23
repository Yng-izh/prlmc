"""Inference-time fusion refinements for the v7 lineage (Part 1 of the
fine-grained fact-relation plan): block-coherent routing and orthogonal fact
injection.

Two operator changes over the sealed v7 classes, neither adding a parameter:

1. Block-coherent routing (调整1后半). The sealed global top-k routinely mixes
   source slots from several discarded blocks into one carrier slot; averaging
   those 4096-D vectors can synthesise a plausible-looking entity that exists
   in no source paragraph (the hotpotqa failure mode: gold "Mika Häkkinen"
   answered "Ayrton Senna", gold "2010" answered "2007"). Every carrier slot
   whose retained top-k spans more than one discarded block is re-routed into
   the SINGLE most affine discarded block and takes its top-k only inside that
   block. Block affinity is the proposal's metric: the mean of the block's
   top-3 mixed routing logits. The mixed logits are recomputed for the
   cross-block rows with the exact sealed formula (native/learned mixture,
   balance offset, prefix mask), so the reroute reuses the routing module's
   own view. Rows whose base top-k already fell inside one block are passed
   through untouched -- they keep their exact sealed weights.

2. Orthogonal fact injection (调整2). The bounded v7 delta is decomposed
   against the carrier slot into a parallel and an orthogonal component; the
   written residual is `ortho + coef * parallel` with
   `coef = 0.15 + 0.85 * sigmoid(redundancy_z)` (detached). redundancy_z is
   the sealed in-block standardised redundancy feature (column 2 of `rel`),
   so block-distinctive core slots keep only ~0.15 of the carrier-parallel
   component -- the only part that can flip a carrier onto a different
   entity -- while high-redundancy background slots take the full delta.

Both mechanisms are zero-safe: a zero-initialised delta decomposes to exactly
zero and zero routing never crosses blocks, so step 0 stays bit-identical to
Selection. The state dict is identical to the sealed v7 modules, so a trained
v7 checkpoint loads strictly with no conversion; the mode is recorded per run
as `fusion_mode="block_ortho"` in the eval identity.

These classes are deliberately NOT registered in `build_merge`: enabling them
is an explicit per-run protocol choice (`--fusion-mode block_ortho`), not a
new architecture generation. Keeping them out of the sealed v7 operator files
(`gated_slot_merge.py` / `native_skip_merge.py`) leaves the checkpoint source
manifests valid -- existing v7 checkpoints still pass strict code-hash
verification.
"""
import torch

from .gated_slot_merge import SupervisedDSlotMergeV7
from .native_skip_merge import NativeSkipSlotMergeV7

# Parallel-component floor on the most block-distinctive (low-redundancy)
# core slots; the proposal's "0.15 * parallel" default.
ORTHO_PARALLEL_FLOOR = .15


class BlockOrthoCoherentMixin:
    """Shared block-coherent routing + orthogonal write for the v7 lineage.

    Inserted before the v7 class in the MRO, so every hook (`_sparse_attention`,
    `_slot_alpha`, `_group_delta`) wraps the sealed v7 implementation instead of
    replacing it.
    """

    def _sparse_attention(self, query, keys, *extra):
        probabilities, indices, top1 = super()._sparse_attention(query, keys, *extra)
        # extra == (q_nat, k_nat, prefix_mask) at every sealed call site.
        if len(extra) >= 2 and extra[0] is not None and extra[1] is not None:
            key_mask = extra[2] if len(extra) > 2 else None
            result = self._reroute_rows(probabilities, indices, top1, query, keys,
                                        extra[0], extra[1], key_mask)
            if result is not None:
                probabilities, indices, top1 = result
        # Per-row retained indices of the last routing chunk, for the tests /
        # telemetry (the sealed trace only reports block-level source lists).
        self._last_indices = indices
        return probabilities, indices, top1

    def _reroute_rows(self, probabilities, indices, top1, query, keys,
                      q_nat, k_nat, key_mask):
        """Block-coherent reroute of the cross-block routing rows.

        Returns None when no row's retained top-k spans more than one
        discarded block (the sealed output is then passed through unchanged).
        Cross-block rows are re-scored with the sealed mixed-logit formula over
        ALL discarded slots, the most affine block is picked by the mean of its
        internal top-3 logits, and the top-k is taken inside that block and
        renormalised. Shapes match the sealed contract:
        [queries, topk] probabilities / global slot indices, plus the raw top-1
        logit per query.
        """
        K = self.block
        if self.topk > K:
            # Production invariant: topk (8) << block (256). When it does not
            # hold, the intra-block top-k would change the retained column
            # count, so the sealed global routing is kept instead.
            return None
        cross = (indices // K != indices[:, :1] // K).any(-1)
        if not bool(cross.any()):
            return None
        rows = cross.nonzero(as_tuple=True)[0]
        q = query[rows].float()
        qn = q_nat[rows].float()
        # Sealed formula, dense over all discarded slots for the cross rows only.
        logits = (self.native_prior_weight * (qn @ k_nat.T)
                  + (1. - self.native_prior_weight) * (q @ keys.T)
                  ) / self.route_mix_temperature
        balance = getattr(self, "_balance_offset", None)
        if balance is not None:
            logits = logits + balance
        if key_mask is not None:
            logits = logits + key_mask.float() * (-1e30)
        n_blocks = k_nat.shape[0] // K
        block_logits = logits.reshape(len(rows), n_blocks, K)
        # Block affinity: mean of the block-internal top-3 mixed logits.
        inner = min(3, K)
        affinity = block_logits.topk(inner, dim=-1).values.mean(-1)
        best = affinity.argmax(-1)
        gathered = block_logits.gather(
            1, best[:, None, None].expand(-1, 1, K)).squeeze(1)
        values, local = gathered.topk(min(self.topk, K), dim=-1)
        probabilities = probabilities.clone()
        indices = indices.clone()
        top1 = top1.clone()
        probabilities[rows] = torch.softmax(values, dim=-1).to(probabilities.dtype)
        indices[rows] = best[:, None] * K + local
        top1[rows] = values[:, 0]
        stats = getattr(self, "_reroute_stats", None)
        if stats is not None:
            stats["queries"] += int(probabilities.shape[0])
            stats["moved"] += int(rows.numel())
        return probabilities, indices, top1

    def _slot_alpha(self, gate_logits, rel, reject=None):
        """Stash the detached per-slot redundancy weight for the write path.

        `rel[..., 2]` is the in-block standardised redundancy feature; its
        sigmoid is the parallel-component coefficient of the proposal. The
        sealed v7 alpha itself is untouched.
        """
        alpha = super()._slot_alpha(gate_logits, rel, reject)
        if rel is not None and rel.dim() == 2 and rel.shape[-1] >= 3:
            self._redundancy_weight = torch.sigmoid(rel[..., 2].detach())
        return alpha

    def _ortho_decompose(self, delta, carrier):
        """Split the bounded delta against the carrier and gate the parallel part.

        `delta` is the sealed v7 bounded write (for the native arm this already
        includes the sharp skip path and the carrier-RMS soft bound). The
        returned residual is `ortho + coef * parallel` with
        `coef = floor + (1-floor) * sigmoid(redundancy_z)` (detached), so the
        component along the carrier's own direction -- the only one that can
        move a slot onto a different entity -- shrinks from the full delta to
        (near) the floor exactly on the block-distinctive slots whose identity
        must survive. A zero delta decomposes to exactly zero: step 0 stays
        bit-identical to Selection.
        """
        c_sq = carrier.square().sum(-1, keepdim=True).clamp_min(1e-6)
        parallel = (delta * carrier).sum(-1, keepdim=True) / c_sq * carrier
        ortho = delta - parallel
        weight = getattr(self, "_redundancy_weight", None)
        if weight is None:
            coef = torch.full_like(parallel[..., :1], ORTHO_PARALLEL_FLOOR)
        else:
            coef = (ORTHO_PARALLEL_FLOOR
                    + (1. - ORTHO_PARALLEL_FLOOR) * weight).unsqueeze(-1)
        with torch.no_grad():
            total = delta.norm(dim=-1).clamp_min(1e-12)
            self._ortho_frac_sum = getattr(self, "_ortho_frac_sum", 0.) \
                + float((ortho.norm(dim=-1) / total).mean())
            self._ortho_layers = getattr(self, "_ortho_layers", 0) + 1
            self._ortho_coef_sum = getattr(self, "_ortho_coef_sum", 0.) \
                + float(coef.mean())
        return ortho + coef * parallel

    def _group_delta(self, probabilities, indices, source_layer, carrier_layer,
                     down_weight, up_weight, group_gain=None):
        delta = super()._group_delta(probabilities, indices, source_layer,
                                     carrier_layer, down_weight, up_weight,
                                     group_gain)
        return self._ortho_decompose(delta, carrier_layer.float().flatten(0, 1))

    def forward(self, *args, **kwargs):
        self._redundancy_weight = None
        self._reroute_stats = dict(queries=0, moved=0)
        self._ortho_frac_sum = 0.
        self._ortho_coef_sum = 0.
        self._ortho_layers = 0
        output = super().forward(*args, **kwargs)
        fusion = dict(self.last_fusion)
        if self._ortho_layers:
            fusion.update(fusion_mode="block_ortho", block_coherent_routing=True,
                ortho_delta_fraction=self._ortho_frac_sum / self._ortho_layers,
                parallel_coef_mean=self._ortho_coef_sum / self._ortho_layers,
                ortho_write=(f"delta = ortho + (floor+(1-floor)*sigmoid(redundancy_z))"
                             f" * parallel, floor={ORTHO_PARALLEL_FLOOR}"))
        if self._reroute_stats["queries"]:
            fusion["rerouted_block_fraction"] = (
                self._reroute_stats["moved"] / self._reroute_stats["queries"])
        self.last_fusion = fusion
        return output


class OrthoSupervisedDSlotMergeV7(BlockOrthoCoherentMixin, SupervisedDSlotMergeV7):
    """Dsup v7 with block-coherent routing and orthogonal fact injection."""

    architecture = "grouped_gated_sparse_slot_v7"


class OrthoNativeSkipSlotMergeV7(BlockOrthoCoherentMixin, NativeSkipSlotMergeV7):
    """Dnative v7 with block-coherent routing and orthogonal fact injection.

    The decomposition wraps the sealed combined write (adapted + tanh-bounded
    native skip), so the sharp full-width entity prototype is subject to the
    same parallel-component floor as the low-rank path.
    """

    architecture = "grouped_gated_sparse_native_v7"


ORTHO_MERGE_CLASSES = {
    "grouped_gated_sparse_slot_v7": OrthoSupervisedDSlotMergeV7,
    "grouped_gated_sparse_native_v7": OrthoNativeSkipSlotMergeV7,
}
