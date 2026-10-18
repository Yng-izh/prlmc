"""D with corrected supervision, optionally one full-width source skip path.

The native path reuses D's routing, slot gates, layer grouping and global cap.
It is a source difference, not a claim of lossless copying or new memory slots.

v5 keeps the same claim but makes the passthrough able to carry rare exact
tokens: it averages only the two sharpest routed slots and scales by the routed
confidence, and it carries one signed gain per layer group instead of one global
scalar (shallow groups address, deep groups read out). Both branches are
zero-initialised, so step 0 stays bit-identical to Selection.
"""
import torch
from torch import nn

from .consolidate import _DELTA_VAR_EPS
from .gated_slot_merge import GraftedSlotMemoryMerge


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
        carrier_rms = carrier.square().mean(-1, keepdim=True).sqrt()
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
        # Do not materialize [2304,k,4096] at once on a 32-GiB V100.
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
        combined = adapted + gain * native
        return combined * torch.rsqrt(combined.square().mean(-1, keepdim=True) + _DELTA_VAR_EPS) * carrier_rms

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        self.last_fusion.update(native_gain=[float(g.detach()) for g in self.native_gain],
            native_path="per-layer-group signed gain on a top-%d sharpened full-width source average minus "
                        "carrier, routed-confidence scaled"
                        % self.top_sharp, sharp_slots=self.top_sharp)
        return output
