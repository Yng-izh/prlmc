"""D with corrected supervision, optionally one full-width source skip path.

The native path reuses D's routing, slot gates, layer grouping and global cap.
It is a source difference, not a claim of lossless copying or new memory slots.
"""
import torch
from torch import nn

from .consolidate import _DELTA_VAR_EPS
from .gated_slot_merge import GraftedSlotMemoryMerge


class SupervisedDSlotMerge(GraftedSlotMemoryMerge):
    architecture = "grouped_gated_sparse_slot_v4"

    def _sparse_attention(self, query, keys, *extra):
        probabilities, indices = super()._sparse_attention(query, keys, *extra)
        if hasattr(self, "_coverage_mass"):
            with torch.no_grad():
                bins=(indices.detach()//self._coverage_block).flatten()
                self._coverage_mass += torch.bincount(bins,weights=probabilities.detach().flatten(),
                                                     minlength=self._coverage_mass.numel())
                self._coverage_queries += query.shape[0]
        return probabilities, indices

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
    architecture = "grouped_gated_sparse_native_v4"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Zero starts at exactly the same function as supervised D: at
        # native_gain=0 the merge is bit-identical to Selection. One signed
        # scalar is sufficient; the existing gates and cap bound both paths.
        # (A per-layer gain would broadcast against [queries, hidden] inside
        # _group_delta, which has no layer index, so it stays scalar.)
        self.native_gain = nn.Parameter(torch.zeros(()))

    def _group_delta(self, probabilities, indices, source_layer, carrier_layer,
                     down_weight, up_weight):
        adapted = super()._group_delta(probabilities, indices, source_layer,
                                      carrier_layer, down_weight, up_weight)
        source = source_layer.float().flatten(0, 1)
        carrier = carrier_layer.float().flatten(0, 1)
        carrier_rms = carrier.square().mean(-1, keepdim=True).sqrt()
        chunks = []
        # Do not materialize [2304,8,4096] at once on a 32-GiB V100.
        for start in range(0, indices.shape[0], self.query_chunk_size):
            ix = indices[start:start+self.query_chunk_size]
            p = probabilities[start:start+self.query_chunk_size]
            chunks.append((source[ix] * p.unsqueeze(-1)).sum(1))
        # Native branch: per-slot RMS normalization first, then rescale to the
        # carrier RMS, so its energy never flattens the adapted delta.
        native = torch.cat(chunks) - carrier
        native = native * torch.rsqrt(native.square().mean(-1, keepdim=True) + _DELTA_VAR_EPS) * carrier_rms
        combined = adapted + self.native_gain * native
        return combined * torch.rsqrt(combined.square().mean(-1, keepdim=True) + _DELTA_VAR_EPS) * carrier_rms

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        self.last_fusion.update(native_gain=float(self.native_gain.detach()),
            native_path="same Top-8 full-width source average minus carrier; per-slot RMS-normalized to carrier scale")
        return output
