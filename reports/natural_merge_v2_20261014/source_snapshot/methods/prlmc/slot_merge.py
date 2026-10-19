"""Query-independent slot fusion into native carriers, with cross-layer routing.

Only discarded blocks supply keys/values. Routing is computed from reference
layer memory slots, then reused at every layer. No question enters this module.
The low-rank value path avoids retaining full hidden-size attention outputs.
"""
import torch
import torch.nn.functional as F

from .consolidate import CarrierResidualCompactor, _DELTA_VAR_EPS, _l2, _select_block_indices


class SlotMemoryMerge(CarrierResidualCompactor):
    architecture = "discarded_slot_attention_v1"

    def __init__(self, *args, query_chunk_size=256, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_chunk_size = int(query_chunk_size)
        if self.query_chunk_size <= 0:
            raise ValueError("positive query chunk size required")
        # Slot logits use cosine in learned rank-dimensional space directly;
        # the old block/native-prior interpolation parameter is not needed.
        del self.route_logit

    def _attention_chunk(self, query, keys):
        return torch.softmax((query @ keys.T) / self.route_temperature, dim=-1)

    def _slot_delta(self, attention, source_layer, carrier_layer):
        carrier = carrier_layer.float()
        values = self.down(source_layer.float().flatten(0, 1))
        context = attention @ values
        carrier_values = self.down(carrier.flatten(0, 1))
        delta = self.up(F.gelu(context - carrier_values)).reshape_as(carrier)
        carrier_rms = carrier.square().mean((-2, -1), keepdim=True).sqrt()
        return delta * torch.rsqrt(delta.square().mean((-2, -1), keepdim=True)
                                  + _DELTA_VAR_EPS) * carrier_rms

    def forward(self, source, weights=None, target=None, block=None,
                goal_direction=None, select_weights=None, carrier_indices=None,
                source_allow_mask=None):
        if source.ndim != 3:
            raise ValueError("[layers, native slots, hidden] source required")
        layers, slots, hidden = source.shape
        block = self.block if block is None else int(block)
        if (block <= 0 or target is None or int(target) <= 0 or int(target) % block
                or slots % block or layers != self.num_layers or hidden != self.hidden):
            raise ValueError("invalid complete native-block source/target")
        target = int(target)
        if target >= slots:
            return source.contiguous()
        count, out_count = slots // block, target // block
        index = (_select_block_indices(source, target, block, weights, goal_direction, select_weights)
                 if carrier_indices is None else torch.as_tensor(carrier_indices,
                     dtype=torch.long, device=source.device))
        if (index.ndim != 1 or index.numel() != out_count or index.unique().numel() != out_count
                or (index < 0).any() or (index >= count).any()
                or not torch.equal(index, index.sort().values)):
            raise ValueError("chronological unique carrier indices required")
        allow = torch.ones(count, dtype=torch.bool, device=source.device)
        if source_allow_mask is not None:
            allow = torch.as_tensor(source_allow_mask, device=source.device)
            if allow.dtype != torch.bool or allow.shape != (count,) or not allow[index].all():
                raise ValueError("source mask must include every carrier")
        blocks = source.reshape(layers, count, block, hidden)
        carriers = blocks.index_select(1, index)
        discarded = allow.clone()
        discarded[index] = False
        source_indices = torch.arange(count, device=source.device)[discarded]
        self.forward_calls += 1
        self.last_gamma = float(self.gamma.detach())
        trace = dict(architecture=self.architecture, carrier_indices=index.detach().cpu().tolist(),
            allowed_source_indices=torch.arange(count, device=source.device)[allow].cpu().tolist(),
            fusion_source_indices=source_indices.cpu().tolist(), gamma=self.last_gamma,
            routing_granularity="carrier slot to discarded source slot, shared across layers",
            interpretation="routing is a computation trace, not proof of readable knowledge fusion")
        if source_indices.numel() == 0:
            self.last_residual_ratio = self.last_routing_entropy = 0.
            self.last_fusion = dict(trace, residual_ratio=0., routing_entropy=0.)
            return carriers.flatten(1, 2).contiguous()
        fusion = blocks.index_select(1, source_indices)
        ref = min(max(self.reference_layer, 0), layers - 1)
        query = _l2(self.q(_l2(carriers[ref].detach().float().flatten(0, 1))))
        keys = _l2(self.k(_l2(fusion[ref].detach().float().flatten(0, 1))))
        use_ckpt = torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters())
        import torch.utils.checkpoint as checkpoint
        chunks = []
        for begin in range(0, target, self.query_chunk_size):
            q = query[begin:begin + self.query_chunk_size]
            chunks.append(checkpoint.checkpoint(self._attention_chunk, q, keys, use_reentrant=False)
                          if use_ckpt else self._attention_chunk(q, keys))
        attention = torch.cat(chunks, dim=0)
        # Chunked score construction plus low-rank values keeps the retained
        # routing at target*discarded_slots, not target*discarded_slots*layers.
        out = torch.empty_like(carriers)
        alpha = self.gamma
        for layer in range(layers):
            args = (attention, fusion[layer], carriers[layer])
            delta = (checkpoint.checkpoint(self._slot_delta, *args, use_reentrant=False)
                     if use_ckpt else self._slot_delta(*args))
            out[layer] = (carriers[layer].float() + alpha * delta).to(source.dtype)
            if layer == ref:
                with torch.no_grad():
                    self.last_residual_ratio = float((alpha * delta).square().mean().sqrt()
                        / carriers[layer].float().square().mean().sqrt().clamp_min(1e-12))
        with torch.no_grad():
            p = attention.detach().clamp_min(1e-12)
            self.last_routing_entropy = float(-(p * p.log()).sum(-1).mean())
        self.last_fusion = dict(trace, residual_ratio=self.last_residual_ratio,
            routing_entropy=self.last_routing_entropy, attention_shape=list(attention.shape))
        return out.flatten(1, 2)
