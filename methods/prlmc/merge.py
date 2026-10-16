"""Checkpoint-compatible merge with fixed-carrier causal source ablation.

The existing operator stays untouched so live experiments retain their protocol.
No extra trainable parameter is introduced; old E6d state_dicts load strictly.
"""
from typing import Optional
import torch
from .consolidate import CarrierResidualCompactor, _select_block_indices, _l2

class SharedMemoryMerge(CarrierResidualCompactor):
    def forward(
        self,
        source: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        target: Optional[int] = None,
        block: Optional[int] = None,
        goal_direction: Optional[torch.Tensor] = None,
        select_weights: Optional[dict] = None,
        carrier_indices=None,
        source_allow_mask=None,
    ) -> torch.Tensor:
        """Reduce [L, K, d] to [L, target, d] = carriers + gamma * delta.

        `target` must be a multiple of the carrier block width and is checked
        against the carrier count that will actually be selected.
        """
        layers, k, d = source.shape
        block = self.block if block is None else int(block)
        if target is None:
            raise ValueError("CarrierResidualCompactor needs an explicit target width")
        target = int(target)
        if block <= 0 or target <= 0 or target % block != 0:
            raise ValueError(
                f"CarrierResidualCompactor: target {target} is not a multiple of block {block}"
            )

        if k % block or layers != self.num_layers or d != self.hidden:
            raise ValueError("invalid complete native-block source shape")
        b = k // block
        b_out = target // block
        if b_out >= b:
            # Nothing to compress. Return the window untouched rather than a
            # residual over every block, so the operator stays a no-op exactly
            # where `select_blocks` is one.
            return source.contiguous()

        if k % block or layers != self.num_layers or d != self.hidden:
            raise ValueError("invalid complete native-block source shape")
        index = (_select_block_indices(source, target, block, weights, goal_direction, select_weights)
                 if carrier_indices is None else torch.as_tensor(carrier_indices, device=source.device, dtype=torch.long))
        if index.ndim != 1 or index.numel() != b_out or index.unique().numel() != b_out:
            raise ValueError("carrier count or uniqueness failure")
        if (index < 0).any() or (index >= b).any() or not torch.equal(index, index.sort().values):
            raise ValueError("carriers must be chronological valid block indices")
        blocks = source.reshape(layers, b, block, d)
        carriers = blocks.index_select(1, index)
        allow = torch.ones(b, dtype=torch.bool, device=source.device)
        if source_allow_mask is not None:
            allow = torch.as_tensor(source_allow_mask, device=source.device)
            if allow.dtype != torch.bool or allow.shape != (b,) or not allow.any() or not allow[index].all():
                raise ValueError("boolean source mask must include every fixed carrier")
        allowed_indices = torch.arange(b, device=source.device)[allow]
        # Remove disabled sources BEFORE summaries, learned keys and value fusion.
        # No masked source can affect this output, even through the softmax normalizer.
        # Preserve the full-input arithmetic and old checkpoint layout by default.
        fusion_blocks = blocks if bool(allow.all()) else blocks.index_select(1, allowed_indices)

        # Routing is computed once, in the reference layer, and reused by all 32
        # layers: a native block is one jointly generated memory event, so
        # per-layer routing could splice together a block that never existed in
        # the training distribution.
        #
        # Only the reference layer is summarised, not the whole [L, b, block, d]:
        # the previous form cast every layer to fp32 (`blocks[...].to(float32)`)
        # and materialised 1.9 GB just to average one layer.
        ref = min(max(self.reference_layer, 0), layers - 1)
        src_summary = fusion_blocks[ref].detach().to(torch.float32).mean(dim=1)  # [b, d]
        car_summary = carriers[ref].detach().to(torch.float32).mean(dim=1)  # [b_out, d]

        # L2 normalisation on the routing representations only -- never on the
        # output path.
        src_unit = _l2(src_summary)
        car_unit = _l2(car_summary)

        # Native cosine prior. Carriers absorb the blocks they are already
        # closest to, which is a sensible starting point rather than noise.
        native_logits = (car_unit @ src_unit.transpose(-1, -2)) / max(
            1e-6, float(self.route_temperature)
        )  # [b_out, b]

        learned_logits = (self.q(car_unit) @ self.k(src_unit).transpose(-1, -2)) / max(
            1.0, self.rank ** 0.5
        )
        logits = native_logits + self.route_scale * learned_logits
        attn = torch.softmax(logits, dim=-1)  # [b_out, b]

        # Casting the whole [L, b_out, block, d] carrier stack to fp32 up front cost
        # 1.2 GB at the E6a shape for no reason: only one layer is alive at a time.
        alpha = self.gamma
        out = torch.empty(layers, b_out, block, d, dtype=source.dtype, device=source.device)
        residual_sq = 0.0
        carrier_sq = 0.0

        # Per-layer checkpointing. Retaining this loop's graph is what made the E6a
        # training step OOM: each layer keeps ~4 fp32 tensors of `b_out*block*d`
        # (37.7 MB at the E6a shape) for backward, so all 32 layers together hold
        # ~4.8 GB. The backbone leaves only ~4.3 GB free on a 32 GB card, which is
        # why training died at this line while inference (no graph) was comfortable.
        # Checkpointing stores just the layer inputs and recomputes one layer during
        # backward, cutting retained memory to a few hundred MB.
        use_ckpt = torch.is_grad_enabled() and any(
            p.requires_grad for p in self.parameters()
        )

        for layer in range(layers):
            carrier_layer = carriers[layer]

            if use_ckpt:
                import torch.utils.checkpoint as checkpoint

                delta = checkpoint.checkpoint(
                    self._layer_delta,
                    attn,
                    fusion_blocks[layer],
                    carrier_layer,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                delta = self._layer_delta(attn, fusion_blocks[layer], carrier_layer)

            out[layer] = (carrier_layer.to(torch.float32) + alpha * delta).to(source.dtype)
            if layer == ref:
                with torch.no_grad():
                    residual_sq = float((alpha * delta).pow(2).mean().sqrt())
                    carrier_sq = float(carrier_layer.to(torch.float32).pow(2).mean().sqrt())

        self.forward_calls += 1
        self.last_gamma = float(alpha.detach())
        self.last_residual_ratio = residual_sq / (carrier_sq + 1e-12)
        with torch.no_grad():
            p = attn.detach().clamp_min(1e-12)
            self.last_routing_entropy = float((-(p * p.log()).sum(dim=-1).mean()).detach())
        self.last_fusion = dict(carrier_indices=index.detach().cpu().tolist(),
            allowed_source_indices=allowed_indices.detach().cpu().tolist(),
            routing_weights=attn.detach().cpu().tolist(), gamma=self.last_gamma,
            residual_ratio=self.last_residual_ratio,
            interpretation="routing is a computation trace, not proof of readable knowledge fusion")
        return out.reshape(layers, b_out * block, d)
