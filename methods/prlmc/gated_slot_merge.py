"""C/D comparison: stronger local writes under one shared global RMS budget.

C keeps v2's dense layer-0 routing and shared value matrices. D changes only
the value grouping, routing and slot gates. Both are query independent, exactly
Selection at initialization, and exactly Selection when discarded sources are
removed. Stored-dtype rounding is included in the budget calculation.

v5 makes D's routing domain independent in three ways, none of which moves the
main line of the method: the pretrained 4096-D representation dominates the
routing score (the learned 32-D projection is a bounded residual view), the
slot gate additionally sees three scale-free geometry features, and the carrier
selector scores blocks in the same layer routing reads. Every added branch is
zero-initialised, so step 0 is still bit-identical to Selection.

v7 (this module, and native_skip_merge.py) removes the forced unit-RMS write --
`carrier_rms * tanh(delta / carrier_rms)` instead of normalising every delta
onto the carrier RMS -- and multiplies the gate by a sparse active-match mask so
the same budget lands on the slots that actually matched a source. The v6
classes keep their v6 numerics unchanged and only opt out of both mechanisms, so
a v6 checkpoint still reproduces bit-identically.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .slot_merge import SlotMemoryMerge
from .consolidate import _DELTA_VAR_EPS, _SELECT_LAYER, _l2

# Sealed v6 plan value (`prepare_natural_merge_v6.TRAIN_V6` variant text says
# "reject-only vs peer in-block baseline, margin .01"). The module default used
# to be 0.0 while the plan claimed .01, and build_merge() passes no kwargs, so
# the shipped behaviour was silently 0.0. One constant, referenced by the
# module, the checkpoint metadata and the tests, keeps the two in sync.
SIM_MARGIN_V6 = 0.01


def _energy(candidate, carrier):
    return torch.stack(((candidate.float() - carrier.float()).square().sum(),
                        carrier.float().square().sum()))


def _write(candidate, carrier, scale):
    return (carrier.float() + scale * (candidate.float() - carrier.float())).to(carrier.dtype)


# Sparse high-SNR gate (v7 C9): the in-block z-score of the top-1 routing
# affinity decides whether a carrier slot actually found a match. Fixed
# constants, not learned: the mask is a discrete reliability decision, and the
# learned part of the gate stays the slot/geometry logit it multiplies.
_ACTIVE_MATCH_SHARPNESS = 4.0
_ACTIVE_MATCH_OFFSET = .15


def _soft_bound(delta, carrier):
    """Bounded residual write: `carrier_rms * tanh(delta / carrier_rms)`.

    Replaces the forced unit-RMS normalisation `delta * rsqrt(mean+eps) *
    carrier_rms`, which mapped *any* magnitude -- including an exactly zero
    (zero-initialised) delta -- onto the carrier RMS, i.e. a zero-point gain of
    `1/sqrt(1e-6) = 1e3`. Three properties the old form did not have:

    * `|out| <= carrier_rms` elementwise (tanh range), so a large delta can no
      longer be amplified, only saturated;
    * `delta == 0` gives exactly `0.0`, so step 0 is bit-identical to Selection;
    * `d out / d delta` at the origin is `1`, so the background noise a slot
      carries is passed through at unit gain instead of being inflated.

    `clamp_min` rather than `+eps`: at normal magnitudes the denominator is the
    carrier RMS itself, so no spurious rescaling enters the forward pass. The
    global `_cap` budget is untouched and still bounds what is stored.
    """
    rms = carrier.square().mean(-1, keepdim=True).sqrt()
    return rms * torch.tanh(delta / rms.clamp_min(_DELTA_VAR_EPS))


def _active_match_weight(affinity_z):
    """Sparse active-match weight in (0,1) from the in-block top-1 affinity z.

    `sigmoid(4*(z + .15))`: z = -3 (a slot that found nothing) gives ~1e-5, z = 0
    gives .65, z = +2 (a sharply matched slot) gives >.999. Multiplying the base
    gate by this concentrates the same 5% global RMS budget on the top quartile
    of slots instead of spreading it over all of them.

    Detached at the call site (`rel` is already detached): the mask must not
    become a way for the gate parameters to learn the mask back.
    """
    return torch.sigmoid(_ACTIVE_MATCH_SHARPNESS * (affinity_z + _ACTIVE_MATCH_OFFSET))


def _sparse_gate_report(weight, alpha):
    """Telemetry for the sparse gate (C9 item 5). Targets are training-time
    observations (~top quartile active, local alpha .10-.20), not assertions."""
    if weight is None or alpha is None:
        return {}
    active = weight > .5
    count = int(active.sum())
    return dict(
        active_slot_fraction=float(active.float().mean()),
        alpha_active_mean=float(alpha[active].mean()) if count else 0.,
        sparse_gate_form="alpha *= sigmoid(4*(top1_affinity_z+.15)), clamped to [0, gamma_max]",
    )


class BudgetedSlotMemoryMerge(SlotMemoryMerge):
    architecture = "budgeted_dense_slot_v3"

    def __init__(self, *args, global_rms_budget=.05, initial_gate=.02, **kwargs):
        kwargs.setdefault("gamma_max", .05)
        super().__init__(*args, **kwargs)
        self.global_rms_budget = float(global_rms_budget)
        if not 0 < initial_gate < self.gamma_max or not 0 < self.global_rms_budget < 1:
            raise ValueError("positive interior initial gate and RMS budget required")
        with torch.no_grad():
            self.gamma_logit.fill_(math.log(initial_gate / (self.gamma_max - initial_gate)))

    def _cap(self, candidate, carriers):
        use_ckpt = torch.is_grad_enabled() and candidate.requires_grad
        energies = []
        for layer in range(candidate.shape[0]):
            args = (candidate[layer], carriers[layer])
            energies.append(checkpoint(_energy, *args, use_reentrant=False)
                            if use_ckpt else _energy(*args))
        residual_sq, carrier_sq = torch.stack(energies).sum(0).unbind()
        # Leave room for the final cast to the physical pool's dtype. This is
        # common to C and D and does not change the declared 5% upper bound.
        margin = torch.finfo(carriers.dtype).eps * (1 + self.global_rms_budget)
        limit = max(0., self.global_rms_budget - margin)
        scale = (limit * torch.sqrt(carrier_sq.clamp_min(1e-24)) /
                 torch.sqrt(residual_sq + 1e-24)).clamp(max=1.)
        outputs = []
        for layer in range(candidate.shape[0]):
            args = (candidate[layer], carriers[layer], scale)
            outputs.append(checkpoint(_write, *args, use_reentrant=False)
                           if use_ckpt else _write(*args))
        out = torch.stack(outputs)
        with torch.no_grad():
            actual = torch.stack([_energy(o, c) for o, c in zip(out, carriers)]).sum(0)
            ratio = float(torch.sqrt(actual[0] / actual[1].clamp_min(1e-24)))
            if ratio > self.global_rms_budget + 1e-6:
                raise ValueError("stored residual exceeds common global RMS budget")
            self.last_residual_ratio = ratio
            self.last_fusion.update(global_rms_budget=self.global_rms_budget,
                budget_scale=float(scale.detach()), residual_ratio=ratio,
                budget_scope="all layers and output slots; measured in stored dtype",
                local_gate_max=self.gamma_max)
        return out

    def forward(self, source, weights=None, **kwargs):
        candidate = super().forward(source, weights=weights, **kwargs)
        if candidate.shape == source.shape:
            return candidate
        index = torch.tensor(self.last_fusion["carrier_indices"], device=source.device)
        block = int(kwargs.get("block") or self.block)
        carriers = source.reshape(self.num_layers, -1, block, self.hidden).index_select(1, index).flatten(1, 2)
        return self._cap(candidate, carriers)

    def state_report(self):
        return dict(self.last_fusion, carrier_rank=self.rank,
                    forward_calls=self.forward_calls, gamma=self.last_gamma)


class GraftedSlotMemoryMerge(BudgetedSlotMemoryMerge):
    architecture = "grouped_gated_sparse_slot_v3"

    def __init__(self, *args, num_groups=8, reference_layer=16, topk=8,
                 native_prior_weight=.6, route_mix_temperature=.04,
                 balance_strength=.15, **kwargs):
        kwargs.setdefault("gamma_max", .20)
        super().__init__(*args, **kwargs)
        self.num_groups = int(num_groups)
        self.reference_layer = int(reference_layer)
        self.topk = int(topk)
        self.native_prior_weight = float(native_prior_weight)
        self.route_mix_temperature = float(route_mix_temperature)
        self.balance_strength = float(balance_strength)
        if not 1 <= self.num_groups <= self.num_layers or self.num_layers % self.num_groups:
            raise ValueError("groups must evenly divide reader layers")
        if not 0 <= self.reference_layer < self.num_layers or self.topk <= 0:
            raise ValueError("valid reference layer and positive top-k required")
        if not 0. <= self.native_prior_weight <= 1. or self.route_mix_temperature <= 0:
            raise ValueError("valid native-prior weight and routing temperature required")
        if not 0. <= self.balance_strength < 1.:
            raise ValueError("load-balance strength must be a small non-negative number")
        del self.down, self.up, self.gamma_logit
        self.down_weight = nn.Parameter(torch.empty(self.num_groups, self.rank, self.hidden))
        self.up_weight = nn.Parameter(torch.zeros(self.num_groups, self.hidden, self.rank))
        for weight in self.down_weight:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.slot_gate = nn.Linear(self.hidden, 1)
        nn.init.zeros_(self.slot_gate.weight)
        # Routing quality gate: three dimensionless per-slot geometry features
        # (top-1 affinity, top-k sharpness, in-block redundancy). Zero
        # initialised, so the gate starts as the hidden-only gate and step 0
        # stays exactly Selection.
        self.rel_gate = nn.Linear(3, 1, bias=False)
        nn.init.zeros_(self.rel_gate.weight)
        # Set by every forward before routing; a plain attribute so it never
        # enters the checkpoint or the state digest.
        self._balance_offset = None
        self._key_mask = None
        # Sparse-gate telemetry for the current forward (plain attributes, so
        # they never enter the checkpoint or the state digest).
        self._active_weight = None
        self._last_alpha = None
        initial_gate = kwargs.get("initial_gate", .02)
        nn.init.constant_(self.slot_gate.bias, math.log(initial_gate / (self.gamma_max - initial_gate)))

    def _sparse_attention(self, query, keys, q_nat=None, k_nat=None, key_mask=None):
        # Two cosine views of the same (carrier slot, discarded slot) pair:
        #   * q_nat @ k_nat.T  -- the pretrained 4096-D representation itself,
        #     domain independent but noisy;
        #   * query @ keys.T   -- the learned 32-D projection, sharp but fitted
        #     on the training domain only.
        # The pretrained view carries the larger weight so a domain shift cannot
        # pull routing away from what the frozen reader already considers
        # similar. A per-key column log-mean term pushes routing away from few
        # overloaded keys. Native views and the prefix mask are explicit
        # arguments so chunked checkpoint recomputation stays a pure function of
        # its inputs.
        if q_nat is None:
            logits = (query @ keys.T) / max(1e-6, float(self.route_temperature))
        else:
            logits = (self.native_prior_weight * (q_nat @ k_nat.T)
                      + (1. - self.native_prior_weight) * (query @ keys.T)
                      ) / self.route_mix_temperature
        balance = getattr(self, "_balance_offset", None)
        if balance is not None:
            # Per-key load-balance prior, computed once over the full query set
            # in forward and detached: chunks stay pure functions of their
            # inputs, so results are independent of query_chunk_size.
            logits = logits + balance
        if key_mask is not None:
            # True marks the leading "Context:" prefix slots of each block;
            # they carry template text, not content, and must never be fused.
            logits = logits + key_mask.float() * (-1e30)
        values, indices = logits.topk(min(self.topk, keys.shape[0]), dim=-1)
        # Only k weights are retained. Dense logits are transient/chunked.
        # The third return value is the top-1 logit per query: the raw affinity
        # feature of the geometry gate, not a renormalised weight.
        return torch.softmax(values, dim=-1), indices, values[:, 0]

    def group_gain(self, group):
        """Optional per-layer-group scalar; the native skip arm overrides this."""
        return None

    def _group_delta(self, probabilities, indices, source_layer, carrier_layer,
                     down_weight, up_weight, group_gain=None):
        values = F.linear(source_layer.float().flatten(0, 1), down_weight)
        # [queries,k,rank], never [queries,k,hidden].
        context = (values[indices] * probabilities.unsqueeze(-1)).sum(1)
        carrier = carrier_layer.float().flatten(0, 1)
        delta = F.linear(F.gelu(context - F.linear(carrier, down_weight)), up_weight)
        # v7 C5: bounded write instead of forced unit RMS. `up_weight=0` (the
        # initialisation) makes `delta` exactly 0, so this is exactly 0 too.
        return _soft_bound(delta, carrier)

    def _slot_alpha(self, gate_logits, rel, reject=None):
        """Per-slot fusion weight: base gate times the sparse active-match mask.

        `rel[..., 0]` is the in-block z-score of the top-1 routing affinity (the
        stack order in `forward` is part of the contract). The v5 lineage has no
        per-group gate and no similarity mask, so `reject` is only there for the
        v6/v7 signature; the clamp keeps `alpha` inside `[0, gamma_max]`, which
        is the local bound the global RMS `_cap` relies on.
        """
        alpha = self.gamma_max * torch.sigmoid(gate_logits)
        if reject is not None:
            alpha = alpha * reject
        weight = _active_match_weight(rel[..., 0])
        # `alpha` is [slots, 1] here (one logit per slot), the mask is [slots].
        alpha = torch.clamp(alpha * weight.unsqueeze(-1), 0., self.gamma_max)
        self._active_weight, self._last_alpha = weight, alpha.detach()
        return alpha

    def _gate_inputs(self, q_nat, probabilities, top1):
        """Hidden gate input plus the three geometry features (v5 and v6).

        Returns `(gate_in, rel, raw)`:
          gate_in -- block-wise z-scored hidden slots, input of the learnable gate
          rel     -- per-block standardised [affinity, sharpness, redundancy]
          raw     -- the same three features before standardisation (tests/telemetry)

        Centroid and redundancy are computed on the RAW normalised slots. The
        z-scored `gate_in` is block-mean-centred by construction, so its block
        centroid is the zero vector and a cosine against it measures only float
        rounding -- not the designed "how redundant is this slot with its block"
        signal.
        """
        slots = q_nat.reshape(-1, self.block, self.hidden)
        centroid = _l2(slots.mean(dim=1, keepdim=True))
        redundancy = (slots * centroid).sum(-1)
        gate_in = (slots - slots.mean(1, keepdim=True)) / (
            slots.std(1, keepdim=True) + 1e-6)
        sharpness = (probabilities[:, 0] - probabilities[:, -1]).reshape(-1, self.block)
        affinity = top1.reshape(-1, self.block)
        raw = torch.stack((affinity, sharpness, redundancy), dim=-1).detach()
        rel = (raw - raw.mean(dim=1, keepdim=True)) / (raw.std(dim=1, keepdim=True) + 1e-6)
        return gate_in, rel, raw

    def forward(self, source, weights=None, target=None, block=None,
                goal_direction=None, select_weights=None, carrier_indices=None,
                source_allow_mask=None):
        # Reuse v2's shape/carrier/mask contract without its value computation.
        from .consolidate import _select_block_indices
        self._active_weight = self._last_alpha = None
        block = self.block if block is None else int(block)
        if (source.ndim != 3 or source.shape[0] != self.num_layers or source.shape[-1] != self.hidden
                or block <= 0 or source.shape[1] % block or target is None
                or int(target) <= 0 or int(target) % block):
            raise ValueError("complete native-block source/target required")
        target = int(target)
        if target >= source.shape[1]:
            return source.contiguous()
        count = source.shape[1] // block
        index = (_select_block_indices(source, target, block, weights, goal_direction, select_weights)
                 if carrier_indices is None else torch.as_tensor(carrier_indices, device=source.device, dtype=torch.long))
        if (index.ndim != 1 or index.numel() != target // block or index.unique().numel() != index.numel()
                or (index < 0).any() or (index >= count).any() or not torch.equal(index, index.sort().values)):
            raise ValueError("chronological unique carrier indices required")
        allow = torch.ones(count, dtype=torch.bool, device=source.device)
        if source_allow_mask is not None:
            allow = torch.as_tensor(source_allow_mask, device=source.device)
            if allow.dtype != torch.bool or allow.shape != (count,) or not allow[index].all():
                raise ValueError("source mask must include every carrier")
        discarded = allow.clone(); discarded[index] = False
        source_indices = torch.arange(count, device=source.device)[discarded]
        blocks = source.reshape(self.num_layers, count, block, self.hidden)
        carriers = blocks.index_select(1, index)
        self.forward_calls += 1
        self.last_fusion = dict(architecture=self.architecture, carrier_indices=index.cpu().tolist(),
            allowed_source_indices=torch.arange(count, device=source.device)[allow].cpu().tolist(),
            fusion_source_indices=source_indices.cpu().tolist(), reference_layer=self.reference_layer,
            groups=self.num_groups, topk=self.topk, routing_granularity="carrier slot to discarded slot; shared across layers",
            interpretation="routing trace alone does not establish readable transfer")
        if not source_indices.numel():
            self.last_gamma = self.last_residual_ratio = self.last_routing_entropy = 0.
            self.last_fusion.update(gamma=0., routing_entropy=0., residual_ratio=0.,
                global_rms_budget=self.global_rms_budget)
            return carriers.flatten(1, 2).contiguous()
        fusion = blocks.index_select(1, source_indices)
        # Native view: the pretrained 4096-D slots themselves.
        q_nat = _l2(carriers[self.reference_layer].detach().float().flatten(0, 1))
        k_nat = _l2(fusion[self.reference_layer].detach().float().flatten(0, 1))
        # Learned view: a bounded residual projection of those same slots.
        query = _l2(self.q(q_nat))
        keys = _l2(self.k(k_nat))
        prefix = int(getattr(self, "prefix_slots", 0))
        if 0 < prefix < self.block:
            # True marks the leading "Context:" prefix slots of each block:
            # they carry template text, not content, and must never be fused.
            self._key_mask = (torch.arange(keys.shape[0], device=keys.device)
                              % self.block) < prefix
        else:
            self._key_mask = None
        prefix_mask = self._key_mask
        self._balance_offset = None
        with torch.no_grad():
            full = (self.native_prior_weight * (q_nat @ k_nat.T)
                    + (1. - self.native_prior_weight) * (query @ keys.T)
                    ) / self.route_mix_temperature
            if prefix_mask is not None:
                full = full + prefix_mask.float() * (-1e30)
            queries = max(1, full.shape[0])
            # Column log-mean over queries: subtract log(N) from the logsumexp so
            # the term is an O(1) logit correction instead of growing with the
            # number of carrier slots.
            column_log_mean = torch.logsumexp(full, dim=0, keepdim=True) - math.log(queries)
            self._balance_offset = -self.balance_strength * column_log_mean
        use_ckpt = torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters())
        chunks = []
        for begin in range(0, target, self.query_chunk_size):
            chunks.append(self._sparse_attention(
                query[begin:begin+self.query_chunk_size], keys,
                q_nat[begin:begin+self.query_chunk_size], k_nat, prefix_mask))
        probabilities = torch.cat([p for p, _, _ in chunks])
        indices = torch.cat([i for _, i, _ in chunks])
        top1 = torch.cat([a for _, _, a in chunks])
        # Geometry gate: the hidden-only gate cannot tell whether a carrier slot
        # actually found a matching source. These three features can, and all of
        # them are scale-free and domain agnostic:
        #   affinity   -- top-1 routing logit (did this slot find a match?)
        #   sharpness  -- p_max - p_min over the retained top-k weights
        #   redundancy -- cos(slot, block centroid); generic slots score high
        # Effects are per-block rather than per-feature: the gate decides *which
        # slots inside one block* to trust, and the global RMS cap handles level.
        gate_in, rel, _ = self._gate_inputs(q_nat, probabilities, top1)
        gate_logits = (self.slot_gate(gate_in.flatten(0, 1))
                       + self.rel_gate(rel.flatten(0, 1)))
        alpha = self._slot_alpha(gate_logits, rel.flatten(0, 1))
        outputs = []
        for layer in range(self.num_layers):
            group = layer // (self.num_layers // self.num_groups)
            args = (probabilities, indices, fusion[layer], carriers[layer],
                    self.down_weight[group], self.up_weight[group], self.group_gain(group))
            delta = (checkpoint(self._group_delta, *args, use_reentrant=False)
                     if use_ckpt else self._group_delta(*args))
            outputs.append((carriers[layer].float().flatten(0, 1) + alpha * delta).to(source.dtype))
        with torch.no_grad():
            self.last_gamma = float(alpha.mean())
            self.last_routing_entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean())
            self.last_fusion.update(gamma=self.last_gamma, gate_min=float(alpha.min()), gate_max=float(alpha.max()),
                routing_entropy=self.last_routing_entropy, sparse_attention_shape=list(indices.shape),
                native_prior_weight=self.native_prior_weight, route_mix_temperature=self.route_mix_temperature,
                balance_strength=self.balance_strength,
                gate_rel_norm=float(self.rel_gate.weight.detach().abs().mean()),
                carrier_selector_layer=int(min(_SELECT_LAYER, self.num_layers - 1)),
                soft_delta_bound="carrier_rms*tanh(delta/carrier_rms); no forced unit-RMS",
                **_sparse_gate_report(self._active_weight, self._last_alpha))
        candidate = torch.stack(outputs)
        return self._cap(candidate, carriers.flatten(1, 2))


class GraftedSlotMemoryMergeV6(GraftedSlotMemoryMerge):
    """v6: soft-bounded delta, layer-group gates, conservative similarity hard mask.

    Three changes over v5, all audited in experiments/v6-rootcause-audit.md:
      1. The per-slot forced unit-RMS normalisation is replaced by a one-sided
         soft bound: tiny raw deltas are no longer amplified to carrier RMS
         (the zero-init self-RMS first-step gain 1/sqrt(eps) disappears) while
         large deltas are still capped at carrier RMS. The global _cap budget
         check is untouched.
      2. slot_gate/rel_gate emit one logit per layer group, so shallow and deep
         layers get independent alpha instead of one gate vector shared across
         all 32 layers.
      3. A similarity hard mask: if the best native 4096-D cosine between a
         carrier slot and its routed discarded slots is below the in-block
         baseline (mean over all in-block discarded-slot candidates), that
         carrier slot's fusion weight is set to zero. Conservative reject-only:
         a low-similarity slot is rejected as merge source, never is a high
         cosine used as positive merge evidence (E7a-sim constraint).
    """

    architecture = "grouped_gated_sparse_slot_v6"

    def __init__(self, *args, sim_margin=SIM_MARGIN_V6, **kwargs):
        super().__init__(*args, **kwargs)
        self.sim_margin = float(sim_margin)
        # Per-layer-group gate: hidden z-score -> one logit per group.
        self.slot_gate = nn.Linear(self.hidden, self.num_groups)
        nn.init.zeros_(self.slot_gate.weight)
        # Geometry gate: three domain-agnostic features -> one logit per group.
        self.rel_gate = nn.Linear(3, self.num_groups, bias=False)
        nn.init.zeros_(self.rel_gate.weight)
        initial_gate = kwargs.get("initial_gate", .02)
        nn.init.constant_(self.slot_gate.bias,
                          math.log(initial_gate / (self.gamma_max - initial_gate)))

    def _group_delta(self, probabilities, indices, source_layer, carrier_layer,
                     down_weight, up_weight, group_gain=None):
        values = F.linear(source_layer.float().flatten(0, 1), down_weight)
        context = (values[indices] * probabilities.unsqueeze(-1)).sum(1)
        carrier = carrier_layer.float().flatten(0, 1)
        delta = F.linear(F.gelu(context - F.linear(carrier, down_weight)), up_weight)
        carrier_rms = carrier.square().mean(-1, keepdim=True).sqrt()
        # One-sided soft bound: delta scaled by carrier_rms only when its own RMS
        # exceeds tau*carrier_rms, otherwise passed through unscaled. Replaces the
        # forced unit-RMS normalisation that amplified a tiny (zero-init) delta
        # all the way to carrier RMS on the first step.
        # eps INSIDE the sqrt (v5 convention): at zero-init up_weight the raw
        # delta is exactly 0, and sqrt(0)+eps has zero gradient while
        # sqrt(0+eps) has a finite one -- outside placement NaNed the whole
        # backward pass (training crashed at the first gradient check).
        delta_rms = torch.sqrt(delta.square().mean(-1, keepdim=True) + _DELTA_VAR_EPS)
        tau = 1.
        scale = torch.clamp(carrier_rms / (tau * delta_rms), max=1.)
        return delta * scale

    def _slot_alpha(self, gate_logits, rel, reject=None):
        """v6 gate, frozen: one logit per layer group times the similarity mask.

        Deliberately *not* the v7 sparse mask -- v6 checkpoints must keep
        reproducing bit-identically (C6), so this override opts out of the base
        class' active-match weight and out of the clamp.
        """
        alpha = self.gamma_max * torch.sigmoid(gate_logits)
        if reject is not None:
            alpha = alpha * reject
        return alpha

    def _similarity_mask(self, q_nat, k_nat, block):
        """Reject-only per-carrier-slot mask from native 4096-D cosine.

        For each carrier slot, the best native cosine is compared with the
        peer-slots' mean best-cosine inside the same block (self excluded).
        Below that baseline (minus margin) the slot is hard-rejected. Detached: this is a
        discrete routing decision, not a differentiable gate.
        """
        with torch.no_grad():
            sim = q_nat @ k_nat.T  # [queries, keys] native cosine in [-1,1]
            best = sim.max(-1).values.reshape(-1, block)
            # In-block baseline: the mean best-cosine of the *peer* carrier
            # slots in the same block (self excluded; max >= own-key mean
            # always, so the baseline must come from peers for the comparison
            # to be able to fire at all).
            n = block
            peer_mean = (best.sum(1, keepdim=True) - best) / max(n - 1, 1)
            reject = best < peer_mean - self.sim_margin
            # 1.0 = keep, 0.0 = rejected carrier slot (alpha zeroed).
            return (1. - reject.reshape(-1).float()).to(sim.dtype)

    def forward(self, source, weights=None, target=None, block=None,
                goal_direction=None, select_weights=None, carrier_indices=None,
                source_allow_mask=None):
        from .consolidate import _select_block_indices
        block = self.block if block is None else int(block)
        if (source.ndim != 3 or source.shape[0] != self.num_layers or source.shape[-1] != self.hidden
                or block <= 0 or source.shape[1] % block or target is None
                or int(target) <= 0 or int(target) % block):
            raise ValueError("complete native-block source/target required")
        target = int(target)
        if target >= source.shape[1]:
            return source.contiguous()
        count = source.shape[1] // block
        index = (_select_block_indices(source, target, block, weights, goal_direction, select_weights)
                 if carrier_indices is None else torch.as_tensor(carrier_indices, device=source.device, dtype=torch.long))
        if (index.ndim != 1 or index.numel() != target // block or index.unique().numel() != index.numel()
                or (index < 0).any() or (index >= count).any() or not torch.equal(index, index.sort().values)):
            raise ValueError("chronological unique carrier indices required")
        allow = torch.ones(count, dtype=torch.bool, device=source.device)
        if source_allow_mask is not None:
            allow = torch.as_tensor(source_allow_mask, device=source.device)
            if allow.dtype != torch.bool or allow.shape != (count,) or not allow[index].all():
                raise ValueError("source mask must include every carrier")
        discarded = allow.clone(); discarded[index] = False
        source_indices = torch.arange(count, device=source.device)[discarded]
        blocks = source.reshape(self.num_layers, count, block, self.hidden)
        carriers = blocks.index_select(1, index)
        self.forward_calls += 1
        self.last_fusion = dict(architecture=self.architecture, carrier_indices=index.cpu().tolist(),
            allowed_source_indices=torch.arange(count, device=source.device)[allow].cpu().tolist(),
            fusion_source_indices=source_indices.cpu().tolist(), reference_layer=self.reference_layer,
            groups=self.num_groups, topk=self.topk, routing_granularity="carrier slot to discarded slot; group-gated",
            interpretation="routing trace alone does not establish readable transfer")
        if not source_indices.numel():
            self.last_gamma = self.last_residual_ratio = self.last_routing_entropy = 0.
            self.last_fusion.update(gamma=0., routing_entropy=0., residual_ratio=0.,
                global_rms_budget=self.global_rms_budget)
            return carriers.flatten(1, 2).contiguous()
        fusion = blocks.index_select(1, source_indices)
        q_nat = _l2(carriers[self.reference_layer].detach().float().flatten(0, 1))
        k_nat = _l2(fusion[self.reference_layer].detach().float().flatten(0, 1))
        query = _l2(self.q(q_nat))
        keys = _l2(self.k(k_nat))
        prefix = int(getattr(self, "prefix_slots", 0))
        if 0 < prefix < self.block:
            self._key_mask = (torch.arange(keys.shape[0], device=keys.device)
                              % self.block) < prefix
        else:
            self._key_mask = None
        prefix_mask = self._key_mask
        self._balance_offset = None
        with torch.no_grad():
            full = (self.native_prior_weight * (q_nat @ k_nat.T)
                    + (1. - self.native_prior_weight) * (query @ keys.T)
                    ) / self.route_mix_temperature
            if prefix_mask is not None:
                full = full + prefix_mask.float() * (-1e30)
            queries = max(1, full.shape[0])
            column_log_mean = torch.logsumexp(full, dim=0, keepdim=True) - math.log(queries)
            self._balance_offset = -self.balance_strength * column_log_mean
        use_ckpt = torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters())
        chunks = []
        for begin in range(0, target, self.query_chunk_size):
            chunks.append(self._sparse_attention(
                query[begin:begin+self.query_chunk_size], keys,
                q_nat[begin:begin+self.query_chunk_size], k_nat, prefix_mask))
        probabilities = torch.cat([p for p, _, _ in chunks])
        indices = torch.cat([i for _, i, _ in chunks])
        top1 = torch.cat([a for _, _, a in chunks])
        # v6 similarity hard mask: reject-only, zero alpha for slots whose best
        # native cosine sits below the in-block discarded-slot baseline.
        sim_reject = self._similarity_mask(q_nat, k_nat, self.block)
        # v6 geometry features: identical three domain-agnostic features to v5,
        # per block of block slots, all detached, then one logit per group.
        gate_in, rel, _ = self._gate_inputs(q_nat, probabilities, top1)
        alpha = self._slot_alpha(
            self.slot_gate(gate_in.flatten(0, 1)) + self.rel_gate(rel.flatten(0, 1)),
            rel.flatten(0, 1), sim_reject.unsqueeze(-1))
        outputs = []
        for layer in range(self.num_layers):
            group = layer // (self.num_layers // self.num_groups)
            args = (probabilities, indices, fusion[layer], carriers[layer],
                    self.down_weight[group], self.up_weight[group], self.group_gain(group))
            delta = (checkpoint(self._group_delta, *args, use_reentrant=False)
                     if use_ckpt else self._group_delta(*args))
            outputs.append((carriers[layer].float().flatten(0, 1) + alpha[:, group:group+1] * delta).to(source.dtype))
        with torch.no_grad():
            self.last_gamma = float(alpha.mean())
            self.last_routing_entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean())
            self.last_fusion.update(gamma=self.last_gamma, gate_min=float(alpha.min()), gate_max=float(alpha.max()),
                routing_entropy=self.last_routing_entropy, sparse_attention_shape=list(indices.shape),
                native_prior_weight=self.native_prior_weight, route_mix_temperature=self.route_mix_temperature,
                balance_strength=self.balance_strength,
                gate_rel_norm=float(self.rel_gate.weight.detach().abs().mean()),
                carrier_selector_layer=int(min(_SELECT_LAYER, self.num_layers - 1)),
                similarity_reject_fraction=float(1. - sim_reject.mean()),
                soft_delta_bound="one-sided delta_rms clamp to carrier_rms; no forced unit-RMS")
        candidate = torch.stack(outputs)
        return self._cap(candidate, carriers.flatten(1, 2))


class SupervisedDSlotMergeV6(GraftedSlotMemoryMergeV6):
    architecture = "grouped_gated_sparse_slot_v6"

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


class NativeSkipSlotMergeV6(SupervisedDSlotMergeV6):
    architecture = "grouped_gated_sparse_native_v6"

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
        carrier_rms = carrier.square().mean(-1, keepdim=True).sqrt()
        sharp = min(self.top_sharp, indices.shape[1])
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
        combined = adapted + gain * native
        # Same one-sided soft bound as the adapted path: no forced unit-RMS
        # renormalisation of the combined residual.
        # eps INSIDE the sqrt (the adapted path's convention). up_weight and
        # native_gain are both zero-initialised, so the first forward has
        # combined == 0 exactly: sqrt(0) has an infinite derivative and
        # 0 * inf NaNs the whole backward pass, while sqrt(0 + eps) has a
        # finite gradient. Outside placement is what NaNed the adapted path.
        combined_rms = torch.sqrt(
            combined.square().mean(-1, keepdim=True) + _DELTA_VAR_EPS)
        scale = torch.clamp(carrier_rms / combined_rms, max=1.)
        return combined * scale

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        self.last_fusion.update(native_gain=[float(g.detach()) for g in self.native_gain],
            native_path="per-layer-group signed gain on a top-%d sharpened full-width source average minus "
                        "carrier, routed-confidence scaled; one-sided soft RMS bound"
                        % self.top_sharp, sharp_slots=self.top_sharp)
        return output


class GraftedSlotMemoryMergeV7(GraftedSlotMemoryMergeV6):
    """v7 main line: bounded tanh write + sparse high-SNR gate.

    Two changes over v6, both audited in experiments/v7-spec-contract.md:
      1. C5: the write is `carrier_rms * tanh(delta / carrier_rms)` instead of
         the one-sided RMS clamp. A zero-initialised `up_weight` therefore emits
         exactly 0 instead of a tiny residual rescaled onto carrier RMS, and the
         zero-point gain is 1 rather than 1/sqrt(eps).
      2. C9: `alpha` is multiplied by `sigmoid(4*(top1_affinity_z + .15))`, so
         the same local/global budget lands on the slots that actually matched a
         source. Background slots are silent instead of uniformly written.

    Inherited from v6 unchanged: per-layer-group gates, the conservative
    similarity hard mask, and the global `_cap` (still `<= .05` of carrier RMS).
    """

    architecture = "grouped_gated_sparse_slot_v7"

    def _group_delta(self, probabilities, indices, source_layer, carrier_layer,
                     down_weight, up_weight, group_gain=None):
        values = F.linear(source_layer.float().flatten(0, 1), down_weight)
        context = (values[indices] * probabilities.unsqueeze(-1)).sum(1)
        carrier = carrier_layer.float().flatten(0, 1)
        delta = F.linear(F.gelu(context - F.linear(carrier, down_weight)), up_weight)
        return _soft_bound(delta, carrier)

    def _slot_alpha(self, gate_logits, rel, reject=None):
        """v6's group gate and similarity mask, times the sparse active mask.

        `rel` is [slots, 3] and `rel[..., 0]` is the in-block top-1 affinity
        z-score; `alpha` is [slots, groups], so the mask broadcasts over groups.
        The clamp is the contract bound `0 <= alpha <= gamma_max` (C9).
        """
        alpha = self.gamma_max * torch.sigmoid(gate_logits)
        if reject is not None:
            alpha = alpha * reject
        weight = _active_match_weight(rel[..., 0])
        alpha = torch.clamp(alpha * weight.unsqueeze(-1), 0., self.gamma_max)
        self._active_weight, self._last_alpha = weight, alpha.detach()
        return alpha

    def forward(self, *args, **kwargs):
        self._active_weight = self._last_alpha = None
        output = super().forward(*args, **kwargs)
        if self._last_alpha is not None:
            self.last_fusion.update(
                soft_delta_bound="carrier_rms*tanh(delta/carrier_rms); no forced unit-RMS",
                **_sparse_gate_report(self._active_weight, self._last_alpha))
        return output


class SupervisedDSlotMergeV7(GraftedSlotMemoryMergeV7):
    architecture = "grouped_gated_sparse_slot_v7"

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
