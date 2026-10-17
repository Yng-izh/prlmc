"""C/D comparison: stronger local writes under one shared global RMS budget.

C keeps v2's dense layer-0 routing and shared value matrices. D changes only
the value grouping, routing and slot gates. Both are query independent, exactly
Selection at initialization, and exactly Selection when discarded sources are
removed. Stored-dtype rounding is included in the budget calculation.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .slot_merge import SlotMemoryMerge
from .consolidate import _DELTA_VAR_EPS, _l2


def _energy(candidate, carrier):
    return torch.stack(((candidate.float() - carrier.float()).square().sum(),
                        carrier.float().square().sum()))


def _write(candidate, carrier, scale):
    return (carrier.float() + scale * (candidate.float() - carrier.float())).to(carrier.dtype)


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

    def __init__(self, *args, num_groups=8, reference_layer=16, topk=8, **kwargs):
        kwargs.setdefault("gamma_max", .20)
        super().__init__(*args, **kwargs)
        self.num_groups = int(num_groups)
        self.reference_layer = int(reference_layer)
        self.topk = int(topk)
        if not 1 <= self.num_groups <= self.num_layers or self.num_layers % self.num_groups:
            raise ValueError("groups must evenly divide reader layers")
        if not 0 <= self.reference_layer < self.num_layers or self.topk <= 0:
            raise ValueError("valid reference layer and positive top-k required")
        del self.down, self.up, self.gamma_logit
        self.down_weight = nn.Parameter(torch.empty(self.num_groups, self.rank, self.hidden))
        self.up_weight = nn.Parameter(torch.zeros(self.num_groups, self.hidden, self.rank))
        for weight in self.down_weight:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.slot_gate = nn.Linear(self.hidden, 1)
        nn.init.zeros_(self.slot_gate.weight)
        initial_gate = kwargs.get("initial_gate", .02)
        nn.init.constant_(self.slot_gate.bias, math.log(initial_gate / (self.gamma_max - initial_gate)))

    def _sparse_attention(self, query, keys, q_raw=None, k_raw=None, key_mask=None):
        # Learned cosine and raw-projection cosine carry equal weight, on a
        # shared 0.03 temperature; the per-key logsumexp term pushes routing
        # away from few overloaded keys (load balancing). Raw projections and
        # the prefix mask are explicit arguments so chunked checkpoint
        # recomputation stays a pure function of its inputs.
        if q_raw is None:
            logits = (query @ keys.T) / max(1e-6, float(self.route_temperature))
        else:
            logits = (0.5 * (q_raw @ k_raw.T) + 0.5 * (query @ keys.T)) / 0.03
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
        return torch.softmax(values, dim=-1), indices

    def _group_delta(self, probabilities, indices, source_layer, carrier_layer,
                     down_weight, up_weight):
        values = F.linear(source_layer.float().flatten(0, 1), down_weight)
        # [queries,k,rank], never [queries,k,hidden].
        context = (values[indices] * probabilities.unsqueeze(-1)).sum(1)
        carrier = carrier_layer.float().flatten(0, 1)
        delta = F.linear(F.gelu(context - F.linear(carrier, down_weight)), up_weight)
        return (delta * torch.rsqrt(delta.square().mean(-1, keepdim=True) + _DELTA_VAR_EPS)
                * carrier.square().mean(-1, keepdim=True).sqrt())

    def forward(self, source, weights=None, target=None, block=None,
                goal_direction=None, select_weights=None, carrier_indices=None,
                source_allow_mask=None):
        # Reuse v2's shape/carrier/mask contract without its value computation.
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
            groups=self.num_groups, topk=self.topk, routing_granularity="carrier slot to discarded slot; shared across layers",
            interpretation="routing trace alone does not establish readable transfer")
        if not source_indices.numel():
            self.last_gamma = self.last_residual_ratio = self.last_routing_entropy = 0.
            self.last_fusion.update(gamma=0., routing_entropy=0., residual_ratio=0.,
                global_rms_budget=self.global_rms_budget)
            return carriers.flatten(1, 2).contiguous()
        fusion = blocks.index_select(1, source_indices)
        ref = carriers[self.reference_layer].detach().float().flatten(0, 1)
        q_raw = self.q(_l2(ref))
        k_raw = self.k(_l2(fusion[self.reference_layer].detach().float().flatten(0, 1)))
        query = _l2(q_raw)
        keys = _l2(k_raw)
        # Block-wise de-mean + z-score of the gate input pulls gate spread apart
        # before the global RMS cap can flatten it. With the zero-initialized
        # gate weight the standardized input leaves step-0 alpha unchanged.
        gate_in = _l2(ref).reshape(-1, self.block, self.hidden)
        gate_in = (gate_in - gate_in.mean(1, keepdim=True)) / (
            gate_in.std(1, keepdim=True) + 1e-6)
        alpha = self.gamma_max * torch.sigmoid(self.slot_gate(gate_in.flatten(0, 1)))
        prefix = int(getattr(self, "prefix_slots", 4))
        if 0 < prefix < self.block:
            # True marks the leading "Context:" prefix slots of each block:
            # they carry template text, not content, and must never be fused.
            self._key_mask = (torch.arange(keys.shape[0], device=keys.device)
                              % self.block) < prefix
        else:
            self._key_mask = None
        prefix_mask = self._key_mask
        self._balance_offset = None
        if q_raw is not None:
            with torch.no_grad():
                full = (0.5 * (q_raw.detach() @ k_raw.detach().T)
                        + 0.5 * (query.detach() @ keys.detach().T)) / 0.03
                queries = max(1, full.shape[0])
                self._balance_offset = (-0.5 * torch.logsumexp(full, dim=0, keepdim=True)
                                        / max(1e-6, math.log(queries)))
        use_ckpt = torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters())
        chunks = []
        for begin in range(0, target, self.query_chunk_size):
            chunks.append(self._sparse_attention(
                query[begin:begin+self.query_chunk_size], keys,
                q_raw[begin:begin+self.query_chunk_size], k_raw, prefix_mask))
        probabilities = torch.cat([p for p, _ in chunks])
        indices = torch.cat([i for _, i in chunks])
        outputs = []
        for layer in range(self.num_layers):
            group = layer // (self.num_layers // self.num_groups)
            args = (probabilities, indices, fusion[layer], carriers[layer],
                    self.down_weight[group], self.up_weight[group])
            delta = (checkpoint(self._group_delta, *args, use_reentrant=False)
                     if use_ckpt else self._group_delta(*args))
            outputs.append((carriers[layer].float().flatten(0, 1) + alpha * delta).to(source.dtype))
        with torch.no_grad():
            self.last_gamma = float(alpha.mean())
            self.last_routing_entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean())
            self.last_fusion.update(gamma=self.last_gamma, gate_min=float(alpha.min()), gate_max=float(alpha.max()),
                routing_entropy=self.last_routing_entropy, sparse_attention_shape=list(indices.shape))
        candidate = torch.stack(outputs)
        return self._cap(candidate, carriers.flatten(1, 2))
