"""V15-LSRM: layered sparse relation modulation of the 2,304-slot window.

What this replaces
------------------
V13 built ONE cross-block context for the whole window: the five utility layers' keys
(12, 16, 20, 24, 28) concatenated into a 5,120-wide vector, one kind-blind top-4 over
that vector, the neighbours' values summed, and the result ADDED into the residual
bottleneck. Measured limits, both from the V13/V14 rounds:

* the five layers never had separate relations -- a slot's neighbours were chosen in a
  space where every layer votes at once, so a neighbour that matters at layer 28 and one
  that matters at layer 12 are averaged into the same similarity;
* the injected signal is a weighted sum of neighbour values, i.e. "what is nearby", with
  no reference to the slot itself, so a neighbour that is merely similar-looking adds as
  much as one that carries what this slot lacks;
* a direct addition has no way to say "inject less": V13 could only shrink everything
  through one `cross_proj`, which is why it needed a 30x learning rate to move at all;
* all 32 memory layers received the same injection, including the shallow ones where the
  reader is still resolving surface form.

What V15 does instead
---------------------
    per layer      five INDEPENDENT top-4 relations, one per utility layer, each in its
                   own 1,024-wide key space: s_ij^l = cos(key_i^l, key_j^l), same-block
                   masked. Five layers, five neighbour sets.
    sink           per layer, a sink logit is appended to the four similarities before the
                   softmax and then DROPPED. The retained mass m_i^l = sum of the four
                   weights is therefore <= 1: when no neighbour is relevant the mass the
                   branch can inject goes to zero instead of being renormalised back to
                   one. The sink logit is not a constant: it is initialised from the
                   per-layer top-4 similarity distribution of the training set
                   (`init_sink`), and both it and the temperature are recorded in the
                   checkpoint so a run can be read back.
    difference     the relation feature is not the neighbours' content but the neighbours
                   MINUS the slot:  f_i^l = sum_j w_ij v_j^l - m_i^l * v_i^l
                                 = m_i^l * (mean neighbour value - own value)
                   so the branch carries what the slot does not already have, and it is
                   attenuated by m_i^l automatically.
    FiLM           the feature enters as a multiplicative gate and an additive offset on
                   the bottleneck:  b <- b * (1 + tanh(W_s c)) + tanh(W_h c), with c the
                   RMS-normalised sum of five per-layer projections. Multiplicative
                   gating is the thing a direct addition cannot express, and the offset
                   is bounded so one bad slot cannot exceed what the residual ceiling
                   allows anyway.
    scale          the five projections are summed and then RMS-normalised before the
                   modulation, so the cross branch cannot drown the self branch as the
                   projections grow: the modulation sees a unit-scale vector whatever
                   the norms of the five maps are.
    shallow        memory layers 0..9 keep their self branch and receive no relation term
                   (`shallow_cutoff = 10`). The reader is still resolving surface form
                   there; V13 injected into all 32.

Initialisation, and why the gradient still flows
------------------------------------------------
`mod_scale` and `mod_shift` are zero-initialised, so at update 0 the module is exactly
V11's calibrator and, on the graft half, exactly AnchorOnly -- the same construction
guarantee V13 had, checked by the trainer before the first update. `cross_proj` is NOT
zero-initialised (orthogonal), unlike V13's: V13's zero init left the whole cross channel
with no gradient on update 1 and it needed `--cross-lr 3e-3` to escape. Here the five
projections carry a live gradient from update 2 onward (`mod_*` moves on update 1, which
re-opens the path), and the trainer records `grad_cross_proj` per update so a silent zero
cannot pass as a result.

What is deliberately NOT here
-----------------------------
No per-layer learned temperature: tau and the sink threshold are fixed at the values the
training distribution implies and stored as buffers, so they are hyperparameters of the
record, not fitted parameters. Making them learnable means keeping 5 x 4 neighbour values
per slot (189 MB at 2,304 slots) inside the graph for two scalars per layer; the run must
earn that first.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from methods.prlmc.core_fusion import apply_delta
from methods.prlmc.cross_block_calibration import conditions

SCHEMA = "hierarchical_relation/1"
KV_LAYERS = 5                            # reader utility layers 12, 16, 20, 24, 28
KV_LAYER_DIM = 8 * 128                   # 8 kv heads x 128 = 1,024 per layer
DEFAULT_KV_DIM = KV_LAYERS * KV_LAYER_DIM  # 5,120, the same width V13 used
SHALLOW_CUTOFF = 10                      # memory layers 0..9 are protected
SINK_PERCENTILE = 10.0                   # per-layer percentile of top-k similarity
DEFAULT_TAU = 20.0


class _RMSNorm(nn.Module):
    """RMSNorm with a learnable scale, without depending on the torch version."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(dim)))
        self.eps = float(eps)

    def forward(self, x):
        x = x.float()
        rms = x.square().mean(dim=-1, keepdim=True).clamp_min(self.eps).sqrt()
        return (x / rms) * self.weight.float()


def _unit(x):
    return x.float() / x.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)


@torch.no_grad()
def relation_feature(window, meta, tau, sink_similarity, k=4, chunk=256,
                     percentiles=(10.0, 50.0, 90.0), return_report=False):
    """Per-utility-layer cross-block relation feature, `[slots, 5 x 1024]` float32.

    `tau` and `sink_similarity` are per-layer: the softmax is over
    `tau_l * [s_1..s_k, sink_similarity_l]`, and the sink slot is dropped after the
    softmax rather than renormalised away, so `sum_j w_j <= 1` carries "how much of the
    neighbourhood is worth injecting" into the feature itself.
    """
    scorer = meta.get("scorer")
    if scorer is None:
        raise ValueError("the metadata carries no reader scorer: the per-layer relations "
                         "cannot be computed")
    slots = int(window.shape[1])
    reading, _ = scorer.reading(window, torch.arange(slots, device=window.device), None)
    width = int(reading.key.shape[-1])
    if width != KV_LAYERS * KV_LAYER_DIM:
        raise ValueError(f"the reader key is {width} wide, V15 expects {KV_LAYERS} "
                         f"utility layers x {KV_LAYER_DIM} = {KV_LAYERS * KV_LAYER_DIM}")
    key = reading.key.float().view(slots, KV_LAYERS, KV_LAYER_DIM)
    value = reading.value.float().view(slots, KV_LAYERS, KV_LAYER_DIM)
    key_unit = _unit(key)
    block = meta["source_block"]
    keep = min(int(k), slots - 1)
    tau = torch.as_tensor(tau, dtype=torch.float64).reshape(-1)
    sink = torch.as_tensor(sink_similarity, dtype=torch.float64).reshape(-1)
    if tau.numel() != KV_LAYERS or sink.numel() != KV_LAYERS:
        raise ValueError("V15 needs one temperature and one sink threshold per utility "
                         "layer")
    # Before `init_sink` the threshold is NaN by construction. That must mean "no sink
    # yet" (weights sum to 1, the gate is open) rather than poison the feature with NaN:
    # the trainer's identity check runs before the thresholds are measured.
    sink_set = bool(torch.isfinite(sink).all())
    rows = []
    sink_mass = [[] for _ in range(KV_LAYERS)]
    retained = [[] for _ in range(KV_LAYERS)]
    similarity_mean = [[] for _ in range(KV_LAYERS)]
    spread = [[] for _ in range(KV_LAYERS)]
    top_values = [[] for _ in range(KV_LAYERS)]
    for start in range(0, slots, chunk):
        stop = min(start + chunk, slots)
        same_block = block[start:stop].unsqueeze(1) == block.unsqueeze(0)
        pieces = []
        for layer in range(KV_LAYERS):
            similarity = key_unit[start:stop, layer] @ key_unit[:, layer].t()
            similarity = similarity.masked_fill(same_block, -2.0)
            best = similarity.topk(keep, dim=-1)
            logits = best.values.double() * float(tau[layer])
            if sink_set:
                logits = torch.cat([
                    logits,
                    torch.full((stop - start, 1), float(tau[layer] * sink[layer]),
                               dtype=torch.float64, device=best.values.device),
                ], dim=-1)
            weights = torch.softmax(logits, dim=-1)[:, :keep].float()
            mass = weights.sum(dim=-1)
            gathered = value[:, layer].index_select(
                0, best.indices.reshape(-1)).reshape(stop - start, keep, -1)
            neighbour = (gathered * weights.unsqueeze(-1)).sum(dim=1)
            pieces.append(neighbour - mass.unsqueeze(-1) * value[start:stop, layer])
            sink_mass[layer].append(1.0 - mass)
            retained[layer].append(mass)
            similarity_mean[layer].append(best.values.mean(dim=-1))
            spread[layer].append(block.index_select(0, best.indices.reshape(-1))
                                 .reshape(stop - start, keep).float().std(dim=-1))
            if return_report:
                top_values[layer].append(best.values.float().reshape(-1))
            del similarity, best, gathered, weights, logits
        rows.append(torch.cat(pieces, dim=-1))
        del same_block, pieces
    feature = torch.cat(rows, dim=0)
    if not return_report:
        return feature

    def _mean(values):
        flat = torch.cat(values, dim=0)
        return float(flat.mean()) if int(flat.numel()) else None

    per_layer = {}
    for layer in range(KV_LAYERS):
        entry = dict(
            sink_mass=(_mean(sink_mass[layer]) if sink_set else None),
            retained_mass=_mean(retained[layer]),
            neighbour_similarity_mean=_mean(similarity_mean[layer]),
            neighbour_block_spread_mean=_mean(spread[layer]),
            feature_rms=float(feature[:, layer * KV_LAYER_DIM:(layer + 1) * KV_LAYER_DIM]
                              .square().mean().sqrt()))
        if top_values[layer]:
            flat = torch.cat(top_values[layer], dim=0).double()
            for percentile in percentiles:
                entry[f"similarity_p{int(percentile)}"] = float(
                    torch.quantile(flat, float(percentile) / 100.0))
        per_layer[f"utility_layer_{layer}"] = entry
    report = dict(k=keep, layers=KV_LAYERS, layer_dim=KV_LAYER_DIM,
                  tau=[float(x) for x in tau],
                  sink_similarity=([float(x) for x in sink] if sink_set else None),
                  sink_set=sink_set,
                  retained_mass_mean=_mean(retained),
                  sink_mass_mean=(_mean(sink_mass) if sink_set else None),
                  feature_rms=float(feature.square().mean().sqrt()),
                  per_layer=per_layer,
                  note="retained_mass <= 1 is the sink at work: the branch injects less "
                       "when no neighbour beats the threshold")
    return feature, report


class HierarchicalRelationResidual(nn.Module):
    """Per-layer relations, sink-gated, as a FiLM modulation of the residual bottleneck.

    `forward(slots, cond, relation, layer_ids)`: `slots` is `[layers, N, D]` for the
    memory layers in this chunk, `layer_ids` their indices in the 32-layer stack, so the
    shallow protection is a mask and not a second module.
    """

    uses_layers = True

    def __init__(self, hidden_size=4096, rank=64, max_ratio=0.5, max_ratio_core=0.15,
                 gate_init=0.0, cond_size=3, cond_rank=32, kv_layers=KV_LAYERS,
                 kv_layer_dim=KV_LAYER_DIM, k_neighbors=4, shallow_cutoff=SHALLOW_CUTOFF,
                 tau_init=DEFAULT_TAU, sink_percentile=SINK_PERCENTILE,
                 core_weight=2.0, graft_weight=0.5):
        super().__init__()
        if rank <= 0 or hidden_size <= 0:
            raise ValueError("rank and hidden_size must be positive")
        if max_ratio_core > max_ratio:
            raise ValueError("the core ceiling must not exceed the graft ceiling")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.max_ratio = float(max_ratio)
        self.max_ratio_core = float(max_ratio_core)
        self.cond_size = int(cond_size)
        self.cond_rank = int(cond_rank)
        self.kv_layers = int(kv_layers)
        self.kv_layer_dim = int(kv_layer_dim)
        self.k_neighbors = int(k_neighbors)
        self.shallow_cutoff = int(shallow_cutoff)
        self.core_weight = float(core_weight)
        self.graft_weight = float(graft_weight)
        self.down = nn.Parameter(torch.empty(self.rank, self.hidden_size))
        self.up_graft = nn.Parameter(torch.zeros(self.hidden_size, self.rank))
        self.up_core = nn.Parameter(torch.zeros(self.hidden_size, self.rank))
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        self.type_gate_logit = nn.Parameter(torch.zeros(()))
        # Random and `cond_up` zero: the FiLM is exactly 0 at init and both halves keep a
        # gradient route out (an all-zero start deadlocked once, in round 6).
        self.cond_down = nn.Parameter(torch.empty(self.cond_rank, self.cond_size))
        self.cond_up = nn.Parameter(torch.zeros(2 * self.rank, self.cond_rank))
        nn.init.orthogonal_(self.down)
        nn.init.orthogonal_(self.cond_down, gain=0.1)
        # One projection per utility layer: the relations are per layer, so the maps that
        # carry them are too. Orthogonal, NOT zero -- see the module docstring.
        self.cross_proj = nn.ModuleList([
            nn.Linear(self.kv_layer_dim, self.rank, bias=False)
            for _ in range(self.kv_layers)])
        for projection in self.cross_proj:
            nn.init.orthogonal_(projection.weight)
        self.cross_norm = _RMSNorm(self.rank)
        self.mod_scale = nn.Linear(self.rank, self.rank, bias=True)
        self.mod_shift = nn.Linear(self.rank, self.rank, bias=True)
        for modulation in (self.mod_scale, self.mod_shift):
            nn.init.zeros_(modulation.weight)
            nn.init.zeros_(modulation.bias)
        # Buffers, not parameters: fixed at the values the training distribution implies
        # and recorded in the checkpoint, so a run can be read back. `init_sink` sets the
        # second one before the first update.
        self.register_buffer("tau", torch.full((self.kv_layers,), float(tau_init)))
        self.register_buffer("sink_similarity",
                             torch.full((self.kv_layers,), float("nan")))
        self.sink_percentile = float(sink_percentile)

    # -- configuration ------------------------------------------------------
    def config(self):
        return dict(hidden_size=self.hidden_size, rank=self.rank,
                    max_ratio=self.max_ratio, max_ratio_core=self.max_ratio_core,
                    gate_init=0.0, cond_size=self.cond_size, cond_rank=self.cond_rank,
                    kv_layers=self.kv_layers, kv_layer_dim=self.kv_layer_dim,
                    k_neighbors=self.k_neighbors, shallow_cutoff=self.shallow_cutoff,
                    tau_init=float(self.tau.reshape(-1)[0]),
                    sink_percentile=self.sink_percentile,
                    core_weight=self.core_weight, graft_weight=self.graft_weight)

    def anchor_view(self):
        """`AnchorResidual` carrying this module's graft parameters, for the identity check."""
        from methods.prlmc.anchor_residual import AnchorResidual

        anchor = AnchorResidual(hidden_size=self.hidden_size, rank=self.rank,
                                max_ratio=self.max_ratio, gate_init=0.0)
        with torch.no_grad():
            anchor.down.copy_(self.down.detach())
            anchor.up.copy_(self.up_graft.detach())
            anchor.gate_logit.copy_(self.gate_logit.detach())
        return anchor

    def relations(self, window, meta, return_report=False):
        """This module's own relation rule -- the only place that knows it."""
        return relation_feature(window, meta, self.tau, self.sink_similarity,
                                k=self.k_neighbors, return_report=return_report)

    @torch.no_grad()
    def init_sink(self, percentiles):
        """Set the per-layer sink threshold from the measured top-k similarity spread.

        `percentiles[l]` is, for utility layer l, the percentile (default the 10th) of
        that layer's top-k neighbour similarities over the training windows. Putting the
        sink logit there means "a neighbour below the bottom decile of what this layer
        normally finds is worth less than the sink", which is the weakest defensible
        threshold and makes the gate measurable instead of decorative.
        """
        values = torch.as_tensor(percentiles, dtype=torch.float32).reshape(-1)
        if values.numel() != self.kv_layers:
            raise ValueError(f"expected {self.kv_layers} thresholds, got "
                             f"{values.numel()}")
        if not torch.isfinite(values).all():
            raise ValueError("the measured sink thresholds are not finite")
        self.sink_similarity.copy_(values)

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
    def _modulation(self, relation):
        """RMS-normalised sum of the five per-layer projections, `[N, rank]`."""
        basis = None
        for layer, projection in enumerate(self.cross_proj):
            chunk = relation[:, layer * self.kv_layer_dim:(layer + 1) * self.kv_layer_dim]
            piece = projection(chunk.float())
            basis = piece if basis is None else basis + piece
        return self.cross_norm(basis)

    def forward(self, slots, cond, relation=None, layer_ids=None):
        if slots.shape[-1] != self.hidden_size:
            raise ValueError(f"expected hidden {self.hidden_size}, got {slots.shape[-1]}")
        if cond.shape != (slots.shape[1], self.cond_size):
            raise ValueError(f"expected conditions {(slots.shape[1], self.cond_size)}, "
                             f"got {tuple(cond.shape)}")
        if relation is not None:
            if relation.shape[0] != slots.shape[1]:
                raise ValueError("one relation row per slot")
            if int(relation.shape[-1]) != self.kv_layers * self.kv_layer_dim:
                raise ValueError(f"expected a {self.kv_layers * self.kv_layer_dim}-wide "
                                 f"relation feature, got {relation.shape[-1]}")
        dtype = slots.dtype
        x = slots.float()
        bottleneck = F.gelu(F.linear(x, self.down.float()))
        film = F.linear(F.gelu(F.linear(cond.float(), self.cond_down.float())),
                        self.cond_up.float())
        scale = torch.tanh(film[:, : self.rank])
        shift = torch.tanh(film[:, self.rank:])
        bottleneck = bottleneck * (1.0 + scale.unsqueeze(0)) + shift.unsqueeze(0)
        if relation is not None:
            basis = self._modulation(relation)
            gain = torch.tanh(self.mod_scale(basis))
            offset = torch.tanh(self.mod_shift(basis))
            if layer_ids is None:
                active = torch.ones((1, 1, 1), device=x.device, dtype=x.dtype)
            else:
                active = (layer_ids.to(x.device) >= self.shallow_cutoff).to(x.dtype)
                active = active.view(-1, 1, 1)
            # Shallow layers keep the self branch exactly: `active` is zero there, so the
            # modulation is not applied at all rather than applied and cancelled.
            bottleneck = (bottleneck * (1.0 + active * gain.unsqueeze(0))
                          + active * offset.unsqueeze(0))
        rms = x.square().mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        graft = F.linear(bottleneck, self.up_graft.float())
        core = F.linear(bottleneck, self.up_core.float())
        mask = cond[:, 2].float().view(1, -1, 1)
        raw = mask * graft + (1.0 - mask) * core
        ceiling = mask * self.max_ratio + (1.0 - mask) * self.max_ratio_core
        gate = torch.sigmoid(self.gate_logit.float()
                             + mask * self.type_gate_logit.float()) * ceiling
        return (gate * rms * torch.tanh(raw / rms)).to(dtype)

    def regulariser_weight(self, cond):
        graft = cond[:, 2] > 0.5
        return torch.where(graft, torch.full_like(cond[:, 2], self.graft_weight),
                           torch.full_like(cond[:, 2], self.core_weight))


@torch.no_grad()
def calibrate(window, meta, module, layer_chunk=8, relation=None, return_report=False):
    """Module over ALL slots and ALL 32 memory layers, with the per-layer mask applied."""
    cond = conditions(meta, window.shape[1])
    relation_report = None
    if relation is None:
        relation, relation_report = module.relations(window, meta, return_report=True)
    pieces = []
    delta_sq = slot_sq = 0.0
    per_type = {0: [0.0, 0.0], 1: [0.0, 0.0]}
    shallow_drift = 0.0
    for start in range(0, window.shape[0], layer_chunk):
        stop = min(start + layer_chunk, window.shape[0])
        chunk = window[start:stop]
        layer_ids = torch.arange(start, stop, device=window.device)
        delta = module(chunk, cond, relation, layer_ids)
        calibrated = apply_delta(chunk, delta.to(chunk.dtype))
        pieces.append(calibrated)
        weight = delta.detach().float().square()
        delta_sq += float(weight.sum())
        slot_sq += float(chunk.detach().float().square().sum())
        if stop <= module.shallow_cutoff:
            # The shallow layers must move only through the self branch; if the relation
            # term leaked there this is where it shows up.
            shallow_drift = max(shallow_drift, float(weight.sum()))
        for kind in (0, 1):
            mask = (meta["slot_type"] == kind)
            if bool(mask.any()):
                per_type[kind][0] += float(weight[:, mask].sum())
                per_type[kind][1] += float(chunk.detach().float()[:, mask].square().sum())
        del delta
    info = dict(
        slots=int(window.shape[1]),
        core_slots=int((meta["slot_type"] == 0).sum()),
        graft_slots=int((meta["slot_type"] == 1).sum()),
        folded_slots=0,
        relative_delta=(delta_sq / max(slot_sq, 1e-24)) ** 0.5,
        core_relative_delta=(per_type[0][0] / max(per_type[0][1], 1e-24)) ** 0.5,
        graft_relative_delta=(per_type[1][0] / max(per_type[1][1], 1e-24)) ** 0.5,
        use_cross=True, neighbor_mode="layered-sink",
        k_neighbors=int(module.k_neighbors), shallow_cutoff=int(module.shallow_cutoff),
        cross_proj_norm=float(torch.stack([p.weight.detach().float().norm()
                                           for p in module.cross_proj]).mean()),
        mod_scale_norm=float(module.mod_scale.weight.detach().float().norm()),
        mod_shift_norm=float(module.mod_shift.weight.detach().float().norm()),
        shallow_relation_leak=shallow_drift,
        gate=float(torch.sigmoid(module.gate_logit.detach().float())))
    if relation_report is not None:
        info["relation"] = relation_report
    if return_report:
        return torch.cat(pieces, dim=0), info, relation
    return torch.cat(pieces, dim=0), info


def load_relation(path, hidden_size, rank=None):
    """Rebuild a `HierarchicalRelationResidual` from its checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA + "/checkpoint":
        raise ValueError(f"{path} is not a {SCHEMA} checkpoint")
    config = dict(payload["config"])
    if int(config.get("hidden_size", -1)) != int(hidden_size):
        raise ValueError(f"{path} was built for hidden {config.get('hidden_size')}, "
                         f"the reader is {hidden_size}")
    if rank is not None and int(config.get("rank", -1)) != int(rank):
        raise ValueError(f"{path} has rank {config.get('rank')}, expected {rank}")
    module = HierarchicalRelationResidual(**config)
    module.load_state_dict(payload["state_dict"], strict=True)
    return module, config, dict(payload.get("geometry") or {})
