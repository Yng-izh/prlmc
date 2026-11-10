"""V16: Hierarchical Band-Aligned & Margin-Gated Relational Calibration (HBAR-Calib).

What was wrong with V15, in code
--------------------------------
Two faults, both real and both visible in the numbers rather than in the design prose:

1. NO BAND-SPECIFIC INJECTION. V15 built five per-layer relation features, then SUMMED
   the five projections into one vector (`basis = piece0 + ... + piece4`), RMS-normalised
   that sum and FiLM-modulated it, broadcasting the SAME vector to memory layers 10..31.
   Five relations went in, one relation came out.
2. THE GATE WAS CANCELED. V15 scaled the relation feature by the sink mass, then divided
   by the RMS of the cross branch before modulating. Measured on the finished V15 run,
   the sink only ever withheld 8.1% of the weight (0.910-0.923 retained on every layer,
   0.902-0.940 across rows): a near-constant 0.92 scale, not a selective gate.

V16 changes exactly those two things and keeps the rest of V13.

The five modifications
----------------------
1. CONSENSUS CANDIDATES, PER-BAND RELATIONS.
   The five reader layers no longer pick neighbours independently. Each layer's key is
   centred over the slot dimension and L2-normalised, the five similarities are averaged
   into one consensus similarity, and ONE cross-block top-4 candidate set is chosen from
   it. Each band then computes its OWN softmax over those same four candidates, in its
   own key space, and aggregates its own RAW values:

       bands[n, b, :] = sum_j w_bj * value_bj,      w_b. = softmax(s_b over the 4)

   The V15 "neighbour minus self" feature is gone; this is V13's value aggregation, kept
   per band instead of pre-mixed.

2. A REAL LAYER-TO-BAND MAP. Fixed, not learned:

       reader L12 -> memory L10-14      reader L24 -> memory L23-26
       reader L16 -> memory L15-18      reader L28 -> memory L27-31
       reader L20 -> memory L19-22      nothing   -> memory L0-9 (self branch only)

   Memory layer `l` reads band `BAND_OF_LAYER[l]` and no other band; a layer below the
   cutoff reads none. There is no path by which one band's vector reaches another band's
   layers: `cross_term` builds a `[chunk_layers, slots, rank]` tensor one memory layer at
   a time from that layer's own band.

   Two-level projection inside a band, both zero-initialised:

       u = band_proj[b](ctx_b) + layer_proj[l](ctx_b)

   so the band shares one main map and every target layer can still learn its own
   direction.

3. POST-PROJECTION MARGIN GATE. Two factors, both multiplicative on `u` AFTER the
   projection, with nothing normalising the product afterwards:

       omega = sigmoid(kappa[type,band] * (margin[band] - tau[type,band]))   confidence
       eta   = sigmoid(eta_logit[type, layer])                              acceptance
       z     = z_self + omega * eta * u

   `margin` is the relation significance of this slot IN THIS BAND:
   the top-4 weighted similarity minus the mean similarity over all LEGAL cross-block
   slots (same-block positions are excluded from the mean, not counted as -2). Core and
   graft carry separate thresholds (0.28 / 0.18) and separate acceptance, and different
   bands and different memory layers get different values.

4. V13'S ADDITIVE LOW-RANK SPINE. `down`, `up_core`, `up_graft`, `cond_down`, `cond_up`,
   `gate_logit`, `type_gate_logit`, the per-type ceilings and `apply_delta` are V13's,
   unchanged, including the conditional FiLM inside the SELF branch -- that one stays;
   only V15's relation FiLM is removed.

5. INITIALISATION AND GRADIENT ROUTE. The self branch is loaded from the AnchorOnly
   checkpoint (core half identity, graft half = AnchorOnly), both cross projections are
   zero, so `u = 0` at update 0 and the module is exactly V11's calibrator. The gate is
   multiplicative on a zero vector, so gate parameters cannot have a gradient at step 1;
   the projections can (`dL/dW = dL/du * gate * ctx`, with gate = omega*eta != 0). Step 1
   trains the projections, step 2 onward the gates too -- the trainer enforces exactly
   that order rather than demanding something impossible.

Everything the relation pass computes is a frozen-reader quantity (`@torch.no_grad()`);
the projections and the gate parameters are the only trainable things here.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from methods.prlmc.core_fusion import apply_delta
from methods.prlmc.cross_block_calibration import conditions

SCHEMA = "band_relation/1"
KV_LAYERS = 5
KV_LAYER_DIM = 8 * 128                     # 8 kv heads x 128 = 1,024 per reader layer
DEFAULT_KV_DIM = KV_LAYERS * KV_LAYER_DIM  # 5,120
READER_LAYERS = (12, 16, 20, 24, 28)
# (first memory layer, last memory layer, inclusive) reached by each reader layer
BAND_LAYERS = ((10, 14), (15, 18), (19, 22), (23, 26), (27, 31))
SHALLOW_CUTOFF = 10                        # memory layers 0..9: self branch only
NUM_MEMORY_LAYERS = 32
CORE_MARGIN = 0.28
GRAFT_MARGIN = 0.18
GATE_KAPPA = 12.0
ETA_INIT = 1.5


def band_of_layer(layer):
    """Memory layer index -> band index, or None for a protected shallow layer."""
    for band, (first, last) in enumerate(BAND_LAYERS):
        if int(first) <= int(layer) <= int(last):
            return band
    return None


def band_members(band):
    first, last = BAND_LAYERS[int(band)]
    return tuple(range(int(first), int(last) + 1))


# The fixed map, materialised once so a checkpoint can be compared against it and the
# report can print it: {12: 0, 16: 1, 20: 2, 24: 3, 28: 4}.
READER_TO_BAND = {layer: band for band, layer in enumerate(READER_LAYERS)}
LAYER_TO_BAND = {layer: band_of_layer(layer) for layer in range(NUM_MEMORY_LAYERS)}


def _normalize(x, eps=1e-6):
    return x.float() / x.float().norm(dim=-1, keepdim=True).clamp_min(eps)


@torch.no_grad()
def band_contexts(window, meta, k=4, chunk=256, return_report=False,
                  kv_layers=KV_LAYERS, kv_layer_dim=KV_LAYER_DIM):
    """Consensus top-4 candidates, then a per-band relation for every slot.

    Returns `bands` `[slots, 5, 1024]` (band relation vectors, raw values, not differences),
    `margins` `[slots, 5]` (top-4 weighted similarity minus the legal cross-block mean),
    and optionally a report with the neighbour statistics and the candidate overlap
    against V13's own top-4.
    """
    scorer = meta.get("scorer")
    if scorer is None:
        raise ValueError("the metadata carries no reader scorer: the band relations "
                         "cannot be computed")
    slots = int(window.shape[1])
    reading, _ = scorer.reading(window, torch.arange(slots, device=window.device), None)
    if int(kv_layers) != len(BAND_LAYERS):
        raise ValueError(f"the band map has {len(BAND_LAYERS)} bands, got {kv_layers}")
    width = int(reading.key.shape[-1])
    expected = int(kv_layers) * int(kv_layer_dim)
    if width != expected:
        raise ValueError(f"the reader key is {width} wide, V16 expects {kv_layers} x "
                         f"{kv_layer_dim} = {expected}")
    key = reading.key.float().view(slots, int(kv_layers), int(kv_layer_dim))
    value = reading.value.float().view(slots, int(kv_layers), int(kv_layer_dim))
    # Centre over the SLOT dimension, then L2-normalise: a slot's key is compared against
    # the population of keys, so the shared component carries no association.
    key_band = _normalize(key - key.mean(dim=0, keepdim=True))
    # V13's own space, kept only to measure how far the candidate sets diverge.
    key_v13 = _normalize(reading.key.float())
    block = meta["source_block"]
    keep = min(int(k), slots - 1)
    contexts = [[] for _ in range(int(kv_layers))]
    margins_out = []
    overlap_rows, similarity_rows, spread_rows, background_rows = [], [], [], []
    for start in range(0, slots, chunk):
        stop = min(start + chunk, slots)
        same_block = block[start:stop].unsqueeze(1) == block.unsqueeze(0)
        legal = ~same_block
        legal_count = legal.sum(dim=-1).clamp_min(1).float()
        per_band = []
        for band in range(int(kv_layers)):
            similarity = key_band[start:stop, band] @ key_band[:, band].t()
            per_band.append(similarity.masked_fill(same_block, -2.0))
        consensus = torch.stack(per_band, dim=0).mean(dim=0)
        index = consensus.topk(keep, dim=-1).indices          # [rows, K] shared candidates
        margin_columns = []
        for band in range(int(kv_layers)):
            similarity = per_band[band]
            picked = similarity.gather(1, index)               # [rows, K]
            weights = torch.softmax(picked, dim=-1)
            gathered = value[:, band].index_select(
                0, index.reshape(-1)).reshape(stop - start, keep, -1)
            contexts[band].append((gathered * weights.unsqueeze(-1)).sum(dim=1))
            weighted = (weights * picked).sum(dim=-1)
            # The background mean uses ONLY legal cross-block slots: the same-block
            # positions were masked to -2 for the top-k and must not enter the mean.
            background = similarity.masked_fill(~legal, 0.0).sum(dim=-1) / legal_count
            margin_columns.append(weighted - background)
            similarity_rows.append(picked.mean(dim=-1))
            background_rows.append(background)
            spread_rows.append(block.index_select(0, index.reshape(-1))
                               .reshape(stop - start, keep).float().std(dim=-1))
            del similarity, picked, gathered, weights
        margins_out.append(torch.stack(margin_columns, dim=-1))
        # How far V16's candidates are from the ones V13 would have picked. Measured,
        # never assumed: centring the keys changes the ranking.
        similarity_v13 = key_v13[start:stop] @ key_v13.t()
        similarity_v13 = similarity_v13.masked_fill(same_block, -2.0)
        index_v13 = similarity_v13.topk(keep, dim=-1).indices
        overlap_rows.append((index.unsqueeze(-1) == index_v13.unsqueeze(1)).any(dim=-1)
                            .float().mean(dim=-1))
        del per_band, consensus, similarity_v13, same_block, legal
    bands = torch.stack([torch.cat(contexts[band], dim=0)
                         for band in range(int(kv_layers))], dim=1)
    margins = torch.cat(margins_out, dim=0)
    if not return_report:
        return bands, margins
    report = dict(
        k=keep, layers=int(kv_layers), layer_dim=int(kv_layer_dim),
        reader_layers=list(READER_LAYERS), band_layers=[list(pair) for pair in BAND_LAYERS],
        candidate_overlap_with_v13=float(torch.cat(overlap_rows).mean()),
        neighbour_similarity_mean=float(torch.cat(similarity_rows).mean()),
        neighbour_block_spread_mean=float(torch.cat(spread_rows).mean()),
        background_similarity_mean=float(torch.cat(background_rows).mean()),
        band_rms=[float(bands[:, band].square().mean().sqrt())
                  for band in range(int(kv_layers))],
        margin_mean=[float(margins[:, band].mean()) for band in range(int(kv_layers))],
        margin_std=[float(margins[:, band].std()) for band in range(int(kv_layers))],
        margin_min=[float(margins[:, band].min()) for band in range(int(kv_layers))],
        margin_max=[float(margins[:, band].max()) for band in range(int(kv_layers))],
        note="candidates are chosen once from the five-layer consensus; each band then "
             "softmaxes its own similarity over those four and aggregates its own values")
    return bands, margins, report


class BandAlignedRelationResidual(nn.Module):
    """V13's low-rank spine plus a band-specific, margin-gated cross-block injection."""

    uses_layers = True

    def __init__(self, hidden_size=4096, rank=64, max_ratio=0.5, max_ratio_core=0.15,
                 gate_init=0.0, cond_size=3, cond_rank=32, kv_layers=KV_LAYERS,
                 kv_layer_dim=KV_LAYER_DIM, k_neighbors=4, shallow_cutoff=SHALLOW_CUTOFF,
                 core_margin=CORE_MARGIN, graft_margin=GRAFT_MARGIN,
                 gate_kappa=GATE_KAPPA, eta_init=ETA_INIT,
                 core_weight=2.0, graft_weight=0.5):
        super().__init__()
        if rank <= 0 or hidden_size <= 0:
            raise ValueError("rank and hidden_size must be positive")
        if max_ratio_core > max_ratio:
            raise ValueError("the core ceiling must not exceed the graft ceiling")
        if int(shallow_cutoff) == 0:
            raise ValueError("a shallow cutoff of 0 would ask for a band for every layer")
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
        # -- the self branch: V13's, verbatim -------------------------------------
        self.down = nn.Parameter(torch.empty(self.rank, self.hidden_size))
        self.up_graft = nn.Parameter(torch.zeros(self.hidden_size, self.rank))
        self.up_core = nn.Parameter(torch.zeros(self.hidden_size, self.rank))
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        self.type_gate_logit = nn.Parameter(torch.zeros(()))
        self.cond_down = nn.Parameter(torch.empty(self.cond_rank, self.cond_size))
        self.cond_up = nn.Parameter(torch.zeros(2 * self.rank, self.cond_rank))
        nn.init.orthogonal_(self.down)
        nn.init.orthogonal_(self.cond_down, gain=0.1)
        # -- the cross branch: one map per band, one per target memory layer -------
        self.band_proj = nn.ModuleList([
            nn.Linear(self.kv_layer_dim, self.rank, bias=False)
            for _ in range(self.kv_layers)])
        self.layer_proj = nn.ModuleList([
            nn.Linear(self.kv_layer_dim, self.rank, bias=False)
            for _ in range(NUM_MEMORY_LAYERS - self.shallow_cutoff)])
        for projection in list(self.band_proj) + list(self.layer_proj):
            nn.init.zeros_(projection.weight)
        # -- the gate: per (type, band) confidence and per (type, layer) acceptance -
        # kappa goes through softplus, so storing 12.0 gives an effective 12.000006.
        self.gate_kappa = nn.Parameter(torch.full((2, self.kv_layers), float(gate_kappa)))
        tau = torch.empty((2, self.kv_layers))
        tau[0].fill_(float(core_margin))
        tau[1].fill_(float(graft_margin))
        self.gate_tau = nn.Parameter(tau)
        self.eta_logit = nn.Parameter(
            torch.full((2, NUM_MEMORY_LAYERS), float(eta_init)))
        # The constructor's values, kept so `config()` describes how the module was BUILT
        # rather than what the gates drifted to; the state dict carries the trained ones.
        self.init_gate = dict(core_margin=float(core_margin),
                              graft_margin=float(graft_margin),
                              gate_kappa=float(gate_kappa), eta_init=float(eta_init))

    # -- configuration ----------------------------------------------------------
    def config(self):
        return dict(hidden_size=self.hidden_size, rank=self.rank,
                    max_ratio=self.max_ratio, max_ratio_core=self.max_ratio_core,
                    gate_init=0.0, cond_size=self.cond_size, cond_rank=self.cond_rank,
                    kv_layers=self.kv_layers, kv_layer_dim=self.kv_layer_dim,
                    k_neighbors=self.k_neighbors, shallow_cutoff=self.shallow_cutoff,
                    core_margin=self.init_gate["core_margin"],
                    graft_margin=self.init_gate["graft_margin"],
                    gate_kappa=self.init_gate["gate_kappa"],
                    eta_init=self.init_gate["eta_init"],
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
        """This module's own relation rule -- the only place that knows the band map.

        The payload is `(bands, margins)`; with `return_report=True` it comes back as
        `(payload, report)`, the shape every call site already expects from V13's
        `neighbours()`.
        """
        result = band_contexts(window, meta, k=self.k_neighbors,
                               return_report=return_report, kv_layers=self.kv_layers,
                               kv_layer_dim=self.kv_layer_dim)
        if not return_report:
            return result
        bands, margins, report = result
        return (bands, margins), report

    @torch.no_grad()
    def load_anchor_only(self, state_dict, config=None):
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
        if float(torch.cat([p.weight.detach().reshape(-1)
                            for p in self.band_proj]).abs().max()) != 0.0:
            raise ValueError("the band projections must start at zero")
        return self

    # -- the cross branch -------------------------------------------------------
    def gate(self, graft, margin, layer):
        """`omega * eta` for one memory layer, `[slots]`; zero below the cutoff."""
        band = band_of_layer(layer)
        if band is None:
            return torch.zeros_like(margin)
        kappa = F.softplus(self.gate_kappa[:, band].float())
        tau = self.gate_tau[:, band].float()
        omega = torch.where(graft,
                            torch.sigmoid(kappa[1] * (margin - tau[1])),
                            torch.sigmoid(kappa[0] * (margin - tau[0])))
        eta = torch.where(graft,
                          torch.sigmoid(self.eta_logit[1, int(layer)].float()),
                          torch.sigmoid(self.eta_logit[0, int(layer)].float()))
        return omega * eta

    def cross_term(self, bands, margins, cond, layer_ids):
        """`[layers_in_chunk, slots, rank]`, one band per memory layer, gated after the map."""
        if bands is None or margins is None:
            raise ValueError("this module is band-aligned: it needs the band relations")
        layer_ids = [int(layer) for layer in layer_ids.reshape(-1).tolist()]
        graft = cond[:, 2].float() > 0.5
        out = bands.new_zeros((len(layer_ids), int(bands.shape[0]), self.rank))
        for position, layer in enumerate(layer_ids):
            band = band_of_layer(layer)
            if band is None:
                continue                      # L0-9: strictly no cross-block injection
            context = bands[:, band, :].float()
            projected = (self.band_proj[band](context)
                         + self.layer_proj[layer - self.shallow_cutoff](context))
            factor = self.gate(graft, margins[:, band].float(), layer)
            out[position] = factor.unsqueeze(-1) * projected
        return out

    # -- the residual -----------------------------------------------------------
    def forward(self, slots, cond, relation=None, layer_ids=None):
        if slots.shape[-1] != self.hidden_size:
            raise ValueError(f"expected hidden {self.hidden_size}, got {slots.shape[-1]}")
        if cond.shape != (slots.shape[1], self.cond_size):
            raise ValueError(f"expected conditions {(slots.shape[1], self.cond_size)}, "
                             f"got {tuple(cond.shape)}")
        dtype = slots.dtype
        x = slots.float()
        bottleneck = F.gelu(F.linear(x, self.down.float()))
        film = F.linear(F.gelu(F.linear(cond.float(), self.cond_down.float())),
                        self.cond_up.float())
        scale = torch.tanh(film[:, : self.rank])
        shift = torch.tanh(film[:, self.rank:])
        bottleneck = bottleneck * (1.0 + scale.unsqueeze(0)) + shift.unsqueeze(0)
        if relation is not None:
            if layer_ids is None:
                raise ValueError("the band injection needs the memory layer indices of "
                                 "this chunk: pass layer_ids")
            bands, margins = relation
            if bands.shape[0] != slots.shape[1]:
                raise ValueError("one band relation row per slot")
            # Added to the bottleneck AFTER the gate, with nothing normalising the
            # product afterwards: whatever the gate decides is what reaches the head.
            bottleneck = bottleneck + self.cross_term(bands, margins, cond, layer_ids)
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
    """Every slot, every memory layer, with the band routing and the gates applied."""
    cond = conditions(meta, window.shape[1])
    relation_report = None
    if relation is None:
        relation, relation_report = module.relations(window, meta, return_report=True)
    bands, margins = relation
    pieces = []
    delta_sq = slot_sq = 0.0
    per_type = {0: [0.0, 0.0], 1: [0.0, 0.0]}
    cross_sq = 0.0
    cross_count = 0
    shallow_leak = 0.0
    for start in range(0, window.shape[0], layer_chunk):
        stop = min(start + layer_chunk, window.shape[0])
        chunk = window[start:stop]
        layer_ids = torch.arange(start, stop, device=window.device)
        cross = module.cross_term(bands, margins, cond, layer_ids)
        delta = module(chunk, cond, relation, layer_ids)
        calibrated = apply_delta(chunk, delta.to(chunk.dtype))
        pieces.append(calibrated)
        weight = delta.detach().float().square()
        delta_sq += float(weight.sum())
        slot_sq += float(chunk.detach().float().square().sum())
        cross = cross.detach().float()
        active = torch.as_tensor([band_of_layer(int(layer)) is not None
                                  for layer in range(start, stop)])
        if bool(active.any()):
            injected = cross[active]
            cross_sq += float(injected.square().sum())
            cross_count += int(injected.numel())
        if bool((~active).any()):
            shallow_leak = max(shallow_leak, float(cross[~active].abs().max()))
        for kind in (0, 1):
            mask = (meta["slot_type"] == kind)
            if bool(mask.any()):
                per_type[kind][0] += float(weight[:, mask].sum())
                per_type[kind][1] += float(chunk.detach().float()[:, mask].square().sum())
        del delta, cross
    calibrated_all = torch.cat(pieces, dim=0)
    graft_mask = (meta["slot_type"] > 0.5)
    gates = {}
    for band in range(bands.shape[1]):
        for layer in band_members(band):
            value = module.gate(graft_mask, margins[:, band].float(), layer).float()
            gates.setdefault(band, []).append(value)
    gate_report = {}
    for band, values in gates.items():
        stacked = torch.stack(values, dim=0)          # [layers in band, slots]
        core_values = stacked[:, ~graft_mask]
        graft_values = stacked[:, graft_mask]
        gate_report[f"band_{band}"] = dict(
            layers=list(band_members(band)),
            gate_core_mean=float(core_values.mean()),
            gate_core_std=float(core_values.std()),
            gate_graft_mean=float(graft_values.mean()),
            gate_graft_std=float(graft_values.std()))
    info = dict(
        slots=int(window.shape[1]),
        core_slots=int((meta["slot_type"] == 0).sum()),
        graft_slots=int((meta["slot_type"] == 1).sum()),
        folded_slots=0,
        relative_delta=(delta_sq / max(slot_sq, 1e-24)) ** 0.5,
        core_relative_delta=(per_type[0][0] / max(per_type[0][1], 1e-24)) ** 0.5,
        graft_relative_delta=(per_type[1][0] / max(per_type[1][1], 1e-24)) ** 0.5,
        use_cross=True, neighbor_mode="band-margin",
        k_neighbors=int(module.k_neighbors), shallow_cutoff=int(module.shallow_cutoff),
        reader_layers=list(READER_LAYERS), band_layers=[list(pair) for pair in BAND_LAYERS],
        cross_injection_rms=(cross_sq / max(cross_count, 1)) ** 0.5,
        shallow_cross_leak=shallow_leak,
        band_proj_norm=[float(projection.weight.detach().float().norm())
                        for projection in module.band_proj],
        layer_proj_norm=float(torch.stack([
            projection.weight.detach().float().norm()
            for projection in module.layer_proj]).mean()),
        gate=float(torch.sigmoid(module.gate_logit.detach().float())),
        gate_bands=gate_report)
    if relation_report is not None:
        info["relation"] = relation_report
    if return_report:
        return calibrated_all, info, relation
    return calibrated_all, info


def load_band(path, hidden_size, rank=None):
    """Rebuild a `BandAlignedRelationResidual` from its checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA + "/checkpoint":
        raise ValueError(f"{path} is not a {SCHEMA} checkpoint")
    config = dict(payload["config"])
    if int(config.get("hidden_size", -1)) != int(hidden_size):
        raise ValueError(f"{path} was built for hidden {config.get('hidden_size')}, "
                         f"the reader is {hidden_size}")
    if rank is not None and int(config.get("rank", -1)) != int(rank):
        raise ValueError(f"{path} has rank {config.get('rank')}, expected {rank}")
    module = BandAlignedRelationResidual(**config)
    module.load_state_dict(payload["state_dict"], strict=True)
    geometry = dict(payload.get("geometry") or {})
    recorded = geometry.get("layer_to_band")
    expected = {str(layer): band for layer, band in LAYER_TO_BAND.items()}
    if recorded is not None and {str(k): v for k, v in dict(recorded).items()} != expected:
        raise ValueError(f"{path} was trained with a different layer-to-band map: "
                         f"{recorded}")
    return module, config, geometry
