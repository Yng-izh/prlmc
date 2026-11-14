"""V17, mechanism one: anchor the keys, release the values.

The problem this file is built around
-------------------------------------
V13's residual is bounded by ONE ceiling per slot kind (`max_ratio_core` 0.15,
`max_ratio` 0.50). That ceiling is a single number on a 4,096-dimensional change, so it
does not distinguish the part of the change the reader's key map can see from the part it
cannot. `key_safe_projection` measured what that costs: a residual that moves the address
of a slot moves the attention that reads it, and the losses from experiment 3 were
systematic rather than noisy (9 broken against 1 fixed on rows that answer without
compression).

V17 splits the residual instead of clipping it harder. For layer `l` the reader computes

    h_norm = h / rms(h) * gamma_l,     k = W_K,l h_norm,     v = W_V,l h_norm

so the key depends on `A_l h` with `A_l = W_K,l diag(gamma_l)` while the value depends on
a DIFFERENT matrix. `build_key_bases` gives an orthonormal basis of `row(A_l)` per layer;
`project_row_space(delta, basis)` is therefore the part of a residual that moves the keys,
and `delta - project_row_space(delta, basis)` is the part that does not.

    p = P_l delta            key-visible part     |p| <= key_ratio   x rms   (0.10)
    n = delta - P_l delta    value-visible part   |n| <= value_ratio x rms   (None => parent)

Both caps are applied PER ELEMENT and only ever shrink a component that exceeds its
bound, so the direction of the residual is never rewritten -- what changes is how much of
it the key map is allowed to see. The key side is held an order of magnitude tighter than
the value side, which is the opposite weighting of V13's single ceiling.

This is deliberately NOT the hard key lock of experiment 4 (50/158). That run set the key
channel's room to zero and threw away the content that travels through it. A ratio of
0.10 is a floor of room, not a lock, and the arm has to show whether the split buys more
than the ceiling it replaces.

Three bands, one mask
---------------------
`BANDS` cuts the layer stack into lexical (0-10), bridging (11-21) and readout (22-31).
A band gate is a learned scalar per band, multiplied into the residual layer by layer,
and band 0 is multiplied by zero through a mask rather than by dropping the parameter:
the parameter stays in the state dict so the report can show what the mask did and a
checkpoint stays loadable under a different `layer_ids`. Under the default utility layers
(12, 16, 20, 24, 28) no window layer lands in band 0, so that gate is structurally inert
here -- `band_layout` says so per window instead of leaving it to be inferred.

What is NOT claimed here
------------------------
The caps are still soft and still ceilings, not guarantees: a graft slot can move its key
by `key_ratio` x rms and the reader will see that. Whether the split is worth anything is
the arm's question (V13 = 67/158 is the baseline), not this file's.

Every default is the "off" value, so `DeepFusionResidual()` is numerically the parent
class: the two new mechanisms sit behind `if` branches that a default configuration never
enters. That is what lets V13 be re-run against the same code this file ships in.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from methods.prlmc import cross_block_calibration
from methods.prlmc.cross_block_calibration import DEFAULT_KV_DIM, CrossBlockResidual
from methods.prlmc.key_safe_projection import project_row_space
from methods.prlmc.reader_utility import UTILITY_LAYERS

SCHEMA = "deep_fusion/1"

# The layer stack the band gate cuts up. Fixed, not learned and not stored per
# checkpoint beyond this module, because a band map that a checkpoint could redefine
# would make two runs' `band_gate` entries mean different things.
BANDS = ((0, 10), (11, 21), (22, 31))
BAND_NAMES = ("lexical", "bridging", "readout")
BAND_MODES = ("off", "three")
KEY_CHANNELS = ("off", "project")


def band_members(band):
    """The layers a band covers, inclusive."""
    if not 0 <= int(band) < len(BANDS):
        raise ValueError(f"band {band} outside {len(BANDS)} bands")
    first, last = BANDS[int(band)]
    return tuple(range(int(first), int(last) + 1))


def band_of_layer(layer):
    """Layer index -> band index, or None for a layer no band covers.

    `layer` is the index the READER uses (`UTILITY_LAYERS` = 12/16/20/24/28), not the
    position of a row in the window: the two coincide only for a 32-layer window, and
    conflating them would put a readout layer's gate on a bridging layer's residual.
    """
    try:
        value = int(layer)
    except (TypeError, ValueError):
        raise ValueError(f"a layer index must be an integer, got {layer!r}")
    for band, (first, last) in enumerate(BANDS):
        if int(first) <= value <= int(last):
            return band
    return None


def band_layout(layer_ids=UTILITY_LAYERS):
    """Which band gates each window layer, and which layers the mask zeroes.

    A checkpoint's config records `layer_ids`, so the layout has to be derivable from
    that alone: the report reads this instead of re-deriving the map at every call site
    and getting a different answer than the forward pass used.

    Returns a dict with `layers`, `bands` (one band index per window layer),
    `band_names`, `zeroed_layers` (window layers whose gate is masked to zero; also
    under `zero_layers` and `zeroed`), `active_layers`, `bands_present` and
    `bands_absent`.
    """
    layers = tuple(int(layer) for layer in layer_ids)
    if not layers:
        raise ValueError("a window needs at least one layer")
    bands = tuple(band_of_layer(layer) for layer in layers)
    missing = [layer for layer, band in zip(layers, bands) if band is None]
    if missing:
        raise ValueError(f"layers {missing} fall in no band of {BANDS}: the band gate "
                         f"would have nothing to index for them")
    present = sorted(set(bands))
    zeroed = tuple(layer for layer, band in zip(layers, bands) if band == 0)
    return dict(layers=layers, bands=bands,
                band_names=tuple(BAND_NAMES[band] for band in bands),
                zeroed_layers=zeroed, zero_layers=zeroed, zeroed=zeroed,
                active_layers=tuple(layer for layer, band in zip(layers, bands) if band != 0),
                bands_present=tuple(present),
                bands_absent=tuple(band for band in range(len(BANDS)) if band not in present),
                # A band with no window layer is not "a gate that learned to be zero":
                # its parameter never sees a gradient, so the 0.0 in the checkpoint is
                # the initial value and nothing else. On the factory layer table
                # (12,16,20,24,28) band 0 is empty, so `band_mode="three"` has two live
                # gates there, not three -- and the report has to say that instead of
                # letting a reader count three zeros.
                band0_observable=bool(0 in present),
                gate_parameters_live=tuple(band for band in present if band != 0))


def _rows_for_layers(basis, layer_ids):
    """The basis rows for `layer_ids`, `[len(layer_ids), hidden, rank]`.

    `build_key_bases` returns one row per reader layer (32 of them), while the window
    carries five utility layers and a chunked call sees a subset of those five. Both are
    indexed here rather than assumed aligned: a basis row used for the wrong layer would
    project onto the wrong row space, and the result would still look like a tensor.
    When the row count already equals the layer count the basis is taken as restricted to
    exactly those layers -- there is nothing left to select.
    """
    if not isinstance(basis, torch.Tensor):
        basis = torch.as_tensor(basis)
    if basis.dim() != 3:
        raise ValueError(f"the key basis must be [layers, hidden, rank], got "
                         f"{tuple(basis.shape)}")
    ids = [int(layer) for layer in layer_ids]
    if int(basis.shape[0]) == len(ids):
        return basis
    if max(ids) >= int(basis.shape[0]):
        raise ValueError(f"the key basis has {int(basis.shape[0])} layer rows and layer "
                         f"{max(ids)} is asked for: it was built for a different reader")
    return basis.index_select(0, torch.as_tensor(ids, dtype=torch.long,
                                                 device=basis.device))


def _clip_to_ratio(part, limit):
    """Shrink one SLOT's component into `|x| <= limit` with a single scale factor.

    `part` is `[layers, slots, hidden]` and `limit` is `[layers, slots, 1]`: one factor per
    slot, not one per coordinate. The per-coordinate version this replaces scaled each
    element independently, which does not preserve the direction of the vector -- and for
    the key half that is not a detail. The key half is *defined* as the part the reader's
    key map can see, so a per-coordinate rescale can rotate a residual that was provably
    orthogonal to the key row space INTO it: the cap then ends up creating the key drift
    it was supposed to bound. One factor per slot keeps each half inside its own subspace,
    so the projection split stays meaningful after the clip.

    The factor is `min(1, limit / |x|_inf)`, the L-infinity norm against the ceiling the
    caller already expressed as a share of the slot RMS. A peak norm rather than an RMS
    norm because the promise being made is a hard bound on every coordinate, which an RMS
    bound does not give (a single large coordinate can sit outside an RMS-preserving cap).
    """
    if part.numel() == 0:
        return part
    peak = part.abs().amax(dim=-1, keepdim=True)
    scale = (limit / peak.clamp_min(1e-12)).clamp_max(1.0)
    return part * scale


def _reader_key_norms(scorer, slots, layer_ids, basis):
    """Mean key norm per slot over `layer_ids`, `[N]`, and the name of how it was measured.

    `ReaderScorer.reading` normalises its keys per layer and again after concatenating
    them, so its vectors have norm 1 by construction and cannot answer "how big is this
    slot's key". The reader's own `input_layernorm` and effective `k_proj` are used
    directly here, on float32, and RoPE is skipped because it is a per-head rotation: it
    cannot change a norm, so what is measured is exactly the norm attention sees.

    A scorer without per-layer maps (a test double, or a reader exposed through another
    interface) falls back to the ROW-SPACE component `|P_l h|`, which the damping scales
    by construction. That is a different constant per layer -- `W_K` is not identity -- but
    the scale factor `min(1, target / |k_i|)` compares one slot against the same layer's
    reference, so the ordering and the factor survive the substitution. The report names
    which of the two was measured instead of leaving the reader to assume the stronger one.
    """
    maps = getattr(scorer, "maps", None)
    if isinstance(maps, dict):
        total = torch.zeros(int(slots.shape[1]), dtype=torch.float32, device=slots.device)
        measured = 0
        for position, layer in enumerate(layer_ids):
            entry = maps.get(int(layer))
            if entry is None:
                # The residual is written into all 32 reader layers while the scorer only
                # indexes the five CONTEXT layers. Refusing the whole pass over four
                # unindexed rows in five made the damping impossible on the real window;
                # the row-space magnitude is the documented substitute, so each such row
                # is measured that way and the report says the mixture is not uniform.
                fallback = _rows_for_layers(basis, [int(layer)])
                total = total + project_row_space(
                    slots[position:position + 1], fallback).norm(dim=-1)[0].to(total.device)
                continue
            weight = entry["k"]
            stored = slots[position].to(weight.device).float()
            normed = entry["norm"](stored)
            total = total + F.linear(normed, weight.float()).norm(dim=-1).to(total.device)
            measured += 1
        count = max(len(list(layer_ids)), 1)
        source = ("reader_k_proj" if measured == count
                  else "mixed_reader_and_row_space" if measured else "row_space_basis")
        return total / count, source
    rows = _rows_for_layers(basis, layer_ids)
    projected = project_row_space(slots, rows)
    return projected.norm(dim=-1).mean(dim=0).to(slots.device), "row_space_basis"


class DeepFusionResidual(CrossBlockResidual):
    """`CrossBlockResidual` with a key-anchored residual and a per-band gate.

    Construction inherits the parent's guarantee: the graft half starts at AnchorOnly and
    the core half at identity, and both new mechanisms are off by default, so at update 0
    this module produces the parent's delta bit for bit.

    Order inside `forward` is the spec's and it matters: the band gate scales the whole
    delta first, then the key/value split bounds the pieces. Clipping first and gating
    after would let the gate push a clipped residual back over the cap it had just
    respected.
    """

    def __init__(self, hidden_size=4096, rank=64, max_ratio=0.5, gate_init=0.0,
                 cond_size=3, cond_rank=32, use_cross=True, k_neighbors=4,
                 neighbor_mode="topk", k_same=2, k_cross=2,
                 kv_dim=DEFAULT_KV_DIM, max_ratio_core=0.15, core_weight=2.0,
                 graft_weight=0.5, span_value=False, context_diffusion=False, csls=False,
                 use_evicted=False, sim_floor=0.0,
                 key_channel="off", key_ratio=0.10, value_ratio=None,
                 band_mode="off", band_gates=(0.0, 1.0, 1.0),
                 layer_ids=UTILITY_LAYERS,
                 graft_tau=1.0, graft_bias=0.0, graft_norm_quantile=0.0):
        super().__init__(hidden_size=hidden_size, rank=rank, max_ratio=max_ratio,
                         gate_init=gate_init, cond_size=cond_size, cond_rank=cond_rank,
                         use_cross=use_cross, k_neighbors=k_neighbors,
                         neighbor_mode=neighbor_mode, k_same=k_same, k_cross=k_cross,
                         kv_dim=kv_dim, max_ratio_core=max_ratio_core,
                         core_weight=core_weight, graft_weight=graft_weight,
                         span_value=span_value, context_diffusion=context_diffusion,
                         csls=csls, use_evicted=use_evicted, sim_floor=sim_floor)
        if key_channel not in KEY_CHANNELS:
            raise ValueError(f"unknown key channel {key_channel!r}: expected one of "
                             f"{KEY_CHANNELS}")
        if band_mode not in BAND_MODES:
            raise ValueError(f"unknown band mode {band_mode!r}: expected one of "
                             f"{BAND_MODES}")
        key_ratio = float(key_ratio)
        # A zero cap would be the hard key lock experiment 4 measured at 50/158, reached
        # by accident instead of by a flag. Refusing it keeps that arm a deliberate
        # choice rather than a default that looks like a ceiling.
        if not 0.0 < key_ratio < 1.0:
            raise ValueError(f"key_ratio is a fraction of the slot RMS and must be in "
                             f"(0, 1), got {key_ratio}")
        if value_ratio is not None:
            value_ratio = float(value_ratio)
            if not 0.0 < value_ratio < 1.0:
                raise ValueError(f"value_ratio is a fraction of the slot RMS and must be "
                                 f"in (0, 1) or None, got {value_ratio}")
        gates = tuple(float(value) for value in band_gates)
        if len(gates) != len(BANDS):
            raise ValueError(f"band_gates wants one initial gate per band "
                             f"({len(BANDS)}: {BAND_NAMES}), got {len(gates)}")
        if not all(math.isfinite(value) for value in gates):
            raise ValueError("band gate initial values must be finite")
        # `layer_ids=None` is the identity mapping: row i of the window IS reader layer i.
        # That is the normal case -- `select_carriers` hands over the reader's full stack --
        # and it is what makes the three bands line up with the reader's real layers
        # (0-10 lexical, 11-21 bridging, 22-31 readout). An explicit tuple is for a caller
        # whose window is a subset and whose row index therefore stops meaning "layer".
        if layer_ids is None:
            layers = None
        else:
            layers = tuple(int(layer) for layer in layer_ids)
            if not layers:
                raise ValueError("layer_ids must name at least one window layer")
            if len(set(layers)) != len(layers):
                raise ValueError(f"layer_ids carries a duplicate layer: {layers}")
            if band_mode == "three":
                layout = band_layout(layers)
                if not layout["active_layers"]:
                    raise ValueError(f"every layer in {layers} falls in band 0, which the "
                                     f"mask zeroes: the band gate would turn the residual off")
        tau = float(graft_tau)
        if not 0.0 < tau <= 1.0:
            raise ValueError(f"graft_tau damps graft keys and must be in (0, 1], got {tau}")
        bias = float(graft_bias)
        quantile = float(graft_norm_quantile)
        if not 0.0 <= quantile < 1.0:
            raise ValueError(f"graft_norm_quantile is a quantile and must be in [0, 1), "
                             f"got {quantile}")
        if not (math.isfinite(bias) and math.isfinite(quantile) and math.isfinite(tau)):
            raise ValueError("the graft attention constants must be finite")
        self.key_channel = str(key_channel)
        self.key_ratio = key_ratio
        self.value_ratio = value_ratio
        self.band_mode = str(band_mode)
        self.band_gate_init = gates
        self.layer_ids = layers
        # The three attention-damping constants live on the module as plain floats so a
        # checkpoint records them in `config()` and an evaluation arm can neutralise them
        # on a copy instead of re-deriving "which arm damped twice".
        self.graft_tau = tau
        self.graft_bias = bias
        self.graft_norm_quantile = quantile
        # The key basis is state, not a parameter: it comes from the frozen reader, it is
        # never trained and it must never enter the state dict (a checkpoint would then
        # be tied to one build of the reader's key map).
        self.key_basis = None
        self.key_basis_layers = ()
        self.band_gate = nn.Parameter(torch.tensor(gates, dtype=torch.float32))
        # Band 0 is multiplied by zero, so its parameter gets no gradient; it is kept so
        # the checkpoint can record what the mask removed.
        self.register_buffer("band_open",
                             torch.tensor([0.0 if band == 0 else 1.0
                                           for band in range(len(BANDS))],
                                          dtype=torch.float32),
                             persistent=False)
        # A chunked caller has to say WHICH layers a chunk holds, or the band gate would
        # be indexed by position inside the chunk. `uses_layers` is the convention the
        # trainer and `band_relation` already use to pass that argument.
        self.uses_layers = True
        if layers is not None and len(layers) != len(UTILITY_LAYERS):
            # The parent sizes the regulariser weights for the standard five-layer window;
            # this module may be built on another layer set, and a weight count that does
            # not match the window is an error in `calibrate` rather than a rounded fit.
            self.register_buffer("layer_norm_weights",
                                 torch.ones(len(layers), dtype=torch.float32),
                                 persistent=False)

    def config(self):
        """Constructor kwargs including every V17 switch, so eval can rebuild the arm."""
        config = super().config()
        config.update(key_channel=self.key_channel, key_ratio=self.key_ratio,
                      value_ratio=self.value_ratio, band_mode=self.band_mode,
                      band_gates=self.band_gate_init, layer_ids=self.layer_ids,
                      graft_tau=self.graft_tau, graft_bias=self.graft_bias,
                      graft_norm_quantile=self.graft_norm_quantile)
        return config

    @property
    def value_ratio_effective(self):
        """The ceiling the value side actually uses, `None` included.

        `value_ratio=None` means "the parent's `max_ratio`", and `max_ratio` is a CLI
        argument whose default is 0.34 -- not this constructor's default 0.5. A sweep
        that sends 0.25/0.30/0.40 and compares the results against the "None" arm is
        therefore comparing against an unlisted 0.34, so the effective ceiling has to be
        readable off the module instead of inferred from the flag.
        """
        return float(self.max_ratio if self.value_ratio is None else self.value_ratio)

    # -- the key row space ------------------------------------------------
    def install_key_basis(self, basis, layer_ids=None):
        """Attach `build_key_bases(model)[0]`; returns self so it can be chained.

        Either the full per-reader-layer stack or one already restricted to the window's
        layers is accepted, and which one it was is recorded alongside it: `forward` may
        be handed a single chunk of the window, and guessing the alignment there is
        exactly how a basis row ends up projecting onto another layer's row space.
        """
        chosen = self.layer_ids if layer_ids is None else layer_ids
        ids = None if chosen is None else tuple(int(layer) for layer in chosen)
        if ids is not None and not ids:
            raise ValueError("the key basis needs at least one layer")
        rows = torch.as_tensor(basis).detach().float()
        if rows.dim() != 3:
            raise ValueError(f"the key basis must be [layers, hidden, rank], got "
                             f"{tuple(rows.shape)}")
        if int(rows.shape[1]) != self.hidden_size:
            raise ValueError(f"the key basis is built for hidden {int(rows.shape[1])}, "
                             f"this module for {self.hidden_size}")
        if ids is None:
            # One row per reader layer and the window is that stack: identity mapping.
            mapping = tuple(range(int(rows.shape[0])))
        elif int(rows.shape[0]) == len(ids):
            mapping = ids
        elif int(rows.shape[0]) > max(ids):
            # One row per reader layer: the window's layers are picked out of it later.
            mapping = tuple(range(int(rows.shape[0])))
        else:
            raise ValueError(f"the key basis has {int(rows.shape[0])} layer rows and "
                             f"layer {max(ids)} is asked for: it was built for a "
                             f"different reader")
        self.key_basis = rows
        self.key_basis_layers = mapping
        return self

    def _basis_for(self, positions):
        """Basis rows for the window positions in `positions`, `[len, hidden, rank]`."""
        basis = self.key_basis
        if basis is None:
            raise ValueError("key_channel='project' reads the reader's key row space: "
                             "install it first with "
                             "install_key_basis(build_key_bases(model)[0])")
        rows = []
        for position in positions:
            layer = self._layer_at(position)
            if layer not in self.key_basis_layers:
                raise ValueError(f"the installed key basis has no row for reader layer "
                                 f"{layer}")
            rows.append(self.key_basis_layers.index(layer))
        index = torch.as_tensor(rows, dtype=torch.long, device=basis.device)
        return basis.index_select(0, index)

    # -- the residual -----------------------------------------------------
    def _positions(self, slots, layer_ids):
        """Window layer POSITIONS for this call, `[rows]`.

        The caller passes positions into `layer_ids`, which is what a chunked trainer can
        know (`arange(start, stop)`); reader layer numbers are rejected rather than
        silently mapped, because `12` would otherwise be read as "window row 12" and the
        wrong band would gate the residual.
        """
        rows = int(slots.shape[0])
        if layer_ids is None:
            if self.layer_ids is None or rows == len(self.layer_ids):
                # Identity mapping: the rows are layers 0..rows-1 in order.
                return list(range(rows))
            raise ValueError(f"{rows} layer rows for a module built on "
                             f"{len(self.layer_ids)} layers: a partial stack must say "
                             f"which layers it holds via `layer_ids`")
        positions = [int(value) for value in
                     torch.as_tensor(layer_ids).reshape(-1).tolist()]
        if len(positions) != rows:
            raise ValueError(f"{len(positions)} layer positions for {rows} layer rows")
        # With an identity layer table a position is an ABSOLUTE reader layer, so a
        # 4-row chunk legitimately carries positions 4..7: the bound is the module's layer
        # table, not the chunk length.
        limit = None if self.layer_ids is None else len(self.layer_ids)
        for position in positions:
            if limit is None:
                if position < 0:
                    raise ValueError(f"layer position {position} is negative")
                continue
            if not 0 <= position < limit:
                raise ValueError(f"layer id {position} is not a position in a window of "
                                 f"{limit} layers: `layer_ids` carries positions, not "
                                 f"reader layer numbers ({self.layer_ids})")
        return positions

    def _band_index(self, positions):
        bands = [band_of_layer(self._layer_at(position)) for position in positions]
        if any(band is None for band in bands):
            raise ValueError(f"the window layers {self.layer_ids} are not all covered by "
                             f"{BANDS}: the band gate has no entry for them")
        return torch.as_tensor(bands, dtype=torch.long, device=self.band_gate.device)

    def _layer_at(self, position):
        """Reader layer number for a window row, identity included."""
        return int(position) if self.layer_ids is None else int(self.layer_ids[position])

    def band_layout(self):
        """`band_layout(...)`: what this module's gate does per window layer.

        Without an explicit layer table every reader layer is a window row, so the layout
        is the full 0..31 map -- which is the case where band 0 (layers 0-10, masked to
        zero) is actually populated.
        """
        if self.layer_ids is None:
            return band_layout(range(BANDS[-1][1] + 1))
        return band_layout(self.layer_ids)

    def forward(self, slots, cond, context=None, layer_ids=None):
        """`[layers, N, D]` slots, `[N, 3]` conditions, context -> bounded delta."""
        delta = super().forward(slots, cond, context)
        if self.band_mode == "off" and self.key_channel == "off":
            # The default path is the parent's, untouched: not even the positions are
            # resolved, so a caller that never asked for a band cannot fail on one.
            return delta.to(slots.dtype)
        positions = self._positions(slots, layer_ids)
        if self.band_mode == "three":
            band = self._band_index(positions)
            open_ = self.band_open.to(self.band_gate.device)
            gate = (self.band_gate.float() * open_)[band]
            delta = delta * gate.to(delta.dtype).view(-1, 1, 1)
        if self.key_channel == "project":
            visible = project_row_space(delta, self._basis_for(positions))
            # In place, deliberately: `delta` is private to this call (it is either the
            # parent's fresh output or the product of the band gate just above) and keeping
            # both it and the complement alive costs another whole `[layers, slots, hidden]`
            # block at the peak, which is exactly where this path runs out of memory.
            hidden = delta.sub_(visible)
            ratio = self.value_ratio_effective
            rms = slots.float().square().mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
            delta = (_clip_to_ratio(visible, self.key_ratio * rms)
                     + _clip_to_ratio(hidden, ratio * rms))
        return delta.to(slots.dtype)


@torch.no_grad()
def init_delta(module, slots, delta):
    """The delta this module MUST produce at initialisation, given the anchor's delta.

    Two deliberate transforms sit between "the AnchorOnly residual" and "what this module
    writes at update 0": the band mask zeroes the layers it will always zero, and the
    key/value split clips the two halves. The identity check cannot compare against the raw
    anchor residual any more -- on a 32-row window the mask alone moves layers 0-10 to
    exactly zero, which reads as "the graft half did not load" (measured: 6.4e+00).

    They are applied here, next to the forward pass that implements them, so the check has
    one implementation to agree with instead of a second copy that can drift.
    """
    rows = int(slots.shape[0])
    out = delta.float()
    if str(getattr(module, "band_mode", "off")) == "three":
        gate = (module.band_gate.detach().float()
                * module.band_open.to(module.band_gate.device))
        band = torch.as_tensor([band_of_layer(module._layer_at(position))
                                for position in range(rows)],
                               dtype=torch.long, device=gate.device)
        out = out * gate[band].to(out.dtype).view(-1, 1, 1)
    if str(getattr(module, "key_channel", "off")) == "project":
        basis = getattr(module, "key_basis", None)
        if basis is None:
            raise ValueError("the identity check needs the key basis the forward uses: "
                             "install it before comparing")
        table = getattr(module, "layer_ids", None)
        layer_ids = list(range(rows)) if table is None else [int(layer) for layer in table]
        if len(layer_ids) != rows:
            raise ValueError(f"{len(layer_ids)} layer ids for {rows} rows")
        visible = project_row_space(out, _rows_for_layers(basis, layer_ids))
        hidden = out - visible
        rms = slots.float().square().mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        out = (_clip_to_ratio(visible, float(module.key_ratio) * rms)
               + _clip_to_ratio(hidden, float(module.value_ratio_effective) * rms))
    return out.to(delta.dtype)


def install_key_basis(module, basis, layer_ids=None):
    """`module.install_key_basis(basis, layer_ids)` as a free function.

    Two entry points for one operation because both call sites exist already: a trainer
    that holds the module and builds the basis once, and an evaluation harness that only
    knows the module as an opaque object. Neither may skip the call, so a module without
    the method is an error rather than a silently unanchored forward.
    """
    hook = getattr(module, "install_key_basis", None)
    if not callable(hook):
        raise ValueError(f"{type(module).__name__} cannot take a key basis: the arm would "
                         f"run with no key anchoring while claiming to have it")
    return hook(basis, layer_ids)


@torch.no_grad()
def graft_attention_calibration(window, meta, basis, tau=1.0, bias=0.0,
                                norm_quantile=0.0, layer_ids=UTILITY_LAYERS,
                                return_report=False):
    """Damp the GRAFT slots' keys; leave every carrier slot bit-identical.

    The 684 graft slots are the ones whose keys nobody trained: they were written in from
    dropped blocks and their key norms sit wherever those blocks' vectors happened to sit.
    An outlier key wins attention by length alone -- `softmax(q.k)` is monotone in `|k|`
    for a fixed direction -- so a few long keys can out-shout the carriers that hold the
    answer. This pass scales the key-visible part of the graft slots only:

        h' = h + (s - 1) * P_l h ,     s = tau * s_quantile * s_bias

    `s_quantile` targets `norm_quantile` of the CORE slots' key norms and can only pull a
    graft key down towards that target (`min(1, target/|k_i|)`); it never lifts one, so it
    is a suppression of outliers and not a rescaling of the whole pool.
    `s_bias` moves the attention logit itself: `s = max(0, 1 + bias / max(logit_proxy,
    eps))` is the FIRST-ORDER step that changes an attended weight by `bias` when the
    logit is `logit_proxy`, so the number a caller writes is a logit change and not a
    multiplier. It is a proxy: the real logit also carries the question's own query, which
    is not available before the question is read. The report says `logit_proxy` so the
    assumption is visible instead of implied.

    Honest limit: only the row-space part is scaled, and the slot's norm therefore moves a
    little (by at most `(1 - s)` of the row-space share, measured at roughly a quarter of
    the vector). Renormalising the slot afterwards would cancel the damping for a slot
    that lives entirely in the row space -- the key would not move at all -- so it is not
    done, and `key_norm_before/after` measure the real reader key instead.
    """
    tau = float(tau)
    bias = float(bias)
    norm_quantile = float(norm_quantile)
    for name, value in (("tau", tau), ("bias", bias), ("norm_quantile", norm_quantile)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 < tau <= 1.0:
        raise ValueError(f"tau must be in (0, 1]: it damps graft keys and cannot raise "
                         f"them above the scale the window already has, got {tau}")
    if not 0.0 <= norm_quantile < 1.0:
        raise ValueError(f"norm_quantile is a quantile of the core key norms and must be "
                         f"in [0, 1), got {norm_quantile}")
    ids = [int(layer) for layer in layer_ids]
    if len(ids) != int(window.shape[0]):
        raise ValueError(f"{len(ids)} reader layers for {int(window.shape[0])} window "
                         f"layers: the basis cannot be matched to the slots")
    slot_type = meta["slot_type"]
    graft = (slot_type > 0.5)
    touched = int(graft.sum())
    core = ~graft
    base = dict(slots_touched=touched, tau=tau, bias=bias, norm_quantile=norm_quantile)
    active = tau != 1.0 or bias != 0.0 or norm_quantile > 0.0
    if touched == 0 or not active:
        report = dict(base, key_scale_mean=None, key_scale_min=None,
                      key_norm_before=None, key_norm_after=None, logit_proxy=None,
                      applied=False,
                      note=("no graft slot in this window" if touched == 0 else
                            "tau=1, bias=0, norm_quantile=0: nothing to damp"))
        return (window, report) if return_report else window
    if int(core.sum()) < 2:
        raise ValueError("the graft attention needs at least two core slots: the core key "
                         "norms are the reference the graft keys are damped towards")
    scorer = meta.get("scorer")
    if scorer is None:
        raise ValueError("the metadata carries no reader scorer: the graft keys cannot be "
                         "measured or damped")
    slots_count = int(window.shape[1])
    positions = torch.arange(slots_count, device=window.device)
    reading, _ = scorer.reading(window, positions, None)
    key = reading.key.float()
    core_index = torch.nonzero(core, as_tuple=False).flatten()
    # A deterministic stride, not a random draw: the report has to reproduce, and 256
    # rows already cover the pool far better than the per-slot estimate needs.
    if int(core_index.numel()) > 256:
        step = int(core_index.numel()) // 256
        core_index = core_index[::step][:256]
    core_keys = key.index_select(0, core_index)
    gram = core_keys @ core_keys.t()
    off_diagonal = ~torch.eye(int(gram.shape[0]), dtype=torch.bool, device=gram.device)
    logit_proxy = float(gram.abs()[off_diagonal].mean())
    norms, norm_source = _reader_key_norms(scorer, window, ids, basis)
    scale = torch.full((slots_count,), tau, dtype=torch.float32, device=window.device)
    if norm_quantile > 0.0:
        # Only ever a downward pull: a graft key shorter than the core quantile is left
        # where it is, so this cannot rescale the whole graft pool.
        target = torch.quantile(norms[core], norm_quantile)
        scale = scale * (target / norms.clamp_min(1e-12)).clamp_max(1.0)
    bias_scale = max(0.0, 1.0 + bias / max(logit_proxy, 1e-12))
    scale = (scale * bias_scale).clamp_min(0.0)
    rows = _rows_for_layers(basis, ids)
    projected = project_row_space(window, rows)
    adjust = (scale.view(1, -1, 1) - 1.0) * projected
    out = window.clone()
    moved = torch.nonzero(graft, as_tuple=False).flatten()
    out[:, moved] = (window[:, moved].float() + adjust[:, moved]).to(window.dtype)
    norms_after, _ = _reader_key_norms(scorer, out, ids, basis)
    before = float(norms[graft].mean())
    after = float(norms_after[graft].mean())
    report = dict(base,
                  key_scale_mean=float(scale[graft].mean()),
                  key_scale_min=float(scale[graft].min()),
                  key_norm_before=before, key_norm_after=after,
                  key_norm_ratio=after / max(before, 1e-12),
                  key_norm_source=norm_source,
                  logit_proxy=logit_proxy, applied=True,
                  note="only the row-space part of the graft slots moved; carriers are "
                       "bit-identical. `logit_proxy` is the mean absolute inner product "
                       "of core keys against a 256-slot core sample, a FIRST-ORDER proxy "
                       "for the attention logit scale: the question's own query is not "
                       "part of it, so `bias` changes the logit by approximately "
                       "`bias` rather than exactly")
    return (out, report) if return_report else out


@torch.no_grad()
def calibrate(window, meta, module, layer_chunk=8, context=None, return_report=False,
              source=None, basis=None):
    """`graft_attention_calibration` (when enabled) then `cross_block_calibration`.

    Damping runs FIRST so the neighbour association is built from the keys the reader will
    actually see: damping afterwards would leave every context row measured against a
    window that no longer exists.

    The return value is the parent's, tuple shape included -- a third element appears only
    when `return_report` is set -- so an arm can be swapped between V13 and V17 without the
    caller learning a second calling convention. The damping report is the only addition,
    under `info["v17"]`.
    """
    tau = float(getattr(module, "graft_tau", 1.0))
    bias = float(getattr(module, "graft_bias", 0.0))
    norm_quantile = float(getattr(module, "graft_norm_quantile", 0.0))
    table = getattr(module, "layer_ids", UTILITY_LAYERS)
    ids = (tuple(range(int(window.shape[0]))) if table is None
           else tuple(int(layer) for layer in table))
    attention_report = None
    if tau != 1.0 or bias != 0.0 or norm_quantile > 0.0:
        if context is not None:
            # A context row is built from the window's keys. Damping rewrites exactly
            # those, so a context computed before it describes a window the reader will
            # never see -- rebuilding it here is the caller's job, not a silent fallback.
            raise ValueError("graft attention damping rewrites the keys the context is "
                             "built from: pass context=None so it is recomputed")
        rows = basis if basis is not None else getattr(module, "key_basis", None)
        if rows is None:
            raise ValueError("this checkpoint damps graft keys but no key basis was "
                             "installed: build_key_bases(model)[0] passed to "
                             "install_key_basis is what the row space is projected on")
        window, attention_report = graft_attention_calibration(
            window, meta, rows, tau=tau, bias=bias, norm_quantile=norm_quantile,
            layer_ids=ids, return_report=True)
    chunk = int(layer_chunk)
    if (str(getattr(module, "band_mode", "off")) == "three"
            or str(getattr(module, "key_channel", "off")) == "project"):
        # Both new paths index a per-layer table by the row's layer, so a chunk boundary
        # would shift a row's band or its basis row. The chunk size never changes the
        # result, it only bounds the working set, so raising it to the whole window is the
        # cheapest way to make that impossible rather than merely unlikely.
        chunk = max(chunk, int(window.shape[0]))
    out = cross_block_calibration.calibrate(
        window, meta, module, layer_chunk=chunk, context=context,
        return_report=return_report, source=source)
    info = out[1]
    # The module's own constants are reported even when nothing was damped: an arm that
    # records "tau=1.0" says the checkpoint was measured WITHOUT damping, which is a
    # different claim from one that records nothing at all.
    damped = dict(tau=tau, bias=bias, norm_quantile=norm_quantile, applied=False,
                  slots_touched=0)
    damped.update(attention_report or {})
    if "note" in (attention_report or {}):
        # The damping report's own note keeps its own name, so the wrapper's note below
        # cannot silently overwrite the reason the caller reads first.
        damped["attention_note"] = damped.pop("note")
    damped.update(
        key_channel=str(getattr(module, "key_channel", "off")),
        key_ratio=getattr(module, "key_ratio", None),
        value_ratio=getattr(module, "value_ratio", None),
        band_mode=str(getattr(module, "band_mode", "off")),
        band_gates=list(getattr(module, "band_gate_init", ())),
        band_layout=band_layout(ids),
        key_basis_installed=bool(getattr(module, "key_basis", None) is not None),
        note="the key/value split bounds the residual after the band gate; the ceilings "
             "are per element and only ever shrink, so the direction is unchanged")
    info["v17"] = damped
    return out


def load_deep(path, hidden_size, rank=None, max_ratio=None):
    """Rebuild a `DeepFusionResidual` from a `deep_fusion/1` checkpoint, refusing drift.

    Strict on purpose: the config is the checkpoint's own `config()`, so a V17 arm cannot
    be rebuilt with fewer switches than it was trained with. A rank or ratio override that
    disagrees with what was trained is an error and not a silent reinterpretation of the
    same weights.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA + "/checkpoint":
        raise ValueError(f"{path} is not a {SCHEMA} checkpoint "
                         f"(schema {payload.get('schema')!r})")
    config = dict(payload["config"])
    if int(config.get("hidden_size", -1)) != int(hidden_size):
        raise ValueError(f"{path} was built for hidden {config.get('hidden_size')}, "
                         f"the reader is {hidden_size}")
    if rank is not None and int(config.get("rank", -1)) != int(rank):
        raise ValueError(f"{path} has rank {config.get('rank')}, expected {rank}")
    if max_ratio is not None and float(config.get("max_ratio", -1)) != float(max_ratio):
        raise ValueError(f"{path} has max_ratio {config.get('max_ratio')}, expected "
                         f"{max_ratio}")
    module = DeepFusionResidual(**config)
    module.load_state_dict(payload["state_dict"], strict=True)
    return module, config


__all__ = ["SCHEMA", "BANDS", "BAND_NAMES", "BAND_MODES", "KEY_CHANNELS",
           "DeepFusionResidual", "band_members", "band_of_layer", "band_layout",
           "install_key_basis", "graft_attention_calibration", "calibrate", "load_deep"]
