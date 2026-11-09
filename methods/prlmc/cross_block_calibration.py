"""V13/V14: cross-block conditioned calibration of every slot in the 2,304-slot window.

What is new relative to V11Evict
--------------------------------
V11's calibration saw each slot alone. Its input was the slot's own vector plus a
three-number condition (position drift, cluster size, slot type), so it could only ask
"how should this vector change to be easier to read WHERE IT NOW SITS". It could not
ask the question the mechanism needs: *what does this slot relate to in the rest of the
memory?* A graft slot that carries a person's birth city and a core slot that carries
the person's identity are useless apart and useful together, and nothing in V11 let the
residual see that.

V13 adds exactly one capability: before a slot is modified, the calibrator is shown a
SMALL, SPARSE set of slots from OTHER blocks that it is most associated with.

    association   s_ij = <key_i, key_j> over the reader's own multi-layer keys
                  (unit rows, computed on the window the reader will read)
    sparsity      same-block neighbours are masked out (the mechanism is cross-block),
                  the top `k = 4` cross-block neighbours survive, weights are a softmax
                  over those k only
    context       c_i = sum_j w_ij * value_j   -- the neighbours' CONTENT, not an average
                  of the slot with them. Nothing is blended into the slot; c_i is only
                  an input to the residual, and the slot itself is moved by the same
                  bounded low-rank map V11 used.

That last sentence is the difference between this and the merges that failed: the
vector is never replaced by a mixture, so the frozen reader never receives a slot that
is a compromise between two facts.

Asymmetric constraint, soft on both sides
-----------------------------------------
Experiment 4 killed the key channel (50/158) and experiment 5 showed the content mostly
travels through it, so locking keys is not an option either. The constraint here is
therefore a pair of different CEILINGS, not a lock:

    core (carrier) slots   |delta| <= max_ratio_core  x rms      (0.15: protect them)
    graft slots            |delta| <= max_ratio_graft x rms      (0.50: let them move)

and the trainer's residual regulariser weights the two halves differently
(`core_weight = 2.0`, `graft_weight = 0.5`). Both are soft: a core slot can still move
up to 15% of its own RMS, which is more than the 0 change a hard key lock allowed.

V14: where the neighbours come from
-----------------------------------
V13 asks every slot for its k most associated cross-block slots and does not care what
kind they are. That is a gap, and it is measurable: the report records, per row,
`neighbour_type_cross_fraction` -- the share of the selected neighbours whose core/graft
kind differs from the query's. The 684 graft slots are the only slots that carry content
the carriers lost, so a carrier whose top-4 neighbours are all carriers is calibrated
against retained knowledge only, and the new knowledge it has to be read alongside never
reaches its context.

`mode="complement"` therefore builds the neighbour set from BOTH sides of that divide:

    k_same  = 2 slots of the query's own kind
    k_cross = 2 slots from the other side (a carrier sees graft slots, graft sees carriers)
    same-block is still masked, and the total is still k = 4

The four weights are one softmax PER GROUP, with the group masses split evenly
(k_same/k and k_cross/k). Even splitting is deliberate rather than a detail: a single
softmax over the four would let the own-kind similarities, which are systematically the
higher ones, starve the complement side towards zero weight, and the arm would silently
degenerate into "top-2 of my own kind". `cross_weight_mass_mean` in the context report
says whether that happened. Whether the complement side then BUYS anything is the arm's
question, not this file's: V14 must beat V13 (67/158) by a margin worth the change, and
must still lose to its own donor arm.

Nothing else changes between V13 and V14 -- same window, same budget, same 1,620 / 684
split, same rank, ceilings, regulariser, loss, updates and cross learning rate. The only
variable is which slots the context is built from.

What is NOT claimed here
------------------------
The sparse association is a proxy for "these two slots belong to the same fact". It is
measured in the reader's key space, which experiment 5 showed is the channel that
carries content AND the channel that breaks existing answers; whether conditioning on
it helps is the thing the arm has to show, not something this file may assume.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from methods.prlmc.core_fusion import apply_delta

SCHEMA = "cross_block_residual/1"
DEFAULT_KV_DIM = 5 * 8 * 128  # 5 utility layers x 8 kv heads x 128
# "topk"       V13: the k most associated cross-block slots, kind-blind
# "complement" V14: k_same of the query's own kind + k_cross from the other kind
NEIGHBOUR_MODES = ("topk", "complement")


def conditions(meta, window_slots):
    """The three conditioning features V11 already used, `[slots, 3]`.

    `delta_position`  where the slot came from vs where it now sits
    `log_cluster`     log |C_i| -- always 0 here (nothing is folded any more), kept so
                      a checkpoint trained with folding stays loadable
    `is_graft`        1 for a dropped-block slot, 0 for a carrier slot
    """
    cluster = meta["cluster_size"].float()
    return torch.stack([
        meta["delta_position"].float().clamp(-1.0, 1.0),
        (torch.log(cluster) / 1.5).clamp(0.0, 1.0),
        meta["slot_type"].float(),
    ], dim=-1).to(torch.float32)


def _neighbour_groups(mode, k, k_same, k_cross, slots):
    """`[(kind, how many)]` and the per-group weight masses; `kind` is None for topk."""
    if mode not in NEIGHBOUR_MODES:
        raise ValueError(f"unknown neighbour mode {mode!r}: expected one of "
                         f"{NEIGHBOUR_MODES}")
    if mode == "topk":
        wanted = [(None, int(k))]
    else:
        if int(k_same) < 0 or int(k_cross) < 0:
            raise ValueError("k_same and k_cross must not be negative")
        if int(k_same) + int(k_cross) != int(k):
            raise ValueError(f"complement neighbours read k = k_same + k_cross, but "
                             f"{k_same} + {k_cross} != {k}")
        wanted = [("same", int(k_same)), ("cross", int(k_cross))]
    wanted = [(kind, count) for kind, count in wanted if count > 0]
    if not wanted:
        raise ValueError("the neighbour set is empty: k must be positive")
    limited = [(kind, min(count, slots - 1)) for kind, count in wanted]
    if any(count <= 0 for _, count in limited):
        raise ValueError("the window is too short for a cross-block neighbour set")
    total = sum(count for _, count in limited)
    return limited, [count / total for _, count in limited]


@torch.no_grad()
def cross_block_context(window, meta, k=4, chunk=256, return_report=False,
                        mode="topk", k_same=2, k_cross=2):
    """Sparse cross-block neighbours of every slot, `[slots, key_dim]` float32.

    Keys and values come from the reader's own projections on the window it will read
    (`ReaderScorer.reading(..., probes=None)`), so the association is measured in the
    space the reader uses, at the positions the slots will actually occupy.

    `mode="topk"` (V13): the k most associated slots from other blocks, whatever kind
    they are. `mode="complement"` (V14): k_same slots of the query's own core/graft kind
    plus k_cross from the other side, so every slot is calibrated against both retained
    knowledge and the new knowledge it has to be read alongside. Each group is softmaxed
    separately and the group masses are equal, so the complement side cannot be starved
    by the systematically higher own-kind similarities.

    Nothing is averaged into the slot: the returned context is an extra input to the
    residual. Same-block neighbours are masked, so every context row is built from
    other blocks by construction -- that mask is what makes this cross-block at all.
    """
    groups, masses = _neighbour_groups(mode, k, k_same, k_cross, int(window.shape[1]))
    scorer = meta.get("scorer")
    if scorer is None:
        raise ValueError("the metadata carries no reader scorer: the cross-block context "
                         "cannot be computed")
    slots = int(window.shape[1])
    reading, _ = scorer.reading(window, torch.arange(slots, device=window.device), None)
    key, value = reading.key.float(), reading.value.float()
    block = meta["source_block"]
    kind = (meta["slot_type"] > 0.5).to(torch.bool)  # True carries new content (graft)
    rows, used_similarity, distinct = [], [], []
    similarity_out, weight_cross, graft_neighbour, query_graft = [], [], [], []
    for start in range(0, slots, chunk):
        stop = min(start + chunk, slots)
        similarity = key[start:stop] @ key.t()
        same = block[start:stop].unsqueeze(1) == block.unsqueeze(0)
        similarity = similarity.masked_fill(same, -2.0)
        same_kind = kind[start:stop].unsqueeze(1) == kind.unsqueeze(0)
        picks, scales = [], []
        for (name, count), mass in zip(groups, masses):
            masked = similarity
            if name == "same":
                masked = similarity.masked_fill(~same_kind, -2.0)
            elif name == "cross":
                masked = similarity.masked_fill(same_kind, -2.0)
            best = masked.topk(count, dim=-1)
            picks.append(best.indices)
            scales.append(torch.softmax(best.values, dim=-1) * mass)
            used_similarity.append(best.values.mean(dim=-1))
        index = torch.cat(picks, dim=-1)                # [rows, K]
        weights = torch.cat(scales, dim=-1)             # [rows, K], one softmax per group
        gathered = value.index_select(0, index.reshape(-1)).reshape(
            stop - start, index.shape[-1], -1)
        rows.append((gathered * weights.unsqueeze(-1)).sum(dim=1))
        distinct.append(block.index_select(0, index.reshape(-1))
                        .reshape(stop - start, index.shape[-1]).float().std(dim=-1))
        # Composition of the neighbour set, which is the number V13 never reported:
        # how much of the context comes from the other side of the core/graft divide.
        neighbour_graft = kind.index_select(0, index.reshape(-1)).reshape(index.shape)
        graft_neighbour.append(neighbour_graft.float().mean(dim=-1))
        query_graft.append(kind[start:stop])
        cross = (neighbour_graft != kind[start:stop].unsqueeze(1)).float()
        similarity_out.append(cross.mean(dim=-1))
        weight_cross.append((weights * cross).sum(dim=-1))
        del similarity, same, same_kind, picks, scales, masked, index, weights, gathered
    context = torch.cat(rows, dim=0)
    if not return_report:
        return context
    used = torch.cat(used_similarity, dim=0)
    cross_each = torch.cat(similarity_out, dim=0)
    cross_mass = torch.cat(weight_cross, dim=0)
    graft_neighbour = torch.cat(graft_neighbour, dim=0)
    is_graft = torch.cat(query_graft, dim=0)

    def _mean(values):
        return float(values.mean()) if int(values.numel()) else None

    report = dict(mode=mode, k=sum(count for _, count in groups), key_dim=int(key.shape[-1]),
                  k_same=groups[0][1] if mode == "complement" else None,
                  k_cross=groups[-1][1] if mode == "complement" else None,
                  neighbour_similarity_mean=_mean(used),
                  neighbour_similarity_min=float(used.min()),
                  neighbour_block_spread_mean=float(torch.cat(distinct, dim=0).mean()),
                  context_rms=float(context.square().mean().sqrt()),
                  neighbour_graft_fraction=_mean(graft_neighbour),
                  neighbour_type_cross_fraction=_mean(cross_each),
                  core_to_graft_neighbour_fraction=_mean(cross_each[~is_graft]),
                  graft_to_core_neighbour_fraction=_mean(cross_each[is_graft]),
                  cross_weight_mass_mean=_mean(cross_mass),
                  own_kind_weight_mass_mean=_mean(1.0 - cross_mass))
    return context, report


class CrossBlockResidual(nn.Module):
    """Bounded low-rank residual for every slot, conditioned on its cross-block context.

    V11's `FullPoolResidual` plus one term: `cross_proj` maps the sparse neighbour
    context into the bottleneck and is zero-initialised, so at update 0 the module is
    EXACTLY V11's calibrator and the identity on the core half. That equality is a
    construction guarantee, not an empirical claim: at step 0 nothing has moved.

    The two halves keep separate output maps and separate residual ceilings, which is
    the soft version of "protect the carriers, free the graft slots".
    """

    def __init__(self, hidden_size=4096, rank=64, max_ratio=0.5, gate_init=0.0,
                 cond_size=3, cond_rank=32, use_cross=True, k_neighbors=4,
                 neighbor_mode="topk", k_same=2, k_cross=2,
                 kv_dim=DEFAULT_KV_DIM, max_ratio_core=0.15, core_weight=2.0,
                 graft_weight=0.5):
        super().__init__()
        if rank <= 0 or hidden_size <= 0:
            raise ValueError("rank and hidden_size must be positive")
        if max_ratio_core > max_ratio:
            raise ValueError("the core ceiling must not exceed the graft ceiling")
        if neighbor_mode not in NEIGHBOUR_MODES:
            raise ValueError(f"unknown neighbour mode {neighbor_mode!r}: expected one of "
                             f"{NEIGHBOUR_MODES}")
        if neighbor_mode == "complement" and int(k_same) + int(k_cross) != int(k_neighbors):
            raise ValueError(f"complement neighbours read k = k_same + k_cross, but "
                             f"{k_same} + {k_cross} != {k_neighbors}")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.max_ratio = float(max_ratio)
        self.max_ratio_core = float(max_ratio_core)
        self.cond_size = int(cond_size)
        self.cond_rank = int(cond_rank)
        self.use_cross = bool(use_cross)
        self.k_neighbors = int(k_neighbors)
        self.neighbor_mode = str(neighbor_mode)
        self.k_same = int(k_same)
        self.k_cross = int(k_cross)
        self.kv_dim = int(kv_dim)
        self.core_weight = float(core_weight)
        self.graft_weight = float(graft_weight)
        self.down = nn.Parameter(torch.empty(self.rank, self.hidden_size))
        self.up_graft = nn.Parameter(torch.zeros(self.hidden_size, self.rank))
        self.up_core = nn.Parameter(torch.zeros(self.hidden_size, self.rank))
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        self.type_gate_logit = nn.Parameter(torch.zeros(()))
        # `cond_down` random and `cond_up` zero: the film is exactly 0 at init, and both
        # halves still have a gradient route out. All-zero deadlocks (measured, round 6).
        self.cond_down = nn.Parameter(torch.empty(self.cond_rank, self.cond_size))
        self.cond_up = nn.Parameter(torch.zeros(2 * self.rank, self.cond_rank))
        nn.init.orthogonal_(self.down)
        nn.init.orthogonal_(self.cond_down, gain=0.1)
        if self.use_cross:
            self.cross_proj = nn.Linear(self.kv_dim, self.rank, bias=False)
            nn.init.zeros_(self.cross_proj.weight)

    def config(self):
        """Constructor kwargs, so a reference copy cannot drift from the trained one."""
        return dict(hidden_size=self.hidden_size, rank=self.rank,
                    max_ratio=self.max_ratio, cond_size=self.cond_size,
                    cond_rank=self.cond_rank, use_cross=self.use_cross,
                    k_neighbors=self.k_neighbors, neighbor_mode=self.neighbor_mode,
                    k_same=self.k_same, k_cross=self.k_cross, kv_dim=self.kv_dim,
                    max_ratio_core=self.max_ratio_core, core_weight=self.core_weight,
                    graft_weight=self.graft_weight)

    def neighbours(self, window, meta, return_report=False):
        """This module's own neighbour rule -- the one place that knows it.

        Callers must not build the context themselves: a V13 checkpoint and a V14
        checkpoint differ in nothing but this rule, so a call site that hard-codes
        `k_neighbors` or the mode would silently evaluate the wrong arm.
        """
        if not self.use_cross:
            if return_report:
                return None, None
            return None
        return cross_block_context(
            window, meta, k=self.k_neighbors, return_report=return_report,
            mode=self.neighbor_mode, k_same=self.k_same, k_cross=self.k_cross)

    def anchor_view(self):
        """`AnchorResidual` carrying this module's graft parameters.

        At initialisation the cross projection is zero and the film is zero, so the
        graft half has to reproduce AnchorOnly exactly; the trainer checks that by
        rebuilding the plain anchor from these parameters and applying it to the same
        slots. Without this view the check cannot distinguish "the module loaded" from
        "the module stayed at zero".
        """
        from methods.prlmc.anchor_residual import AnchorResidual

        anchor = AnchorResidual(hidden_size=self.hidden_size, rank=self.rank,
                               max_ratio=self.max_ratio, gate_init=0.0)
        with torch.no_grad():
            anchor.down.copy_(self.down.detach())
            anchor.up.copy_(self.up_graft.detach())
            anchor.gate_logit.copy_(self.gate_logit.detach())
        return anchor

    # -- initialisation from the finished AnchorOnly run --------------------
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
    def forward(self, slots, cond, context=None):
        """`[layers, N, D]` slots, `[N, 3]` conditions, `[N, kv]` context -> delta."""
        if slots.shape[-1] != self.hidden_size:
            raise ValueError(f"expected hidden {self.hidden_size}, got {slots.shape[-1]}")
        if cond.shape != (slots.shape[1], self.cond_size):
            raise ValueError(f"expected conditions {tuple((slots.shape[1], self.cond_size))},"
                             f" got {tuple(cond.shape)}")
        if self.use_cross:
            if context is None:
                raise ValueError("this module is cross-block conditioned: it needs a "
                                 "context, and got None")
            if context.shape[0] != slots.shape[1]:
                raise ValueError("one context row per slot")
        dtype = slots.dtype
        x = slots.float()
        bottleneck = F.gelu(F.linear(x, self.down.float()))
        film = F.linear(F.gelu(F.linear(cond.float(), self.cond_down.float())),
                        self.cond_up.float())
        scale = torch.tanh(film[:, : self.rank])
        shift = torch.tanh(film[:, self.rank:])
        bottleneck = bottleneck * (1.0 + scale.unsqueeze(0)) + shift.unsqueeze(0)
        if self.use_cross:
            # Zero-initialised: the first update sees V11's calibrator exactly.
            bottleneck = bottleneck + self.cross_proj(
                context.float().to(bottleneck.device)).unsqueeze(0)
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
        """Per-slot weight for the residual penalty: carriers held tighter than graft."""
        graft = cond[:, 2] > 0.5
        return torch.where(graft, torch.full_like(cond[:, 2], self.graft_weight),
                           torch.full_like(cond[:, 2], self.core_weight))


def _apply(chunk, delta):
    """RMS-preserving update -- `core_fusion.apply_delta`, the one V11 used.

    A plain `chunk + delta` moves the slot's RMS by up to the residual ceiling
    (15% / 50%), and the frozen reader expects the native norm range: the first
    evaluation run failed its own RMS check at 4.55e-2 for exactly this reason.
    """
    return apply_delta(chunk, delta.to(chunk.dtype))


@torch.no_grad()
def calibrate(window, meta, module, layer_chunk=8, context=None, return_report=False):
    """Module over ALL 2,304 slots: bounded residual, RMS restored, context supplied.

    Layer-chunked because the module is shared across layers: the chunk size cannot
    change the result, it only bounds the float32 working set.
    """
    cond = conditions(meta, window.shape[1])
    if module.use_cross and context is None:
        context, context_report = module.neighbours(window, meta, return_report=True)
    else:
        context_report = None
    pieces = []
    delta_sq = slot_sq = 0.0
    per_type = {0: [0.0, 0.0], 1: [0.0, 0.0]}
    for start in range(0, window.shape[0], layer_chunk):
        stop = min(start + layer_chunk, window.shape[0])
        chunk = window[start:stop]
        delta = module(chunk, cond, None if context is None else context)
        calibrated = _apply(chunk, delta)
        pieces.append(calibrated)
        weight = delta.detach().float().square()
        delta_sq += float(weight.sum())
        slot_sq += float(chunk.detach().float().square().sum())
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
        use_cross=bool(module.use_cross),
        neighbor_mode=(module.neighbor_mode if module.use_cross else None),
        k_neighbors=(module.k_neighbors if module.use_cross else None),
        cross_proj_norm=float(module.cross_proj.weight.detach().float().norm())
        if module.use_cross else None,
        gate=float(torch.sigmoid(module.gate_logit.detach().float())))
    if context_report is not None:
        info["cross_context"] = context_report
    if return_report:
        return torch.cat(pieces, dim=0), info, context
    return torch.cat(pieces, dim=0), info


def load_cross(path, hidden_size, rank=None, max_ratio=None):
    """Rebuild a `CrossBlockResidual` from its checkpoint, refusing a mismatch."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA + "/checkpoint":
        raise ValueError(f"{path} is not a {SCHEMA} checkpoint")
    config = dict(payload["config"])
    if int(config.get("hidden_size", -1)) != int(hidden_size):
        raise ValueError(f"{path} was built for hidden {config.get('hidden_size')}, "
                         f"the reader is {hidden_size}")
    if rank is not None and int(config.get("rank", -1)) != int(rank):
        raise ValueError(f"{path} has rank {config.get('rank')}, expected {rank}")
    if max_ratio is not None and float(config.get("max_ratio", -1)) != float(max_ratio):
        raise ValueError(f"{path} has max_ratio {config.get('max_ratio')}")
    module = CrossBlockResidual(**config)
    module.load_state_dict(payload["state_dict"], strict=True)
    return module, config
