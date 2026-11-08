"""Round 3, experiment 1: the anchor alone, with the candidate path removed.

The counterfactual ablation (`tools/eval_counterfactual_candidates.py`) replaced
every candidate slot with another document's and left all 158 answers unchanged,
so experiment 2's +10.13 points cannot come from the leftovers. This module
removes the candidate path instead of replacing it:

    SourceCompaction   360 anchors -> attention over 2,200 candidates -> residual
    AnchorResidual     360 anchors -> a low-rank map of the anchor     -> residual

Same 360 native anchors, same frozen reader, same answer cross-entropy, same
update count, same three bounds on the residual, and -- because the map is
`hidden -> rank -> hidden` -- exactly the same parameter count at rank 64:
2 x 64 x 4096 + 1 = 524,289, against the attention module's 4 x 32 x 4096 + 1.

What the comparison decides
---------------------------
* AnchorOnly close to Fusion (57/158): the gain is a representation adjustment of
  the anchors themselves, and the candidate path was decoration.
* AnchorOnly close to Graft (41/158): the candidate path does carry something the
  anchors do not -- not the identity of the slots (the counterfactual ruled that
  out) but some statistic of the block.

`forward` takes the anchors and nothing else. The module cannot read a candidate,
which makes "it does not use the candidates" a property of the code rather than a
claim in a report -- the same reason experiment 2's anchors are verbatim copies
instead of "approximately preserved" vectors.

Why the residual is a function of the anchor here and not there
--------------------------------------------------------------
SourceCompaction's delta comes from the attention context alone, so its direction
is a property of the candidates and only its scale comes from the anchor. This
one is the mirror image: the direction is the anchor's, which is what a "the gain
is just the anchor being adjusted" hypothesis predicts.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.prlmc.core_fusion import apply_delta
from tools.v9_geometry_50block import GEOMETRY_50B

SCHEMA = "anchor_residual/1"


class AnchorResidual(nn.Module):
    """`gelu` bottleneck from the anchor to its own residual, 0.52 M parameters.

    The residual is bounded exactly as in experiment 2: `tanh` keeps it within
    `max_ratio` of the anchor's RMS, the sigmoid gate scales that bound down, and
    the caller restores the anchor's RMS afterwards, so no key or value leaves the
    native norm range the frozen reader expects. `up` starts at zero, so the
    compacted window is bit-for-bit the grafted one at initialisation and every
    later change is a measured addition rather than a replacement.
    """

    def __init__(self, hidden_size=GEOMETRY_50B.hidden_size,
                 rank=2 * GEOMETRY_50B.ortho_adapter_rank, max_ratio=0.5,
                 gate_init=0.0):
        super().__init__()
        if rank <= 0 or hidden_size <= 0:
            raise ValueError("rank and hidden_size must be positive")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.max_ratio = float(max_ratio)
        self.down = nn.Parameter(torch.empty(self.rank, self.hidden_size))
        self.up = nn.Parameter(torch.zeros(self.hidden_size, self.rank))
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        nn.init.orthogonal_(self.down)

    def forward(self, anchors):
        """`[K, N, D]` anchors -> `[K, N, D]` delta. No candidates by construction."""
        if anchors.shape[-1] != self.hidden_size:
            raise ValueError(f"expected hidden {self.hidden_size}, got "
                             f"{anchors.shape[-1]}")
        dtype = anchors.dtype
        anchor = anchors.float()
        hidden = F.gelu(F.linear(anchor, self.down.float()))
        raw = F.linear(hidden, self.up.float())
        rms = anchor.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        gate = torch.sigmoid(self.gate_logit.float()) * self.max_ratio
        return (gate * rms * torch.tanh(raw / rms)).to(dtype)


def compact_anchors(output, parts, module, layer_chunk=8):
    """The compacted graft slots alone -- `[layers, 360, hidden]`, with the graph.

    The same return shape as `source_compaction.compact_anchors`, minus the window
    the candidates would have come from: this path has no candidate tensor to
    gather, so the signature cannot accidentally carry one.
    """
    anchors = output.index_select(1, parts["anchor_offsets"])
    pieces = []
    delta_sq = anchor_sq = 0.0
    for start in range(0, output.shape[0], layer_chunk):
        stop = min(start + layer_chunk, output.shape[0])
        chunk = anchors[start:stop]
        delta = module(chunk)
        pieces.append(apply_delta(chunk, delta))
        delta_sq += float(delta.detach().float().square().sum())
        anchor_sq += float(chunk.detach().float().square().sum())
        del delta
    return torch.cat(pieces, dim=0), dict(
        anchor_slots=int(anchors.shape[1]),
        candidate_slots=0,
        relative_delta=(delta_sq / max(anchor_sq, 1e-24)) ** 0.5,
        gate=float(torch.sigmoid(module.gate_logit.detach().float())),
    )


def compact_window(output, parts, module, layer_chunk=8):
    """The whole window with the compacted graft slots written back; core untouched."""
    anchors, info = compact_anchors(output, parts, module, layer_chunk)
    return output.index_copy(1, parts["anchor_offsets"], anchors), info


# ---------------------------------------------------------------------------
# Round 6, module 3: the same bounded residual, over the whole pool
# ---------------------------------------------------------------------------
FULL_POOL_SCHEMA = "full_pool_residual/1"


class FullPoolResidual(nn.Module):
    """`f_theta(h, dpos, log|C|, t)` -- `AnchorResidual` widened to every slot.

    Round 3 established that a 0.52 M low-rank map from an anchor to its own
    bounded residual is the only mechanism in this project with a paired gain over
    the grafted window (56/158 vs 41/158, p = 0.011). This keeps that map exactly
    -- same `down`, same rank, same `tanh` bound, same RMS restore -- and adds the
    two things a joint window needs:

    1. **Two output halves.** The graft half is initialised from the finished
       AnchorOnly checkpoint, so the calibration that already works is where
       training starts. The core half is zero-initialised, so at update 0 no carrier
       slot moves at all and the only change to the window is the one that was
       already measured to help.
    2. **A condition vector**, `[delta_position, log(|C|), is_graft]`, injected as a
       zero-initialised FiLM on the bottleneck. A folded slot (`|C| > 1`) and a
       verbatim one are not the same object, and the calibration has to be able to
       treat them differently without being told which arm it is evaluating.

    At initialisation `up_core = 0` and `cond_up = 0`, so the core slots are
    bit-identical to the window and the graft slots are bit-identical to what
    `AnchorResidual` would produce. That equality is asserted, not assumed: a joint
    architecture that cannot reproduce its own baseline on its first step is not a
    baseline.
    """

    def __init__(self, hidden_size=GEOMETRY_50B.hidden_size,
                 rank=2 * GEOMETRY_50B.ortho_adapter_rank, max_ratio=0.5,
                 gate_init=0.0, cond_size=3, cond_rank=32):
        super().__init__()
        if rank <= 0 or hidden_size <= 0:
            raise ValueError("rank and hidden_size must be positive")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.max_ratio = float(max_ratio)
        self.cond_size = int(cond_size)
        self.cond_rank = int(cond_rank)
        self.down = nn.Parameter(torch.empty(self.rank, self.hidden_size))
        self.up_graft = nn.Parameter(torch.zeros(self.hidden_size, self.rank))
        self.up_core = nn.Parameter(torch.zeros(self.hidden_size, self.rank))
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        # Zero, so `sigmoid(gate_logit)` and `sigmoid(gate_logit + 0)` agree at the
        # start: the graft half begins exactly where AnchorOnly left it.
        self.type_gate_logit = nn.Parameter(torch.zeros(()))
        # Only the SECOND half of the FiLM is zero-initialised. Zeroing both would
        # be a deadlock, not a conservative start: `cond_up`'s gradient is its own
        # input `gelu(cond @ cond_down^T)`, which is zero when `cond_down` is zero,
        # and `cond_down`'s gradient is `cond_up^T (...)`, which is zero when
        # `cond_up` is zero -- so neither ever leaves zero and the condition stays
        # off forever. With `cond_down` small-random and `cond_up` zero, the film is
        # still EXACTLY zero at initialisation (multiplying by a zero matrix), so the
        # identity-at-step-0 property is untouched, and both halves now have a route
        # out. Measured on the smoke run: the all-zero version died here.
        self.cond_down = nn.Parameter(torch.empty(self.cond_rank, self.cond_size))
        self.cond_up = nn.Parameter(torch.zeros(2 * self.rank, self.cond_rank))
        nn.init.orthogonal_(self.down)
        nn.init.orthogonal_(self.cond_down, gain=0.1)

    # -- initialisation from the finished AnchorOnly run --------------------
    @torch.no_grad()
    def load_anchor_only(self, state_dict, config=None):
        """Copy the measured calibration into the graft half and check it landed."""
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

    def anchor_view(self):
        """`AnchorResidual` with this module's graft parameters, for the equality check."""
        anchor = AnchorResidual(hidden_size=self.hidden_size, rank=self.rank,
                                max_ratio=self.max_ratio,
                                gate_init=0.0)
        with torch.no_grad():
            anchor.down.copy_(self.down.detach())
            anchor.up.copy_(self.up_graft.detach())
            anchor.gate_logit.copy_(self.gate_logit.detach())
        return anchor

    # -- the residual -------------------------------------------------------
    def forward(self, slots, cond):
        """`[layers, N, D]` slots and `[N, 3]` conditions -> `[layers, N, D]` delta."""
        if slots.shape[-1] != self.hidden_size:
            raise ValueError(f"expected hidden {self.hidden_size}, got {slots.shape[-1]}")
        if cond.shape != (slots.shape[1], self.cond_size):
            raise ValueError(f"expected conditions {tuple((slots.shape[1], self.cond_size))}, "
                             f"got {tuple(cond.shape)}")
        dtype = slots.dtype
        x = slots.float()
        bottleneck = F.gelu(F.linear(x, self.down.float()))
        # Zero-initialised FiLM: exactly 0 at update 0, so the graft half is the
        # frozen anchor's map and the core half is the identity.
        film = F.linear(F.gelu(F.linear(cond.float(), self.cond_down.float())),
                        self.cond_up.float())
        scale = torch.tanh(film[:, : self.rank]).to(dtype).float()
        shift = torch.tanh(film[:, self.rank:]).to(dtype).float()
        bottleneck = bottleneck * (1.0 + scale.unsqueeze(0)) + shift.unsqueeze(0)
        # `sqrt` at zero has an infinite derivative; clamp the MEAN SQUARE, which is
        # the quantity that is actually floored, not the RMS.
        rms = x.square().mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        graft = F.linear(bottleneck, self.up_graft.float())
        core = F.linear(bottleneck, self.up_core.float())
        mask = cond[:, 2].float().view(1, -1, 1)
        raw = mask * graft + (1.0 - mask) * core
        gate = torch.sigmoid(self.gate_logit.float()
                             + mask * self.type_gate_logit.float()) * self.max_ratio
        return (gate * rms * torch.tanh(raw / rms)).to(dtype)


def load_full_pool(path, hidden_size, rank=None, max_ratio=None):
    """Rebuild a `FullPoolResidual` from its checkpoint, refusing a mismatched reader."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != FULL_POOL_SCHEMA + "/checkpoint":
        raise ValueError(f"{path} is not a {FULL_POOL_SCHEMA} checkpoint")
    config = dict(payload["config"])
    if int(config.get("hidden_size", -1)) != int(hidden_size):
        raise ValueError(f"{path} was built for hidden {config.get('hidden_size')}, "
                         f"the reader is {hidden_size}")
    if rank is not None and int(config.get("rank", -1)) != int(rank):
        raise ValueError(f"{path} has rank {config.get('rank')}, expected {rank}")
    if max_ratio is not None and float(config.get("max_ratio", -1)) != float(max_ratio):
        raise ValueError(f"{path} has max_ratio {config.get('max_ratio')}")
    module = FullPoolResidual(**config)
    module.load_state_dict(payload["state_dict"], strict=True)
    return module, config
