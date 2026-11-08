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

