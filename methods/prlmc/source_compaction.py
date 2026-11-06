"""Experiment 2: many-to-one compaction into the 360 graft slots.

Native grafting copies 360 slots of the ten dropped blocks verbatim into the
output and throws the other 2,200 away: nothing in those 2,200 reaches the
reader. This module is the only difference between the two arms of experiment 2:

    Graft       360 native anchors, copied verbatim                (V10, unchanged)
    Compaction  360 native anchors + a bounded residual that each anchor pulls
                from the slots of its OWN source block that were not selected

Capacity is untouched: 1,944 core + 360 graft, same positions, same order, same
quota. The core slots come back byte-identical -- the mirror image of experiment
1B, which moved the core and left the graft alone.

Why the anchors stay: an anchor is a verbatim copy of a real slot, so with a
zero-initialised output projection `Compaction` is bit-for-bit `Graft` at
initialisation and every later change is a measured addition rather than a
replacement. That is what makes "same capacity, more knowledge" a checked
property instead of a claim.

Why the source block only: the experiment asks whether the 2,200 discarded slots
can be summarised into the 360 that survive. A cross-block read would be a
second, unrelated fusion channel, and 1B already showed that a channel the loss
does not ask for can cost accuracy.

Supervision is real answer cross-entropy (`tools/train_source_compaction.py`),
not the readout distance experiment 1B used: 1B showed the readout distance can be
optimised to completion while accuracy falls, so it is not a training target here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.prlmc.core_fusion import apply_delta
from tools.v9_geometry_50block import GEOMETRY_50B

SCHEMA = "source_compaction/1"


# ---------------------------------------------------------------------------
# the anchors, and the leftovers of their source blocks
# ---------------------------------------------------------------------------
def split_source_anchors(telemetry, block, window_blocks, device=None):
    """The 360 anchors and, for each, the unselected slots of its source block.

    `telemetry["bridge_slot_offsets"]` is where the graft slots land in the
    output and `telemetry["slot_source_positions"]` is the window position each
    output slot was copied from. The anchors are therefore the graft slots
    themselves -- never "the last 360 output entries", because the output is
    sorted chronologically and the graft slots are interleaved with the carriers.

    An anchor's candidates are every position of its own source block that did
    not itself become an anchor. The result carries the block ids so the module
    can mask the cross-attention to same-block pairs instead of trusting the
    caller to have grouped the rows.
    """
    offsets = [int(offset) for offset in telemetry["bridge_slot_offsets"]]
    positions = [int(position) for position in telemetry["slot_source_positions"]]
    if not offsets or max(offsets) >= len(positions):
        raise ValueError("a graft offset falls outside the compressed window")
    selected = set(int(block_index) for block_index in telemetry["selected_blocks"])
    dropped = sorted(set(range(window_blocks)) - selected)
    anchor_positions = [positions[offset] for offset in offsets]
    anchor_blocks = [position // block for position in anchor_positions]
    stray = sorted(set(anchor_blocks) - set(dropped))
    if stray:
        raise ValueError(f"an anchor came from block {stray}, which this compaction "
                         f"did not drop: the anchors are not the graft slots")

    chosen = set(anchor_positions)
    candidate_positions, candidate_blocks = [], []
    for block_index in dropped:
        for position in range(block_index * block, (block_index + 1) * block):
            if position not in chosen:
                candidate_positions.append(position)
                candidate_blocks.append(block_index)
    if not candidate_positions:
        raise ValueError("the dropped blocks have no unselected slot to compact")

    def tensor(values):
        return torch.as_tensor(values, dtype=torch.long, device=device)

    anchor_blocks_t = tensor(anchor_blocks)
    candidate_blocks_t = tensor(candidate_blocks)
    same_block = anchor_blocks_t[:, None] == candidate_blocks_t[None, :]
    if not bool(same_block.any(dim=-1).all()):
        raise ValueError("an anchor has no unselected slot of its own block")

    def per_block(blocks):
        return {str(block_index): int(sum(1 for b in blocks if b == block_index))
                for block_index in dropped}

    return dict(
        anchor_offsets=tensor(offsets),
        anchor_positions=tensor(anchor_positions),
        anchor_blocks=anchor_blocks_t,
        candidate_positions=tensor(candidate_positions),
        candidate_blocks=candidate_blocks_t,
        same_block=same_block,
        dropped_blocks=dropped,
        anchors_per_block=per_block(anchor_blocks),
        candidates_per_block=per_block(candidate_blocks),
    )


# ---------------------------------------------------------------------------
# the module
# ---------------------------------------------------------------------------
class SourceCompaction(nn.Module):
    """One cross-attention pass from the anchors to their blocks' leftovers.

    A single projection set is shared by every reader layer: 4 x rank x hidden + 1
    parameters (0.52 M at rank 32 / hidden 4096) against the 8 B-parameter reader.
    The residual is normalised by each anchor's own per-layer RMS, so the shared
    map sees a comparable input at every depth.

    The residual is bounded three ways, all inherited from the V10 operator:
    `tanh` keeps it within `max_ratio` of the anchor's RMS, the sigmoid gate
    scales that bound down, and the caller restores the anchor's RMS afterwards,
    so no key or value leaves the native norm range the frozen reader expects.

    The gate starts at sigmoid(0) = 0.5: `out_proj` is exactly zero, so the window
    already starts as the identity and a damped gate would only slow the first
    updates down by 8x (measured on 1B, which started at -2.0 by mistake).
    """

    def __init__(self, hidden_size=GEOMETRY_50B.hidden_size,
                 rank=GEOMETRY_50B.ortho_adapter_rank, max_ratio=0.5,
                 gate_init=0.0):
        super().__init__()
        if rank <= 0 or hidden_size <= 0:
            raise ValueError("rank and hidden_size must be positive")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.max_ratio = float(max_ratio)
        self.q_proj = nn.Parameter(torch.empty(self.rank, self.hidden_size))
        self.k_proj = nn.Parameter(torch.empty(self.rank, self.hidden_size))
        self.v_proj = nn.Parameter(torch.empty(self.rank, self.hidden_size))
        # Zero, so the compacted window starts out identical to the grafted one.
        self.out_proj = nn.Parameter(torch.zeros(self.hidden_size, self.rank))
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        for projection in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.orthogonal_(projection)
        self.scale = self.rank ** -0.5

    def forward(self, anchors, candidates, same_block):
        """`[K, N, D]` anchors + `[K, M, D]` candidates -> `[K, N, D]` delta."""
        if anchors.shape[0] != candidates.shape[0]:
            raise ValueError("anchors and candidates must carry the same layers")
        if anchors.shape[-1] != self.hidden_size:
            raise ValueError(f"expected hidden {self.hidden_size}, got "
                             f"{anchors.shape[-1]}")
        if same_block.shape != (anchors.shape[1], candidates.shape[1]):
            raise ValueError(f"the same-block mask is {tuple(same_block.shape)}, "
                             f"expected {(anchors.shape[1], candidates.shape[1])}")
        dtype = anchors.dtype
        anchor = anchors.float()
        candidate = candidates.float()
        query = F.linear(anchor, self.q_proj.float())
        key = F.linear(candidate, self.k_proj.float())
        value = F.linear(candidate, self.v_proj.float())
        logits = torch.matmul(query, key.transpose(1, 2)) * self.scale
        logits = logits.masked_fill(~same_block[None], -1e4)
        weights = torch.softmax(logits, dim=-1)
        context = torch.matmul(weights, value)
        raw = F.linear(F.gelu(context), self.out_proj.float())
        rms = anchor.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        gate = torch.sigmoid(self.gate_logit.float()) * self.max_ratio
        return (gate * rms * torch.tanh(raw / rms)).to(dtype)


def compact_anchors(output, window_memory, parts, module, layer_chunk=8):
    """The compacted graft slots alone -- `[layers, 360, hidden]`, with the graph.

    Training injects only these 360 slots into the reader's pool, so it never has
    to materialise a 2,304-slot window or its gradient (604 MB each way at 32
    layers, fp16). Evaluation wants the whole window and calls `compact_window`.

    `output` is the operator's compressed window and `window_memory` the
    pre-compression one the candidates are read from. Layers are processed in
    chunks purely to bound the fp32 working set; the module is shared across
    layers, so the chunk size changes nothing about the result.
    """
    anchors = output.index_select(1, parts["anchor_offsets"])
    pieces = []
    delta_sq = anchor_sq = 0.0
    for start in range(0, output.shape[0], layer_chunk):
        stop = min(start + layer_chunk, output.shape[0])
        candidates = window_memory[start:stop].index_select(1, parts["candidate_positions"])
        chunk = anchors[start:stop]
        delta = module(chunk, candidates, parts["same_block"])
        # `apply_delta` upcasts to fp32; doing it per chunk keeps that working set
        # at 8 layers instead of 32, which is ~300 MB of the peak this step has to
        # fit on a 40 GB card.
        pieces.append(apply_delta(chunk, delta))
        delta_sq += float(delta.detach().float().square().sum())
        anchor_sq += float(chunk.detach().float().square().sum())
        del candidates, delta
    return torch.cat(pieces, dim=0), dict(
        anchor_slots=int(anchors.shape[1]),
        candidate_slots=int(parts["candidate_positions"].numel()),
        relative_delta=(delta_sq / max(anchor_sq, 1e-24)) ** 0.5,
        gate=float(torch.sigmoid(module.gate_logit.detach().float())),
    )


def compact_window(output, window_memory, parts, module, layer_chunk=8):
    """The whole window with the compacted graft slots written back; core untouched."""
    anchors, info = compact_anchors(output, window_memory, parts, module, layer_chunk)
    return output.index_copy(1, parts["anchor_offsets"], anchors), info
