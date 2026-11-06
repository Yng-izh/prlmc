"""Round 3, experiment 2: keep the anchor calibration, try to add knowledge back.

Experiment 1 (anchor-only) showed the +10.13 points of experiment 2 survive the
DELETION of the 2,200 candidate slots: 56/158 against 57/158. The counterfactual
had already shown they survive their REPLACEMENT (158/158 answers unchanged). So
neither the identity nor the presence of those slots produced the gain -- the
anchors' own representation adjustment did.

That leaves the one question this module exists for: once the anchors are
calibrated, can reading the leftovers ADD anything the calibration does not
already give?

    AnchorOnly      graft slots -> frozen AnchorResidual -> 360 slots   (56/158)
    AnchorFusion    the same, plus a trainable branch that reads the block's
                    leftovers and adds a bounded residual

The AnchorResidual is frozen byte for byte -- only the new branch has a gradient.
The branch starts at exactly zero (`fuse_up` is zero-initialised), so the fused
window IS the AnchorOnly window at initialisation: 56/158 is the starting point
and every later point is a measured addition, not a replacement.

What is different from experiment 2's module
--------------------------------------------
Three things, each aimed at a measured property of `SourceCompaction`:

1. sparse selection, not full aggregation. Experiment 2 mixed all ~220 leftovers
   of a block with a softmax whose measured entropy was 2.61 nats (about 14
   effective slots) and whose maximum weight was 0.382: a broad average, not a
   lookup. Here the branch scores every leftover of the block and keeps the best
   `top_k = 4`, then softmaxes over those four alone.

2. native width, not rank width. Experiment 2 projected every candidate to rank
   32 BEFORE mixing, so the mixture could never contain a direction the 32-row
   projection had dropped. Here the mixing happens at the candidates' native
   4,096 dimensions -- the four winners are combined as full-width vectors, and
   only the result is compressed.

3. a norm clamp, not an element-wise `tanh`. Experiment 2's bound was applied
   coordinate by coordinate, and 64% of the trained residual's entries sat at
   |tanh| > 0.99: the residual was a sign pattern whose magnitude the bound had
   pinned, so the loss could only move it by rotating it. This branch bounds the
   residual's VECTOR norm and leaves its direction alone:

       bound = gate * max_ratio * rms(anchor)
       scale = bound / (bound + rms(raw))
       delta = raw * scale

   `scale` lies in (0, 1), its derivative `bound^2 / (bound + rms)^2` decays
   polynomially rather than exponentially, and no coordinate saturates on its
   own: the residual can still grow past the bound instead of freezing at it.

What the branch reads
---------------------
The novelty, `fused - base`: what the selected leftovers say that the calibrated
anchor does not already say. A branch that only re-states the anchor produces a
zero residual, which is the behaviour wanted here -- the gain being measured is
knowledge the anchor does not have.

Capacity, again
---------------
4 x 32 x 4096 + 1 = 524,289 trainable parameters, the same count as experiment 2's
module and as the frozen AnchorResidual it sits on. The memory budget is
untouched: 1,944 core + 360 graft, same positions, same order, same quota.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.prlmc.core_fusion import apply_delta
from methods.prlmc.key_safe_projection import key_safe_merge
from tools.v9_geometry_50block import GEOMETRY_50B

SCHEMA = "anchor_fusion/1"


# ---------------------------------------------------------------------------
# the trainable half
# ---------------------------------------------------------------------------
class SparseCandidateFusion(nn.Module):
    """Top-k over a block's leftovers, mixed at native width, bounded by norm.

    4 x rank x hidden + 1 parameters: a bilinear score (`score_q`, `score_k`), a
    bottleneck on the novelty (`fuse_down`, `fuse_up`) and one gate. At rank 32 /
    hidden 4096 that is 524,289, exactly experiment 2's module and exactly the
    frozen `AnchorResidual` it sits on.

    The score is bilinear rather than a plain dot product because a plain dot
    product has no parameters: it would fix the selection at initialisation and
    the only thing training could move is how the four winners are mixed.
    """

    def __init__(self, hidden_size=GEOMETRY_50B.hidden_size,
                 rank=GEOMETRY_50B.ortho_adapter_rank, top_k=4, max_ratio=0.5,
                 gate_init=0.0):
        super().__init__()
        if rank <= 0 or hidden_size <= 0:
            raise ValueError("rank and hidden_size must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.top_k = int(top_k)
        self.max_ratio = float(max_ratio)
        self.score_q = nn.Parameter(torch.empty(self.rank, self.hidden_size))
        self.score_k = nn.Parameter(torch.empty(self.rank, self.hidden_size))
        self.fuse_down = nn.Parameter(torch.empty(self.rank, self.hidden_size))
        # Zero, so the fused window starts out bit-identical to the AnchorOnly one.
        self.fuse_up = nn.Parameter(torch.zeros(self.hidden_size, self.rank))
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        for projection in (self.score_q, self.score_k, self.fuse_down):
            nn.init.orthogonal_(projection)
        self.scale = self.rank ** -0.5

    def forward(self, anchors, candidates, same_block):
        """`[K,N,D]` anchors + `[K,M,D]` candidates -> `([K,N,D]` delta, stats).

        Returns the statistics with the delta because they are free here and
        expensive to recover afterwards: whether the top-4 collapsed onto one
        slot, whether the novelty carries any signal, and how far the residual
        moved are what tell a null result from a broken run.
        """
        if anchors.shape[0] != candidates.shape[0]:
            raise ValueError("anchors and candidates must carry the same layers")
        if anchors.shape[-1] != self.hidden_size:
            raise ValueError(f"expected hidden {self.hidden_size}, got "
                             f"{anchors.shape[-1]}")
        if same_block.shape != (anchors.shape[1], candidates.shape[1]):
            raise ValueError(f"the same-block mask is {tuple(same_block.shape)}, "
                             f"expected {(anchors.shape[1], candidates.shape[1])}")
        if same_block.shape[1] < self.top_k:
            raise ValueError(f"a block has fewer than {self.top_k} leftovers to select")
        dtype = anchors.dtype
        anchor = anchors.float()
        candidate = candidates.float()

        query = F.linear(anchor, self.score_q.float())
        key = F.linear(candidate, self.score_k.float())
        logits = torch.matmul(query, key.transpose(1, 2)) * self.scale
        logits = logits.masked_fill(~same_block[None], -1e4)
        best = torch.topk(logits, self.top_k, dim=-1)
        weights = torch.softmax(best.values, dim=-1)

        # Native width: the winners are gathered at their own 4,096 dimensions and
        # only the mixture is compressed, so no direction is dropped before mixing.
        gathered = torch.gather(
            candidate.unsqueeze(1).expand(-1, anchor.shape[1], -1, -1), 2,
            best.indices.unsqueeze(-1).expand(-1, -1, -1, candidate.shape[-1]))
        fused = (weights.unsqueeze(-1) * gathered).sum(dim=2)

        novelty = fused - anchor
        raw = F.linear(F.gelu(F.linear(novelty, self.fuse_down.float())),
                       self.fuse_up.float())
        anchor_rms = anchor.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        # `clamp_min` BEFORE the square root, not after: `sqrt` has an infinite
        # derivative at zero and `raw` is exactly zero at initialisation, which is
        # where a NaN would silently poison the first update. Clamping the mean
        # square puts the clamp's own zero derivative in front of the singularity.
        raw_rms = raw.square().mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        bound = torch.sigmoid(self.gate_logit.float()) * self.max_ratio * anchor_rms
        # Vector-level clamp: one scale per slot, so the direction of `raw` is
        # untouched and no coordinate can saturate against the bound on its own.
        scale = bound / (bound + raw_rms).clamp_min(1e-12)
        delta = raw * scale

        with torch.no_grad():
            stats = dict(
                slots=int(anchor.shape[0] * anchor.shape[1]),
                max_weight=float(weights.max(dim=-1).values.mean()),
                entropy=float(-(weights * weights.clamp_min(1e-9).log()).sum(dim=-1)
                              .mean()),
                novelty_ratio=float(novelty.square().mean().sqrt()
                                    / anchor.square().mean().sqrt().clamp_min(1e-12)),
                raw_rms_ratio=float(raw_rms.mean() / anchor_rms.mean().clamp_min(1e-12)),
            )
        return delta.to(dtype), stats


# ---------------------------------------------------------------------------
# the frozen half, and the two together
# ---------------------------------------------------------------------------
class FrozenAnchorFusion(nn.Module):
    """A frozen `AnchorResidual` with a trainable `SparseCandidateFusion` on top.

    The composition is `base = apply_delta(anchors, anchor(anchors))` and then the
    branch's residual, i.e. exactly the AnchorOnly arm followed by the new module.
    With `fuse_up = 0` the second step is the identity -- bit for bit, since
    `rms_preserve` divides by the same RMS it multiplied by -- so the fused window
    starts as the AnchorOnly window.

    Two ways to apply that residual:

    `plain`      `apply_delta(base, delta)`: the residual moves all 4,096 dimensions
                 the reader uses for BOTH its keys and its values. This is what
                 experiment 3 did, and it is why experiment 3 lost six questions.
    `key-safe`   `key_safe_merge(base, delta, basis)`: the residual is confined to
                 the null space of the layer's key map and the slot's L2 norm is
                 held, so the reader's keys do not move while its values do. See
                 `methods/prlmc/key_safe_projection.py` for why both halves of that
                 condition are needed and what "does not move" means at fp16.

    The mode is part of the module, not a flag at the call site: a checkpoint that
    was trained one way must be evaluated the same way, and `fusion_mode` is what
    the driver records and rebuilds from.

    `forward` returns the frozen stage as well: it is the AnchorOnly arm, and the
    driver compares it against that arm's own code path to prove the baseline
    underneath the new branch is the baseline being claimed.
    """

    def __init__(self, anchor, fusion, key_bases=None, key_safe=False):
        super().__init__()
        self.anchor = anchor
        self.fusion = fusion
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)
        self.anchor.eval()
        self.key_safe = bool(key_safe)
        # Non-persistent: the basis is 32 x 4096 x 1024 floats (537 MB) and is a
        # function of the frozen reader, so it is rebuilt from the reader and
        # matched by digest instead of being copied into every checkpoint.
        self.register_buffer("key_bases",
                             torch.empty(0) if key_bases is None else key_bases.float(),
                             persistent=False)

    @property
    def fusion_mode(self):
        return "key-safe" if self.key_safe else "plain"

    @property
    def gate_logit(self):
        """The FUSION gate; the frozen anchor gate is fixed at its trained value."""
        return self.fusion.gate_logit

    def anchor_slots(self, anchors):
        return apply_delta(anchors, self.anchor(anchors))

    def forward(self, anchors, candidates, same_block, layer_start=0):
        base = self.anchor_slots(anchors)
        delta, stats = self.fusion(base, candidates, same_block)
        if not self.key_safe:
            return apply_delta(base, delta), base, stats
        if self.key_bases.numel() == 0:
            raise ValueError("this module is in key-safe mode but carries no key "
                             "basis: build it with `build_key_bases(reader)`")
        basis = self.key_bases[layer_start:layer_start + base.shape[0]]
        if basis.shape[0] != base.shape[0]:
            raise ValueError(f"the key basis covers layers "
                             f"{layer_start}..{layer_start + basis.shape[0]}, the "
                             f"window carries {base.shape[0]}")
        fused, merge = key_safe_merge(base, delta, basis)
        # No `apply_delta` here on purpose: it would rescale the merged slot and
        # move the RMSNorm denominator the constraint just fixed.
        return fused, base, dict(stats, **merge)


# ---------------------------------------------------------------------------
# window plumbing, the same shape as source_compaction's
# ---------------------------------------------------------------------------
def frozen_stage(output, parts, module, layer_chunk=8):
    """The frozen anchor stage alone -- the AnchorOnly arm, from this object."""
    anchors = output.index_select(1, parts["anchor_offsets"])
    pieces = []
    with torch.no_grad():
        for start in range(0, output.shape[0], layer_chunk):
            pieces.append(module.anchor_slots(anchors[start:start + layer_chunk]))
    return torch.cat(pieces, dim=0)


def compact_anchors(output, window_memory, parts, module, layer_chunk=8):
    """The fused graft slots alone -- `[layers, 360, hidden]`, with the graph.

    Training injects only these 360 slots into the reader's pool, so it never has
    to materialise a 2,304-slot window or its gradient. Layers are processed in
    chunks purely to bound the fp32 working set; the module is shared across
    layers, so the chunk size changes nothing about the result.
    """
    anchors = output.index_select(1, parts["anchor_offsets"])
    pieces = []
    base_sq = fused_sq = delta_sq = anchor_sq = source_sq = 0.0
    weighted = dict(max_weight=0.0, entropy=0.0, novelty_ratio=0.0, raw_rms_ratio=0.0)
    # Only present in key-safe mode; summed the same way so the report has one row
    # per sample whatever the mode.
    optional = ("row_space_fraction", "null_shift", "row_leak", "norm_preserved")
    extra = {name: 0.0 for name in optional}
    present = set()
    slots = 0
    for start in range(0, output.shape[0], layer_chunk):
        stop = min(start + layer_chunk, output.shape[0])
        candidates = window_memory[start:stop].index_select(1, parts["candidate_positions"])
        chunk = anchors[start:stop]
        # `layer_start` is what lets a key-safe module pick the basis of the layer it
        # is writing into; the module is shared across layers, the basis is not.
        fused, base, stats = module(chunk, candidates, parts["same_block"],
                                    layer_start=start)
        pieces.append(fused)
        base_sq += float(base.detach().float().square().sum())
        fused_sq += float(fused.detach().float().square().sum())
        delta_sq += float((fused.detach().float() - base.detach().float()).square().sum())
        anchor_sq += float((base.detach().float() - chunk.detach().float()).square().sum())
        source_sq += float(chunk.detach().float().square().sum())
        for name in weighted:
            weighted[name] += stats[name] * stats["slots"]
        for name in optional:
            if name in stats:
                extra[name] += stats[name] * stats["slots"]
                present.add(name)
        slots += stats["slots"]
        del candidates, fused, base
    info = dict(
        anchor_slots=int(anchors.shape[1]),
        candidate_slots=int(parts["candidate_positions"].numel()),
        top_k=int(module.fusion.top_k),
        # The fusion branch's own contribution: RMS(fused - base) / RMS(base).
        relative_delta=(delta_sq / max(base_sq, 1e-24)) ** 0.5,
        # The frozen branch's contribution: RMS(base - anchor) / RMS(anchor).
        anchor_relative_delta=(anchor_sq / max(source_sq, 1e-24)) ** 0.5,
        gate=float(torch.sigmoid(module.fusion.gate_logit.detach().float())),
        # `rms_preserve` restores each slot's RMS, so this must be 1.0: the branch
        # may change what a slot says but never how loud it is.
        rms_preserved=(fused_sq / max(base_sq, 1e-24)) ** 0.5,
        fusion_mode=module.fusion_mode,
    )
    info.update({name: value / max(slots, 1) for name, value in weighted.items()})
    info.update({name: extra[name] / max(slots, 1) for name in sorted(present)})
    return torch.cat(pieces, dim=0), info


def compact_window(output, window_memory, parts, module, layer_chunk=8):
    """The whole window with the fused graft slots written back; core untouched."""
    anchors, info = compact_anchors(output, window_memory, parts, module, layer_chunk)
    return output.index_copy(1, parts["anchor_offsets"], anchors), info
