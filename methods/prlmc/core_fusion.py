"""Experiment 1B: many-to-one core fusion, applied before the graft.

Native grafting deletes 40 of a carrier block's 256 slots and refills those 40
positions with slots copied verbatim from the blocks the compaction drops.
Whatever the deleted 40 held is gone the moment they are overwritten. This
module is the ONLY difference between the two arms of experiment 1B:

    Graft    delete the 40 slots, graft the 360           (V10, unchanged)
    Fusion   first fuse the 40 into the 216 KEPT slots of the SAME carrier
             block, then delete and graft exactly as above

The fusion is one cross-attention pass restricted to same-block pairs: every kept
slot asks the 40 doomed slots of its own carrier block what they hold, and adds a
bounded, norm-preserving residual to itself. The output projection is
zero-initialised, so at initialisation `Fusion` is bit-for-bit `Graft`; the module
only becomes non-trivial after training (`tools/train_core_fusion.py`).

Scope, deliberately narrow
--------------------------
The source is the 40 slots the SAME carrier block is about to lose. The ten
fully-dropped blocks are not a fusion source: they reach the reader only through
the 360 graft slots, exactly as in the Graft arm. Fusing them into the carriers
would be a different experiment (cross-block condensation), which is what the
outcome of 1B is supposed to decide whether to run at all.

Nothing here touches the reader, the writer or the V10 operator. `fuse_output`
writes back only at the core offsets, so the graft slots of the fused window are
byte-identical to the unfused one. That is what turns "same slots, same 360-slot
quota, same everything except the fusion" into a checked property of every
sample instead of a claim about the code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.v9_geometry_50block import GEOMETRY_50B

SCHEMA = "core_fusion/1"


# ---------------------------------------------------------------------------
# what the compaction keeps, what it deletes
# ---------------------------------------------------------------------------
def split_core_and_deleted(positions, graft_offsets, carrier_blocks, block,
                           core_slots_per_block, device=None):
    """Split one compressed window into its core slots and each carrier's 40 losses.

    `positions` is the operator's `slot_source_positions`: the window position of
    every output slot, in the order the reader will see them. `graft_offsets` is
    its `bridge_slot_offsets`, which is NOT "the last 360 entries" because the
    output is sorted chronologically.

    The deleted slots are not in the output at all, so they are reconstructed
    from the carrier blocks themselves: for each carrier, the block's 256
    positions minus the ones that survived. The result is checked to be exactly
    `block - core_slots_per_block` per carrier -- if the operator ever keeps a
    different number, the fusion would be aimed at the wrong set and the arm
    would silently stop being the experiment.
    """
    positions = [int(position) for position in positions]
    graft = set(int(offset) for offset in graft_offsets)
    if not graft <= set(range(len(positions))):
        raise ValueError("a graft offset falls outside the compressed window")
    core_offsets = [index for index in range(len(positions)) if index not in graft]
    core_positions = [positions[index] for index in core_offsets]

    carriers = [int(block_index) for block_index in carrier_blocks]
    if len(carriers) != len(set(carriers)):
        raise ValueError("a carrier block is named twice")
    kept = {block_index: [] for block_index in carriers}
    for position in core_positions:
        block_index, _ = divmod(position, block)
        if block_index not in kept:
            raise ValueError(f"a core slot came from block {block_index}, which is "
                             f"not a carrier: the window is not the one this "
                             f"compression produced")
        kept[block_index].append(position)
    deleted = []
    for block_index in carriers:
        if len(kept[block_index]) != core_slots_per_block:
            raise ValueError(f"carrier block {block_index} kept "
                             f"{len(kept[block_index])} slots, expected "
                             f"{core_slots_per_block}")
        survivors = set(kept[block_index])
        gone = [block_index * block + offset for offset in range(block)
                if block_index * block + offset not in survivors]
        if len(gone) != block - core_slots_per_block:
            raise ValueError(f"carrier block {block_index} deletes {len(gone)} slots, "
                             f"expected {block - core_slots_per_block}")
        deleted.extend(gone)

    def tensor(values):
        return torch.as_tensor(values, dtype=torch.long, device=device)

    return dict(
        core_offsets=tensor(core_offsets),
        core_positions=core_positions,
        deleted_positions=deleted,
        core_block_ids=tensor([position // block for position in core_positions]),
        deleted_block_ids=tensor([position // block for position in deleted]),
    )


# ---------------------------------------------------------------------------
# the module
# ---------------------------------------------------------------------------
class LearnedCoreFusion(nn.Module):
    """Same-block cross-attention from the kept slots to the doomed ones.

    One projection set is shared by every layer. That is the deliberate "light"
    part: 4 x rank x hidden + 1 parameters (0.52 M at rank 32 / hidden 4096),
    against the 8 B-parameter reader it has to work with. Layers are not
    interchangeable, but the residual is normalised by each slot's own per-layer
    RMS, so a shared map sees a comparable input at every depth.

    The residual is bounded three ways, all of them from the V10 operator:
    `tanh` keeps it within `max_ratio` of the slot's RMS, the sigmoid gate scales
    that bound down, and the caller restores the slot's original RMS afterwards,
    so no key or value leaves the native norm range the frozen reader expects.

    The gate starts at sigmoid(0) = 0.5 rather than at V10's -2.0: V10 needs the
    small initial gate because its adapter starts from a non-zero residual, while
    here `out_proj` is exactly zero, so the window is already the identity and a
    damped gate would only slow the first thousand updates by 8x.
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
        # Zero, so the fused window starts out identical to the unfused one.
        self.out_proj = nn.Parameter(torch.zeros(self.hidden_size, self.rank))
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        for projection in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.orthogonal_(projection)
        self.scale = self.rank ** -0.5

    def forward(self, core_layers, deleted_layers, core_block_ids, deleted_block_ids):
        """`[K, Nc, D]` kept slots + `[K, Nd, D]` doomed slots -> `[K, Nc, D]` delta."""
        if core_layers.shape[0] != deleted_layers.shape[0]:
            raise ValueError("core and deleted tensors must carry the same layers")
        if core_layers.shape[-1] != self.hidden_size:
            raise ValueError(f"expected hidden {self.hidden_size}, got "
                             f"{core_layers.shape[-1]}")
        if core_block_ids.numel() != core_layers.shape[1]:
            raise ValueError("one block id per core slot is required")
        if deleted_block_ids.numel() != deleted_layers.shape[1]:
            raise ValueError("one block id per deleted slot is required")
        dtype = core_layers.dtype
        core = core_layers.float()
        doomed = deleted_layers.float()
        query = F.linear(core, self.q_proj.float())
        key = F.linear(doomed, self.k_proj.float())
        value = F.linear(doomed, self.v_proj.float())
        logits = torch.matmul(query, key.transpose(1, 2)) * self.scale
        # Same-block only: a kept slot may not pull from another carrier's losses.
        mask = core_block_ids[:, None] == deleted_block_ids[None, :]
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("a core slot has no deleted slot of its own block")
        logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=-1)
        context = torch.matmul(weights, value)
        raw = F.linear(F.gelu(context), self.out_proj.float())
        rms = core.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        gate = torch.sigmoid(self.gate_logit.float()) * self.max_ratio
        return (gate * rms * torch.tanh(raw / rms)).to(dtype)


def rms_preserve(base, mixed):
    """`mixed` rescaled to `base`'s per-slot RMS (V10's `_geodesic_rms_preserve`)."""
    original = base.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    mixed = mixed.float()
    now = mixed.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    return (mixed * (original / now)).to(base.dtype)


def apply_delta(core, delta):
    """The one place a residual becomes a window: add it, restore the slot's RMS."""
    return rms_preserve(core, core + delta)


def fuse_output(output, window_memory, parts, fusion, layer_chunk=8):
    """Apply the fusion to a compressed window; the graft slots come back untouched.

    `output` is the operator's compressed window and `window_memory` the
    pre-compression one the deleted slots are read from. Layers are processed in
    chunks purely to bound the fp32 working set: the module is shared across
    layers, so the chunk size changes nothing about the result.
    """
    core = output.index_select(1, parts["core_offsets"])
    deleted_index = torch.as_tensor(parts["deleted_positions"], dtype=torch.long,
                                    device=output.device)
    deleted = window_memory.index_select(1, deleted_index)
    pieces = []
    for start in range(0, output.shape[0], layer_chunk):
        stop = min(start + layer_chunk, output.shape[0])
        pieces.append(fusion(core[start:stop], deleted[start:stop],
                             parts["core_block_ids"], parts["deleted_block_ids"]))
    delta = torch.cat(pieces, dim=0)
    fused_core = apply_delta(core, delta)
    fused = output.index_copy(1, parts["core_offsets"], fused_core)
    return fused, dict(
        core_slots=int(core.shape[1]),
        deleted_slots=int(deleted.shape[1]),
        relative_delta=float(delta.detach().float().norm()
                             / core.detach().float().norm().clamp_min(1e-6)),
        gate=float(torch.sigmoid(fusion.gate_logit.detach().float())),
    )


# ---------------------------------------------------------------------------
# the readout distance the module is trained against
# ---------------------------------------------------------------------------
def readout_distance(model, pre_windows, post_windows, query_hidden, layers,
                     query_tokens=8, slot_stride=1):
    """Reader-readout distance, identical maths to
    `tools.objective.compute_local_window_readout_loss` but with the distillation
    taps as an argument, so a HELD-OUT layer set can be measured without touching
    the sealed training objective.

    The reader is GQA (32 query heads, 8 KV heads), so K/V are expanded to the
    query-head count before the readout. `pre_windows` may be a plain dict of
    detached tensors and `post_windows` may keep its graph.
    """
    config = model.config
    n_heads = int(config.num_attention_heads)
    n_kv = int(getattr(config, "num_key_value_heads", n_heads) or n_heads)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // n_heads))
    repeat = max(1, n_heads // n_kv)
    layer_modules = model.model.layers
    device = query_hidden[layers[0]].device
    total = torch.zeros((), device=device, dtype=torch.float32)
    for layer in layers:
        block = layer_modules[layer]
        attention = block.self_attn
        model_dtype = block.input_layernorm.weight.dtype

        def project(memory):
            normed = block.input_layernorm(memory.unsqueeze(0))
            key = F.linear(normed, attention.k_proj.weight)
            value = F.linear(normed, attention.v_proj.weight)
            key = key.view(1, -1, n_kv, head_dim).transpose(1, 2)
            value = value.view(1, -1, n_kv, head_dim).transpose(1, 2)
            if repeat > 1:
                key = key.repeat_interleave(repeat, dim=1)
                value = value.repeat_interleave(repeat, dim=1)
            return key.float(), value.float()

        query = F.linear(query_hidden[layer].to(model_dtype), attention.q_proj.weight)
        query = query.view(1, -1, n_heads, head_dim).transpose(1, 2)[
            :, :, -query_tokens:, :].float()
        scale = head_dim ** -0.5
        with torch.no_grad():
            key_pre, value_pre = project(pre_windows[layer][::slot_stride])
            weights = F.softmax(torch.matmul(query, key_pre.transpose(-1, -2)) * scale,
                                dim=-1)
            readout_pre = torch.matmul(weights, value_pre)
        key_post, value_post = project(post_windows[layer][::slot_stride])
        weights_post = F.softmax(torch.matmul(query, key_post.transpose(-1, -2)) * scale,
                                 dim=-1)
        readout_post = torch.matmul(weights_post, value_post)
        cosine = 1.0 - F.cosine_similarity(readout_post.reshape(1, -1),
                                           readout_pre.reshape(1, -1), dim=-1).mean()
        l2 = F.smooth_l1_loss(readout_post, readout_pre)
        total = total + (cosine + 0.5 * l2)
    return total / len(layers)


def prompt_hidden_states(model, prompt_ids, layers, device=None):
    """Hidden states of the QUESTION only -- no answer tokens ever enter here.

    `tools.objective._query_hidden_states` runs the prompt plus the gold answer
    because it needs the teacher distribution at answer positions. The fusion is
    trained without answers, so the tap is taken on the prompt alone; the last
    eight prompt tokens are the ones the readout loss attends with.
    """
    device = device or model.memory.device
    ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False, return_dict=True,
                    output_hidden_states=True)
    hidden = out.hidden_states
    return {layer: hidden[layer + 1].float() for layer in layers}
