"""Memory-pool injection helpers shared by the training objective.

`_pool_in_place` writes a candidate merge result into the frozen reader's memory
pool while keeping the graph alive, `differentiable_memory` routes the reader's
reads to that pool, `active_mask_override` pins the logical slot budget for the
duration of a read, and `_compacted_mask`/`_compacted_pool` describe what the
merge left behind. Nothing here mutates reader weights.
"""
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


def _mem(label, device):
    """Phase-boundary memory trace, enabled by PRLMC_MEM_TRACE=1.

    Kept because the E6a training step failed with ~31 GiB already live *before*
    the first student forward, which no amount of checkpointing can fix: the leak
    is upstream of the optimizer step and has to be located by measurement.
    """
    if not os.environ.get("PRLMC_MEM_TRACE"):
        return
    if not torch.cuda.is_available():
        return
    print(
        "[mem] %-26s alloc=%.2fGB reserved=%.2fGB"
        % (
            label,
            torch.cuda.memory_allocated(device) / 1e9,
            torch.cuda.memory_reserved(device) / 1e9,
        ),
        flush=True,
    )


def _checkpoint_func():
    """Checkpoint function for the fork's per-layer branch.

    `transformers` normally installs `self._gradient_checkpointing_func` on the
    module it is called on. `gradient_checkpointing_enable()` is invoked on
    `MemoryLLM`, but the fork's decoder loop reads it from the inner `LlamaModel`
    (`self.model._gradient_checkpointing_func`). `use_reentrant=False` is required
    because the loop passes the layer's arguments positionally and the reentrant
    implementation would warn/fail on the non-tensor `prefix_token_length` and
    `memory_attention_bias` values.
    """
    import torch.utils.checkpoint as checkpoint

    def func(function, *args, **kwargs):
        return checkpoint.checkpoint(
            function, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
        )

    return func


@contextmanager
def differentiable_memory(model, memory: torch.Tensor):
    """Present `memory` to the model's forward pass while keeping autograd.

    Assigning through `.data` would sever the graph, which is what makes naively
    "writing" a trainable candidate a silent no-op: the functional loss would be
    detached and the compressor would receive no gradient at all. So the
    parameter is temporarily moved out of `_parameters` and the trainable tensor
    takes its place, letting `cat_memory_and_hiddens` read it as usual. The
    original parameter object is restored on exit, including on exception.
    """
    parameters = model._parameters
    had_parameter = "memory" in parameters
    original = parameters.pop("memory", None)
    object.__setattr__(model, "memory", memory)
    try:
        yield
    finally:
        model.__dict__.pop("memory", None)
        if had_parameter:
            parameters["memory"] = original


@contextmanager
def active_mask_override(model, mask):
    """Temporarily publish `mask` as the reader's active-slot mask.

    `_compacted_pool` rearranges the memory tensor, but the model also holds a
    `memory_active_mask` buffer that decides which pool slots the reader may
    attend to. If that buffer is left describing the *pre-compaction* layout, the
    freed slots stay active and still contain the blocks the compaction just
    dropped, so the student would read the carriers and the originals side by
    side and the probe would score the uncompacted memory. Syncing the mask is
    what makes the measurement a measurement of the compaction.
    """
    if mask is None:
        yield
        return
    buffers = model._buffers
    had = "memory_active_mask" in buffers
    original = buffers.get("memory_active_mask")
    buffers["memory_active_mask"] = mask
    try:
        yield
    finally:
        if had:
            buffers["memory_active_mask"] = original
        else:
            buffers.pop("memory_active_mask", None)


_READER_HIDDEN_LAYERS = (8, 16, 24, 31)


_CARRIER_CURRICULUM = (19, 15, 19, 12, 19, 9)


def _compacted_mask(manager, new_used, device):
    """Active-slot mask for a pool whose distinct suffix is `new_used` wide.

    Returns None when the run does not use the active mask, so a config with
    `active_memory_mask: false` keeps reading the pool exactly as before.
    """
    if not getattr(manager, "active_memory_mask", False):
        return None
    total = manager.layout.total
    k_anchor = manager.layout.k_anchor
    mask = torch.zeros(total, dtype=torch.bool, device=device)
    mask[:k_anchor] = True
    if new_used > 0:
        mask[total - new_used :] = True
    return mask


def _compacted_pool(manager, device, candidate, window_used, target):
    """The reader's pool after `window_used` slots are compacted into `target`.

    Reproduces `compact_with_report`'s placement exactly (layout.py:1105-1120),
    and it must be a *rearrangement*, never a shortening.

    Two properties are load-bearing:

    1. The pool keeps its config width (`num_tokens * num_blocks`). This fork
       derives `cache_position` from the config constants, so a shortened pool
       makes `rotary_emb` emit cos/sin for 12801 positions against a shorter q and
       `apply_rotary_pos_emb` raises `The size of tensor a (12) must match the
       size of tensor b (10252)`.
    2. Carriers land flush against the trailing end, with the protected newest
       block immediately after them. Substituting in place at `start` would leave
       the freed slots inside the distinct suffix, so the reader would attend to
       the carriers *and* the blocks they replaced.

    The freed slots at the front are the base pool's existing filler, which the
    active mask removes from attention. They are not re-tiled here: the filler's
    only role is to avoid reading uninitialised rows when the mask is off, and the
    base already provides that.
    """
    base = manager._memory.to(device).detach()
    total = manager.total
    used = manager.used
    new_used = used - window_used + target
    new_start = total - new_used
    remainder = base[:, total - (used - window_used) :, :]
    pool = torch.cat([base[:, :new_start, :], candidate.to(base.dtype), remainder], dim=1)
    if pool.shape[1] != total:
        raise ValueError(
            "_compacted_pool width %d != pool width %d" % (pool.shape[1], total)
        )
    return pool


class _InjectSlice(torch.autograd.Function):
    """Write `candidate` into `pool[:, start:start+target, :]` in place, differentiably.

    A plain `pool[...] = candidate` cannot be used here. `pool` is the frozen 8B
    model's `memory` parameter and does **not** require grad, and PyTorch only
    records an in-place assignment when it lands on a tensor that already carries
    a graph. Assigning into a non-requiring tensor silently drops the graph, so
    `candidate` would receive no gradient at all and the compactor would never
    train. (Verified on the host: `torch.autograd.grad(out, candidate)` raises
    "One of the differentiated Tensors does not require grad".)

    Doing the write inside a custom `Function` restores it: the forward performs
    the in-place write under no-grad, and the backward routes the output gradient
    for exactly the written slice back to `candidate`. The returned tensor is the
    same object as `pool`, so no second pool is allocated — the entire point,
    since `_compacted_pool`'s `torch.cat` would allocate another 3.36 GB here.

    The dtype cast lives inside `forward` because ops inside a custom `Function`
    are not tracked; the fp16 storage therefore stays fp16 while `candidate` keeps
    its fp32 gradient.
    """

    @staticmethod
    def forward(ctx, pool, candidate, start, target):
        ctx.start = int(start)
        ctx.target = int(target)
        pool[:, ctx.start : ctx.start + ctx.target, :] = candidate.to(pool.dtype)
        return pool

    @staticmethod
    def backward(ctx, grad_output):
        start, target = ctx.start, ctx.target
        return (
            None,
            grad_output[:, start : start + target, :].clone(),
            None,
            None,
        )


class _PoolInjection:
    """Live in-place injection of a compacted candidate into `model.memory`.

    Writes the candidate at the inference placement and restores the displaced
    slots on exit. `reinject()` re-runs the write and returns a *fresh* pool
    tensor carrying its own `_InjectSlice` node.

    The fresh node per probe is the point. One shared node would force every
    probe's backward to pass `retain_graph=True`, which keeps all reader graphs
    alive at once and defeats per-layer checkpointing. A fresh node lets each
    probe free its graph immediately, while gradients accumulate on the candidate
    leaf — which is where they are summed anyway.
    """

    def __init__(self, base, candidate, new_start, target):
        self.base = base
        self.candidate = candidate
        self.new_start = new_start
        self.target = target
        # The displaced slots are read only by `restore()`, at the very end of
        # the step, and they never take part in autograd. Keeping the clone on
        # the device therefore holds 604 MB live across the whole reader
        # backward for nothing -- and the full-width memory gradient that
        # backward has to materialise is 3.12 GiB, which is exactly where the
        # 32 GB V100 ran out (needed 3.12 GiB, had 3.09 GiB free). Offloading
        # costs one 604 MB copy per step and returns that headroom at the peak.
        saved = base[:, new_start : new_start + target, :].clone()
        self.saved = saved.cpu() if saved.is_cuda else saved

    def reinject(self):
        return _InjectSlice.apply(
            self.base, self.candidate, self.new_start, self.target
        )

    def restore(self):
        with torch.no_grad():
            self.base[
                :, self.new_start : self.new_start + self.target, :
            ] = self.saved.to(self.base.device)


@contextmanager
def _pool_in_place(manager, device, candidate, window_used, target):
    """Rearrange the window in place, yielding a live `_PoolInjection`.

    Same layout as `_compacted_pool`, but without the `torch.cat`. That `cat`
    allocates a second full-width pool: 3.36 GB at the E6a shape
    (`32 x 12800 x 4096` fp16), which is most of why the student forward OOM'd.
    Only `target` slots actually change, so the copy is almost entirely waste.

    The layout is contiguous by construction (layout.py:1105-1120):

        [ filler ... | candidate (target) | remainder (used - window) ]

    with `new_start = total - (used - window + target)`. The remainder's source
    range *is* its destination range, so it is already in place and needs no
    copy. Only `[new_start, new_start + target)` has to be written, via
    `_InjectSlice` so the graph survives.

    The displaced slots are saved and restored on exit, including on exception,
    so `model.memory` is unchanged after training.
    """
    # Deliberately the `nn.Parameter` rather than `manager._memory` (which
    # returns `.data`): `.data` bypasses autograd bookkeeping.
    base = manager.model.memory
    total = manager.total
    used = manager.used
    new_used = used - window_used + target
    new_start = total - new_used
    injection = _PoolInjection(base, candidate, new_start, target)
    try:
        yield injection
    finally:
        injection.restore()


def compressed_pool(
    manager,
    compactor,
    window_blocks: int,
    target_blocks: int,
    device=None,
) -> torch.Tensor:
    """The pool that compacting `window_blocks` into `target_blocks` actually yields.

    Deliberately mirrors the training loop's substitution rather than calling
    `compact_with_report`: the probe has to score the same object training
    optimises, and going through `manager.compact` would take a different path
    (block alignment, `newest_width` protection, the `native_filler_only`
    filler rewrite) and could differ from the trained operator's input.
    """
    device = manager.model.memory.device if device is None else device
    start = manager.total - manager.used
    window = int(window_blocks) * manager.block
    target = int(target_blocks) * manager.block
    if window > manager.used:
        # `_compacted_pool` derives `new_start` from `used - window + target`, so a
        # window wider than the pool makes that arithmetic negative and the
        # placement silently produces an over-wide pool. The failure it used to
        # raise ("width 13056 != 12800") pointed at the placement rather than at
        # the real cause, which is always a window the caller never wrote.
        raise ValueError(
            "compressed_pool: window %d slots exceeds the %d slots written"
            % (window, manager.used)
        )
    source = manager._memory[:, start : start + window, :].to(device)
    # The importance weights are not optional here. `_select_block_indices` scores
    # importance at weight 0.4, and without them it falls back to the
    # dispersion-based stand-in -- a *different* set of carriers than
    # `compact_with_report` picks for the same source. A screen that derives the
    # dropped blocks from the weighted scorer and then measures the unweighted one
    # is comparing two different operators, and every probe whose fact sat in a
    # dropped block would still be readable in the scored pool.
    weights = manager._window_importance(start, window)
    candidate = compactor(
        source,
        weights,
        target=target,
        block=manager.block,
        goal_direction=manager.goal_direction,
        select_weights=manager.select_weights,
    )
    pool = _compacted_pool(manager, device, candidate, window, target)
    mask = _compacted_mask(manager, manager.used - window + target, device)
    return pool, mask
