#!/usr/bin/env python3
"""Streaming memory runtime: append writes, compress at the trigger, answer.

One write appends one native block; when the resident width reaches the trigger
the oldest `window` blocks are compressed to `target` blocks and the newest
native blocks are left untouched. The arm decides what the compressed window
becomes: `Selection` keeps native carrier blocks, `SharedE6d`/`SharedMerge`
runs the trainable operator, `Recent` keeps the newest blocks, `Full` and
`NativeDefault` are reference arms that never compress.

The reader is frozen. Nothing here mutates reader weights; the only trainable
parameters are the shared merge operator's.
"""
from contextlib import contextmanager
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

E6D_SHA = "a9fe2ea3f7a4e2c21590cf2219591edb11fc65297ce36ddeed96c236d8020c5d"
ARMS = ("Full", "Selection", "Recent", "SharedE6d", "NativeDefault")


@contextmanager
def isolated_query(model):
    """Queries cannot become writes, including when generation raises."""
    import torch
    memory = model.memory
    snapshot = memory.detach().cpu().clone()
    initialized = model.initialized.clone()
    mask = model._buffers.get("memory_active_mask")
    mask_value = None if mask is None else mask.clone()
    modes = [(m, m.training) for m in model.modules()]
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    device = memory.device
    try:
        model.eval()
        yield
    finally:
        # Restore parameter binding, storage, values, mask and RNG even if the
        # native generation path replaces memory.data or the parameter itself.
        if model.memory is not memory:
            model.memory = memory
        with torch.no_grad():
            if memory.shape == snapshot.shape and memory.device == device:
                memory.copy_(snapshot)
            else:
                memory.data = snapshot.to(device)
            model.initialized.copy_(initialized)
        if mask is None:
            model._buffers.pop("memory_active_mask", None)
        else:
            model._buffers["memory_active_mask"] = mask
            mask.copy_(mask_value)
        for module, mode in modes:
            module.training = mode
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def author_score(prediction, target):
    target = target.replace("</s>", "").strip()
    if not target:
        raise ValueError("empty scoring target")
    return target in prediction


def generate(model, tokenizer, sample, target):
    """Greedy 10-token readout of one prepared sample; the query is isolated."""
    import torch
    ids = torch.tensor([sample["prompt_ids"]], device=model.memory.device, dtype=torch.long)
    with isolated_query(model), torch.inference_mode():
        # Exact author generation length and no newline stop. The prefix width
        # is physical and fixed; logical masking is applied by the reader fork.
        attention = torch.ones((1, model.memory.shape[1] + ids.shape[1]),
                               dtype=torch.long, device=ids.device)
        output = model.generate(input_ids=ids, attention_mask=attention, max_new_tokens=10,
                                do_sample=False, use_cache=True,
                                pad_token_id=tokenizer.pad_token_id)
        tokens = output[0, ids.shape[1]:].detach().cpu().tolist()
        prediction = tokenizer.decode(tokens, skip_special_tokens=False)
    return dict(prediction=prediction, token_ids=tokens, score=author_score(prediction, target))


def budget_geometry(budget_blocks):
    """Pre-registered (window_blocks, target_blocks) per managed budget."""
    table = {8: (9, 4), 10: (11, 5), 12: (11, 5), 20: (19, 9), 50: (19, 9)}
    if budget_blocks not in table:
        raise ValueError("budget_blocks must be one of 8/10/12/20/50")
    return table[budget_blocks]


def compaction_trigger(budget_blocks, window_blocks):
    """Resident block count at which the managed stream compacts.

    The rule is `used >= min(budget, window + 1)`: the window has to be fully
    resident before it can be compressed, and a stream whose budget is smaller
    than its window has to be allowed to reach the window (transient overshoot
    of one block, then back to target). A budget with room beyond the window
    fills to the budget first, so the reader actually carries what the budget
    declares instead of stopping at window+1.
    """
    if int(budget_blocks) > int(window_blocks) + 1:
        return int(budget_blocks)
    return min(int(budget_blocks), int(window_blocks) + 1)


class StreamingMemory:
    """Explicit preappend 19→9 compression; no answer-derived goals/anchors."""
    def __init__(self, manager, compactor, arm, budget_blocks=20, window_blocks=None,
                 target_blocks=None, strict_budget=False, trigger_blocks=None):
        if arm not in ARMS:
            raise ValueError("unknown arm")
        self.m = manager
        self.compactor = compactor
        self.arm = arm
        if budget_blocks not in (8, 10, 12, 20, 50):
            raise ValueError("budget_blocks must be one of 8/10/12/20/50")
        from runtime import resolve_geometry
        self.budget_blocks = int(budget_blocks)
        # The pre-registered compression geometry is a fixed table, not a
        # formula; an explicit pair overrides it and a conflict raises instead of
        # being silently corrected.
        self.window_blocks, self.target_blocks = resolve_geometry(
            budget_blocks, window_blocks, target_blocks,
            total_blocks=manager.total // manager.block,
        )
        # Declared vs. actually allowed capacity. For the 8/10-block geometries
        # the pre-registered window (9/11) is one block larger than the budget,
        # so the stream may transiently hold 9/11 blocks before compacting back
        # to the target. That is a real capacity advantage against any method
        # held at 8/10, so it is reported, never absorbed into a budget label.
        self.trigger_blocks = None if trigger_blocks is None else int(trigger_blocks)
        if self.trigger_blocks is not None:
            if self.trigger_blocks < self.window_blocks:
                raise ValueError(
                    f"trigger_blocks {self.trigger_blocks} is smaller than the "
                    f"{self.window_blocks}-block compression window; the window "
                    "would not be fully resident when compaction fires")
            if self.trigger_blocks > self.budget_blocks:
                raise ValueError(
                    f"trigger_blocks {self.trigger_blocks} exceeds the declared "
                    f"budget {self.budget_blocks}; the stream would hold more "
                    "blocks than it reports")
        self.effective_trigger_blocks = (self.trigger_blocks if self.trigger_blocks is not None
                                         else compaction_trigger(self.budget_blocks,
                                                                 self.window_blocks))
        self.peak_allowed_blocks = max(self.budget_blocks, self.window_blocks,
                                       self.effective_trigger_blocks)
        self.capacity_overshoot_blocks = self.peak_allowed_blocks - self.budget_blocks
        if strict_budget and self.capacity_overshoot_blocks:
            raise ValueError(
                f"strict budget {self.budget_blocks} violated by the compression "
                f"window {self.window_blocks}; pass explicit window_blocks <= "
                "budget_blocks (and target_blocks < window_blocks) to stay inside "
                "the declared capacity")
        self.budget = self.budget_blocks * manager.block
        self.window = self.window_blocks * manager.block
        self.target = self.target_blocks * manager.block
        self.events = []
        self.peak = 0

    def compact(self, write_index):
        """Reference-arm compression: native carrier selection or recent blocks."""
        import torch
        from methods.prlmc.consolidate import select_block_compact, recent_block_compact, _select_block_indices
        m = self.m
        start = m.total - m.used
        used_before = int(m.used)
        source = m._memory[:, start:start + self.window].contiguous()
        weights = m._window_importance(start, self.window)
        ages = m.window_block_ages(start, self.window)
        kwargs = dict(target=self.target, block=m.block, weights=weights,
                      goal_direction=None, select_weights=m.select_weights,
                      block_ages=ages)
        begin = time.monotonic()
        if self.arm == "SharedE6d":
            output = self.compactor(source, **kwargs).to(m.dtype)
            selected = _select_block_indices(source, self.target, m.block, weights, None,
                                             m.select_weights, block_ages=ages)
        elif self.arm == "Selection":
            output = select_block_compact(source, **kwargs).to(m.dtype)
            selected = _select_block_indices(source, self.target, m.block, weights, None,
                                             m.select_weights, block_ages=ages)
        elif self.arm == "Recent":
            output = recent_block_compact(source, target=self.target, block=m.block).to(m.dtype)
            selected = torch.arange(self.window // m.block - self.target // m.block, self.window // m.block)
        else:
            raise ValueError("reference arm cannot use managed compression")
        if output.shape != (m.num_layers, self.target, m.hidden) or not torch.isfinite(output).all():
            raise ValueError("invalid compactor output; no eviction fallback")
        # Keep the newest native block unmodified so the next injection has the
        # same predecessor position. No fact is placed in a protected anchor.
        m.used = self.target + (used_before - self.window)
        head = m.total - m.used
        m._memory[:, head:head + self.target].copy_(output)
        m.note_merge(selected, self.window // m.block,
                     target_blocks=self.target // m.block, used_before=used_before)
        m._tile_free_region()
        self.events.append(dict(before_write=write_index, source_slots=self.window, target_slots=self.target,
            remaining_slots=m.used, selected_blocks=selected.cpu().tolist(),
            selected_block_ages=ages[selected].cpu().tolist(),
            seconds=time.monotonic() - begin))

    def write(self, record, index):
        import torch
        m = self.m
        with torch.no_grad():
            if self.arm not in ("Full", "NativeDefault") \
                    and m.used >= self.window \
                    and m.used >= self.effective_trigger_blocks * m.block:
                self.compact(index)
            if self.arm == "Full" and m.used + m.block > m.total:
                return None  # Honest Full-capacity failure; never truncate the stream.
            ids = torch.tensor([record["token_ids"]], dtype=torch.long, device=m._memory.device)
            attention = torch.ones((1, m.block + ids.shape[1]), dtype=torch.long, device=ids.device)
            delta = m.model.inject_memory(ids, context_attention_mask=attention,
                                          update_memory=self.arm == "NativeDefault")
            if self.arm == "NativeDefault":
                # Published native random-slot eviction operates on the entire
                # checkpoint pool; report separately from same-budget arms.
                m.used = m.total
                m.model.memory_active_mask.fill_(True)
            else:
                delta = delta.detach()[0]
                if delta.shape != (m.num_layers, m.block, m.hidden):
                    raise ValueError("unexpected injection shape; refuse dropping/tiling original writes")
                if m.used:
                    old_start = m.total - m.used
                    # On-device block shift: ascending order is safe because every
                    # destination sits exactly one block below its source.
                    for b in range(m.used // m.block):
                        src = old_start + b * m.block
                        m._memory[:, src - m.block:src].copy_(m._memory[:, src:src + m.block])
                m._memory[:, -m.block:].copy_(delta)
                m.used += m.block
                m.note_append(appended=1)
                m._tile_free_region()
            active = int(m.model.memory_active_mask.sum())
            # The allowed ceiling is the declared peak, not the declared budget:
            # for the 8/10 geometries the window is one block larger and the
            # overshoot is reported, not hidden.
            limit = m.total if self.arm in ("Full", "NativeDefault") \
                else self.peak_allowed_blocks * m.block
            if active != m.used or active > limit:
                raise ValueError("active budget violation")
            self.peak = max(self.peak, active)
        return dict(index=index, role=record["role"], input_tokens=len(record["token_ids"]),
            below_author_16_token_training_floor=len(record["token_ids"]) < 16,
            input_sha256=record["token_sha256"], active_slots=active,
            compression_count=len(self.events), native_random_evicted_slots=m.block if self.arm == "NativeDefault" else 0)
