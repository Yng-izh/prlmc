"""Bridge-masked low-rank K/V calibration for the native memory prefix.

Why a hook and not a modeling edit: this repo does not vendor the MemoryLLM
modeling file -- the reader is loaded from a fp16 checkpoint and the memory is
injected by `methods/prlmc/probes.py`, so the attention module reads
`model.memory` itself. The only mask-respecting place for a K/V delta is a
forward pre-hook on `self_attn` that perturbs the *memory rows* of the incoming
hidden states before the native `k_proj`/`v_proj` run: that is exactly
`dK = W_k dh`, `dV = W_v dh` with a rank-8 `dh` on the bridge slots, while the
other memory slots, the whole query region and the lm_head path stay
bit-identical to the frozen reader.

Consequences that matter for the A/B comparison:

* on the Selection arm (no cross-block fusion) the mask is empty, so the hook
  returns `None`, no tensor is built and the forward is the untouched native one
  -- the arm cannot degrade by construction;
* only slots that already accepted a bridge write are touched, i.e. a few hundred
  of the 12,800 pool slots, so the cost stays a few rank-8 projections per layer
  and layers 12-31 (including the layer-16 routing core) fit in the budget.
"""
import hashlib

import torch
from torch import nn


class BridgeKVAdapter(nn.Module):
    """One rank-r correction on the memory rows, gated and zero-initialised once.

    Only `up` is zero-initialised. The gate starts at one, so at step zero the
    output is still zero (identity start) but `dL/d(up) = grad . (down(x) * scale
    * tanh(gate))` is non-zero and the branch can leave the origin. Zeroing both
    the projection *and* the gate would make every parameter gradient zero and
    the adapter could never move, which is why the gate is not zeroed here.
    """

    def __init__(self, hidden, rank=8, lora_alpha=16):
        super().__init__()
        self.down = nn.Linear(hidden, rank, bias=False)
        self.up = nn.Linear(rank, hidden, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)          # single zero init: identity at step 0
        self.scaling = lora_alpha / rank
        self.gate = nn.Parameter(torch.ones(1))
        self.enabled = True

    def delta(self, x):
        return self.up(self.down(x)) * (self.scaling * torch.tanh(self.gate))

    def forward(self, hidden, mask, memory_len):
        """`hidden` [1, T, H]; `mask` bool [memory_len] over pool slots.

        Returns `hidden` itself when nothing applies, so the caller can tell the
        identity case apart by object identity and skip the hook result.
        """
        if not self.enabled or mask is None:
            return hidden
        if hidden.dim() != 3 or not bool(mask.any()):
            return hidden
        # MemoryLLM reuses the cached memory K/V on every incremental step, so the
        # hidden states there are a single query token: nothing to calibrate.
        if hidden.shape[1] <= 1:
            return hidden
        span = min(int(memory_len), hidden.shape[1])
        index = mask.nonzero(as_tuple=False).flatten()
        index = index[index < span]
        if index.numel() == 0:
            return hidden
        rows = hidden.index_select(1, index)
        moved = rows + self.delta(rows.to(self.down.weight.dtype)).to(hidden.dtype)
        return hidden.index_copy(1, index, moved)


class AdapterHandle:
    """Same surface as `reader_lora_train.AdapterHandle` so the trainer is agnostic."""

    spec = "bridge_kv_masked"

    def __init__(self, model, manager, layers, rank=8, lora_alpha=16):
        self.model, self.manager = model, manager
        self.layers = sorted(int(x) for x in layers)
        self.blocks = nn.ModuleDict()
        self.handles = []
        self.stage_active = True
        self.event_bridge = None
        device = model.memory.device
        hidden = int(model.config.hidden_size)
        named = dict(model.named_modules())
        for layer in self.layers:
            target = f"layers.{layer}.self_attn"
            key = next((name for name in named if name.endswith(target)), None)
            if key is None:
                raise KeyError(f"self_attn of layer {layer} not found")
            adapter = BridgeKVAdapter(hidden, rank, lora_alpha)
            self.blocks[f"L{layer}"] = adapter
            # `LlamaDecoderLayer.forward` calls `self.self_attn(hidden_states=...)`
            # by keyword, so a positional-only pre-hook would receive no args.
            self.handles.append(
                named[key].register_forward_pre_hook(self._hook(adapter), with_kwargs=True))
        self.blocks.to(device=device, dtype=torch.float32)   # params live with the reader
        self.names = [f"bridge.{k}.{n}" for k, blk in self.blocks.items()
                      for n, _ in blk.named_parameters()]

    def _hook(self, adapter):
        def hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states")
            if hidden is None and args:
                hidden = args[0]
            if hidden is None:
                return None
            mask = self.mask()
            if mask is None or not bool(mask.any()):
                return None                     # identity: nothing is even built
            memory_len = int(self.model.memory.shape[1])
            moved = adapter.forward(hidden, mask, memory_len)
            if moved is hidden:
                return None
            if "hidden_states" in kwargs:
                return args, {**kwargs, "hidden_states": moved}
            return (moved,) + tuple(args[1:]), kwargs
        return hook

    def mask(self):
        """Bridge slots of the current pool, plus this event's not-yet-installed ones."""
        getter = getattr(self.manager, "fused_slot_mask", None)
        fused = None if getter is None else getter()
        if self.event_bridge is None:
            return fused
        if fused is None:
            out = torch.zeros(self.manager.total, dtype=torch.bool,
                              device=self.manager._memory.device)
        else:
            out = fused.clone()
        out[self.event_bridge] = True
        return out

    def set_event_bridge(self, index):
        self.event_bridge = index

    def parameters(self):
        return [p for blk in self.blocks.values() for p in blk.parameters()]

    def state_dict(self):
        return {f"{k}.{n}": v for k, blk in self.blocks.items()
                for n, v in blk.state_dict().items()}

    def load_state_dict(self, state):
        own = self.state_dict()
        missing, extra = set(own) - set(state), set(state) - set(own)
        if missing or extra:
            raise KeyError("bridge adapter state mismatch: missing=%s extra=%s"
                           % (sorted(missing)[:4], sorted(extra)[:4]))
        with torch.no_grad():
            for key, value in state.items():
                own[key].copy_(value.to(device=own[key].device, dtype=own[key].dtype))
        return sorted(own)

    def digest(self):
        h = hashlib.sha256()
        for k, v in sorted(self.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def enable_grad(self):
        for p in self.parameters():
            p.requires_grad_(True)

    def on(self):
        for blk in self.blocks.values():
            blk.enabled = True

    def off(self):
        for blk in self.blocks.values():
            blk.enabled = False

    def active(self):
        return any(blk.enabled for blk in self.blocks.values())


def attach_bridge_adapter(model, manager, layers=range(12, 32), rank=8, lora_alpha=16):
    return AdapterHandle(model, manager, layers, rank, lora_alpha)
