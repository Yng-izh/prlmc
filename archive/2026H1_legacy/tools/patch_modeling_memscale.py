#!/usr/bin/env python3
"""Readout-probe patch: multiplicative gain on reader attention to memory.

Extends the Phase 2 active-mask patch with `memory_attn_scale`, a scalar
buffer on MemoryLLM. During the READER pass only (never injection), every
ACTIVE pool slot's attention logit gets `+ log(scale)`; inactive slots stay
at `finfo.min`. scale == 1.0 reproduces the Phase 2 path bit-for-bit.

Two changes to the attention-bias plumbing:

1. `_memory_bias_for` builds `log(scale)` into the bias on active slots, and
   stays live even when the active mask is all-true (that branch previously
   returned None, which would silently disable the gain).
2. The SDPA fold becomes two steps: `masked_fill` for blocked keys (compare
   against `finfo.min / 2`, not `< 0`, so a DOWN-weighting `log(scale) < 0`
   is not mistaken for a block), then an additive pass of the remaining bias.
   The additive term is exactly zero on every non-memory key, and on every
   key when scale == 1.0, so the unpatched behaviour is unchanged.

Anchor texts are taken from the Phase 2 patched file (md5
13fc12ae8f1b3a487cb1cd1cfdebf027). Idempotent, with --check / --revert.
"""

import argparse
import hashlib
import os
import shutil
import sys

# --- patch A: register the scale buffer next to the active-mask buffer -------

INIT_OLD = """        self.register_buffer(
            "memory_active_mask",
            torch.ones(self.num_blocks * self.num_tokens, dtype=torch.bool),
            persistent=False,
        )
"""

INIT_NEW = """        self.register_buffer(
            "memory_active_mask",
            torch.ones(self.num_blocks * self.num_tokens, dtype=torch.bool),
            persistent=False,
        )
        # Readout probe (PRLM-C): multiplicative gain on the reader's attention
        # to ACTIVE pool slots. 1.0 keeps the Phase 2 behaviour bit-for-bit.
        self.register_buffer(
            "memory_attn_scale",
            torch.tensor(1.0, dtype=torch.float32),
            persistent=False,
        )
"""

# --- patch B1: keep the bias alive when the mask is all-true -----------------

GATE_OLD = """        memory_active = getattr(self, "memory_active_mask", None)
        if memory_active is not None and bool(memory_active.all()):
            memory_active = None
"""

GATE_NEW = """        memory_active = getattr(self, "memory_active_mask", None)
        memory_scale = float(getattr(self, "memory_attn_scale", 1.0))
        if memory_active is not None and bool(memory_active.all()) and memory_scale == 1.0:
            memory_active = None
"""

# --- patch B2: early-out condition inside _memory_bias_for --------------------

EARLY_OLD = """            if memory_active is None or is_injection:
"""

EARLY_NEW = """            if is_injection or (memory_active is None and memory_scale == 1.0):
"""

# --- patch B3: build log(scale) into the bias on active slots -----------------

BIAS_OLD = """            bias = torch.zeros(key_len, dtype=hidden_states.dtype, device=hidden_states.device)
            bias[memory_offset : memory_offset + memory_total].masked_fill_(
                ~memory_active.to(hidden_states.device),
                torch.finfo(hidden_states.dtype).min,
            )
"""

BIAS_NEW = """            bias = torch.zeros(key_len, dtype=hidden_states.dtype, device=hidden_states.device)
            region = bias[memory_offset : memory_offset + memory_total]
            if memory_active is not None:
                region.masked_fill_(
                    ~memory_active.to(hidden_states.device),
                    torch.finfo(hidden_states.dtype).min,
                )
            if memory_scale != 1.0:
                # Gain on the reader's attention to ACTIVE slots only. Masked
                # cells sit at finfo.min and are never touched here, so the
                # additive term cannot overflow float16.
                gain = math.log(memory_scale)
                if memory_active is not None:
                    region[memory_active.to(hidden_states.device)] += gain
                else:
                    region += gain
"""

# --- patch C: two-step SDPA fold (masked_fill for blocks, add for the gain) ---

FOLD_OLD = """            if bias.shape[-1] == causal_mask.shape[-1]:
                # masked_fill rather than an add: adding two `finfo.min` values
                # overflows to -inf on float16, and a bias of exactly
                # `finfo.min` is what the caller sends for inactive keys.
                causal_mask = causal_mask.masked_fill(
                    bias < 0, torch.finfo(query_states.dtype).min
                )
"""

FOLD_NEW = """            if bias.shape[-1] == causal_mask.shape[-1]:
                # masked_fill rather than an add: adding two `finfo.min` values
                # overflows to -inf on float16, and a bias of exactly
                # `finfo.min` is what the caller sends for inactive keys.
                # The blocked test compares against finfo.min / 2 so that a
                # DOWN-weighting log(scale) < 0 is not mistaken for a block.
                blocked = bias <= torch.finfo(query_states.dtype).min / 2
                causal_mask = causal_mask.masked_fill(
                    blocked, torch.finfo(query_states.dtype).min
                )
                # Readout probe: log(memory_attn_scale) rides on active memory
                # slots as an additive term. Zero on every other key, and on
                # every key when scale == 1.0, so that case is bit-identical
                # to the masked_fill-only path above.
                causal_mask = causal_mask + bias.masked_fill(blocked, 0.0).reshape(1, 1, -1)
"""

PATCHES = [
    ("register memory_attn_scale buffer", INIT_OLD, INIT_NEW),
    ("keep bias alive for all-true mask", GATE_OLD, GATE_NEW),
    ("_memory_bias_for early-out", EARLY_OLD, EARLY_NEW),
    ("_memory_bias_for active-slot gain", BIAS_OLD, BIAS_NEW),
    ("two-step SDPA bias fold", FOLD_OLD, FOLD_NEW),
]

DEFAULT_ROOT = "/share/latent/PRLM-C"
RELATIVE = os.path.join("methods", "memoryllm_modeling", "modeling_memoryllm.py")
BACKUP_SUFFIX = ".bak2"


def _sub_once(text, old, new, name):
    count = text.count(old)
    if count == 0:
        raise SystemExit("anchor not found for patch %r" % name)
    if count > 1:
        raise SystemExit("anchor for patch %r is not unique (%d matches)" % (name, count))
    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT, help="benchmark checkout root")
    parser.add_argument("--check", action="store_true", help="report status without writing")
    parser.add_argument("--revert", action="store_true", help="restore the .bak2 copy")
    args = parser.parse_args()

    path = os.path.join(args.root, RELATIVE)
    backup = path + BACKUP_SUFFIX
    if not os.path.exists(path):
        raise SystemExit("missing modeling file: %s" % path)

    if args.revert:
        if not os.path.exists(backup):
            raise SystemExit("no backup to restore at %s" % backup)
        shutil.copyfile(backup, path)
        print("reverted %s from %s" % (path, backup))
        return 0

    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    applied = []
    missing = []
    for name, old, new in PATCHES:
        if new in text:
            applied.append(name)
            continue
        if text.count(old) == 1:
            missing.append((name, old, new))
        else:
            raise SystemExit(
                "patch %r: anchor found %d times, and the patched form is absent"
                % (name, text.count(old))
            )

    if args.check:
        print("file      : %s" % path)
        print("md5       : %s" % hashlib.md5(text.encode("utf-8")).hexdigest())
        print("patched   : %d/%d" % (len(applied), len(PATCHES)))
        for name in applied:
            print("  [x] %s" % name)
        for name, _, _ in missing:
            print("  [ ] %s" % name)
        return 0

    if not missing:
        print("already patched: %s" % path)
        return 0

    if not os.path.exists(backup):
        shutil.copyfile(path, backup)
        print("backup    : %s" % backup)

    for name, old, new in missing:
        text = _sub_once(text, old, new, name)
        print("  [x] %s" % name)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    print("md5       : %s" % hashlib.md5(text.encode("utf-8")).hexdigest())
    print("patched   : %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
