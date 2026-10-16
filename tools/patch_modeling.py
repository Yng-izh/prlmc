#!/usr/bin/env python3
"""Phase 2 patch: let the reader skip the free region of the memory pool.

Why a patch script rather than an edit
--------------------------------------
`methods/memoryllm_modeling/modeling_memoryllm.py` is a verbatim fork of the
upstream MemoryAgentBench file. It is not tracked in this repo, so the edit is
expressed here as an idempotent, reversible set of exact-string substitutions
against the 1993-line original (md5 0d8dbe8eef709b29fda03b5bec75f995).

What the patch does
-------------------
The pool is a fixed-width tensor and stays that way: only the attention bias
changes. `MemoryLLM.memory_active_mask` is a bool buffer of length
`num_blocks * num_tokens`; `MemoryLLM.forward` turns it into a float additive
bias of shape `[key_len]` -- 0.0 for active keys, `finfo(dtype).min` for
inactive ones -- and threads it to every attention layer as
`memory_attention_bias`.

Three details that are easy to get wrong:

1. The shipped SDPA block at the `prefix_token_length > 0` branch rebuilds
   `causal_mask` as a fresh *bool* tensor, discarding anything passed through
   `attention_mask`. So the bias cannot be added upstream in
   `_update_causal_mask`; it has to be folded in at that rebuild. With no bias
   present the original bool path is reproduced byte-for-byte, which keeps
   E5a/E5b bit-identical after patching and isolates the mask as the only
   difference in E5c.
2. Applying the bias only at the prefill step is not enough. Memory keys live
   in the KV cache during `generate`, so decode steps attend to them with
   `prefix_token_length == 0` and no rebuilt mask. The bias therefore covers
   `past_seen + q` keys and is applied on the decode path too.
3. The memory region sits at a fixed offset in the key sequence --
   `int(add_bos_embedding)` -- because `cat_memory_and_hiddens` prepends the
   BOS embedding. The bias is written at that offset, not left-aligned.
"""

import argparse
import hashlib
import os
import shutil
import sys

# --- patch 1: register the mask buffer -------------------------------------

INIT_OLD = """        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
"""

INIT_NEW = """        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        # Phase 2 (PRLM-C): which pool slots the reader is allowed to attend to.
        # All-true by default, so an unpatched -- or unmanaged -- run reads the
        # pool exactly as before. `memory_active_mask` is non-persistent: it is
        # a run-time view of the manager's layout, not part of the checkpoint.
        self.register_buffer(
            "memory_active_mask",
            torch.ones(self.num_blocks * self.num_tokens, dtype=torch.bool),
            persistent=False,
        )
"""

# --- patch 2: SDPA signature ------------------------------------------------

SDPA_SIG_OLD = """        cache_position: Optional[torch.LongTensor] = None,
        prefix_token_length: Optional[int] = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:"""

SDPA_SIG_NEW = """        cache_position: Optional[torch.LongTensor] = None,
        prefix_token_length: Optional[int] = 0,
        memory_attention_bias: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:"""

# --- patch 3: build the mask as float and fold in the bias ------------------

SDPA_MASK_OLD = """        if prefix_token_length > 0:
            causal_mask = torch.cat(
                [
                    torch.ones(q_len - prefix_token_length, prefix_token_length, dtype=torch.bool),
                    torch.ones(q_len - prefix_token_length, q_len - prefix_token_length, dtype=torch.bool).tril(diagonal=0),
                ], dim=1
            ).to(query_states.device)
"""

SDPA_MASK_NEW = """        if prefix_token_length > 0:
            if memory_attention_bias is None:
                # Shipped path, reproduced byte-for-byte so a run without the
                # Phase 2 mask is unaffected by this patch.
                causal_mask = torch.cat(
                    [
                        torch.ones(q_len - prefix_token_length, prefix_token_length, dtype=torch.bool),
                        torch.ones(q_len - prefix_token_length, q_len - prefix_token_length, dtype=torch.bool).tril(diagonal=0),
                    ], dim=1
                ).to(query_states.device)
            else:
                # Rebuilt as a float mask because the memory-key bias has to be
                # folded into the same tensor. The shipped bool version silently
                # dropped any additive mask, which is why the free region could
                # not be excluded from here before.
                masked_q = q_len - prefix_token_length
                causal_mask = torch.zeros(
                    (masked_q, q_len), dtype=query_states.dtype, device=query_states.device
                )
                if masked_q > 1:
                    causal_mask[:, prefix_token_length:] = torch.triu(
                        torch.full(
                            (masked_q, masked_q),
                            torch.finfo(query_states.dtype).min,
                            dtype=query_states.dtype,
                            device=query_states.device,
                        ),
                        diagonal=1,
                    )
        elif memory_attention_bias is not None:
            # Decode step: the pool is already in the KV cache, so the bias has
            # to cover the cached keys as well as this query token.
            causal_mask = torch.zeros(
                (q_len, key_states.shape[-2]), dtype=query_states.dtype, device=query_states.device
            )

        if memory_attention_bias is not None and causal_mask is not None:
            bias = memory_attention_bias.to(
                device=query_states.device, dtype=query_states.dtype
            ).reshape(1, -1)
            if bias.shape[-1] == causal_mask.shape[-1]:
                # masked_fill rather than an add: adding two `finfo.min` values
                # overflows to -inf on float16, and a bias of exactly
                # `finfo.min` is what the caller sends for inactive keys.
                causal_mask = causal_mask.masked_fill(
                    bias < 0, torch.finfo(query_states.dtype).min
                )
"""

# --- patch 4: decoder layer signature ---------------------------------------

DEC_SIG_OLD = """        cache_position: Optional[torch.LongTensor] = None,
        prefix_token_length: Optional[int] = 0,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:"""

DEC_SIG_NEW = """        cache_position: Optional[torch.LongTensor] = None,
        prefix_token_length: Optional[int] = 0,
        memory_attention_bias: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:"""

# --- patch 5: decoder layer self-attention call ------------------------------

DEC_CALL_OLD = """            cache_position=cache_position,
            prefix_token_length=prefix_token_length,
            **kwargs,
        )"""

DEC_CALL_NEW = """            cache_position=cache_position,
            prefix_token_length=prefix_token_length,
            memory_attention_bias=memory_attention_bias,
            **kwargs,
        )"""

# --- patch 6: build the bias once per forward --------------------------------

LOOP_OLD = """        for idx, decoder_layer in enumerate(self.model.layers):
"""

LOOP_NEW = """        # Phase 2 (PRLM-C): the free region of the pool is left in the tensor
        # but removed from the reader's attention. `memory_active_mask` marks
        # the active slots; the bias puts `finfo.min` on the rest. `None`
        # whenever every slot is active, so the unmanaged path never allocates
        # it and stays bit-identical to the unpatched model.
        memory_active = getattr(self, "memory_active_mask", None)
        if memory_active is not None and bool(memory_active.all()):
            memory_active = None
        memory_attention_bias = None
        _memory_bias_cache = {}

        def _memory_bias_for(prefix_len):
            # Built here rather than before the loop because `prefix_token_length`
            # is only known inside it, and cached because every one of the 32
            # layers in a forward pass needs the same tensor. The pool sits at a
            # fixed offset (`int(add_bos_embedding)`) because
            # `cat_memory_and_hiddens` prepends the BOS embedding.
            if memory_active is None or is_injection:
                # Injection reads only the trailing block, not the whole pool,
                # so the pool offset does not describe those keys.
                return None
            memory_total = int(memory_active.numel())
            memory_offset = int(self.add_bos_embedding)
            if prefix_len > 0:
                key_len = hidden_states.shape[1]
            else:
                past_seen = (
                    past_key_values.get_seq_length() if past_key_values is not None else 0
                )
                # Decode steps attend to the pool through the KV cache, so the
                # bias has to cover the cached keys as well as this query token.
                key_len = past_seen + hidden_states.shape[1]
            if key_len < memory_offset + memory_total:
                # The pool is not in this sequence yet: the cold pre-`initialized`
                # forward and the short first write have no memory prefix.
                return None
            cached = _memory_bias_cache.get((prefix_len, key_len))
            if cached is not None:
                return cached
            bias = torch.zeros(key_len, dtype=hidden_states.dtype, device=hidden_states.device)
            bias[memory_offset : memory_offset + memory_total].masked_fill_(
                ~memory_active.to(hidden_states.device),
                torch.finfo(hidden_states.dtype).min,
            )
            _memory_bias_cache[(prefix_len, key_len)] = bias
            return bias

        for idx, decoder_layer in enumerate(self.model.layers):
"""

# --- patch 6b: attach the bias once the prefix length is known --------------

PREFIX_OLD = """            else:
                prefix_token_length = 0

            if self.model.gradient_checkpointing and self.model.training:
"""

PREFIX_NEW = """            else:
                prefix_token_length = 0

            memory_attention_bias = _memory_bias_for(prefix_token_length)

            if self.model.gradient_checkpointing and self.model.training:
"""

# --- patch 8: gradient-checkpointing positional call -------------------------

CKPT_OLD = """                    cache_position,
                    prefix_token_length
                )"""

CKPT_NEW = """                    cache_position,
                    prefix_token_length,
                    memory_attention_bias
                )"""

# --- patch 9: keyword decoder-layer call -------------------------------------

KW_OLD = """                    cache_position=cache_position,
                    prefix_token_length=prefix_token_length,
                )"""

KW_NEW = """                    cache_position=cache_position,
                    prefix_token_length=prefix_token_length,
                    memory_attention_bias=memory_attention_bias,
                )"""

PATCHES = [
    ("register memory_active_mask buffer", INIT_OLD, INIT_NEW),
    ("LlamaSdpaAttention.forward signature", SDPA_SIG_OLD, SDPA_SIG_NEW),
    ("LlamaSdpaAttention causal-mask rebuild", SDPA_MASK_OLD, SDPA_MASK_NEW),
    ("LlamaDecoderLayer.forward signature", DEC_SIG_OLD, DEC_SIG_NEW),
    ("LlamaDecoderLayer self_attn call", DEC_CALL_OLD, DEC_CALL_NEW),
    ("MemoryLLM.forward memory bias helper", LOOP_OLD, LOOP_NEW),
    ("MemoryLLM.forward bias attachment", PREFIX_OLD, PREFIX_NEW),
    ("MemoryLLM.forward checkpointing call", CKPT_OLD, CKPT_NEW),
    ("MemoryLLM.forward decoder call", KW_OLD, KW_NEW),
]

DEFAULT_ROOT = "/share/latent/PRLM-C"
RELATIVE = os.path.join("methods", "memoryllm_modeling", "modeling_memoryllm.py")


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
    parser.add_argument("--revert", action="store_true", help="restore the .bak copy")
    args = parser.parse_args()

    path = os.path.join(args.root, RELATIVE)
    backup = path + ".bak"
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
