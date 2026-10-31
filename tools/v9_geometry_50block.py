#!/usr/bin/env python3
"""Frozen 50-block geometry for the paper main table.

Import this instead of hard-coding numbers in the trainer, the prepare script and
the runtime, so a single edit moves every stage of the pipeline together.
"""

CONTEXT_LENGTH = 128000          # declared capability of the protocol
TRAIN_STREAM_TOKENS = 64000      # ~8 merge events at trigger 50 / chunk 512
CHUNK_TOKENS = 512
MEMORY_BLOCKS = 50
WINDOW_BLOCKS = 19
TARGET_BLOCKS = 9
TRIGGER_BLOCKS = 50              # merge only once the full pool is occupied

FUSION = dict(architecture="DnativeV9", topk=8, rank=32,
              bridge_fraction=0.156, global_rms_budget=0.08)

GEOMETRY = dict(budget_blocks=MEMORY_BLOCKS, window_blocks=WINDOW_BLOCKS,
                target_blocks=TARGET_BLOCKS, trigger_blocks=TRIGGER_BLOCKS)


def stream_arguments():
    return dict(GEOMETRY)


def apply(cfg):
    cfg.update(context_length=CONTEXT_LENGTH, memory_blocks=MEMORY_BLOCKS,
               window_blocks=WINDOW_BLOCKS, target_blocks=TARGET_BLOCKS,
               trigger_blocks=TRIGGER_BLOCKS, k_max=MEMORY_BLOCKS * 256,
               fusion=dict(FUSION))
    return cfg


def assert_stream_covers_pool(stream_tokens=TRAIN_STREAM_TOKENS):
    need = MEMORY_BLOCKS * CHUNK_TOKENS
    if stream_tokens < need:
        raise ValueError(f"stream {stream_tokens} < {need} tokens: the pool would never "
                         f"fill, so no merge event would be trained")
    merges = max(0, (stream_tokens // CHUNK_TOKENS - MEMORY_BLOCKS)) // (WINDOW_BLOCKS - TARGET_BLOCKS)
    return merges
