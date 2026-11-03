#!/usr/bin/env python3
"""Frozen 50-block geometry and budgets for the PRLM-C main table.

Single source of truth: import this instead of hard-coding numbers in the
trainer, the prepare script, the runtime and the evaluators, so one edit moves
every stage of the pipeline together.
"""

from dataclasses import dataclass, asdict

CONTEXT_LENGTH = 128000          # declared capability of the protocol
TRAIN_STREAM_TOKENS = 64000      # ~8 merge events at trigger 50 / chunk 512
CHUNK_TOKENS = 512
MEMORY_BLOCKS = 50
WINDOW_BLOCKS = 19
TARGET_BLOCKS = 9
TRIGGER_BLOCKS = 50              # merge only once the full pool is occupied

# V10: manifold-preserving slot grafting, no reader adapter.
CORE_SLOTS_PER_BLOCK = 216       # protected native core
GRAFT_SLOTS_PER_BLOCK = 40       # evicted redundant slots, refilled from discarded blocks
ANCHOR_BLOCKS_QUOTA = 6
BRIDGE_BLOCKS_QUOTA = 3
MAX_OLD_GENERATION_BLOCKS = 4
AGE_DECAY = 0.80
INTRA_DISCARD_CLUSTER_COS_THR = 0.78
CROSS_BLOCK_HOMO_COS_THR = 0.85
HOMO_REINFORCE_MAX_ALPHA = 0.15
ORTHO_ADAPTER_RANK = 32
ORTHO_RESIDUAL_MAX_RATIO = 0.05
GLOBAL_RMS_BUDGET = 0.08
DISTILL_LAYERS = (8, 16, 24, 31)   # V10 readout distillation taps

FUSION = dict(architecture="DnativeV9", topk=8, rank=ORTHO_ADAPTER_RANK,
              bridge_fraction=GRAFT_SLOTS_PER_BLOCK / 256,
              global_rms_budget=GLOBAL_RMS_BUDGET,
              core_slots_per_block=CORE_SLOTS_PER_BLOCK,
              graft_slots_per_block=GRAFT_SLOTS_PER_BLOCK)

GEOMETRY = dict(budget_blocks=MEMORY_BLOCKS, window_blocks=WINDOW_BLOCKS,
                target_blocks=TARGET_BLOCKS, trigger_blocks=TRIGGER_BLOCKS)


@dataclass(frozen=True)
class ConsolidationGeometry50B:
    budget_blocks: int = MEMORY_BLOCKS
    window_blocks: int = WINDOW_BLOCKS
    target_blocks: int = TARGET_BLOCKS
    trigger_blocks: int = TRIGGER_BLOCKS
    slots_per_block: int = 256
    num_layers: int = 32
    hidden_size: int = 4096
    reference_layer: int = 16
    select_layer: int = 16
    core_slots_per_block: int = CORE_SLOTS_PER_BLOCK
    graft_slots_per_block: int = GRAFT_SLOTS_PER_BLOCK
    anchor_blocks_quota: int = ANCHOR_BLOCKS_QUOTA
    bridge_blocks_quota: int = BRIDGE_BLOCKS_QUOTA
    max_old_generation_blocks: int = MAX_OLD_GENERATION_BLOCKS
    age_decay: float = AGE_DECAY
    intra_discard_cluster_cos_thr: float = INTRA_DISCARD_CLUSTER_COS_THR
    cross_block_homo_cos_thr: float = CROSS_BLOCK_HOMO_COS_THR
    homo_reinforce_max_alpha: float = HOMO_REINFORCE_MAX_ALPHA
    ortho_adapter_rank: int = ORTHO_ADAPTER_RANK
    ortho_residual_max_ratio: float = ORTHO_RESIDUAL_MAX_RATIO
    global_rms_budget: float = GLOBAL_RMS_BUDGET
    chunk_tokens: int = CHUNK_TOKENS
    tail_tokens: int = 2048
    max_eval_tokens: int = 131072


GEOMETRY_50B = ConsolidationGeometry50B()


def get_canonical_config_dict():
    return asdict(GEOMETRY_50B)


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
