"""Single Source of Truth for 50-block PRLM-C Consolidation Geometry and Budgets (V10)."""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ConsolidationGeometry50B:
    budget_blocks: int = 50
    window_blocks: int = 19
    target_blocks: int = 9
    slots_per_block: int = 256
    num_layers: int = 32
    hidden_size: int = 4096
    reference_layer: int = 16
    select_layer: int = 16
    # 块内分区：每块 216 个原生核心保护槽位 + 40 个可腾退嫁接槽位 (15.625%)
    core_slots_per_block: int = 216
    graft_slots_per_block: int = 40
    # 选块器配额：9 个目标块 = 6 个高显著主锚点块 + 3 个二跳互补桥接块
    anchor_blocks_quota: int = 6
    bridge_blocks_quota: int = 3
    max_old_generation_blocks: int = 4
    age_decay: float = 0.80
    # 流形保范凝聚与正交残差安全边界
    intra_discard_cluster_cos_thr: float = 0.78
    cross_block_homo_cos_thr: float = 0.85
    homo_reinforce_max_alpha: float = 0.15
    ortho_adapter_rank: int = 32
    ortho_residual_max_ratio: float = 0.05
    global_rms_budget: float = 0.08
    # 流式训练与评测常量
    chunk_tokens: int = 512
    tail_tokens: int = 2048
    max_eval_tokens: int = 131072


GEOMETRY_50B = ConsolidationGeometry50B()


def get_canonical_config_dict():
    return asdict(GEOMETRY_50B)
