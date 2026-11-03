import torch
import torch.nn.functional as F

from tools.v9_geometry_50block import GEOMETRY_50B

_SELECT_LAYER = GEOMETRY_50B.select_layer


@torch.no_grad()
def select_window_carrier_blocks(
    window_memory: torch.Tensor,
    target_blocks: int = GEOMETRY_50B.target_blocks,
    block_size: int = GEOMETRY_50B.slots_per_block,
    block_ages: torch.Tensor = None,
    condensed_slot_mask: torch.Tensor = None,
    bridge_quota: int = GEOMETRY_50B.bridge_blocks_quota,
) -> list:
    """
    统一的 Anchor-Bridge 两阶段选块函数。
    Selection 臂与 SharedMerge 臂必须调用完全相同的此函数，保证选块轨迹 100% 同轴。

    Args:
        window_memory: [num_layers, window_blocks * block_size, hidden_dim]
        target_blocks: 保留块数 (默认 9)
        block_size: 每块槽位数 (默认 256)
        block_ages: [window_blocks] 每个块经历过的压缩代数
        condensed_slot_mask: [window_blocks * block_size] 布尔掩码，标记已嫁接凝聚槽位
    Returns:
        按时间升序排列的保留块索引列表，长度为 target_blocks
    """
    total_slots = window_memory.shape[1]
    num_blocks = total_slots // block_size
    if target_blocks >= num_blocks:
        return list(range(num_blocks))

    ref = window_memory[_SELECT_LAYER].float().view(num_blocks, block_size, -1)
    # 1. 计算块内信息离散度 (到块质心的正交能量) + 槽位范数显著性
    slot_norms = ref.norm(dim=-1)  # [B, S]
    ref_unit = F.normalize(ref, dim=-1)
    centroids = F.normalize(ref_unit.mean(dim=1, keepdim=True), dim=-1)  # [B, 1, D]
    dispersion = 1.0 - (ref_unit * centroids).sum(dim=-1).clamp(-1.0, 1.0)  # [B, S]

    slot_salience = slot_norms * (0.5 + dispersion)
    # 如果包含上一轮已凝聚嫁接的槽位，给予归档保护加权，防止二次丢失
    if condensed_slot_mask is not None:
        cond_mask = condensed_slot_mask.view(num_blocks, block_size).float()
        slot_salience = slot_salience * (1.0 + 0.35 * cond_mask)

    # 取每块 Top-64 显著槽位均值作为块基础得分
    top_k = min(64, block_size)
    base_scores = slot_salience.topk(k=top_k, dim=-1).values.mean(dim=-1)  # [B]
    base_scores = base_scores / (base_scores.mean().clamp_min(1e-6))

    # 2. 加入时间新近性微偏置与压缩代数衰减
    recency = torch.linspace(0.0, 0.15, steps=num_blocks, device=ref.device)
    scores = base_scores + recency
    if block_ages is not None:
        ages = block_ages.to(device=ref.device, dtype=torch.float32)
        scores = scores * (GEOMETRY_50B.age_decay ** ages)

    # 3. 第一阶段：选出 (target_blocks - bridge_quota) 个主锚点块 (兼顾分代配额)
    actual_bridge_quota = min(bridge_quota, max(0, target_blocks - 1))
    anchor_quota = target_blocks - actual_bridge_quota

    sorted_candidates = torch.argsort(scores, descending=True).tolist()
    anchors = []
    old_count = 0
    max_old = GEOMETRY_50B.max_old_generation_blocks

    for idx in sorted_candidates:
        if len(anchors) >= anchor_quota:
            break
        is_old = (block_ages is not None) and (int(block_ages[idx].item()) > 0)
        if is_old and old_count >= max_old:
            continue
        anchors.append(idx)
        if is_old:
            old_count += 1

    # 若因代数限制未满，按分数补齐
    for idx in sorted_candidates:
        if len(anchors) >= anchor_quota:
            break
        if idx not in anchors:
            anchors.append(idx)

    # 4. 第二阶段：从剩余块中选出与已选锚点“局部共享桥接实体、但全块不雷同”的二跳互补块
    remaining = [i for i in range(num_blocks) if i not in anchors]
    if actual_bridge_quota > 0 and len(remaining) > 0:
        anchor_tensor = ref_unit[anchors]  # [A, S, D]
        rem_tensor = ref_unit[remaining]  # [R, S, D]

        # 计算剩余块与锚点块之间的槽位级最大相似度 (局部实体桥接强度)
        flat_anchors = anchor_tensor.reshape(-1, ref_unit.shape[-1])  # [A*S, D]
        # [R, S, A*S] -> 每个剩余块内前 16 个最强跨块连接槽位的均值
        sim_matrix = torch.matmul(rem_tensor, flat_anchors.t())  # [R, S, A*S]
        max_slot_sim = sim_matrix.max(dim=-1).values  # [R, S]
        local_bridge_score = max_slot_sim.topk(k=16, dim=-1).values.mean(dim=-1)  # [R]

        # 计算全块质心雷同惩罚 (排除纯重复块)
        anchor_cent = centroids[anchors].squeeze(1)  # [A, D]
        rem_cent = centroids[remaining].squeeze(1)  # [R, D]
        global_redundancy = torch.matmul(rem_cent, anchor_cent.t()).max(dim=-1).values.clamp_min(0.0)

        rem_base = scores[remaining]
        bridge_fitness = 0.45 * rem_base + 0.80 * local_bridge_score - 0.50 * (global_redundancy ** 2)
        top_bridge_rel = torch.argsort(bridge_fitness, descending=True)[:actual_bridge_quota].tolist()
        bridges = [remaining[r_idx] for r_idx in top_bridge_rel]
    else:
        bridges = []

    selected = sorted(anchors + bridges)
    assert len(selected) == target_blocks, f"Expected {target_blocks} blocks, got {len(selected)}"
    return selected
