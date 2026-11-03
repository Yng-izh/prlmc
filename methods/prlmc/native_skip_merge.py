import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.v9_geometry_50block import GEOMETRY_50B


class ManifoldGraftedSlotMerge(nn.Module):
    """
    PRLM-C V10 核心算子：流形保范槽位嫁接与双路凝聚模块。

    - 100% 保持 MemoryLLM 原生 32 层 K/V 流形与逐层 RMS 能量分布，无需任何 Reader LoRA / K/V Adapter。
    - 保留块内 216 个核心槽位 (84.4%) 原生受保护；腾退 40 个冗余槽位 (15.6%)。
    - 从 10 个淘汰块 (2560 slots) 中提取 360 个高信息量锚点槽位，在各自同源块内执行保范测地线凝聚，
      加上零初始化正交关系微调 (上限 0.05)，并按【严格原始时间顺序】拼入最终的 2304 个槽位。
    """

    def __init__(
        self,
        num_layers: int = GEOMETRY_50B.num_layers,
        hidden_size: int = GEOMETRY_50B.hidden_size,
        num_groups: int = 8,
        rank: int = GEOMETRY_50B.ortho_adapter_rank,
        reference_layer: int = GEOMETRY_50B.reference_layer,
        block_size: int = GEOMETRY_50B.slots_per_block,
        core_slots_per_block: int = GEOMETRY_50B.core_slots_per_block,
        graft_slots_per_block: int = GEOMETRY_50B.graft_slots_per_block,
        ortho_residual_max: float = GEOMETRY_50B.ortho_residual_max_ratio,
    ):
        super().__init__()
        assert core_slots_per_block + graft_slots_per_block == block_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.layers_per_group = num_layers // num_groups
        self.rank = rank
        self.reference_layer = reference_layer
        self.block_size = block_size
        self.core_slots_per_block = core_slots_per_block
        self.graft_slots_per_block = graft_slots_per_block
        self.ortho_residual_max = ortho_residual_max

        # 轻量参考层路由微调投影 (零初始化残差，保证训练起点即为纯净原生流形几何)
        self.route_q = nn.Linear(hidden_size, rank, bias=False)
        self.route_k = nn.Linear(hidden_size, rank, bias=False)
        nn.init.orthogonal_(self.route_q.weight, gain=0.1)
        nn.init.zeros_(self.route_k.weight)

        # 8 个层组的正交上下文低秩适配器 (up_proj 严格零初始化)
        self.down_proj = nn.Parameter(torch.empty(num_groups, rank, hidden_size))
        self.up_proj = nn.Parameter(torch.zeros(num_groups, hidden_size, rank))
        for g in range(num_groups):
            nn.init.orthogonal_(self.down_proj[g], gain=0.1)

        # 可学习的同源凝聚温度与正交门控偏置 (零初始化)
        self.cluster_logit_scale = nn.Parameter(torch.tensor(2.5))
        self.ortho_gate_logits = nn.Parameter(torch.full((num_groups,), -2.0))
        self.homo_gate_logits = nn.Parameter(torch.full((num_groups,), -2.0))

    @staticmethod
    def _span3_context(x_block: torch.Tensor) -> torch.Tensor:
        """块内 3-gram 局部上下文池化 [B, S, D]，增强实体—关系绑定匹配。"""
        left = torch.cat([x_block[:, :1, :], x_block[:, :-1, :]], dim=1)
        right = torch.cat([x_block[:, 1:, :], x_block[:, -1:, :]], dim=1)
        return F.normalize(0.70 * x_block + 0.15 * left + 0.15 * right, dim=-1)

    @staticmethod
    def _geodesic_rms_preserve(base: torch.Tensor, neighbor_mean: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        流形保范球面插值：将 base 与同源邻域均值按权重融合后，严格恢复 base 在每一层的原生 RMS 范数。
        彻底消除线性加权导致的注意力 Key 模长坍缩问题。
        base, neighbor_mean: [L, N, D], weight: [1, N, 1] or [L, N, 1]
        """
        orig_rms = base.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        mixed = (1.0 - weight) * base.float() + weight * neighbor_mean.float()
        mixed_rms = mixed.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        return (mixed * (orig_rms / mixed_rms)).to(base.dtype)

    def _orthogonal_context_delta(
        self, anchor_layers: torch.Tensor, ctx_layers: torch.Tensor
    ) -> torch.Tensor:
        """
        计算严格正交于 anchor 主方向的有界低秩关系增量 (幅度 <= ortho_residual_max * RMS)。
        anchor_layers, ctx_layers: [L, N, D]
        """
        L, N, D = anchor_layers.shape
        G = self.num_groups
        lpg = self.layers_per_group

        a_grp = anchor_layers.float().view(G, lpg, N, D)
        c_grp = ctx_layers.float().view(G, lpg, N, D)
        diff = c_grp - a_grp  # [G, lpg, N, D]

        # 低秩瓶颈投影: [G, lpg, N, D] -> [G, lpg, N, R] -> [G, lpg, N, D]
        low_rank = F.gelu(torch.einsum("glnd,grd->glnr", diff, self.down_proj.float()))
        raw_delta = torch.einsum("glnr,gdr->glnd", low_rank, self.up_proj.float())

        # Gram-Schmidt 正交化：剔除平行于原实体向量的分量，绝不翻转实体身份
        a_norm_sq = a_grp.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        proj = (raw_delta * a_grp).sum(dim=-1, keepdim=True) / a_norm_sq
        ortho_delta = raw_delta - proj * a_grp

        # 软有界幅度约束 (严格 <= 0.05 * anchor_rms)
        a_rms = a_grp.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        gate = torch.sigmoid(self.ortho_gate_logits.float()).view(G, 1, 1, 1) * self.ortho_residual_max
        bounded_ortho = gate * a_rms * torch.tanh(ortho_delta / a_rms)
        return bounded_ortho.reshape(L, N, D)

    def forward(
        self,
        window_memory: torch.Tensor,
        selected_blocks: list,
        condensed_slot_mask: torch.Tensor = None,
        disable_source_fusion: bool = False,
        enable_grafting: bool = True,
    ):
        """
        Args:
            window_memory: [L=32, W_blocks * 256, D=4096]
            selected_blocks: 长度为 target_blocks (9) 的保留块索引列表
            condensed_slot_mask: [W_blocks * 256] 标记哪些槽位已是历史凝聚槽位 (享有写保护)
            disable_source_fusion: 若为 True (源屏蔽契约测试)，精确退化为纯 Selection
            enable_grafting: 是否开启冗余槽位腾退与丢弃块原生槽位嫁接
        Returns:
            compacted_memory: [L=32, target_blocks * 256, D=4096]
            new_condensed_mask: [target_blocks * 256] 更新后的凝聚保护掩码
            telemetry: 诊断统计字典
        """
        L, total_slots, D = window_memory.shape
        B = self.block_size
        num_blocks = total_slots // B
        sel_set = set(int(i) for i in selected_blocks)
        discard_blocks = [i for i in range(num_blocks) if i not in sel_set]

        win_blocks_tensor = window_memory.view(L, num_blocks, B, D)
        if condensed_slot_mask is None:
            condensed_slot_mask = torch.zeros(total_slots, dtype=torch.bool, device=window_memory.device)
        cond_mask_blocks = condensed_slot_mask.view(num_blocks, B)

        # 若无丢弃块或显式关闭源融合，直接返回 Selection 原生块
        if len(discard_blocks) == 0 or disable_source_fusion:
            sel_idx = torch.as_tensor(selected_blocks, dtype=torch.long, device=window_memory.device)
            out_mem = win_blocks_tensor.index_select(1, sel_idx).reshape(L, len(selected_blocks) * B, D)
            out_mask = cond_mask_blocks.index_select(0, sel_idx).reshape(-1)
            return out_mem, out_mask, {"mode": "selection_passthrough", "grafted_slots": 0}

        # 提取第 16 层参考表示
        ref_all = win_blocks_tensor[self.reference_layer].float()  # [num_blocks, B, D]
        ref_unit = F.normalize(ref_all, dim=-1)
        ref_span = self._span3_context(ref_unit)  # [num_blocks, B, D]

        # 加入轻量可学习路由修正 (初始化为 0)
        q_learned = self.route_q(ref_span)
        k_learned = self.route_k(ref_span)
        route_repr = F.normalize(ref_span + 0.1 * F.linear(q_learned, self.route_k.weight.t()), dim=-1)

        # ==================== 1. 保留块内分区：识别 216 个 Core Slots 与 40 个待腾退冗余槽位 ====================
        sel_t = torch.as_tensor(selected_blocks, dtype=torch.long, device=window_memory.device)
        disc_t = torch.as_tensor(discard_blocks, dtype=torch.long, device=window_memory.device)

        carrier_ref = ref_all.index_select(0, sel_t)  # [9, B, D]
        carrier_unit = ref_unit.index_select(0, sel_t)  # [9, B, D]
        carrier_cond_mask = cond_mask_blocks.index_select(0, sel_t)  # [9, B]

        carrier_cent = F.normalize(carrier_unit.mean(dim=1, keepdim=True), dim=-1)
        carrier_disp = 1.0 - (carrier_unit * carrier_cent).sum(dim=-1)  # 越大越独特
        carrier_salience = carrier_ref.norm(dim=-1) * (0.5 + carrier_disp)
        # 历史已凝聚槽位与每块末尾锚点槽位强制保护，不被腾退
        carrier_salience = carrier_salience + 1e4 * carrier_cond_mask.float()

        num_target = len(selected_blocks)
        num_graft_total = num_target * self.graft_slots_per_block if enable_grafting else 0

        if num_graft_total > 0:
            # 每块保留显著性最高的 216 个 Core 槽位 (保持块内原始先后顺序)
            core_local_idx = carrier_salience.topk(k=self.core_slots_per_block, dim=-1).indices  # [9, 216]
            core_local_idx, _ = torch.sort(core_local_idx, dim=-1)
        else:
            core_local_idx = torch.arange(B, device=window_memory.device).unsqueeze(0).expand(num_target, B)

        # 计算保留 Core 槽位在原始 19-block 窗口中的全局时间坐标 (用于最终严格时序重排)
        carrier_global_pos = sel_t.unsqueeze(1) * B + core_local_idx  # [9, 216]
        flat_core_pos = carrier_global_pos.reshape(-1)  # [1944]
        flat_core_layers = window_memory.index_select(1, flat_core_pos)  # [L, 1944, D]
        flat_core_route = route_repr.view(total_slots, D).index_select(0, flat_core_pos)  # [1944, D]

        # ==================== 2. 通道 B：Core 槽位的跨块同体保范强化 (仅当余弦 > 0.85) ====================
        disc_global_base = disc_t.unsqueeze(1) * B + torch.arange(B, device=window_memory.device).unsqueeze(0)
        flat_disc_pos = disc_global_base.reshape(-1)  # [2560]
        flat_disc_layers = window_memory.index_select(1, flat_disc_pos)  # [L, 2560, D]
        flat_disc_route = route_repr.view(total_slots, D).index_select(0, flat_disc_pos)  # [2560, D]

        # 计算 Core 与丢弃槽位的相似度 [1944, 2560]
        core_disc_sim = torch.matmul(flat_core_route, flat_disc_route.t())
        max_sim_val, max_sim_idx = core_disc_sim.max(dim=-1)  # [1944]
        homo_mask = (max_sim_val > GEOMETRY_50B.cross_block_homo_cos_thr).float()  # [1944]

        matched_disc_layers = flat_disc_layers.index_select(1, max_sim_idx)  # [L, 1944, D]
        G = self.num_groups
        lpg = self.layers_per_group
        homo_alpha_grp = (
            torch.sigmoid(self.homo_gate_logits.float()).view(G, 1, 1, 1)
            * GEOMETRY_50B.homo_reinforce_max_alpha
        )
        homo_alpha = homo_alpha_grp.expand(G, lpg, 1, 1).reshape(L, 1, 1) * homo_mask.view(1, -1, 1)
        reinforced_core_layers = self._geodesic_rms_preserve(
            flat_core_layers, matched_disc_layers, homo_alpha
        )

        if num_graft_total == 0:
            return (
                reinforced_core_layers,
                condensed_slot_mask.index_select(0, flat_core_pos),
                {"mode": "homo_only", "grafted_slots": 0, "homo_matched": int(homo_mask.sum().item())},
            )

        # ==================== 3. 通道 A：从 10 个淘汰块中凝聚 360 个高信息量原生槽位 ====================
        disc_ref = ref_all.index_select(0, disc_t)  # [10, B, D]
        disc_unit = ref_unit.index_select(0, disc_t)  # [10, B, D]
        disc_cent = F.normalize(disc_unit.mean(dim=1, keepdim=True), dim=-1)
        disc_disp = 1.0 - (disc_unit * disc_cent).sum(dim=-1)  # [10, B]

        # 槽位信息杠杆率 = 自身独特显著性 × 与保留 Core 的桥接相关度 × (1 - 与 Core 的完全重复度)
        disc_to_core_max = core_disc_sim.max(dim=0).values.view(len(discard_blocks), B)  # [10, B]
        novelty_factor = (1.0 - (disc_to_core_max - 0.85).clamp_min(0.0) * 5.0).clamp_min(0.1)
        bridge_relevance = disc_to_core_max.clamp(0.2, 0.82)
        disc_leverage = disc_ref.norm(dim=-1) * (0.6 + disc_disp) * (0.5 + bridge_relevance) * novelty_factor

        flat_leverage = disc_leverage.reshape(-1)  # [2560]
        graft_pick_idx = flat_leverage.topk(k=num_graft_total, dim=0).indices  # [360]
        graft_pick_idx, _ = torch.sort(graft_pick_idx)  # 保持原始时间顺序

        graft_global_pos = flat_disc_pos.index_select(0, graft_pick_idx)  # [360]
        graft_anchor_layers = flat_disc_layers.index_select(1, graft_pick_idx)  # [L, 360, D]

        # 同源块内局部簇凝聚：每个被选中的淘汰 Anchor 仅聚合【同一个淘汰块内】余弦 > 0.78 的同义槽位
        graft_block_ids = graft_pick_idx // B  # [360] 属于第几个淘汰块
        all_disc_block_ids = torch.arange(len(discard_blocks), device=window_memory.device).unsqueeze(1).expand(-1, B).reshape(-1)
        same_block_mask = (graft_block_ids.unsqueeze(1) == all_disc_block_ids.unsqueeze(0))  # [360, 2560]

        graft_route = flat_disc_route.index_select(0, graft_pick_idx)  # [360, D]
        intra_sim = torch.matmul(graft_route, flat_disc_route.t())  # [360, 2560]
        valid_neighbor_mask = same_block_mask & (intra_sim > GEOMETRY_50B.intra_discard_cluster_cos_thr)

        temp = self.cluster_logit_scale.float().clamp(0.5, 10.0)
        masked_logits = torch.where(
            valid_neighbor_mask, intra_sim * temp, torch.full_like(intra_sim, -1e9)
        )
        # 自身必定在邻域内 (intra_sim = 1.0 > 0.78)，因此 softmax 恒良定义且数值稳定
        cluster_weights = F.softmax(masked_logits, dim=-1).to(flat_disc_layers.dtype)  # [360, 2560]
        cluster_ctx_layers = torch.einsum("gn,lnd->lgd", cluster_weights.float(), flat_disc_layers.float())  # [L, 360, D]

        # 流形保范凝聚 + 有界正交关系微调
        condensed_graft_layers = self._geodesic_rms_preserve(
            graft_anchor_layers,
            cluster_ctx_layers.to(graft_anchor_layers.dtype),
            weight=torch.tensor(0.25, device=window_memory.device).view(1, 1, 1),
        )
        ortho_delta = self._orthogonal_context_delta(condensed_graft_layers, cluster_ctx_layers)
        condensed_graft_layers = self._geodesic_rms_preserve(
            condensed_graft_layers,
            (condensed_graft_layers.float() + ortho_delta).to(condensed_graft_layers.dtype),
            weight=torch.tensor(1.0, device=window_memory.device).view(1, 1, 1),
        )

        # ==================== 4. 严格按原始时间顺序合并 1944 个 Core 与 360 个 Graft 槽位 ====================
        all_global_pos = torch.cat([flat_core_pos, graft_global_pos], dim=0)  # [2304]
        all_layers = torch.cat([reinforced_core_layers, condensed_graft_layers], dim=1)  # [L, 2304, D]

        core_cond_flags = condensed_slot_mask.index_select(0, flat_core_pos)
        graft_cond_flags = torch.ones(num_graft_total, dtype=torch.bool, device=window_memory.device)
        all_cond_flags = torch.cat([core_cond_flags, graft_cond_flags], dim=0)

        # 按全局时间位置升序排列，保证 RoPE 与跨槽位时序 100% 单调连续
        chronological_order = torch.argsort(all_global_pos, stable=True)
        final_memory = all_layers.index_select(1, chronological_order)
        final_cond_mask = all_cond_flags.index_select(0, chronological_order)

        telemetry = {
            "mode": "manifold_graft_v10",
            "selected_blocks": [int(x) for x in selected_blocks],
            "core_slots_kept": int(flat_core_pos.numel()),
            "grafted_slots": int(num_graft_total),
            "homo_reinforced_slots": int(homo_mask.sum().item()),
            "total_condensed_protected": int(final_cond_mask.sum().item()),
        }
        return final_memory, final_cond_mask, telemetry
