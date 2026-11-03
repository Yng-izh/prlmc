import torch

from methods.prlmc.consolidate import select_window_carrier_blocks
from methods.prlmc.native_skip_merge import ManifoldGraftedSlotMerge
from tools.v9_geometry_50block import GEOMETRY_50B


class CoaxialMergeStream:
    """
    严格同轴的 50-block 流式记忆压缩管理器。
    保证 Selection、CarrierOnlyFusion、SharedMerge 使用 100% 相同的选块器与状态追踪。
    """

    def __init__(
        self,
        manager,
        merge_operator: ManifoldGraftedSlotMerge = None,
        arm: str = "SharedMerge",
        budget_blocks: int = GEOMETRY_50B.budget_blocks,
        window_blocks: int = GEOMETRY_50B.window_blocks,
        target_blocks: int = GEOMETRY_50B.target_blocks,
        carrier_schedule: list = None,
        on_compression=None,
    ):
        assert arm in ("Selection", "CarrierOnlyFusion", "SharedMerge", "Full"), f"Unknown arm: {arm}"
        self.manager = manager
        self.merge_operator = merge_operator
        self.arm = arm
        self.block = manager.block
        self.budget = int(budget_blocks) * self.block
        self.window = int(window_blocks) * self.block
        self.target = int(target_blocks) * self.block
        self.carrier_schedule = list(carrier_schedule) if carrier_schedule is not None else None
        self.on_compression = on_compression

        self.total_blocks = manager.total // self.block
        self.block_ages = torch.zeros(self.total_blocks, dtype=torch.long, device=manager._memory.device)
        self.condensed_slot_mask = torch.zeros(manager.total, dtype=torch.bool, device=manager._memory.device)
        self.compression_count = 0
        self.selected_history = []

    def reset(self, initial_memory: torch.Tensor):
        with torch.no_grad():
            self.manager._memory.copy_(initial_memory)
            self.manager.used = 0
            self.manager._tile_free_region()
        self.block_ages.zero_()
        self.condensed_slot_mask.zero_()
        self.compression_count = 0
        self.selected_history.clear()

    def write_chunk(self, delta_slots: torch.Tensor):
        """写入一个 512-token chunk 生成的 256 个原生 latent slots；满池前自动触发同轴压缩。"""
        if self.arm != "Full" and (self.manager.used + delta_slots.shape[1] > self.budget):
            self.compact()

        with torch.no_grad():
            # 左移追加新块
            m = self.manager
            n_new = delta_slots.shape[1]
            m._memory = torch.roll(m._memory, shifts=-n_new, dims=1)
            m._memory[:, -n_new:, :].copy_(delta_slots.to(m.dtype))
            m.used = min(m.total, m.used + n_new)
            m._tile_free_region()

            # 同步维护块年龄与凝聚槽位保护掩码
            self.block_ages = torch.roll(self.block_ages, shifts=-1, dims=0)
            self.block_ages[-1] = 0
            self.condensed_slot_mask = torch.roll(self.condensed_slot_mask, shifts=-n_new, dims=0)
            self.condensed_slot_mask[-n_new:] = False

    def compact(self):
        m = self.manager
        start = m.total - m.used
        win_mem = m._memory[:, start : start + self.window, :]
        tail_mem = m._memory[:, start + self.window :, :].clone()

        b_start = start // self.block
        b_win = self.window // self.block
        win_ages = self.block_ages[b_start : b_start + b_win]
        tail_ages = self.block_ages[b_start + b_win :].clone()

        win_cond_mask = self.condensed_slot_mask[start : start + self.window]
        tail_cond_mask = self.condensed_slot_mask[start + self.window :].clone()

        # 【核心修复 P0-2】：无论 Selection 还是 SharedMerge，若无外部 schedule，均调用完全相同的两阶段选块器
        if self.carrier_schedule is not None and len(self.carrier_schedule) > self.compression_count:
            selected = list(self.carrier_schedule[self.compression_count])
        else:
            selected = select_window_carrier_blocks(
                window_memory=win_mem,
                target_blocks=self.target // self.block,
                block_size=self.block,
                block_ages=win_ages,
                condensed_slot_mask=win_cond_mask,
            )

        self.selected_history.append(selected)

        if self.arm == "Selection":
            sel_t = torch.as_tensor(selected, dtype=torch.long, device=win_mem.device)
            win_b = win_mem.view(win_mem.shape[0], b_win, self.block, win_mem.shape[-1])
            candidate = win_b.index_select(1, sel_t).reshape(win_mem.shape[0], self.target, win_mem.shape[-1])
            new_win_cond = win_cond_mask.view(b_win, self.block).index_select(0, sel_t).reshape(-1)
            telemetry = {"mode": "Selection", "selected_blocks": selected}
        elif self.arm == "CarrierOnlyFusion":
            candidate, new_win_cond, telemetry = self.merge_operator(
                win_mem, selected, win_cond_mask, disable_source_fusion=True, enable_grafting=False
            )
        else:  # SharedMerge
            assert self.merge_operator is not None, "SharedMerge requires a loaded ManifoldGraftedSlotMerge"
            candidate, new_win_cond, telemetry = self.merge_operator(
                win_mem, selected, win_cond_mask, disable_source_fusion=False, enable_grafting=True
            )

        # 写回压缩后的 9 块 + 尾部未压缩块
        with torch.no_grad():
            replacement = torch.cat([candidate.to(m.dtype), tail_mem], dim=1)
            m.used = replacement.shape[1]
            m._memory[:, m.total - m.used :, :].copy_(replacement)
            m._tile_free_region()

            sel_t = torch.as_tensor(selected, dtype=torch.long, device=win_ages.device)
            new_ages = torch.cat([win_ages.index_select(0, sel_t) + 1, tail_ages], dim=0)
            self.block_ages.zero_()
            self.block_ages[self.total_blocks - new_ages.numel() :].copy_(new_ages)

            updated_cond = torch.cat([new_win_cond, tail_cond_mask], dim=0)
            self.condensed_slot_mask.zero_()
            self.condensed_slot_mask[m.total - updated_cond.numel() :].copy_(updated_cond)

        self.compression_count += 1
        if self.on_compression is not None:
            self.on_compression(self.compression_count, selected, telemetry)
        return candidate, selected, telemetry
