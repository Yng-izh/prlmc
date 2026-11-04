import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.v9_geometry_50block import GEOMETRY_50B


class ManifoldGraftedSlotMerge(nn.Module):
    """V10 core operator: manifold-preserving slot grafting.

    Keeps the frozen reader's native K/V manifold: every emitted slot is a
    native slot, or a norm-matched geodesic condensation of same-block native
    slots. No reader LoRA, no K/V adapter.
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
        # "full" is the trained V10 operator. "native" is the diagnostic mode
        # (experiment D): carriers and graft slots are copied verbatim, with no
        # same-block condensation, no orthogonal context delta and no core
        # reinforcement, so only native slots ever reach the reader.
        self.operator_mode = "full"

        # Low-rank routing residual, zero-initialised: training starts exactly on
        # the native manifold.
        self.route_q = nn.Linear(hidden_size, rank, bias=False)
        self.route_k = nn.Linear(hidden_size, rank, bias=False)
        nn.init.orthogonal_(self.route_q.weight, gain=0.1)
        nn.init.zeros_(self.route_k.weight)

        # Per-layer-group orthogonal context adapter (up_proj strictly zero).
        self.down_proj = nn.Parameter(torch.empty(num_groups, rank, hidden_size))
        self.up_proj = nn.Parameter(torch.zeros(num_groups, hidden_size, rank))
        for g in range(num_groups):
            nn.init.orthogonal_(self.down_proj[g], gain=0.1)

        self.cluster_logit_scale = nn.Parameter(torch.tensor(2.5))
        self.ortho_gate_logits = nn.Parameter(torch.full((num_groups,), -2.0))
        self.homo_gate_logits = nn.Parameter(torch.full((num_groups,), -2.0))

    @staticmethod
    def _span3_context(x_block: torch.Tensor) -> torch.Tensor:
        left = torch.cat([x_block[:, :1, :], x_block[:, :-1, :]], dim=1)
        right = torch.cat([x_block[:, 1:, :], x_block[:, -1:, :]], dim=1)
        return F.normalize(0.70 * x_block + 0.15 * left + 0.15 * right, dim=-1)

    @staticmethod
    def _geodesic_rms_preserve(base: torch.Tensor, neighbor_mean: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Blend then restore base's per-layer RMS so key norms never collapse."""
        orig_rms = base.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        mixed = (1.0 - weight) * base.float() + weight * neighbor_mean.float()
        mixed_rms = mixed.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        return (mixed * (orig_rms / mixed_rms)).to(base.dtype)

    def _orthogonal_context_delta(self, anchor_layers: torch.Tensor, ctx_layers: torch.Tensor) -> torch.Tensor:
        L, N, D = anchor_layers.shape
        G = self.num_groups
        lpg = self.layers_per_group

        a_grp = anchor_layers.float().view(G, lpg, N, D)
        c_grp = ctx_layers.float().view(G, lpg, N, D)
        diff = c_grp - a_grp

        low_rank = F.gelu(torch.einsum("glnd,grd->glnr", diff, self.down_proj.float()))
        raw_delta = torch.einsum("glnr,gdr->glnd", low_rank, self.up_proj.float())

        a_norm_sq = a_grp.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        proj = (raw_delta * a_grp).sum(dim=-1, keepdim=True) / a_norm_sq
        ortho_delta = raw_delta - proj * a_grp

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
        L, total_slots, D = window_memory.shape
        B = self.block_size
        num_blocks = total_slots // B
        sel_set = set(int(i) for i in selected_blocks)
        discard_blocks = [i for i in range(num_blocks) if i not in sel_set]

        win_blocks_tensor = window_memory.view(L, num_blocks, B, D)
        if condensed_slot_mask is None:
            condensed_slot_mask = torch.zeros(total_slots, dtype=torch.bool, device=window_memory.device)
        cond_mask_blocks = condensed_slot_mask.view(num_blocks, B)

        if len(discard_blocks) == 0 or disable_source_fusion:
            sel_idx = torch.as_tensor(selected_blocks, dtype=torch.long, device=window_memory.device)
            out_mem = win_blocks_tensor.index_select(1, sel_idx).reshape(L, len(selected_blocks) * B, D)
            out_mask = cond_mask_blocks.index_select(0, sel_idx).reshape(-1)
            return out_mem, out_mask, {"mode": "selection_passthrough", "grafted_slots": 0}

        ref_all = win_blocks_tensor[self.reference_layer].float()
        ref_unit = F.normalize(ref_all, dim=-1)
        ref_span = self._span3_context(ref_unit)

        q_learned = self.route_q(ref_span)
        route_repr = F.normalize(ref_span + 0.1 * F.linear(q_learned, self.route_k.weight.t()), dim=-1)

        sel_t = torch.as_tensor(selected_blocks, dtype=torch.long, device=window_memory.device)
        disc_t = torch.as_tensor(discard_blocks, dtype=torch.long, device=window_memory.device)

        carrier_ref = ref_all.index_select(0, sel_t)
        carrier_unit = ref_unit.index_select(0, sel_t)
        carrier_cond_mask = cond_mask_blocks.index_select(0, sel_t)

        carrier_cent = F.normalize(carrier_unit.mean(dim=1, keepdim=True), dim=-1)
        carrier_disp = 1.0 - (carrier_unit * carrier_cent).sum(dim=-1)
        carrier_salience = carrier_ref.norm(dim=-1) * (0.5 + carrier_disp)
        carrier_salience = carrier_salience + 1e4 * carrier_cond_mask.float()

        num_target = len(selected_blocks)
        num_graft_total = num_target * self.graft_slots_per_block if enable_grafting else 0

        if num_graft_total > 0:
            core_local_idx = carrier_salience.topk(k=self.core_slots_per_block, dim=-1).indices
            core_local_idx, _ = torch.sort(core_local_idx, dim=-1)
        else:
            core_local_idx = torch.arange(B, device=window_memory.device).unsqueeze(0).expand(num_target, B)

        carrier_global_pos = sel_t.unsqueeze(1) * B + core_local_idx
        flat_core_pos = carrier_global_pos.reshape(-1)
        flat_core_layers = window_memory.index_select(1, flat_core_pos)
        flat_core_route = route_repr.view(total_slots, D).index_select(0, flat_core_pos)

        disc_global_base = disc_t.unsqueeze(1) * B + torch.arange(B, device=window_memory.device).unsqueeze(0)
        flat_disc_pos = disc_global_base.reshape(-1)
        flat_disc_layers = window_memory.index_select(1, flat_disc_pos)
        flat_disc_route = route_repr.view(total_slots, D).index_select(0, flat_disc_pos)

        core_disc_sim = torch.matmul(flat_core_route, flat_disc_route.t())
        max_sim_val, max_sim_idx = core_disc_sim.max(dim=-1)
        homo_mask = (max_sim_val > GEOMETRY_50B.cross_block_homo_cos_thr).float()

        matched_disc_layers = flat_disc_layers.index_select(1, max_sim_idx)
        G = self.num_groups
        lpg = self.layers_per_group
        homo_alpha_grp = (
            torch.sigmoid(self.homo_gate_logits.float()).view(G, 1, 1, 1)
            * GEOMETRY_50B.homo_reinforce_max_alpha
        )
        homo_alpha = homo_alpha_grp.expand(G, lpg, 1, 1).reshape(L, 1, 1) * homo_mask.view(1, -1, 1)
        if self.operator_mode == "native":
            reinforced_core_layers = flat_core_layers
            homo_matched = 0
        else:
            reinforced_core_layers = self._geodesic_rms_preserve(flat_core_layers, matched_disc_layers, homo_alpha)
            homo_matched = int(homo_mask.sum().item())

        if num_graft_total == 0:
            return (
                reinforced_core_layers,
                condensed_slot_mask.index_select(0, flat_core_pos),
                {"mode": "homo_only", "grafted_slots": 0, "homo_matched": int(homo_mask.sum().item())},
            )

        disc_ref = ref_all.index_select(0, disc_t)
        disc_unit = ref_unit.index_select(0, disc_t)
        disc_cent = F.normalize(disc_unit.mean(dim=1, keepdim=True), dim=-1)
        disc_disp = 1.0 - (disc_unit * disc_cent).sum(dim=-1)

        disc_to_core_max = core_disc_sim.max(dim=0).values.view(len(discard_blocks), B)
        novelty_factor = (1.0 - (disc_to_core_max - 0.85).clamp_min(0.0) * 5.0).clamp_min(0.1)
        bridge_relevance = disc_to_core_max.clamp(0.2, 0.82)
        disc_leverage = disc_ref.norm(dim=-1) * (0.6 + disc_disp) * (0.5 + bridge_relevance) * novelty_factor

        flat_leverage = disc_leverage.reshape(-1)
        graft_pick_idx = flat_leverage.topk(k=num_graft_total, dim=0).indices
        graft_pick_idx, _ = torch.sort(graft_pick_idx)

        graft_global_pos = flat_disc_pos.index_select(0, graft_pick_idx)
        graft_anchor_layers = flat_disc_layers.index_select(1, graft_pick_idx)

        graft_block_ids = graft_pick_idx // B
        all_disc_block_ids = torch.arange(len(discard_blocks), device=window_memory.device).unsqueeze(1).expand(-1, B).reshape(-1)
        same_block_mask = (graft_block_ids.unsqueeze(1) == all_disc_block_ids.unsqueeze(0))

        graft_route = flat_disc_route.index_select(0, graft_pick_idx)
        intra_sim = torch.matmul(graft_route, flat_disc_route.t())
        valid_neighbor_mask = same_block_mask & (intra_sim > GEOMETRY_50B.intra_discard_cluster_cos_thr)

        temp = self.cluster_logit_scale.float().clamp(0.5, 10.0)
        masked_logits = torch.where(
            valid_neighbor_mask, intra_sim * temp, torch.full_like(intra_sim, -1e9)
        )
        cluster_weights = F.softmax(masked_logits, dim=-1).to(flat_disc_layers.dtype)
        L_ = flat_disc_layers.shape[0]
        cluster_ctx_layers = torch.bmm(
            cluster_weights.float().unsqueeze(0).expand(L_, -1, -1), flat_disc_layers.float()
        )

        if self.operator_mode == "native":
            # Experiment D: the picked native slots are copied unchanged.
            condensed_graft_layers = graft_anchor_layers
        else:
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

        all_global_pos = torch.cat([flat_core_pos, graft_global_pos], dim=0)
        all_layers = torch.cat([reinforced_core_layers, condensed_graft_layers], dim=1)

        core_cond_flags = condensed_slot_mask.index_select(0, flat_core_pos)
        graft_cond_flags = torch.ones(num_graft_total, dtype=torch.bool, device=window_memory.device)
        all_cond_flags = torch.cat([core_cond_flags, graft_cond_flags], dim=0)

        chronological_order = torch.argsort(all_global_pos, stable=True)
        final_memory = all_layers.index_select(1, chronological_order)
        final_cond_mask = all_cond_flags.index_select(0, chronological_order)

        core_count = int(flat_core_pos.numel())
        graft_out_mask = chronological_order >= core_count
        bridge_slot_offsets = torch.nonzero(graft_out_mask, as_tuple=False).flatten().tolist()
        graft_src_blocks = disc_t[graft_pick_idx // B]
        core_src_blocks = sel_t.repeat_interleave(self.core_slots_per_block)
        slot_source_blocks = torch.cat([core_src_blocks, graft_src_blocks])[chronological_order].tolist()

        telemetry = {
            "mode": "manifold_graft_v10",
            "bridge_slot_offsets": bridge_slot_offsets,
            "slot_source_blocks": slot_source_blocks,
            "core_slot_count": core_count,
            "selected_blocks": [int(x) for x in selected_blocks],
            "core_slots_kept": int(flat_core_pos.numel()),
            "grafted_slots": int(num_graft_total),
            "operator_mode": self.operator_mode,
            "homo_reinforced_slots": homo_matched,
            "total_condensed_protected": int(final_cond_mask.sum().item()),
        }
        return final_memory, final_cond_mask, telemetry
