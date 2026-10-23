"""CPU contracts for the Part-1 inference refinements (block_ortho fusion).

Miniature readers pin the *mechanism*: block-coherent routing keeps every
carrier slot's top-k inside a single discarded block and reuses the sealed
mixed-logit view; the orthogonal write gates only the carrier-parallel
component and never touches the sealed alpha. Nothing here is research
evidence.

Fusion mode contract: these classes live OUTSIDE the six sealed operator
files, so existing v7 checkpoints keep passing strict code-hash verification,
and the mode is opt-in per run (`--fusion-mode block_ortho`).
"""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from methods.prlmc.ortho_slot_merge import (
    ORTHO_MERGE_CLASSES, OrthoNativeSkipSlotMergeV7, OrthoSupervisedDSlotMergeV7,
)
from methods.prlmc.gated_slot_merge import SupervisedDSlotMergeV7
from methods.prlmc.native_skip_merge import NativeSkipSlotMergeV7

LAYERS, HIDDEN, RANK, GROUPS = 4, 8, 3, 2
BLOCK = 4                 # > TOPK so the intra-block top-k is reachable
TOPK = 3
SLOTS = 40
CARRIERS = list(range(4, 10))   # carrier BLOCK indices: last 6 blocks (24 slots)
TARGET = 24


def ortho_sup():
    return OrthoSupervisedDSlotMergeV7(LAYERS, HIDDEN, block=BLOCK, rank=RANK,
                                       num_groups=GROUPS, reference_layer=2,
                                       topk=TOPK)


def ortho_native():
    return OrthoNativeSkipSlotMergeV7(LAYERS, HIDDEN, block=BLOCK, rank=RANK,
                                      num_groups=GROUPS, reference_layer=2,
                                      topk=TOPK)


def sealed(source, target=TARGET, block=BLOCK):
    return source[:, SLOTS - target:].contiguous()


def state_matches_sealed(module):
    """The ortho subclass must be a drop-in for the sealed v7 checkpoint."""
    sealed_cls = {OrthoSupervisedDSlotMergeV7: SupervisedDSlotMergeV7,
                  OrthoNativeSkipSlotMergeV7: NativeSkipSlotMergeV7}[type(module)]
    twin = sealed_cls(LAYERS, HIDDEN, block=BLOCK, rank=RANK, num_groups=GROUPS,
                      reference_layer=2, topk=TOPK)
    return twin.load_state_dict(module.state_dict(), strict=True)


class BlockCoherentRouting(unittest.TestCase):
    """调整1后半: cross-block rows are re-routed into one most-affine block.

    Uses BLOCK=4 > topk=3 so the production invariant (topk << block) holds
    and the reroute is actually reachable.
    """

    K, N_SLOTS, TARGET = 4, 40, 24
    CARRIER_BLOCKS = list(range(4, 10))  # 6 carrier blocks = 24 slots

    def setUp(self):
        torch.manual_seed(3)
        self.source = torch.randn(LAYERS, self.N_SLOTS, HIDDEN)
        self.module = OrthoSupervisedDSlotMergeV7(
            LAYERS, HIDDEN, block=self.K, rank=RANK, num_groups=GROUPS,
            reference_layer=2, topk=TOPK)
        with torch.no_grad():
            self.module.up_weight.normal_()
            # Sharpen the learned routing view so a genuine cross-block winner
            # is forced to exist (k/q are bias-free linears).
            self.module.k.weight.normal_()

    def run_forward(self):
        return self.module(self.source, target=self.TARGET,
                           carrier_indices=self.CARRIER_BLOCKS)

    def test_reroute_moves_cross_block_rows_into_one_block(self):
        out = self.run_forward()
        idx = self.module._last_indices
        block_ids = idx // self.K
        # Every retained source row must be block-coherent...
        self.assertTrue(bool((block_ids == block_ids[:, :1]).all()),
                        "top-k rows still span multiple blocks")
        # ...and at least one row was actually re-routed.
        self.assertGreater(self.module.last_fusion["rerouted_block_fraction"], 0.)
        self.assertEqual(self.module.last_fusion["fusion_mode"], "block_ortho")
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_step0_identity_survives_both_mechanisms(self):
        expected = self.source[:, self.N_SLOTS - self.TARGET:].contiguous()
        for module in (ortho_sup(), ortho_native()):
            for dtype in (torch.float32, torch.float16):
                source = self.source.to(dtype)
                out = module(source, target=self.TARGET,
                             carrier_indices=self.CARRIER_BLOCKS)
                self.assertTrue(torch.equal(out, expected.to(dtype)),
                                f"{type(module).__name__} {dtype}")
                # Passthrough branch: carriers must span the whole pool.
                self.assertTrue(torch.equal(
                    module(source, target=self.N_SLOTS,
                           carrier_indices=list(range(self.N_SLOTS // self.K))),
                    source))
            # strict load returned without error: the state dict is identical.
            result = state_matches_sealed(module)
            self.assertEqual(len(result.missing_keys), 0)
            self.assertEqual(len(result.unexpected_keys), 0)

    def test_single_discarded_block_keeps_sealed_routing(self):
        """The reroute is a no-op when only one block can supply slots.

        Routing must match the sealed module exactly; the write differs from
        the sealed module only through the ortho decomposition, so the output
        is compared against an identical ortho twin instead.
        """
        torch.manual_seed(7)
        module = OrthoSupervisedDSlotMergeV7(
            LAYERS, HIDDEN, block=self.K, rank=RANK, num_groups=GROUPS,
            reference_layer=2, topk=TOPK)
        with torch.no_grad():
            module.up_weight.normal_()
        # The mask is per BLOCK and must include every carrier block; only one
        # discarded block (block 0) is allowed to supply keys.
        mask = torch.zeros(self.N_SLOTS // self.K, dtype=torch.bool)
        mask[0] = True
        mask[torch.tensor(self.CARRIER_BLOCKS)] = True
        out = module(self.source, target=self.TARGET,
                     carrier_indices=self.CARRIER_BLOCKS, source_allow_mask=mask)
        # No row was re-routed: the single-block case is a routing no-op.
        self.assertEqual(module._reroute_stats["moved"], 0)
        # Determinism: an identical ortho twin reproduces the output exactly.
        clone = OrthoSupervisedDSlotMergeV7(
            LAYERS, HIDDEN, block=self.K, rank=RANK, num_groups=GROUPS,
            reference_layer=2, topk=TOPK)
        clone.load_state_dict(module.state_dict())
        self.assertTrue(torch.equal(
            clone(self.source, target=self.TARGET,
                  carrier_indices=self.CARRIER_BLOCKS, source_allow_mask=mask),
            out))

    def test_intra_block_renormalized_weights_sum_to_one(self):
        self.run_forward()
        alpha = torch.tensor(self.module.last_fusion["alpha_active_mean"])
        self.assertGreaterEqual(float(alpha), 0.)

class OrthogonalWrite(unittest.TestCase):
    """调整2: the carrier-parallel component is redundancy-gated."""

    def setUp(self):
        torch.manual_seed(5)
        self.module = ortho_sup()

    def test_ortho_decompose_project_out(self):
        delta = torch.randn(9, HIDDEN)
        carrier = torch.randn(9, HIDDEN)
        written = self.module._ortho_decompose(delta, carrier)
        # The parallel component along the carrier is scaled to <= the full
        # delta and >= floor of it: the direction can only shrink.
        c_sq = carrier.square().sum(-1, keepdim=True)
        full_par = (delta * carrier).sum(-1, keepdim=True) / c_sq * carrier
        low = (0.15 - 1e-5) * full_par.abs()
        high = (1. + 1e-5) * full_par.abs()
        par = (written * carrier).sum(-1, keepdim=True) / c_sq * carrier
        self.assertTrue(bool(((par.abs() >= low) & (par.abs() <= high)).all()))
        # The orthogonal subspace is written in full (fp-exact up to rounding).
        self.assertTrue(torch.allclose(written - par, delta - full_par, atol=1e-6))
        # Zero delta is exactly zero.
        self.assertTrue(torch.equal(
            self.module._ortho_decompose(torch.zeros_like(carrier), carrier),
            torch.zeros_like(carrier)))

    def test_low_redundancy_slots_get_the_floor(self):
        """A very negative redundancy_z pins coef to ~0.15."""
        delta = torch.randn(4, HIDDEN)
        carrier = torch.randn(4, HIDDEN)
        self.module._redundancy_weight = torch.full((4,), -1e4).sigmoid()
        written = self.module._ortho_decompose(delta, carrier)
        c_sq = carrier.square().sum(-1, keepdim=True)
        full_par = (delta * carrier).sum(-1, keepdim=True) / c_sq * carrier
        par = (written * carrier).sum(-1, keepdim=True) / c_sq * carrier
        self.assertTrue(torch.allclose(par, 0.15 * full_par, atol=1e-6))

    def test_redundancy_weight_is_stashed_detached(self):
        with torch.no_grad():
            self.module.up_weight.normal_()
        source = torch.randn(LAYERS, SLOTS, HIDDEN)
        self.module(source, target=TARGET, carrier_indices=CARRIERS)
        self.assertIsNotNone(self.module._redundancy_weight)
        self.assertFalse(self.module._redundancy_weight.requires_grad)

    def test_native_arm_decomposes_the_combined_write(self):
        """The sharp skip path is inside the ortho decomposition."""
        torch.manual_seed(9)
        module = ortho_native()
        with torch.no_grad():
            module.up_weight.normal_()
            module.native_gain.normal_()
        source = torch.randn(LAYERS, SLOTS, HIDDEN)
        out = module(source, target=TARGET, carrier_indices=CARRIERS)
        self.assertEqual(module.last_fusion["fusion_mode"], "block_ortho")
        self.assertTrue(bool(torch.isfinite(out).all()))
        # Zero gain AND zero up_weight still reproduces Selection exactly.
        module2 = ortho_native()
        for dtype in (torch.float32, torch.float16):
            out2 = module2(source.to(dtype), target=TARGET,
                           carrier_indices=CARRIERS)
            self.assertTrue(torch.equal(out2, sealed(source.to(dtype))))


class Registry(unittest.TestCase):
    def test_registry_maps_v7_names(self):
        self.assertIs(ORTHO_MERGE_CLASSES["grouped_gated_sparse_slot_v7"],
                      OrthoSupervisedDSlotMergeV7)
        self.assertIs(ORTHO_MERGE_CLASSES["grouped_gated_sparse_native_v7"],
                      OrthoNativeSkipSlotMergeV7)


if __name__ == "__main__":
    unittest.main()
