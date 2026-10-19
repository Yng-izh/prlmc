#!/usr/bin/env python3
"""v6 CPU tests: soft delta bound, layer-group gates, similarity hard mask, runtime disambiguation."""
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from test_natural_merge_v2 import tiny_manager, tiny_qa  # noqa: F401  (fixture contract)
from methods.prlmc.gated_slot_merge import (GraftedSlotMemoryMergeV6, SupervisedDSlotMergeV6,
                                            NativeSkipSlotMergeV6)
from natural_merge_runtime import build_merge, training_for, validate_merge_checkpoint
from prepare_natural_merge_v6 import TRAIN_V6, validate_v6_plan
from prepare_natural_merge_v4 import TRAIN_V4


def model(cls=SupervisedDSlotMergeV6, **kw):
    torch.manual_seed(7)
    defaults = dict(num_groups=2, reference_layer=2, topk=3)
    defaults.update(kw)
    return cls(4, 8, block=2, rank=3, **defaults)


def source():
    g = torch.Generator().manual_seed(11)
    return torch.randn(4, 38, 8, generator=g)


class SoftDeltaBound(unittest.TestCase):
    def test_tiny_delta_stays_tiny(self):
        m = model()
        src = source()
        out = m(src, target=18, carrier_indices=list(range(10, 19)))
        carriers = (src.reshape(4, 19, 2, 8).index_select(1, torch.arange(10, 19)).flatten(1, 2))
        self.assertTrue(torch.allclose(out, carriers, atol=1e-6),
                        "zero-initialized v6 must reproduce carriers (no unit-RMS amplification)")

    def test_one_sided_bound_scales_only_large_deltas(self):
        m = GraftedSlotMemoryMergeV6(4, 8, block=2, rank=3, num_groups=2, reference_layer=2, topk=3)
        with torch.no_grad():
            m.up_weight.normal_(0, .05)
        prob = torch.softmax(torch.randn(8, 3), -1)  # top-k=3 retained weights
        idx = torch.randint(0, 6, (8, 3))
        carrier = (torch.randn(2, 4, 8) * 1e-3)
        delta = m._group_delta(prob, idx, source(), carrier, m.down_weight[0], m.up_weight[0])
        self.assertLess(float(delta.square().mean().sqrt()) / float(carrier.flatten(0, 1).square().mean().sqrt()), 1.05,
                        "oversized delta must be bounded at carrier RMS")
        # tiny delta passes through unscaled (no amplification)
        with torch.no_grad():
            m.up_weight.zero_()
        delta0 = m._group_delta(prob, idx, source(), carrier, m.down_weight[0], m.up_weight[0])
        self.assertEqual(float(delta0.abs().max()), 0.)

    def test_cap_budget_untouched(self):
        m = model()
        m(source(), target=18, carrier_indices=list(range(10, 19)))
        self.assertLessEqual(m.last_residual_ratio, 0.05 + 1e-6)


class GroupGates(unittest.TestCase):
    def test_gate_shapes_are_per_group(self):
        m = model(num_groups=2)
        self.assertEqual(tuple(m.slot_gate.weight.shape), (2, 8))
        self.assertEqual(tuple(m.rel_gate.weight.shape), (2, 3))
        m2 = model(num_groups=4)
        self.assertEqual(tuple(m2.slot_gate.weight.shape), (4, 8))
        self.assertEqual(tuple(m2.rel_gate.weight.shape), (4, 3))

    def test_zero_init_gates_match_initial_gate_bias(self):
        m = model()
        self.assertEqual(float(m.slot_gate.weight.abs().sum()), 0.)
        self.assertEqual(float(m.rel_gate.weight.abs().sum()), 0.)
        import math
        expected = math.log(.02 / (.20 - .02))
        self.assertAlmostEqual(float(m.slot_gate.bias[0]), expected, places=5)

    def test_group_gate_gradients_flow(self):
        m = model(num_groups=2)
        with torch.no_grad():
            m.up_weight.normal_(0, .01); m.slot_gate.weight.normal_(0, .5); m.rel_gate.weight.normal_(0, .5)
        out = m(source(), target=18, carrier_indices=list(range(10, 19)))
        out.sum().backward()
        for p in (m.up_weight, m.slot_gate.weight, m.rel_gate.weight):
            self.assertIsNotNone(p.grad)
            self.assertGreater(float(p.grad.abs().sum()), 0.)

    def test_native_v6_group_gain(self):
        m = model(NativeSkipSlotMergeV6, num_groups=2)
        self.assertEqual(tuple(m.native_gain.shape), (2,))
        self.assertEqual(float(m.native_gain.abs().sum()), 0.)
        with torch.no_grad():
            m.up_weight.normal_(0, .01); m.native_gain.normal_(0, .1)
        out = m(source(), target=18, carrier_indices=list(range(10, 19)))
        out.sum().backward()
        self.assertGreater(float(m.native_gain.grad.abs().sum()), 0.)
        self.assertLessEqual(m.last_residual_ratio, 0.05 + 1e-6)


class SimilarityHardMask(unittest.TestCase):
    def test_reject_only_low_similarity_blocks(self):
        m = model(sim_margin=.5)
        q = torch.nn.functional.normalize(torch.randn(8, 8), dim=-1)
        # 3 low-sim blocks vs 1 exact block; peer-baseline comparison must fire
        # on the clearly below-baseline carriers and keep the exact ones.
        k = torch.cat([torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
                       for _ in range(3)] + [q[6:]])
        keep = m._similarity_mask(q, k, 2)
        self.assertEqual(set(keep.unique().tolist()) <= {0., 1.}, True)
        self.assertEqual(float(keep[6:].min()), 1.)
        rejected = (keep[:6] == 0.).sum().item()
        self.assertGreaterEqual(rejected, 2, "clearly below-baseline carriers must be rejected")

    def test_mask_is_detached(self):
        m = model()
        q = torch.randn(4, 8, requires_grad=True)
        keep = m._similarity_mask(q, torch.randn(4, 8), 2)
        self.assertIsNone(keep.grad_fn)
        self.assertFalse(keep.requires_grad)

    def test_reject_fraction_telemetry(self):
        m = model(num_groups=2)
        m(source(), target=18, carrier_indices=list(range(10, 19)))
        frac = m.last_fusion["similarity_reject_fraction"]
        self.assertTrue(0. <= frac <= 1.)
        self.assertIn("soft_delta_bound", m.last_fusion)
        self.assertIn("no forced unit-RMS", m.last_fusion["soft_delta_bound"])


class RuntimeDisambiguation(unittest.TestCase):
    def test_v6_architectures_build_and_register(self):
        for arch, cls in (("grouped_gated_sparse_slot_v6", (GraftedSlotMemoryMergeV6, SupervisedDSlotMergeV6)),
                          ("grouped_gated_sparse_native_v6", (NativeSkipSlotMergeV6,))):
            m = build_merge(arch, 32, 8, 2)
            self.assertIsInstance(m, cls[0])
            self.assertEqual(m.gamma_max, .20)
            self.assertEqual(m.num_groups, 8)
        training = training_for("grouped_gated_sparse_slot_v6")
        self.assertEqual(training, TRAIN_V6)
        self.assertIsNot(training, TRAIN_V4)

    def test_v6_plan_validates_and_leaves_v4_alone(self):
        self.assertEqual(TRAIN_V4["variants"].get("DsupV6"), None,
                         "TRAIN_V4 must not gain v6 variants (old plans embed deep equality)")
        self.assertEqual(TRAIN_V4["variants"]["Dsup"]["architecture"], "grouped_gated_sparse_slot_v5")

    def test_checkpoint_validation_accepts_v6_training_record(self):
        variant = TRAIN_V6["variants"]["DsupV6"]
        saved = dict(schema="natural_merge_checkpoint/1", variant="DsupV6",
                     architecture=variant["architecture"], evaluation_eligible=True,
                     module_configuration=variant, effective_gamma_max=.05,
                     global_rms_budget=TRAIN_V6["global_rms_budget"], rank=TRAIN_V6["rank"],
                     training=TRAIN_V6)
        validate_merge_checkpoint(saved, "grouped_gated_sparse_slot_v6")

    def test_checkpoint_mismatch_fails_loudly(self):
        variant = TRAIN_V6["variants"]["DsupV6"]
        saved = dict(schema="natural_merge_checkpoint/1", variant="Dsup",
                     architecture="grouped_gated_sparse_slot_v5", evaluation_eligible=True,
                     module_configuration=TRAIN_V4["variants"]["Dsup"], effective_gamma_max=.05,
                     global_rms_budget=TRAIN_V4["global_rms_budget"], rank=TRAIN_V4["rank"],
                     training=TRAIN_V4)
        with self.assertRaises(ValueError):
            validate_merge_checkpoint(saved, "grouped_gated_sparse_slot_v6")

    def test_unknown_architecture_loud_error(self):
        with self.assertRaises(ValueError):
            build_merge("grouped_gated_sparse_slot_v4", 32, 8, 2)

    def test_v6_plan_deep_validates(self):
        plan = dict(schema="natural_merge_v6_training_plan/1", original_v4_plan={"schema": "bogus"},
                    base_plan={}, tokenizer={}, eos_token_id=0, training=TRAIN_V6,
                    episodes=[], qa_events=[], content_sha256="x")
        with self.assertRaises(ValueError):
            validate_v6_plan(plan)


if __name__ == "__main__":
    unittest.main(verbosity=2)
