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


class ZeroInitBackward(unittest.TestCase):
    """Zero-initialised parameters must not NaN the first backward pass."""

    def _assert_finite_grads(self, m):
        out = m(source(), target=18, carrier_indices=list(range(10, 19)))
        self.assertTrue(torch.isfinite(out).all())
        out.sum().backward()
        checked = 0
        for name, p in m.named_parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all(), f"non-finite grad in {name}")
                checked += 1
        self.assertGreater(checked, 0)

    def test_native_v6_zero_init_has_no_nan_gradient(self):
        # Both up_weight and native_gain are zero at init, so the combined
        # residual is exactly zero: sqrt(0) outside the eps NaNed the backward.
        m = model(NativeSkipSlotMergeV6, num_groups=2)
        self.assertEqual(float(m.up_weight.abs().sum()), 0.)
        self.assertEqual(float(m.native_gain.abs().sum()), 0.)
        self._assert_finite_grads(m)

    def test_adapted_v6_zero_init_has_no_nan_gradient(self):
        m = model(SupervisedDSlotMergeV6, num_groups=2)
        self.assertEqual(float(m.up_weight.abs().sum()), 0.)
        self._assert_finite_grads(m)


class GeometryFeatures(unittest.TestCase):
    """Redundancy must be measured on the raw slots, not on z-scored ones."""

    def test_redundancy_matches_raw_block_centroid(self):
        m = model()
        torch.manual_seed(3)
        q = torch.nn.functional.normalize(torch.randn(8, 8), dim=-1)
        probs = torch.full((8, 3), 1. / 3.)
        _, _, raw = m._gate_inputs(q, probs, torch.zeros(8))
        redundancy = raw[..., 2]
        slots = q.reshape(-1, 2, 8)
        centroid = torch.nn.functional.normalize(slots.mean(1, keepdim=True), dim=-1)
        self.assertTrue(torch.allclose(redundancy, (slots * centroid).sum(-1), atol=1e-6))

    def test_z_scored_block_centroid_collapses_to_noise(self):
        # The old code fed the de-meaned slots into the centroid: the block
        # centroid is then the zero vector, and `_l2` divides by 1e-6, turning
        # float rounding into a unit-length random direction. Documented here so
        # the regression cannot come back silently.
        m = model()
        torch.manual_seed(5)
        q = torch.nn.functional.normalize(torch.randn(8, 8), dim=-1)
        _, _, raw = m._gate_inputs(q, torch.full((8, 3), 1. / 3.), torch.zeros(8))
        slots = q.reshape(-1, 2, 8)
        raw_centroid = slots.mean(1, keepdim=True)
        centred = (slots - slots.mean(1, keepdim=True)) / (slots.std(1, keepdim=True) + 1e-6)
        self.assertGreater(float(raw_centroid.norm(dim=-1).min()), .5,
                           "the raw block centroid must be a real direction")
        self.assertLess(float(centred.mean(1, keepdim=True).norm(dim=-1).max()), 1e-5,
                        "the z-scored block centroid is the zero vector")
        self.assertGreater(float(raw[..., 2].abs().mean()), 1e-2,
                           "raw-slot redundancy must carry real signal")


class SimilarityMargin(unittest.TestCase):
    """The sealed plan says margin .01; the built module must actually use it."""

    def test_built_v6_modules_use_the_plan_margin(self):
        from methods.prlmc.gated_slot_merge import SIM_MARGIN_V6
        self.assertEqual(SIM_MARGIN_V6, .01)
        for arch in ("grouped_gated_sparse_slot_v6", "grouped_gated_sparse_native_v6"):
            self.assertEqual(float(build_merge(arch, 32, 8, 2).sim_margin), .01)
        self.assertIn("margin .01", TRAIN_V6["variants"]["DsupV6"]["routing_score"])
        self.assertIn("margin .01", TRAIN_V6["variants"]["DnativeV6"]["routing_score"])

    def test_margin_reaches_the_mask(self):
        m = build_merge("grouped_gated_sparse_slot_v6", 32, 8, 2)
        self.assertEqual(m.sim_margin, .01)
        q = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
        k = torch.cat([q[:2] - .01, q[2:] + .01])
        strict = model(sim_margin=0.)._similarity_mask(q, k, 2)
        forgiving = m._similarity_mask(q, k, 2)
        # reject = best < peer_mean - margin: a larger margin lowers the bar, so
        # the conservative mask keeps at least as many slots as margin 0.
        self.assertGreaterEqual(float(forgiving.sum()), float(strict.sum()))
        self.assertGreater(float(forgiving.sum()), 0., "the mask must not reject everything")

    def test_checkpoint_margin_mismatch_is_refused(self):
        from natural_merge_runtime import check_similarity_margin
        merge = build_merge("grouped_gated_sparse_slot_v6", 32, 8, 2)
        check_similarity_margin(dict(sim_margin=.01), merge)
        check_similarity_margin({}, merge)  # legacy checkpoints carry no record
        with self.assertRaises(ValueError):
            check_similarity_margin(dict(sim_margin=0.), merge)


if __name__ == "__main__":
    unittest.main(verbosity=2)
