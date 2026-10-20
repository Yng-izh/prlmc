"""CPU contracts for v7 C5/C9: bounded tanh writes and the sparse high-SNR gate.

Nothing here is research evidence: the fixtures are miniature readers used to
pin the *mechanism* -- zero-init identity, boundedness, the shape of the sparse
mask, and the untouched global RMS budget.
"""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from methods.prlmc.consolidate import _DELTA_VAR_EPS
from methods.prlmc.gated_slot_merge import (
    GraftedSlotMemoryMerge, GraftedSlotMemoryMergeV6, GraftedSlotMemoryMergeV7,
    NativeSkipSlotMergeV6, SupervisedDSlotMergeV6, SupervisedDSlotMergeV7,
    _active_match_weight, _soft_bound,
)
from methods.prlmc.native_skip_merge import (
    NativeSkipSlotMerge, NativeSkipSlotMergeV7, SupervisedDSlotMerge,
)

LAYERS, HIDDEN, BLOCK, RANK, GROUPS = 4, 8, 2, 3, 2
SLOTS = 38
CARRIERS = list(range(10, 19))


def v5(**kwargs):
    return GraftedSlotMemoryMerge(LAYERS, HIDDEN, block=BLOCK, rank=RANK, num_groups=GROUPS,
                                  reference_layer=2, topk=3, **kwargs)


def v7(**kwargs):
    return GraftedSlotMemoryMergeV7(LAYERS, HIDDEN, block=BLOCK, rank=RANK, num_groups=GROUPS,
                                    reference_layer=2, topk=3, **kwargs)


def v7_native(**kwargs):
    return NativeSkipSlotMergeV7(LAYERS, HIDDEN, block=BLOCK, rank=RANK, num_groups=GROUPS,
                                 reference_layer=2, topk=3, **kwargs)


def selection(source, target=18, block=BLOCK):
    """The Selection output the modules must reproduce at step 0."""
    return source[:, SLOTS - target:].contiguous()


class SoftBound(unittest.TestCase):
    """`_soft_bound`: bounded, exact at zero, unit gain at the origin."""

    def test_zero_delta_is_exactly_zero(self):
        carrier = torch.randn(18, HIDDEN)
        out = _soft_bound(torch.zeros_like(carrier), carrier)
        self.assertTrue(torch.equal(out, torch.zeros_like(carrier)))

    def test_bound_is_the_carrier_rms(self):
        torch.manual_seed(4)
        carrier = torch.randn(18, HIDDEN)
        rms = carrier.square().mean(-1, keepdim=True).sqrt()
        delta = torch.randn(18, HIDDEN) * 1e3
        out = _soft_bound(delta, carrier)
        self.assertTrue(bool((out.abs() <= rms * (1 + 1e-6)).all()))
        # Saturation, not rescaling: a huge delta lands on the bound.
        self.assertTrue(bool((out.abs() > .9 * rms).all()))

    def test_tiny_delta_is_not_amplified(self):
        """The old unit-RMS form mapped 1e-6*rms onto rms (a 1e6 gain)."""
        torch.manual_seed(5)
        carrier = torch.randn(18, HIDDEN)
        rms = carrier.square().mean(-1, keepdim=True).sqrt()
        delta = 1e-6 * rms
        new = _soft_bound(delta, carrier)
        old = (delta * torch.rsqrt(delta.square().mean(-1, keepdim=True) + _DELTA_VAR_EPS) * rms)
        self.assertLess(float(((new - delta).abs() / delta.abs()).max()), 1e-3)
        # Unit gain at the origin: a 1e-6*rms residual stays a 1e-6*rms residual.
        self.assertLess(abs(float(new.abs().mean() / delta.abs().mean()) - 1.), 1e-3)
        # The retired form multiplied the same residual by 1/sqrt(eps) = 1e3.
        self.assertGreater(float(old.abs().mean() / delta.abs().mean()), 900.)

    def test_zero_carrier_stays_finite_and_small(self):
        carrier = torch.zeros(4, HIDDEN)
        out = _soft_bound(torch.randn(4, HIDDEN) * 1e6, carrier)
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertTrue(bool((out.abs() <= _DELTA_VAR_EPS).all()))

    def test_origin_gain_is_one(self):
        carrier = torch.randn(6, HIDDEN)
        delta = torch.zeros(6, HIDDEN, requires_grad=True)
        _soft_bound(delta, carrier).sum().backward()
        self.assertTrue(torch.allclose(delta.grad, torch.ones_like(delta), atol=1e-6))


class SparseGate(unittest.TestCase):
    """`_active_match_weight` and its placement in `_slot_alpha`."""

    def test_background_slots_are_silent(self):
        weight = _active_match_weight(torch.tensor([-3.]))
        self.assertAlmostEqual(float(weight), 1.1e-5, places=6)
        self.assertLess(float(weight), 1e-4)

    def test_high_snr_slots_pass(self):
        self.assertGreater(float(_active_match_weight(torch.tensor([2.]))), .999)

    def test_weight_is_monotone_and_detached(self):
        z = torch.linspace(-4, 4, 9, requires_grad=True)
        weight = _active_match_weight(z.detach())
        self.assertTrue(bool((weight[1:] > weight[:-1]).all()))
        self.assertIsNone(weight.grad_fn)
        self.assertTrue(bool(((weight > 0) & (weight < 1)).all()))

    def test_alpha_clamped_and_background_zeroed(self):
        module = v7()
        slots = 9
        gate_logits = torch.full((slots, GROUPS), 50.)  # base gate wide open
        rel = torch.zeros(slots, 3)
        rel[:, 0] = torch.linspace(-4., 4., slots)  # affinity_z is column 0
        alpha = module._slot_alpha(gate_logits, rel)
        self.assertEqual(alpha.shape, (slots, GROUPS))
        self.assertTrue(bool((alpha >= 0).all() and (alpha <= module.gamma_max).all()))
        # The most negative z slot is effectively off; the most positive is on.
        self.assertLess(float(alpha[0].max()), 1e-3 * module.gamma_max)
        self.assertGreater(float(alpha[-1].min()), .99 * module.gamma_max)
        # A saturating gate logit cannot escape the local bound either.
        extreme = module._slot_alpha(torch.full((slots, GROUPS), 200.),
                                     torch.stack((torch.full((slots,), 50.),
                                                  torch.zeros(slots), torch.zeros(slots)), -1))
        self.assertTrue(bool((extreme <= module.gamma_max).all()))
        self.assertTrue(bool((extreme >= 0).all()))

    def test_v6_gate_has_no_sparse_mask(self):
        """v6 is frozen: its alpha must not depend on the affinity feature."""
        module = GraftedSlotMemoryMergeV6(LAYERS, HIDDEN, block=BLOCK, rank=RANK,
                                          num_groups=GROUPS, reference_layer=2, topk=3)
        slots = 7
        gate_logits = torch.zeros(slots, GROUPS)
        low = torch.zeros(slots, 3); low[:, 0] = -3.
        high = torch.zeros(slots, 3); high[:, 0] = 3.
        self.assertTrue(torch.equal(module._slot_alpha(gate_logits, low),
                                    module._slot_alpha(gate_logits, high)))

    def test_mask_is_detached_inside_a_forward(self):
        torch.manual_seed(6)
        module = v7()
        with torch.no_grad():
            module.up_weight.normal_()
        source = torch.randn(LAYERS, SLOTS, HIDDEN)
        module(source, target=18, carrier_indices=CARRIERS)
        self.assertIsNotNone(module._active_weight)
        self.assertIsNone(module._active_weight.grad_fn)
        fusion = module.last_fusion
        self.assertIn("sparse_gate_form", fusion)
        self.assertGreaterEqual(fusion["active_slot_fraction"], 0.)
        self.assertLessEqual(fusion["active_slot_fraction"], 1.)
        self.assertGreaterEqual(fusion["alpha_active_mean"], 0.)
        self.assertLessEqual(fusion["alpha_active_mean"], module.gamma_max)
        self.assertIn("no forced unit-RMS", fusion["soft_delta_bound"])


class StepZeroIdentity(unittest.TestCase):
    """up_weight / native_gain zero init must reproduce Selection bit-exactly."""

    def setUp(self):
        torch.manual_seed(11)
        self.source = torch.randn(LAYERS, SLOTS, HIDDEN)

    def check(self, module):
        for dtype in (torch.float32, torch.float16):
            source = self.source.to(dtype)
            out = module(source, target=18, carrier_indices=CARRIERS)
            self.assertTrue(torch.equal(out, selection(source)), f"{type(module).__name__} {dtype}")
            self.assertTrue(torch.equal(module(source, target=SLOTS), source))

    def test_v5_lineage_identity(self):
        self.check(v5())
        self.check(SupervisedDSlotMerge(LAYERS, HIDDEN, block=BLOCK, rank=RANK, num_groups=GROUPS,
                                        reference_layer=2, topk=3))
        self.check(NativeSkipSlotMerge(LAYERS, HIDDEN, block=BLOCK, rank=RANK, num_groups=GROUPS,
                                       reference_layer=2, topk=3))

    def test_v7_lineage_identity(self):
        self.check(v7())
        self.check(SupervisedDSlotMergeV7(LAYERS, HIDDEN, block=BLOCK, rank=RANK, num_groups=GROUPS,
                                          reference_layer=2, topk=3))
        self.check(v7_native())
        # The native branch is exactly zero at native_gain=0.
        module = v7_native()
        self.assertTrue(torch.equal(module.native_gain, torch.zeros(GROUPS)))

    def test_v5_lineage_reports_the_sparse_gate(self):
        """The shared base carries both v7 mechanisms, so v5 reports them too."""
        module = v5()
        with torch.no_grad():
            module.up_weight.normal_()
        source = torch.randn(LAYERS, SLOTS, HIDDEN)
        module(source, target=18, carrier_indices=CARRIERS)
        self.assertIsNotNone(module._active_weight)
        self.assertIn("active_slot_fraction", module.last_fusion)
        self.assertIn("soft_delta_bound", module.last_fusion)
        self.assertIn("no forced unit-RMS", module.last_fusion["soft_delta_bound"])

    def test_masked_out_sources_degrade_to_selection(self):
        for module in (v5(), v7(), v7_native()):
            with torch.no_grad():
                module.up_weight.normal_()
                if hasattr(module, "native_gain"):
                    module.native_gain.fill_(1.)
            # Only the carriers survive the mask: no fusion source is left, so
            # the output is Selection again.
            mask = torch.zeros(19, dtype=torch.bool)
            mask[CARRIERS] = True
            out = module(self.source, target=18, carrier_indices=CARRIERS, source_allow_mask=mask)
            self.assertTrue(torch.equal(out, selection(self.source)))
            # Masking a carrier out is a caller error, not a silent degradation.
            with self.assertRaises(ValueError):
                module(self.source, target=18, carrier_indices=CARRIERS,
                       source_allow_mask=torch.zeros(19, dtype=torch.bool))

    def test_masked_sources_cannot_affect_the_output(self):
        module = v7_native()
        with torch.no_grad():
            module.up_weight.normal_(); module.native_gain.fill_(1.)
        mask = torch.zeros(19, dtype=torch.bool); mask[CARRIERS] = True; mask[2] = True
        changed = self.source.clone(); changed[:, :2 * BLOCK] *= 1e3
        self.assertTrue(torch.equal(
            module(self.source, target=18, carrier_indices=CARRIERS, source_allow_mask=mask),
            module(changed, target=18, carrier_indices=CARRIERS, source_allow_mask=mask)))


class BoundedWrite(unittest.TestCase):
    """The write stays inside the carrier scale, and `_cap` is not loosened."""

    def setUp(self):
        torch.manual_seed(13)
        self.source = torch.randn(LAYERS, SLOTS, HIDDEN)

    def test_delta_is_bounded_by_carrier_rms(self):
        for module in (v5(), v7()):
            torch.manual_seed(14)
            blocks = self.source.reshape(LAYERS, 19, BLOCK, HIDDEN)
            carrier_blocks = blocks.index_select(1, torch.tensor(CARRIERS))
            for scale in (1., 1e3):
                with torch.no_grad():
                    module.up_weight.mul_(0).add_(torch.randn_like(module.up_weight) * scale)
                probabilities = torch.softmax(torch.randn(18, 3), -1)
                indices = torch.randint(0, SLOTS, (18, 3))
                delta = module._group_delta(probabilities, indices, blocks[0], carrier_blocks[0],
                                            module.down_weight[0], module.up_weight[0])
                rms = carrier_blocks[0].float().flatten(0, 1).square().mean(-1, keepdim=True).sqrt()
                self.assertTrue(bool((delta.abs() <= rms * (1 + 1e-6)).all()), f"scale {scale}")

    def test_global_budget_is_not_relaxed(self):
        for module in (v5(), v7(), v7_native(),
                       NativeSkipSlotMerge(LAYERS, HIDDEN, block=BLOCK, rank=RANK,
                                           num_groups=GROUPS, reference_layer=2, topk=3)):
            with torch.no_grad():
                module.up_weight.normal_()
                module.slot_gate.bias.fill_(10.)
                if hasattr(module, "native_gain"):
                    module.native_gain.fill_(1e3)
            out = module(self.source, target=18, carrier_indices=CARRIERS)
            reference = selection(self.source).float()
            ratio = (out.float() - reference).norm() / reference.norm()
            self.assertLessEqual(float(ratio), .050001, type(module).__name__)
            self.assertGreater(float(ratio), 0.)
            self.assertLessEqual(module.last_residual_ratio, .050001)
            self.assertLessEqual(module.last_fusion["residual_ratio"], .050001)
            self.assertEqual(module.global_rms_budget, .05)

    def test_native_gain_is_tanh_bounded(self):
        module = v7_native()
        with torch.no_grad():
            module.up_weight.normal_()
            module.native_gain.fill_(1e3)  # tanh saturates to 1 instead of 1e3
        out = module(self.source, target=18, carrier_indices=CARRIERS)
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertLessEqual(module.last_residual_ratio, .050001)
        # The recorded gain is the raw parameter; the applied one is tanh of it.
        self.assertEqual(module.last_fusion["native_gain"], [1e3] * GROUPS)

    def test_zero_init_gradients_are_finite(self):
        for module in (v5(), v7(), v7_native()):
            module.zero_grad()
            out = module(self.source, target=18, carrier_indices=CARRIERS)
            (out * torch.randn_like(out)).sum().backward()
            for name, parameter in module.named_parameters():
                if parameter.grad is None:
                    continue
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()), f"{type(module).__name__}.{name}")
            self.assertGreater(float(module.up_weight.grad.abs().sum()), 0.)


class V7Classes(unittest.TestCase):
    """Registry-facing surface the t6 wiring depends on (C6)."""

    def test_architecture_names(self):
        self.assertEqual(GraftedSlotMemoryMergeV7.architecture, "grouped_gated_sparse_slot_v7")
        self.assertEqual(SupervisedDSlotMergeV7.architecture, "grouped_gated_sparse_slot_v7")
        self.assertEqual(NativeSkipSlotMergeV7.architecture, "grouped_gated_sparse_native_v7")
        self.assertEqual(GraftedSlotMemoryMergeV6.architecture, "grouped_gated_sparse_slot_v6")
        self.assertEqual(NativeSkipSlotMergeV6.architecture, "grouped_gated_sparse_native_v6")
        self.assertEqual(NativeSkipSlotMerge.architecture, "grouped_gated_sparse_native_v5")

    def test_inheritance_and_geometry(self):
        module = v7_native()
        self.assertIsInstance(module, SupervisedDSlotMergeV7)
        self.assertIsInstance(module, GraftedSlotMemoryMergeV7)
        self.assertEqual(module.gamma_max, .20)
        self.assertEqual(module.num_groups, GROUPS)
        self.assertEqual(module.top_sharp, 2)
        self.assertEqual(module.native_gain.shape, (GROUPS,))
        self.assertTrue(torch.equal(module.up_weight, torch.zeros_like(module.up_weight)))

    def test_v7_and_v6_do_not_share_the_write_path(self):
        """Same weights, different delta: the tanh bound is not the v6 clamp."""
        torch.manual_seed(21)
        source = torch.randn(LAYERS, SLOTS, HIDDEN)
        blocks = source.reshape(LAYERS, 19, BLOCK, HIDDEN)
        carrier_blocks = blocks.index_select(1, torch.tensor(CARRIERS))
        probabilities = torch.softmax(torch.randn(18, 3), -1)
        indices = torch.randint(0, SLOTS, (18, 3))
        seven = v7()
        six = GraftedSlotMemoryMergeV6(LAYERS, HIDDEN, block=BLOCK, rank=RANK,
                                       num_groups=GROUPS, reference_layer=2, topk=3)
        with torch.no_grad():
            seven.up_weight.normal_()
        six.load_state_dict(seven.state_dict(), strict=True)
        bound = seven._group_delta(probabilities, indices, blocks[0], carrier_blocks[0],
                                   seven.down_weight[0], seven.up_weight[0])
        clamped = six._group_delta(probabilities, indices, blocks[0], carrier_blocks[0],
                                   six.down_weight[0], six.up_weight[0])
        rms = carrier_blocks[0].float().flatten(0, 1).square().mean(-1, keepdim=True).sqrt()
        # tanh bounds every element; the v6 clamp bounds the per-slot RMS, so a
        # single coordinate may still exceed it. Same weights, different write.
        self.assertTrue(bool((bound.abs() <= rms * (1 + 1e-6)).all()))
        self.assertFalse(torch.allclose(bound, clamped))


if __name__ == "__main__":
    unittest.main()
