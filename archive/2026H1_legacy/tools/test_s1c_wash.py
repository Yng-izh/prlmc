"""Tests of the S1c-basis-wash gates, bases and displacement shapes."""
import unittest

import torch
from torch.nn.utils import parametrize

from s1c_wash import (make_arm_displacement, margin_gate, principal_angles,
                      random_basis, wash_gate)


def entry(protect_harmed=0, net=9, rescue=15):
    return {"protect_harmed": protect_harmed, "net_vs_select": net,
            "rescue_recovered": rescue}


class BasisWashTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_random_basis_reproducible_and_orthonormal(self):
        first = random_basis(256, 13, 0, torch)
        second = random_basis(256, 13, 0, torch)
        other = random_basis(256, 13, 1, torch)
        self.assertEqual(first.shape, (256, 13))
        torch.testing.assert_close(first, second)
        self.assertGreater(float((first - other).abs().max()), 0.1)
        gram = first.T @ first
        torch.testing.assert_close(gram, torch.eye(13, dtype=torch.float64),
                                   rtol=0, atol=1e-10)

    def test_principal_angles_self_and_disjoint(self):
        a = random_basis(256, 13, 0, torch)
        b = random_basis(256, 13, 1, torch)
        self.assertLess(float(principal_angles(a, a, torch).max()), 1e-6)
        angles = principal_angles(a, b, torch)
        for value in angles.tolist():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, torch.pi / 2 + 1e-9)
        self.assertGreater(float(angles.min()), 0.9)

    def test_wash_gate_pass_fail_and_margin(self):
        self.assertEqual(wash_gate({"corrected_ul": entry(), "random_13d": entry(net=3, rescue=8)}), "PASS")
        self.assertEqual(wash_gate({"corrected_ul": entry(net=7), "random_13d": entry(net=3, rescue=8)}), "FAIL")
        self.assertEqual(wash_gate({"corrected_ul": entry(net=9, rescue=12),
                                    "random_13d": entry(net=5, rescue=9)}), "MARGIN")
        self.assertEqual(wash_gate({"corrected_ul": entry(net=9, rescue=11),
                                    "random_13d": entry(net=8, rescue=10)}), "FAIL")
        self.assertEqual(wash_gate({"corrected_ul": entry(protect_harmed=2),
                                    "random_13d": entry(net=3, rescue=8)}), "MARGIN")

    def test_margin_gate_uses_best_random_draw(self):
        draws = [entry(net=5, rescue=12), entry(net=6, rescue=11)]
        result = margin_gate(entry(net=9, rescue=15), draws)
        self.assertEqual(result["rescue_gap"], 3)
        self.assertEqual(result["net_gap"], 3)
        self.assertTrue(result["resolved_pass"])
        weak = margin_gate(entry(net=9, rescue=13), draws)
        self.assertEqual(weak["rescue_gap"], 1)
        self.assertFalse(weak["resolved_pass"])

    def test_displacement_shapes_identity_and_clamp(self):
        weight = torch.randn(32, 4096)
        basis = random_basis(weight.numel(), 13, 0, torch).to(weight.dtype)
        low = make_arm_displacement(weight, basis, torch)
        self.assertEqual(low.value.numel(), 13)
        self.assertTrue(torch.equal(low.forward(weight), weight))
        with torch.no_grad():
            low.value.normal_()
            low.clamp()
        self.assertLessEqual(float(low.value.norm()), 1.0 + 1e-6)
        full = make_arm_displacement(weight, None, torch)
        self.assertEqual(full.value.shape, weight.shape)
        self.assertTrue(torch.equal(full.forward(weight), weight))
        with torch.no_grad():
            full.normal = None
            full.value.fill_(2.0)
            full.clamp()
        self.assertLessEqual(float(full.value.norm()), 1.0 + 1e-6)

    def test_displacement_gradient_flows_through_basis(self):
        weight = torch.randn(8, 16)
        basis = random_basis(weight.numel(), 5, 2, torch).to(weight.dtype)
        layer = torch.nn.Linear(16, 8, bias=False)
        layer.weight.data = weight.clone()
        layer.weight.requires_grad_(False)
        displacement = make_arm_displacement(layer.weight, basis, torch)
        parametrize.register_parametrization(layer, "weight", displacement)
        loss = layer(torch.randn(3, 16)).square().sum()
        loss.backward()
        self.assertGreater(float(displacement.value.grad.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()
