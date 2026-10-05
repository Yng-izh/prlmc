"""Numerical contract tests for S1-local protected parameter directions."""
import unittest
import torch
from s1_local_protected import orth, protect


class ProtectedBasisTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_orthonormality_and_row_span(self):
        rows = torch.randn(7, 53)
        u = orth(rows)
        self.assertTrue(torch.allclose(u.T @ u, torch.eye(7, dtype=u.dtype), atol=1e-10))
        residual = rows.double().T - u @ (u.T @ rows.double().T)
        self.assertLess(float(residual.norm()), 1e-10)

    def test_protection_and_projected_span(self):
        ul, uk = orth(torch.randn(5, 53)), orth(torch.randn(9, 53))
        safe, energies = protect(ul, uk)
        self.assertLess(float((uk.T @ safe).norm()), 1e-10)
        self.assertTrue(torch.allclose(safe.T @ safe, torch.eye(5, dtype=safe.dtype), atol=1e-10))
        projected = ul - uk @ (uk.T @ ul)
        self.assertLess(float((projected - safe @ (safe.T @ projected)).norm()), 1e-10)
        self.assertTrue(bool((energies <= 1 + 1e-10).all()))

    def test_fully_overlapping_is_empty(self):
        ul = orth(torch.randn(5, 53))
        safe, _ = protect(ul, ul)
        self.assertEqual(safe.shape, (53, 0))

    def test_partial_overlap_drops_null_direction(self):
        ul = torch.eye(8, dtype=torch.float64)[:, :4]
        uk = ul[:, :2]
        safe, _ = protect(ul, uk)
        self.assertEqual(safe.shape, (8, 2))
        self.assertLess(float((uk.T @ safe).norm()), 1e-10)

    def test_oracle_checkpoint_dispatch_and_gradient(self):
        import os
        from unittest.mock import patch
        import torch.utils.checkpoint as checkpoint
        os.environ["S0B_GPU_SET"] = "1"
        from s0b_coefficient_oracle import OracleCompactor
        from methods.prlmc.consolidate import CarrierResidualCompactor, FixedBasisCarrierResidualCompactor
        from methods.prlmc.functional_subspace import CoefficientOracle
        warm = CarrierResidualCompactor(num_layers=2, hidden=8, block=2, rank=2)
        with torch.no_grad():
            warm.up.weight.normal_(std=0.1)
        arm = FixedBasisCarrierResidualCompactor.from_warmup(warm)
        for parameter in arm.parameters():
            parameter.requires_grad_(False)
        oracle = CoefficientOracle(2, max_deviation=0.5)
        arm.add_module("_s1_oracle", oracle)
        wrapper = OracleCompactor(arm, oracle)
        arm._layer_delta = wrapper._layer_delta
        with patch.object(checkpoint, "checkpoint", wraps=checkpoint.checkpoint) as called:
            result = wrapper(torch.randn(2, 8, 8), target=4, block=2)
            self.assertEqual(called.call_count, 2)
            result.sum().backward()
        self.assertIsNotNone(oracle.residual.grad)
        self.assertGreater(float(oracle.residual.grad.norm()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in arm.parameters()
                            if parameter is not oracle.residual))

    def test_no_protection_and_nonfinite(self):
        ul = orth(torch.randn(3, 12))
        safe, _ = protect(ul, torch.empty(12, 0, dtype=torch.float64))
        self.assertTrue(torch.allclose(safe @ safe.T, ul @ ul.T, atol=1e-10))
        with self.assertRaises(ValueError):
            orth(torch.full((2, 12), float("nan")))


if __name__ == "__main__":
    unittest.main()
