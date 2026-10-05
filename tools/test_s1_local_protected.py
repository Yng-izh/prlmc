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

    def test_no_protection_and_nonfinite(self):
        ul = orth(torch.randn(3, 12))
        safe, _ = protect(ul, torch.empty(12, 0, dtype=torch.float64))
        self.assertTrue(torch.allclose(safe @ safe.T, ul @ ul.T, atol=1e-10))
        with self.assertRaises(ValueError):
            orth(torch.full((2, 12), float("nan")))


if __name__ == "__main__":
    unittest.main()
