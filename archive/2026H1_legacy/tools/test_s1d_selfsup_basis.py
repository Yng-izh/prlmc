#!/usr/bin/env python
"""Unit tests for s1d_selfsup_basis: coverage math, orthogonality, aggregation."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from s1d_selfsup_basis import energy_coverage, orth


def _orthonormal_cols(U):
    return torch.allclose(U.t() @ U, torch.eye(U.shape[1], dtype=U.dtype),
                          atol=1e-9)


class TestEnergyCoverage(unittest.TestCase):
    def test_full_rank_covers_all_energy(self):
        torch.manual_seed(0)
        G = torch.randn(20, 64, dtype=torch.float64)
        U = orth(G)                       # spans G's row space entirely
        self.assertAlmostEqual(energy_coverage(U, G), 1.0, places=9)

    def test_zero_overlap_covers_nothing(self):
        torch.manual_seed(0)
        G = torch.randn(20, 64, dtype=torch.float64)
        # random rows with G's row space projected out -> null-space basis
        X = torch.randn(44, 64, dtype=torch.float64)
        Q = orth(G)                       # (64, 20) spans G's rows
        X = X - (X @ Q) @ Q.t()
        U = orth(X)[:, :13]
        self.assertLess(energy_coverage(U, G), 1e-9)

    def test_energy_weighted_matches_rowwise(self):
        torch.manual_seed(1)
        G = torch.randn(37, 128, dtype=torch.float64)
        U = orth(torch.randn(128, 128, dtype=torch.float64))[:, :13]
        agg = energy_coverage(U, G)
        num = sum(float((U.t() @ g) @ (U.t() @ g)) for g in G)
        den = sum(float(g @ g) for g in G)
        self.assertAlmostEqual(agg, num / den, places=9)

    def test_rank_slice_subset_of_full_basis(self):
        # rank-13 slice of a basis covers no more than the full basis
        torch.manual_seed(2)
        G = torch.randn(25, 96, dtype=torch.float64)
        full = orth(G)
        r13 = full[:, :13]
        self.assertGreaterEqual(energy_coverage(full, G),
                                energy_coverage(r13, G))


class TestOrth(unittest.TestCase):
    def test_columns_orthonormal(self):
        torch.manual_seed(3)
        rows = torch.randn(30, 128, dtype=torch.float64)
        U = orth(rows)
        self.assertTrue(_orthonormal_cols(U))
        self.assertEqual(U.shape, (128, 30))

    def test_span_preserved(self):
        # orth keeps the column space: projecting rows onto U recovers them
        torch.manual_seed(4)
        rows = torch.randn(15, 80, dtype=torch.float64)
        U = orth(rows)
        # rows as columns live in span(U): residual after projection ~ 0
        resid = rows.t() - U @ (U.t() @ rows.t())
        self.assertLess(resid.norm(), 1e-9)

    def test_more_rows_than_dim_raises_nothing(self):
        rows = torch.randn(140, 128, dtype=torch.float64)  # rank <= 128
        U = orth(rows)
        self.assertTrue(_orthonormal_cols(U[:, :128]))
        self.assertEqual(U.shape, (128, 128))


if __name__ == "__main__":
    unittest.main()
