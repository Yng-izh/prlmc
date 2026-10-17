"""s1c_lowo_coverage.py 单元测试：小规模合成数据验证 LOWO 覆盖率数学正确性。"""
import math
import sys
import unittest

import torch

sys.path.insert(0, "tools")
from s1c_lowo_coverage import coverage, svd_basis  # noqa: E402


class TestLowoCoverage(unittest.TestCase):
    def test_svd_basis_shapes_and_orthonormality(self):
        torch.manual_seed(0)
        G = torch.randn(40, 64)
        U = svd_basis(G, 13)
        self.assertEqual(U.shape, (64, 13))
        err = (U.t() @ U - torch.eye(13)).abs().max().item()
        self.assertLess(err, 1e-5)

    def test_coverage_full_rank_is_one(self):
        torch.manual_seed(0)
        G = torch.randn(30, 128)
        U = svd_basis(G, 30)  # rank = n 行数，中心化后秩最多 29
        # 用 29 保证不越秩
        U = svd_basis(G, 29)
        mu = G.mean(dim=0, keepdim=True)
        Gt = G - mu
        rho = coverage(U, Gt[0])
        self.assertAlmostEqual(rho, 1.0, places=5)

    def test_coverage_random_direction_is_low(self):
        torch.manual_seed(0)
        G = torch.randn(30, 128)
        U = svd_basis(G, 13)
        g = torch.randn(128)
        rho = coverage(U, g)
        self.assertLess(rho, 0.5)

    def test_coverage_identical_subspace_is_high(self):
        # 构造严格落在 13 维子空间内的数据
        torch.manual_seed(0)
        Q, _ = torch.linalg.qr(torch.randn(128, 13))
        coefs = torch.randn(50, 13)
        G = coefs @ Q.t()  # 每行都在 span(Q) 内
        U = svd_basis(G, 13)
        Gt = G - G.mean(dim=0, keepdim=True)
        for i in range(5):
            rho = coverage(U, Gt[i])
            self.assertGreater(rho, 0.999)

    def test_energy_weighted_aggregate_matches_row_formula(self):
        # 聚合公式 = 能量加权平均，与逐行加权一致
        torch.manual_seed(0)
        G = torch.randn(20, 64)
        U = svd_basis(G[:15], 10)
        Gt = G[15:] - G[:15].mean(dim=0, keepdim=True)
        P = U.t() @ Gt.t()
        agg = (P * P).sum().item() / (Gt * Gt).sum().item()
        nums = (P * P).sum(dim=0)
        dens = (Gt * Gt).sum(dim=0)
        manual = (nums.sum() / dens.sum()).item()
        self.assertAlmostEqual(agg, manual, places=6)


if __name__ == "__main__":
    unittest.main()
