"""Tests of the S1c geometry audit's analyze() math on synthetic rows."""
import unittest

import torch

import s1c_geometry_audit as G


def build_cache(ul, rescue_scale=2.0, protect_vec=None, noise=1e-4, seed=5):
    torch.manual_seed(seed)
    rows, cohort_of = {}, {}
    for i in range(36):
        q = "r%02d" % i
        rows[q] = (ul[:, 0] * rescue_scale + noise * torch.randn(ul.shape[0], dtype=torch.float64)).float()
        cohort_of[q] = "rescue"
    for i in range(31):
        q = "p%02d" % i
        rows[q] = (protect_vec + noise * torch.randn(ul.shape[0], dtype=torch.float64)).float()
        cohort_of[q] = "protect"
    for i in range(44):
        q = "t%02d" % i
        rows[q] = (noise * torch.randn(ul.shape[0], dtype=torch.float64)).float()
        cohort_of[q] = "repair"
    return {"basis_sha256": "x" * 64, "checkpoint_sha256": "y" * 64,
            "rows": rows, "cohort_of": cohort_of}


class GeometryAnalyzeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.dim = 4096
        self.ul, _ = torch.linalg.qr(torch.randn(self.dim, 13, dtype=torch.float64))
        self.basis_file = {"corrected_ul": self.ul.float()}
        self._real_load = torch.load
        self._real_digest = G.digest

    def _patch(self, cache):
        def fake_load(path, map_location=None, weights_only=False):
            return cache if str(path).endswith("rows.pt") else self.basis_file
        torch.load = fake_load
        G.digest = lambda p: "x" * 64

    def tearDown(self):
        torch.load = self._real_load
        G.digest = self._real_digest

    def test_orthogonal_protect_reads_decoupling(self):
        torch.manual_seed(7)
        outside = torch.randn(self.dim, dtype=torch.float64)
        outside = outside - self.ul @ (self.ul.T @ outside)
        cache = build_cache(self.ul, protect_vec=outside * 3.0)
        self._patch(cache)
        result = G.analyze("rows.pt", "basis.pt")
        self.assertEqual(result["cohort_counts"],
                         {"rescue": 36, "protect": 31, "repair": 44})
        self.assertGreater(result["projected_energy_fraction"]["rescue"], 0.99)
        self.assertLess(result["projected_energy_fraction"]["protect"], 0.01)
        self.assertLess(abs(result["rescue_protect_cosine"]), 0.2)
        self.assertTrue(result["pre_registered_reading"].startswith("decoupling"))

    def test_co_directional_protect_reads_shared_benefit(self):
        cache = build_cache(self.ul, protect_vec=self.ul[:, 0] * 1.0)
        self._patch(cache)
        result = G.analyze("rows.pt", "basis.pt")
        self.assertGreater(result["rescue_protect_cosine"], 0.9)
        self.assertGreater(result["conflict_energy_value"], 0.9)
        self.assertTrue(result["pre_registered_reading"].startswith("co-directional"))

    def test_per_question_records_split_norms(self):
        torch.manual_seed(11)
        outside = torch.randn(self.dim, dtype=torch.float64)
        outside = outside - self.ul @ (self.ul.T @ outside)
        cache = build_cache(self.ul, protect_vec=outside * 3.0)
        self._patch(cache)
        result = G.analyze("rows.pt", "basis.pt")
        self.assertEqual(len(result["per_question"]), 111)
        entry = next(e for e in result["per_question"] if e["cohort"] == "rescue")
        self.assertAlmostEqual(
            entry["grad_norm"],
            (entry["proj_norm"] ** 2 + entry["residual_norm"] ** 2) ** 0.5,
            places=3)
        self.assertGreaterEqual(entry["projected_fraction"], 0.0)
        self.assertLessEqual(entry["projected_fraction"], 1.0)

    def test_conflict_energy_is_abs_cosine_of_projected_means(self):
        cache = build_cache(self.ul, protect_vec=self.ul[:, 0] * -1.0)
        self._patch(cache)
        result = G.analyze("rows.pt", "basis.pt")
        # anti-aligned means: signed cos < 0, |cos| (conflict) high
        self.assertLess(result["rescue_protect_cosine"], -0.9)
        self.assertGreater(result["conflict_energy_value"], 0.9)
        self.assertTrue(result["pre_registered_reading"].startswith("decoupling"))


if __name__ == "__main__":
    unittest.main()
