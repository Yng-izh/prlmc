"""Focused SIC tests; run with python3 tools/test_sic_selection.py."""

import importlib.util
from pathlib import Path
import unittest

import torch


# Load the pure tensor module without importing optional reader dependencies.
_SPEC = importlib.util.spec_from_file_location(
    "sic_under_test", Path(__file__).resolve().parents[1] / "methods/prlmc/sic.py"
)
sic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sic)


class OptimizationStatisticsTests(unittest.TestCase):
    def test_exact_percentage_boundaries(self):
        from sic_optimize import reduction_gate
        self.assertTrue(reduction_gate(10, 9, 10))
        self.assertTrue(reduction_gate(10, 7, 30))
        self.assertFalse(reduction_gate(11, 10, 10))
        self.assertFalse(reduction_gate(0, 0, 10))
        self.assertFalse(reduction_gate(10, 11, 10))

    def test_exact_pairing_and_eligibility(self):
        from sic_optimize import stats
        records = [dict(eligible=True, arms=dict(select=False, candidate=True)) for _ in range(6)]
        records.append(dict(eligible=False, arms=dict(select=True, candidate=False)))
        result = stats(records, "candidate")
        self.assertEqual(result["n"], 6)
        self.assertEqual(result["net"], 6)
        self.assertEqual(result["exact_mcnemar_p"], 0.03125)
        self.assertEqual(stats(records, "candidate", False)["harmed"], 1)

    def test_no_discordance(self):
        from sic_optimize import stats
        result = stats([dict(eligible=True, arms=dict(select=True, candidate=True))], "candidate")
        self.assertIsNone(result["reduction"])
        self.assertEqual(result["exact_mcnemar_p"], 1.)


class NormalizeProbeScoresTests(unittest.TestCase):
    def test_l1_and_per_probe_scale_invariance(self):
        scores = torch.tensor([[2., 3., 5.], [1., 0., 3.]])
        normalized = sic.normalize_probe_scores(scores)
        torch.testing.assert_close(normalized.sum(1), torch.ones(2))
        torch.testing.assert_close(
            normalized,
            sic.normalize_probe_scores(scores * torch.tensor([[17.], [0.125]])),
        )
        torch.testing.assert_close(scores, torch.tensor([[2., 3., 5.], [1., 0., 3.]]))

    def test_zero_rows_and_empty_axes(self):
        scores = torch.tensor([[0., 0.], [1., 3.]])
        torch.testing.assert_close(
            sic.normalize_probe_scores(scores), torch.tensor([[0., 0.], [.25, .75]])
        )
        for shape in ((0, 4), (3, 0), (0, 0)):
            with self.subTest(shape=shape):
                self.assertEqual(tuple(sic.normalize_probe_scores(torch.zeros(shape)).shape), shape)

    def test_large_and_tiny_finite_rows(self):
        for dtype, magnitude in ((torch.float32, 3e38), (torch.float64, 1e300),
                                 (torch.float64, 1e-300)):
            with self.subTest(dtype=dtype, magnitude=magnitude):
                out = sic.normalize_probe_scores(torch.tensor([[magnitude, magnitude]], dtype=dtype))
                torch.testing.assert_close(out, torch.tensor([[.5, .5]]))
                self.assertEqual(out.dtype, torch.float32)

    def test_integer_and_detached_output(self):
        torch.testing.assert_close(
            sic.normalize_probe_scores(torch.tensor([[1, 3]])), torch.tensor([[.25, .75]])
        )
        self.assertFalse(sic.normalize_probe_scores(torch.ones(2, 2, requires_grad=True)).requires_grad)

    def test_invalid_scores(self):
        for scores in (torch.ones(3), torch.ones(1, 2, 3), torch.tensor([[-1.]]),
                       torch.tensor([[float('nan')]]), torch.tensor([[float('inf')]]),
                       torch.tensor([[-float('inf')]]), torch.ones(1, 2, dtype=torch.complex64)):
            with self.subTest(scores=scores):
                with self.assertRaises(ValueError):
                    sic.normalize_probe_scores(scores)


class CoverageSelectTests(unittest.TestCase):
    def test_coverage_avoids_redundant_blocks(self):
        scores = torch.tensor([[1., 1., 0.], [0., 0., 1.]])
        self.assertEqual(sic.coverage_select(scores, 2), [1, 2])

    def test_per_probe_scaling_does_not_change_selection(self):
        scores = torch.tensor([[8., 2., 0., 1.], [0., 0., 7., 3.]])
        self.assertEqual(sic.coverage_select(scores, 2),
                         sic.coverage_select(scores * torch.tensor([[100.], [.01]]), 2))

    def test_newer_ties_and_temporal_order(self):
        self.assertEqual(sic.coverage_select(torch.ones(3, 5), 3), [2, 3, 4])
        self.assertEqual(sic.coverage_select(torch.tensor([[4., 1., 9.]]), 2), [0, 2])

    def test_zero_and_no_probe_recency(self):
        for scores in (torch.zeros(2, 5), torch.empty(0, 5)):
            self.assertEqual(sic.coverage_select(scores, 2), [3, 4])

    def test_budget_boundaries(self):
        scores = torch.ones(2, 3)
        self.assertEqual(sic.coverage_select(scores, 0), [])
        self.assertEqual(sic.coverage_select(scores, 3), [0, 1, 2])
        self.assertEqual(sic.coverage_select(torch.empty(0, 0), 0), [])
        for budget in (-1, 4, 1.5, True):
            with self.subTest(budget=budget):
                with self.assertRaises(ValueError):
                    sic.coverage_select(scores, budget)

    def test_validation_even_for_empty_budget(self):
        with self.assertRaises(ValueError):
            sic.coverage_select(torch.tensor([[float('nan')]]), 0)

    def test_matches_explicit_greedy_objective(self):
        normalized = sic.normalize_probe_scores(torch.tensor([
            [4., 1., 0., 3., 1.], [0., 2., 5., 1., 0.], [1., 3., 1., 0., 4.]
        ]))
        kept = []
        for budget in range(1, 6):
            candidate = max(
                (j for j in range(5) if j not in kept),
                key=lambda j: (float(normalized[:, kept + [j]].sum(1).sqrt().sum()), j),
            )
            kept.append(candidate)
            self.assertEqual(sic.coverage_select(normalized, budget), sorted(kept))


class BlockTaylorSalienceTests(unittest.TestCase):
    def test_signed_contributions_cancel_before_absolute_value(self):
        grad = torch.tensor([[[1.], [-1.], [-2.], [-3.]]])
        memory = torch.ones_like(grad)
        torch.testing.assert_close(sic.block_taylor_salience(grad, memory, 2), torch.tensor([0., 5.]))
        torch.testing.assert_close(sic.block_salience(grad, 2), torch.tensor([2., 13.]))

    def test_layer_and_feature_cancellation(self):
        grad = torch.tensor([[[1., -1.]], [[2., -2.]]])
        torch.testing.assert_close(sic.block_taylor_salience(grad, torch.ones_like(grad), 1), torch.zeros(1))

    def test_fp16_noncontiguous_and_no_mutation(self):
        grad = torch.arange(24, dtype=torch.float16).reshape(2, 3, 4).transpose(1, 2)
        memory = -torch.ones_like(grad)
        original = grad.clone()
        expected = torch.stack([(grad[:, i:i + 2].float() * memory[:, i:i + 2].float()).sum().abs()
                                for i in (0, 2)])
        out = sic.block_taylor_salience(grad.requires_grad_(), memory, 2)
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(grad, original)
        self.assertEqual(out.dtype, torch.float32)
        self.assertFalse(out.requires_grad)

    def test_empty_window(self):
        empty = torch.empty(2, 0, 3)
        self.assertEqual(tuple(sic.block_taylor_salience(empty, empty, 2).shape), (0,))

    def test_invalid_shapes_and_blocks(self):
        good = torch.ones(1, 4, 2)
        for grad, memory, block in ((good, torch.ones(1, 2, 2), 2),
                                    (torch.ones(4, 2), torch.ones(4, 2), 2),
                                    (good, good, 0), (good, good, -1), (good, good, 3),
                                    (good, good, 1.5), (good, good, True),
                                    (good.to(torch.complex64), good, 2),
                                    (torch.empty(0, 4, 2), torch.empty(0, 4, 2), 2)):
            with self.subTest(shape=grad.shape, block=block):
                with self.assertRaises(ValueError):
                    sic.block_taylor_salience(grad, memory, block)

    def test_nonfinite_in_either_input_and_overflow(self):
        for bad_value in (float('nan'), float('inf'), -float('inf')):
            bad = torch.tensor([[[1.], [bad_value]]])
            good = torch.ones_like(bad)
            for grad, memory in ((bad, good), (good, bad)):
                with self.assertRaises(ValueError):
                    sic.block_taylor_salience(grad, memory, 1)
        with self.assertRaises(ValueError):
            sic.block_taylor_salience(torch.full((1, 1, 1), 3e38), torch.full((1, 1, 1), 3e38), 1)


class LegacyBehaviorTests(unittest.TestCase):
    def test_salience_select_retains_strict_budget_and_recency(self):
        self.assertEqual(sic.salience_select(torch.ones(4), 2), [2, 3])
        for budget in (0, 4):
            with self.assertRaises(ValueError):
                sic.salience_select(torch.ones(4), budget)

    def test_exports(self):
        for name in ('normalize_probe_scores', 'coverage_select', 'block_taylor_salience'):
            self.assertIn(name, sic.__all__)


if __name__ == '__main__':
    unittest.main()
