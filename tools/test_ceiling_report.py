"""CPU contracts for tools/ceiling_report.py: headroom, indicators, verdicts.

The no-compression reference is a reference, not a monotone upper bound (it wins
9 / loses 11 / ties 76 per sample on hotpotqa), so the report must never reject a
treatment for exceeding it. These tests pin the three complementary indicators,
their paired deltas, the exact-match normalisation, and the softened verdict.

Run with:  CUDA_VISIBLE_DEVICES="" python tools/test_ceiling_report.py
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import ceiling_report as cr


def write_run(path, scores, status="complete", task="hotpotqa", answers=None,
              predictions=None):
    answers = answers or [["gold%d" % i] for i in range(len(scores))]
    predictions = predictions or ["" for _ in scores]
    samples = [dict(task=task, id=f"id{i}", sample_index=i,
                    official_qa_f1_truncated=float(s),
                    answers=answers[i], prediction=predictions[i])
               for i, s in enumerate(scores)]
    Path(path).write_text(json.dumps(dict(
        schema="longbench_prlmc_results/1", status=status,
        protocol={"merge_arm": "Selection"}, identity={"arm": "Selection"},
        samples=samples)))
    return path


def hit_or_miss(scores):
    return ["gold%d" % i if s >= 0.99 else "" for i, s in enumerate(scores)]


class LoadRunContracts(unittest.TestCase):
    def test_rejects_partial_without_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_run(Path(tmp) / "r.json", [1.0], status="running")
            with self.assertRaises(ValueError):
                cr.load_run(str(p), allow_partial=False)
            scores, meta = cr.load_run(str(p), allow_partial=True)
            self.assertTrue(meta["partial"])
            self.assertEqual(len(scores), 1)

    def test_rejects_empty_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.json"
            p.write_text(json.dumps(dict(schema="longbench_prlmc_results/1",
                                         status="complete", samples=[])))
            with self.assertRaises(ValueError):
                cr.load_run(str(p), allow_partial=True)


class NormalisationContracts(unittest.TestCase):
    def test_normalise_drops_articles_case_and_punctuation(self):
        self.assertEqual(cr.normalise("The Hakim Abdullah Jamal."), "hakim abdullah jamal")
        self.assertEqual(cr.normalise("  A  Test,  "), "test")

    def test_exact_match_reads_the_truncated_prediction(self):
        sample = dict(answers=["Mary Astor"], prediction="Mary Astor\nextra prose")
        self.assertTrue(cr.is_exact(sample))
        self.assertFalse(cr.is_exact(dict(answers=["Mary Astor"], prediction="Joan Crawford")))
        self.assertFalse(cr.is_exact(dict(answers=["Mary Astor"], prediction="")))


class PowerContracts(unittest.TestCase):
    def test_required_n_matches_normal_approximation(self):
        # sd 0.3256, effect 1.7pp -> ~2880 samples
        self.assertEqual(cr.required_n(0.3256, 0.017), 2880)

    def test_required_n_guards_degenerate_input(self):
        self.assertIsNone(cr.required_n(0.3, 0.0))
        self.assertIsNone(cr.required_n(0.0, 0.01))


class IndicatorContracts(unittest.TestCase):
    def test_three_indicators_on_one_arm(self):
        scores = {("t", "a", 0): 0.0, ("t", "b", 1): 0.4, ("t", "c", 2): 1.0}
        keys = sorted(scores)
        exact = {k: (1.0 if k[2] == 2 else 0.0) for k in keys}
        out = cr.indicators(scores, keys, exact, threshold=0.8)
        self.assertAlmostEqual(out["mean_f1"], 1.4 / 3)
        self.assertAlmostEqual(out["nonzero_rate"], 2 / 3)
        self.assertAlmostEqual(out["high_precision_rate"], 1 / 3)
        self.assertAlmostEqual(out["exact_match_rate"], 1 / 3)

    def test_high_precision_threshold_is_inclusive(self):
        scores = {("t", "a", 0): 0.8}
        out = cr.indicators(scores, sorted(scores), threshold=0.8)
        self.assertAlmostEqual(out["high_precision_rate"], 1.0)

    def test_paired_deltas_cover_every_indicator(self):
        a = {("t", "a", 0): 1.0, ("t", "b", 1): 0.0}
        b = {("t", "a", 0): 0.4, ("t", "b", 1): 0.0}
        keys = sorted(a)
        exact = {k: 0.0 for k in keys}
        out = cr.indicator_deltas(a, b, keys, exact, exact, threshold=0.8)
        self.assertEqual(set(out), {"mean_f1", "nonzero_rate",
                                    "high_precision_rate", "exact_match_rate"})
        self.assertAlmostEqual(out["mean_f1"]["mean"], 0.3)
        self.assertAlmostEqual(out["high_precision_rate"]["mean"], 0.5)
        self.assertAlmostEqual(out["nonzero_rate"]["mean"], 0.0)


class VerdictContracts(unittest.TestCase):
    def _report(self, head_low, head_high, eff_low=None, eff_mean=None):
        report = dict(headroom=dict(low=head_low, high=head_high,
                                    mean=(head_low + head_high) / 2))
        if eff_low is not None:
            report["effect"] = dict(low=eff_low, mean=eff_mean)
        return report

    def test_headroom_undetected_when_the_reference_is_not_better(self):
        r = self._report(-0.06, 0.065, -0.06, -0.005)
        self.assertTrue(cr.verdict(r).startswith("headroom_undetected"))

    def test_headroom_alone_is_a_valid_outcome(self):
        self.assertEqual(cr.verdict(self._report(0.001, 0.05)), "headroom_measured")

    def test_positive_within_headroom(self):
        r = self._report(0.001, 0.050, 0.002, 0.017)
        self.assertEqual(cr.verdict(r), "positive_within_headroom")

    def test_no_effect(self):
        r = self._report(0.001, 0.050, -0.02, -0.005)
        self.assertEqual(cr.verdict(r), "no_effect_detected")

    def test_no_impossible_effect_branch_exists(self):
        # The reference is not a monotone upper bound, so no verdict may reject a
        # treatment for beating it.
        r = self._report(0.001, 0.004, 0.010, 0.017)
        self.assertEqual(cr.verdict(r), "positive_within_headroom")
        self.assertFalse(hasattr(cr, "effect_exceeds_headroom"))


class EndToEndContracts(unittest.TestCase):
    def run_report(self, tmp, sel_scores, ceil_scores, treat_scores, extra=()):
        sel = write_run(tmp / "sel.json", sel_scores, predictions=hit_or_miss(sel_scores))
        ceil = write_run(tmp / "ceil.json", ceil_scores, predictions=hit_or_miss(ceil_scores))
        treat = write_run(tmp / "treat.json", treat_scores, predictions=hit_or_miss(treat_scores))
        argv = ["ceiling_report.py", "--selection", str(sel), "--ceiling", str(ceil),
                "--treatment", str(treat), *extra]
        old = sys.argv
        sys.argv = argv
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cr.main()
            return json.loads(buf.getvalue())
        finally:
            sys.argv = old

    def test_paired_report_on_shared_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.run_report(Path(tmp), [0.0, 1.0, 0.5, 0.0],
                                     [0.0, 1.0, 0.5, 1.0], [0.0, 1.0, 0.0, 0.0])
        self.assertEqual(report["schema"], "prlmc_ceiling_report/2")
        self.assertEqual(report["paired_samples"], 4)
        self.assertAlmostEqual(report["mean_selection"], 0.375)
        self.assertAlmostEqual(report["mean_ceiling"], 0.625)
        self.assertAlmostEqual(report["headroom"]["mean"], 0.25)
        self.assertAlmostEqual(report["effect"]["mean"], -0.125)
        self.assertEqual(report["effect"]["wins"], 0)
        self.assertEqual(report["effect"]["losses"], 1)
        self.assertAlmostEqual(report["metrics"]["selection"]["nonzero_rate"], 0.5)
        self.assertAlmostEqual(report["metrics"]["ceiling"]["high_precision_rate"], 0.5)
        self.assertAlmostEqual(report["metrics"]["treatment"]["exact_match_rate"], 0.25)

    def test_treatment_above_headroom_is_flagged_not_rejected(self):
        # 20 samples so the bootstrap CI is not degenerate: the reference gains
        # half the set, the treatment gains all of it.
        with tempfile.TemporaryDirectory() as tmp:
            report = self.run_report(Path(tmp), [0.0] * 20,
                                     [1.0] * 10 + [0.0] * 10, [1.0] * 20)
        self.assertGreater(report["headroom"]["low"], 0)
        self.assertEqual(report["verdict"], "positive_within_headroom")
        self.assertTrue(any(n.startswith("effect_above_headroom_mean")
                            for n in report["notes"]))

    def test_precision_threshold_is_configurable(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.run_report(Path(tmp), [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
                                     extra=("--precision-threshold", "0.5"))
        self.assertEqual(report["metrics"]["selection"]["high_precision_threshold"], 0.5)
        self.assertAlmostEqual(report["metrics"]["selection"]["high_precision_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
