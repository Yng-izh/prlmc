"""CPU contracts for tools/ceiling_report.py: headroom vs effect, power, verdicts.

Run with:  CUDA_VISIBLE_DEVICES="" python tools/test_ceiling_report.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import ceiling_report as cr


def write_run(path, scores, status="complete", task="hotpotqa"):
    samples = [dict(task=task, id=f"id{i}", sample_index=i,
                    official_qa_f1_truncated=float(s))
               for i, s in enumerate(scores)]
    Path(path).write_text(json.dumps(dict(
        schema="longbench_prlmc_results/1", status=status,
        protocol={"merge_arm": "Selection"}, identity={"arm": "Selection"},
        samples=samples)))
    return path


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


class PowerContracts(unittest.TestCase):
    def test_required_n_matches_normal_approximation(self):
        # sd 0.3256, effect 1.7pp -> ~2880 samples
        self.assertEqual(cr.required_n(0.3256, 0.017), 2880)

    def test_required_n_guards_degenerate_input(self):
        self.assertIsNone(cr.required_n(0.3, 0.0))
        self.assertIsNone(cr.required_n(0.0, 0.01))


class VerdictContracts(unittest.TestCase):
    def _report(self, head_low, head_high, eff_low, eff_mean):
        return dict(headroom=dict(low=head_low, high=head_high),
                    effect=dict(low=eff_low, mean=eff_mean))

    def test_headroom_absent_when_ceiling_not_better(self):
        r = self._report(-0.06, 0.065, -0.06, -0.005)
        self.assertTrue(cr.verdict(r).startswith("headroom_absent"))

    def test_effect_exceeding_headroom_is_impossible(self):
        r = self._report(0.001, 0.004, 0.010, 0.017)
        self.assertTrue(cr.verdict(r).startswith("effect_exceeds_headroom"))

    def test_positive_within_headroom(self):
        r = self._report(0.001, 0.050, 0.002, 0.017)
        self.assertEqual(cr.verdict(r), "positive_within_headroom")

    def test_no_effect(self):
        r = self._report(0.001, 0.050, -0.02, -0.005)
        self.assertEqual(cr.verdict(r), "no_effect_detected")


class EndToEndContracts(unittest.TestCase):
    def test_paired_report_on_shared_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sel = write_run(tmp / "sel.json", [0.0, 1.0, 0.5, 0.0])
            ceil = write_run(tmp / "ceil.json", [0.0, 1.0, 0.5, 1.0])
            treat = write_run(tmp / "treat.json", [0.0, 1.0, 0.0, 0.0])
            argv = ["ceiling_report.py", "--selection", str(sel),
                    "--ceiling", str(ceil), "--treatment", str(treat)]
            old = sys.argv
            sys.argv = argv
            try:
                import contextlib, io
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    cr.main()
                report = json.loads(buf.getvalue())
            finally:
                sys.argv = old
            self.assertEqual(report["schema"], "prlmc_ceiling_report/1")
            self.assertEqual(report["paired_samples"], 4)
            self.assertAlmostEqual(report["mean_selection"], 0.375)
            self.assertAlmostEqual(report["mean_ceiling"], 0.625)
            self.assertAlmostEqual(report["headroom"]["mean"], 0.25)
            self.assertAlmostEqual(report["effect"]["mean"], -0.125)
            self.assertEqual(report["effect"]["wins"], 0)
            self.assertEqual(report["effect"]["losses"], 1)


if __name__ == "__main__":
    unittest.main()
