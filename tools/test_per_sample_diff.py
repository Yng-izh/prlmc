#!/usr/bin/env python3
"""CPU contract for tools/per_sample_diff.py (the three-way per-question diff).

The report is only useful if its categories are exact: "lost_to_compression" is
the headroom set, "rescued" is what the merge operator actually recovered, and
"damaged" is what it broke. These tests pin the category rule and the pairing.

Run with:  python tools/test_per_sample_diff.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import per_sample_diff as diff


def run_file(directory, name, samples, status="complete", arm=None):
    path = Path(directory) / name
    path.write_text(json.dumps(dict(
        schema="longbench_prlmc_results/1", status=status,
        identity=dict(arm=arm), protocol=dict(merge_arm=arm), samples=samples)))
    return str(path)


def sample(index, f1, prediction="p", answers=("gold",), task="hotpotqa"):
    return dict(task=task, id=f"id{index}", sample_index=index,
                official_qa_f1_truncated=f1, prediction=prediction,
                answers=list(answers), compression_count=1)


class Categories(unittest.TestCase):
    def test_ceiling_and_selection_agree(self):
        category, flags = diff.classify(sample(0, 1.0), sample(0, 1.0), None)
        self.assertEqual(category, "both_correct")
        self.assertEqual(flags, dict(selection=True, ceiling=True, treatment=None))
        category, _ = diff.classify(sample(1, 0.0), sample(1, 0.0), None)
        self.assertEqual(category, "both_wrong")

    def test_headroom_is_the_lost_to_compression_set(self):
        category, _ = diff.classify(sample(0, 0.0), sample(0, 1.0), None)
        self.assertEqual(category, "lost_to_compression")
        category, _ = diff.classify(sample(1, 1.0), sample(1, 0.0), None)
        self.assertEqual(category, "compression_helped")

    def test_treatment_splits_headroom_into_rescued_and_missed(self):
        lost, ceiling = sample(0, 0.0), sample(0, 1.0)
        self.assertEqual(diff.classify(lost, ceiling, sample(0, 0.8))[0], "rescued")
        self.assertEqual(diff.classify(lost, ceiling, sample(0, 0.0))[0], "missed")

    def test_treatment_gains_and_damage(self):
        # Selection wrong, ceiling wrong, treatment right -> gained.
        self.assertEqual(diff.classify(sample(0, 0.0), sample(0, 0.0),
                                       sample(0, 0.5))[0], "gained")
        # Selection right, treatment wrong -> damaged.
        self.assertEqual(diff.classify(sample(0, 1.0), sample(0, 1.0),
                                       sample(0, 0.0))[0], "damaged")

    def test_threshold_is_strict(self):
        # 0.0 is never correct; a 0.5 overlap is correct only above a 0.0 threshold.
        self.assertFalse(diff.correct(sample(0, 0.0)))
        self.assertTrue(diff.correct(sample(0, 0.5)))
        self.assertFalse(diff.correct(sample(0, 0.5), threshold=0.5))


class Pairing(unittest.TestCase):
    def test_only_shared_keys_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            sel = run_file(tmp, "sel.json", [sample(0, 0.0), sample(1, 1.0)])
            ceil = run_file(tmp, "full.json", [sample(0, 1.0), sample(2, 1.0)])
            report = diff.build_report(type("A", (), dict(
                selection=sel, ceiling=ceil, treatment=None, allow_partial=False,
                correct_threshold=0.0, data_dir=None)))
        self.assertEqual(report["paired_samples"], 1)
        self.assertEqual(report["rows"][0]["sample_index"], 0)
        self.assertEqual(report["rows"][0]["category"], "lost_to_compression")

    def test_partial_runs_are_refused_unless_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            sel = run_file(tmp, "sel.json", [sample(0, 0.0)], status="running")
            ceil = run_file(tmp, "full.json", [sample(0, 1.0)])
            with self.assertRaises(ValueError) as caught:
                diff.load_run(sel, allow_partial=False)
            self.assertIn("--allow-partial", str(caught.exception))
            samples, meta = diff.load_run(sel, allow_partial=True)
            self.assertTrue(meta["partial"])
            self.assertEqual(len(samples), 1)

    def test_question_text_is_looked_up_by_sample_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "hotpotqa.jsonl").write_text(
                json.dumps(dict(context="c", input="Q0", answers=["a"])) + "\n"
                + json.dumps(dict(context="c", input="Q1", answers=["a"])) + "\n")
            self.assertEqual(diff.question_text(tmp, "hotpotqa", 1), "Q1")
            self.assertIsNone(diff.question_text(tmp, "hotpotqa", 9))
            self.assertIsNone(diff.question_text(None, "hotpotqa", 0))


if __name__ == "__main__":
    unittest.main()
