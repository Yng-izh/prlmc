"""CPU contracts for --budget-blocks plumbing and the fused-subset report tool.

Run with:  CUDA_VISIBLE_DEVICES="" python tools/test_budget_and_fused_report.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_natural_merge_v2 import tiny_manager
from test_public_retention import write_record
from natural_merge_runtime import MergeStream
from public_retention_eval import StreamingMemory
from methods.prlmc.slot_consolidate import SlotConsolidateMerge
from fused_subset_report import build_report, bootstrap_ci


class BudgetContracts(unittest.TestCase):
    def _run(self, arm, budget_blocks, writes=40):
        manager = tiny_manager()
        merge = SlotConsolidateMerge(manager.num_layers, manager.hidden, manager.block) \
            if arm == "Consolidate" else None
        stream = MergeStream(manager, merge, arm, budget_blocks=budget_blocks)
        for i in range(writes):
            stream.write(write_record(i), i)
        return manager, stream

    def test_window_target_per_budget(self):
        # Pre-registered geometry: 8 -> 9->4, 10 -> 11->5, 12 -> 11->5, 20 -> 19->9.
        for budget, window, target in ((8, 9, 4), (10, 11, 5), (12, 11, 5), (20, 19, 9)):
            m, stream = self._run("Consolidate", budget)
            self.assertEqual(stream.budget, budget * m.block)
            self.assertEqual(stream.window, window * m.block)
            self.assertEqual(stream.target, target * m.block)

    def test_repeated_compression_9_to_4_and_11_to_5(self):
        # budget 8: repeated (9 -> 4); budget 10: repeated (11 -> 5);
        # budget 12: repeated (11 -> 5) unchanged. The window may transiently
        # exceed the budget (budget+1 blocks) but the settled state after every
        # completed write cycle never does.
        # Peak contract: used never exceeds max(budget, window) at any point.
        # (After the last compact the stream refills toward the window; the
        # budget is enforced by the next compact firing, so the hard bound is
        # max(budget, window).)
        for budget, shape, peak, end in ((8, (9, 4), 9, 9), (10, (11, 5), 11, 11),
                                         (12, (11, 5), 12, 12), (20, (19, 9), 20, 20)):
            m, stream = self._run("Consolidate", budget)
            shapes = {(e["source_slots"] // m.block, e["target_slots"] // m.block)
                      for e in stream.events}
            self.assertEqual(shapes, {shape}, (budget, shapes))
            self.assertGreater(len(stream.events), 1, "budget must trigger repeated compressions")
            self.assertLessEqual(stream.peak // m.block, peak, (budget, stream.peak))
            self.assertLessEqual(m.used // m.block, end)

    def test_selection_arm_repeated_compression(self):
        m, stream = self._run("Selection", 12)
        shapes = {(e["source_slots"] // m.block, e["target_slots"] // m.block)
                  for e in stream.events}
        self.assertEqual(shapes, {(11, 5)})
        self.assertLessEqual(stream.peak // m.block, 12)
        self.assertLessEqual(m.used // m.block, 12)

    def test_budget_20_unchanged(self):
        m, stream = self._run("Consolidate", 20)
        self.assertEqual((stream.window // m.block, stream.target // m.block), (19, 9))

    def test_invalid_budget_rejected(self):
        with self.assertRaises(ValueError):
            MergeStream(tiny_manager(), None, "Consolidate", budget_blocks=9)


class FusedSubsetReportContracts(unittest.TestCase):
    @staticmethod
    def make_run(tmp, name, compression_counts, f1s):
        samples = [dict(task="hotpotqa", id=f"qa{i}", sample_index=i,
                        compression_count=c, official_qa_f1_truncated=f)
                   for i, (c, f) in enumerate(zip(compression_counts, f1s))]
        path = Path(tmp) / name
        path.write_text(json.dumps(dict(schema="longbench_prlmc_results/1",
                                        status="complete", samples=samples)))
        return str(path)

    def test_fused_subset_and_paired_wl(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = self.make_run(tmp, "t.json", [1, 0, 3, 1], [0.6, 0.2, 0.8, 0.4])
            b = self.make_run(tmp, "b.json", [0, 0, 0, 0], [0.4, 0.3, 0.5, 0.4])
            report = build_report(t, [b])
            # fused subset = indices with compression_count > 0 -> samples 0, 2, 3
            self.assertEqual(report["treatment"]["fused"]["n"], 3)
            self.assertAlmostEqual(report["treatment"]["fused"]["mean"],
                                   (0.6 + 0.8 + 0.4) / 3)
            self.assertAlmostEqual(report["treatment"]["full"]["mean"],
                                   (0.6 + 0.2 + 0.8 + 0.4) / 4)
            paired = report["baselines"][0]["paired"]
            self.assertEqual(paired["paired_n"], 3)
            self.assertEqual(paired["wins"], 2)   # 0.6>0.4, 0.8>0.5, 0.4==0.4 tie
            self.assertEqual(paired["ties"], 1)
            self.assertEqual(paired["losses"], 0)
            self.assertAlmostEqual(paired["fused_delta_mean"], (0.2 + 0.3 + 0.0) / 3)
            ci = paired["bootstrap_ci"]
            self.assertEqual(ci["n"], 3)
            self.assertLessEqual(ci["low"], paired["fused_delta_mean"])
            self.assertGreaterEqual(ci["high"], paired["fused_delta_mean"])

    def test_bootstrap_ci_bounds_and_degenerate(self):
        ci = bootstrap_ci([0.1] * 50, resamples=500)
        self.assertEqual(ci["low"], 0.1)
        self.assertEqual(ci["high"], 0.1)
        self.assertEqual(bootstrap_ci([], resamples=100)["n"], 0)

    def test_no_fused_samples_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = self.make_run(tmp, "t.json", [0, 0], [0.5, 0.5])
            b = self.make_run(tmp, "b.json", [0, 0], [0.5, 0.5])
            with self.assertRaises(ValueError):
                build_report(t, [b])

    def test_incomplete_run_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            path.write_text(json.dumps(dict(schema="longbench_prlmc_results/1",
                                            status="running", samples=[])))
            with self.assertRaises(ValueError):
                build_report(str(path), [str(path)])


if __name__ == "__main__":
    unittest.main()
