"""CPU contracts for --budget-blocks plumbing, the fused-subset report tool,
LongBench run identity/--resume checks and the dual MemoryLLM protocol labels.

Run with:  CUDA_VISIBLE_DEVICES="" python tools/test_budget_and_fused_report.py
"""
import argparse
import contextlib
import io
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
from fused_subset_report import build_report, bootstrap_ci, fused_indices, summarize
import longbench_memoryllm_eval as memoryllm
import longbench_prlmc_eval as prlmc


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
        # Pre-existing failure (t1 baseline, test_budget_and_fused_report.py:113):
        # the mean of fifty 0.1 draws is 0.09999999999999996, not exactly 0.1, so
        # this is a float-equality bug in the assertion, not a v7 regression.
        ci = bootstrap_ci([0.1] * 50, resamples=500)
        self.assertAlmostEqual(ci["low"], 0.1, places=12)
        self.assertAlmostEqual(ci["high"], 0.1, places=12)
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


class EvalIdentityContracts(unittest.TestCase):
    """--resume identity, CLI geometry, --dry-run and eligibility plumbing."""

    @staticmethod
    def args(*argv):
        return prlmc.build_parser().parse_args(list(argv))

    def test_cli_defaults_preserve_existing_semantics(self):
        args = self.args()
        self.assertEqual((args.window_blocks, args.target_blocks), (None, None))
        self.assertEqual((args.budget_blocks, args.total_tokens, args.tail_tokens),
                         (20, 16384, 2048))
        self.assertIs(args.strict_code_hash, True)
        self.assertFalse(args.dry_run)
        self.assertEqual(prlmc.effective_geometry(args), (19, 9))

    def test_geometry_defaults_follow_the_budget_table(self):
        for budget, geometry in ((8, (9, 4)), (10, (11, 5)), (12, (11, 5)), (20, (19, 9))):
            args = self.args("--budget-blocks", str(budget))
            self.assertEqual(prlmc.effective_geometry(args), geometry)

    def test_explicit_geometry_is_validated(self):
        args = self.args("--window-blocks", "13", "--target-blocks", "9")
        self.assertEqual(prlmc.effective_geometry(args), (13, 9))
        for argv in (("--window-blocks", "9", "--target-blocks", "9"),
                     ("--window-blocks", "13", "--target-blocks", "20"),
                     ("--window-blocks", "51", "--target-blocks", "9")):
            with self.assertRaises(ValueError):
                prlmc.effective_geometry(self.args(*argv))

    def test_explicit_geometry_needs_the_v7_stream(self):
        args = self.args("--window-blocks", "13", "--target-blocks", "9")

        class LegacyStream:
            def __init__(self, manager, compactor, arm, budget_blocks=20):
                pass

        class V7Stream:
            def __init__(self, manager, compactor, arm, budget_blocks=20,
                         window_blocks=19, target_blocks=9):
                pass

        with self.assertRaises(RuntimeError):
            prlmc.stream_kwargs(args, LegacyStream)
        kwargs, geometry = prlmc.stream_kwargs(args, V7Stream)
        self.assertEqual(geometry, (13, 9))
        self.assertEqual(kwargs, dict(budget_blocks=20, window_blocks=13, target_blocks=9))
        # Defaults must not require the new runtime signature at all.
        defaults, geometry = prlmc.stream_kwargs(self.args(), LegacyStream)
        self.assertEqual(defaults, dict(budget_blocks=20))
        self.assertEqual(geometry, (19, 9))

    def test_total_tokens_32768_is_rejected_explicitly(self):
        args = self.args("--total-tokens", "32768", "--arm", "Selection")
        with self.assertRaises(ValueError) as caught:
            prlmc.evaluate(args)
        self.assertIn("not enabled", str(caught.exception))

    def test_identity_records_every_field_and_geometry(self):
        args = self.args("--window-blocks", "13", "--target-blocks", "9")
        geometry = prlmc.effective_geometry(args)
        saved = dict(architecture="grouped_gated_sparse_slot_v7", variant="DsupV7",
                     module_configuration=dict(carrier_selector_layer=16, reference_layer=16))
        identity = prlmc.current_identity(args, saved, geometry, "c" * 64, True)
        for key in prlmc.IDENTITY_KEYS:
            self.assertIn(key, identity)
        self.assertEqual(identity["arm"], "SharedMerge")
        self.assertEqual((identity["window_blocks"], identity["target_blocks"]), (13, 9))
        self.assertEqual((identity["select_layer"], identity["reference_layer"]), (16, 16))
        self.assertEqual(identity["critical_source_manifest"], prlmc.critical_manifest())

    def test_resume_rejects_identity_and_sample_mismatch(self):
        current = dict(arm="SharedMerge", checkpoint_sha256="a" * 64, total_tokens=16384)
        self.assertEqual(prlmc.identity_mismatch(current, current), {})
        self.assertEqual(prlmc.identity_mismatch(dict(current, arm="Selection"), current),
                         {"arm": ("Selection", "SharedMerge")})
        good = [dict(task="hotpotqa", sample_index=0, arm="SharedMerge",
                     checkpoint_sha256="a" * 64)]
        prlmc.validate_resume_samples(good, current)
        with self.assertRaises(RuntimeError):
            prlmc.validate_resume_samples([dict(good[0], arm="Selection")], current)
        with self.assertRaises(RuntimeError):
            prlmc.validate_resume_samples([dict(good[0], checkpoint_sha256="b" * 64)], current)

    def test_unverified_checkpoint_is_not_evaluation_eligible(self):
        flags = prlmc.evaluation_flags("ckpt.pt", False)
        self.assertFalse(flags["evaluation_eligible"])
        self.assertIn("historical diagnosis only", flags["warning"])
        self.assertTrue(prlmc.evaluation_flags("ckpt.pt", True)["evaluation_eligible"])
        self.assertTrue(prlmc.evaluation_flags(None, None)["evaluation_eligible"])

    def _dry_run(self, tmp):
        checkpoint = Path(tmp) / "ckpt.pt"
        torch.save(dict(architecture="block_v1", state_dict={"w": torch.zeros(1)},
                        source_manifest=prlmc.source_manifest(),
                        effective_gamma_max=.05, evaluation_eligible=True), checkpoint)
        out = Path(tmp) / "results.json"
        args = self.args("--dry-run", "--arm", "SharedMerge", "--checkpoint", str(checkpoint),
                         "--out", str(out), "--window-blocks", "13", "--target-blocks", "9")
        stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            plan = prlmc.evaluate(args)
        return plan, out, stderr.getvalue()

    def test_dry_run_writes_no_file_and_prints_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, out, _ = self._dry_run(tmp)
            self.assertFalse(out.exists(), "--dry-run must not write the result file")
        self.assertTrue(plan["dry_run"])
        self.assertEqual(plan["geometry"]["window_blocks"], 13)
        self.assertEqual(plan["protocol"]["compaction"], "13->9 block merge before the "
                                                        "write that would exceed the budget")
        self.assertTrue(plan["identity"]["code_hash_verified"])
        self.assertTrue(plan["evaluation_eligible"])
        self.assertEqual(plan["warnings"], [])

    def test_dry_run_marks_historical_checkpoint_ineligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "legacy.pt"
            torch.save(dict(architecture="block_v1", state_dict={"w": torch.zeros(1)},
                            effective_gamma_max=.05, evaluation_eligible=True), checkpoint)
            out = Path(tmp) / "r.json"
            args = self.args("--dry-run", "--arm", "SharedMerge", "--checkpoint", str(checkpoint),
                             "--out", str(out))
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                plan = prlmc.evaluate(args)
            self.assertFalse(out.exists(), "--dry-run must not write the result file")
        self.assertFalse(plan["evaluation_eligible"])
        self.assertFalse(plan["identity"]["code_hash_verified"])
        self.assertIn("historical diagnosis only", plan["warning"])
        self.assertTrue(any("source_manifest" in w for w in plan["warnings"]))
        self.assertIn("WARNING", stderr.getvalue())

    def test_active_merge_subset_matches_fused_report(self):
        samples = [dict(task="hotpotqa", id=f"qa{i}", sample_index=i, compression_count=c,
                        official_qa_f1=f, official_qa_f1_truncated=f)
                   for i, (c, f) in enumerate(zip([0, 1, 2, 0], [0.2, 0.6, 0.8, 0.4]))]
        summary = prlmc.summarize_task(samples, "hotpotqa")
        fused = fused_indices(samples)
        self.assertEqual(summary["active_merge_subset"]["n"], len(fused))
        self.assertEqual(summary["active_merge_subset"]["n"], 2)
        self.assertAlmostEqual(summary["active_merge_subset"]["qa_f1"],
                               100 * summarize(fused)["mean"])
        self.assertEqual(summary["no_merge_subset"]["n"], 2)
        self.assertAlmostEqual(summary["qa_f1"], 100 * summarize(samples)["mean"])
        self.assertEqual(summary["active_merge_subset"]["n"] + summary["no_merge_subset"]["n"],
                         summary["n"])
        self.assertEqual(prlmc.summarize_task([], "hotpotqa")["n"], 0)


class MemoryLLMProtocolContracts(unittest.TestCase):
    def test_protocol_mode_is_recorded_in_the_report(self):
        for mode in memoryllm.PROTOCOL_MODES:
            args = argparse.Namespace(total_tokens=16384, tail_tokens=2048,
                                      protocol_mode=mode, seed=0)
            protocol = memoryllm.build_protocol(args, ("hotpotqa",), False)
            self.assertEqual(protocol["protocol_mode"], mode)
            self.assertEqual(protocol["direct_query_tokens"], 2048)
            self.assertEqual(protocol["write_prefix_tokens"], 16384 - 2048)
        official = memoryllm.protocol_mode_record("official_memoryllm")
        self.assertIn("middle truncation", official["prompt_truncation"])
        self.assertIn("tail", memoryllm.protocol_mode_record("prlmc_aligned")["prompt_truncation"])
        with self.assertRaises(ValueError):
            memoryllm.protocol_mode_record("unknown")

    def test_protocol_modes_change_only_the_truncation_axis(self):
        ids = list(range(10))
        self.assertEqual(memoryllm.truncate_ids(ids, 10, "prlmc_aligned"), ids)
        self.assertEqual(memoryllm.truncate_ids(ids, 6, "prlmc_aligned"), [4, 5, 6, 7, 8, 9])
        self.assertEqual(memoryllm.truncate_ids(ids, 6, "official_memoryllm"), [0, 1, 2, 7, 8, 9])


if __name__ == "__main__":
    unittest.main()
