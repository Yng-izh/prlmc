#!/usr/bin/env python3
"""CPU contract: the no-compression "Full" reference arm in the LongBench entry.

The Full arm is not a new oracle. It is the arm `StreamingMemory` already
implements for the natural-data streams (`ARMS`, `write` skips the compaction
trigger for it, capacity limit = the whole physical pool). These tests pin the
wiring into `tools/longbench_prlmc_eval.py`:

* `--arm Full` is selectable, loads no merge module, and refuses the managed
  geometry/budget flags that would mislabel a keep-everything run;
* the stream it builds compacts nothing, keeps every written block resident past
  the 20-block managed budget, and reports the native 50-block capacity;
* past the physical pool it refuses the write instead of dropping blocks
  (an honest ceiling, not a silent truncation);
* the recorded protocol/identity carry no compaction geometry.

Run with:  CUDA_VISIBLE_DEVICES="" python tools/test_full_arm.py
"""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import longbench_prlmc_eval as prlmc
from natural_merge_runtime import MergeStream
from test_public_retention import manager, write_record


def cli_args(*argv):
    return prlmc.build_parser().parse_args(["--dry-run", *argv])


def drive(stream, count, start=0):
    results = []
    for i in range(start, start + count):
        results.append(stream.write(write_record(i), i))
    return results


class FullArmWiring(unittest.TestCase):
    def test_arm_is_selectable(self):
        parser = prlmc.build_parser()
        arm_action = next(a for a in parser._actions if a.dest == "arm")
        self.assertIn(prlmc.FULL_ARM, arm_action.choices)
        self.assertEqual(cli_args("--arm", "Full").arm, "Full")

    def test_full_refuses_a_checkpoint(self):
        args = cli_args("--arm", "Full", "--checkpoint", "/nonexistent.pt")
        with self.assertRaises(ValueError) as caught:
            prlmc.evaluate(args)
        self.assertIn("drop --checkpoint", str(caught.exception))

    def test_full_refuses_managed_geometry_flags(self):
        for argv in (("--window-blocks", "13"), ("--target-blocks", "9")):
            args = cli_args("--arm", "Full", *argv)
            with self.assertRaises(ValueError) as caught:
                prlmc.evaluate(args)
            self.assertIn("no compaction", str(caught.exception))

    def test_full_refuses_a_managed_budget_label(self):
        args = cli_args("--arm", "Full", "--budget-blocks", "8")
        with self.assertRaises(ValueError) as caught:
            prlmc.evaluate(args)
        self.assertIn("50-block capacity", str(caught.exception))

    def test_stream_kwargs_reports_no_compaction_geometry(self):
        kwargs, geometry = prlmc.stream_kwargs(cli_args("--arm", "Full"))
        self.assertIsNone(geometry)
        self.assertEqual(kwargs, dict(budget_blocks=prlmc.BUDGET_BLOCKS))


class FullStreamKeepsEverything(unittest.TestCase):
    def stream(self):
        m, compactor = manager()
        kwargs, geometry = prlmc.stream_kwargs(cli_args("--arm", "Full"))
        self.assertIsNone(geometry)
        return m, MergeStream(m, compactor, prlmc.FULL_ARM, **kwargs)

    def test_no_compaction_past_the_managed_budget(self):
        m, stream = self.stream()
        results = drive(stream, 28)
        self.assertTrue(all(r is not None for r in results))
        self.assertEqual(stream.events, [], "the Full arm must never compact")
        self.assertEqual(m.used, 28 * m.block)
        self.assertGreater(m.used, 20 * m.block, "must exceed the managed 20-block budget")
        self.assertEqual(stream.peak, 28 * m.block)
        self.assertEqual(int(m.model.memory_active_mask.sum()), m.used)

    def test_refuses_to_drop_blocks_at_the_physical_pool(self):
        m, stream = self.stream()
        blocks = m.total // m.block
        results = drive(stream, blocks + 1)
        self.assertIsNone(results[-1], "past the physical pool the write must be refused")
        self.assertTrue(all(r is not None for r in results[:-1]))
        self.assertEqual(m.used, m.total)
        self.assertEqual(stream.events, [])


class FullArmRecordsNativeCapacity(unittest.TestCase):
    def test_protocol_declares_the_physical_pool(self):
        args = cli_args("--arm", "Full")
        protocol = prlmc.protocol_record(args, ("hotpotqa",), None, None)
        self.assertEqual(protocol["capacity"], "native_50_blocks")
        self.assertEqual(protocol["budget_blocks"], prlmc.PHYSICAL_BLOCKS)
        self.assertIsNone(protocol["window_blocks"])
        self.assertIsNone(protocol["target_blocks"])
        self.assertIn("none", protocol["compaction"])
        self.assertEqual(protocol["merge_arm"], "Full")

    def test_identity_carries_no_compaction_geometry(self):
        args = cli_args("--arm", "Full")
        identity = prlmc.current_identity(args, None, None, None, None)
        self.assertEqual(identity["budget_blocks"], prlmc.PHYSICAL_BLOCKS)
        self.assertIsNone(identity["window_blocks"])
        self.assertIsNone(identity["target_blocks"])
        self.assertIsNone(identity["architecture"])
        self.assertIsNone(identity["select_layer"])

    def test_dry_run_reports_native_capacity_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "plan.json"
            args = cli_args("--arm", "Full", "--out", str(out))
            with contextlib.redirect_stdout(io.StringIO()):
                plan = prlmc.evaluate(args)
            self.assertFalse(out.exists(), "--dry-run must not write the result file")
        self.assertEqual(plan["geometry"]["budget_blocks"], prlmc.PHYSICAL_BLOCKS)
        self.assertIsNone(plan["geometry"]["window_blocks"])
        self.assertEqual(plan["protocol"]["merge_arm"], "Full")
        self.assertTrue(plan["evaluation_eligible"])


if __name__ == "__main__":
    unittest.main()
