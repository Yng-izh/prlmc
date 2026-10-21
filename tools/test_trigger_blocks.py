#!/usr/bin/env python3
"""CPU contract: the full-load compaction trigger (budget actually spent).

The pre-registered rule compacts at `min(budget, window+1)`, which for the v7
20/13/9 geometry fires at 14 resident blocks: the stream then cycles 10..14
blocks and the declared 15th..20th blocks are never used. `--trigger-blocks`
moves the firing point to the budget so the stream fills to 20, compresses the
oldest 13 into 9, and keeps the newest 7 uncompressed -- the layout the
independent-window training path sees.

These tests pin: the CLI flag and its validation, the kwarg plumbing into
MergeStream, the resident-width cycle it produces, and the recorded
protocol/identity.

Run with:  CUDA_VISIBLE_DEVICES="" python tools/test_trigger_blocks.py
"""
import sys
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


def drive(stream, count):
    for i in range(count):
        stream.write(write_record(i), i)


class TriggerCli(unittest.TestCase):
    def test_flag_defaults_to_the_preregistered_rule(self):
        args = cli_args("--arm", "Selection")
        self.assertIsNone(args.trigger_blocks)
        self.assertEqual(prlmc.effective_trigger(args, 13), 14)

    def test_explicit_trigger_is_used(self):
        args = cli_args("--arm", "Selection", "--trigger-blocks", "20")
        self.assertEqual(prlmc.effective_trigger(args, 13), 20)

    def test_trigger_below_the_window_is_refused(self):
        args = cli_args("--arm", "Selection", "--trigger-blocks", "12")
        with self.assertRaises(ValueError) as caught:
            prlmc.effective_trigger(args, 13)
        self.assertIn("smaller than the 13-block", str(caught.exception))

    def test_trigger_above_the_budget_is_refused(self):
        args = cli_args("--arm", "Selection", "--trigger-blocks", "21")
        with self.assertRaises(ValueError) as caught:
            prlmc.effective_trigger(args, 13)
        self.assertIn("exceeds --budget-blocks", str(caught.exception))

    def test_full_arm_refuses_the_trigger_flag(self):
        args = cli_args("--arm", "Full", "--trigger-blocks", "20")
        with self.assertRaises(ValueError) as caught:
            prlmc.evaluate(args)
        self.assertIn("performs no compaction", str(caught.exception))


class TriggerPlumbing(unittest.TestCase):
    def test_kwargs_carry_the_trigger(self):
        args = cli_args("--arm", "Selection", "--budget-blocks", "20",
                        "--window-blocks", "13", "--target-blocks", "9",
                        "--trigger-blocks", "20")
        kwargs, geometry = prlmc.stream_kwargs(args)
        self.assertEqual(kwargs["trigger_blocks"], 20)
        self.assertEqual(geometry, (13, 9))

    def test_kwargs_omit_the_trigger_by_default(self):
        args = cli_args("--arm", "Selection")
        kwargs, _ = prlmc.stream_kwargs(args)
        self.assertNotIn("trigger_blocks", kwargs)

    def test_a_stream_without_the_kwarg_is_refused(self):
        class Legacy(MergeStream):
            def __init__(self, manager, compactor, arm, budget_blocks=20,
                         window_blocks=None, target_blocks=None):
                super().__init__(manager, compactor, arm, budget_blocks=budget_blocks,
                                 window_blocks=window_blocks, target_blocks=target_blocks)

        args = cli_args("--arm", "Selection", "--trigger-blocks", "20")
        with self.assertRaises(RuntimeError) as caught:
            prlmc.stream_kwargs(args, merge_stream_cls=Legacy)
        self.assertIn("trigger_blocks", str(caught.exception))


class TriggerCycle(unittest.TestCase):
    """Resident width and firing points for the two rules on the same geometry."""

    def build(self, trigger):
        m, compactor = manager()
        return MergeStream(m, compactor, "Selection", budget_blocks=20,
                           window_blocks=13, target_blocks=9,
                           trigger_blocks=trigger), m

    def test_default_rule_fires_at_fourteen_and_wastes_the_budget(self):
        stream, m = self.build(None)
        drive(stream, 30)
        self.assertEqual(stream.events[0]["before_write"], 14)
        self.assertEqual(stream.effective_trigger_blocks, 14)
        self.assertLessEqual(stream.peak, 14 * m.block)

    def test_full_load_rule_fires_at_twenty_and_keeps_a_seven_block_tail(self):
        stream, m = self.build(20)
        drive(stream, 30)
        self.assertEqual(stream.events[0]["before_write"], 20)
        self.assertEqual(stream.effective_trigger_blocks, 20)
        self.assertEqual(stream.peak, 20 * m.block)
        # 13 -> 9 plus the 7 blocks that were outside the window.
        self.assertEqual(stream.events[0]["remaining_slots"], 16 * m.block)
        self.assertEqual([e["before_write"] for e in stream.events], [20, 24, 28])

    def test_full_load_uses_more_resident_blocks_than_the_default(self):
        default_stream, m = self.build(None)
        drive(default_stream, 30)
        full_stream, _ = self.build(20)
        drive(full_stream, 30)
        self.assertGreater(full_stream.peak, default_stream.peak)
        self.assertEqual(full_stream.capacity_overshoot_blocks, 0)


class TriggerRecord(unittest.TestCase):
    def test_protocol_and_identity_record_the_trigger(self):
        args = cli_args("--arm", "Selection", "--budget-blocks", "20",
                        "--window-blocks", "13", "--target-blocks", "9",
                        "--trigger-blocks", "20")
        geometry = prlmc.effective_geometry(args)
        protocol = prlmc.protocol_record(args, ("hotpotqa",), None, geometry)
        self.assertEqual(protocol["trigger_blocks"], 20)
        self.assertIn("full-load", protocol["compaction"])
        self.assertIn("used>=20", protocol["compaction"])
        identity = prlmc.current_identity(args, None, geometry, None, None)
        self.assertEqual(identity["trigger_blocks"], 20)
        self.assertIn("trigger_blocks", prlmc.IDENTITY_KEYS)

    def test_default_protocol_keeps_the_old_wording(self):
        args = cli_args("--arm", "Selection", "--budget-blocks", "20",
                        "--window-blocks", "13", "--target-blocks", "9")
        geometry = prlmc.effective_geometry(args)
        protocol = prlmc.protocol_record(args, ("hotpotqa",), None, geometry)
        self.assertEqual(protocol["trigger_blocks"], 14)
        self.assertIn("trigger=min(budget,window+1)", protocol["compaction"])

    def test_full_arm_records_no_trigger(self):
        args = cli_args("--arm", "Full")
        protocol = prlmc.protocol_record(args, ("hotpotqa",), None, None)
        self.assertIsNone(protocol["trigger_blocks"])


if __name__ == "__main__":
    unittest.main()
