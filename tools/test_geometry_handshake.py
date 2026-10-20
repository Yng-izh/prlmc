#!/usr/bin/env python3
"""CPU handshake contract: the CLI geometry must reach the streaming runtime (v7 C8).

t8 found that `--dry-run --window-blocks 13 --target-blocks 9` was a false
green: `evaluate` returned before `stream_kwargs`, so the CLI parsed a geometry
the runtime never saw. These tests close that gap without loading any model:

* the CLI kwargs are fed to the real `MergeStream` on a tiny CPU reader, and the
  resolved window/target/trigger are read back off the stream object;
* the value that arrives is the value that compacts -- first compaction at 14
  blocks for 13/9, at 20 blocks for the default 19/9;
* `--dry-run` must go through the same handshake, so a runtime without the v7
  geometry signature fails loudly instead of printing a green plan.

Run with:  CUDA_VISIBLE_DEVICES="" python tools/test_geometry_handshake.py
"""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import longbench_prlmc_eval as prlmc
from natural_merge_runtime import MergeStream
from public_retention_eval import compaction_trigger
from test_public_retention import manager, write_record


def cli_args(*argv):
    return prlmc.build_parser().parse_args(["--dry-run", *argv])


def drive(stream, count, start=0):
    """One write per native block; `start` keeps the write index monotonic."""
    for i in range(start, start + count):
        stream.write(write_record(i), i)
    return stream


class CliToRuntimeHandshake(unittest.TestCase):
    def test_explicit_13_9_reaches_the_stream_and_compacts_at_14(self):
        args = cli_args("--window-blocks", "13", "--target-blocks", "9")
        kwargs, geometry = prlmc.stream_kwargs(args)
        self.assertEqual(geometry, (13, 9))
        self.assertEqual(kwargs, dict(budget_blocks=20, window_blocks=13, target_blocks=9))
        m, compactor = manager()
        stream = MergeStream(m, compactor, "Selection", **kwargs)
        # The resolved pair, in blocks and in slots, is on the stream itself.
        self.assertEqual((stream.window_blocks, stream.target_blocks), (13, 9))
        self.assertEqual((stream.window, stream.target), (13 * m.block, 9 * m.block))
        self.assertEqual(stream.budget_blocks, 20)
        self.assertEqual(compaction_trigger(stream.budget_blocks, stream.window_blocks), 14)
        # One write per block: 13 resident blocks do not compact, the 14th does.
        drive(stream, 13)
        self.assertEqual(stream.events, [], "13 resident blocks must not compact")
        drive(stream, 2, start=13)
        self.assertEqual([e["before_write"] for e in stream.events], [14])
        self.assertEqual(stream.events[0]["source_slots"] // m.block, 13)
        self.assertEqual(stream.events[0]["target_slots"] // m.block, 9)

    def test_default_geometry_still_compacts_at_20(self):
        args = cli_args()
        kwargs, geometry = prlmc.stream_kwargs(args)
        self.assertEqual(geometry, (19, 9))
        self.assertEqual(kwargs, dict(budget_blocks=20))
        m, compactor = manager()
        stream = MergeStream(m, compactor, "Selection", **kwargs)
        self.assertEqual((stream.window_blocks, stream.target_blocks), (19, 9))
        drive(stream, 20)
        self.assertEqual(stream.events, [])
        drive(stream, 1, start=20)
        self.assertEqual([e["before_write"] for e in stream.events], [20])
        self.assertEqual(stream.events[0]["source_slots"] // m.block, 19)

    def test_protocol_label_names_the_real_trigger(self):
        """The text written into every v7 result JSON must state the C8 trigger."""
        args = cli_args("--window-blocks", "13", "--target-blocks", "9")
        geometry = prlmc.effective_geometry(args)
        text = prlmc.protocol_record(args, ("hotpotqa",), None, geometry)["compaction"]
        self.assertIn("13->9", text)
        self.assertIn("used>=14", text)
        self.assertNotIn("used>=20", text)
        default = cli_args()
        default_text = prlmc.protocol_record(default, ("hotpotqa",), None,
                                             prlmc.effective_geometry(default))["compaction"]
        self.assertIn("19->9", default_text)
        self.assertIn("used>=20", default_text)

    def test_registered_budget_geometries_stay_grid_identical(self):
        for budget, geometry, first in ((8, (9, 4), 9), (10, (11, 5), 11),
                                        (12, (11, 5), 12), (20, (19, 9), 20)):
            args = cli_args("--budget-blocks", str(budget))
            kwargs, resolved = prlmc.stream_kwargs(args)
            self.assertEqual(resolved, geometry, budget)
            m, compactor = manager()
            stream = MergeStream(m, compactor, "Selection", **kwargs)
            self.assertEqual((stream.window_blocks, stream.target_blocks), geometry)
            drive(stream, first)
            self.assertEqual(stream.events, [], budget)
            drive(stream, 1, start=first)
            self.assertEqual([e["before_write"] for e in stream.events], [first], budget)


class DryRunUsesTheHandshake(unittest.TestCase):
    def test_dry_run_calls_stream_kwargs_before_printing(self):
        calls = []
        original = prlmc.stream_kwargs

        def spy(args, merge_stream_cls=prlmc.MergeStream):
            calls.append(prlmc.effective_geometry(args))
            return original(args, merge_stream_cls)

        prlmc.stream_kwargs = spy
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "plan.json"
                args = cli_args("--arm", "Selection", "--window-blocks", "13",
                                "--target-blocks", "9", "--out", str(out))
                with contextlib.redirect_stdout(io.StringIO()):
                    plan = prlmc.evaluate(args)
                self.assertFalse(out.exists())
        finally:
            prlmc.stream_kwargs = original
        self.assertEqual(calls, [(13, 9)], "--dry-run must validate the runtime handshake")
        self.assertEqual(plan["geometry"]["window_blocks"], 13)
        self.assertEqual(plan["geometry"]["target_blocks"], 9)

    def test_dry_run_fails_loudly_on_a_runtime_without_v7_geometry(self):
        class LegacyStream:
            def __init__(self, manager, compactor, arm, budget_blocks=20):
                pass

        original = prlmc.stream_kwargs

        def legacy(args, merge_stream_cls=prlmc.MergeStream):
            return original(args, LegacyStream)

        prlmc.stream_kwargs = legacy
        try:
            args = cli_args("--arm", "Selection", "--window-blocks", "13",
                            "--target-blocks", "9")
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError) as caught:
                    prlmc.evaluate(args)
        finally:
            prlmc.stream_kwargs = original
        self.assertIn("window_blocks", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
