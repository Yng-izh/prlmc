"""Tests of the S1c global-oracle gate and window metric helpers."""
import unittest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from s1c_global_oracle import (ORACLE_ARMS, global_gate, window_metric,
                               window_sets)


def window_fixture(net_g, net_r, net_u, net_l, R_g, R_r, R_u, R_l):
    def final(net, rescue):
        return {"net_vs_select": net, "rescue_recovered": rescue,
                "retained_harmed": 0}
    return {"arms": {arm: {"final": final(net, rescue)} for arm, net, rescue in
                     (("global_13d", net_g, R_g), ("random_13d", net_r, R_r),
                      ("unconstrained", net_u, R_u), ("local_13d", net_l, R_l))}}


class GlobalGateTest(unittest.TestCase):
    def test_pass_when_all_margins_clear(self):
        gate = global_gate([window_fixture(6, 0, 1, 8, 9, 2, 3, 10)])
        self.assertEqual(gate["verdict"], "PASS")
        self.assertEqual(gate["net"], 6)
        self.assertEqual(gate["rescue_gap"], 7)

    def test_fail_when_random_ties_both_margins(self):
        gate = global_gate([window_fixture(3, 3, 1, 5, 6, 6, 2, 7)])
        self.assertEqual(gate["verdict"], "FAIL")

    def test_fail_on_negative_net(self):
        gate = global_gate([window_fixture(-2, -6, -1, 0, 2, 0, 3, 5)])
        self.assertEqual(gate["verdict"], "FAIL")

    def test_fail_when_unconstrained_dominates(self):
        gate = global_gate([window_fixture(3, -5, 8, 5, 8, 0, 10, 7)])
        self.assertEqual(gate["verdict"], "FAIL")

    def test_margin_on_rescue_gap_two(self):
        gate = global_gate([window_fixture(4, 0, 1, 5, 6, 4, 2, 7)])
        self.assertEqual(gate["verdict"], "MARGIN")

    def test_margin_on_low_net(self):
        gate = global_gate([window_fixture(2, -1, 0, 3, 5, 0, 1, 4)])
        self.assertEqual(gate["verdict"], "MARGIN")

    def test_margin_when_unconstrained_slightly_above(self):
        gate = global_gate([window_fixture(6, 0, 7, 8, 9, 2, 10, 11)])
        self.assertEqual(gate["verdict"], "MARGIN")

    def test_two_windows_summed(self):
        one = window_fixture(2, -1, 0, 3, 3, 0, 1, 4)
        gate = global_gate([one, dict(one)])
        self.assertEqual(gate["verdict"], "PASS")
        self.assertEqual(gate["totals"]["global_13d"]["rescue_recovered"], 6)
        self.assertEqual(gate["net"], 4)


class WindowHelpersTest(unittest.TestCase):
    def test_window_sets_partitions(self):
        full = {"a": True, "b": True, "c": False, "d": True}
        select = {"a": True, "b": False, "c": False, "d": True}
        d0 = {"a": True, "b": False, "c": True, "d": False}
        sets = window_sets(full, select, d0)
        self.assertEqual(sets["rescue"], ["b"])
        self.assertEqual(sets["retained"], ["a", "d"])
        self.assertEqual(sets["protect"], ["a"])
        self.assertEqual(sets["repair"], ["d"])
        self.assertEqual(sets["rescue_d0_wrong"], ["b"])

    def test_window_sets_universe_check(self):
        with self.assertRaises(ValueError):
            window_sets({"a": True}, {"b": True}, {"a": False})

    def test_window_metric_counts(self):
        sets = {"original_lost": [], "rescue": ["a", "b"], "rescue_d0_correct": [],
                "repair": ["c"], "protect": ["d"], "retained": ["d", "e"]}
        correct = {"a": True, "b": False, "c": True, "d": True, "e": False,
                   "f": True}
        entry = window_metric(correct, sets, step=3, deviation=0.5)
        self.assertEqual(entry["rescue_recovered"], 1)
        self.assertEqual(entry["repair_recovered"], 1)
        self.assertEqual(entry["retained_harmed"], 1)
        self.assertEqual(entry["net_vs_select"], 0)
        self.assertEqual(entry["step"], 3)
        self.assertAlmostEqual(entry["accuracy"], 4 / 6)
        self.assertEqual(entry["rescue_recovered_questions"], ["a"])


if __name__ == "__main__":
    unittest.main()
