"""Tests for the block-provenance helpers.

The pure functions are what the conclusions rest on, so they are pinned here:
block replay over recorded compactions, fusion-window absorption, token span to
block mapping, and the evidence-location buckets.
"""
import unittest

from block_provenance import absorbed, answer_forms, blocks_of, replay
from evidence_location import classify, tally

WINDOW = 13


def sample(events):
    return {"compression_events": events}


class ReplayTest(unittest.TestCase):
    def test_without_compaction_every_block_is_resident(self):
        self.assertEqual(replay(sample([]), WINDOW, 20), list(range(20)))

    def test_compaction_keeps_the_carriers_and_drops_the_rest(self):
        event = {"before_write": 14, "selected_blocks": list(range(9))}
        resident = replay(sample([event]), WINDOW, 20)
        self.assertEqual(resident[:9], list(range(9)))
        self.assertEqual(resident[-1], 19)
        self.assertNotIn(9, resident)
        self.assertNotIn(12, resident)
        self.assertIn(13, resident)          # never entered the window
        self.assertEqual(len(resident), 16)

    def test_window_is_the_oldest_blocks_only(self):
        event = {"before_write": 20, "selected_blocks": [0, 1, 2]}
        resident = replay(sample([event]), WINDOW, 21)
        self.assertEqual(resident[:3], [0, 1, 2])
        self.assertIn(13, resident)          # block 13 sits past the window
        self.assertEqual(resident[-1], 20)

    def test_two_compactions_chain(self):
        events = [{"before_write": 14, "selected_blocks": list(range(9))},
                  {"before_write": 18, "selected_blocks": [0, 1, 2, 3, 4, 5, 6, 7, 9]}]
        resident = replay(sample(events), WINDOW, 24)
        self.assertEqual(resident[-1], 23)
        self.assertEqual(len(resident), 24 - 4 - 4)


class AbsorbedTest(unittest.TestCase):
    def test_reports_the_write_that_absorbed_the_block(self):
        event = {"before_write": 14, "selected_blocks": list(range(9))}
        self.assertEqual(absorbed({0, 5}, sample([event]), WINDOW, 20),
                         [(14, [0, 5])])

    def test_block_that_never_entered_a_window_is_not_absorbed(self):
        event = {"before_write": 14, "selected_blocks": list(range(9))}
        self.assertEqual(absorbed({19}, sample([event]), WINDOW, 20), [])

    def test_no_compaction_means_no_absorption(self):
        self.assertEqual(absorbed({0}, sample([]), WINDOW, 20), [])


class BlocksOfTest(unittest.TestCase):
    def test_span_inside_one_block(self):
        self.assertEqual(blocks_of((10, 20), 10240), {0})

    def test_boundary_token_belongs_to_the_next_block(self):
        self.assertEqual(blocks_of((511, 512), 10240), {0})
        self.assertEqual(blocks_of((512, 513), 10240), {1})

    def test_span_across_blocks(self):
        self.assertEqual(blocks_of((500, 1100), 10240), {0, 1, 2})

    def test_span_is_clipped_to_the_written_prefix(self):
        self.assertEqual(blocks_of((20000, 21000), 10240), {19})


class AnswerFormsTest(unittest.TestCase):
    def test_full_answer_first_then_suffixes(self):
        self.assertEqual(answer_forms("Hakim Abdullah Jamal"),
                         ["Hakim Abdullah Jamal", "Abdullah Jamal", "Jamal"])

    def test_short_forms_are_dropped(self):
        self.assertEqual(answer_forms("no"), [])
        self.assertEqual(answer_forms("Troy, Montana"), ["Troy, Montana", "Montana"])


class ClassifyTest(unittest.TestCase):
    def test_evidence_only_in_the_written_prefix(self):
        self.assertEqual(classify([(100, 200)], 16384, 14336), "memory_only")

    def test_evidence_only_in_the_tail(self):
        self.assertEqual(classify([(15000, 15100)], 16384, 14336), "tail_only")

    def test_evidence_reachable_both_ways(self):
        self.assertEqual(classify([(14000, 14500)], 16384, 14336), "both")

    def test_missing_evidence(self):
        self.assertEqual(classify([], 16384, 14336), "absent")


class TallyTest(unittest.TestCase):
    def test_counts_and_sums_per_bucket(self):
        rows = [("both", {"Full": 1.0}), ("both", {"Full": 0.0}),
                ("absent", {"Full": 0.5})]
        stats = tally(rows, {"Full": {}})
        self.assertEqual(stats["both"]["n"], 2)
        self.assertEqual(stats["both"]["Full_right"], 1)
        self.assertAlmostEqual(stats["both"]["Full_f1"], 1.0)
        self.assertEqual(stats["absent"]["Full_right"], 1)


if __name__ == "__main__":
    unittest.main()
