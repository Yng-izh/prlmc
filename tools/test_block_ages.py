"""CPU contracts for generational block ages and the quota selector (v7 C1/C2).

Two things are being pinned here:

* the first-round invariance -- with no age information the selector must return
  the pre-v7 blocks *bit for bit*, not merely something close;
* the age bookkeeping -- the pool ages must follow the same physical moves the
  slots make, so a compaction can never launder old content into the young
  generation or lose the tail block's age.

The fixtures are the tiny CPU readers the other contract files use; they are
never experiment data.
"""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from methods.prlmc.consolidate import _greedy_diverse, _select_block_indices
from methods.prlmc.native_skip_merge import SupervisedDSlotMerge
from natural_merge_runtime import MergeStream, install, reset, state_digest
from test_natural_merge_v2 import tiny_manager
from test_public_retention import write_record


def selector_source(seed=3):
    """The exact input appendix A of the v7 contract pins the vectors on."""
    torch.manual_seed(seed)
    source = torch.randn(4, 38, 8)
    weights = torch.randn(38).abs()
    goal = torch.randn(8)
    return source, weights, goal


def ages_with(blocks, old, new):
    """Ages for `blocks` blocks: the first `old` old, the rest `new`."""
    ages = torch.zeros(blocks, dtype=torch.long)
    ages[:old] = new
    return ages


class FirstRoundInvariance(unittest.TestCase):
    def test_first_round_bit_identical(self):
        source, weights, goal = selector_source()
        recency_heavy = {"importance": .1, "recency": .9, "goal": 0, "diversity": 0}
        golden = [
            ((None, None, None), [9, 10, 11, 12, 13, 14, 15, 17, 18]),
            ((weights, None, None), [4, 9, 10, 12, 13, 15, 16, 17, 18]),
            ((weights, goal, None), [4, 7, 9, 10, 12, 13, 15, 16, 17]),
            ((weights, goal, recency_heavy), [10, 11, 12, 13, 14, 15, 16, 17, 18]),
        ]
        zeros = torch.zeros(19, dtype=torch.long)
        for (w, g, sw), expected in golden:
            without = _select_block_indices(source, 18, 2, w, g, sw)
            with_zeros = _select_block_indices(source, 18, 2, w, g, sw, block_ages=zeros)
            self.assertTrue(torch.equal(without, with_zeros))
            self.assertEqual(without.tolist(), expected)
            # Deterministic and device-stable: the selector carries no RNG.
            self.assertTrue(torch.equal(without, _select_block_indices(source, 18, 2, w, g, sw)))

    def test_zero_ages_cannot_move_the_quota_path(self):
        """The generational path with all-zero ages equals the plain greedy path.

        This is the second guarantee: even if the early exit were removed,
        `0.75 ** 0 == 1.0` exactly and the quota mask never fires, so the picks
        are unchanged.
        """
        self.assertEqual(float(torch.pow(torch.tensor(0.75), torch.tensor(0.0))), 1.0)
        for seed in range(5):
            torch.manual_seed(seed)
            score = torch.rand(19)
            vectors = torch.nn.functional.normalize(torch.randn(19, 4), dim=-1)
            self.assertEqual(
                _greedy_diverse(score, vectors, 9, .1),
                _greedy_diverse(score, vectors, 9, .1,
                                group=torch.zeros(19, dtype=torch.bool), group_cap=4),
            )
            # A quota that is already exhausted and has no young block left must
            # backfill instead of shortening the output.
            self.assertEqual(
                len(_greedy_diverse(score, vectors, 9, .1,
                                    group=torch.ones(19, dtype=torch.bool), group_cap=0)),
                9,
            )


class Quota(unittest.TestCase):
    def test_old_quota_cap(self):
        torch.manual_seed(5)
        source = torch.randn(4, 38, 8)
        ages = torch.zeros(19, dtype=torch.long)
        ages[:14] = 1  # 14 old, 5 young
        selected = _select_block_indices(source, 18, 2, None, None, None, block_ages=ages)
        old = int((ages[selected] > 0).sum())
        self.assertEqual(selected.numel(), 9)
        self.assertLessEqual(old, 4)
        self.assertGreaterEqual(9 - old, 5)
        self.assertEqual(sorted(selected.tolist()), selected.tolist())

    def test_backfill_when_new_scarce(self):
        torch.manual_seed(6)
        source = torch.randn(4, 38, 8)
        ages = torch.ones(19, dtype=torch.long)
        ages[-2:] = 0  # only 2 young blocks
        selected = _select_block_indices(source, 18, 2, None, None, None, block_ages=ages)
        self.assertEqual(selected.numel(), 9)
        self.assertEqual(int((ages[selected] > 0).sum()), 7)
        self.assertEqual(int((ages[selected] == 0).sum()), 2)

    def test_quota_is_exhaustive_at_defaults(self):
        """The cap is 4 of 9 at the registered defaults, not a tunable accident."""
        for old in (1, 4, 9, 18):
            torch.manual_seed(7 + old)
            source = torch.randn(4, 38, 8)
            ages = ages_with(19, old, 1)
            young = 19 - old
            selected = _select_block_indices(source, 18, 2, None, None, None, block_ages=ages)
            self.assertEqual(selected.numel(), 9)
            taken_old = int((ages[selected] > 0).sum())
            taken_young = int((ages[selected] == 0).sum())
            self.assertEqual(taken_old + taken_young, 9)
            if young >= 5:
                self.assertLessEqual(taken_old, 4)
                self.assertGreaterEqual(taken_young, 5)
            else:
                # Backfill: the young generation is exhausted, so the remaining
                # seats go to old blocks and the output keeps its width.
                self.assertEqual(taken_young, young)
                self.assertEqual(taken_old, 9 - young)

    def test_age_decay_monotone(self):
        """Same score, different age: the younger block wins the seat."""
        torch.manual_seed(9)
        source = torch.ones(4, 38, 8)
        weights = torch.zeros(38)
        weights[6:8] = 1.  # block 3
        weights[10:12] = 1.  # block 5
        ages = torch.zeros(19, dtype=torch.long)
        ages[3] = 5
        one_seat = dict(target=2, block=2, weights=weights, goal_direction=None,
                        select_weights={"importance": 1., "recency": 0., "goal": 0.,
                                        "diversity": 0.},
                        block_ages=ages, max_old_ratio=1.)
        self.assertEqual(_select_block_indices(source, **one_seat).tolist(), [5])
        ages[3], ages[5] = 0, 5
        self.assertEqual(_select_block_indices(source, **one_seat).tolist(), [3])

    def test_slot_level_ages_are_reduced_per_block(self):
        torch.manual_seed(13)
        source = torch.randn(4, 38, 8)
        block_ages = torch.zeros(38, dtype=torch.long)
        block_ages[6:8] = 2  # block 3
        by_block = _select_block_indices(source, 18, 2, None, None, None,
                                         block_ages=torch.tensor([0, 0, 0, 2] + [0] * 15))
        by_slot = _select_block_indices(source, 18, 2, None, None, None, block_ages=block_ages)
        self.assertTrue(torch.equal(by_block, by_slot))
        with self.assertRaises(ValueError):
            _select_block_indices(source, 18, 2, None, None, None,
                                  block_ages=torch.zeros(5, dtype=torch.long))
        with self.assertRaises(ValueError):
            _select_block_indices(source, 18, 2, None, None, None,
                                  block_ages=torch.full((19,), -1, dtype=torch.long))


class AgeBookkeeping(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.m = tiny_manager()  # total 80 slots, block 2 -> 40 blocks, budget 40
        self.assertEqual(self.m.num_blocks, 40)

    def test_initial_ages_are_zero(self):
        self.assertEqual(int(self.m._block_ages.sum()), 0)
        self.assertEqual(self.m.window_block_ages(0, 38).numel(), 19)
        self.assertEqual(int(self.m.window_block_ages(0, 38).sum()), 0)

    def test_roll_mapping(self):
        ages = self.m._block_ages
        ages[0], ages[1], ages[2], ages[3] = 0, 1, 2, 0
        self.m.note_append(1, 0)
        # old[1] is now at index 0, old[2] at 1, old[3] at 2, and the new block
        # (index 3) is young.
        self.assertEqual(self.m._block_ages[:4].tolist(), [1, 2, 0, 0])
        self.assertEqual(self.m.age_append_events, 1)

    def test_evict_then_append(self):
        ages = self.m._block_ages
        ages[0], ages[1], ages[2], ages[3] = 0, 1, 2, 0
        self.m.note_append(1, 1)
        # Dropping the oldest and appending a new block: survivors shift left by
        # one and the tail is young. Never a wraparound copy of the head.
        self.assertEqual(self.m._block_ages[:4].tolist(), [1, 2, 0, 0])

    def test_valid_span_tracks_used(self):
        self.m.used = 6
        first, end = self.m._valid_age_span()
        self.assertEqual((first, end), (37, 40))
        self.m.used = 0
        self.assertEqual(self.m._valid_age_span(), (40, 40))

    def test_merge_permutation(self):
        ages = self.m._block_ages
        for index, age in enumerate([0, 1, 2, 3, 4]):
            ages[index] = age
        self.m.note_block_move(1)
        # Block 1's content lands at the physical tail, everything after it
        # shifts one block left, and the content before it is untouched.
        self.assertEqual(self.m._block_ages[:5].tolist(), [0, 2, 3, 4, 0])
        self.assertEqual(int(self.m._block_ages[-1]), 1)

    def test_reset_clears_ages(self):
        self.m._block_ages[3] = 7
        self.m.reset()
        self.assertEqual(int(self.m._block_ages.sum()), 0)
        self.m._block_ages[3] = 7
        self.m.reset_block_ages()
        self.assertEqual(int(self.m._block_ages.sum()), 0)

    def test_snapshot_restore_carries_ages(self):
        self.m._block_ages[5] = 3
        state = self.m.snapshot()
        self.m.note_append(2, 0)
        self.assertNotEqual(self.m._block_ages.tolist(), state["block_ages"].tolist())
        self.m.restore(state)
        self.assertEqual(self.m._block_ages.tolist(), state["block_ages"].tolist())

    def test_ages_are_not_pool_state(self):
        """D-06: ages must never reach a state dict or a determinism digest."""
        merge = SupervisedDSlotMerge(1, 4, block=2, rank=2, num_groups=1,
                                     reference_layer=0, topk=2)
        digest = state_digest(merge)
        self.m._block_ages[2] = 5
        self.assertEqual(digest, state_digest(merge))
        for state in (merge.state_dict(), self.m.model.state_dict()):
            self.assertFalse([name for name in state if "age" in name.lower()])
        # They are diagnostic state: reported by `stats`, carried by the
        # rollback snapshot, absent from anything that is hashed.
        self.assertIn("block_age_max", self.m.stats())
        self.assertIn("block_ages", self.m.snapshot())
        self.assertFalse(hasattr(self.m, "state_dict"))


class InstallAges(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.m = tiny_manager()
        self.m.used = 40  # 20 resident blocks: window 19 + tail 1
        self.m._tile_free_region()
        self.m.reset_block_ages()

    def test_install_ages(self):
        m = self.m
        first = (m.total - m.used) // m.block  # 20
        m._block_ages[first + 1] = 3
        m._block_ages[first + 3] = 2
        m._block_ages[first + 5] = 1
        m._block_ages[first + 19] = 4  # tail block
        candidate = torch.randn(1, 6, 4)  # three carriers
        install(m, candidate, 38, selected_blocks=[1, 3, 5])
        out = m._block_ages
        self.assertEqual(m.used, 8)
        first_new = (m.total - m.used) // m.block  # 36
        # Carrier age + 1, in output order.
        self.assertEqual(out[first_new:first_new + 3].tolist(), [4, 3, 2])
        # The uncompressed tail keeps its age and moves with its content.
        self.assertEqual(int(out[first_new + 3]), 4)
        self.assertEqual(int(out.sum()), 4 + 3 + 2 + 4)
        self.assertEqual(out.numel(), 40)

    def test_install_without_selection_leaves_ages_untouched(self):
        m = self.m
        before = m._block_ages.clone()
        install(m, torch.randn(1, 6, 4), 38)
        self.assertTrue(torch.equal(m._block_ages, before))

    def test_note_merge_matches_install(self):
        m = self.m
        first = (m.total - m.used) // m.block
        m._block_ages[first + 2] = 5
        m._block_ages[first + 19] = 7
        m.note_merge([2], 19, target_blocks=1, used_before=40)
        first_new = (m.total - 4) // m.block  # one carrier + one tail block
        self.assertEqual(int(m._block_ages[first_new]), 6)
        self.assertEqual(int(m._block_ages[first_new + 1]), 7)
        self.assertEqual(int(m._block_ages.sum()), 13)


class ManagerWritePath(unittest.TestCase):
    """The managed write path keeps the ages aligned with `_place`."""

    def test_place_appends_roll_ages(self):
        torch.manual_seed(31)
        m = tiny_manager()
        self.assertEqual(m.cfg.write_mode, "append")  # always allocates a block
        for _ in range(3):
            m.write("x" * 32)
        first, end = m._valid_age_span()
        self.assertEqual(end - first, m.used // m.block)
        self.assertEqual(int(m._block_ages.sum()), 0)
        # Mark what is resident, then append once more: the survivors shift one
        # physical block left and the appended block is young.
        m._block_ages[first:end] = 1
        m.write("x" * 32)
        first_after, end_after = m._valid_age_span()
        self.assertEqual(end_after - first_after, end - first + 1)
        self.assertEqual(m._block_ages[first_after:end_after - 1].tolist(), [1] * (end - first))
        self.assertEqual(int(m._block_ages[end_after - 1]), 0)

    def test_short_chunk_does_not_desynchronise_ages(self):
        """A sub-block chunk must not roll the array by a whole block."""
        torch.manual_seed(37)
        m = tiny_manager()
        m.write("x" * 32)
        first, end = m._valid_age_span()
        before = m._block_ages.clone()
        # A chunk that encodes to fewer slots than one block still appends.
        m._place(torch.zeros(m.num_layers, 1, m.hidden))
        self.assertTrue(torch.equal(m._block_ages, before))
        self.assertEqual(m._valid_age_span()[1], end)


class StreamingRounds(unittest.TestCase):
    """Three real compaction rounds through the streaming path."""

    def setUp(self):
        torch.manual_seed(23)
        self.m = tiny_manager()
        merge = SupervisedDSlotMerge(1, 4, block=2, rank=2, num_groups=1,
                                     reference_layer=0, topk=2)
        self.stream = MergeStream(self.m, merge, "SharedMerge")
        reset(self.m, self.m._memory.clone())

    def resident(self):
        first, end = self.m._valid_age_span()
        return self.m._block_ages[first:end]

    def run_rounds(self, writes=41):
        """Write until three compactions have happened, recording each one."""
        m = self.m
        rounds = []
        for index in range(writes):
            ages_before = m._block_ages.clone()
            used_before = int(m.used)
            before = len(self.stream.events)
            self.stream.write(write_record(index), index)
            if len(self.stream.events) == before:
                continue
            event = self.stream.events[-1]
            first = (m.total - used_before) // m.block
            selected = event["selected_blocks"]
            rounds.append(dict(
                event=event,
                selected=selected,
                window_ages=ages_before[first:first + self.stream.window // m.block],
                resident=self.resident(),
            ))
        return rounds

    def test_three_rounds_are_monotone_and_capped(self):
        m = self.m
        self.assertEqual((self.stream.window, self.stream.target), (38, 18))
        rounds = self.run_rounds()
        self.assertEqual(len(rounds), 3)
        for round_ in rounds:
            selected = round_["selected"]
            self.assertEqual(len(selected), 9)
            self.assertEqual(len(set(selected)), 9)
            self.assertEqual(round_["event"]["selected_blocks"], selected)
            # Selected carriers are blocks of the window the selector was given.
            self.assertTrue(all(0 <= index < 19 for index in selected))
            # The cap is evaluated against the ages the selector saw.
            old = int((round_["window_ages"][selected] > 0).sum())
            self.assertLessEqual(old, 4)
            self.assertGreaterEqual(9 - old, 5)
        # The age array never shrinks, and one round is enough for the carriers.
        self.assertEqual(int(rounds[0]["resident"].max()), 1)
        self.assertEqual(int((rounds[0]["resident"] == 1).sum()), 9)
        masses = [int(round_["resident"].sum()) for round_ in rounds]
        self.assertEqual(masses, sorted(masses))
        maxima = [int(round_["resident"].max()) for round_ in rounds]
        self.assertEqual(maxima, sorted(maxima))
        self.assertLessEqual(int((rounds[-1]["resident"] >= 2).sum()), 4)

    def test_repeated_selection_ages_one_round_at_a_time(self):
        """Deterministic monotone accumulation: carrier age + 1 per round."""
        m = self.m
        selected = [1, 3, 5, 7, 9, 11, 13, 15, 17]
        for round_ in range(3):
            m.used = 40  # window 19 blocks + one uncompressed tail block
            m.reset_block_ages()
            first = (m.total - m.used) // m.block
            for index in selected:
                m._block_ages[first + index] = round_
            m._tile_free_region()
            install(m, torch.randn(1, 18, 4), 38, selected_blocks=selected)
            first_new = (m.total - m.used) // m.block
            out = m._block_ages[first_new:first_new + 9]
            # The same nine carriers survive every round, so their age must be
            # the round number exactly: 1, 2, 3. Monotone, no reset to 0.
            self.assertEqual(out.tolist(), [round_ + 1] * 9)


class Geometry(unittest.TestCase):
    """C8/D-08: explicit window/target reach the stream, defaults do not move."""

    def stream(self, **kwargs):
        torch.manual_seed(29)
        manager = tiny_manager()
        merge = SupervisedDSlotMerge(1, 4, block=2, rank=2, num_groups=1,
                                     reference_layer=0, topk=2)
        return MergeStream(manager, merge, "SharedMerge", **kwargs), manager

    def test_defaults_are_the_pre_registered_table(self):
        stream, manager = self.stream()
        self.assertEqual((stream.window, stream.target), (38, 18))
        self.assertEqual((stream.window // manager.block, stream.target // manager.block),
                         (19, 9))
        for budget, expected in ((8, (9, 4)), (10, (11, 5)), (12, (11, 5)), (20, (19, 9))):
            stream, manager = self.stream(budget_blocks=budget)
            self.assertEqual((stream.window // manager.block, stream.target // manager.block),
                             expected)

    def test_explicit_geometry_wins_and_default_is_bit_identical(self):
        explicit, manager = self.stream(window_blocks=13, target_blocks=9)
        self.assertEqual((explicit.window // manager.block, explicit.target // manager.block),
                         (13, 9))
        # The 19/9 geometry passed explicitly must equal the default.
        passed, manager = self.stream(window_blocks=19, target_blocks=9)
        default, manager = self.stream()
        self.assertEqual((passed.window, passed.target), (default.window, default.target))

    def test_invalid_geometry_is_refused(self):
        for kwargs in (dict(window_blocks=9, target_blocks=9),
                       dict(window_blocks=13, target_blocks=14),
                       dict(target_blocks=21),
                       dict(window_blocks=41),
                       dict(window_blocks=0)):
            with self.assertRaises(ValueError):
                self.stream(**kwargs)

    def test_geometry_reaches_the_stream_selector(self):
        """13/9 really compacts 13 blocks into 9 at the registered cap."""
        stream, manager = self.stream(window_blocks=13, target_blocks=9)
        reset(manager, manager._memory.clone())
        for index in range(30):
            stream.write(write_record(index), index)
        self.assertEqual(manager.compaction_count, 0)  # streaming path, not manager
        self.assertTrue(stream.events)
        event = stream.events[0]
        self.assertEqual(event["source_slots"], 13 * manager.block)
        self.assertEqual(event["target_slots"], 9 * manager.block)
        self.assertEqual(len(event["selected_blocks"]), 9)


if __name__ == "__main__":
    unittest.main()
