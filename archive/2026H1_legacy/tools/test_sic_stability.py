import copy
import unittest
from sic_stability import calibration_gate, episode_summary, stability_select


class StabilityTests(unittest.TestCase):
    def test_repeat_and_scaling(self):
        rows = [[100., 1., 0., 0.], [2., 1., 0., 0.]]
        original = copy.deepcopy(rows)
        first = stability_select(rows, 2, seed=17)
        self.assertEqual(first, stability_select(rows, 2, seed=17))
        self.assertEqual(first['kept'], [0, 1])
        scaled = [[x * 7 for x in rows[0]], [x / 11 for x in rows[1]]]
        self.assertEqual(first['kept'], stability_select(scaled, 2, seed=17)['kept'])
        self.assertEqual(rows, original)
        self.assertAlmostEqual(sum(first['frequencies']), 2)

    def test_ties_are_seeded_not_always_recent(self):
        choices = [stability_select([[0.] * 19], 9, seed=s)['kept'] for s in range(5)]
        self.assertGreater(len({tuple(x) for x in choices}), 1)
        self.assertTrue(all(x == sorted(set(x)) and len(x) == 9 for x in choices))

    def test_validation(self):
        for rows in ([], [[]], [[1., -1.]], [[float('nan')]], [[1.], [1., 2.]]):
            with self.assertRaises(ValueError):
                stability_select(rows, 1)
        for budget in (0, 3, True, 1.5):
            with self.assertRaises(ValueError):
                stability_select([[1., 2., 3.]], budget)

    def test_gate_fallback_and_gain(self):
        self.assertFalse(calibration_gate([True] * 8, [True] * 8)['accept'])
        self.assertFalse(calibration_gate([False] * 4, [True] * 4)['accept'])
        self.assertTrue(calibration_gate([False] * 5, [True] * 5)['accept'])
        self.assertFalse(calibration_gate([True] * 5, [False] * 5)['accept'])
        self.assertFalse(calibration_gate([False] * 5, [True] * 5)['statistical_guarantee'])

    def test_cluster_summary_uses_episode_not_questions(self):
        record = dict(arms=dict(select=dict(em=False), candidate=dict(em=True)))
        episodes = [dict(evaluation=[copy.deepcopy(record)] * 50) for _ in range(4)]
        result = episode_summary(episodes, 'candidate', bootstraps=200)
        self.assertEqual(result['sign_flip_p'], .125)
        self.assertEqual(result['rescued'], 200)
        self.assertEqual(result['episode_bootstrap_ci95'], [1., 1.])
        self.assertEqual(result['macro_em_delta'], 1.)

    def test_mixed_zero_effect(self):
        records = [dict(arms=dict(select=dict(em=True), candidate=dict(em=True)))]
        result = episode_summary([dict(evaluation=records)] * 4, 'candidate', bootstraps=200)
        self.assertEqual(result['sign_flip_p'], 1.)
        self.assertEqual(result['macro_em_delta'], 0.)


if __name__ == '__main__':
    unittest.main()
