import copy
import contextlib
import hashlib
import json
import math
import sys
import unittest
from unittest import mock

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import sic_refine as R
import sic_optimize as O


ARMS = list(R.SOURCE_ARMS)
KEPT = list(range(9))


class FakeTorch:
    def __init__(self):
        self.cuda = mock.Mock()
        self.cuda.empty_cache = mock.Mock()

    def no_grad(self):
        return contextlib.nullcontext()


class FakeS:
    def pool_from_blocks(self, manager, blocks):
        return object(), object()


class FakeE:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def score_probes(self, manager, probes, pool, batch, mask):
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return copy.deepcopy(result)


def score_result(probes, *, correct=None, gold_loss=1.0, n=None):
    questions = [p.question for p in probes]
    return {"correct": dict(correct or {q: True for q in questions}),
            "n": len(questions) if n is None else n,
            "gold_loss": gold_loss}


class SicRefineTests(unittest.TestCase):
    def make_fixture(self):
        plans = []
        records = []
        windows = []
        discovery_count = [24, 24, 24, 25]
        for row, count in enumerate(discovery_count):
            discovery = []
            development = []
            for i in range(count):
                discovery.append({"question": "d%d_%d" % (row, i),
                                  "answer": "a%d_%d" % (row, i),
                                  "key": "dk%d_%d" % (row, i)})
            for i in range(50):
                development.append({"question": "v%d_%d" % (row, i),
                                    "answer": "b%d_%d" % (row, i),
                                    "key": "vk%d_%d" % (row, i)})
            plans.append({"row": row, "start": 30,
                          "context_sha256": hashlib.sha256(("context%d" % row).encode()).hexdigest(),
                          "discovery": discovery, "development": development,
                          "confirmation": [{"question": "confirmation%d" % row}]})
            arm_values = {name: {x["question"]: (i % 3 != 0)
                                 for i, x in enumerate(development)} for name in ARMS}
            for x in development:
                q = x["question"]
                records.append(dict(x, row=row, full=(q.endswith("_1")),
                                    no_memory=False, eligible=(q.endswith("_1")),
                                    arms={name: arm_values[name][q] for name in ARMS}))
            matrix = [[float((i + j) % 19 + 1) for j in range(19)] for i in range(count)]
            windows.append({"row": row, "kept": {name: list(KEPT) for name in ARMS},
                            "energy": matrix, "taylor": matrix,
                            "discovery_losses": [1.0] * count,
                            "gold_loss": {name: 2.5 for name in ARMS},
                            "repeat_equal": True})
        plan = {"seed": 1, "plans": plans,
                "protocol": {"window_blocks": 19, "target_blocks": 9}}
        plan_bytes = json.dumps(plan, sort_keys=True).encode()
        source = {"plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
                  "stage": "develop", "selected": "raw", "windows": windows,
                  "records": records, "confirmation": {"sourceplanhash": "ignored"},
                  "metadata": {"confirmation": "ignored"}}
        source["eligible"] = {name: O.stats(records, name, True) for name in ARMS}
        source["all"] = {name: O.stats(records, name, False) for name in ARMS}
        source_bytes = json.dumps(source, sort_keys=True).encode()
        return plan, source, plan_bytes, source_bytes

    def test_finite_loss_and_checked_blocks(self):
        for value in (0, 1, 1.5):
            self.assertTrue(R.finite_loss(value))
        for value in (True, False, -1, float("nan"), float("inf"), "1", None):
            self.assertFalse(R.finite_loss(value))
        self.assertEqual(R.checked_blocks(KEPT), KEPT)
        for bad in ([0] * 9, list(range(8)), list(range(10)),
                    [0, 1, 2, 3, 4, 5, 6, 7, 19],
                    [0, 1, 2, 3, 4, 5, 6, 8, 7],
                    [0, 1, 2, 3, 4, 5, 6, 7, True]):
            with self.assertRaises(ValueError):
                R.checked_blocks(bad)

    def test_validate_inputs_complete_fixture_and_mutations(self):
        plan, source, plan_bytes, source_bytes = self.make_fixture()
        result = R.validate_inputs(plan_bytes, source_bytes)
        self.assertEqual(result[2], hashlib.sha256(plan_bytes).hexdigest())

        cases = []
        changed_plan = copy.deepcopy(plan)
        changed_plan["plans"][0]["answer"] = "unused"
        cases.append((json.dumps(changed_plan, sort_keys=True).encode(), source_bytes, "hash"))
        missing_window = copy.deepcopy(source)
        missing_window["windows"].pop()
        cases.append((plan_bytes, json.dumps(missing_window, sort_keys=True).encode(), "window"))
        missing_record = copy.deepcopy(source)
        missing_record["records"].pop()
        cases.append((plan_bytes, json.dumps(missing_record, sort_keys=True).encode(), "record"))
        bad_metadata = copy.deepcopy(source)
        bad_metadata["records"][0]["answer"] = "changed"
        cases.append((plan_bytes, json.dumps(bad_metadata, sort_keys=True).encode(), "metadata"))
        bad_eligible = copy.deepcopy(source)
        bad_eligible["records"][1]["eligible"] = False
        cases.append((plan_bytes, json.dumps(bad_eligible, sort_keys=True).encode(), "eligible"))
        bad_score = copy.deepcopy(source)
        bad_score["windows"][0]["gold_loss"]["raw"] = float("nan")
        cases.append((plan_bytes, json.dumps(bad_score, sort_keys=True).encode(), "score"))
        bad_kept = copy.deepcopy(source)
        bad_kept["windows"][0]["kept"]["raw"] = [0] * 9
        cases.append((plan_bytes, json.dumps(bad_kept, sort_keys=True).encode(), "kept"))
        conflict = copy.deepcopy(source)
        conflict["windows"][0]["gold_loss"]["raw"] = 3.5
        cases.append((plan_bytes, json.dumps(conflict, sort_keys=True).encode(), "conflict"))
        fake_stats = copy.deepcopy(source)
        fake_stats["all"]["raw"]["wrong"] += 1
        cases.append((plan_bytes, json.dumps(fake_stats, sort_keys=True).encode(), "stats"))
        for pbytes, sbytes, label in cases:
            with self.subTest(label=label), self.assertRaises((ValueError, KeyError, json.JSONDecodeError)):
                R.validate_inputs(pbytes, sbytes)

        # Confirmation content is intentionally outside the validation contract.
        altered_plan = copy.deepcopy(plan)
        altered_plan["plans"][0]["confirmation"] = [{"question": "deleted"}]
        altered_plan_bytes = json.dumps(altered_plan, sort_keys=True).encode()
        altered_source = copy.deepcopy(source)
        altered_source["plan_sha256"] = hashlib.sha256(altered_plan_bytes).hexdigest()
        altered_source["confirmation"] = {"sourceplanhash": "changed"}
        altered_source_bytes = json.dumps(altered_source, sort_keys=True).encode()
        R.validate_inputs(altered_plan_bytes, altered_source_bytes)

    def test_verify_reference_exact_and_question_level_changes(self):
        expected = {"q1": True, "q2": False, "q3": True}
        R.verify_reference(dict(expected), expected, "same")
        flipped = dict(expected)
        flipped["q1"] = False
        swapped = {"q1": False, "q2": True, "q3": True}
        for changed in (flipped, swapped):
            with self.assertRaises(RuntimeError):
                R.verify_reference(changed, expected, "changed")
        missing = dict(expected)
        del missing["q2"]
        with self.assertRaises(RuntimeError):
            R.verify_reference(missing, expected, "missing")

    def test_score_blocks_repeat_and_release(self):
        probes = [mock.Mock(question="q1"), mock.Mock(question="q2")]
        good = score_result(probes, correct={"q1": True, "q2": False})
        for repeat, calls in ((False, 1), (True, 2)):
            torch = FakeTorch()
            scorer = FakeE([good, good])
            result = R.score_blocks(mock.Mock(), KEPT, probes, scorer, FakeS(), torch,
                                    repeat=repeat)
            self.assertEqual(result["repeat_equal"], repeat)
            self.assertEqual(result["correct"], good["correct"])
            self.assertEqual(torch.cuda.empty_cache.call_count, 1)
            self.assertEqual(scorer.calls, calls)

        bad_cases = [score_result(probes, correct={"q1": True}),
                     score_result(probes, gold_loss=float("nan"))]
        for bad in bad_cases:
            torch = FakeTorch()
            with self.assertRaises(ValueError):
                R.score_blocks(mock.Mock(), KEPT, probes, FakeE([bad]), FakeS(), torch, repeat=False)
            self.assertEqual(torch.cuda.empty_cache.call_count, 1)

        first = score_result(probes, correct={"q1": True, "q2": False})
        second = score_result(probes, correct={"q1": False, "q2": False})
        torch = FakeTorch()
        with self.assertRaises(RuntimeError):
            R.score_blocks(mock.Mock(), KEPT, probes, FakeE([first, second]), FakeS(), torch, repeat=True)
        self.assertEqual(torch.cuda.empty_cache.call_count, 1)


if __name__ == "__main__":
    unittest.main()
