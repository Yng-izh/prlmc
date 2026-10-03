"""CPU-only regression checks; no torch/model imports or remote experiments."""
import json
from pathlib import Path
import tempfile
import unittest

from native_qa_compare import atomic_json, exact_match, native_pairs, split_context, summarize


class NativeQAComparisonTests(unittest.TestCase):
    def test_aliases_and_ids(self):
        pairs = native_pairs({"questions": ["Q1", "Q2"],
                              "answers": [["A1", "Alias1"], "A2"],
                              "metadata": {"qa_pair_ids": ["native1", "native2"]}})
        self.assertEqual(pairs[0]["answers"], ["A1", "Alias1"])
        self.assertEqual(pairs[1]["id"], "native2")

    def test_misaligned_gold_fails(self):
        with self.assertRaises(ValueError):
            native_pairs({"questions": ["Q"], "answers": []})

    def test_misaligned_ids_fail(self):
        with self.assertRaises(ValueError):
            native_pairs({"questions": ["Q"], "answers": ["A"],
                          "metadata": {"qa_pair_ids": []}})

    def test_empty_alias_fails(self):
        with self.assertRaises(ValueError):
            native_pairs({"questions": ["Q"], "answers": [[]]})

    def test_lossless_chunks_and_template_budget(self):
        text = "Instructions\n1. Fact one.\n2. Conflicting fact two.\n" * 3
        render = lambda chunk: "PREFIX:" + chunk + ":SUFFIX"
        chunks = split_context(text, len, render, 35)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(render(c)) <= 35 for c in chunks))

    def test_template_overflow_fails(self):
        with self.assertRaises(ValueError):
            split_context("text", len, lambda c: "long prefix" + c, 3)

    def test_alias_em(self):
        self.assertTrue(exact_match(" The Philippines. ", ["Philippines"]))
        self.assertTrue(exact_match("alias", ["first", "Alias"]))
        self.assertFalse(exact_match("Italy because of reasons", ["Italy"]))

    def test_paired_metrics(self):
        pairs = [{"full_correct": a, "compressed_correct": b}
                 for a, b in [(True, True), (True, False), (False, True), (False, False)]]
        report = summarize(pairs)
        self.assertEqual((report["lost"], report["gained"], report["retained"]), (1, 1, 1))
        self.assertEqual(report["net_change"], 0)
        self.assertEqual(report["full_em"], 0.5)

    def test_atomic_json_preserves_old_result_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            atomic_json(path, {"status": "valid"})
            with self.assertRaises(TypeError):
                atomic_json(path, {"bad": object()})
            self.assertEqual(json.loads(path.read_text()), {"status": "valid"})
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
