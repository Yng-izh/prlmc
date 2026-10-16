import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from longbench_memoryllm_eval import MAX_NEW, PROMPTS, qa_f1, split_prompt


class Tokenizer:
    def __call__(self, text, **kwargs):
        return type("Encoded", (), {"input_ids": list(range(len(text)))})()


class LongBenchContracts(unittest.TestCase):
    def test_official_task_lengths_and_prompts(self):
        self.assertEqual(MAX_NEW["hotpotqa"], 32)
        self.assertEqual(MAX_NEW["qasper"], 128)
        self.assertEqual(MAX_NEW["multifieldqa_en"], 64)
        self.assertIn("{context}", PROMPTS["hotpotqa"])

    def test_fixed_two_k_query_suffix(self):
        _, all_ids, write_ids, query_ids = split_prompt(
            Tokenizer(), "hotpotqa", {"context": "x" * 10000, "input": "q"}, 8192)
        self.assertEqual(len(all_ids), 8192)
        self.assertEqual(len(query_ids), 2048)
        self.assertEqual(len(write_ids), 6144)
        self.assertEqual(max(map(len, [write_ids[i:i+512] for i in range(0, len(write_ids), 512)])), 512)

    def test_qa_f1(self):
        self.assertEqual(qa_f1("The Eiffel Tower", "Eiffel Tower"), 1.0)
        self.assertLess(qa_f1("wrong", "right"), 1.0)


if __name__ == "__main__":
    unittest.main()
