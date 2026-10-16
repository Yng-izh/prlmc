import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from longbench_memoryllm_eval import MAX_NEW, PROMPTS, qa_f1, split_prompt, reset_model


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
        self.assertEqual(qa_f1("", ""), 0.0)
        self.assertEqual(qa_f1("the", "a"), 0.0)

    def test_short_input_has_no_writes(self):
        _, ids, writes, query = split_prompt(
            Tokenizer(), "hotpotqa", {"context": "x", "input": "q"}, 8192)
        self.assertEqual(writes, [])
        self.assertEqual(ids, query)

    def test_16k_chronological_chunks_are_lossless_after_truncation(self):
        _, ids, writes, query = split_prompt(
            Tokenizer(), "qasper", {"context": "x"*20000, "input": "q"}, 16384)
        self.assertEqual(len(writes), 14336)
        self.assertEqual(writes+query, ids)
        self.assertEqual(query[-1], ids[-1])

    def test_memory_reset_restores_values_initialized_and_mask(self):
        import torch
        import types
        initial = torch.randn(2, 8, 3)
        model = types.SimpleNamespace(memory=torch.zeros_like(initial),
            initialized=torch.tensor(0), memory_active_mask=torch.zeros(8, dtype=torch.bool))
        reset_model(model, initial, torch.tensor(1))
        self.assertTrue(torch.equal(model.memory, initial))
        self.assertEqual(model.initialized.item(), 1)
        self.assertTrue(model.memory_active_mask.all().item())


if __name__ == "__main__":
    unittest.main()
