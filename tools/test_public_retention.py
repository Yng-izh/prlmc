"""CPU contract tests; fixtures below are never used as experiment data."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_public_retention import (choose_indices, digest, group_distractors,
                                      make_writes, nq_rows, processed_inputs, squad_rows)
from public_retention_eval import (ARMS, StreamingMemory, author_score, generate,
                                   E6D_SHA, isolated_query, summarize, validate_plan, shard_samples)
from verify_public_retention import verify
from run_public_retention_parallel import merge_reports
from methods.prlmc.config import PRLMCConfig
from methods.prlmc.consolidate import CarrierResidualCompactor, select_block_compact
from methods.prlmc.layout import ProtectedResidualMemory


class Tokenizer:
    eos_token_id = 1
    pad_token_id = 1

    def __call__(self, text, **kwargs):
        return types.SimpleNamespace(input_ids=[ord(c) for c in text])

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


class Reader(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_tokens = 2
        self.num_blocks = 30
        self.memory = torch.nn.Parameter(torch.randn(1, 60, 4), requires_grad=False)
        self.register_buffer("initialized", torch.tensor(1, dtype=torch.uint8))
        self.register_buffer("memory_active_mask", torch.ones(60, dtype=torch.bool))
        self.inputs = []

    def inject_memory(self, ids, context_attention_mask=None, update_memory=False):
        self.inputs.append(ids.tolist())
        value = float(ids[0, 0])
        delta = torch.full((1, 1, 2, 4), value)
        if update_memory:
            self.memory.data = torch.cat((self.memory[:, 2:], delta[0]), dim=1)
        return delta

    def generate(self, input_ids, **kwargs):
        self.generation_kwargs = kwargs
        self.memory.data.fill_(123.)
        self.initialized.zero_()
        return torch.cat((input_ids, torch.tensor([[ord("A")]])), dim=1)


def manager():
    model = Reader()
    cfg = PRLMCConfig(k_anchor=0, anchor_source="none", k_max=40,
        compact_mode="carrier_residual", active_memory_mask=True,
        native_filler_only=False, trigger="none", projection="none", basis_mode="none")
    m = ProtectedResidualMemory(model, Tokenizer(), cfg)
    compactor = CarrierResidualCompactor(num_layers=1, hidden=4, block=2, rank=2,
                                        gamma_max=.05).eval().requires_grad_(False)
    return m, compactor


def write_record(i):
    ids = [i + 1]
    return dict(token_ids=ids, token_sha256=digest(ids), role="related" if i == 0 else "unrelated")


class DataContracts(unittest.TestCase):
    def test_processed_release_original_order(self):
        with tempfile.TemporaryDirectory() as directory:
            dev, train = Path(directory) / "dev.json", Path(directory) / "train.json"
            data = [dict(context="untouched " + str(i), question="Q?", answer="A") for i in range(3)]
            dev.write_text(json.dumps(data))
            train.write_text(json.dumps(["original train first", "original train second"]))
            samples, groups = processed_inputs(dev, train, 2, 1)
            self.assertEqual([s["context"] for s in samples], [d["context"] for d in data[:2]])
            self.assertEqual(groups[0]["text"], "original train first")
            with self.assertRaises(ValueError):
                processed_inputs(dev, train, 4, 1)

    def test_fixed_unsorted_indices_and_duplicates(self):
        picked = choose_indices(iter([dict(id=i) for i in range(8)]), [7, 0, 4])
        self.assertEqual([r["id"] for r in picked], [7, 0, 4])
        with self.assertRaises(ValueError):
            choose_indices(iter([{}]), [5])

    def test_squad_keeps_originals_and_first_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "squad.json"
            p.write_text(json.dumps(dict(data=[dict(paragraphs=[dict(context="real original", qas=[
                dict(id="q", is_impossible=False, question="original?", answers=[dict(text="A"), dict(text="B")]),
                dict(id="no", is_impossible=True, question="no?", answers=[])])])])))
            rows = list(squad_rows(p, True))
            self.assertEqual((rows[0]["context"], rows[0]["question"], rows[0]["answer"]),
                             ("real original", "original?", "A"))
            self.assertEqual(len(rows), 1)

    def test_lossless_and_author_truncation(self):
        tok = Tokenizer()
        text = "X" * 1203
        full = make_writes(text, "related", lambda s: tok(s).input_ids, tok.decode, "lossless")
        self.assertEqual("".join(tok.decode(w["token_ids"]) for w in full), text)
        self.assertEqual(sum(w["truncated_tokens"] for w in full), 0)
        cropped = make_writes(text, "related", lambda s: tok(s).input_ids, tok.decode, "author512")
        self.assertEqual(cropped[0]["truncated_tokens"], 691)
        self.assertEqual(len(cropped[0]["token_ids"]), 512)

    def test_original_train_group_thresholds(self):
        tokenize = lambda s: list(s)
        texts = [dict(context="x" * 400) for _ in range(10)]
        groups = group_distractors(texts, 2, tokenize, "squad")
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]["sources"]), 2)
        with self.assertRaises(ValueError):
            group_distractors([], 1, tokenize, "squad")

    def test_nq_original_span_and_suspect_fallback(self):
        row = dict(example_id=3, question_text="Q", document_tokens=[
            dict(token="<p>", html_token=True), dict(token="A", html_token=False),
            dict(token="B", html_token=False)], annotations=[dict(yes_no_answer="NONE",
                long_answer=dict(start_token=0, end_token=3), short_answers=[dict(start_token=1, end_token=2)])])
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "nq.jsonl"
            p.write_text(json.dumps(row) + "\n")
            result = list(nq_rows(p, True, "author512"))[0]
            self.assertEqual((result["context"], result["answer"]), ("A B", "A"))
            row["document_text"] = "<p> A B"
            del row["document_tokens"]
            p.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "character"):
                list(nq_rows(p, True, "author512"))
            self.assertEqual(list(nq_rows(p, True, "lossless"))[0]["context"], "A B")

    def test_case_sensitive_author_score(self):
        self.assertTrue(author_score("Answer A</s>", " A</s> "))
        self.assertFalse(author_score("answer a", "A"))
        with self.assertRaises(ValueError):
            author_score("any", "</s>")


class StreamingContracts(unittest.TestCase):
    def test_shards_cover_original_samples_and_merge_rejects_duplicates(self):
        plan = dict(samples=[dict(sample_index=i) for i in range(100)])
        reports = []
        for index in range(4):
            chosen = shard_samples(plan, index, 4)
            self.assertEqual(len(chosen), 25)
            samples = [dict(sample_index=s["sample_index"], context_sha256=str(s["sample_index"]),
                arms={a: dict(status="complete", compressions=[{}],
                    trace=[dict(role="related", generation=dict(score=True))]) for a in ARMS}) for s in chosen]
            reports.append(dict(status="complete", identity={"same": True}, protocol={},
                physical_slots=60, matched_budget_slots=40, shared_state_sha256_before="same",
                shared_state_sha256_after="same", training_performed=False, optimizer_created=False,
                sharding=dict(index=index, count=4, sample_indices=[s["sample_index"] for s in chosen]),
                samples=samples, runtime=dict(gpu=index), costs=dict(elapsed_seconds=1., peak_allocated_bytes=100)))
        merged = merge_reports(plan, reports[::-1])
        self.assertEqual([s["sample_index"] for s in merged["samples"]], list(range(100)))
        self.assertEqual(merged["summary"]["n"], 100)
        with self.assertRaisesRegex(ValueError, "duplicate shard"):
            merge_reports(plan, reports[:3] + [reports[0]])
        changed = copy.deepcopy(reports)
        changed[1]["identity"] = {"same": False}
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            merge_reports(plan, changed)
        changed = copy.deepcopy(reports)
        changed[2]["samples"].pop()
        with self.assertRaisesRegex(ValueError, "membership"):
            merge_reports(plan, changed)
        with self.assertRaises(ValueError):
            shard_samples(plan, 4, 4)

    def test_saved_prediction_verifier_detects_corruption(self):
        sample = dict(sample_index=0, filtered_index=0, id="real-id", context="Original",
                      context_sha256=digest("Original"), question="Q", answer="A",
                      related_writes=[write_record(0)])
        protocol = dict(budget_blocks=20, window_blocks=19, target_blocks=9, anchor_blocks=0,
                        max_new_tokens=10, rank=32, gamma_max=.05, training=False,
                        arms=list(ARMS), do_sample=False, stop_strings=[])
        plan = dict(schema="public_retention/1", status="prepared", protocol=protocol,
                    samples=[sample], official_indices=[0], unrelated_writes=[])
        plan["content_sha256"] = digest(plan)
        arms = {a: dict(status="complete", compressions=[], trace=[dict(index=0, role="related",
            active_slots=2, input_sha256=digest([1]), generation=dict(token_ids=[65], prediction="A", score=True))]) for a in ARMS}
        report = dict(status="complete", protocol=protocol, training_performed=False, optimizer_created=False,
            identity=dict(checkpoint_sha256=E6D_SHA), shared_state_sha256_before="same", shared_state_sha256_after="same",
            physical_slots=60, matched_budget_slots=40, samples=[dict(sample_index=0, original_id="real-id",
            context_sha256=sample["context_sha256"], question="Q", original_answer="A", scoring_target="A</s>", arms=arms)])
        report["summary"] = summarize(report)
        self.assertEqual(verify(plan, report, Tokenizer())["status"], "passed")
        changed = copy.deepcopy(report)
        changed["samples"][0]["arms"]["SharedE6d"]["trace"][0]["generation"]["score"] = False
        with self.assertRaisesRegex(ValueError, "scoring mismatch"):
            verify(plan, changed, Tokenizer())
        changed_plan = copy.deepcopy(plan)
        changed_plan["samples"][0]["question"] = "changed"
        with self.assertRaisesRegex(ValueError, "integrity"):
            validate_plan(changed_plan)

    def test_short_writes_repeated_compression_same_budget(self):
        outcomes = {}
        for arm in ("Selection", "Recent", "SharedE6d"):
            m, compactor = manager()
            stream = StreamingMemory(m, compactor, arm)
            for i in range(43):
                stream.write(write_record(i), i)
            self.assertEqual(len(m.model.inputs), 43)  # No 16-token skip.
            self.assertEqual([e["before_write"] for e in stream.events], [20, 30, 40])
            self.assertLessEqual(stream.peak, 40)
            self.assertTrue(torch.equal(m._memory[:, -2:], torch.full((1, 2, 4), 43.)))
            outcomes[arm] = m._distinct().clone()
        # Zero-initialized residual has exactly the Selection carrier behavior.
        self.assertTrue(torch.equal(outcomes["Selection"], outcomes["SharedE6d"]))

    def test_nonzero_residual_is_applied(self):
        m, compactor = manager()
        with torch.no_grad():
            compactor.up.weight.normal_(std=.2)
        stream = StreamingMemory(m, compactor, "SharedE6d")
        for i in range(20):
            stream.write(write_record(i), i)
        start = m.total - m.used
        source = m._memory[:, start:start + stream.window].clone()
        weights = m._window_importance(start, stream.window)
        selection = select_block_compact(source, stream.target, block=m.block, weights=weights,
                                         select_weights=m.select_weights)
        stream.compact(20)
        self.assertFalse(torch.equal(m._distinct()[:, :stream.target], selection))

    def test_full_capacity_refuses_omission(self):
        m, compactor = manager()
        stream = StreamingMemory(m, compactor, "Full")
        for i in range(30):
            self.assertIsNotNone(stream.write(write_record(i), i))
        self.assertIsNone(stream.write(write_record(30), 30))
        self.assertEqual(len(m.model.inputs), 30)
        self.assertEqual(stream.events, [])

    def test_query_restores_mutation_and_exceptions(self):
        m, _ = manager()
        original = m.model.memory.detach().clone()
        binding = m.model.memory
        for throws in (False, True):
            try:
                with isolated_query(m.model):
                    m.model.memory = torch.nn.Parameter(torch.zeros_like(original))
                    m.model.initialized.zero_()
                    m.model.memory_active_mask.zero_()
                    if throws:
                        raise RuntimeError("generation failed")
            except RuntimeError:
                pass
            self.assertIs(m.model.memory, binding)
            self.assertTrue(torch.equal(m.model.memory, original))
            self.assertEqual(int(m.model.initialized), 1)
            self.assertTrue(m.model.memory_active_mask.all())

    def test_generation_official_protocol(self):
        m, _ = manager()
        old = m.model.memory.detach().clone()
        result = generate(m.model, Tokenizer(), dict(prompt_ids=[81]), "A</s>")
        self.assertTrue(result["score"])
        self.assertEqual(m.model.generation_kwargs["max_new_tokens"], 10)
        self.assertNotIn("stop_strings", m.model.generation_kwargs)
        self.assertEqual(m.model.generation_kwargs["attention_mask"].shape[1], 61)
        self.assertTrue(torch.equal(old, m.model.memory))

    def test_reference_gate_and_strict_rescue(self):
        def report(compressed, full, selection, shared):
            arms = {a: dict(status="complete", compressions=[{}] if compressed else [],
                trace=[dict(role="related", generation=dict(score=full)),
                       dict(role="unrelated", generation=dict(score=(selection if a == "Selection" else shared if a == "SharedE6d" else full)))]) for a in ARMS}
            return dict(samples=[dict(sample_index=0, arms=arms)])
        r = summarize(report(False, True, False, True))
        self.assertEqual(r["validity"], "not_applicable_no_compression")
        self.assertFalse(r["conclusion_allowed"])
        r = summarize(report(True, False, False, True))
        self.assertEqual(r["validity"], "reader_reference_failed")
        self.assertEqual(r["strict_rescued"], [])
        r = summarize(report(True, True, False, True))
        self.assertEqual(r["strict_rescued"], [0])


if __name__ == "__main__":
    unittest.main()
