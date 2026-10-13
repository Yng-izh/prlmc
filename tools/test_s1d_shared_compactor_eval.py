"""CPU contract tests for zero-training shared compactor evaluation."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

from s1d_shared_compactor_eval import DEVELOPMENT_IDS, PROTOCOL, arm_summary, choose_development, validate_write_plan

HERE = Path(__file__).parent


class SharedEvalContract(unittest.TestCase):
    def test_only_development_episode_ids_are_accepted(self):
        test_ids = [1, 3, 5, 6, 7, 9, 10, 11, 12, 13, 16, 19, 20, 21, 22, 23]
        plan = dict(status="completed",
                    episodes=[dict(episode=i, partition="development") for i in DEVELOPMENT_IDS]
                    + [dict(episode=i, partition="test") for i in test_ids])
        for episode in DEVELOPMENT_IDS:
            self.assertEqual(choose_development(plan, episode)["episode"], episode)
        with self.assertRaises(ValueError):
            choose_development(plan, 1)

    def test_write_shape_and_hash(self):
        lines = ["%d. fact %d" % (i + 1, i) for i in range(410)]
        import hashlib
        episode = dict(lines=lines, context_sha256=hashlib.sha256("\n".join(lines).encode()).hexdigest())
        checked, chunks = validate_write_plan(episode)
        self.assertEqual(checked, lines)
        self.assertEqual([len(chunk) for chunk in chunks], [30] + [20] * 19)

    def test_arm_summary_is_paired(self):
        maps = {name: {"q%d" % i: (name in ("Full", "SharedE6d") or i % 2 == 0)
                       for i in range(64)} for name in PROTOCOL["arms"]}
        summary = arm_summary(maps)
        self.assertEqual(summary["shared_vs_selection"]["net"], 32)
        bad = dict(maps)
        bad["SharedE6d"] = {"q0": True}
        with self.assertRaises(ValueError):
            arm_summary(bad)

    def test_runtime_contains_no_training_primitives(self):
        source = (HERE / "s1d_shared_compactor_eval.py").read_text()
        tree = ast.parse(source)
        forbidden = {"backward", "step", "zero_grad", "AdamW", "optim"}
        hits = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in forbidden:
                hits.append(node.attr)
            if isinstance(node, ast.Name) and node.id in forbidden:
                hits.append(node.id)
        self.assertEqual(hits, [], hits)
        self.assertIn('basis_constructed=False', source)
        self.assertIn('alpha_adapted=False', source)

    def test_report_artifact_is_not_overwritten_by_partial_predictions(self):
        source = (HERE / "s1d_shared_compactor_eval.py").read_text()
        self.assertNotIn('atomic_json(output_path', source)
        self.assertIn('status="complete"', source)


if __name__ == "__main__":
    unittest.main()

class SharedEvalTensorRuntime(unittest.TestCase):
    def test_real_candidate_pools_and_zero_carrier_match(self):
        import torch
        from types import SimpleNamespace
        from s1d_shared_compactor_eval import _candidate_pool, state_digest
        from methods.prlmc.consolidate import CarrierResidualCompactor
        torch.manual_seed(1)
        memory = torch.randn(2, 48, 8)
        manager = SimpleNamespace(_memory=memory, total=48, used=38, block=2,
            num_layers=2, hidden=8,
            layout=SimpleNamespace(total=48, k_anchor=4), active_memory_mask=True,
            goal_direction=None, select_weights=None,
            _window_importance=lambda start, width: torch.linspace(0, 1, width))
        compactor = CarrierResidualCompactor(num_layers=2, hidden=8, block=2, rank=3, gamma_max=.05)
        compactor.requires_grad_(False)
        before = state_digest(compactor)
        with torch.no_grad():
            selection, mask = _candidate_pool(manager, compactor, 'Selection', 'cpu')
            shared, shared_mask = _candidate_pool(manager, compactor, 'SharedE6d', 'cpu')
        self.assertEqual(selection.shape, memory.shape)
        self.assertEqual(int(mask.sum()), 22)
        self.assertTrue(torch.equal(selection, shared))
        self.assertTrue(torch.equal(mask, shared_mask))
        self.assertEqual(state_digest(compactor), before)

    def test_generation_restores_every_question_and_source_on_exception(self):
        import torch
        from types import SimpleNamespace
        from s1d_shared_compactor_eval import _generate
        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 2
            def __call__(self, *args, **kwargs):
                return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))
            def decode(self, *args, **kwargs):
                return 'correct'
        class Reader(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.memory = torch.nn.Parameter(torch.zeros(2, 48, 8), requires_grad=False)
                self.register_buffer('memory_active_mask', torch.ones(48, dtype=torch.bool))
                self.fail = False
                self.calls = 0
            def generate(self, input_ids, **kwargs):
                self.calls += 1
                if self.memory.sum() != 0:
                    raise AssertionError('previous question polluted this query')
                self.memory.data.add_(1)
                if self.fail:
                    raise RuntimeError('deliberate generation failure')
                return torch.cat([input_ids, torch.tensor([[3]])], dim=1)
        model = Reader()
        original, mask = model.memory, model.memory_active_mask
        manager = SimpleNamespace(tokenizer=Tokenizer())
        items = [dict(question='q1', answer='correct'), dict(question='q2', answer='correct')]
        for candidate in (None, torch.zeros_like(original)):
            result = _generate(model, manager, items, candidate, mask.clone())
            self.assertTrue(all(row['em'] for row in result.values()))
            self.assertIs(model.memory, original)
            self.assertIs(model.memory_active_mask, mask)
            self.assertEqual(float(original.sum()), 0)
        model.fail = True
        candidate = torch.zeros_like(original)
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            _generate(model, manager, items, candidate, mask.clone())
        self.assertEqual(float(candidate.sum()), 0)
        self.assertIs(model.memory, original)
        self.assertIs(model.memory_active_mask, mask)
