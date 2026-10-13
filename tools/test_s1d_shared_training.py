"""CPU tests for scene isolation, native block selection and shared scheduling."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from prepare_s1d_shared_training import build_training_plan, validate_training_plan
from sic_paper_data import digest, generate
from s1d_shared_compactor_eval import (_candidate_pool, arm_summary, baseline_indices,
                                     evaluation_protocol)
from s1d_shared_compactor_train import train_events, training_schedule
from test_sic_paper_data import fixture, SPECS


class SharedTrainingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contexts = fixture(800)
        cls.reserved = generate(cls.contexts, SPECS, episodes=2, dev=1)
        cls.plan = build_training_plan(cls.contexts, SPECS, cls.reserved, episodes=2)

    def test_training_is_disjoint_from_all_reserved_and_other_training_scenes(self):
        validate_training_plan(self.plan, self.reserved)
        blocked = {f["subject"] for e in self.reserved["episodes"] for f in e["facts"]}
        seen = set()
        for episode in self.plan["episodes"]:
            subjects = {fact["subject"] for fact in episode["facts"]}
            self.assertFalse(subjects & (blocked | seen))
            seen |= subjects
            self.assertEqual(episode["partition"], "train")
            self.assertNotIn("splits", episode)
            self.assertEqual(len(episode["training_probes"]), 20)

    def test_data_shortage_does_not_recycle_reserved_entities(self):
        plan = build_training_plan(self.contexts, SPECS, self.reserved, episodes=4)
        self.assertEqual(plan["status"], "insufficient_data")
        self.assertEqual(plan["episodes"], [])

    def test_rehashed_overlap_rejected(self):
        plan = copy.deepcopy(self.plan)
        plan["episodes"][0]["facts"][0]["subject"] = self.reserved["episodes"][0]["facts"][0]["subject"]
        episode = plan["episodes"][0]
        episode.pop("episode_sha256")
        episode["episode_sha256"] = digest(episode)
        plan.pop("plan_sha256")
        plan["plan_sha256"] = digest(plan)
        with self.assertRaises(ValueError):
            validate_training_plan(plan, self.reserved)

    def test_training_probe_cannot_be_changed_even_with_rehashed_plan(self):
        plan = copy.deepcopy(self.plan)
        episode = plan["episodes"][0]
        episode["training_probes"][0]["answer"] = "wrong"
        episode.pop("episode_sha256")
        episode["episode_sha256"] = digest(episode)
        plan.pop("plan_sha256")
        plan["plan_sha256"] = digest(plan)
        with self.assertRaisesRegex(ValueError, "latest"):
            validate_training_plan(plan, self.reserved)

    def test_interleave_totals_and_persistent_training_state(self):
        events = training_schedule(8, 0)
        self.assertEqual(events, training_schedule(8, 1))
        for episode in range(8):
            self.assertEqual(sum(n for stage, e, n in events if e == episode and stage == "A"), 4)
            self.assertEqual(sum(n for stage, e, n in events if e == episode and stage == "B"), 96)
        self.assertEqual([e for _, e, _ in events[8:16]], list(range(8)))
        # Mimic a module/optimizer that accumulates state across all visits.
        state, logged = {"steps": 0}, []
        def visit(stage, episode, steps):
            state["steps"] += steps
            return dict(optimizer_steps=state["steps"])
        train_events(events, visit, logged.append)
        self.assertEqual(state["steps"], 800)
        self.assertEqual(logged[-1]["cumulative_steps"], 800)
        self.assertEqual([e["training"]["optimizer_steps"] for e in logged],
                         [e["cumulative_steps"] for e in logged])
        with self.assertRaises(ValueError):
            training_schedule(8, 0, steps_b=95)

    def test_extra_baseline_paired_recovery_and_damage(self):
        maps = {"Full": {"a": True, "b": True, "c": False},
                "Selection": {"a": False, "b": True, "c": False},
                "SharedE6d": {"a": True, "b": False, "c": True},
                "Recent9": {"a": False, "b": True, "c": True},
                "Random9": {"a": False, "b": False, "c": False}}
        summary = arm_summary(maps)
        self.assertEqual(summary["shared_recovers_lost"], ["a"])
        self.assertEqual(summary["paired_comparisons"]["Recent9"]["gained"], ["a"])
        self.assertEqual(summary["paired_comparisons"]["Recent9"]["harmed"], ["b"])
        self.assertEqual(summary["paired_comparisons"]["Recent9"]["net"], 0)
        with self.assertRaises(ValueError):
            evaluation_protocol(["Full", "Random9"])

    def test_recent_and_random_pools_copy_same_blocks_at_every_layer(self):
        import torch
        from methods.prlmc.consolidate import CarrierResidualCompactor
        memory = torch.arange(2 * 48 * 8).reshape(2, 48, 8).float()
        manager = SimpleNamespace(_memory=memory, total=48, used=38, block=2,
            num_layers=2, hidden=8, layout=SimpleNamespace(total=48, k_anchor=4),
            active_memory_mask=True, goal_direction=None, select_weights=None,
            _window_importance=lambda begin, width: torch.zeros(width))
        compactor = CarrierResidualCompactor(2, 8, block=2, rank=3)
        original = memory.clone()
        for name in ("Recent9", "Random9"):
            pool, mask = _candidate_pool(manager, compactor, name, "cpu")
            indices = baseline_indices(name)
            self.assertEqual(len(indices), 9)
            self.assertEqual(indices, sorted(set(indices)))
            source = original[:, 10:, :].reshape(2, 19, 2, 8)
            expected = source[:, indices, :, :].reshape(2, 18, 8)
            self.assertTrue(torch.equal(pool[:, -18:, :], expected))
            self.assertTrue(torch.equal(pool[:, :4, :], original[:, :4, :]))
            self.assertEqual(int(mask.sum()), 22)
        self.assertTrue(torch.equal(memory, original))
        self.assertEqual(baseline_indices("Recent9"), list(range(10, 19)))
        self.assertEqual(baseline_indices("Random9"), baseline_indices("Random9"))

    def test_actual_training_backend_updates_only_compactor_across_scenes(self):
        import torch
        from methods.prlmc.consolidate import CarrierResidualCompactor
        from methods.prlmc.probes import Probe, clear_teacher_cache, train_carrier_compactor
        torch.manual_seed(1)
        class Tokenizer:
            def __call__(self, text, return_tensors=None, **kwargs):
                ids = [1, 2, 3] if text.endswith(" correct") else [1, 2]
                return SimpleNamespace(input_ids=torch.tensor([ids]) if return_tensors else ids)
        class Reader(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.memory = torch.nn.Parameter(torch.randn(2, 48, 8), requires_grad=False)
                self.register_buffer("memory_active_mask", torch.ones(48, dtype=torch.bool))
                self.register_buffer("projection", torch.randn(8, 5))
            def forward(self, input_ids, **kwargs):
                value = self.memory[:, self.memory_active_mask, :].mean(dim=(0, 1))
                hidden = value[None, None, :].expand(1, input_ids.shape[1], 8)
                return SimpleNamespace(logits=hidden @ self.projection, hidden_states=[hidden] * 32)
        reader = Reader().eval()
        original_memory = reader.memory
        manager = SimpleNamespace(model=reader, _memory=reader.memory.data, total=48, used=38, block=2,
            num_layers=2, hidden=8, layout=SimpleNamespace(total=48, k_anchor=4), cfg=SimpleNamespace(),
            active_memory_mask=True, goal_direction=None, select_weights=None, tokenizer=Tokenizer(),
            _window_importance=lambda begin, width: torch.zeros(width))
        compactor = CarrierResidualCompactor(2, 8, block=2, rank=3, gamma_max=.05)
        initial = compactor.up.weight.detach().clone()
        optimizer = torch.optim.AdamW(compactor.parameters(), lr=.001)
        clear_teacher_cache()
        for scene in range(2):
            with torch.no_grad():
                reader.memory.copy_(torch.randn_like(reader.memory))
            source = reader.memory.detach().clone()
            train_carrier_compactor(manager, [Probe("q", ["correct"])], steps=1, stage="B",
                compactor=compactor, optimiser=optimizer, window_blocks=19, target_blocks=9,
                use_checkpoint=False, teacher_cache_key=(scene,))
            self.assertIs(reader.memory, original_memory)
            self.assertTrue(torch.equal(reader.memory, source))
            self.assertFalse(reader.memory.requires_grad)
        self.assertFalse(torch.equal(compactor.up.weight, initial))
        self.assertEqual(int(optimizer.state[compactor.up.weight]["step"]), 2)
        clear_teacher_cache()

    def test_sealed_access_requires_exact_freeze_and_single_open_ledger(self):
        from s1d_shared_compactor_eval import sha, validate_test_access
        from s1d_controlled_contract import claim_sealed_test
        with tempfile.TemporaryDirectory() as directory:
            plan, freeze = Path(directory) / "plan.json", Path(directory) / "freeze.json"
            plan.write_text("{}")
            manifest = {"checkpoint_sha256": "candidate"}
            freeze.write_text(json.dumps(dict(status="frozen", manifest=manifest,
                plan_sha256=sha(plan), all_development_complete=True, test_scored=False)))
            with self.assertRaisesRegex(ValueError, "single-open"):
                validate_test_access(plan, freeze, manifest)
            claim_sealed_test(plan, freeze, manifest)
            validate_test_access(plan, freeze, manifest)
            with self.assertRaises(FileExistsError):
                claim_sealed_test(plan, freeze, manifest)
            with self.assertRaisesRegex(ValueError, "freeze"):
                validate_test_access(plan, freeze, {"checkpoint_sha256": "different"})

    def test_natural_eventqa_requires_all_elements_and_complete_context_admission(self):
        from s1d_shared_natural_eval import context_capacity, eventqa_score
        self.assertTrue(eventqa_score("Alice and BOB", ["alice", "Bob"])[0])
        self.assertFalse(eventqa_score("Alice", ["alice", "Bob"])[0])
        with self.assertRaises(ValueError):
            eventqa_score("Alice", [])
        self.assertEqual(context_capacity(["chunk"] * 50, 256, 12288), "capacity_exceeded")
        self.assertEqual(context_capacity(["chunk"] * 10, 256, 12288), "insufficient_blocks_for_9_block_comparison")
        self.assertEqual(context_capacity(["chunk"] * 20, 256, 12288), "eligible")

    def test_variable_natural_window_preserves_whole_block_budget(self):
        import torch
        from methods.prlmc.consolidate import CarrierResidualCompactor
        memory = torch.randn(2, 48, 8)
        manager = SimpleNamespace(_memory=memory, total=48, used=24, block=2, num_layers=2, hidden=8,
            layout=SimpleNamespace(total=48, k_anchor=4), active_memory_mask=True,
            goal_direction=None, select_weights=None, _window_importance=lambda begin, width: torch.zeros(width))
        compactor = CarrierResidualCompactor(2, 8, block=2, rank=3, gamma_max=.05)
        for name in ("Recent9", "Random9", "Selection", "SharedE6d"):
            pool, mask = _candidate_pool(manager, compactor, name, "cpu", window_blocks=12)
            self.assertEqual(pool.shape, memory.shape)
            self.assertEqual(int(mask.sum()), 22)
            if name == "Recent9":
                self.assertTrue(torch.equal(pool[:, -18:, :], memory[:, -18:, :]))

    def test_fleet_aggregation_rescores_and_never_drops_failed_episodes(self):
        from unittest.mock import patch
        import run_s1d_shared_compactor_eval as launcher
        from s1d_shared_compactor_eval import COMPARISON_ARMS, DEVELOPMENT_IDS, TEST_IDS, sha
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "ckpt.pt"
            checkpoint.write_bytes(b"checkpoint")
            plan_path = root / "plan.json"
            episodes = [dict(episode=episode, partition="development", splits=dict(evaluation=[
                dict(question="q%d" % question, answer="correct") for question in range(64)])) for episode in DEVELOPMENT_IDS]
            episodes += [dict(episode=episode, partition="test") for episode in TEST_IDS]
            plan_path.write_text(json.dumps(dict(status="completed", episodes=episodes)))
            protocol = evaluation_protocol(COMPARISON_ARMS)
            maps = {name: {"q%d" % question: name == "SharedE6d" for question in range(64)} for name in COMPARISON_ARMS}
            predictions = {name: {q: dict(prediction="correct" if correct else "wrong", em=correct, substring_em=correct)
                                  for q, correct in values.items()} for name, values in maps.items()}
            paths = []
            for episode in DEVELOPMENT_IDS:
                metadata = {name: dict(active_slots=5376 if name == "Full" else 2816) for name in COMPARISON_ARMS}
                for name in ("Recent9", "Random9"):
                    metadata[name]["kept_blocks"] = baseline_indices(name)
                payload = dict(status="complete", training_performed=False, optimizer_created=False,
                    backward_performed=False, basis_constructed=False, alpha_adapted=False,
                    source_memory_unchanged=True, compactor_unchanged=True,
                    compactor_state_sha256_before="frozen", compactor_state_sha256_after="frozen",
                    identity=dict(stage="development", episode=episode, context_sha256=str(episode),
                        protocol=protocol, plan_sha256=sha(plan_path), checkpoint_sha256=sha(checkpoint), source_manifest={}),
                    predictions=predictions, generation_questions=list(maps["Full"]), arm_metadata=metadata,
                    summary=arm_summary(maps))
                path = root / ("ep%d.json" % episode)
                path.write_text(json.dumps(payload))
                paths.append(path)
            with patch.object(launcher, "source_manifest", return_value={}):
                result = launcher.aggregate(paths, checkpoint, plan_path, protocol)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["paired_comparisons"]["Recent9"]["mean_delta"], 1)
                payload = json.loads(paths[0].read_text())
                payload["predictions"]["Random9"]["q0"]["em"] = True
                paths[0].write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, "rescored"):
                    launcher.aggregate(paths, checkpoint, plan_path, protocol)
                payload["status"] = "failed"
                paths[0].write_text(json.dumps(payload))
                result = launcher.aggregate(paths, checkpoint, plan_path, protocol)
                self.assertEqual(result["status"], "incomplete")
                self.assertNotIn("bootstrap_episode_delta_ci95", result)


if __name__ == "__main__":
    unittest.main()
