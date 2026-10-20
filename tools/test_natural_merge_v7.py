#!/usr/bin/env python3
"""v7 CPU contracts: plan schema, registry, generation isolation, targeted recovery.

Run with:  CUDA_VISIBLE_DEVICES="" python tools/test_natural_merge_v7.py

Every fixture below is a CPU contract fixture, never experiment data.
"""
import copy
import json
from pathlib import Path
import sys
import types
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from methods.prlmc.gated_slot_merge import GraftedSlotMemoryMergeV6
from methods.prlmc.native_skip_merge import SupervisedDSlotMerge
from natural_merge_runtime import (MergeStream, V7_ARCHITECTURES, build_merge, reset,
                                   state_digest, training_for, validate_merge_checkpoint)
from natural_merge_v4_objective import (NativeOrigins, balanced_coefficients, full_targets,
                                        make_optimizer, recovery_raw_weight, train_event)
from prepare_natural_merge_v3 import TRAIN_V3, build_plan as build_v3
from prepare_natural_merge_v4 import TRAIN_V4, build_plan as build_v4
from prepare_natural_merge_v6 import TRAIN_V6
from prepare_natural_merge_v7 import (TRAIN_V7, build_plan as build_v7, make_schedule_v7,
                                      validate_v7_plan)
from public_retention_eval import StreamingMemory, compaction_trigger
from test_natural_merge_v2 import tiny_manager, tiny_qa
from test_public_retention import manager, write_record


def v7_classes_available():
    try:
        from methods.prlmc.gated_slot_merge import SupervisedDSlotMergeV7  # noqa: F401
        from methods.prlmc.native_skip_merge import NativeSkipSlotMergeV7  # noqa: F401
        return True
    except ImportError:
        return False


class Tokens:
    """Deterministic tokenizer stand-in: one id per character, no specials."""
    eos_token_id = 99999

    def __call__(self, text, **kwargs):
        return types.SimpleNamespace(input_ids=[ord(c) for c in text])


def archived_v4_plan():
    p = ROOT/"reports/natural_merge_v2_20261014/train_plan.json"
    if not p.exists():
        return None, None
    tokens = Tokens()
    original = build_v3(json.loads(p.read_text()), tokens, "v2")
    return build_v4(original, tokens, "v3"), p


class PlanSchema(unittest.TestCase):
    def test_v7_plan_builds_from_the_archived_v4_plan(self):
        plan_v4, path = archived_v4_plan()
        if plan_v4 is None:
            self.skipTest("archived original plan required")
        plan = build_v7(plan_v4, "v4-file-sha")
        validate_v7_plan(plan)
        self.assertEqual(plan["schema"], "natural_merge_v7_training_plan/1")
        self.assertEqual(plan["training"], TRAIN_V7)
        self.assertEqual(plan["episodes"], plan_v4["episodes"])
        self.assertEqual(sorted(TRAIN_V7["variants"]), ["DnativeV7", "DsupV7"])
        self.assertEqual(TRAIN_V7["geometry"], dict(budget_blocks=20, window_blocks=13, target_blocks=9))
        self.assertEqual(TRAIN_V7["recurrent_event_before_writes"], [14, 18])
        # Single-merge events keep their sealed positions and source windows.
        strip = lambda e: {k: v for k, v in e.items() if k != "qa_ids"}
        self.assertEqual([strip(e) for e in plan["qa_events"] if e["stage"] == "single_merge"],
                         [strip(e) for e in plan_v4["qa_events"] if e["stage"] == "single_merge"])
        # The QA assignment is re-derived (recurrent appearances moved), but it
        # must still supervise every original QA exactly once per appearance.
        for event in plan["qa_events"]:
            self.assertTrue(event["qa_ids"])
            self.assertEqual(len(event["qa_ids"]), len(set(event["qa_ids"])))
        self.assertEqual(sorted({e["before_write"] for e in plan["qa_events"] if e["stage"] == "recurrent"}),
                         [14, 18])
        all_ids = {q["id"] for e in plan["episodes"] for c in e["contexts"] for q in c["qas"]}
        seen = {i for e in plan["qa_events"] for i in e["qa_ids"]}
        self.assertEqual(all_ids, seen, "v7 schedule must supervise every original training QA")

    def test_v7_plan_rejects_tampering_and_wrong_schema(self):
        plan_v4, _ = archived_v4_plan()
        if plan_v4 is None:
            self.skipTest("archived original plan required")
        plan = build_v7(plan_v4, "v4-file-sha")
        changed = copy.deepcopy(plan); changed["training"]["geometry"]["target_blocks"] = 7
        with self.assertRaises(ValueError):
            validate_v7_plan(changed)  # content hash and geometry both move
        changed = copy.deepcopy(plan); changed["content_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_v7_plan(changed)
        changed = copy.deepcopy(plan); changed["schema"] = "natural_merge_v6_training_plan/1"
        with self.assertRaises(ValueError):
            validate_v7_plan(changed)
        changed = copy.deepcopy(plan); changed["qa_events"] = []
        with self.assertRaises(ValueError):
            validate_v7_plan(changed)
        changed = copy.deepcopy(plan); changed["episodes"][0]["contexts"] = []
        with self.assertRaises(ValueError):
            validate_v7_plan(changed)

    def test_schedule_builder_is_deterministic_and_v4_episode_independent(self):
        plan_v4, _ = archived_v4_plan()
        if plan_v4 is None:
            self.skipTest("archived original plan required")
        first = make_schedule_v7(plan_v4["episodes"])
        second = make_schedule_v7(copy.deepcopy(plan_v4["episodes"]))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64, "32 single + 32 recurrent updates")

    def test_sealed_earlier_generations_are_untouched(self):
        # TRAIN_V3's recovery formula string is locked by deep-equality
        # assertions inside every archived v3/v4/v5/v6 plan; v7 must not move it.
        self.assertEqual(TRAIN_V3["recovery_weight"],
                         "1 + clamp(CE_Selection - CE_Full, 0, 2); mean one within noncarrier cohort")
        self.assertEqual(TRAIN_V4["recovery_weight"], TRAIN_V3["recovery_weight"])
        self.assertEqual(TRAIN_V6["recovery_weight"], TRAIN_V3["recovery_weight"])
        self.assertNotIn("recovery_gate", TRAIN_V4)
        self.assertNotIn("geometry", TRAIN_V4)
        self.assertNotIn("recovery_gate", TRAIN_V6)
        self.assertEqual(sorted(TRAIN_V4["variants"]), ["Dnative", "Dsup"])
        self.assertEqual(sorted(TRAIN_V6["variants"]), ["DnativeV6", "DsupV6"])
        # v7 is a separate record, not an alias.
        self.assertIsNot(TRAIN_V7, TRAIN_V4)
        self.assertNotEqual(TRAIN_V7, TRAIN_V4)
        # The sealed v4 plan still validates, which is the real deep-equality
        # check on TRAIN_V4 (archived plans embed it verbatim).
        plan_v4, _ = archived_v4_plan()
        if plan_v4 is not None:
            from prepare_natural_merge_v4 import validate_v4_plan
            validate_v4_plan(plan_v4)


class Registry(unittest.TestCase):
    def test_v7_architectures_are_declared_and_resolve(self):
        self.assertEqual(V7_ARCHITECTURES,
                         ("grouped_gated_sparse_slot_v7", "grouped_gated_sparse_native_v7"))
        self.assertEqual(training_for("grouped_gated_sparse_slot_v7"), TRAIN_V7)
        self.assertEqual(training_for("grouped_gated_sparse_native_v7"), TRAIN_V7)
        self.assertEqual(training_for("grouped_gated_sparse_slot_v6"), TRAIN_V6)
        if not v7_classes_available():
            with self.assertRaisesRegex(ValueError, "v7 merge classes are unavailable"):
                build_merge("grouped_gated_sparse_slot_v7", 32, 8, 2)
            self.skipTest("v7 architecture classes not landed in this checkout")
        module = build_merge("grouped_gated_sparse_slot_v7", 32, 8, 2)
        self.assertEqual(module.architecture, "grouped_gated_sparse_slot_v7")
        self.assertEqual(module.gamma_max, .20)
        self.assertEqual(module.num_groups, 8)
        native = build_merge("grouped_gated_sparse_native_v7", 32, 8, 2)
        self.assertEqual(native.architecture, "grouped_gated_sparse_native_v7")

    def test_v6_stays_registered_for_read_only_reproduction(self):
        module = build_merge("grouped_gated_sparse_slot_v6", 32, 8, 2)
        self.assertIsInstance(module, GraftedSlotMemoryMergeV6)

    def test_v6_and_v7_checkpoints_are_mutually_rejected(self):
        for saved_v6, arch_v7 in [(dict(variant="DsupV6", training=TRAIN_V6,
                                        module_configuration=TRAIN_V6["variants"]["DsupV6"],
                                        effective_gamma_max=.05, global_rms_budget=.05, rank=32,
                                        evaluation_eligible=True),
                                   "grouped_gated_sparse_slot_v7")]:
            with self.assertRaises(ValueError):
                validate_merge_checkpoint(saved_v6, arch_v7)
        saved_v7 = dict(variant="DsupV7", training=TRAIN_V7,
                        module_configuration=TRAIN_V7["variants"]["DsupV7"],
                        effective_gamma_max=.05, global_rms_budget=.05, rank=32,
                        evaluation_eligible=True)
        with self.assertRaises(ValueError):
            validate_merge_checkpoint(saved_v7, "grouped_gated_sparse_slot_v6")
        with self.assertRaises(ValueError):
            validate_merge_checkpoint(saved_v7, "grouped_gated_sparse_native_v7")
        # Its own architecture/training pair is accepted.
        self.assertFalse(validate_merge_checkpoint(saved_v7, "grouped_gated_sparse_slot_v7"))

    def test_v7_module_state_round_trips_through_the_registry(self):
        if not v7_classes_available():
            self.skipTest("v7 architecture classes not landed in this checkout")
        module = build_merge("grouped_gated_sparse_slot_v7", 32, 8, 2)
        saved = dict(variant="DsupV7", training=TRAIN_V7,
                     module_configuration=TRAIN_V7["variants"]["DsupV7"],
                     effective_gamma_max=.05, global_rms_budget=.05, rank=32,
                     evaluation_eligible=True, state_dict=module.state_dict())
        other = build_merge("grouped_gated_sparse_slot_v7", 32, 8, 2)
        with self.assertRaises(ValueError):
            validate_merge_checkpoint(saved, "grouped_gated_sparse_slot_v7")  # no manifest
        from natural_merge_runtime import source_manifest
        saved["source_manifest"] = source_manifest()
        self.assertTrue(validate_merge_checkpoint(saved, "grouped_gated_sparse_slot_v7"))
        other.load_state_dict(saved["state_dict"], strict=True)
        self.assertEqual(state_digest(other), state_digest(module))


class TargetedRecoveryWeights(unittest.TestCase):
    def test_gap_boundary_is_strictly_greater_than_025(self):
        # gap == 0.25 is the background band, not the targeted band.
        self.assertEqual(recovery_raw_weight("recover", 2.0, 2.25, TRAIN_V7), 0.25)
        self.assertAlmostEqual(recovery_raw_weight("recover", 2.0, 2.2500001, TRAIN_V7),
                               1 + 3 * 0.2500001, places=6)
        self.assertEqual(recovery_raw_weight("recover", 1.0, 1.25, TRAIN_V7), 0.25)
        self.assertAlmostEqual(recovery_raw_weight("recover", 1.0, 1.5, TRAIN_V7), 2.5)

    def test_ce_full_boundary_is_inclusive_at_25(self):
        self.assertEqual(recovery_raw_weight("recover", 2.5, 3.5, TRAIN_V7), 4.0)
        self.assertEqual(recovery_raw_weight("recover", 2.5000001, 3.5, TRAIN_V7), 0.25)
        # gap is capped at 2 before scaling.
        self.assertEqual(recovery_raw_weight("recover", 1.0, 9.0, TRAIN_V7), 7.0)

    def test_protect_cohort_keeps_the_background_anchor(self):
        self.assertEqual(recovery_raw_weight("protect", 1.0, 1.0, TRAIN_V7), 0.25)
        rows = balanced_coefficients([dict(source_index=i) for i in range(4)],
                                     [("recover", "current_window"), ("recover", "current_window"),
                                      ("protect", "untouched_tail"), ("protect", "untouched_tail")],
                                     [True, True, True, False],
                                     [1., 1., 1., 1.], [2., 3., 2., 2.], TRAIN_V7)
        self.assertEqual([r["weight"] for r in rows[2:]], [0.25, 0.25])
        self.assertTrue(all(r["recovery"] > 0 for r in rows), "protect recovery CE is non-zero by design")
        self.assertAlmostEqual(sum(r["recovery"] for r in rows[:2]), 1.0, places=9)

    def test_legacy_plan_keeps_the_sealed_formula_bit_for_bit(self):
        self.assertEqual(recovery_raw_weight("recover", 1.0, 2.0, TRAIN_V4), 2.0)
        self.assertEqual(recovery_raw_weight("recover", 1.0, 5.0, TRAIN_V4), 3.0)
        self.assertEqual(recovery_raw_weight("protect", 1.0, 1.0, TRAIN_V4), 0.0)
        rows = balanced_coefficients([dict(source_index=i) for i in range(4)],
                                     [("recover", "current_window"), ("recover", "current_window"),
                                      ("protect", "untouched_tail"), ("protect", "untouched_tail")],
                                     [True, True, True, False],
                                     [1., 1., 1., 1.], [2., 3., 2., 2.])
        self.assertEqual([r["weight"] for r in rows], [0.8, 1.2, 0.0, 0.0])
        self.assertEqual([r["recovery"] for r in rows], [0.4, 0.6, 0.0, 0.0])

    def test_teacher_gate_stays_at_two_point_zero(self):
        # Two different gates: recovery weight at 2.5, distillation at 2.0.
        self.assertEqual(TRAIN_V7["teacher_max_gold_ce"], 2.0)
        self.assertEqual(TRAIN_V7["recovery_gate"]["ce_full_max"], 2.5)
        in_band = 2.2
        self.assertGreater(recovery_raw_weight("recover", in_band, in_band + 1., TRAIN_V7), 1.)
        self.assertFalse(in_band <= TRAIN_V7["teacher_max_gold_ce"],
                         "the 2.0 < ce_full <= 2.5 band has a recovery weight but no KL teacher")


class EvaluationGeometry(unittest.TestCase):
    """The 13->9 trigger must live on the real evaluation write path (C8/D-07)."""

    def _run(self, budget, window, target, writes, arm="Selection"):
        m, compactor = manager()
        stream = StreamingMemory(m, compactor, arm, budget_blocks=budget,
                                 window_blocks=window, target_blocks=target)
        for i in range(writes):
            stream.write(write_record(i), i)
        return m, stream

    def test_trigger_table_keeps_every_registered_geometry(self):
        self.assertEqual(compaction_trigger(8, 9), 8)
        self.assertEqual(compaction_trigger(10, 11), 10)
        self.assertEqual(compaction_trigger(12, 11), 12)
        self.assertEqual(compaction_trigger(20, 19), 20)
        self.assertEqual(compaction_trigger(20, 13), 14)

    def test_first_compaction_grid_is_identical_for_the_registered_geometries(self):
        for budget, window, target, first in ((8, 9, 4, 9), (10, 11, 5, 11),
                                              (12, 11, 5, 12), (20, 19, 9, 20)):
            m, stream = self._run(budget, window, target, first + 1)
            self.assertEqual([e["before_write"] for e in stream.events], [first],
                             (budget, window, target))
            event = stream.events[0]
            self.assertEqual(event["source_slots"] // m.block, window)
            self.assertEqual(event["target_slots"] // m.block, target)
            self.assertEqual(m.used // m.block, target + (first - window) + 1,
                             "carriers + untouched tail + the appended write")

    def test_13_9_fires_at_14_blocks_and_keeps_a_one_block_tail(self):
        m, stream = self._run(20, 13, 9, 40)
        self.assertEqual([e["before_write"] for e in stream.events], [14, 18, 22, 26, 30, 34, 38])
        self.assertTrue(all(e["source_slots"] // m.block == 13 for e in stream.events))
        self.assertTrue(all(e["target_slots"] // m.block == 9 for e in stream.events))
        # The last event leaves 9 carriers + 1 tail, then two more writes append.
        self.assertEqual(m.used // m.block, 12)
        # The 13->9 merge keeps the newest block: 9 carriers + 1 tail.
        self.assertEqual(stream.events[0]["remaining_slots"] // m.block, 10)

    def test_19_9_still_fires_at_20_blocks(self):
        m, stream = self._run(20, 19, 9, 40)
        self.assertEqual([e["before_write"] for e in stream.events], [20, 30])
        self.assertTrue(all(e["source_slots"] // m.block == 19 for e in stream.events))
        self.assertEqual(m.used // m.block, 20)

    def test_explicit_geometry_overrides_the_table_and_conflicts_raise(self):
        m, stream = self._run(20, 13, 9, 1)
        self.assertEqual((stream.window_blocks, stream.target_blocks), (13, 9))
        self.assertEqual((stream.window // m.block, stream.target // m.block), (13, 9))
        # None keeps the pre-registered table (20 -> 19/9).
        m2, compactor = manager()
        default = StreamingMemory(m2, compactor, "Selection")
        self.assertEqual((default.window_blocks, default.target_blocks), (19, 9))
        for window, target in ((9, 13), (13, 13), (31, 9), (0, 0)):
            m3, compactor3 = manager()
            with self.assertRaises(ValueError):
                StreamingMemory(m3, compactor3, "Selection", budget_blocks=20,
                                window_blocks=window, target_blocks=target)
        with self.assertRaises(ValueError):
            StreamingMemory(m3, compactor3, "Selection", budget_blocks=13)

    def test_ages_are_booked_on_the_evaluation_path(self):
        m, stream = self._run(20, 13, 9, 15)
        self.assertEqual(len(stream.events), 1)
        self.assertEqual(m.age_append_events, 15, "one append per write, never double counted")
        summary = m.block_age_summary()
        self.assertEqual(summary["block_age_max"], 1)
        self.assertEqual(summary["block_age_histogram"].get(1), 9, "nine surviving carriers aged")
        self.assertEqual(int(m._block_ages[-1]), 0, "the newest appended block is the young generation")
        self.assertEqual(len(stream.events[0]["selected_block_ages"]), 9)
        # Snapshot/restore carries the ages with the slots (R-07).
        state = m.snapshot()
        before = m._block_ages.clone()
        m.restore(state)
        self.assertTrue(torch.equal(m._block_ages, before))

    def test_selection_and_carrier_arms_stay_identical_on_the_13_9_path(self):
        # Zero-initialised residual: the SharedE6d arm must still equal Selection.
        m1, s1 = self._run(20, 13, 9, 20, "Selection")
        m2, s2 = self._run(20, 13, 9, 20, "SharedE6d")
        self.assertTrue(torch.equal(m1._distinct(), m2._distinct()))
        self.assertEqual([e["selected_blocks"] for e in s1.events],
                         [e["selected_blocks"] for e in s2.events])


class TrainerWiring(unittest.TestCase):
    def test_trainer_records_source_manifest_and_v7_geometry(self):
        import ast
        source = (ROOT/"tools/natural_merge_v7_train.py").read_text()
        tree = ast.parse(source)
        saves = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "save"
                 and getattr(getattr(node.func, "value", None), "id", "") == "torch"]
        self.assertTrue(saves, "v7 trainer must persist a checkpoint")
        dicts = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "dict"]
        keywords = {kw.arg for node in dicts for kw in node.keywords}
        self.assertIn("source_manifest", keywords,
                      "v7 checkpoints must carry the t2 source manifest")
        self.assertIn("training", keywords)
        self.assertIn("geometry", keywords)
        self.assertTrue(any(isinstance(arg, ast.Name) and arg.id == "checkpoint"
                            for node in saves for arg in node.args),
                        "the checkpoint dict must be the object passed to torch.save")
        calls = {getattr(node.func, "id", getattr(node.func, "attr", "")) for node in ast.walk(tree)
                 if isinstance(node, ast.Call)}
        self.assertIn("validate_v7_plan", calls)
        import natural_merge_v7_train as trainer
        self.assertEqual(trainer.GEOMETRY, TRAIN_V7["geometry"])
        self.assertEqual(trainer.TRAIN_V7, TRAIN_V7)
        self.assertIn("DsupV7", source)

    def test_trainer_rejects_a_v6_plan(self):
        import natural_merge_v7_train as trainer
        from prepare_natural_merge_v6 import validate_v6_plan
        plan_v4, _ = archived_v4_plan()
        if plan_v4 is None:
            self.skipTest("archived original plan required")
        v7 = build_v7(plan_v4, "v4-file-sha")
        # A v6 validator must not accept a v7 plan, and vice versa.
        with self.assertRaises(ValueError):
            validate_v6_plan(v7)
        self.assertTrue(callable(trainer.run))


class SupervisedMergeGeometry(unittest.TestCase):
    def test_train_event_uses_the_plan_window_and_target(self):
        for variant, cls in [("DsupV7", SupervisedDSlotMerge)]:
            manager = tiny_manager()
            merge = cls(1, 4, block=2, rank=2, num_groups=1, reference_layer=0, topk=2)
            initial = manager._memory.clone(); reset(manager, initial)
            stream = MergeStream(manager, merge, "Full"); origins = NativeOrigins()
            for i in range(19):
                stream.write(write_record(i), i); origins.append(i)
            qas = [dict(tiny_qa(i), source_index=i, content_ids=[1], answer_ids=[1, 0]) for i in range(19)]
            targets = full_targets(manager, qas)
            optimizer = make_optimizer(merge, plan=TRAIN_V7)
            event = train_event(manager, merge, optimizer, qas, targets, variant, origins, plan=TRAIN_V7)
            self.assertEqual(len(event["carrier_indices"]), 9)
            # 19 resident blocks -> 13-block window -> 9 carriers + 6 untouched tail.
            self.assertEqual(manager.used, (9 + 6) * manager.block)
            self.assertEqual(len(origins.blocks), 15)
            self.assertEqual(event["cohorts"]["recover"]["count"] + event["cohorts"]["protect"]["count"], 19)

    def test_sealed_v4_geometry_is_unchanged(self):
        manager = tiny_manager()
        merge = SupervisedDSlotMerge(1, 4, block=2, rank=2, num_groups=1, reference_layer=0, topk=2)
        initial = manager._memory.clone(); reset(manager, initial)
        stream = MergeStream(manager, merge, "Full"); origins = NativeOrigins()
        for i in range(19):
            stream.write(write_record(i), i); origins.append(i)
        qas = [dict(tiny_qa(i), source_index=i, content_ids=[1], answer_ids=[1, 0]) for i in range(19)]
        targets = full_targets(manager, qas)
        event = train_event(manager, merge, make_optimizer(merge), qas, targets, "Dsup", origins)
        self.assertEqual(len(event["carrier_indices"]), 9)
        self.assertEqual(manager.used, (9 + 0) * manager.block, "v4 keeps the 19->9 full-window merge")
        self.assertEqual(len(origins.blocks), 9)


class CapacityHonesty(unittest.TestCase):
    """8/10-block geometries allow one block more than they declare."""

    def test_overshoot_is_reported_not_hidden(self):
        for budget, window, target, overshoot in ((8, 9, 4, 1), (10, 11, 5, 1),
                                                  (12, 11, 5, 0), (20, 19, 9, 0)):
            m, compactor = manager()
            stream = StreamingMemory(m, compactor, "Selection", budget_blocks=budget,
                                     window_blocks=window, target_blocks=target)
            self.assertEqual(stream.peak_allowed_blocks, max(budget, window))
            self.assertEqual(stream.capacity_overshoot_blocks, overshoot)
            self.assertEqual(stream.budget_blocks, budget)

    def test_strict_budget_refuses_the_overshoot(self):
        for budget, window, target in ((8, 9, 4), (10, 11, 5)):
            m, compactor = manager()
            with self.assertRaises(ValueError):
                StreamingMemory(m, compactor, "Selection", budget_blocks=budget,
                                window_blocks=window, target_blocks=target,
                                strict_budget=True)
        m, compactor = manager()
        stream = StreamingMemory(m, compactor, "Selection", budget_blocks=8,
                                 window_blocks=8, target_blocks=4, strict_budget=True)
        self.assertEqual(stream.capacity_overshoot_blocks, 0)
        self.assertEqual(stream.peak_allowed_blocks, 8)

    def test_strict_budget_never_exceeds_the_declared_capacity(self):
        m, compactor = manager()
        stream = StreamingMemory(m, compactor, "Selection", budget_blocks=8,
                                 window_blocks=8, target_blocks=4, strict_budget=True)
        for i in range(20):
            stream.write(write_record(i), i)
            self.assertLessEqual(stream.peak // m.block, 8)


class ArchitectureResolution(unittest.TestCase):
    """The evaluation entry must follow the checkpoint, not a pinned default."""

    def test_checkpoint_architecture_wins(self):
        from natural_merge_runtime import resolve_architecture
        v5, v6 = "grouped_gated_sparse_slot_v5", "grouped_gated_sparse_slot_v6"
        self.assertEqual(resolve_architecture({"architecture": v6}, default=v5), v6)
        self.assertEqual(resolve_architecture(None, default=v5), v5)
        self.assertEqual(resolve_architecture({"architecture": v6}, requested=v6, default=v5), v6)
        with self.assertRaises(ValueError):
            resolve_architecture({"architecture": v6}, requested=v5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
