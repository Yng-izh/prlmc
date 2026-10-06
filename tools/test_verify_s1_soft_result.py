"""Synthetic regression tests for independent S1-soft verifiers."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from verify_s1_soft_result import validate
from audit_s1_soft_checkpoints import audit


def entry(report, correct, name, step):
    sets = report["sets"]
    ce = {q: 1.25 for q in correct}
    groups = [("original_rescue", "original_rescued_questions", "original_lost", True),
              ("rescue_recovered", "rescue_recovered_questions", "rescue", True),
              ("rescue_d0_harmed", "rescue_d0_harmed_questions", "rescue_d0_correct", False),
              ("repair_recovered", "repair_recovered_questions", "repair", True),
              ("protect_harmed", "protect_harmed_questions", "protect", False),
              ("retained_harmed", "retained_harmed_questions", "retained", False)]
    result = {"step": step, "correct": dict(correct), "ce": ce,
              "accuracy": sum(correct.values()) / 296, "deviation": 0.0,
              "loss_original": sum(ce[q] for q in sets["original_lost"])}
    for count, label, group, target in groups:
        values = [q for q in sets[group] if correct[q] == target]
        result[count], result[label] = len(values), values
    harmed = [q for q, v in report["baselines"]["d0"]["correct"].items() if v and not correct[q]]
    result["d0_new_harmed_all"] = len(harmed)
    result["d0_new_harmed_questions_all"] = harmed
    result["net_vs_select"] = result["rescue_recovered"] - result["retained_harmed"]
    result["penalty"] = 0 if name == "ul" else 13 * .25 ** 2
    return result


def set_arm(report, name, correct):
    report["arms"][name] = {"trainable_dof": 13, "lambda": 0 if name == "ul" else 1,
                            "protection_set": [] if name == "ul" else report["sets"]["protect" if name == "soft_p" else "retained"],
                            "history": [entry(report, correct, name, step) for step in range(5, 61, 5)]}
    report["arms"][name]["final"] = copy.deepcopy(report["arms"][name]["history"][-1])


def fixture():
    qs = ["q%d" % i for i in range(296)]
    full = {q: i < 101 for i, q in enumerate(qs)}
    select = {q: i < 75 for i, q in enumerate(qs)}
    d0 = {q: i < 31 or 95 <= i < 101 for i, q in enumerate(qs)}
    report = {"schema_version": 1, "status": "complete", "verdict": "LOCAL-PASS",
              "protocol": {"window": [3, 17890], "steps": 60, "measure_every": 5,
                           "lr": .003, "weight_decay": .01, "parameter_budget": 1,
                           "seed": 0, "smoke": False, "lambda": 1, "expanded_probes": 296,
                           **{k: "0" * 64 for k in ("checkpoint_sha256", "basis_sha256", "historical_result_sha256", "driver_sha256", "gradient_repair_audit_sha256")}},
              "probes": [{"question": q, "answers": ["a"]} for q in qs],
              "baselines": {name: {"correct": values, "ce": {q: 1.0 for q in qs}}
                            for name, values in (("full", full), ("select", select), ("d0", d0))},
              "sets": {"original_lost": qs[75:88], "rescue": qs[75:101],
                       "protect": qs[:31], "repair": qs[31:75], "retained": qs[:75],
                       "rescue_d0_correct": qs[95:101], "rescue_d0_wrong": qs[75:95]}, "arms": {}}
    ul = dict(select)
    for q in qs[31:37]:
        ul[q] = False
    for q in qs[75:82] + qs[88:95]:
        ul[q] = True
    soft = dict(ul)
    for q in qs[31:37]:
        soft[q] = True
    for name, values in (("ul", ul), ("soft_p", soft), ("soft_k", soft)):
        set_arm(report, name, values)
    historical = {"status": "complete", "protocol": {"smoke": False, "checkpoint_sha256": "0" * 64},
                  "sets": {"lost": qs[75:88]}, "checkpoint_diagnostic": {"correct": d0},
                  "arms": {"select": {"final": {"correct": select}}, "subspace_ul": {"final": {"correct": ul}}}}
    digest = hashlib.sha256(json.dumps(historical).encode()).hexdigest()
    report["protocol"]["historical_result_sha256"] = digest
    return report, historical, digest


class ResultTests(unittest.TestCase):
    def test_complete_pass_and_historical_required(self):
        report, old, digest = fixture()
        self.assertTrue(all(validate(report, True, old, digest)["gate"].values()))
        with self.assertRaises(AssertionError):
            validate(report, True)

    def test_corrupted_metrics_and_maps(self):
        for key, value in (("original_rescue", 8), ("original_rescued_questions", ["q0"]),
                           ("net_vs_select", 999), ("loss_original", 0), ("penalty", 0),
                           ("ce", {"q0": 1}), ("ce", {"q%d" % i: float("nan") for i in range(296)})):
            with self.subTest(key=key, value=str(value)[:20]):
                report, _, _ = fixture()
                report["arms"]["soft_p"]["history"][-1][key] = value
                report["arms"]["soft_p"]["final"][key] = value
                with self.assertRaises(AssertionError):
                    validate(report)

    def test_missing_full_and_history(self):
        for target in ("full", "history"):
            report, _, _ = fixture()
            if target == "full":
                del report["baselines"]["full"]
            else:
                del report["arms"]["soft_p"]["history"]
            with self.assertRaises((KeyError, AssertionError)):
                validate(report)

    def test_history_and_final_integrity(self):
        for change in ("missing", "step", "final", "duplicate_list"):
            report, _, _ = fixture()
            arm = report["arms"]["soft_p"]
            if change == "missing":
                arm["history"].pop(0)
            elif change == "step":
                arm["history"][0]["step"] = 6
            elif change == "final":
                arm["final"]["accuracy"] = 0
            else:
                arm["history"][-1]["original_rescued_questions"].append("q75")
                arm["final"] = copy.deepcopy(arm["history"][-1])
            with self.assertRaises(AssertionError):
                validate(report)

    def test_gate_failures(self):
        for mode in ("strict", "protect", "original", "rescue", "repair", "harm"):
            report, _, _ = fixture()
            correct = dict(report["arms"]["soft_p"]["final"]["correct"])
            if mode == "strict":
                correct = dict(report["arms"]["ul"]["final"]["correct"])
            elif mode == "protect":
                correct["q0"] = False
            elif mode == "original":
                correct["q75"] = False
                correct["q82"] = False
                correct["q95"] = True
            elif mode == "rescue":
                correct["q88"] = False
            elif mode == "repair":
                for i in range(31, 38):
                    correct["q%d" % i] = False
            else:
                for i in range(31, 38):
                    correct["q%d" % i] = False
            set_arm(report, "soft_p", correct)
            report["verdict"] = "NO-IMPROVEMENT"
            self.assertFalse(all(validate(report)["gate"].values()), mode)
            report["verdict"] = "LOCAL-PASS"
            with self.assertRaises(AssertionError):
                validate(report)

    def test_smoke_and_control(self):
        report, old, digest = fixture()
        report["protocol"].update(smoke=True, steps=2, measure_every=1)
        report["verdict"] = "SMOKE-ONLY"
        for arm in report["arms"].values():
            arm["history"] = arm["history"][:2]
            for step, measurement in enumerate(arm["history"], 1):
                measurement["step"] = step
            arm["final"] = copy.deepcopy(arm["history"][-1])
        with self.assertRaises(AssertionError):
            validate(report)
        self.assertIsNone(validate(report, allow_smoke=True)["control_verified"])
        with self.assertRaises(AssertionError):
            validate(report, True, old, digest, allow_smoke=True)
        with self.assertRaises(AssertionError):
            validate(report, True, old, digest)
        report, old, digest = fixture()
        old["arms"]["subspace_ul"]["final"]["correct"] = dict(old["arms"]["subspace_ul"]["final"]["correct"])
        old["arms"]["subspace_ul"]["final"]["correct"]["q200"] = True
        # Repaired regime: the historical per-question match is diagnostic, not
        # gating; the control holds only on the count invariants 7/6/+8.
        result = validate(report, historical=old, historical_sha256=digest)
        self.assertTrue(result["control_verified"])
        self.assertFalse(result["control_detail"]["historical_per_question_match"])
        report, old, digest = fixture()
        # Drop one original rescue while keeping recorded counts consistent:
        # original_7 fails on its own, so control is False. The verdict then
        # must be recorded as INVALID-CONTROL.
        set_arm(report, "ul", {**report["arms"]["ul"]["final"]["correct"], "q75": False})
        report["verdict"] = "INVALID-CONTROL"
        result = validate(report, historical=old, historical_sha256=digest)
        self.assertFalse(result["control_verified"])
        report, old, digest = fixture()
        changed = dict(report["arms"]["ul"]["final"]["correct"])
        changed["q75"] = False
        set_arm(report, "ul", changed)
        report["verdict"] = "INVALID-CONTROL"
        self.assertFalse(validate(report, historical=old, historical_sha256=digest)["control_verified"])

    def test_recorded_gate_corruption(self):
        report, _, _ = fixture()
        report["gate"] = validate(report)["gate"]
        validate(report)
        for key in report["gate"]:
            broken = copy.deepcopy(report)
            broken["gate"][key] = not broken["gate"][key]
            with self.assertRaises(AssertionError):
                validate(broken)
            broken["gate"][key] = 1
            with self.assertRaises(AssertionError):
                validate(broken)
        broken = copy.deepcopy(report)
        broken["gate"].pop("net_at_least_8")
        with self.assertRaises(AssertionError):
            validate(broken)

    def test_partial_running_and_invalid_control(self):
        report, old, digest = fixture()
        report["status"] = "running"
        report["arms"] = {"soft_p": report["arms"]["soft_p"]}
        report["arms"]["soft_p"]["history"] = []
        report["arms"]["soft_p"]["final"]["step"] = 0
        result = validate(report)
        self.assertIsNone(result["gate"])
        self.assertIsNone(result["control_verified"])
        report, old, digest = fixture()
        report["status"] = "invalid-control"
        report["verdict"] = "INVALID-CONTROL"
        report["arms"] = {"ul": report["arms"]["ul"]}
        changed = dict(report["arms"]["ul"]["final"]["correct"])
        changed["q75"] = False
        set_arm(report, "ul", changed)
        result = validate(report, historical=old, historical_sha256=digest)
        self.assertFalse(result["control_verified"])
        self.assertIsNone(result["gate"])

    def test_hash_and_sets(self):
        report, old, digest = fixture()
        with self.assertRaises(AssertionError):
            validate(report, True, old, "f" * 64)
        report["sets"]["protect"] = report["sets"]["protect"][1:]
        with self.assertRaises(AssertionError):
            validate(report)


class CheckpointTests(unittest.TestCase):
    def test_frozen_basis_keys_and_budget(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, source_path, formal_path = root / "report.json", root / "source.pt", root / "formal.pt"
            report, _, _ = fixture()
            source = {"down.weight": torch.zeros(13, 1), "up.weight": torch.ones(1, 13),
                      "q.weight": torch.ones(1), "k.weight": torch.ones(1),
                      "gamma_logit": torch.ones(1), "route_logit": torch.ones(1), "extra_buffer": torch.ones(1)}
            torch.save({"state_dict": source}, source_path)
            generator = torch.Generator().manual_seed(0)
            formal_ul = torch.linalg.qr(torch.randn(13, 13, dtype=torch.float64, generator=generator)).Q
            torch.save({"ul": formal_ul, "lost": report["sets"]["original_lost"]}, formal_path)
            report["protocol"]["checkpoint_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
            report["protocol"]["basis_sha256"] = hashlib.sha256(formal_path.read_bytes()).hexdigest()
            path.write_text(json.dumps(report))
            saved_basis = {"ul": formal_ul, "original_lost": report["sets"]["original_lost"],
                           "source_basis_sha256": report["protocol"]["basis_sha256"]}
            torch.save(saved_basis, path.with_suffix(".basis.pt"))
            state = {key: value.clone() for key, value in source.items() if key != "down.weight"}
            state.update({"down.parametrizations.weight.original": source["down.weight"].clone(),
                          "down.parametrizations.weight.0.basis": formal_ul.float(),
                          "down.parametrizations.weight.0.value": torch.zeros(13)})
            for name in ("ul", "soft_p", "soft_k"):
                torch.save(state, path.with_suffix("." + name + ".pt"))
            self.assertEqual(len(audit(path, source_path, formal_path)["saved_checkpoints_verified"]), 3)
            for changed_ul in (formal_ul.float(), formal_ul + 1e-12):
                torch.save({**saved_basis, "ul": changed_ul}, path.with_suffix(".basis.pt"))
                with self.assertRaises(AssertionError):
                    audit(path, source_path, formal_path)
            torch.save(saved_basis, path.with_suffix(".basis.pt"))
            smoke = copy.deepcopy(report)
            smoke["protocol"].update(smoke=True, steps=2, measure_every=1)
            for arm in smoke["arms"].values():
                arm["final"]["step"] = 2
            path.write_text(json.dumps(smoke))
            with self.assertRaises(AssertionError):
                audit(path, source_path, formal_path)
            self.assertEqual(len(audit(path, source_path, formal_path, allow_smoke=True)["saved_checkpoints_verified"]), 3)
            path.write_text(json.dumps(report))
            mutations = [("extra_buffer", torch.zeros(1)),
                         ("down.parametrizations.weight.original", torch.ones(13, 1)),
                         ("down.parametrizations.weight.0.basis", -torch.eye(13)),
                         ("down.parametrizations.weight.0.value", torch.ones(13)),
                         ("down.parametrizations.weight.0.value", torch.full((13,), float("nan")))]
            for key, value in mutations:
                broken = dict(state)
                broken[key] = value
                torch.save(broken, path.with_suffix(".ul.pt"))
                with self.assertRaises(AssertionError):
                    audit(path, source_path, formal_path)
            broken = dict(state)
            del broken["extra_buffer"]
            torch.save(broken, path.with_suffix(".ul.pt"))
            with self.assertRaises(AssertionError):
                audit(path, source_path, formal_path)


if __name__ == "__main__":
    unittest.main()
