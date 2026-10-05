"""Independently validate S1-local JSON metrics; never select a training step."""
import argparse
import hashlib
import json
from pathlib import Path


ARMS = ("select", "unconstrained", "coefficient_oracle", "subspace_ul", "protected_usafe")


def validate(report, require_complete=False):
    protocol = report["protocol"]
    assert not protocol["smoke"], "smoke output is not a formal result"
    assert protocol["window"] == [3, 17890]
    assert protocol["steps"] == 60 and protocol["seed"] == 0
    assert protocol["lr"] == 0.003 and protocol["parameter_budget"] == 1.0
    assert protocol["coefficient_budget"] == 0.5
    lost = report["sets"]["lost"]
    retained = report["sets"]["retained_full_and_select"]
    historical = report["historical_s0_audit"]
    assert len(lost) == len(set(lost)) == 13
    assert set(lost) == set(historical["historical_lost"])
    assert set(lost) <= set(historical["deterministic_lost"])
    assert len(retained) == len(set(retained)) >= 10
    baseline = report["checkpoint_diagnostic"]["correct"]
    native = report["arms"]["select"]["final"]["correct"]
    assert all(native[q] for q in retained)
    retained_at_d0 = [q for q in retained if baseline[q]]
    assert set(retained_at_d0) == set(report["sets"]["retained_also_correct_at_D0"])
    assert all(not baseline[q] for q in lost)
    assert report["reader_config"]["attention_dropout"] == 0
    assert report["reader_dropout_audit"], "dropout module audit is missing"
    if "basis" in report:
        assert report["basis"]["protection_overlap"] <= 1e-5
        assert 0 < report["basis"]["safe_rank"] <= report["basis"]["lost_rank"] <= 13
    summary = []
    for name, arm in report["arms"].items():
        assert name in ARMS
        measurements = arm.get("history", []) or [arm["final"]]
        for result in measurements:
            correct = result["correct"]
            assert set(correct) == set(baseline) == set(native)
            assert all(type(value) is bool for value in correct.values())
            rescue = [q for q in lost if correct[q]]
            harm = [q for q in retained if not correct[q]]
            added = [q for q in retained_at_d0 if not correct[q]]
            added_all = [q for q, previous in baseline.items() if previous and not correct[q]]
            for count_key, list_key, expected in [
                ("rescued", "rescued_questions", rescue),
                ("harmed_vs_select", "harmed_questions", harm),
                ("newly_harmed_vs_D0", "newly_harmed_questions", added),
                ("newly_harmed_vs_D0_all", "newly_harmed_questions_all", added_all),
            ]:
                assert result[count_key] == len(expected), (name, count_key)
                assert set(result[list_key]) == set(expected), (name, list_key)
            assert abs(result["accuracy"] - sum(correct.values()) / len(correct)) < 1e-12
            assert abs(result["harm_rate"] - len(harm) / len(retained)) < 1e-12
        if arm.get("history"):
            assert arm["final"] == arm["history"][-1]
            steps = [item["step"] for item in arm["history"]]
            assert steps == list(range(5, steps[-1] + 1, 5))
        final = arm["final"]
        summary.append({"arm": name, "step": final["step"],
                        "rescue": final["rescued"], "harm": final["harmed_vs_select"],
                        "new_harm_all": final["newly_harmed_vs_D0_all"],
                        "correct": sum(final["correct"].values()), "n": len(baseline),
                        "net_full_eligible_vs_select": final["net_vs_select"]})
    complete = report["status"] == "complete"
    if require_complete:
        assert complete, "formal run is still incomplete"
    gate = None
    if complete:
        assert set(report["arms"]) == set(ARMS)
        assert all(report["arms"][name]["final"]["step"] == 60 for name in ARMS[1:])
        assert all(len(report["arms"][name]["history"]) == 12 for name in ARMS[1:])
        finals = {name: arm["final"] for name, arm in report["arms"].items()}
        protected, oracle = finals["protected_usafe"], finals["coefficient_oracle"]
        gate = {
            "rescue_at_least_7": protected["rescued"] >= 7,
            "harm_strictly_below_oracle": protected["harmed_vs_select"] < oracle["harmed_vs_select"],
            "harm_at_most_half_oracle": protected["harmed_vs_select"] <= oracle["harmed_vs_select"] // 2,
            "harm_not_above_free_or_ul": all(protected["harmed_vs_select"] <= finals[name]["harmed_vs_select"]
                                             for name in ("unconstrained", "subspace_ul")),
            "nonnegative_net_vs_select": protected["net_vs_select"] >= 0,
            "retained_at_least_10": len(retained) >= 10,
        }
        assert report["verdict"] == ("GO" if all(gate.values()) else "NO-GO")
    return {"status": report["status"], "metrics_verified": True, "arms": summary,
            "gate": gate, "net_metric_note": "Net gate is checked as recorded; expanded full-pool correctness is not serialized."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    path = Path(args.report)
    payload = path.read_bytes()
    result = validate(json.loads(payload), args.require_complete)
    result["report_sha256"] = hashlib.sha256(payload).hexdigest()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
