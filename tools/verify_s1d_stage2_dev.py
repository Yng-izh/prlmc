#!/usr/bin/env python3
"""Independently verify the four completed S1d stage2 dev shards."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"selfsup_13d": 13, "random_13d": 13, "unconstrained": 131072}
LEGACY_DRIVER_SHA = "39bbbcc8e4657c00c04dc1d683a929e6b4c533f7d03b6ebf5ae9859aa80b3c5e"
TOTALS = {arm: [0, 0, 0] for arm in ARMS}


def verify_metric(metric, sets, baseline):
    correct = metric["correct"]
    assert set(correct) == set(baseline), "question universe mismatch"
    recovered = [q for q in sets["rescue_d0_wrong"] if correct[q]]
    harmed = [q for q in baseline if baseline[q] and not correct[q]]
    assert metric["new_recovery"] == len(recovered)
    assert metric["new_recovery_questions"] == recovered
    assert metric["d0_harmed"] == len(harmed)
    assert metric["d0_harmed_questions"] == harmed
    assert metric["net_vs_d0"] == len(recovered) - len(harmed)
    assert math.isclose(metric["accuracy"], sum(correct.values()) / len(correct))


def main():
    driver_sha = hashlib.sha256((ROOT / "tools/s1d_stage2_dev.py").read_bytes()).hexdigest()
    accepted_driver_shas = {driver_sha, LEGACY_DRIVER_SHA}
    for start in (16370, 17510):
        shards = []
        for suffix in ("sr", "u"):
            path = ROOT / "runs" / f"s1d_stage2_dev.w{start}.{suffix}.json"
            report = json.loads(path.read_text())
            assert report["status"] == "complete", path
            assert report["driver_sha256"] in accepted_driver_shas, "driver hash mismatch"
            protocol = report["protocol"]
            assert (protocol["steps"], protocol["measure_every"], protocol["lr"],
                    protocol["budget"], protocol["max_new_tokens"]) == (60, 10, 0.003, 1.0, 32)
            assert len(report["windows"]) == 1
            window = report["windows"][0]
            assert window["window"] == [3, start]
            expected = {"selfsup_13d", "random_13d"} if suffix == "sr" else {"unconstrained"}
            assert set(window["arms"]) == expected
            assert set(protocol["selected_arms"]) == expected
            shards.append(window)
        assert shards[0]["baselines"] == shards[1]["baselines"]
        assert shards[0]["sets"] == shards[1]["sets"]
        for window in shards:
            baseline = window["baselines"]
            sets = window["sets"]
            assert sets["rescue_d0_wrong"] == [q for q in sets["rescue"]
                                               if not baseline["d0"]["correct"][q]]
            for name in ("full", "select", "d0"):
                verify_metric(baseline[name], sets, baseline["d0"]["correct"])
            for name in ("gen_full", "gen_d0"):
                verify_metric(baseline[name], sets, baseline["gen_d0"]["correct"])
            for arm, entry in window["arms"].items():
                assert entry["initial_identity_verified"] is True
                assert entry["trainable_dof"] == ARMS[arm]
                history = entry["history"]
                assert len(history) == 6
                assert entry["final"] == history[-1]
                for point in history:
                    verify_metric(point, sets, baseline["d0"]["correct"])
                    assert math.isfinite(point["sup_loss"])
                    assert 0 <= point["deviation"] <= 1.0 + 1e-6
                final = entry["final"]
                verify_metric(final["gen"], sets, baseline["gen_d0"]["correct"])
                assert set(final["gen"]["predictions"]) == set(final["gen"]["correct"])
                scores = [final[k] for k in ("new_recovery", "d0_harmed", "net_vs_d0")]
                TOTALS[arm] = [a + b for a, b in zip(TOTALS[arm], scores)]
                print(f"3,{start} {arm}: recovery/harm/net={scores}")
    assert TOTALS == {"selfsup_13d": [13, 6, 7], "random_13d": [0, 1, -1],
                      "unconstrained": [14, 10, 4]}
    print("PASS: four completed shards, exact driver hashes, shared baselines, "
          "six arms, all TF/GEN metrics, six measurements per arm, norm budgets")


if __name__ == "__main__":
    main()
