"""Tests of verify_s1c_wash.check_report over synthetic reports."""
import unittest

from verify_s1c_wash import check_report


def final(arm, rescue=15, harm=6, protect_harm=0, original=7, correct_len=296):
    return {
        "original_rescue": original, "rescue_recovered": rescue,
        "repair_recovered": 38, "protect_harmed": protect_harm,
        "retained_harmed": harm, "net_vs_select": rescue - harm,
        "correct": {"q%03d" % i: False for i in range(correct_len)},
    }


def make_report(**overrides):
    arms = {
        "corrected_ul": {"final": final("corrected_ul"), "history": [{}] * 12,
                         "trainable_dof": 13, "initial_identity_verified": True},
        "old_ul": {"final": final("old_ul"), "history": [{}] * 12,
                   "trainable_dof": 13, "initial_identity_verified": True},
        "random_13d": {"final": final("random_13d", rescue=8, harm=5),
                       "history": [{}] * 12, "trainable_dof": 13,
                       "initial_identity_verified": True},
        "unconstrained": {"final": final("unconstrained", rescue=10, harm=17),
                          "history": [{}] * 12, "trainable_dof": 131072,
                          "initial_identity_verified": True},
    }
    report = {
        "status": "complete", "gate": "PASS", "verdict": "PASS",
        "protocol": {
            "smoke": False, "steps": 60, "measure_every": 5, "seed": 0,
            "random_seed": 0,
            "checkpoint_sha256": "a" * 64, "old_basis_sha256": "b" * 64,
            "historical_result_sha256": "c" * 64,
            "gradient_repair_audit_sha256": "d" * 64,
            "historical_driver_sha256": "e" * 64, "driver_sha256": "f" * 64,
        },
        "baseline_checks": {"d0_repeat": True},
        "sets": {
            "rescue": ["r%02d" % i for i in range(36)],
            "protect": ["p%02d" % i for i in range(31)],
            "repair": ["x%02d" % i for i in range(44)],
            "retained": ["t%02d" % i for i in range(75)],
        },
        "arms": arms,
    }
    report.update(overrides)
    return report


class VerifyWashTest(unittest.TestCase):
    def test_clean_report_passes(self):
        self.assertEqual(check_report(make_report()), [])

    def test_missing_arm_and_bad_dof(self):
        report = make_report()
        del report["arms"]["random_13d"]
        findings = check_report(report)
        self.assertTrue(any("arm missing: random_13d" in f for f in findings))
        self.assertTrue(any("four finals unavailable" in f for f in findings))

        report = make_report()
        report["arms"]["corrected_ul"]["trainable_dof"] = 7
        self.assertTrue(any("trainable_dof" in f for f in check_report(report)))

    def test_control_invariants_enforced(self):
        # Break only the net field, leaving the counts self-consistent with the
        # frozen control, so the arithmetic cross-check is what must fire.
        report = make_report()
        report["arms"]["old_ul"]["final"]["net_vs_select"] = 99
        findings = check_report(report)
        self.assertTrue(any("net inconsistent" in f for f in findings))
        # The count-invariant failures must also fire: original 7 and harm 6 are
        # frozen expectations on the old_ul control arm.
        report = make_report()
        report["arms"]["old_ul"]["final"]["original_rescue"] = 6
        self.assertTrue(any("original rescue" in f for f in check_report(report)))

        report = make_report()
        report["arms"]["old_ul"]["final"]["retained_harmed"] = 5
        report["arms"]["old_ul"]["final"]["net_vs_select"] = 10
        self.assertTrue(any("retained harm" in f for f in check_report(report)))

    def test_gate_value_and_protocol_guards(self):
        report = make_report(gate="MAYBE")
        self.assertTrue(any("gate value" in f for f in check_report(report)))
        report = make_report()
        report["protocol"]["steps"] = 30
        self.assertTrue(any("steps/measure_every" in f for f in check_report(report)))
        report = make_report(status="failed")
        self.assertEqual(len(check_report(report)), 1)


if __name__ == "__main__":
    unittest.main()
