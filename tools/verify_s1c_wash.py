"""Verify an S1c-basis-wash report: provenance, arms, control and gate."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def check_report(report, basis_path=None):
    findings = []

    def need(condition, message):
        if not condition:
            findings.append(message)
        return bool(condition)

    protocol = report.get("protocol", {})
    if report.get("status") != "complete":
        findings.append("status not complete: %s" % report.get("status"))
        return findings
    if protocol.get("smoke"):
        findings.append("report is smoke; verification requires formal 60/5")
    if protocol.get("steps") != 60 or protocol.get("measure_every") != 5:
        findings.append("protocol steps/measure_every not 60/5")
    if protocol.get("seed") != 0 or protocol.get("random_seed") != 0:
        findings.append("protocol seed not 0")
    for key in ("checkpoint_sha256", "old_basis_sha256", "historical_result_sha256",
                "gradient_repair_audit_sha256", "historical_driver_sha256",
                "driver_sha256"):
        value = protocol.get(key, "")
        if not (isinstance(value, str) and len(value) == 64
                and all(c in "0123456789abcdef" for c in value)):
            findings.append("protocol %s missing or not a sha256 hex" % key)

    for arm in ("corrected_ul", "old_ul", "random_13d", "unconstrained"):
        entry = report.get("arms", {}).get(arm)
        if not need(isinstance(entry, dict), "arm missing: %s" % arm):
            continue
        final = entry.get("final")
        if not need(isinstance(final, dict), "arm final missing: %s" % arm):
            continue
        need(final.get("rescue_recovered") >= 0, "%s: negative rescue" % arm)
        need(0 <= final.get("protect_harmed", 99) <= 31, "%s: protect harm outside [0,31]" % arm)
        need(entry.get("initial_identity_verified") is True, "%s: identity at init unverified" % arm)
        history = entry.get("history") or []
        need(len(history) in (0, 12), "%s: history length %d" % (arm, len(history)))
        need(entry.get("trainable_dof") in (13, 131072),
             "%s: trainable_dof %s" % (arm, entry.get("trainable_dof")))
        need(isinstance(final.get("correct"), dict) and len(final["correct"]) == 296,
             "%s: final correct map not 296 questions" % arm)

    sets = report.get("sets", {})
    need(len(sets.get("rescue", [])) == 36, "rescue cohort not 36")
    need(len(sets.get("protect", [])) == 31, "protect cohort not 31")
    need(len(sets.get("repair", [])) == 44, "repair cohort not 44")
    need(len(sets.get("retained", [])) == 75, "retained cohort not 75")
    need(report.get("baseline_checks", {}).get("d0_repeat") is True, "d0 repeat check absent")

    finals = {arm: report["arms"][arm]["final"]
              for arm in ("corrected_ul", "old_ul", "random_13d", "unconstrained")
              if arm in report.get("arms", {})}
    if len(finals) == 4:
        # Count invariants must hold on the control arm itself, same as the
        # s1_soft_local one-sided ruling: original 7 / retained harm 6 / net>=8.
        control = finals["old_ul"]
        need(control["original_rescue"] == 7, "old_ul original rescue %s != 7" % control["original_rescue"])
        need(control["retained_harmed"] == 6, "old_ul retained harm %s != 6" % control["retained_harmed"])
        need(control["net_vs_select"] >= 8, "old_ul net %s < 8" % control["net_vs_select"])
        for arm, final in finals.items():
            expected = final["rescue_recovered"] - final["retained_harmed"]
            need(final["net_vs_select"] == expected, "%s: net inconsistent with counts" % arm)
        gate = report.get("gate")
        need(gate in ("PASS", "FAIL", "MARGIN"), "gate value %r" % gate)
    else:
        findings.append("four finals unavailable for gate cross-check")

    if basis_path is not None:
        try:
            import torch
            saved = torch.load(basis_path, map_location="cpu", weights_only=True)
            corrected = saved["corrected_ul"]
            need(tuple(corrected.shape) == (131072, 13),
                 "corrected basis shape %s" % (tuple(corrected.shape),))
            torch.testing.assert_close(corrected.T @ corrected,
                                       torch.eye(13, dtype=corrected.dtype), atol=1e-10, rtol=1e-10)
            old = saved["old_ul"]
            torch.testing.assert_close(old.T @ old, torch.eye(13, dtype=old.dtype), atol=1e-10, rtol=1e-10)
            # rows is keyed by question text
            need(len(saved["rows"]) == 13, "rows count %d" % len(saved["rows"]))
            random_ul = saved["random_ul"]
            torch.testing.assert_close(random_ul.T @ random_ul,
                                       torch.eye(13, dtype=random_ul.dtype), atol=1e-10, rtol=1e-10)
            need(set(saved["lost"]) == set(saved["rows"]),
                 "lost questions and Jacobian rows disagree")
        except Exception as exc:  # noqa: BLE001 - verification reports the failure
            findings.append("basis file unreadable: %s: %s" % (type(exc).__name__, exc))
    return findings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True)
    p.add_argument("--basis", default=None)
    p.add_argument("--require-complete", action="store_true")
    p.add_argument("--require-gate", default=None,
                   help="require this gate verdict (PASS/FAIL/MARGIN)")
    args = p.parse_args()
    report = json.loads(Path(args.report).read_text())
    findings = check_report(report, args.basis)
    if args.require_gate and report.get("gate") != args.require_gate:
        findings.append("gate %r != required %r" % (report.get("gate"), args.require_gate))
    control = (report.get("arms", {}).get("old_ul", {}).get("final") or {})
    summary = {
        "report": args.report,
        "report_sha256": digest(args.report),
        "driver_sha256": report.get("protocol", {}).get("driver_sha256"),
        "old_ul_control": {k: control.get(k) for k in
                           ("original_rescue", "rescue_recovered", "repair_recovered",
                            "protect_harmed", "retained_harmed", "net_vs_select")},
        "corrected_ul": {k: (report.get("arms", {}).get("corrected_ul", {}).get("final") or {}).get(k)
                         for k in ("rescue_recovered", "protect_harmed", "retained_harmed", "net_vs_select")},
        "random_13d": {k: (report.get("arms", {}).get("random_13d", {}).get("final") or {}).get(k)
                       for k in ("rescue_recovered", "protect_harmed", "retained_harmed", "net_vs_select")},
        "unconstrained": {k: (report.get("arms", {}).get("unconstrained", {}).get("final") or {}).get(k)
                          for k in ("rescue_recovered", "protect_harmed", "retained_harmed", "net_vs_select")},
        "gate": report.get("gate"),
        "verdict": report.get("verdict"),
        "control_verified": not findings,
        "findings": findings,
    }
    if args.require_complete and findings:
        raise SystemExit("verification failed:\n" + "\n".join(findings))
    print(json.dumps(summary, indent=2))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
