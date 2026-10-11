#!/usr/bin/env python3
"""Verify the isolated S1d stage2 oracle diagnostic."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = list(range(10, 61, 10))
HASHES = ("checkpoint_sha256", "selfsup_rows_sha256", "gold_rows_sha256")
BASIS = "frozen per-window selfsup SVD-13 basis (runs/s1d_selfsup_rows_v2.pt)"


def req(ok, msg):
    if not ok:
        raise AssertionError(msg)


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def sha(x, label):
    req(isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x), label + " invalid sha256")


def metric(m, sets, baseline, label):
    c = m["correct"]
    req(isinstance(c, dict) and set(c) == set(baseline), label + " universe")
    req(all(type(v) is bool for v in c.values()), label + " booleans")
    rec = [q for q in sets["rescue_d0_wrong"] if c[q]]
    harm = [q for q in baseline if baseline[q] and not c[q]]
    expected = {"new_recovery": len(rec), "new_recovery_questions": rec,
                "rescue_recovered_s1c_style": sum(c[q] for q in sets["rescue"]),
                "d0_harmed": len(harm), "d0_harmed_questions": harm,
                "d0_new_recovery": len(rec), "net_vs_d0": len(rec) - len(harm)}
    for k, v in expected.items():
        req(m[k] == v, label + " " + k)
    req(finite(m["accuracy"]) and math.isclose(m["accuracy"], sum(c.values()) / len(c), rel_tol=1e-9, abs_tol=1e-12), label + " accuracy")


def predictions(p, c, label):
    req(isinstance(p, dict) and set(p) == set(c), label + " predictions universe")
    req(all(isinstance(v, str) for v in p.values()), label + " predictions strings")


def verify_sets(w):
    b, s = w["baselines"], w["sets"]
    f, se, d = b["full"]["correct"], b["select"]["correct"], b["d0"]["correct"]
    req(bool(f) and set(f) == set(se) == set(d), "TF baseline universes")
    retained = [q for q in f if f[q] and se[q]]
    rescue = [q for q in f if f[q] and not se[q]]
    want = {"rescue": rescue, "retained": retained,
            "protect": [q for q in retained if d[q]],
            "repair": [q for q in retained if not d[q]],
            "rescue_d0_correct": [q for q in rescue if d[q]],
            "rescue_d0_wrong": [q for q in rescue if not d[q]]}
    req(s == want, "cohort sets")


def baselines(w, label):
    verify_sets(w)
    b, s = w["baselines"], w["sets"]
    for n in ("full", "select", "d0"):
        metric(b[n], s, b["d0"]["correct"], label + "/" + n)
    for n in ("gen_full", "gen_d0"):
        metric(b[n], s, b["gen_d0"]["correct"], label + "/" + n)
        predictions(b["gen_predictions_" + n[4:]], b[n]["correct"], label + "/" + n)


def score(m):
    return "recovery/harm/net=%d/%d/%d accuracy=%.6f" % (m["new_recovery"], m["d0_harmed"], m["net_vs_d0"], m["accuracy"])


def show(name, e):
    print(name + " final TF: " + score(e["final"]))
    print(name + " final GEN: " + score(e["final"]["gen"]))
    for i, p in enumerate(e["history"], 1):
        step = p.get("step", i * 10)
        print("%s history step=%d loss=%.8f norm=%.8f TF: %s" % (name, step, p["sup_loss"], p["deviation"], score(p)))
        if "gen" in p:
            print("%s history step=%d GEN: %s" % (name, step, score(p["gen"])))


def verify(r, old):
    req(r["schema_version"] == "s1d_stage2_oracle/1", "schema")
    req(r["status"] == "complete", "oracle incomplete")
    req(r["diagnostic_only"] is True and r["excluded_from_method_results"] is True, "oracle flags")
    current = hashlib.sha256((ROOT / "tools/s1d_stage2_dev.py").read_bytes()).hexdigest()
    req(r["driver_sha256"] in (current, "8053c26cc34e11ecb4b9a0b845334c23e755ba2e63dad2cca40ee71c54a6f190"), "current driver sha mismatch")
    p = r["protocol"]
    for k, v in (("steps", 60), ("measure_every", 10), ("lr", .003), ("budget", 1.), ("seed", 0), ("max_new_tokens", 32)):
        req(p[k] == v, "protocol " + k)
    req(p["arms"] == p["selected_arms"] == ["oracle_13d"], "oracle arm")
    req(p["dev_windows"] == [[3, 17510]], "oracle window")
    req(p["basis"] == BASIS, "basis declaration")
    req(p["supervision"] == "real QA answer-token CE on rescue_d0_wrong only; first gold alias", "supervision")
    req(p["optimizer"] == "AdamW" and p["weight_decay"] == .01, "optimizer")
    req(p["gradient_reduction"] == "sum of per-question mean answer-token CE", "gradient reduction")
    req(p["mathematical_upper_bound"] is False, "upper bound claim")
    od = r["oracle_diagnostic"]
    req(od["basis"] == "unchanged selfsup SVD-13 basis; no QA-based basis rebuild", "diagnostic basis")
    req(od["supervision"] == "real QA answer-token CE on rescue_d0_wrong only", "diagnostic supervision")
    req(old["status"] == "complete", "old incomplete")
    for k in HASHES:
        sha(p[k], k); req(p[k] == old["protocol"][k], "hash mismatch " + k)
    req(len(r["windows"]) == len(old["windows"]) == 1, "window count")
    w, ow = r["windows"][0], old["windows"][0]
    req(w["window"] == ow["window"] == [3, 17510], "window")
    req(set(w["arms"]) == {"oracle_13d"}, "oracle window arms")
    req(w["sets"] == ow["sets"], "sets mismatch")
    for n in ("full", "select", "d0"):
        req(w["baselines"][n] == ow["baselines"][n], "TF baseline mismatch " + n)
    baselines(w, "oracle"); baselines(ow, "old")
    e = w["arms"]["oracle_13d"]
    req(e["trainable_dof"] == 13 and e["initial_identity_verified"] is True, "DOF/identity")
    sha(e["basis_sha256"], "basis")
    q = w["sets"]["rescue_d0_wrong"]
    req(q and e["supervision_questions"] == q and e["n_supervision_questions"] == len(q), "training set")
    req(set(e["supervision_answers"]) == set(q) and all(isinstance(x, str) and x.strip() for x in e["supervision_answers"].values()), "training answers")
    req([x["step"] for x in e["history"]] == STEPS, "history steps")
    req(e["final"] == e["history"][-1], "final history")
    for x in e["history"]:
        lab = "oracle step %d" % x["step"]
        metric(x, w["sets"], w["baselines"]["d0"]["correct"], lab)
        req(finite(x["sup_loss"]) and finite(x["deviation"]) and 0 <= x["deviation"] <= 1.000001, lab + " loss/norm")
        req(x["basis_unchanged"] is True, lab + " basis changed")
        a = x["alpha"]
        req(isinstance(a, list) and len(a) == 13 and all(finite(v) for v in a), lab + " alpha")
        n = math.hypot(*a)
        req(n <= 1.000001 and math.isclose(n, x["deviation"], rel_tol=1e-5, abs_tol=1e-6), lab + " alpha norm")
        if "gen" in x:
            metric(x["gen"], w["sets"], w["baselines"]["gen_d0"]["correct"], lab + " GEN")
            predictions(x["gen"]["predictions"], x["gen"]["correct"], lab + " GEN")
    req("gen" in e["final"], "oracle final GEN")
    oe = ow["arms"]["selfsup_13d"]
    for x in oe["history"]:
        metric(x, ow["sets"], ow["baselines"]["d0"]["correct"], "old selfsup")
    req(oe["final"] == oe["history"][-1], "old final history")
    metric(oe["final"]["gen"], ow["sets"], ow["baselines"]["gen_d0"]["correct"], "old GEN")
    predictions(oe["final"]["gen"]["predictions"], oe["final"]["gen"]["correct"], "old GEN")
    for n in ("gen_full", "gen_d0"):
        oc, nc = ow["baselines"][n]["correct"], w["baselines"][n]["correct"]
        op, np = ow["baselines"]["gen_predictions_" + n[4:]], w["baselines"]["gen_predictions_" + n[4:]]
        print("Non-decisive GEN baseline %s: correctness_flips=%d prediction_changes=%d" % (n, sum(oc[k] != nc[k] for k in oc), sum(op[k] != np[k] for k in op)))
    show("old selfsup_13d", oe); show("oracle_13d", e)
    print("PASS: oracle flags, current driver SHA, hashes, TF baselines/sets, frozen basis, training set, TF/GEN metrics, finite losses/alpha, norm budgets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=ROOT / "runs/s1d_stage2_oracle.w17510.json")
    ap.add_argument("--previous", type=Path, default=ROOT / "runs/s1d_stage2_dev.w17510.sr.json")
    a = ap.parse_args()
    if not a.report.is_file():
        print("SKIP: oracle report does not exist: %s" % a.report)
        return
    verify(json.loads(a.report.read_text()), json.loads(a.previous.read_text()))


if __name__ == "__main__":
    main()
