#!/usr/bin/env python3
"""Verify the fixed-weight protected S1d QA oracle diagnostic."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = list(range(10, 61, 10))
BASIS = "frozen per-window selfsup SVD-13 basis (runs/s1d_selfsup_rows_v2.pt)"

def req(ok, msg):
    if not ok:
        raise AssertionError(msg)

def finite(x):
    return type(x) in (int, float) and math.isfinite(x)

def sha(x, label):
    req(isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x), label)

def metric(m, sets, baseline, label):
    c = m["correct"]
    req(set(c) == set(baseline) and all(type(v) is bool for v in c.values()), label)
    rec = [q for q in sets["rescue_d0_wrong"] if c[q]]
    harm = [q for q in baseline if baseline[q] and not c[q]]
    expected = {
        "new_recovery": len(rec), "new_recovery_questions": rec,
        "rescue_recovered_s1c_style": sum(c[q] for q in sets["rescue"]),
        "d0_harmed": len(harm), "d0_harmed_questions": harm,
        "d0_new_recovery": len(rec), "net_vs_d0": len(rec) - len(harm),
    }
    for k, v in expected.items():
        req(m[k] == v, label + "/" + k)
    req(finite(m["accuracy"]) and math.isclose(
        m["accuracy"], sum(c.values()) / max(1, len(c)), abs_tol=1e-12), label + "/accuracy")

def verify(r, old, allow_running=False):
    req(r["schema_version"] == "s1d_stage2_oracle_protected/1", "schema")
    complete = r["status"] == "complete"
    req(complete or (allow_running and r["status"] == "running"), "incomplete")
    if complete:
        req(r["launcher"]["torchrun_exit_code"] == 0 and
            r["launcher"]["all_workers_succeeded"] is True and
            r["launcher"]["world_size"] == 8, "worker group completion")
    else:
        req("launcher" not in r and "failure" not in r, "running report has terminal certificate")
    req(r["diagnostic_only"] is True and r["excluded_from_method_results"] is True, "flags")
    current = hashlib.sha256((ROOT / "tools/s1d_stage2_dev.py").read_bytes()).hexdigest()
    req(r["driver_sha256"] == current, "driver sha")
    p = r["protocol"]
    for key in ("checkpoint_sha256", "selfsup_rows_sha256", "gold_rows_sha256"):
        sha(p[key], key)
        req(p[key] == old["protocol"][key], "original artifact/" + key)
    for k, v in (("steps", 60), ("measure_every", 10), ("lr", .003), ("budget", 1.),
                 ("seed", 0), ("max_new_tokens", 32), ("qa_max_tokens", 256),
                 ("protect_weight", 1.0),
                 ("skip_generation", True)):
        req(p[k] == v, "protocol/" + k)
    parallel = p["parallel_qa"]
    req(parallel["world_size"] == 8, "parallel/world size")
    devices = parallel["physical_devices"]
    req(len(devices) == len(set(devices)) == 8 and
        all(type(device) is int and device >= 0 for device in devices), "parallel/unique devices")
    cache_manifest = ROOT / "runs/s1d_reader_fp16_manifest.json"
    req(parallel["reader_cache_manifest_sha256"] ==
        hashlib.sha256(cache_manifest.read_bytes()).hexdigest(), "reader cache certificate")
    req(p["optimizer"] == "AdamW" and p["weight_decay"] == .01, "optimizer")
    req(parallel["gradient_reduction"] ==
        "SUM local alpha gradients with global cohort denominators", "parallel/gradient SUM")
    req(p["arms"] == p["selected_arms"] == ["oracle_13d_protected"], "arm")
    req(p["dev_windows"] == [[3, 17510]], "window")
    req(p["basis"] == BASIS, "basis declaration")
    req(p["protection_set"] == "all questions D0 answers correctly", "protection set")
    req(p["protect_loss"] == "mean over questions ReLU(current answer-token CE - D0 answer-token CE)", "protect loss")
    req(p["gradient_reduction"] == "mean rescue answer-token CE plus mean protected hinge CE", "gradient reduction")
    req(p["mathematical_upper_bound"] is False, "upper bound")
    req(r["oracle_diagnostic"]["protect_weight"] == 1.0, "diagnostic weight")
    req(len(r["windows"]) == len(old["windows"]) == 1, "window count")
    w, ow = r["windows"][0], old["windows"][0]
    req(w["window"] == ow["window"] == [3, 17510], "window id")
    req(w["sets"] == ow["sets"], "sets")
    for n in ("full", "select", "d0"):
        req(w["baselines"][n] == ow["baselines"][n], "baseline/" + n)
    e = w["arms"]["oracle_13d_protected"]
    req(e["trainable_dof"] == 13 and e["initial_identity_verified"] is True, "dof/identity")
    sha(e["basis_sha256"], "basis hash")
    req(e["basis_sha256"] == ow["arms"]["oracle_13d"]["basis_sha256"], "original frozen basis")
    q = w["sets"]["rescue_d0_wrong"]
    protect = [x for x, ok in w["baselines"]["d0"]["correct"].items() if ok]
    req(e["supervision_questions"] == q and e["n_supervision_questions"] == len(q), "training set")
    req(e["protection_questions"] == protect and e["n_protection_questions"] == len(protect), "protection set")
    req(e["protection_weight"] == 1.0, "arm weight")
    req(set(e["protection_baseline_ce"]) == set(protect), "baseline CE universe")
    req(all(finite(value) and value >= 0 for value in e["protection_baseline_ce"].values()),
        "baseline CE finite")
    measured_steps = [x["step"] for x in e["history"]]
    req(bool(measured_steps) and measured_steps == STEPS[:len(measured_steps)], "history prefix")
    if complete:
        req(measured_steps == STEPS and e["final"] == e["history"][-1], "history")
    for x in e["history"]:
        metric(x, w["sets"], w["baselines"]["d0"]["correct"], "step")
        req(finite(x["sup_loss"]) and finite(x["rescue_loss"]) and finite(x["protect_penalty"]), "loss")
        req(x["rescue_loss"] >= 0 and x["protect_penalty"] >= 0 and
            math.isclose(x["sup_loss"], x["rescue_loss"] + x["protect_penalty"],
                         rel_tol=1e-10, abs_tol=1e-10), "fixed-weight loss decomposition")
        req(finite(x["deviation"]) and 0 <= x["deviation"] <= 1.000001, "norm")
        req(x["basis_unchanged"] is True, "basis changed")
        req(len(x["alpha"]) == 13 and all(finite(v) for v in x["alpha"]), "alpha")
        req(math.isclose(math.hypot(*x["alpha"]), x["deviation"], rel_tol=1e-5, abs_tol=1e-6), "alpha norm")
        req(x.get("gen") is None, "GEN must be explicitly skipped")
    if not complete:
        print("PARTIAL PASS: measured steps %s; status=running; worker completion uncertified" % measured_steps)
        return False
    print("protected final TF: recovery/harm/net=%d/%d/%d" %
          (e["final"]["new_recovery"], e["final"]["d0_harmed"], e["final"]["net_vs_d0"]))
    print("PASS: protected oracle flags, fixed weight, D0 CE baseline, cohorts, frozen basis, alpha audit, TF/GEN metrics")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=ROOT / "runs/s1d_stage2_oracle_protected.w17510.json")
    ap.add_argument("--previous", type=Path, default=ROOT / "runs/s1d_stage2_oracle.w17510.json")
    ap.add_argument("--allow-running", action="store_true",
                    help="audit recorded checkpoints only; completion remains uncertified")
    a = ap.parse_args()
    if not a.report.is_file():
        print("SKIP: protected report does not exist: %s" % a.report)
        return
    verify(json.loads(a.report.read_text()), json.loads(a.previous.read_text()),
           allow_running=a.allow_running)

if __name__ == "__main__":
    main()
