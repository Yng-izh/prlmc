"""Server-side verifier for the s1c global oracle report and artifacts.

Checks (all read-only, CPU):
  1. protocol hashes in the report match the actual files (driver, checkpoint,
     wash basis, wash report, gradient audit).
  2. every expected window x arm pair is present, finished (60 steps) and
     identity-at-init verified.
  3. gate arithmetic cross-check: totals recomputed from per-window arm
     ``final`` blocks must reproduce the stored gate totals, gaps and verdict.
  4. uglobal.pt: shape (131072, 13), orthonormal columns, rank 13; recomputed
     SVD energy fraction over rows.pt matches the stored u_global_audit value.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

EXPECTED_WINDOWS = [(2, 3450), (2, 3830)]
EXPECTED_ARMS = ("global_13d", "random_13d", "unconstrained", "local_13d")
EXPECTED_STEPS = 60
FLAT_DIM = 131072
RANK = 13

HASH_FILES = {
    "driver_sha256": "tools/s1c_global_oracle.py",
    "checkpoint_sha256": "runs/e6d_interleave_ckpt.pt",
    "wash_basis_sha256": "runs/s1c_wash.basis.pt",
    "wash_report_sha256": "runs/s1c_wash.json",
    "gradient_repair_audit_sha256": "runs/s1_checkpoint_recompute_production_cpu.json",
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="runs/s1c_global_oracle.json")
    ap.add_argument("--uglobal", default="runs/s1c_global_oracle.uglobal.pt")
    ap.add_argument("--rows", default="runs/s1c_global_rows.pt")
    args = ap.parse_args()

    failures = []

    def check(name, ok, detail=""):
        tag = "PASS" if ok else "FAIL"
        print(f"[verify] {tag} {name}" + (f" ({detail})" if detail else ""))
        if not ok:
            failures.append(name)

    report = json.load(open(args.report))
    protocol = report.get("protocol", {})

    # 1. hash chain
    for key, path in HASH_FILES.items():
        stored = protocol.get(key)
        if stored is None:
            check(f"hash:{key}", False, "missing in report")
            continue
        actual = sha256_of(path)
        check(f"hash:{key}", stored == actual, f"stored={stored[:12]} actual={actual[:12]}")

    # 2. window/arm completeness
    windows = {tuple(w["window"]): w for w in report.get("windows", [])}
    for wkey in EXPECTED_WINDOWS:
        w = windows.get(wkey)
        check(f"window_present:{wkey}", w is not None)
        if w is None:
            continue
        for arm in EXPECTED_ARMS:
            a = w.get("arms", {}).get(arm)
            ok = (
                a is not None
                and a.get("final", {}).get("step") == EXPECTED_STEPS
                and len(a.get("history", [])) == EXPECTED_STEPS // 5
            )
            check(f"arm_done:{wkey}:{arm}", ok,
                  f"final_step={a.get('final', {}).get('step') if a else 'missing'}")
            if a is not None:
                check(f"identity_verified:{wkey}:{arm}",
                      a.get("initial_identity_verified") is True)

    # 3. gate arithmetic cross-check
    per_arm = {arm: {"rescue_recovered": 0, "retained_harmed": 0, "net_vs_select": 0}
               for arm in EXPECTED_ARMS}
    for wkey in EXPECTED_WINDOWS:
        w = windows.get(wkey)
        if w is None:
            continue
        for arm in EXPECTED_ARMS:
            a = w.get("arms", {}).get(arm)
            if not a:
                continue
            f = a.get("final", {})
            if f.get("step") != EXPECTED_STEPS:
                continue
            per_arm[arm]["rescue_recovered"] += f.get("rescue_recovered", 0)
            per_arm[arm]["retained_harmed"] += f.get("retained_harmed", 0)
            per_arm[arm]["net_vs_select"] += f.get("net_vs_select", 0)

    gate = report.get("gate", {})
    gt = gate.get("totals", {})
    for arm in EXPECTED_ARMS:
        stored = gt.get(arm, {})
        for field in ("rescue_recovered", "retained_harmed", "net_vs_select"):
            ok = stored.get(field) == per_arm[arm][field]
            check(f"gate_totals:{arm}:{field}", ok,
                  f"stored={stored.get(field)} recomputed={per_arm[arm][field]}")

    net = gt.get("global_13d", {}).get("net_vs_select")
    net_random = gt.get("random_13d", {}).get("net_vs_select")
    net_unc = gt.get("unconstrained", {}).get("net_vs_select")
    rr_global = gt.get("global_13d", {}).get("rescue_recovered", 0)
    rr_random = gt.get("random_13d", {}).get("rescue_recovered", 0)
    rescue_gap = rr_global - rr_random
    net_gap = net - net_random if net is not None and net_random is not None else None

    check("gate:net", gate.get("net") == net, f"stored={gate.get('net')} recomputed={net}")
    check("gate:net_random", gate.get("net_random") == net_random)
    check("gate:net_unconstrained", gate.get("net_unconstrained") == net_unc)
    check("gate:rescue_gap", gate.get("rescue_gap") == rescue_gap, f"recomputed={rescue_gap}")
    check("gate:net_gap", gate.get("net_gap") == net_gap, f"recomputed={net_gap}")

    if all(v is not None for v in (net, net_random, net_unc, rescue_gap, net_gap)):
        if net >= 3 and rescue_gap >= 3 and net_gap >= 3 and net >= net_unc:
            expected_verdict = "PASS"
        elif net < 0 or (rescue_gap <= 0 and net_gap <= 0) or net < net_unc - 2:
            expected_verdict = "FAIL"
        else:
            expected_verdict = "MARGIN"
        check("gate:verdict", gate.get("verdict") == expected_verdict,
              f"stored={gate.get('verdict')} recomputed={expected_verdict}")

    # 4. uglobal geometry
    try:
        blob = torch.load(args.uglobal, map_location="cpu")
        u = blob["u_global"] if isinstance(blob, dict) else blob
        if not isinstance(u, torch.Tensor):
            u = torch.as_tensor(u)
        check("uglobal:shape", tuple(u.shape) == (FLAT_DIM, RANK), str(tuple(u.shape)))
        gram = (u.T @ u).float()
        eye = torch.eye(RANK)
        orth_err = (gram - eye).abs().max().item()
        check("uglobal:orthonormal", orth_err < 1e-4, f"max_err={orth_err:.2e}")
        rank = torch.linalg.matrix_rank(u.float()).item()
        check("uglobal:rank", rank == RANK, str(rank))

        rows_blob = torch.load(args.rows, map_location="cpu")
        rbw = rows_blob["rows_by_window"]
        n_rows = sum(len(v) for v in rbw.values())
        check("rows:count", n_rows == 240, f"n_rows={n_rows}")
        check("rows:checkpoint_hash",
              rows_blob.get("checkpoint_sha256") == protocol.get("checkpoint_sha256"))
        row_mat = torch.stack([
            t for v in rbw.values() for t in v.values()
        ]).float()
        s = torch.linalg.svdvals(row_mat.float())
        total = (s ** 2).sum().item()
        top = (s ** 2)[:RANK].sum().item()
        energy = top / total
        stored_energy = report.get("u_global_audit", {}).get("energy_fraction")
        check("uglobal:energy_fraction",
              stored_energy is not None and math.isclose(energy, stored_energy, rel_tol=1e-3),
              f"recomputed={energy:.4f} stored={stored_energy}")
        # top singular values cross-check
        stored_sv = report.get("u_global_audit", {}).get("top_singular_values", [])
        if stored_sv:
            sv_err = max(abs(a - b) for a, b in zip(s[:RANK].tolist(), stored_sv))
            check("uglobal:top_sv", sv_err < 1e-2, f"max_err={sv_err:.2e}")
    except Exception as exc:  # noqa: BLE001
        check("uglobal:load", False, str(exc)[:120])

    print()
    if failures:
        print(f"[verify] RESULT: {len(failures)} failure(s): {failures}")
        raise SystemExit(1)
    print("[verify] RESULT: all checks passed")


if __name__ == "__main__":
    main()
