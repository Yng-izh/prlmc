"""Summarise the no-training V17 evaluations and check that they are trustworthy.

For every report:
  * V13 must reproduce the reference report (eval_v16.json) row for row;
  * V17NoAttn (when present) must reproduce V13 row for row -- it is V13's weights with
    the damping switched off, so any difference is a wrapper bug, not a result;
  * each treatment arm is paired with V13 on the same questions (rescued / harmed,
    exact two-sided McNemar), plus its loss on the 47 questions Full answers.

Standard library only, so it runs with any python3.
"""
import argparse
import json
import math
import os
import sys

TREATMENTS = ("V13Density", "V17", "V17NoAttn")


def correct(row, arm):
    entry = (row.get("arms") or {}).get(arm)
    return None if entry is None else int(entry["correct"])


def mcnemar(rescued, harmed):
    n = rescued + harmed
    if n == 0:
        return 1.0
    k = min(rescued, harmed)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def by_id(report):
    return {str(row["id"]): row for row in report["samples"] if row.get("arms")}


def same_predictions(rows_a, arm_a, rows_b, arm_b):
    shared = sorted(set(rows_a) & set(rows_b))
    differing = [key for key in shared
                 if (rows_a[key]["arms"].get(arm_a) or {}).get("prediction")
                 != (rows_b[key]["arms"].get(arm_b) or {}).get("prediction")]
    return dict(compared=len(shared), differing=len(differing), ids=differing[:10])


def summarise(path, reference):
    report = json.load(open(path))
    rows = by_id(report)
    out = dict(report=path, status=report.get("status"), rows=len(rows),
               peak_gib=round(report.get("peak_allocated_bytes", 0) / 2 ** 30, 2))
    out["v13_vs_reference"] = same_predictions(rows, "V13", reference, "V13")
    arms = sorted({arm for row in rows.values() for arm in row["arms"]})
    out["accuracy"] = {arm: sum(correct(r, arm) or 0 for r in rows.values())
                       for arm in arms}
    protect = [key for key, row in rows.items() if correct(row, "Full") == 1]
    out["protect_lost"] = {arm: sum(1 for key in protect if correct(rows[key], arm) == 0)
                           for arm in arms}
    pairs = {}
    for arm in TREATMENTS:
        if arm not in arms:
            continue
        rescued = sum(1 for r in rows.values()
                      if correct(r, arm) == 1 and correct(r, "V13") == 0)
        harmed = sum(1 for r in rows.values()
                     if correct(r, arm) == 0 and correct(r, "V13") == 1)
        pairs[arm] = dict(rescued=rescued, harmed=harmed, net=rescued - harmed,
                          mcnemar_p=round(mcnemar(rescued, harmed), 4))
    out["versus_V13"] = pairs
    if "V17NoAttn" in arms:
        out["v17noattn_equals_v13"] = same_predictions(rows, "V17NoAttn", rows, "V13")
    density = [row.get("density") for row in report["samples"] if row.get("density")]
    if density:
        absorbed = [((d.get("telemetry") or {}).get("absorbed_pairs") or 0) for d in density]
        spread = [((d.get("quota") or {}).get("max") or 0) - ((d.get("quota") or {}).get("min")
                                                               or 0) for d in density]
        out["window"] = dict(quota_mode=density[0].get("quota_mode"),
                             absorb_cos=density[0].get("absorb_cos"),
                             absorbed_pairs_mean=round(sum(absorbed) / len(absorbed), 1),
                             rows_with_no_absorb=sum(1 for a in absorbed if a == 0),
                             quota_spread_mean=round(sum(spread) / len(spread), 1))
    problems = []
    if out["v13_vs_reference"]["differing"]:
        problems.append("V13 does not reproduce the reference")
    if out.get("v17noattn_equals_v13", {}).get("differing"):
        problems.append("V17NoAttn differs from V13: wrapper bug")
    out["trustworthy"] = not problems
    out["problems"] = problems
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--reference", default="reports/cross_block/eval_v16.json")
    p.add_argument("--out", required=True)
    p.add_argument("reports", nargs="+")
    args = p.parse_args(argv)
    reference = by_id(json.load(open(args.reference)))
    results = []
    for path in args.reports:
        if not os.path.exists(path):
            results.append(dict(report=path, status="missing"))
            continue
        try:
            results.append(summarise(path, reference))
        except Exception as error:  # noqa: BLE001 - recorded, the others still run
            results.append(dict(report=path, status="unreadable", error=repr(error)))
    json.dump(results, open(args.out, "w"), indent=1)
    json.dump(results, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
