"""Exact support oracle for native 19->9 block selection.

This is a gold-support upper bound, not a reader-CE oracle: it uses the known
answer-support block of development questions and must never be used at test
or deployment time.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path


def save(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--row", type=int, default=None)
    args = ap.parse_args()
    plan_path, source_path = Path(args.plan), Path(args.source)
    plan_bytes, source_bytes = plan_path.read_bytes(), source_path.read_bytes()
    plan, source = json.loads(plan_bytes), json.loads(source_bytes)
    plan_hash = hashlib.sha256(plan_bytes).hexdigest()
    if source.get("plan_sha256") != plan_hash:
        raise ValueError("source does not match plan")
    rows = [p for p in plan["plans"] if args.row is None or p["row"] == args.row]
    if not rows:
        raise ValueError("no requested row")
    records = {(r["row"], r["question"]): r for r in source["records"]}
    report = dict(experiment="SIC-support-oracle", stage="develop",
                  plan_sha256=plan_hash,
                  source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                  objective="maximize eligible development questions whose final support block is retained",
                  total_combos=92378, rows=[])
    for p in rows:
        row = p["row"]
        eligible = [x for x in p["development"]
                    if records[(row, x["question"])]["eligible"]]
        support = {x["question"]: (int(x["line"])-60)//20 for x in eligible}
        select = next(w for w in source["windows"] if w["row"] == row)["kept"]["select"]
        best = None
        for blocks in itertools.combinations(range(19), 9):
            kept = set(blocks)
            retained = sum(support[q] in kept for q in support)
            # Minimize native block changes only after maximizing support.
            overlap = sum(b in select for b in blocks)
            key = (retained, overlap, tuple(-b for b in blocks))
            if best is None or key > best["key"]:
                best = dict(key=key, kept=list(blocks), retained=retained,
                            lost=len(eligible)-retained,
                            eligible=len(eligible),
                            support_counts={q: support[q] for q in support})
        select_retained = sum(support[q] in set(select) for q in support)
        best["select"] = dict(kept=select, retained=select_retained,
                              lost=len(eligible)-select_retained,
                              eligible=len(eligible))
        best.pop("key")
        report["rows"].append(dict(row=row, **best))
    report["oracle_exact"] = True
    report["notes"] = [
        "Uses development gold support block, so it is an upper bound/diagnostic only.",
        "It does not measure reader CE, block redundancy, or answer formatting.",
        "Confirmation questions are not read or used.",
    ]
    save(args.out, report)
    print(json.dumps({"rows": [(x["row"], x["eligible"], x["select"]["lost"], x["lost"], x["kept"]) for x in report["rows"]]}, indent=2))


if __name__ == "__main__":
    main()
