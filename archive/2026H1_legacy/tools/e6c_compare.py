#!/usr/bin/env python3
"""E6c step 3: per-question comparison of the three benchmark arms.

Reads the three `*_results.json` files produced by `run_e6c_bench.sh` and joins
them on `qa_pair_id`, which is unique in the factconsolidation result files (100/100
in the E6 run). `query_id` is a valid alternate key.

Answers only one question: under the same tightened budget, does the trained
carrier-residual compactor get more questions right than the two baselines that
do not use it?

    carrier vs selection:  rescued = carrier correct, selection wrong
                           harmed  = carrier wrong,  selection correct
    carrier vs eviction:   rescued = carrier correct, eviction wrong
                           harmed  = carrier wrong,  eviction correct

The direction is only credible if both balances are positive. A single arm's
accuracy is not enough: at n=100 the standard error on an accuracy is about 4.5
points, so a two-point lead is noise. The paired counts are the honest reading,
because every question is the same question for all three arms.

Usage (on the host, from the repo root):

    python tools/e6c_compare.py \
        --evict   outputs/prlmc-e6c-evict/Conflict_Resolution/*_results.json \
        --select  outputs/prlmc-e6c-select/Conflict_Resolution/*_results.json \
        --carrier outputs/prlmc-e6c-carrier/Conflict_Resolution/*_results.json

With no arguments it globs the three default output directories.
"""

import argparse
import glob
import json
import os
import sys

RUNS = ("evict", "select", "carrier")
METRIC = "exact_match"
ALT_METRIC = "substring_exact_match"


def load(path):
    with open(path) as handle:
        payload = json.load(handle)
    rows = payload["data"]
    out = {}
    for row in rows:
        key = row.get("qa_pair_id")
        if key is None:
            key = row.get("query_id")
        if key is None:
            raise SystemExit("row without qa_pair_id or query_id in %s" % path)
        out[str(key)] = row
    averaged = payload.get("averaged_metrics", {})
    return out, averaged, payload.get("agent_config", {})


def resolve(pattern):
    if pattern:
        hits = sorted(glob.glob(pattern))
    else:
        hits = []
    return hits


def default_patterns():
    root = "outputs"
    return {
        run: os.path.join(root, "prlmc-e6c-%s" % run, "Conflict_Resolution",
                          "*_results.json")
        for run in RUNS
    }


def hit(rows, key, metric):
    row = rows.get(key)
    if row is None:
        return None
    value = row.get(metric)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) >= 0.5
    return bool(value)


def paired(rows_a, rows_b, a_name, b_name, metric):
    """A = the arm under test (carrier), B = the baseline."""
    shared = sorted(set(rows_a) & set(rows_b))
    rescued, harmed, both, neither = [], [], [], []
    for key in shared:
        a = hit(rows_a, key, metric)
        b = hit(rows_b, key, metric)
        if a is None or b is None:
            continue
        if a and not b:
            rescued.append(key)
        elif b and not a:
            harmed.append(key)
        elif a and b:
            both.append(key)
        else:
            neither.append(key)
    return {
        "under_test": a_name,
        "baseline": b_name,
        "n_shared": len(shared),
        "rescued": rescued,
        "harmed": harmed,
        "both": len(both),
        "neither": len(neither),
        "delta": len(rescued) - len(harmed),
    }


def accuracy(rows, metric):
    scored = [v for v in (hit(rows, k, metric) for k in rows) if v is not None]
    if not scored:
        return float("nan"), 0
    return sum(1 for v in scored if v) / len(scored), len(scored)


def main():
    parser = argparse.ArgumentParser()
    defaults = default_patterns()
    for run in RUNS:
        parser.add_argument("--%s" % run, default=defaults[run])
    parser.add_argument("--metric", default=METRIC)
    parser.add_argument("--show-questions", action="store_true",
                        help="print the question text for every rescued/harmed id")
    args = parser.parse_args()

    loaded = {}
    for run in RUNS:
        paths = resolve(getattr(args, run))
        if len(paths) != 1:
            raise SystemExit(
                "[e6c] --%s matched %d files (%s); expected exactly one"
                % (run, len(paths), getattr(args, run))
            )
        rows, averaged, _ = load(paths[0])
        loaded[run] = rows
        acc, n = accuracy(rows, args.metric)
        print("[e6c] %-8s %-22s %.4f  (n=%d)  file %s"
              % (run, args.metric, acc, n, paths[0]))

    selection = paired(loaded["carrier"], loaded["select"], "carrier", "select",
                       args.metric)
    eviction = paired(loaded["carrier"], loaded["evict"], "carrier", "evict",
                      args.metric)

    for block in (selection, eviction):
        print("[e6c] carrier vs %-8s rescued %d  harmed %d  delta %+d  "
              "(both %d, neither %d, shared %d)"
              % (block["baseline"], len(block["rescued"]), len(block["harmed"]),
                 block["delta"], block["both"], block["neither"], block["n_shared"]))

    ok_select = selection["delta"] > 0
    ok_evict = eviction["delta"] > 0
    if ok_select and ok_evict:
        direction = "carrier > selection and carrier > eviction"
    elif not ok_select and not ok_evict:
        direction = "no direction: carrier does not lead either baseline"
    else:
        direction = ("mixed: carrier leads %s but not %s"
                     % ("selection" if ok_select else "eviction",
                        "eviction" if ok_select else "selection"))
    print("[e6c] direction: %s" % direction)

    if args.show_questions:
        for block in (selection, eviction):
            for kind in ("rescued", "harmed"):
                for key in block[kind]:
                    question = loaded["carrier"].get(key, {}).get("query", "")
                    print("[e6c]   %s vs %s %s %s :: %s"
                          % (block["under_test"], block["baseline"], kind, key,
                             " ".join(str(question).split())[:160]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
