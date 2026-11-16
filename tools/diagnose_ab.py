#!/usr/bin/env python3
"""Compare two (or more) evaluation reports sample by sample, without re-running.

The question this answers is not "did accuracy move" but "did the fusion move the
model's belief at all": for every sample it reads the four option logits the
evaluator stored (`option_scores`, one per letter) and reports

    P(correct option) = softmax(option logits)[gold letter]

for each arm, plus the rank of the gold letter. A pair of arms with identical
accuracy but different P(correct) means the fusion changed the decision margin
without changing the winner; identical P(correct) on every sample means the
fusion did not reach the readout at all.

Samples are paired by LongBench v2 `id`, never by position: two reports produced
with different `--length` / `--max-samples` / `--sample-ids` orderings would
otherwise be compared question against question. Reports without usable ids
(merged shard reports restart `sample_index` at 0) fall back to `sample_index`
and say so. Reports from different entrypoint revisions are flagged, because a
difference between them is not an arm effect.

Usage:
    python tools/diagnose_ab.py reports/main_table/eval_selection_50b_n20.json \
                               reports/main_table/eval_v10_50b_n20.json
    python tools/diagnose_ab.py a.json b.json --labels Selection V10 --csv out.csv
"""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

LETTERS = ("A", "B", "C", "D")


def softmax_correct(scores, answer):
    """P(gold option) under a softmax over the four stored single-token logits."""
    if not isinstance(scores, dict):
        return None
    values = [scores.get(letter) for letter in LETTERS]
    if any(value is None for value in values):
        return None
    top = max(values)
    exps = [math.exp(value - top) for value in values]
    total = sum(exps)
    gold = scores.get(answer)
    return None if gold is None else math.exp(gold - top) / total


def logit_spread(scores):
    """max-min of the four option logits: how decisive the reader is at all."""
    if not isinstance(scores, dict):
        return None
    values = [scores.get(letter) for letter in LETTERS]
    if any(value is None for value in values):
        return None
    return max(values) - min(values)


def rank_correct(scores, answer):
    if not isinstance(scores, dict) or scores.get(answer) is None:
        return None
    values = {letter: scores.get(letter) for letter in LETTERS if scores.get(letter) is not None}
    if len(values) < len(LETTERS):
        return None
    ordered = sorted(values, key=lambda letter: values[letter], reverse=True)
    return ordered.index(answer) + 1


def load(path):
    """Return (report, samples keyed for pairing, key kind)."""
    report = json.loads(Path(path).read_text())
    samples = report.get("samples") or []
    by_id = {str(s["id"]): s for s in samples if s.get("id") is not None}
    if len(by_id) == len(samples):
        return report, by_id, "id"
    return report, {int(s["sample_index"]): s for s in samples}, "sample_index"


def identity(report):
    proto = report.get("protocol") or {}
    ident = report.get("identity") or {}
    return dict(
        arm=proto.get("merge_arm") or proto.get("arm"),
        geometry=proto.get("geometry"),
        fusion_policy=proto.get("fusion_policy"),
        operator_variant=proto.get("operator_variant"),
        entrypoint=ident.get("entrypoint_sha256"),
        checkpoint=ident.get("checkpoint_sha256"))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", nargs="+", help="two or more result JSONs")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--csv", default=None, help="write the per-sample table here")
    parser.add_argument("--show", type=int, default=0, help="print this many rows (0 = all)")
    args = parser.parse_args()
    if len(args.reports) < 2:
        parser.error("need at least two reports to compare")
    labels = args.labels or [Path(p).stem for p in args.reports]
    if len(labels) != len(args.reports):
        parser.error("--labels must match the number of reports")

    loaded = [load(path) for path in args.reports]
    kinds = {kind for _, _, kind in loaded}
    if len(kinds) > 1:
        print("warning: some reports carry ids and some do not; pairing by "
              "sample_index instead", file=sys.stderr)
        loaded = [(report, {int(s["sample_index"]): s for s in report.get("samples") or []},
                   "sample_index") for report, _, _ in loaded]
        kinds = {"sample_index"}
    join = kinds.pop()
    keys = [set(samples) for _, samples, _ in loaded]
    common = sorted(set.intersection(*keys), key=str)
    if not common:
        parser.error("the reports share no sample id")

    print(f"paired samples: {len(common)}  (pairing key: {join})")
    for label, (report, samples, _) in zip(labels, loaded):
        info = identity(report)
        print(f"  {label:>16}: n={len(samples):<4} arm={info['arm']} "
              f"geometry={info['geometry']} policy={info['fusion_policy']} "
              f"variant={info['operator_variant']} "
              f"entry={str(info['entrypoint'])[:8]}")

    disagreeing = [key for key in common
                   if len({str(samples[key].get("answer")) for _, samples, _ in loaded}) > 1]
    if disagreeing:
        parser.error(f"{len(disagreeing)} paired samples disagree on the gold answer "
                     f"(first: {disagreeing[0]}); these reports are not the same evaluation")
    if len({str(identity(report)["entrypoint"]) for report, _, _ in loaded}) > 1:
        print("warning: the reports were produced by different entrypoint revisions; "
              "differences below may be code effects, not arm effects", file=sys.stderr)

    print()
    header = f"{'arm':>16} {'n':>4} {'correct':>8} {'option':>7} {'cue':>6} " \
             f"{'P(gold)':>8} {'rank1':>6} {'median':>7} {'spread':>7} {'P>=.5':>6}"
    print(header)
    print("-" * len(header))
    per_arm = []
    for label, (report, samples, _) in zip(labels, loaded):
        rows = [samples[key] for key in common]
        n = len(rows)
        correct = sum(int(bool(row.get("correct"))) for row in rows)
        option = sum(int(bool(row.get("option_correct"))) for row in rows)
        cue = sum(int(bool(row.get("cue_correct"))) for row in rows)
        probs = [p for p in (softmax_correct(row.get("option_scores"), row.get("answer"))
                             for row in rows) if p is not None]
        ranks = [r for r in (rank_correct(row.get("option_scores"), row.get("answer"))
                             for row in rows) if r is not None]
        spreads = [value for value in (logit_spread(row.get("option_scores")) for row in rows)
                   if value is not None]
        decisive = sum(1 for value in probs if value >= 0.5)
        per_arm.append(dict(label=label, rows=rows, probs=probs))
        mean_p = statistics.fmean(probs) if probs else float("nan")
        median_p = statistics.median(probs) if probs else float("nan")
        mean_spread = statistics.fmean(spreads) if spreads else float("nan")
        print(f"{label:>16} {n:>4} {correct:>4}/{n:<3} {option:>3}/{n:<3} {cue:>3}/{n:<3} "
              f"{mean_p:>8.3f} {sum(1 for r in ranks if r == 1):>3}/{len(ranks):<2} "
              f"{median_p:>7.3f} {mean_spread:>7.2f} {decisive:>3}/{len(probs):<2}")

    base = per_arm[0]
    print()
    for arm in per_arm[1:]:
        moved = changed_letter = changed_correct = up = down = 0
        for a, b in zip(base["rows"], arm["rows"]):
            pa = softmax_correct(a.get("option_scores"), a.get("answer"))
            pb = softmax_correct(b.get("option_scores"), b.get("answer"))
            delta = None if pa is None or pb is None else pb - pa
            if a.get("option_letter") != b.get("option_letter"):
                changed_letter += 1
            if int(bool(a.get("correct"))) != int(bool(b.get("correct"))):
                changed_correct += 1
            if delta is not None and abs(delta) > 1e-9:
                moved += 1
                up += delta > 0
                down += delta < 0
        print(f"{base['label']} vs {arm['label']}: "
              f"P(gold) moved on {moved}/{len(common)} samples (up {up}, down {down}); "
              f"predicted letter differs on {changed_letter}; "
              f"correctness differs on {changed_correct}")

    if args.csv:
        fieldnames = ["sample", "id", "length", "answer"]
        for arm in per_arm:
            fieldnames += [f"letter_{arm['label']}", f"correct_{arm['label']}",
                           f"p_{arm['label']}", f"rank_{arm['label']}"]
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for position, key in enumerate(common):
                head = per_arm[0]["rows"][position]
                record = dict(sample=key, id=head.get("id"), length=head.get("length"),
                              answer=head.get("answer"))
                for arm in per_arm:
                    row, tag = arm["rows"][position], arm["label"]
                    p = softmax_correct(row.get("option_scores"), row.get("answer"))
                    record[f"letter_{tag}"] = row.get("option_letter")
                    record[f"correct_{tag}"] = int(bool(row.get("correct")))
                    record[f"p_{tag}"] = None if p is None else round(p, 4)
                    record[f"rank_{tag}"] = rank_correct(row.get("option_scores"),
                                                         row.get("answer"))
                writer.writerow(record)
        print(f"per-sample table -> {args.csv}")

    limit = args.show if args.show else len(common)
    print()
    head = f"{'sample':>8} {'len':>7} {'gold':>4} " + " ".join(
        f"{arm['label'][:10]:>12}" for arm in per_arm)
    print(head)
    for position, key in enumerate(common[:limit]):
        base_row = per_arm[0]["rows"][position]
        cells = []
        for arm in per_arm:
            row = arm["rows"][position]
            p = softmax_correct(row.get("option_scores"), row.get("answer"))
            cells.append(f"{str(row.get('option_letter'))}/"
                         f"{'-' if p is None else format(p, '.2f')}")
        print(f"{str(key)[-8:]:>8} {str(base_row.get('length')):>7} "
              f"{str(base_row.get('answer')):>4} " + " ".join(f"{cell:>12}" for cell in cells))


if __name__ == "__main__":
    main()
