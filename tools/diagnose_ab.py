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

Usage:
    python tools/diagnose_ab.py reports/main_table/eval_selection_50b_n20.json \
                               reports/main_table/eval_v10_50b_n20.json
    python tools/diagnose_ab.py a.json b.json --labels Selection V10 --csv out.csv
"""
import argparse
import csv
import json
import math
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
    report = json.loads(Path(path).read_text())
    samples = {int(s["sample_index"]): s for s in report.get("samples", [])}
    return report, samples


def letter_of(sample, key="option_letter"):
    return sample.get(key)


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
    keys = [set(samples) for _, samples in loaded]
    common = sorted(set.intersection(*keys))
    if not common:
        parser.error("the reports share no sample_index")

    print(f"samples in every report: {len(common)}")
    print()
    header = f"{'arm':>16} {'n':>4} {'correct':>8} {'option':>7} {'cue':>6} " \
             f"{'P(gold)':>8} {'rank1':>6} {'median':>7} {'spread':>7} {'P>=.5':>6}"
    print(header)
    print("-" * len(header))
    per_arm = []
    for label, (report, samples) in zip(labels, loaded):
        rows = [samples[i] for i in common]
        n = len(rows)
        correct = sum(int(bool(row.get("correct"))) for row in rows)
        option = sum(int(bool(row.get("option_correct"))) for row in rows)
        cue = sum(int(bool(row.get("cue_correct"))) for row in rows)
        probs = [softmax_correct(row.get("option_scores"), row.get("answer")) for row in rows]
        probs = [p for p in probs if p is not None]
        ranks = [rank_correct(row.get("option_scores"), row.get("answer")) for row in rows]
        ranks = [r for r in ranks if r is not None]
        median = sorted(probs)[len(probs) // 2] if probs else float("nan")
        spreads = [logit_spread(row.get("option_scores")) for row in rows]
        spreads = [value for value in spreads if value is not None]
        decisive = sum(1 for value in probs if value >= 0.5)
        per_arm.append(dict(label=label, rows=rows, probs=probs))
        print(f"{label:>16} {n:>4} {correct:>4}/{n:<3} {option:>3}/{n:<3} {cue:>3}/{n:<3} "
              f"{(sum(probs) / len(probs) if probs else float('nan')):>8.3f} "
              f"{sum(1 for r in ranks if r == 1):>3}/{len(ranks):<2} {median:>7.3f} "
              f"{(sum(spreads) / len(spreads) if spreads else float('nan')):>7.2f} "
              f"{decisive:>3}/{n:<2}")

    base = per_arm[0]
    print()
    for arm in per_arm[1:]:
        moved = changed_letter = changed_correct = 0
        up = down = 0
        rows_out = []
        for index in common:
            a, b = base["rows"][common.index(index)], arm["rows"][common.index(index)]
            pa = softmax_correct(a.get("option_scores"), a.get("answer"))
            pb = softmax_correct(b.get("option_scores"), b.get("answer"))
            delta = None if pa is None or pb is None else pb - pa
            if letter_of(a) != letter_of(b):
                changed_letter += 1
            if int(bool(a.get("correct"))) != int(bool(b.get("correct"))):
                changed_correct += 1
            if delta is not None and abs(delta) > 1e-9:
                moved += 1
                up += delta > 0
                down += delta < 0
            rows_out.append(dict(
                sample=index, id=a.get("id"), length=a.get("length"),
                answer=a.get("answer"),
                letter_a=letter_of(a), letter_b=letter_of(b),
                correct_a=int(bool(a.get("correct"))), correct_b=int(bool(b.get("correct"))),
                p_a=None if pa is None else round(pa, 4),
                p_b=None if pb is None else round(pb, 4),
                delta=None if delta is None else round(delta, 4),
                compressed_a=a.get("compression_count"), compressed_b=b.get("compression_count")))
        print(f"{base['label']} vs {arm['label']}: "
              f"P(gold) moved on {moved}/{len(common)} samples "
              f"(up {up}, down {down}); "
              f"predicted letter differs on {changed_letter}; "
              f"correctness differs on {changed_correct}")
        if args.csv:
            with open(args.csv, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
                writer.writeheader()
                writer.writerows(rows_out)
            print(f"per-sample table -> {args.csv}")
        limit = args.show if args.show else len(rows_out)
        print()
        print(f"{'idx':>4} {'len':>7} {'gold':>4} {'A':>2} {'B':>2} {'corr':>4} "
              f"{'P_A':>6} {'P_B':>6} {'dP':>7}")
        for row in rows_out[:limit]:
            p_a = "-" if row["p_a"] is None else format(row["p_a"], ".3f")
            p_b = "-" if row["p_b"] is None else format(row["p_b"], ".3f")
            delta = "-" if row["delta"] is None else format(row["delta"], "+.3f")
            print(f"{row['sample']:>4} {str(row['length']):>7} {str(row['answer']):>4} "
                  f"{str(row['letter_a']):>2} {str(row['letter_b']):>2} "
                  f"{row['correct_a']}/{row['correct_b']:<2} "
                  f"{p_a:>6} {p_b:>6} {delta:>7}")


if __name__ == "__main__":
    main()
