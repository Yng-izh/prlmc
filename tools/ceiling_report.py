#!/usr/bin/env python3
"""Ceiling report: how much headroom does compression actually leave?

A merge operator can only recover information that compression destroyed, so its
effect is bounded by the no-compression run (the "ceiling"). Comparing a merge
run against its same-budget selector alone cannot distinguish "the operator is
weak" from "there was nothing to recover".

The no-compression run is a *reference*, not a monotone upper bound: on hotpotqa
it wins 9, loses 11 and ties 76 against the same-budget selector, because a
28-block context also carries more distractors. So this report never rejects a
treatment for exceeding the headroom; it reports three complementary indicators
for every arm and the paired deltas between them:

    mean_f1               continuous F1 average
    nonzero_rate          share of samples with F1 > 0 (did it find the entity
                          at all, as opposed to answering something unrelated)
    high_precision_rate   share of samples with F1 >= --precision-threshold
                          (0.80 by default: the answer is complete enough)
    exact_match_rate      share of samples whose truncated answer equals a gold
                          answer after the LongBench normalisation

Plus the paired bootstrap CI of `ceiling - selection` (headroom) and
`treatment - selection` (effect) for each indicator, and the sample count needed
to detect --target-effect at 80% power.

Usage:
  python tools/ceiling_report.py --selection sel.json --ceiling full.json \
      [--treatment merge.json] [--target-effect 0.017] [--precision-threshold 0.8] \
      [--allow-partial] [--out r.json]

Inputs are run JSONs from tools/longbench_prlmc_eval.py or
tools/longbench_memoryllm_eval.py. Partial runs are rejected unless
--allow-partial is given, in which case the report is stamped partial=true.
"""
import argparse
import json
import math
import random
import re
import string
from pathlib import Path

import numpy as np

ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalise(text):
    """LongBench / SQuAD-style answer normalisation used for exact match."""
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = ARTICLES.sub(" ", text)
    return " ".join(text.split())


def truncated_prediction(sample):
    """The answer the scorer actually reads: everything before the first newline."""
    return (sample.get("prediction") or "").split("\n")[0]


def is_exact(sample):
    gold = [normalise(a) for a in (sample.get("answers") or [])]
    guess = normalise(truncated_prediction(sample))
    return bool(guess) and guess in gold


def load_run(path, allow_partial):
    data = json.loads(Path(path).read_text())
    status = data.get("status")
    if status != "complete" and not allow_partial:
        raise ValueError(f"{path}: run is not complete (status={status}); "
                         "pass --allow-partial to report on a partial run")
    samples = data.get("samples") or []
    if not samples:
        raise ValueError(f"{path}: run has no samples")
    scores, exact = {}, {}
    for sample in samples:
        key = (sample["task"], sample["id"], sample["sample_index"])
        scores[key] = float(sample["official_qa_f1_truncated"])
        exact[key] = 1.0 if is_exact(sample) else 0.0
    return scores, dict(status=status, partial=status != "complete",
                        protocol=(data.get("protocol") or {}),
                        identity=(data.get("identity") or {}),
                        exact_match=exact)


def bootstrap_ci(deltas, resamples=2000, seed=0, level=0.95):
    rng = random.Random(seed)
    deltas = list(deltas)
    n = len(deltas)
    means = []
    for _ in range(resamples):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    alpha = (1.0 - level) / 2.0
    low = means[max(0, int(math.floor(alpha * resamples)))]
    high = means[min(resamples - 1, int(math.ceil((1.0 - alpha) * resamples)) - 1)]
    return dict(n=n, mean=sum(deltas) / n, low=low, high=high,
                resamples=resamples, seed=seed)


def required_n(sd, effect, power=0.8):
    """Two-sided 0.05 test with the normal approximation."""
    if effect <= 0 or sd <= 0:
        return None
    z_alpha, z_beta = 1.959963985, 0.8416212336
    return int(math.ceil(((z_alpha + z_beta) * sd / effect) ** 2))


def paired(a, b, keys):
    return [a[k] - b[k] for k in keys]


def summarise(deltas, label):
    ci = bootstrap_ci(deltas)
    sd = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
    ci["sd"] = sd
    ci["label"] = label
    ci["wins"] = sum(1 for d in deltas if d > 0)
    ci["losses"] = sum(1 for d in deltas if d < 0)
    ci["ties"] = sum(1 for d in deltas if d == 0)
    return ci


def indicators(scores, keys, exact=None, threshold=0.80):
    """The three (four) complementary indicators for one arm."""
    n = len(keys)
    f1 = [scores[k] for k in keys]
    out = dict(n=n,
               mean_f1=sum(f1) / n,
               nonzero_rate=sum(1 for v in f1 if v > 0) / n,
               high_precision_rate=sum(1 for v in f1 if v >= threshold) / n,
               high_precision_threshold=threshold)
    if exact is not None:
        out["exact_match_rate"] = sum(exact[k] for k in keys) / n
    return out


def indicator_deltas(a, b, keys, exact_a=None, exact_b=None, threshold=0.80):
    """Paired CI for every indicator, so no single metric carries the verdict."""
    out = dict(mean_f1=summarise(paired(a, b, keys), "mean_f1"))
    for name, test in (
            ("nonzero_rate", lambda value, key: 1.0 if value > 0 else 0.0),
            ("high_precision_rate", lambda value, key: 1.0 if value >= threshold else 0.0)):
        out[name] = summarise([test(a[k], k) - test(b[k], k) for k in keys], name)
    if exact_a is not None and exact_b is not None:
        out["exact_match_rate"] = summarise(
            [exact_a[k] - exact_b[k] for k in keys], "exact_match_rate")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selection", required=True, help="same-budget selector run")
    p.add_argument("--ceiling", required=True,
                   help="no-compression reference run (same entrypoint, --arm Full)")
    p.add_argument("--treatment", default=None, help="merge run to compare")
    p.add_argument("--target-effect", type=float, default=0.017,
                   help="effect size to power for (default 0.017 = v5 hotpotqa claim)")
    p.add_argument("--precision-threshold", type=float, default=0.80,
                   help="F1 at or above which an answer counts as complete (default 0.80)")
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    sel, sel_meta = load_run(args.selection, args.allow_partial)
    ceil, ceil_meta = load_run(args.ceiling, args.allow_partial)
    treat = treat_meta = None
    if args.treatment:
        treat, treat_meta = load_run(args.treatment, args.allow_partial)

    keys = sorted(set(sel) & set(ceil) & (set(treat) if treat else set(sel)))
    if not keys:
        raise SystemExit("no shared samples between the given runs")
    tasks = sorted({k[0] for k in keys})
    threshold = args.precision_threshold

    sel_exact = sel_meta.get("exact_match")
    ceil_exact = ceil_meta.get("exact_match")
    treat_exact = (treat_meta or {}).get("exact_match")

    headroom = summarise(paired(ceil, sel, keys), "ceiling - selection")
    metrics = dict(
        selection=indicators(sel, keys, sel_exact, threshold),
        ceiling=indicators(ceil, keys, ceil_exact, threshold),
    )
    deltas = dict(headroom=indicator_deltas(ceil, sel, keys, ceil_exact, sel_exact, threshold))
    report = dict(
        schema="prlmc_ceiling_report/2",
        selection=args.selection, ceiling=args.ceiling, treatment=args.treatment,
        partial=any(m["partial"] for m in (sel_meta, ceil_meta)),
        paired_samples=len(keys), tasks=tasks,
        mean_selection=sum(sel[k] for k in keys) / len(keys),
        mean_ceiling=sum(ceil[k] for k in keys) / len(keys),
        metrics=metrics,
        deltas=deltas,
        headroom=headroom,
        required_n=dict(target_effect=args.target_effect, power=0.8,
                        n=required_n(headroom["sd"], args.target_effect)),
    )
    notes = []
    if treat is not None:
        report["mean_treatment"] = sum(treat[k] for k in keys) / len(keys)
        report["effect"] = summarise(paired(treat, sel, keys), "treatment - selection")
        metrics["treatment"] = indicators(treat, keys, treat_exact, threshold)
        deltas["effect"] = indicator_deltas(treat, sel, keys, treat_exact, sel_exact, threshold)
        report["verdict"] = verdict(report)
        if report["effect"]["low"] > report["headroom"]["high"]:
            notes.append(
                "effect_above_headroom_mean: the treatment's mean gain exceeds the "
                "no-compression reference's mean gain. This is reported, not "
                "rejected: the reference is not a monotone upper bound "
                f"(ceiling wins/losses/ties {headroom['wins']}/{headroom['losses']}/"
                f"{headroom['ties']} per sample), so compression helps some "
                "questions and hurts others.")
    report["notes"] = notes

    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


def verdict(report):
    """One-line reading of the paired mean-F1 comparison (indicators carry the detail)."""
    head = report["headroom"]
    eff = report.get("effect")
    if head["low"] <= 0:
        return ("headroom_undetected: the no-compression reference is not "
                "demonstrably better than the same-budget selector at this sample "
                f"size (headroom {head['mean']:+.4f}, CI [{head['low']:+.4f}, "
                f"{head['high']:+.4f}] covers 0), so neither a positive nor a zero "
                "headroom is established; check nonzero_rate/high_precision_rate and "
                "raise n before concluding")
    if eff is None:
        return "headroom_measured"
    if eff["low"] > 0 and eff["mean"] > 0:
        return "positive_within_headroom"
    return "no_effect_detected"


if __name__ == "__main__":
    main()
