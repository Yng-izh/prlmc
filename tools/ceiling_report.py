#!/usr/bin/env python3
"""Ceiling report: how much headroom does compression actually leave?

A merge operator can only recover information that compression destroyed, so its
effect is bounded by the no-compression run (the "ceiling"). Comparing a merge
run against its same-budget selector alone cannot distinguish "the operator is
weak" from "there was nothing to recover". This tool reports both:

  - headroom = mean(ceiling) - mean(selection) with a paired bootstrap CI
  - effect   = mean(treatment) - mean(selection) with a paired bootstrap CI
  - required_n: samples needed to detect --target-effect at 80% power, using the
    observed per-sample paired standard deviation

A treatment whose CI lies inside a headroom CI that covers zero has not been
shown to recover anything; a treatment claiming an effect larger than the
headroom is claiming the impossible.

Usage:
  python tools/ceiling_report.py --selection sel.json --ceiling full.json \
      [--treatment merge.json] [--target-effect 0.017] [--allow-partial] [--out r.json]

Inputs are run JSONs from tools/longbench_prlmc_eval.py or
tools/longbench_memoryllm_eval.py. Partial runs are rejected unless
--allow-partial is given, in which case the report is stamped partial=true.
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np


def load_run(path, allow_partial):
    data = json.loads(Path(path).read_text())
    status = data.get("status")
    if status != "complete" and not allow_partial:
        raise ValueError(f"{path}: run is not complete (status={status}); "
                         "pass --allow-partial to report on a partial run")
    samples = data.get("samples") or []
    if not samples:
        raise ValueError(f"{path}: run has no samples")
    out = {}
    for sample in samples:
        key = (sample["task"], sample["id"], sample["sample_index"])
        out[key] = float(sample["official_qa_f1_truncated"])
    return out, dict(status=status, partial=status != "complete",
                     protocol=(data.get("protocol") or {}),
                     identity=(data.get("identity") or {}))


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selection", required=True, help="same-budget selector run")
    p.add_argument("--ceiling", required=True, help="no-compression run (upper bound)")
    p.add_argument("--treatment", default=None, help="merge run to judge against headroom")
    p.add_argument("--target-effect", type=float, default=0.017,
                   help="effect size to power for (default 0.017 = v5 hotpotqa claim)")
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    sel, sel_meta = load_run(args.selection, args.allow_partial)
    ceil, ceil_meta = load_run(args.ceiling, args.allow_partial)
    treat = None
    if args.treatment:
        treat, treat_meta = load_run(args.treatment, args.allow_partial)

    keys = sorted(set(sel) & set(ceil) & (set(treat) if treat else set(sel)))
    if not keys:
        raise SystemExit("no shared samples between the given runs")
    tasks = sorted({k[0] for k in keys})

    headroom = summarise(paired(ceil, sel, keys), "ceiling - selection")
    report = dict(
        schema="prlmc_ceiling_report/1",
        selection=args.selection, ceiling=args.ceiling, treatment=args.treatment,
        partial=any(m["partial"] for m in (sel_meta, ceil_meta)),
        paired_samples=len(keys), tasks=tasks,
        mean_selection=sum(sel[k] for k in keys) / len(keys),
        mean_ceiling=sum(ceil[k] for k in keys) / len(keys),
        headroom=headroom,
        required_n=dict(target_effect=args.target_effect,
                        power=0.8,
                        n=required_n(headroom["sd"], args.target_effect)),
    )
    if treat is not None:
        report["mean_treatment"] = sum(treat[k] for k in keys) / len(keys)
        report["effect"] = summarise(paired(treat, sel, keys), "treatment - selection")
        report["verdict"] = verdict(report)

    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


def verdict(report):
    head = report["headroom"]
    eff = report["effect"]
    if head["low"] <= 0:
        return ("headroom_absent: the no-compression run is not demonstrably better "
                "than the same-budget selector (headroom CI covers 0), so a merge "
                "operator has no recoverable information to exploit on this task set")
    if eff["low"] > head["high"]:
        return ("effect_exceeds_headroom: the reported treatment effect is larger "
                "than the measured headroom, which cannot be a real recovery effect")
    if eff["low"] > 0 and eff["mean"] > 0:
        return "positive_within_headroom"
    return "no_effect_detected"


if __name__ == "__main__":
    main()
