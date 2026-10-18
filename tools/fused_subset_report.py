#!/usr/bin/env python3
"""Fused-subset report: compare a run against a baseline on the fused subset.

Input: two or more run JSONs produced by tools/longbench_prlmc_eval.py (schema
"longbench_prlmc_results/1"). The first file is the treatment run; additional
files are baselines. The treatment run defines the fused subset: every sample
with compression_count > 0. Metrics per comparison:

  - fused mean / full mean of official_qa_f1_truncated (both runs)
  - paired wins/losses/ties on the fused subset (treatment > baseline counts
    as a win; ties broken as neither)
  - bootstrap 95% CI (resamples=2000, seed=0) of the paired fused-subset mean
    delta (treatment - baseline)

Usage:
  python tools/fused_subset_report.py --treatment run_t.json \
      --baseline run_b.json [--baseline run_b2.json ...] [--out report.json]
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np


def load_samples(path):
    data = json.loads(Path(path).read_text())
    if data.get("schema") != "longbench_prlmc_results/1":
        raise ValueError(f"{path}: expected schema longbench_prlmc_results/1")
    if data.get("status") != "complete":
        raise ValueError(f"{path}: run is not complete (status={data.get('status')})")
    samples = data["samples"]
    if not samples:
        raise ValueError(f"{path}: run has no samples")
    return samples


def key(sample):
    return sample["task"], sample["id"], sample["sample_index"]


def score(sample):
    return float(sample["official_qa_f1_truncated"])


def bootstrap_ci(deltas, resamples=2000, seed=0, level=0.95):
    rng = random.Random(seed)
    deltas = list(deltas)
    if not deltas:
        return dict(n=0, low=None, high=None, resamples=resamples, seed=seed)
    n = len(deltas)
    means = []
    for _ in range(resamples):
        draw = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(draw) / n)
    means.sort()
    alpha = (1.0 - level) / 2.0
    low = means[max(0, int(np.floor(alpha * resamples)))]
    high = means[min(resamples - 1, int(np.ceil((1.0 - alpha) * resamples)) - 1)]
    return dict(n=n, low=low, high=high, resamples=resamples, seed=seed, level=level)


def fused_indices(samples):
    """Fused subset in original order: compression_count > 0."""
    return [s for s in samples if int(s.get("compression_count", 0)) > 0]


def paired_report(treatment, baseline, subset):
    t_map = {key(s): s for s in treatment}
    b_map = {key(s): s for s in baseline}
    common = [k for k in (key(s) for s in subset) if k in t_map and k in b_map]
    deltas, wins, losses, ties = [], 0, 0, 0
    for k in common:
        d = score(t_map[k]) - score(b_map[k])
        deltas.append(d)
        if d > 0:
            wins += 1
        elif d < 0:
            losses += 1
        else:
            ties += 1
    fused_delta_mean = sum(deltas) / len(deltas) if deltas else None
    return dict(
        paired_n=len(common),
        wins=wins, losses=losses, ties=ties,
        fused_delta_mean=fused_delta_mean,
        bootstrap_ci=bootstrap_ci(deltas))


def summarize(samples):
    scores = [score(s) for s in samples]
    return dict(n=len(samples),
                mean=sum(scores) / len(scores) if scores else None)


def build_report(treatment_path, baseline_paths):
    treatment = load_samples(treatment_path)
    subset = fused_indices(treatment)
    if not subset:
        raise ValueError(
            f"{treatment_path}: no fused samples (compression_count > 0); "
            "the run never compressed, so no fused subset exists")
    report = dict(
        schema="fused_subset_report/1",
        treatment=dict(path=treatment_path,
                       full=summarize(treatment),
                       fused=summarize(subset),
                       compression_counts=[int(s["compression_count"]) for s in subset]),
        baselines=[])
    for path in baseline_paths:
        baseline = load_samples(path)
        report["baselines"].append(dict(
            path=path,
            full=summarize(baseline),
            fused=summarize(fused_indices(baseline)),
            paired=paired_report(treatment, baseline, subset)))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--treatment", required=True, help="treatment run JSON (defines the fused subset)")
    p.add_argument("--baseline", action="append", required=True,
                   help="baseline run JSON; repeatable")
    p.add_argument("--out", default=None, help="write the report JSON here (also printed)")
    args = p.parse_args()
    report = build_report(args.treatment, args.baseline)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
