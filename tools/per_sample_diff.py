#!/usr/bin/env python3
"""Per-question diff across a Selection / Full(ceiling) / treatment run triple.

`tools/ceiling_report.py` answers "is there headroom at all" with a paired CI.
This tool answers the complementary question the CI hides: *which questions* the
no-compression run gets right that the same-budget selector gets wrong, and which
of those the merge operator actually recovers or damages.

Inputs are run JSONs from tools/longbench_prlmc_eval.py (or
tools/longbench_memoryllm_eval.py); samples are paired on
(task, id, sample_index). Correctness is `official_qa_f1_truncated > 0`, the same
rule tools/public_retention_eval.py uses for its gained/harmed/lost/rescued sets.

Categories per paired sample (Selection = same-budget control):

  both_correct / both_wrong          selector and ceiling agree
  lost_to_compression                ceiling correct, selection wrong -> headroom
  compression_helped                 selection correct, ceiling wrong -> noise
  rescued                            lost_to_compression and treatment correct
  missed                             lost_to_compression and treatment wrong
  gained                             selection wrong, treatment correct
  damaged                            selection correct, treatment wrong

Usage:
  python tools/per_sample_diff.py --selection sel.json --ceiling full.json \
      [--treatment merge.json] [--data-dir DIR] [--out r.json] [--allow-partial]

--data-dir is optional: when given, the question text is looked up from
<data-dir>/<task>.jsonl and printed next to the answers.
"""
import argparse
import json
from pathlib import Path


CATEGORIES = ("both_correct", "both_wrong", "lost_to_compression",
              "compression_helped")


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
        out[key] = sample
    return out, dict(path=str(path), status=status, partial=status != "complete",
                     arm=(data.get("identity") or {}).get("arm"),
                     protocol=(data.get("protocol") or {}))


def correct(sample, threshold=0.0):
    return float(sample["official_qa_f1_truncated"]) > threshold


def question_text(data_dir, task, sample_index):
    if not data_dir:
        return None
    path = Path(data_dir) / (task + ".jsonl")
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if sample_index >= len(rows):
        return None
    return rows[sample_index].get("input")


def classify(selection, ceiling, treatment, threshold=0.0):
    """Category plus the three correctness flags for one paired sample."""
    sel, ceil = correct(selection, threshold), correct(ceiling, threshold)
    if sel and ceil:
        category = "both_correct"
    elif not sel and not ceil:
        category = "both_wrong"
    elif ceil:
        category = "lost_to_compression"
    else:
        category = "compression_helped"
    treat = None if treatment is None else correct(treatment, threshold)
    if treat is not None:
        if category == "lost_to_compression":
            category = "rescued" if treat else "missed"
        elif not sel and treat:
            category = "gained"
        elif sel and not treat:
            category = "damaged"
    return category, dict(selection=sel, ceiling=ceil, treatment=treat)


def build_report(args):
    sel, sel_meta = load_run(args.selection, args.allow_partial)
    ceil, ceil_meta = load_run(args.ceiling, args.allow_partial)
    treat, treat_meta = (None, None)
    if args.treatment:
        treat, treat_meta = load_run(args.treatment, args.allow_partial)
    keys = sorted(set(sel) & set(ceil) & (set(treat) if treat else set(sel)))
    if not keys:
        raise SystemExit("no shared samples between the given runs")

    rows, counts = [], {}
    for key in keys:
        selection, ceiling = sel[key], ceil[key]
        treatment = treat[key] if treat else None
        category, flags = classify(selection, ceiling, treatment, args.correct_threshold)
        counts[category] = counts.get(category, 0) + 1
        task, sample_id, sample_index = key
        rows.append(dict(
            task=task, id=sample_id, sample_index=sample_index,
            category=category, **flags,
            f1_selection=float(selection["official_qa_f1_truncated"]),
            f1_ceiling=float(ceiling["official_qa_f1_truncated"]),
            f1_treatment=(None if treatment is None
                          else float(treatment["official_qa_f1_truncated"])),
            prediction_selection=selection.get("prediction"),
            prediction_ceiling=ceiling.get("prediction"),
            prediction_treatment=None if treatment is None else treatment.get("prediction"),
            answers=selection.get("answers"),
            compression_count=int(selection.get("compression_count", 0)),
            question=question_text(args.data_dir, task, sample_index)))
    order = {name: i for i, name in enumerate(
        ("lost_to_compression", "rescued", "missed", "gained", "damaged",
         "compression_helped", "both_correct", "both_wrong"))}
    rows.sort(key=lambda r: (order.get(r["category"], 99), r["task"], r["sample_index"]))
    return dict(
        schema="prlmc_per_sample_diff/1",
        selection=sel_meta, ceiling=ceil_meta, treatment=treat_meta,
        partial=any(m and m["partial"] for m in (sel_meta, ceil_meta, treat_meta)),
        correct_threshold=args.correct_threshold,
        paired_samples=len(keys), categories=counts, rows=rows)


def _flat(value, width):
    """Single-line cell: newlines become markers so the table stays aligned."""
    text = str(value).replace("\n", "⏎")
    return text[:width]


def print_table(report):
    header = (f"{'idx':>4} {'category':<20} {'sel':>6} {'ceil':>6} {'treat':>6} "
              f"{'ans':<28} {'selection':<24} {'ceiling':<24} treatment")
    print(header)
    print("-" * len(header))
    for row in report["rows"]:
        answers = _flat("|".join(row["answers"] or []), 28)
        treatment = row["prediction_treatment"]
        print(f"{row['sample_index']:>4} {row['category']:<20} "
              f"{row['f1_selection']:>6.2f} {row['f1_ceiling']:>6.2f} "
              f"{(row['f1_treatment'] if row['f1_treatment'] is not None else float('nan')):>6.2f} "
              f"{answers:<28} {_flat(row['prediction_selection'], 24):<24} "
              f"{_flat(row['prediction_ceiling'], 24):<24} {_flat(treatment, 24)}")
    print()
    for name, count in sorted(report["categories"].items(), key=lambda kv: -kv[1]):
        print(f"{name:<22} {count:>4}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selection", required=True, help="same-budget selector run")
    p.add_argument("--ceiling", required=True, help="no-compression Full run")
    p.add_argument("--treatment", default=None, help="merge run to judge")
    p.add_argument("--data-dir", default=None,
                   help="LongBench test dir; when given, the question text is printed")
    p.add_argument("--correct-threshold", type=float, default=0.0,
                   help="a sample counts as correct when truncated QA F1 exceeds "
                        "this value (default 0.0, i.e. any token overlap)")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--json", action="store_true", help="print JSON instead of the table")
    args = p.parse_args()
    report = build_report(args)
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    if args.json:
        print(text)
    else:
        print_table(report)


if __name__ == "__main__":
    main()
