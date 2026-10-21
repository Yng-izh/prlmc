#!/usr/bin/env python3
"""Where the gold evidence lives, and what the reader can do with it.

For every sample this rebuilds the prompt, locates the sentence containing the
gold answer, and classifies the sample by the cheapest way to reach that
evidence:

    tail_only        visible verbatim in the last 2048 queried tokens
    memory_only      only in the written prefix -> has to come out of memory
    both             reachable either way
    truncated_out    the answer occurs in the prompt but only in the part the
                     16384-token tail truncation dropped (never written, so no
                     memory could hold it)
    no_literal_match the answer string never occurs (alias, paraphrase, or a
                     case/punctuation difference) -> the bucket is unknown, not
                     "absent from memory"

Cross-tabulating those buckets against each arm's F1 splits the loss into a
write-side term (memory_only vs both is what "going through memory" costs) and a
read-side term (the arms only differ on the compression they apply afterwards).

Usage:
    python tools/evidence_location.py
    python tools/evidence_location.py --runs-dir runs/longbench_prlmc_v7_eval --arms Full,Selection
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from block_provenance import answer_forms, sentence_spans  # noqa: E402

TOTAL = 16384
TAIL = 2048


def retained(spans, n_all, tail=TAIL):
    """Spans clipped to the retained region, dropping matches cut away entirely.

    `sentence_spans` reports positions in `all_ids` coordinates, so a match that
    sits in the part of the prompt the tail truncation dropped comes back with a
    negative start. Those matches are *not* in memory -- nothing was written from
    them -- so they must never be bucketed as `memory_only`.
    """
    out = []
    for start, end in spans:
        if end <= 0 or start >= n_all:
            continue
        out.append((max(start, 0), min(end, n_all)))
    return out


def classify(spans, n_all, write_tokens, tail=TAIL, has_match=None):
    """Bucket name for one sample given its gold-evidence token spans.

    `spans` are in `all_ids` coordinates and may be negative (the match exists in
    the prompt but sits before the retained window). `has_match` distinguishes
    "the answer string occurs in the prompt but was truncated away"
    (`truncated_out`) from "the answer never occurs literally" (`no_literal_match`,
    e.g. an alias or a punctuation/case difference).
    """
    if has_match is None:
        has_match = bool(spans)
    if not has_match:
        return "no_literal_match"
    kept = retained(spans, n_all, tail)
    if not kept:
        return "truncated_out"
    tail_edge = n_all - tail
    in_tail = any(end > tail_edge for _, end in kept)
    in_prefix = any(0 <= start < write_tokens for start, _ in kept)
    if in_tail and in_prefix:
        return "both"
    if in_tail:
        return "tail_only"
    return "memory_only"


def tally(rows, arm_f1):
    """Aggregate {bucket: counts} for {arm: {sample_index: f1}}."""
    out = {}
    for bucket, scores in rows:
        cell = out.setdefault(bucket, {"n": 0})
        cell["n"] += 1
        for arm, value in scores.items():
            cell.setdefault(arm + "_right", 0)
            cell.setdefault(arm + "_f1", 0.0)
            cell[arm + "_right"] += value > 0
            cell[arm + "_f1"] += value
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="hotpotqa")
    ap.add_argument("--data-dir", default="/share/latent/benchmarks/LongBench/data")
    ap.add_argument("--runs-dir", default=str(ROOT / "runs/longbench_prlmc_v7_eval"))
    ap.add_argument("--model", default="/share/latent/MemoryLLM-8B")
    ap.add_argument("--arms", default="Full,Selection")
    ap.add_argument("--limit", type=int, default=0, help="0 = every sample in the run")
    ap.add_argument("--show-indices", default="",
                    help="comma separated sample_index list to print per sample")
    args = ap.parse_args()
    show = {int(x) for x in args.show_indices.split(",") if x.strip()}

    sys.path.insert(0, str(ROOT))
    from transformers import AutoTokenizer
    from longbench_memoryllm_eval import load_rows, split_prompt

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = load_rows(args.data_dir, args.task)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    runs = {}
    for arm in arms:
        data = json.loads(Path(args.runs_dir, "%s_%s.json" % (arm, args.task)).read_text())
        runs[arm] = {s["sample_index"]: s["official_qa_f1_truncated"] for s in data["samples"]}
    indices = sorted(set.intersection(*[set(v) for v in runs.values()]))
    if args.limit:
        indices = indices[:args.limit]

    table = []
    for index in indices:
        row = rows[index]
        prompt, all_ids, write_ids, _ = split_prompt(
            tokenizer, args.task, row, TOTAL, tail_tokens=TAIL)
        dropped = len(tokenizer(prompt, add_special_tokens=False).input_ids) - len(all_ids)
        spans = []
        for form in answer_forms(row["answers"][0]):
            spans = sentence_spans(tokenizer, prompt, form, dropped)
            if spans:
                break
        bucket = classify(spans, len(all_ids), len(write_ids))
        table.append((bucket, {arm: runs[arm][index] for arm in arms}))
        if index in show:
            kept = retained(spans, len(all_ids))
            detail = ("kept %d/%d spans" % (len(kept), len(spans))) if spans else "no literal match"
            print("idx %-3d %-15s %-24s %s" % (
                index, bucket, detail,
                "  ".join("%s=%.2f" % (arm, runs[arm][index]) for arm in arms)))

    if show:
        print()

    stats = tally(table, runs)
    head = "%-12s %4s" % ("evidence", "n")
    for arm in arms:
        head += " %14s" % (arm + " 正确")
        head += " %9s" % (arm + " F1")
    print(head)
    for bucket, cell in sorted(stats.items(), key=lambda kv: -kv[1]["n"]):
        line = "%-12s %4d" % (bucket, cell["n"])
        for arm in arms:
            right, f1 = cell[arm + "_right"], cell[arm + "_f1"]
            line += " %8d(%3.0f%%)" % (right, 100 * right / cell["n"])
            line += " %8.1f%%" % (100 * f1 / cell["n"])
        print(line)
    total = sum(c["n"] for c in stats.values())
    print("samples: %d" % total)


if __name__ == "__main__":
    main()
