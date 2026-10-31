#!/usr/bin/env python3
"""Three-way LoRA diagnosis on one frozen sample set.

    A  V9 fusion + joint Reader LoRA    the failing configuration
    B  V9 fusion, LoRA disabled         is the adapter itself collapsing generation
    C  Selection eviction + same LoRA   does the adapter generate without fusion

All three runs share the same --sample-ids file, the same 50-block / 19->9
geometry, the same model path and the same 128k input, so the only differences
are the merge arm and the adapter. The id list is taken from an earlier report,
so the questions can never drift between runs.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "main_table"
CKPT = "runs/natural_merge_v9/DnativeV9.50block.joint.pt"
LORA = "runs/natural_merge_v9/reader_lora_joint.pt"

CONFIGS = (
    ("A_ours_lora", "V9 + joint LoRA",
     dict(arm="SharedMerge", checkpoint=CKPT, reader_lora=LORA)),
    ("B_ours_nolora", "V9, LoRA off",
     dict(arm="SharedMerge", checkpoint=CKPT)),
    ("C_selection_lora", "Selection + joint LoRA",
     dict(arm="Selection", reader_lora=LORA)),
)


def pin_ids(source, out, expect):
    """Freeze the id list once; never silently replace an existing pin."""
    report = json.loads(Path(source).read_text())
    ids = [s.get("id") for s in report["samples"]]
    if any(i is None for i in ids):
        raise SystemExit(f"{source} has samples without ids")
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{source} has duplicate ids")
    if expect is not None and len(ids) != expect:
        raise SystemExit(f"{source} holds {len(ids)} samples, --expect-ids says {expect}")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        previous = json.loads(out.read_text())
        if previous != ids:
            raise SystemExit(f"{out} already pins different ids; refusing to overwrite")
        print(f"pinned ids reused: {out} ({len(ids)} ids)")
    else:
        out.write_text(json.dumps(ids, indent=1))
        print(f"pinned ids written: {out} ({len(ids)} ids)")
    return ids


def build(spec, args, ids_path, out):
    cmd = [sys.executable, str(ROOT / "tools" / "longbench_v2_prlmc_eval.py"),
           "--data-json", args.data_json, "--length", args.length,
           "--sample-ids", str(ids_path), "--arm", spec["arm"],
           "--budget-blocks", "50", "--window-blocks", "19",
           "--target-blocks", "9", "--trigger-blocks", "50",
           "--model-path", args.model_path, "--total-tokens", "128000",
           "--write-cache-slots", "1", "--num-shards", str(args.num_shards),
           "--out", str(out)]
    # --arm Selection refuses a checkpoint on purpose: C must not inherit V9 fusion.
    if spec.get("checkpoint"):
        cmd += ["--checkpoint", spec["checkpoint"]]
    if spec.get("reader_lora"):
        cmd += ["--reader-lora", spec["reader_lora"]]
    return cmd


def run_one(name, spec, args, ids_path, tag):
    out = OUT_DIR / f"ab13_{tag}_{name}.json"
    log = OUT_DIR / f"ab13_{tag}_{name}.log"
    if out.exists() or list(OUT_DIR.glob(out.name + ".shard*")):
        raise SystemExit(f"output exists; refuse overwrite: {out}")
    cmd = build(spec, args, ids_path, out)
    print(f"[{name}] {' '.join(cmd)}", flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        code = subprocess.run(cmd, cwd=str(ROOT), stdout=fh,
                              stderr=subprocess.STDOUT).returncode
    if code != 0:
        raise SystemExit(f"[{name}] exit {code}; see {log}")
    return json.loads(out.read_text())


def summarize(report):
    samples = report["samples"]
    n = max(1, len(samples))
    return dict(n=len(samples), correct=sum(s["correct"] for s in samples),
                empty=sum(1 for s in samples if not s["prediction"].strip()),
                mean_output_tokens=round(
                    sum(len(s["output_tokens"]) for s in samples) / n, 1))


def clipped(text, width=60):
    text = (text or "").replace("\n", "\\n").strip()
    return text[:width] + ("..." if len(text) > width else "")


def compare(reports, ids, tag):
    rows = []
    for index, sid in enumerate(ids):
        row = dict(index=index, id=sid)
        for name, _, _ in CONFIGS:
            samples = sorted(reports[name]["samples"], key=lambda s: s["sample_index"])
            if len(samples) != len(ids):
                raise SystemExit(f"[{name}] produced {len(samples)} samples, pinned {len(ids)}")
            sample = samples[index]
            if str(sample["id"]) != str(sid):
                raise SystemExit(f"[{name}] sample order drifted at {index}")
            row[name] = dict(letter=sample["extracted_letter"],
                             correct=sample["correct"], answer=sample["answer"],
                             output_tokens=len(sample["output_tokens"]),
                             prediction=clipped(sample["prediction"]),
                             seconds=round(sample["seconds"], 1))
        rows.append(row)

    print("\n## per-sample (letter / n_out)")
    header = " | ".join([f"{name} ({label})" for name, label, _ in CONFIGS])
    print(f"| # | id | answer | {header} |")
    print("|---|---|---|" + "---|" * len(CONFIGS))
    for row in rows:
        cells = " | ".join(
            f"{row[name]['letter'] or '-'} / {row[name]['output_tokens']}"
            for name, _, _ in CONFIGS)
        print(f"| {row['index']} | `{row['id']}` | {row['A_ours_lora']['answer']} | {cells} |")

    print("\n## per-config")
    for name, label, _ in CONFIGS:
        samples = reports[name]["samples"]
        n = max(1, len(samples))
        print(f"{name:18s} {label:26s} n={len(samples):2d} "
              f"correct={sum(s['correct'] for s in samples):2d} "
              f"empty={sum(1 for s in samples if not s['prediction'].strip()):2d} "
              f"mean_out={sum(len(s['output_tokens']) for s in samples) / n:5.1f} "
              f"compressions={[s['compression_count'] for s in samples]} "
              f"elapsed={reports[name].get('elapsed_seconds', 0.0):.0f}s")

    deltas = sum(1 for row in rows
                 if len({row[name]["prediction"] for name, _, _ in CONFIGS}) > 1)
    print(f"\nprediction differs on {deltas}/{len(rows)} samples")

    payload = dict(
        schema="ab_lora_diagnosis/1", tag=tag, sample_ids=ids,
        configs={name: dict(label=label, spec=spec,
                            summary=summarize(reports[name]),
                            identity=reports[name].get("identity"),
                            elapsed_seconds=reports[name].get("elapsed_seconds"))
                 for name, label, spec in CONFIGS},
        rows=rows)
    path = OUT_DIR / f"ab13_{tag}_compare.json"
    if path.exists():
        raise SystemExit(f"compare output exists; refuse overwrite: {path}")
    path.write_text(json.dumps(payload, indent=1))
    print(f"written: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-json",
                   default="/data/zhangjiayang/yyz/datasets/LongBench-v2/data.json")
    p.add_argument("--model-path", default="/dev/shm/s1d_reader_fp16")
    p.add_argument("--length", default="medium",
                   choices=("all", "short", "medium", "long"))
    p.add_argument("--source-report",
                   default="reports/main_table/smoke50_ours_joint.json",
                   help="report whose samples define the frozen id list")
    p.add_argument("--ids-out", default="reports/main_table/ab13_ids.json")
    p.add_argument("--expect-ids", type=int, default=13)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--tag", default="r1")
    p.add_argument("--only", default=None,
                   help="comma separated config names to run (default: all three)")
    args = p.parse_args()

    ids_path = ROOT / args.ids_out
    ids = pin_ids(ROOT / args.source_report, ids_path, args.expect_ids)

    wanted = set(args.only.split(",")) if args.only else None
    reports = {}
    for name, _, spec in CONFIGS:
        if wanted is not None and name not in wanted:
            continue
        reports[name] = run_one(name, spec, args, ids_path, args.tag)

    if len(reports) == len(CONFIGS):
        compare(reports, ids, args.tag)
    else:
        print(json.dumps({n: summarize(r) for n, r in reports.items()},
                         ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
