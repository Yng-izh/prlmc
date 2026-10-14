#!/usr/bin/env python3
"""Run fixed retention samples on independent GPUs and verify before merging."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_qa_compare import atomic_json
from prepare_public_retention import sha, tokenizer_manifest
from public_retention_eval import shard_samples, summarize, validate_plan
from verify_public_retention import verify


def merge_reports(plan, reports):
    if not reports:
        raise ValueError("no shard reports")
    expected_count = len(reports)
    seen = set()
    first = reports[0]
    for report in reports:
        shard = report.get("sharding", {})
        index = shard.get("index")
        if shard.get("count") != expected_count or index in seen:
            raise ValueError("missing/duplicate shard identity")
        expected = [s["sample_index"] for s in shard_samples(plan, index, expected_count)]
        if [s["sample_index"] for s in report["samples"]] != expected or shard.get("sample_indices") != expected:
            raise ValueError("missing/duplicate sample membership")
        seen.add(index)
        for field in ("status", "identity", "protocol", "physical_slots", "matched_budget_slots",
                      "shared_state_sha256_before", "shared_state_sha256_after",
                      "training_performed", "optimizer_created"):
            if report.get(field) != first.get(field):
                raise ValueError("inconsistent shard field: " + field)
        if report["status"] != "complete":
            raise ValueError("incomplete shard")
    merged = copy.deepcopy(first)
    merged["samples"] = sorted([s for report in reports for s in report["samples"]], key=lambda s: s["sample_index"])
    if [s["sample_index"] for s in merged["samples"]] != [s["sample_index"] for s in plan["samples"]]:
        raise ValueError("full sample coverage failed")
    merged["sharding"] = dict(index=0, count=1, sample_indices=[s["sample_index"] for s in plan["samples"]])
    merged["summary"] = summarize(merged)
    merged["worker_runtime"] = [report["runtime"] for report in reports]
    merged["costs"] = dict(worker_seconds_sum=sum(r["costs"]["elapsed_seconds"] for r in reports),
        longest_worker_seconds=max(r["costs"]["elapsed_seconds"] for r in reports),
        peak_allocated_bytes=max(r["costs"]["peak_allocated_bytes"] for r in reports))
    merged.pop("progress", None)
    return merged


def run(args):
    plan = json.loads(Path(args.plan).read_text())
    validate_plan(plan)
    if plan["task"] != "squad" or plan["mode"] != "author512":
        raise ValueError("current authorized fleet is SQuAD author512 only")
    devices = [int(s) for s in args.devices.split(",")]
    if len(devices) != 4 or len(set(devices)) != 4:
        raise ValueError("four distinct GPU devices required")
    if tokenizer_manifest(args.model_path) != plan["tokenizer"]:
        raise ValueError("tokenizer changed")
    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    ledger_path = Path(str(prefix) + ".fleet.json")
    outputs = [Path(str(prefix) + f".shard{i}.json") for i in range(len(devices))]
    if ledger_path.exists() or any(p.exists() for p in outputs) or Path(str(prefix) + ".json").exists():
        raise ValueError("fleet output exists; refuse overwrite")
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free",
                                     "--format=csv,noheader,nounits"], text=True)
    usage = {int(parts[0]): (int(parts[1]), int(parts[2])) for row in query.splitlines()
             for parts in [row.split(",")]}
    for gpu in devices:
        used, free = usage[gpu]
        if used > 256 or free < 30000:
            raise ValueError(f"GPU {gpu} no longer idle: used={used} free={free} MiB")
    ledger = dict(status="starting", plan_sha256=sha(args.plan), started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  task="squad", mode="author512", tasks=[])
    atomic_json(ledger_path, ledger)
    processes, handles = [], []
    try:
        for index, gpu in enumerate(devices):
            log = Path(str(prefix) + f".shard{index}.log")
            command = [sys.executable, str(ROOT / "tools/public_retention_eval.py"),
                "--plan", args.plan, "--model-path", args.model_path, "--checkpoint", args.checkpoint,
                "--cuda-device", str(gpu), "--shard-index", str(index), "--shard-count", str(len(devices)),
                "--out", str(outputs[index])]
            handle = log.open("x")
            handles.append(handle)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=handle, stderr=subprocess.STDOUT)
            processes.append(proc)
            ledger["tasks"].append(dict(index=index, gpu=gpu, pid=proc.pid, log=str(log), out=str(outputs[index]),
                sample_indices=[s["sample_index"] for s in shard_samples(plan, index, len(devices))], status="running"))
            atomic_json(ledger_path, ledger)
        ledger["status"] = "running"
        atomic_json(ledger_path, ledger)
        print(json.dumps(ledger), flush=True)
        while any(proc.poll() is None for proc in processes):
            for proc, task in zip(processes, ledger["tasks"]):
                code = proc.poll()
                task["exit_code"] = code
                task["status"] = "running" if code is None else "complete" if code == 0 else "failed"
                p = Path(task["out"])
                if p.exists():
                    try:
                        report = json.loads(p.read_text())
                        task["progress"] = report.get("progress")
                    except (OSError, ValueError):
                        pass
            atomic_json(ledger_path, ledger)
            time.sleep(30)
        for proc, task in zip(processes, ledger["tasks"]):
            task["exit_code"] = proc.returncode
            task["status"] = "complete" if proc.returncode == 0 else "failed"
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("one or more workers failed; partial outputs retained, no partial aggregate")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        reports = []
        for task in ledger["tasks"]:
            report = json.loads(Path(task["out"]).read_text())
            if report["identity"]["plan_sha256"] != sha(args.plan):
                raise ValueError("worker used a different input plan")
            task["verification"] = verify(plan, report, tokenizer)
            reports.append(report)
        merged = merge_reports(plan, reports)
        merged["worker_artifacts"] = {str(p): sha(p) for p in outputs}
        certificate = verify(plan, merged, tokenizer)
        aggregate = Path(str(prefix) + ".json")
        atomic_json(aggregate, merged)
        certificate.update(plan_sha256=sha(args.plan), report_sha256=sha(aggregate))
        atomic_json(str(prefix) + ".certificate.json", certificate)
        ledger.update(status="complete", aggregate=str(aggregate), summary=merged["summary"])
    except Exception as exc:
        ledger.update(status="failed", failure=dict(type=type(exc).__name__, message=str(exc)))
        # Only terminate this launcher's own children on a startup exception.
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            proc.wait()
        raise
    finally:
        for handle in handles:
            handle.close()
        atomic_json(ledger_path, ledger)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--devices", required=True)
    p.add_argument("--model-path", default="/dev/shm/s1d_reader_fp16")
    p.add_argument("--checkpoint", default="runs/e6d_interleave_ckpt.pt")
    p.add_argument("--prefix", required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
