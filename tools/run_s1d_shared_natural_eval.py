#!/usr/bin/env python3
"""Queue prespecified complete EventQA rows and aggregate by independent context."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time

from s1d_shared_compactor_eval import COMPARISON_ARMS, arm_summary, atomic_json, sha, source_manifest
from s1d_shared_natural_eval import OFFICIAL, eventqa_score, validate_natural_freeze


def aggregate(paths, dataset, checkpoint, prompt):
    import pandas as pd
    frame = pd.read_parquet(dataset)
    rows, failures = [], []
    contexts = set()
    common = None
    for index, path in paths.items():
        result = json.loads(Path(path).read_text()) if Path(path).exists() else {}
        if result.get("status") != "complete":
            failures.append(dict(row=index, status=result.get("status", "missing"), failure=result.get("failure")))
            continue
        if (result.get("dataset_sha256") != sha(dataset) or result.get("checkpoint_sha256") != sha(checkpoint) or
                result.get("prompt_sha256") != sha(prompt) or result.get("source_manifest") != source_manifest() or
                result.get("official_scorer_sha256") != sha(OFFICIAL) or
                result.get("source_memory_unchanged") is not True or result.get("compactor_unchanged") is not True or
                result.get("compactor_state_sha256_before") != result.get("compactor_state_sha256_after")):
            raise ValueError("natural frozen provenance mismatch")
        provenance = {key: result.get(key) for key in ("model_manifest_sha256", "freeze_sha256", "protocol")}
        if common is not None and common != provenance:
            raise ValueError("mixed natural model/freeze/protocol across contexts")
        common = provenance
        import hashlib
        context_hash = hashlib.sha256(frame.iloc[index]["context"].encode()).hexdigest()
        if context_hash != result["context_sha256"] or context_hash in contexts:
            raise ValueError("changed or duplicate natural contexts cannot be independent units")
        contexts.add(context_hash)
        from native_qa_compare import native_pairs
        items = native_pairs(frame.iloc[index])
        answers = {item["question"]: item["answers"] for item in items}
        maps = {}
        for name in COMPARISON_ARMS:
            predictions = result["predictions"][name]
            if set(predictions) != set(answers):
                raise ValueError("natural question universe mismatch")
            maps[name] = {}
            for question, value in predictions.items():
                correct, _ = eventqa_score(value["prediction"], answers[question])
                if bool(correct) != value["em"]:
                    raise ValueError("EventQA score differs from upstream rescore")
                maps[name][question] = bool(correct)
            expected = 512 + (result["written_block_count"] if name == "Full" else 9) * 256
            if result["arm_metadata"][name]["active_slots"] != expected:
                raise ValueError("natural active-slot budget mismatch")
        summary = arm_summary(maps)
        if summary != result.get("summary"):
            raise ValueError("natural summary mismatch")
        rows.append(dict(row=index, context_sha256=context_hash, summary=summary))
    output = dict(status="incomplete" if failures else "complete", metric="official eventqa_recall",
                  inference_unit="independent natural context", rows=rows, failures=failures)
    if not failures and rows:
        output["paired_comparisons"] = {}
        for name in COMPARISON_ARMS:
            if name == "SharedE6d":
                continue
            differences = [row["summary"]["paired_comparisons"][name]["delta_em"] for row in rows]
            rng = random.Random(20261012)
            samples = sorted(sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(10000))
            output["paired_comparisons"][name] = dict(mean_delta=sum(differences) / len(differences),
                context_count=len(rows), ci95=[samples[249], samples[9749]] if len(rows) > 1 else None)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rows", required=True, help="fixed distinct-context row IDs, comma separated")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--model-path", default="/dev/shm/s1d_reader_fp16")
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    ids = [int(item) for item in args.rows.split(",")]
    devices = [int(item) for item in args.devices.split(",")]
    if not ids or len(set(ids)) != len(ids) or not 1 <= len(devices) <= 8 or len(set(devices)) != len(devices):
        raise ValueError("unique rows and one to eight GPUs required")
    protocol = validate_natural_freeze(args.freeze, args.dataset, args.checkpoint, args.prompt_file, args.model_path, ids)
    if ids != protocol["rows"]:
        raise ValueError("natural fleet must cover all prespecified row IDs in the frozen order")
    import pandas as pd
    frame = pd.read_parquet(args.dataset, columns=["context"])
    if any(index < 0 or index >= len(frame) for index in ids) or len({frame.iloc[index]["context"] for index in ids}) != len(ids):
        raise ValueError("all requested rows must be distinct complete contexts")
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,memory.used",
                                     "--format=csv,noheader,nounits"], text=True)
    available = {int(row.split(",")[0]): [int(value.strip()) for value in row.split(",")[1:]] for row in query.splitlines()}
    if any(device not in available or available[device][0] < 30000 or available[device][1] > 256 for device in devices):
        raise ValueError("GPU preflight failed")
    prefix = Path(args.prefix)
    paths = {index: Path(str(prefix) + ".row%d.json" % index) for index in ids}
    fleet = Path(str(prefix) + ".fleet.json")
    if fleet.exists() or any(path.exists() or path.with_suffix(".log").exists() for path in paths.values()):
        raise ValueError("natural artifacts exist; refuse overwrite")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    jobs, workers, pending = [], {}, list(ids)
    def launch(device):
        index = pending.pop(0)
        handle = paths[index].with_suffix(".log").open("w")
        command = [sys.executable, "-u", "tools/s1d_shared_natural_eval.py", "--dataset", args.dataset,
                   "--row", str(index), "--checkpoint", args.checkpoint, "--prompt-file", args.prompt_file,
                   "--freeze", args.freeze, "--model-path", args.model_path, "--cuda-device", str(device), "--out", str(paths[index])]
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        job = dict(row=index, device=device, pid=process.pid, command=command, exit_code=None)
        jobs.append(job)
        workers[device] = (process, handle, job)
    def save(status):
        atomic_json(fleet, dict(status=status, jobs=jobs, pending_rows=pending))
    for device in devices[:len(ids)]:
        launch(device)
    save("running")
    while workers:
        for device, (process, handle, job) in list(workers.items()):
            code = process.poll()
            if code is None:
                continue
            job["exit_code"] = code
            handle.close()
            del workers[device]
            if pending:
                launch(device)
        save("running")
        if workers:
            time.sleep(5)
    summary = aggregate(paths, args.dataset, args.checkpoint, args.prompt_file)
    atomic_json(str(prefix) + ".summary.json", summary)
    failed = summary["status"] != "complete" or any(job["exit_code"] != 0 for job in jobs)
    save("failed" if failed else "complete")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
