#!/usr/bin/env python3
"""Run the two prespecified shared-training seeds on one or two idle GPUs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from prepare_s1d_shared_training import validate_training_plan
from s1d_shared_compactor_eval import atomic_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--training-plan", required=True)
    parser.add_argument("--reserved-plan", default="runs/sic_paper_controlled_plan.json")
    parser.add_argument("--model-path", default="/dev/shm/s1d_reader_fp16")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    devices = [int(value) for value in args.devices.split(",")]
    if not 1 <= len(devices) <= 2 or len(set(devices)) != len(devices):
        raise ValueError("one or two distinct GPUs required")
    plan = json.loads(Path(args.training_plan).read_text())
    reserved = json.loads(Path(args.reserved_plan).read_text())
    validate_training_plan(plan, reserved)
    if plan.get("token_budget_verified") is not True:
        raise ValueError("training token audit missing")
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,memory.used",
                                     "--format=csv,noheader,nounits"], text=True)
    available = {int(row.split(",")[0]): [int(value.strip()) for value in row.split(",")[1:]]
                 for row in query.splitlines()}
    if any(device not in available or available[device][0] < 30000 or available[device][1] > 256 for device in devices):
        raise ValueError("GPU preflight failed; no training started")
    prefix = Path(args.prefix)
    paths = {seed: {kind: Path(str(prefix) + ".seed%d.%s" % (seed, suffix))
                   for kind, suffix in (("out", "json"), ("checkpoint", "pt"), ("log", "log"))}
             for seed in (0, 1)}
    fleet = Path(str(prefix) + ".fleet.json")
    if fleet.exists() or any(path.exists() for values in paths.values() for path in values.values()):
        raise ValueError("artifacts exist; refuse overwrite")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    jobs, workers, pending = [], {}, [0, 1]

    def launch(device):
        seed = pending.pop(0)
        command = [sys.executable, "-u", "tools/s1d_shared_compactor_train.py",
                   "--training-plan", args.training_plan, "--reserved-plan", args.reserved_plan,
                   "--model-path", args.model_path, "--cuda-device", str(device), "--seed", str(seed),
                   "--out", str(paths[seed]["out"]), "--checkpoint-out", str(paths[seed]["checkpoint"])]
        if args.smoke:
            command += ["--smoke", "--steps-a", "1", "--steps-b", "4"]
        handle = paths[seed]["log"].open("w")
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        job = dict(seed=seed, device=device, pid=process.pid, command=command, exit_code=None,
                   **{kind: str(path) for kind, path in paths[seed].items()})
        jobs.append(job)
        workers[device] = (process, handle, job)
        print("START seed=%d gpu=%d pid=%d" % (seed, device, process.pid), flush=True)

    def save(status):
        atomic_json(fleet, dict(status=status, jobs=jobs, pending_seeds=pending,
            training_plan_sha256=sha(args.training_plan), reserved_plan_sha256=sha(args.reserved_plan),
            seeds=[0, 1], smoke=args.smoke, development_scored=False, test_opened=False))

    for device in devices:
        launch(device)
    save("running")
    while workers:
        for device, (process, handle, job) in list(workers.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            job["exit_code"] = code
            del workers[device]
            if pending:
                launch(device)
        save("running")
        if workers:
            time.sleep(5)
    expected = "smoke_complete" if args.smoke else "complete"
    failed = False
    for job in jobs:
        path = Path(job["out"])
        result = json.loads(path.read_text()) if path.exists() else {}
        checkpoint = Path(job["checkpoint"])
        job["status"] = result.get("status", "missing")
        if (job["exit_code"] != 0 or job["status"] != expected or
                not checkpoint.exists() or result.get("checkpoint_sha256") != sha(checkpoint) or
                result.get("training_plan_sha256") != sha(args.training_plan) or
                result.get("protocol", {}).get("seed") != job["seed"]):
            failed = True
    save("failed" if failed else expected)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
