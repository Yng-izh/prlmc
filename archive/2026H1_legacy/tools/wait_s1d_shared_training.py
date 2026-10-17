#!/usr/bin/env python3
"""Queue seed0/1 GPU smoke, shared training and frozen development evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from s1d_shared_compactor_eval import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-plan", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--baseline-prefix", required=True)
    args = parser.parse_args()
    prefix = Path(args.prefix)
    scheduler = Path(str(prefix) + ".queue.json")
    if scheduler.exists():
        raise ValueError("queue exists; refuse duplicate scheduler")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pending = [0, 1]
    jobs, workers = [], {}

    def launch(seed, device, phase):
        stem = str(prefix) + ".seed%d" % seed
        if phase == "evaluation":
            command = [sys.executable, "-u", "tools/run_s1d_shared_compactor_eval.py",
                       "--devices", str(device), "--checkpoint", stem + ".pt",
                       "--arms", "Full,Selection,SharedE6d,Recent9,Random9",
                       "--prefix", stem + ".development"]
            log = Path(stem + ".development.launcher.log")
        else:
            out = stem + (".smoke.json" if phase == "smoke" else ".json")
            checkpoint = stem + (".smoke.pt" if phase == "smoke" else ".pt")
            if Path(out).exists() or Path(checkpoint).exists():
                raise ValueError("artifacts exist: " + out)
            command = [sys.executable, "-u", "tools/s1d_shared_compactor_train.py",
                       "--training-plan", args.training_plan, "--seed", str(seed),
                       "--cuda-device", str(device), "--out", out, "--checkpoint-out", checkpoint]
            if phase == "smoke":
                command += ["--smoke", "--steps-a", "1", "--steps-b", "4"]
            log = Path(stem + (".smoke.log" if phase == "smoke" else ".log"))
        if log.exists():
            raise ValueError("log exists: " + str(log))
        handle = log.open("w")
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        job = dict(seed=seed, device=device, phase=phase, pid=process.pid,
                   command=command, log=str(log), exit_code=None)
        jobs.append(job)
        workers[device] = (process, handle, job)
        print("START", json.dumps(job), flush=True)

    def save(status):
        atomic_json(scheduler, dict(status=status, pending_seeds=pending, jobs=jobs,
                                   training_plan=args.training_plan, test_opened=False))

    save("waiting_for_idle_gpu")
    while pending or workers:
        for device, (process, handle, job) in list(workers.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            job["exit_code"] = code
            del workers[device]
            print("EXIT", json.dumps(job), flush=True)
            if code == 0 and job["phase"] == "smoke":
                launch(job["seed"], device, "training")
            elif code == 0 and job["phase"] == "training":
                launch(job["seed"], device, "evaluation")
        # Avoid racing the baseline launcher's next queued episode on its GPUs.
        fleet = Path(args.baseline_prefix + ".fleet.json")
        baseline = json.loads(fleet.read_text()) if fleet.exists() else {}
        reserved = set(baseline.get("devices", [])) if baseline.get("status") == "running" else set()
        query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,memory.used",
                                         "--format=csv,noheader,nounits"], text=True)
        for row in query.splitlines():
            device, free, used = [int(value.strip()) for value in row.split(",")]
            if pending and device not in workers and device not in reserved and free >= 30000 and used <= 256:
                launch(pending.pop(0), device, "smoke")
        save("running" if workers else "waiting_for_idle_gpu")
        if pending or workers:
            time.sleep(30)
    failed = any(job["exit_code"] != 0 for job in jobs)
    save("failed" if failed else "complete")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
