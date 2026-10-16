#!/usr/bin/env python3
"""Run the fixed native-50 LongBench matrix with one process per GPU.

The six task/length pairs are independent. A GPU receives at most one model
process at a time; completed jobs are followed by the remaining queue entries.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import sys

from longbench_memoryllm_eval import atomic_json

TASKS = ("hotpotqa", "2wikimqa", "musique", "qasper", "multifieldqa_en", "narrativeqa")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--devices", required=True, help="comma-separated physical GPU IDs")
    p.add_argument("--python", default=sys.executable)
    args = p.parse_args()
    devices = [x.strip() for x in args.devices.split(",") if x.strip()]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("devices must be a nonempty unique list")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    jobs = [(task, length) for length in (8192, 16384) for task in TASKS]
    pending = list(jobs)
    state = dict(schema="longbench_native50_queue/1", status="running", devices=devices,
                 launcher_pid=os.getpid(), started=time.time(),
                 model_path=args.model_path, data_dir=args.data_dir,
                 jobs=[dict(task=t, total_tokens=n, status="queued") for t, n in jobs])
    state_path = out_dir / "launcher.json"

    def save():
        atomic_json(state_path, state)

    def start(device, task, total_tokens):
        out = out_dir / f"native50_{total_tokens}_{task}.json"
        log = out_dir / f"native50_{total_tokens}_{task}.log"
        command = [args.python, "tools/longbench_memoryllm_eval.py",
                   "--data-dir", args.data_dir, "--model-path", args.model_path,
                   "--max_length", str(total_tokens), "--capacity-blocks", "50",
                   "--cuda-device", device, "--tasks", task, "--out", str(out)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, PYTHONUNBUFFERED="1",
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        handle = log.open("w")
        proc = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                                env=env, stdout=handle, stderr=subprocess.STDOUT)
        handle.close()
        return proc, command, out, log

    running = {}
    locks = {}
    save()
    while pending or running:
        for device in devices:
            if device in running or not pending:
                continue
            # Recheck the actual device immediately before each launch; do not
            # queue a model on a GPU another user has begun to occupy.
            if device not in locks:
                handle = open(f"/tmp/prlmc_longbench_gpu_{device}.lock", "a+")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                locks[device] = handle
            check = subprocess.run(["nvidia-smi", f"--id={device}",
                "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15)
            if check.returncode or int(check.stdout.strip()) < 28000:
                continue
            task, length = pending.pop(0)
            index = next(i for i, x in enumerate(state["jobs"])
                         if x["task"] == task and x["total_tokens"] == length and x["status"] == "queued")
            proc, command, out, log = start(device, task, length)
            state["jobs"][index].update(status="running", device=device, pid=proc.pid,
                                         command=command, output=str(out), log=str(log), started=time.time())
            running[device] = (proc, index)
        save()
        for device, (proc, index) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            state["jobs"][index].update(status="complete" if code == 0 else "failed",
                                         exit_code=code, ended=time.time())
            del running[device]
        save()
        if running or pending:
            time.sleep(5)
    state["status"] = "complete" if all(x["status"] == "complete" for x in state["jobs"]) else "failed"
    state["ended"] = time.time()
    save()
    raise SystemExit(0 if state["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
