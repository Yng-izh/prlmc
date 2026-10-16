#!/usr/bin/env python3
"""Detached torchrun launcher with verified smoke gate and all-worker certification."""
import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def finalize_report(path, returncode, world_size):
    report = json.loads(path.read_text()) if path.exists() else {}
    success = returncode == 0 and report.get("status") == "complete"
    report["launcher"] = {
        "torchrun_exit_code": returncode,
        "world_size": world_size,
        "all_workers_succeeded": success,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not success:
        report["status"] = "failed"
        report.setdefault("failure", {
            "type": "TorchrunFailure" if returncode else "IncompleteReport",
            "message": "worker group exit code: %d" % returncode,
        })
    temporary = path.with_suffix(path.suffix + ".launcher.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    temporary.replace(path)
    return success


def verify_smoke(path):
    report = json.loads(path.read_text())
    previous = json.loads((ROOT / "runs/s1d_stage2_oracle.w17510.json").read_text())
    window, old = report["windows"][0], previous["windows"][0]
    if window["sets"] != old["sets"] or any(
            window["baselines"][name] != old["baselines"][name]
            for name in ("full", "select", "d0")):
        raise RuntimeError("multi-GPU smoke changed original baselines/cohorts")
    arm = window["arms"]["oracle_13d_protected"]
    history = arm["history"]
    if len(history) != 1 or history[0]["step"] != 1 or not history[0]["basis_unchanged"]:
        raise RuntimeError("incomplete multi-GPU smoke history")
    if not 0 < history[0]["deviation"] <= 1:
        raise RuntimeError("multi-GPU smoke alpha did not update")
    print("PASS smoke: all workers succeeded; original cohorts; nonzero alpha; frozen basis", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1,2,3,4,6,8,9")
    parser.add_argument("--out", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--smoke-first", action="store_true")
    parser.add_argument("--cache-dir")
    parser.add_argument("--cache-pid", type=int)
    args, remaining = parser.parse_known_args()
    devices = [int(value) for value in args.devices.split(",")]
    if len(devices) != len(set(devices)) or len(devices) < 2:
        raise SystemExit("need at least two distinct GPUs")
    output, log = ROOT / args.out, ROOT / args.log
    if output.exists():
        raise SystemExit("output exists: " + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        marker = Path(args.cache_dir) / "fp16_cache_manifest.json"
        if not marker.exists() and args.cache_pid:
            print("waiting for verified shared reader cache", flush=True)
            try:
                descriptor = os.pidfd_open(args.cache_pid)
            except ProcessLookupError:
                descriptor = None
            if descriptor is not None:
                try:
                    poller = select.poll()
                    poller.register(descriptor, select.POLLIN)
                    if not poller.poll(30 * 60 * 1000):
                        raise RuntimeError("reader cache process exceeded 30 minutes")
                finally:
                    os.close(descriptor)
        if not marker.exists():
            raise RuntimeError("reader cache failed or is incomplete")
        remaining += ["--model-path", args.cache_dir]
        print("verified shared reader cache ready", flush=True)
    environment = dict(os.environ)
    environment.setdefault("GLOO_SOCKET_IFNAME", "lo")

    def run_worker_group(report_path, log_path, extra):
        if report_path.exists():
            raise RuntimeError("output exists: " + str(report_path))
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   "--nnodes=1", "--nproc-per-node=%d" % len(devices),
                   str(ROOT / "tools/s1d_stage2_dev.py"),
                   "--cuda-devices", args.devices, "--out", str(report_path)] + remaining + extra
        print("launching %d-GPU group: %s" % (len(devices), report_path.name), flush=True)
        with log_path.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, env=environment,
                                    stdin=subprocess.DEVNULL, stdout=stream,
                                    stderr=subprocess.STDOUT)
        success = finalize_report(report_path, result.returncode, len(devices))
        print("worker group exit=%d verified_complete=%s" % (result.returncode, success), flush=True)
        if not success:
            raise RuntimeError("distributed worker group failed; inspect " + str(log_path))

    if args.smoke_first:
        smoke = output.with_name(output.stem + ".parallel_smoke.json")
        smoke_log = log.with_name(log.stem + ".parallel_smoke.log")
        run_worker_group(smoke, smoke_log, ["--steps", "1", "--measure-every", "1"])
        verify_smoke(smoke)
    run_worker_group(output, log, [])


if __name__ == "__main__":
    main()
