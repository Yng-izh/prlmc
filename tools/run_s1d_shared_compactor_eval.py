#!/usr/bin/env python3
"""Parallel launcher and episode-level aggregator for frozen shared E6d eval."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from s1d_shared_compactor_eval import DEVELOPMENT_IDS, PROTOCOL, arm_summary, sha, source_manifest


def aggregate(paths, checkpoint):
    rows, failures = [], []
    common_identity = None
    for episode, path in zip(DEVELOPMENT_IDS, paths):
        if not Path(path).exists():
            failures.append(dict(episode=episode, status="missing", path=str(path)))
            continue
        report = json.loads(Path(path).read_text())
        if report.get("status") != "complete":
            failures.append(dict(episode=episode, status=report.get("status"),
                                 path=str(path), failure=report.get("failure")))
            continue
        identity = report.get("identity", {})
        if (report.get("training_performed") is not False or
                report.get("optimizer_created") is not False or
                report.get("backward_performed") is not False or
                report.get("basis_constructed") is not False or
                report.get("alpha_adapted") is not False or
                report.get("source_memory_unchanged") is not True or
                report.get("compactor_unchanged") is not True or
                report.get("compactor_state_sha256_before") != report.get("compactor_state_sha256_after")):
            raise ValueError("frozen evaluation integrity failed")
        if identity.get("episode") != episode or identity.get("protocol") != PROTOCOL:
            raise ValueError("episode/protocol mismatch")
        if identity.get("checkpoint_sha256") != sha(checkpoint):
            raise ValueError("shared checkpoint mismatch")
        if identity.get("source_manifest") != source_manifest():
            raise ValueError("code manifest mismatch")
        comparable = {key: value for key, value in identity.items()
                      if key not in ("episode", "context_sha256")}
        if common_identity is None:
            common_identity = comparable
        elif comparable != common_identity:
            raise ValueError("mixed provenance across episode reports")
        plan = json.loads(Path("runs/sic_paper_controlled_plan.json").read_text())
        from s1d_shared_compactor_eval import choose_development
        from readout_probe import official_score
        answers = {item["question"]: item["answer"] for item in
                   choose_development(plan, episode)["splits"]["evaluation"]}
        if len(answers) != 64 or set(report["generation_questions"]) != set(answers):
            raise ValueError("evaluation question universe changed")
        maps = {}
        for name in PROTOCOL["arms"]:
            predictions = report["predictions"][name]
            if set(predictions) != set(answers):
                raise ValueError("incomplete arm predictions")
            expected_slots = 5376 if name == "Full" else 2816
            if report["arm_metadata"][name]["active_slots"] != expected_slots:
                raise ValueError("active-slot budget changed")
            maps[name] = {}
            for question, row in predictions.items():
                em, substring = official_score(row["prediction"], [answers[question]])
                if row["em"] != em or row["substring_em"] != substring:
                    raise ValueError("saved score differs from rescored prediction")
                maps[name][question] = bool(em)
        summary = arm_summary(maps)
        if summary != report.get("summary"):
            raise ValueError("summary mismatch; report may have been edited")
        rows.append(dict(episode=episode, status="complete", context_sha256=identity["context_sha256"],
                         Full_em=summary["arms"]["Full"]["em"],
                         Selection_em=summary["arms"]["Selection"]["em"],
                         SharedE6d_em=summary["arms"]["SharedE6d"]["em"],
                         delta_em=summary["shared_vs_selection"]["delta_em"],
                         net=summary["shared_vs_selection"]["net"],
                         R=len(summary["shared_recovers_lost"]),
                         H=len(summary["shared_harmed_vs_selection"]),
                         summary=summary, costs=report.get("costs", {})))
    result = dict(schema_version="s1d_shared_compactor_eval_summary/1",
                  status="complete" if not failures else "incomplete", stage="development",
                  protocol=PROTOCOL, checkpoint_sha256=sha(checkpoint),
                  source_manifest=source_manifest(), episodes=rows, failures=failures,
                  sealed_test_opened=False, inference_unit="episode")
    if not failures:
        deltas = [row["delta_em"] for row in rows]
        result.update(mean_delta=sum(deltas) / len(deltas),
                      positive_episodes=sum(delta > 0 for delta in deltas),
                      negative_episodes=sum(delta < 0 for delta in deltas),
                      zero_episodes=sum(delta == 0 for delta in deltas),
                      mean_em={name: sum(row[name + "_em"] for row in rows) / len(rows)
                               for name in ("Full", "Selection", "SharedE6d")})
        # Fixed episode bootstrap, with no question-level independence claim.
        import random
        rng = random.Random(20261012)
        samples = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas)
                         for _ in range(10000))
        result["bootstrap_episode_delta_ci95"] = [samples[249], samples[9749]]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--checkpoint", default="runs/e6d_interleave_ckpt.pt")
    parser.add_argument("--prefix", default="runs/s1d_shared_e6d_dev_v1")
    parser.add_argument("--model-path", default="/dev/shm/s1d_reader_fp16")
    parser.add_argument("--gamma-max", type=float, default=.05)
    args = parser.parse_args()
    devices = [int(item) for item in args.devices.split(",")]
    if not 1 <= len(devices) <= 8 or len(set(devices)) != len(devices):
        raise ValueError("one to eight distinct devices required")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise ValueError("shared checkpoint missing")
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,memory.used",
                                     "--format=csv,noheader,nounits"], text=True)
    available = {int(line.split(",")[0]): [int(value.strip()) for value in line.split(",")[1:]]
                 for line in query.splitlines()}
    if any(device not in available or available[device][0] < 30000 or available[device][1] > 256
           for device in devices):
        raise ValueError("GPU preflight failed; no workers started")
    prefix = Path(args.prefix)
    paths = [Path(str(prefix) + ".ep%d.json" % episode) for episode in DEVELOPMENT_IDS]
    logs = [path.with_suffix(".log") for path in paths]
    fleet = Path(str(prefix) + ".fleet.json")
    if fleet.exists() or any(path.exists() for path in paths + logs):
        raise ValueError("existing artifacts; refuse overwrite")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    jobs = []
    workers = {}
    pending = list(zip(DEVELOPMENT_IDS, paths, logs))

    def launch(device):
        episode, path, log = pending.pop(0)
        command = [sys.executable, "-u", "tools/s1d_shared_compactor_eval.py",
                   "--episode", str(episode), "--cuda-device", str(device),
                   "--checkpoint", str(checkpoint), "--model-path", args.model_path,
                   "--gamma-max", str(args.gamma_max), "--out", str(path)]
        handle = log.open("w")
        worker = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, start_new_session=True)
        job = dict(episode=episode, device=device, pid=worker.pid,
                   out=str(path), log=str(log), command=command, exit_code=None)
        workers[device] = (worker, handle, job)
        jobs.append(job)
        print("START episode=%d gpu=%d pid=%d" % (episode, device, worker.pid), flush=True)

    def save_fleet(status):
        from native_qa_compare import atomic_json
        atomic_json(fleet, dict(status=status, protocol=PROTOCOL, devices=devices,
                               pending_episodes=[episode for episode, _, _ in pending], jobs=jobs))

    for device in devices:
        launch(device)
    save_fleet("running")
    try:
        while workers:
            for device, (worker, handle, job) in list(workers.items()):
                code = worker.poll()
                if code is None:
                    continue
                handle.close()
                job["exit_code"] = code
                del workers[device]
                print("EXIT episode=%d gpu=%d code=%d" % (job["episode"], device, code), flush=True)
                if pending:
                    launch(device)
            save_fleet("running" if workers else "complete")
            if workers:
                time.sleep(5)
        summary = aggregate(paths, checkpoint)
        Path(str(prefix) + ".summary.json").write_text(json.dumps(summary, indent=2))
        print("FINAL", json.dumps({key: summary[key] for key in summary if key in
                                   ("status", "mean_delta", "positive_episodes", "negative_episodes",
                                    "bootstrap_episode_delta_ci95", "failures")}), flush=True)
    except Exception:
        save_fleet("failed")
        raise


if __name__ == "__main__":
    main()
