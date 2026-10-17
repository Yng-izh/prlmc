#!/usr/bin/env python3
"""Create a reviewable method freeze from complete development evidence only."""
import argparse
import json
from pathlib import Path

from s1d_shared_compactor_eval import (COMPARISON_ARMS, DEVELOPMENT_IDS, atomic_json,
                                     evaluation_protocol, freeze_manifest, sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="runs/sic_paper_controlled_plan.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-path", default="/dev/shm/s1d_reader_fp16")
    parser.add_argument("--development-prefixes", required=True,
                        help="comma-separated complete fleets: old + new seed0 + new seed1")
    parser.add_argument("--decision", required=True, help="prespecified method choice and rationale")
    parser.add_argument("--natural-dataset")
    parser.add_argument("--natural-prompt-file")
    parser.add_argument("--natural-rows", help="prespecified complete natural context row IDs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise ValueError("freeze exists; refuse overwrite")
    for suffix in (".test_opened.json", ".s1d_test_opened.json"):
        if Path(args.plan).with_suffix(suffix).exists():
            raise ValueError("test already opened; cannot make a new method selection")
    from run_s1d_shared_compactor_eval import aggregate
    protocol = evaluation_protocol(COMPARISON_ARMS)
    evidence = []
    candidates = set()
    seeds = set()
    old_e6d_seen = False
    for prefix in args.development_prefixes.split(","):
        fleet_path = Path(prefix + ".fleet.json")
        fleet = json.loads(fleet_path.read_text())
        jobs = fleet.get("jobs", [])
        if (len(jobs) != 8 or {job["episode"] for job in jobs} != set(DEVELOPMENT_IDS) or
                any(job["exit_code"] != 0 for job in jobs) or fleet.get("status") != "complete"):
            raise ValueError("all eight successful development workers required")
        paths = [Path(prefix + ".ep%d.json" % episode) for episode in DEVELOPMENT_IDS]
        first = json.loads(paths[0].read_text())
        command = next(job["command"] for job in jobs)
        checkpoint = command[command.index("--checkpoint") + 1]
        result = aggregate(paths, checkpoint, args.plan, protocol)
        if (result["status"] != "complete" or first["identity"]["gamma_max"] != .05 or
                first["identity"]["model_manifest_sha256"] != sha(Path(args.model_path) / "fp16_cache_manifest.json")):
            raise ValueError("complete comparable baseline evidence required")
        training = first.get("checkpoint_training_protocol")
        if training:
            if training.get("name") != "S1d-independent-shared-E6d-training-v1" or training.get("smoke") is not False:
                raise ValueError("new candidate must be formal independent shared training")
            seeds.add(training["seed"])
        else:
            if sha(checkpoint) != "a9fe2ea3f7a4e2c21590cf2219591edb11fc65297ce36ddeed96c236d8020c5d":
                raise ValueError("legacy reference must be archived original E6d checkpoint")
            old_e6d_seen = True
        candidates.add(sha(checkpoint))
        evidence.append(dict(prefix=prefix, fleet_sha256=sha(fleet_path),
                             checkpoint_sha256=sha(checkpoint), summary=result))
    if not old_e6d_seen or not {0, 1}.issubset(seeds) or len(evidence) < 3:
        raise ValueError("old E6d and both prespecified new-training seeds must be evaluated before freeze")
    if sha(args.checkpoint) not in candidates:
        raise ValueError("chosen checkpoint was not in complete development comparisons")
    manifest = freeze_manifest(args.plan, args.checkpoint, args.model_path,
                               evaluation_protocol(COMPARISON_ARMS, stage="test"), .05)
    natural = None
    if any((args.natural_dataset, args.natural_prompt_file, args.natural_rows)):
        if not all((args.natural_dataset, args.natural_prompt_file, args.natural_rows)):
            raise ValueError("natural dataset, prompt and row IDs must be frozen together")
        rows = [int(value) for value in args.natural_rows.split(",")]
        if not rows or len(set(rows)) != len(rows) or any(row < 0 for row in rows):
            raise ValueError("unique nonnegative natural row IDs required")
        natural = dict(dataset_sha256=sha(args.natural_dataset), prompt_sha256=sha(args.natural_prompt_file),
                       rows=rows, max_new_tokens=32, stop_strings=["\n"], scorer="official eventqa_recall")
    atomic_json(args.out, dict(status="frozen", manifest=manifest, plan_sha256=sha(args.plan),
        all_development_complete=True, test_scored=False, development_evidence=evidence,
        decision=args.decision, test_episode_count=16, inference_unit="episode", natural_protocol=natural))


if __name__ == "__main__":
    main()
