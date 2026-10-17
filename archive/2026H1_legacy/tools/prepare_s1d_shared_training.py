#!/usr/bin/env python3
"""Prepare training-only controlled scenes disjoint from all reserved entities."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import sys

# Deployment includes some old top-level copies of tool modules. Load the
# maintained tools/ versions consistently from scripts, tests, and launchers.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sic_paper_data import (check_token_budget, digest, extract, file_hash,
                            generate, save_json, templates, tokenizer_manifest, validate)

ROOT = Path(__file__).resolve().parents[1]


def entity_inventory(plan):
    # Integrity inspection only: no predictions, scores, or test ledger.
    validate(plan)
    return {fact["subject"] for episode in plan["episodes"] for fact in episode["facts"]}


def historical_entities(contexts, specs):
    schedule = ROOT / "tools/e6d_schedule.py"
    if schedule.is_file():
        tree = ast.parse(schedule.read_text())
        windows = next(ast.literal_eval(node.value) for node in tree.body
                       if isinstance(node, ast.Assign) and any(
                           isinstance(target, ast.Name) and target.id == "DEFAULT_WINDOWS" for target in node.targets))
    else:
        # The source snapshot is not required on a deployment checkout. These
        # are the frozen E6d windows recorded in the experiment protocol.
        windows = [(3, 17890), (3, 17130), (3, 16750), (3, 14470),
                   (3, 17510), (3, 14850), (3, 16370), (3, 15230)]
    result = set()
    for row, start in windows:
        lines = [line for line in contexts[row].splitlines() if line[:1].isdigit()]
        result.update(fact["subject"] for fact in extract("\n".join(lines[start:start + 410]), specs))
    return result


def build_training_plan(contexts, specs, reserved, episodes=8, seed=20261014, source_hashes=None):
    if episodes < 1:
        raise ValueError("positive training episode count required")
    blocked = entity_inventory(reserved) | historical_entities(contexts, specs)
    raw = generate(contexts, specs, episodes=episodes, dev=episodes, seed=seed,
                   source_hashes=source_hashes, excluded_subjects=blocked)
    if raw["status"] != "completed":
        return raw  # Fail explicitly; never recycle reserved entities.
    for episode in raw["episodes"]:
        episode["partition"] = "train"
        # New episodes are training data throughout. No imitation of a held-out
        # evaluation split inside the training file; supervision is prespecified.
        episode["training_probes"] = episode.pop("splits")["discovery"][:20]
        episode.pop("episode_sha256")
        episode["episode_sha256"] = digest(episode)
    raw.pop("plan_sha256")
    raw["actual_counts"].update(development=0, test=0, training=episodes)
    raw["protocol"]["splits"] = {"training_probes": 20}
    raw["protocol"]["test_scored"] = False
    raw.update(schema_version="s1d_shared_training_plan/1", requested_development=0,
               isolation=dict(reserved_plan_content_sha256=reserved["plan_sha256"],
                   excluded_entities=sorted(blocked), entity_disjoint=True,
                   historical_e6d_windows_excluded=True, sealed_test_scored=False),
               supervision="20 prespecified relation-balanced latest write-time facts per episode; B2 prompt",
               test_scored=False)
    raw["plan_sha256"] = digest(raw)
    validate_training_plan(raw, reserved)
    return raw


def validate_training_plan(plan, reserved=None):
    if plan.get("schema_version") != "s1d_shared_training_plan/1" or plan.get("status") != "completed":
        raise ValueError("completed training-only plan required")
    copy = dict(plan)
    copy.pop("plan_sha256", None)
    if digest(copy) != plan.get("plan_sha256"):
        raise ValueError("training plan hash mismatch")
    if plan.get("test_scored") is not False:
        raise ValueError("test scoring is prohibited during preparation")
    blocked = set(plan["isolation"]["excluded_entities"])
    if reserved is not None:
        if reserved.get("plan_sha256") != plan["isolation"]["reserved_plan_content_sha256"]:
            raise ValueError("reserved plan changed")
        blocked |= entity_inventory(reserved)
    seen = set()
    from s1d_shared_compactor_eval import validate_write_plan
    for index, episode in enumerate(plan["episodes"]):
        if episode.get("partition") != "train" or episode["episode"] != index or "splits" in episode:
            raise ValueError("only new sequential training episodes accepted")
        copy = dict(episode)
        copy.pop("episode_sha256", None)
        if digest(copy) != episode["episode_sha256"]:
            raise ValueError("training episode hash mismatch")
        validate_write_plan(episode)
        if len(episode["facts"]) != 410:
            raise ValueError("training scene requires 410 facts")
        for line, fact in enumerate(episode["facts"]):
            begin, end = fact["answer_span"]
            rendered = "%d. %s" % (line + 1, fact["source_text"][:begin] + fact["answer"] + fact["source_text"][end:])
            if (episode["lines"][line] != rendered or fact["line"] != line or fact["serial"] != line + 1 or
                    fact["block"] != (None if line < 30 else (line - 30) // 20)):
                raise ValueError("training fact rendering or position mismatch")
            if fact["answer"] not in plan["answer_banks"].get(str(fact["relation"]), []):
                raise ValueError("training answer outside source bank")
        anchor = {fact["subject"] for fact in episode["facts"][:30]}
        written = episode["facts"][30:]
        counts = Counter(fact["subject"] for fact in written)
        if len(anchor) != 30 or len(counts) != 128 or anchor & set(counts) or sorted(counts.values()) != [2] * 4 + [3] * 124:
            raise ValueError("training fact/entity budget changed")
        if any(len({fact["answer"] for fact in written if fact["subject"] == subject}) != count
               for subject, count in counts.items()):
            raise ValueError("training versions must be distinct")
        subjects = {fact["subject"] for fact in episode["facts"]}
        if len(subjects) != 158 or subjects & (seen | blocked):
            raise ValueError("training/reserved or cross-training entity overlap")
        seen |= subjects
        probes = episode["training_probes"]
        if len(probes) != 20 or len({probe["question"] for probe in probes}) != 20:
            raise ValueError("exactly 20 fixed training probes required")
        latest = {fact["subject"]: fact for fact in episode["facts"]}
        for probe in probes:
            if any(probe[field] != latest[probe["subject"]][field] for field in
                   ("question", "relation", "answer", "line", "block", "serial", "source_serial")):
                raise ValueError("training answer differs from latest written fact")
    if len(plan["episodes"]) != plan["requested_episodes"]:
        raise ValueError("training episode count changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--templates", required=True)
    parser.add_argument("--reserved-plan", default="runs/sic_paper_controlled_plan.json")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--tokenizer", default="/dev/shm/s1d_reader_fp16")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise ValueError("training plan exists; refuse overwrite")
    import pandas as pd
    contexts = pd.read_parquet(args.dataset, columns=["context"])["context"].tolist()
    reserved = json.loads(Path(args.reserved_plan).read_text())
    plan = build_training_plan(contexts, templates(args.templates), reserved, args.episodes, args.seed,
                               {name: file_hash(path) for name, path in (
                                   ("dataset", args.dataset), ("templates", args.templates),
                                   ("reserved_plan", args.reserved_plan), ("generator", __file__))})
    if plan["status"] == "completed":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        plan["tokenizer_manifest"] = tokenizer_manifest(args.tokenizer)
        check_token_budget(plan, lambda text: tokenizer(text, add_special_tokens=False).input_ids)
        if plan["status"] == "completed":
            validate_training_plan(plan, reserved)
    save_json(args.out, plan)
    print(json.dumps({key: plan[key] for key in ("status", "capacity", "feasible_adjustments") if key in plan}))
    if plan["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
