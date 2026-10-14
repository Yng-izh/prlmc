#!/usr/bin/env python3
"""Independently rescore every saved token sequence and recompute paired results."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_public_retention import sha, tokenizer_manifest, write_new
from public_retention_eval import ARMS, E6D_SHA, author_score, summarize, validate_plan, shard_samples


def verify(plan, report, tokenizer):
    validate_plan(plan)
    if report.get("status") != "complete" or report.get("protocol") != plan["protocol"]:
        raise ValueError("complete fixed-protocol report required")
    if report.get("training_performed") is not False or report.get("optimizer_created") is not False:
        raise ValueError("test-time training was not disabled")
    if report["identity"]["checkpoint_sha256"] != E6D_SHA:
        raise ValueError("unexpected shared checkpoint")
    if report["shared_state_sha256_before"] != report["shared_state_sha256_after"]:
        raise ValueError("shared parameter state changed")
    sharding = report.get("sharding", dict(index=0, count=1))
    expected_samples = shard_samples(plan, sharding["index"], sharding["count"])
    if "sample_indices" in sharding and sharding["sample_indices"] != [s["sample_index"] for s in expected_samples]:
        raise ValueError("sample shard membership changed")
    if len(report["samples"]) != len(expected_samples):
        raise ValueError("missing original samples")
    rescored = 0
    for sample, actual in zip(expected_samples, report["samples"]):
        for key, original in (("sample_index", sample["sample_index"]), ("original_id", sample["id"]),
                              ("question", sample["question"]), ("original_answer", sample["answer"]),
                              ("context_sha256", sample["context_sha256"])):
            if actual[key] != original:
                raise ValueError("original input identity changed")
        target = tokenizer.decode(tokenizer(sample["answer"] + "</s>", add_special_tokens=False).input_ids)
        if actual["scoring_target"] != target:
            raise ValueError("scoring target was modified")
        writes = sample["related_writes"] + plan["unrelated_writes"]
        if set(actual["arms"]) != set(ARMS):
            raise ValueError("missing comparison arm")
        for arm in ARMS:
            result = actual["arms"][arm]
            if result["status"] != "complete" and not (arm == "Full" and result["status"] == "capacity_exceeded"):
                raise ValueError("unexpected arm failure")
            expected_count = len(writes) if result["status"] == "complete" else result["failed_before_write"]
            if len(result["trace"]) != expected_count:
                raise ValueError("missing write/readout records")
            for index, entry in enumerate(result["trace"]):
                write = writes[index]
                if entry["index"] != index or entry["role"] != write["role"] or entry["input_sha256"] != write["token_sha256"]:
                    raise ValueError("write order/input changed")
                generated = entry["generation"]
                prediction = tokenizer.decode(generated["token_ids"], skip_special_tokens=False)
                if prediction != generated["prediction"] or author_score(prediction, target) != generated["score"]:
                    raise ValueError("saved generation/scoring mismatch")
                if len(generated["token_ids"]) > 10:
                    raise ValueError("generation protocol changed")
                cap = report["physical_slots"] if arm in ("Full", "NativeDefault") else report["matched_budget_slots"]
                if not 0 < entry["active_slots"] <= cap:
                    raise ValueError("active memory budget violation")
                rescored += 1
            if arm in ("Selection", "Recent", "SharedE6d"):
                schedule = [(e["before_write"], e["source_slots"], e["target_slots"]) for e in result["compressions"]]
                if arm == "Selection":
                    reference = schedule
                elif schedule != reference:
                    raise ValueError("managed arms used different compression schedules")
    if summarize(report) != report["summary"]:
        raise ValueError("paired summary disagrees with saved scores")
    return dict(status="passed", samples=len(expected_samples), generated_readouts=rescored,
                validity=report["summary"]["validity"], inference_unit="context, not independent QA")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from transformers import AutoTokenizer
    plan = json.loads(Path(a.plan).read_text())
    report = json.loads(Path(a.report).read_text())
    if report["identity"]["plan_sha256"] != sha(a.plan) or tokenizer_manifest(a.model_path) != plan["tokenizer"]:
        raise ValueError("plan/tokenizer hash changed")
    result = verify(plan, report, AutoTokenizer.from_pretrained(a.model_path, local_files_only=True))
    result.update(plan_sha256=sha(a.plan), report_sha256=sha(a.report), verifier_sha256=sha(__file__))
    write_new(a.out, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
