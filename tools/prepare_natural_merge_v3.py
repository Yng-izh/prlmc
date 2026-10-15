#!/usr/bin/env python3
"""Seal the C/D protocol and re-encode original training answers without '</s>'."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_natural_merge_v2 import TRAIN_V2, validate_v2_plan, selected_qas
from prepare_public_retention import digest, sha, tokenizer_manifest, write_new

TRAIN_V3 = {k: copy.deepcopy(TRAIN_V2[k]) for k in (
    "seed", "streams", "rounds", "planned_updates", "single_merge_updates", "recurrent_updates",
    "rank", "gamma_max", "optimizer", "weight_decay", "gradient_clip", "qa_schedule",
    "curriculum", "single_window_starts", "recurrent_event_before_writes", "teacher_weight",
    "teacher_temperature", "teacher_max_gold_ce", "teacher_scope", "distillation",
    "truncated_gradient_at_compression", "checkpoint")}
TRAIN_V3.update(matrix_learning_rate=1e-3, gate_learning_rate=1e-2,
    global_rms_budget=.05, initial_gate=.02, protection_weight=1., protection_margin=0.,
    answer_supervision="tokenizer(original first answer, add_special_tokens=False); no appended terminator",
    objective="noncarrier recovery weighted CE + eligible Full KL; retained-native QA protection hinge",
    recovery_weight="1 + clamp(CE_Selection - CE_Full, 0, 2); mean one within noncarrier cohort",
    cohort_policy="native carrier origins plus untouched tail are protected; all other written original contexts recover",
    carrier_policy="unchanged content-only selector shared by Selection/C/D; no QA-conditioned selection",
    variants={
        "C": dict(architecture="budgeted_dense_slot_v3", groups=1, reference_layer=0,
                  routing="dense", local_gate_max=.05, gate="uniform scalar"),
        "D": dict(architecture="grouped_gated_sparse_slot_v3", groups=8, reference_layer=16,
                  routing="top8", topk=8, local_gate_max=.20, gate="slotwise shared across layers")})


def validate_v3_plan(plan):
    identity = dict(plan); expected = identity.pop("content_sha256", None)
    if (expected != digest(identity) or plan.get("schema") != "natural_merge_v3_training_plan/1"
            or plan.get("training") != TRAIN_V3):
        raise ValueError("v3 plan integrity/protocol failure")
    original = plan["original_v2_plan"]; validate_v2_plan(original)
    if (plan["base_plan"] != original["base_plan"] or plan["tokenizer"] != original["tokenizer"]
            or len(plan["episodes"]) != len(original["episodes"])):
        raise ValueError("v3 original training scope changed")
    for old_ep, ep in zip(original["episodes"], plan["episodes"]):
        if {k:v for k,v in ep.items() if k != "contexts"} != {k:v for k,v in old_ep.items() if k != "contexts"}:
            raise ValueError("v3 raw writes changed")
        if len(ep["contexts"]) != len(old_ep["contexts"]):
            raise ValueError("v3 context coverage changed")
        for old_c, c in zip(old_ep["contexts"], ep["contexts"]):
            if {k:v for k,v in c.items() if k != "qas"} != {k:v for k,v in old_c.items() if k != "qas"}:
                raise ValueError("v3 original contexts changed")
            if len(c["qas"]) != len(old_c["qas"]):
                raise ValueError("v3 original QA coverage changed")
            for old, q in zip(old_c["qas"], c["qas"]):
                excluded = {"answer_ids", "answer_sha256", "training_answer_text"}
                if {k:v for k,v in q.items() if k not in excluded} != {k:v for k,v in old.items() if k not in excluded}:
                    raise ValueError("v3 original questions/answers/scoring changed")
                if (q.get("training_answer_text") != old["answer"] or not q["answer_ids"]
                        or any(type(i) is not int or i < 0 for i in q["answer_ids"])
                        or digest(q["answer_ids"]) != q["answer_sha256"]
                        or len(q["prompt_ids"]) + len(q["answer_ids"]) > 512):
                    raise ValueError("v3 answer-only supervision integrity failure")


def check_answer_encoding(plan, tokenizer):
    for ep in plan["episodes"]:
        for c in ep["contexts"]:
            for q in c["qas"]:
                if tokenizer(q["answer"], add_special_tokens=False).input_ids != q["answer_ids"]:
                    raise ValueError("original answer token encoding mismatch")


def build_plan(original, tokenizer, source_sha256):
    validate_v2_plan(original)
    episodes = copy.deepcopy(original["episodes"])
    for ep in episodes:
        for c in ep["contexts"]:
            for q in c["qas"]:
                q["training_answer_text"] = q["answer"]
                q["answer_ids"] = tokenizer(q["answer"], add_special_tokens=False).input_ids
                q["answer_sha256"] = digest(q["answer_ids"])
    plan = dict(schema="natural_merge_v3_training_plan/1", original_v2_plan=original,
        original_v2_file_sha256=source_sha256, base_plan=original["base_plan"],
        tokenizer=original["tokenizer"], training=TRAIN_V3, episodes=episodes)
    plan["content_sha256"] = digest(plan)
    validate_v3_plan(plan); check_answer_encoding(plan, tokenizer)
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v2-train-plan", required=True); p.add_argument("--out", required=True)
    p.add_argument("--model-path", default="/dev/shm/s1d_reader_fp16")
    args = p.parse_args()
    original = json.loads(Path(args.v2_train_plan).read_text())
    if tokenizer_manifest(args.model_path) != original["tokenizer"]:
        raise ValueError("training tokenizer changed")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    plan = build_plan(original, tokenizer, sha(args.v2_train_plan)); write_new(args.out, plan)
    print(json.dumps(dict(out=args.out, content_sha256=plan["content_sha256"],
        contexts=sum(len(e["contexts"]) for e in plan["episodes"]),
        qas=sum(len(c["qas"]) for e in plan["episodes"] for c in e["contexts"]))))


if __name__ == "__main__": main()
