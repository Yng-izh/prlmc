#!/usr/bin/env python3
"""v6 protocol plan: same sealed v4 episodes/QAs, new DsupV6/DnativeV6 variants.

The v6 route keeps the v4 supervision protocol byte-for-byte (same episodes,
same QA schedule, same objective constants) and changes only the merge
architecture (soft-bounded delta, layer-group gates, similarity hard mask) and
its variant registry. The plan embeds TRAIN_V6 deep-equality like every earlier
generation, so old v4/v5 plans and checkpoints stay valid and untouched.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_natural_merge_v4 import TRAIN_V4, validate_v4_plan, check_answer_encoding
from prepare_public_retention import digest, sha

TRAIN_V6 = copy.deepcopy(TRAIN_V4)
TRAIN_V6.update(
    variants={
        "DsupV6": dict(architecture="grouped_gated_sparse_slot_v6", groups=8, reference_layer=16,
                       routing="top8", topk=8, local_gate_max=.20,
                       routing_score="0.6*native 4096-D cosine + 0.4*learned rank-32 cosine, tau .04,"
                                     " column log-mean load balance .15; per-slot similarity hard mask"
                                     " (reject-only vs peer in-block baseline, margin .01)",
                       gate="slotwise hidden z-score + 3 domain-agnostic geometry features"
                            " (top1 affinity, top-k sharpness, in-block redundancy); zero-initialized;"
                            " one logit per layer group (8 groups)",
                       carrier_selector_layer=16,
                       delta_bound="one-sided soft RMS bound (delta scaled to carrier RMS only when"
                                   " its own RMS exceeds carrier RMS); no forced unit-RMS normalisation;"
                                   " global 5% RMS cap unchanged"),
        "DnativeV6": dict(architecture="grouped_gated_sparse_native_v6", groups=8, reference_layer=16,
                          routing="top8", topk=8, local_gate_max=.20,
                          routing_score="0.6*native 4096-D cosine + 0.4*learned rank-32 cosine, tau .04,"
                                        " column log-mean load balance .15; per-slot similarity hard mask"
                                        " (reject-only vs peer in-block baseline, margin .01)",
                          gate="slotwise hidden z-score + 3 domain-agnostic geometry features"
                               " (top1 affinity, top-k sharpness, in-block redundancy); zero-initialized;"
                               " one logit per layer group (8 groups)",
                          carrier_selector_layer=16,
                          native_path="top-2 sharpened full-width source average minus carrier,"
                                      " routed-confidence scaled; one zero-initialized signed gain per"
                                      " layer group; combined residual one-sided soft RMS bound",
                          delta_bound="one-sided soft RMS bound on the adapted+native combined delta;"
                                      " global 5% RMS cap unchanged")})


def validate_v6_plan(plan):
    identity = dict(plan); expected = identity.pop("content_sha256", None)
    if (expected != digest(identity) or plan.get("schema") != "natural_merge_v6_training_plan/1"
            or plan["training"] != TRAIN_V6):
        raise ValueError("v6 plan integrity/protocol failure")
    old = plan["original_v4_plan"]; validate_v4_plan(old)
    if (plan["base_plan"] != old["base_plan"] or plan["tokenizer"] != old["tokenizer"]
            or plan["eos_token_id"] != old["eos_token_id"]):
        raise ValueError("base protocol/tokenizer changed")
    if len(plan["episodes"]) != len(old["episodes"]):
        raise ValueError("training streams changed")
    for previous, ep in zip(old["episodes"], plan["episodes"]):
        if previous != ep:
            raise ValueError("original episode changed")
    if plan["qa_events"] != old["qa_events"]:
        raise ValueError("v6 QA schedule differs from sealed v4 schedule")


def build_plan(original_v4, source_sha):
    validate_v4_plan(original_v4)
    plan = dict(schema="natural_merge_v6_training_plan/1",
                original_v4_plan=original_v4, original_v4_file_sha256=source_sha,
                base_plan=original_v4["base_plan"], tokenizer=original_v4["tokenizer"],
                eos_token_id=original_v4["eos_token_id"], training=TRAIN_V6,
                episodes=copy.deepcopy(original_v4["episodes"]),
                qa_events=copy.deepcopy(original_v4["qa_events"]))
    plan["content_sha256"] = digest(plan)
    validate_v6_plan(plan)
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v4-train-plan", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    source = Path(args.v4_train_plan)
    original = json.loads(source.read_text())
    plan = build_plan(original, sha(source))
    Path(args.out).write_text(json.dumps(plan, sort_keys=True) + "\n")
    print(json.dumps(dict(schema=plan["schema"], content_sha256=plan["content_sha256"],
                          variants=sorted(TRAIN_V6["variants"]), out=args.out)))


if __name__ == "__main__":
    main()
