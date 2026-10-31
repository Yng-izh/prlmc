#!/usr/bin/env python3
"""Build the v9 training plan: v8 curriculum, v9 architecture.

Sealed-plan rule: TRAIN_V4/V7/V8 validators deep-compare against their frozen
records, so v9 is a NEW schema (`natural_merge_v9_training_plan/1`). No earlier
record is mutated.

Changes over v8 (all structural, none in the curriculum):
1. the per-token knowledge protection term is switched off (weights 0): the v9
   in-block partition protects the carrier core structurally, so a token-level
   hinge on the same slots would be a second, conflicting way of saying the same
   thing;
2. the global RMS budget rises .05 -> .08 and is spent only on the 15.6% bridge
   slots, which is what makes the residual readable at the trigger without
   touching the protected core;
3. the variant is `DnativeV9` (`grouped_gated_sparse_native_v9`): top-1 sharp
   native skip, orthogonal low-rank branch, per-layer-group gate, anchor-bridge
   selection and the write-once lock.

Geometry, curriculum, streams, events and QA supervision are inherited from the
sealed v8 plan unchanged, so the update count (128) and the single/recurrent
split (64/64) stay comparable with v7 and v8.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_public_retention import digest  # noqa: E402
from prepare_natural_merge_v8 import make_schedule_v8  # noqa: E402

SCHEMA = "natural_merge_v9_training_plan/1"
TRAIN_V9 = None
RECURRENT_EVENTS = [20, 24]
TRIGGER_BLOCKS = 20
SINGLE_STARTS = [[0, 20], [1, 20]]
SINGLE_WINDOW_CONTEXTS = 19
ARCHITECTURE = "grouped_gated_sparse_native_v9"
GLOBAL_RMS_BUDGET = .08
BRIDGE_BLOCKS = 3
NGRAM_KERNEL = (.25, .5, .25)
NATIVE_INIT = 0.


def build_training(v8_training):
    training = copy.deepcopy(v8_training)
    variant = copy.deepcopy(training["variants"]["DnativeV7"])
    variant.update(dict(
        architecture=ARCHITECTURE,
        top_sharp=1,
        route_temperature_sharp=.1,
        bridge_fraction=.156,
        bridge_gamma_max=.60,
        significance_affinity_weight=.5,
        write_threshold=.5,
    ))
    training.update(dict(
        schema=SCHEMA,
        variants={"DnativeV9": variant},
        # The structural core replaces the token-level protection term.
        token_protection_weight=0.,
        token_kl_weight=0.,
        global_rms_budget=GLOBAL_RMS_BUDGET,
        bridge_blocks=BRIDGE_BLOCKS,
        ngram_kernel=list(NGRAM_KERNEL),
        native_init=NATIVE_INIT,
        objective="context-balanced targeted recovery content CE (ce_full<=2.5 and gap>0.25 ->"
                  " 1+3*min(gap,2), else 0.25) + eligible Full content KL + protection content hinge"
                  " + first-token Selection KL anchor + separate EOS CE; per-token protection is"
                  " disabled (weight 0) because the 84.4% protected core is masked structurally",
    ))
    return training


def freeze_v9_training(training):
    global TRAIN_V9
    TRAIN_V9 = training
    return TRAIN_V9


def build_plan(v8_plan, hotpot_payload):
    if v8_plan.get("schema") != "natural_merge_v8_training_plan/1":
        raise ValueError("sealed v8 plan required as the v9 base")
    if v8_plan["content_sha256"] != digest(
            {k: v for k, v in v8_plan.items() if k != "content_sha256"}):
        raise ValueError("sealed v8 plan file integrity failure")
    training = freeze_v9_training(build_training(v8_plan["training"]))
    plan = dict(schema=SCHEMA,
                original_v8_plan_sha256=v8_plan["content_sha256"],
                original_v7_plan=v8_plan["original_v7_plan"],
                original_v4_plan=v8_plan["original_v4_plan"],
                base_plan=v8_plan["base_plan"], tokenizer=v8_plan["tokenizer"],
                eos_token_id=v8_plan["eos_token_id"], training=training,
                episodes=v8_plan["episodes"],
                hotpot_episodes=v8_plan["hotpot_episodes"],
                hotpot_provenance=v8_plan["hotpot_provenance"],
                hotpot_parquet_sha256=v8_plan["hotpot_parquet_sha256"],
                qa_events=make_schedule_v8(v8_plan["episodes"]))
    plan["content_sha256"] = digest(plan)
    validate_v9_plan(plan, base=v8_plan)
    return plan


def validate_v9_plan(plan, base=None):
    if plan.get("schema") != SCHEMA:
        raise ValueError("v9 schema required")
    training = plan["training"]
    allowed_geometries = (dict(budget_blocks=20, window_blocks=13, target_blocks=9),
                          dict(budget_blocks=50, window_blocks=19, target_blocks=9))
    if training["geometry"] not in allowed_geometries:
        raise ValueError("unsupported merge geometry (sealed 20/13->9 or paper 50/19->9)")
    allowed_triggers = (TRIGGER_BLOCKS, 50)
    if training["trigger_blocks"] not in allowed_triggers:
        raise ValueError("trigger must be 20 (sealed) or 50 (paper)")
    allowed_events = (RECURRENT_EVENTS, [50, 60, 70, 80, 90, 100, 110, 120])
    if training["recurrent_event_before_writes"] not in allowed_events:
        raise ValueError("unsupported recurrent event schedule")
    if training["single_window_contexts"] != SINGLE_WINDOW_CONTEXTS:
        raise ValueError("single-merge curriculum stays sealed")
    if training["single_window_starts"] != SINGLE_STARTS:
        raise ValueError("single-merge window starts stay sealed")
    if training["planned_updates"] != 128 or training["streams"] != 16:
        raise ValueError("v9 runs 16 episodes x 4 rounds = 128 updates")
    if (training["token_protection_weight"], training["token_kl_weight"]) != (0., 0.):
        raise ValueError("v9 disables the per-token protection term")
    if training["global_rms_budget"] != GLOBAL_RMS_BUDGET:
        raise ValueError("v9 budget must be the bridge-scaled .08")
    if list(training["variants"]) != ["DnativeV9"]:
        raise ValueError("v9 trains the Dnative main line only")
    variant = training["variants"]["DnativeV9"]
    if variant["architecture"] != ARCHITECTURE:
        raise ValueError("v9 variant architecture mismatch")
    if not variant["local_gate_max"] <= variant["bridge_gamma_max"] < 1.:
        raise ValueError("bridge cap must sit between the core cap and one")
    if not 0. < variant["bridge_fraction"] < 1.:
        raise ValueError("bridge fraction must be a strict interior ratio")
    if training["native_init"] != 0.:
        raise ValueError("v9 starts at the Selection identity (native gain 0)")
    if training["bridge_blocks"] != BRIDGE_BLOCKS:
        raise ValueError("v9 bridge seat count changed")
    if plan["qa_events"] != make_schedule_v8(plan["episodes"]):
        raise ValueError("v9 QA event schedule changed")
    seen = {i for e in plan["qa_events"] for i in e["qa_ids"]}
    expected = {q["id"] for e in plan["episodes"] for c in e["contexts"] for q in c["qas"]}
    if seen != expected:
        raise ValueError("v9 must supervise every original training QA")
    if base is not None:
        if plan["episodes"] != base["episodes"] or plan["qa_events"] != base["qa_events"]:
            raise ValueError("v9 curriculum must be inherited unchanged from v8")
        if plan["tokenizer"] != base["tokenizer"]:
            raise ValueError("v9 tokenizer changed")
    for ep in plan["episodes"]:
        for c in ep["contexts"]:
            for q in c["qas"]:
                if q["context_sha256"] != c["context_sha256"]:
                    raise ValueError("QA/context provenance hash mismatch")


def ensure_train_v9():
    """Deterministically rebuild TRAIN_V9 for checkpoint validation."""
    global TRAIN_V9
    if TRAIN_V9 is not None:
        return TRAIN_V9
    plan_path = ROOT / "data" / "natural_merge_v9" / "train.json"
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != SCHEMA:
        raise ValueError("v9 plan file missing or wrong schema")
    v8_plan = json.loads((ROOT / "data" / "natural_merge_v8" / "train.json").read_text())
    hotpot = json.loads((ROOT / "data" / "natural_merge_v8" / "hotpot_train_episodes.json").read_text())
    rebuilt = build_plan(v8_plan, hotpot)
    if rebuilt["content_sha256"] != plan["content_sha256"]:
        raise ValueError("v9 plan rebuild does not match the sealed plan file")
    TRAIN_V9 = rebuilt["training"]
    return TRAIN_V9


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v8-train-plan", required=True)
    p.add_argument("--hotpot-episodes", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    v8_plan = json.loads(Path(args.v8_train_plan).read_text())
    hotpot_payload = json.loads(Path(args.hotpot_episodes).read_text())
    plan = build_plan(v8_plan, hotpot_payload)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(plan, sort_keys=True) + "\n")
    print(json.dumps(dict(schema=plan["schema"], out=args.out,
                          content_sha256=plan["content_sha256"],
                          episodes=len(plan["episodes"]),
                          qa_events=len(plan["qa_events"]),
                          architecture=plan["training"]["variants"]["DnativeV9"]["architecture"],
                          global_rms_budget=plan["training"]["global_rms_budget"],
                          trigger_blocks=plan["training"]["trigger_blocks"])))


if __name__ == "__main__":
    main()
