#!/usr/bin/env python3
"""v7 protocol plan: same sealed v4 episodes/QAs, v7 variants, targeted recovery.

The v7 route keeps the sealed v4 supervision corpus (same episodes, same raw
writes, same answer/EOS encoding, same objective constants) and changes exactly
three things:

1. the merge architecture (v7 classes: soft-bounded tanh delta, sparse high-SNR
   gate, v6 layer-group routing inherited unchanged);
2. the merge geometry the student stream runs: 13 -> 9 native blocks at
   `used >= min(budget, window+1)` = 14 blocks, which is the geometry the v7
   evaluation stream compacts. `TRAIN_V4`/`TRAIN_V6` are never mutated: archived
   v4/v5/v6 plans embed them under deep-equality assertions.
3. the recovery weight: a targeted `1 + 3*min(gap, 2)` inside the
   `ce_full <= 2.5 and gap > 0.25` band instead of the flat v4 clamp.

The QA schedule is re-derived from the v7 constants (not copied from v4)
because the recurrent events have to fire at the 13/9 streaming trigger: the
sealed rotation distributes a context's QAs over *all* of its event
appearances, so moving the recurrent events necessarily moves the single-merge
QA split. The single-merge event positions and source windows stay sealed.
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

TRAIN_V7 = copy.deepcopy(TRAIN_V4)
TRAIN_V7.update(
    # The supervised merge window is the evaluation merge window: 13 -> 9 at a
    # 20-block managed budget, so the trigger is min(20, 14) = 14 blocks.
    geometry=dict(budget_blocks=20, window_blocks=13, target_blocks=9),
    # The sealed v4 single-merge curriculum accumulates 19 original contexts
    # before the supervised event. That is the episode length, not the merge
    # window; changing it would break the sealed QA coverage.
    single_window_contexts=19,
    # Rounds 2/3 events fire where the 13/9 student stream would compact:
    # used = 14 blocks = window + 1, before writes 14 and 18. Two events per
    # stream keeps the sealed 32 single + 32 recurrent = 64 update budget.
    recurrent_event_before_writes=[14, 18],
    curriculum="rounds 0/1: two isolated 19-context windows (sealed v4 accumulation,"
               " supervised 13->9 merge); rounds 2/3: 39-context recurrent streams compacted"
               " at the 13/9 trigger (events before writes 14 and 18)",
    # Targeted recovery gate (C10). `teacher_max_gold_ce` stays 2.0 and is a
    # *different* gate: a sample in 2.0 < ce_full <= 2.5 carries a recovery
    # weight but no KL teacher, by design.
    recovery_gate=dict(ce_full_max=2.5, gap_min=0.25, gap_cap=2.0, scale=3.0, background=0.25),
    recovery_weight="gap = CE_Selection - CE_Full; raw = 1 + 3*min(gap, 2) if recover and"
                    " CE_Full <= 2.5 and gap > 0.25, else 0.25 background anchor; recover weights"
                    " normalised to mean one per event, the 0.25 anchor is absolute; the protect"
                    " cohort keeps the 0.25 anchor, so its recovery CE term is non-zero by design;"
                    " teacher KL eligibility stays at CE_Full <= 2.0 (2.0 < CE_Full <= 2.5 carries"
                    " recovery weight but no distillation)",
    objective="context-balanced targeted recovery content CE (ce_full<=2.5 and gap>0.25 ->"
              " 1+3*min(gap,2), else 0.25) + eligible Full content KL + protection content hinge"
              " + first-token Selection KL anchor + separate EOS CE",
    variants={
        "DsupV7": dict(architecture="grouped_gated_sparse_slot_v7", groups=8, reference_layer=16,
                       routing="top8", topk=8, local_gate_max=.20,
                       routing_score="0.6*native 4096-D cosine + 0.4*learned rank-32 cosine, tau .04,"
                                     " column log-mean load balance .15; per-slot similarity hard mask"
                                     " (reject-only vs peer in-block baseline, margin .01)",
                       gate="slotwise hidden z-score + 3 domain-agnostic geometry features"
                            " (top1 affinity, top-k sharpness, in-block redundancy); zero-initialized;"
                            " one logit per layer group (8 groups); sparse high-signal mask"
                            " sigmoid(4*(affinity_z + .15)) keeps only the top-affinity slots active",
                       carrier_selector_layer=16,
                       delta_bound="soft-bounded tanh: delta = carrier_rms*tanh(raw/carrier_rms);"
                                   " no forced unit-RMS normalisation, no 1/sqrt(var+eps) zero-point"
                                   " amplification; global 5% RMS cap unchanged"),
        "DnativeV7": dict(architecture="grouped_gated_sparse_native_v7", groups=8, reference_layer=16,
                          routing="top8", topk=8, local_gate_max=.20,
                          routing_score="0.6*native 4096-D cosine + 0.4*learned rank-32 cosine, tau .04,"
                                        " column log-mean load balance .15; per-slot similarity hard mask"
                                        " (reject-only vs peer in-block baseline, margin .01)",
                          gate="slotwise hidden z-score + 3 domain-agnostic geometry features"
                               " (top1 affinity, top-k sharpness, in-block redundancy); zero-initialized;"
                               " one logit per layer group (8 groups); sparse high-signal mask"
                               " sigmoid(4*(affinity_z + .15)) keeps only the top-affinity slots active",
                          carrier_selector_layer=16,
                          native_path="top-2 sharpened full-width source average minus carrier,"
                                      " routed-confidence scaled; one zero-initialized signed gain per"
                                      " layer group; combined residual soft-bounded with tanh",
                          delta_bound="soft-bounded tanh on the adapted+native combined delta;"
                                      " no forced unit-RMS normalisation; global 5% RMS cap unchanged")})


def make_schedule_v7(episodes):
    """Deterministic v7 QA schedule (same distribution rule as the sealed v4 one).

    Single-merge events keep the sealed 19-context windows, so that half of the
    schedule is identical to v4; recurrent events sit at the 13/9 streaming
    trigger instead of the old 19/9 positions 20/30.
    """
    events = []
    for rnd in range(4):
        for ep in episodes:
            if rnd < 2:
                for start in TRAIN_V7["single_window_starts"][rnd]:
                    events.append(dict(round=rnd, stream=ep["id"], stage="single_merge",
                                       window_start=start,
                                       source_indices=list(range(start, start + TRAIN_V7["single_window_contexts"]))))
            else:
                for before in TRAIN_V7["recurrent_event_before_writes"]:
                    events.append(dict(round=rnd, stream=ep["id"], stage="recurrent",
                                       before_write=before, source_indices=list(range(before))))
    appearances = {}
    for i, event in enumerate(events):
        for source in event["source_indices"]:
            appearances.setdefault((event["stream"], source), []).append(i)
    by_id = {ep["id"]: ep for ep in episodes}
    for i, event in enumerate(events):
        ids = []
        for source in event["source_indices"]:
            qas = by_id[event["stream"]]["contexts"][source]["qas"]
            occurrences = appearances[(event["stream"], source)]
            j = occurrences.index(i)
            chosen = qas[j::len(occurrences)] or [qas[j % len(qas)]]
            ids.extend(q["id"] for q in chosen)
        event["qa_ids"] = ids
    return events


def validate_v7_plan(plan):
    identity = dict(plan); expected = identity.pop("content_sha256", None)
    if (expected != digest(identity) or plan.get("schema") != "natural_merge_v7_training_plan/1"
            or plan["training"] != TRAIN_V7):
        raise ValueError("v7 plan integrity/protocol failure")
    old = plan["original_v4_plan"]; validate_v4_plan(old)
    if (plan["base_plan"] != old["base_plan"] or plan["tokenizer"] != old["tokenizer"]
            or plan["eos_token_id"] != old["eos_token_id"]):
        raise ValueError("base protocol/tokenizer changed")
    if plan["episodes"] != old["episodes"]:
        raise ValueError("sealed v4 training episodes changed")
    # The single-merge half of the sealed v4 curriculum keeps its event
    # positions and source windows verbatim. Only the QA-to-event assignment is
    # re-derived (the sealed rotation splits a context's QAs over all of its
    # appearances, and the recurrent appearances moved to the 13/9 trigger).
    def without_qa_ids(event):
        return {k: v for k, v in event.items() if k != "qa_ids"}
    if ([without_qa_ids(e) for e in plan["qa_events"] if e["stage"] == "single_merge"]
            != [without_qa_ids(e) for e in old["qa_events"] if e["stage"] == "single_merge"]):
        raise ValueError("sealed v4 single-window events changed")
    if plan["qa_events"] != make_schedule_v7(plan["episodes"]):
        raise ValueError("v7 QA event schedule changed")
    all_ids = {q["id"] for e in plan["episodes"] for c in e["contexts"] for q in c["qas"]}
    seen = {i for e in plan["qa_events"] for i in e["qa_ids"]}
    if all_ids != seen:
        raise ValueError("incomplete original QA training coverage")
    geometry = plan["training"]["geometry"]
    if not 0 < geometry["target_blocks"] < geometry["window_blocks"] <= geometry["budget_blocks"]:
        raise ValueError("invalid v7 merge geometry")


def build_plan(original_v4, source_sha):
    validate_v4_plan(original_v4)
    plan = dict(schema="natural_merge_v7_training_plan/1",
                original_v4_plan=original_v4, original_v4_file_sha256=source_sha,
                base_plan=original_v4["base_plan"], tokenizer=original_v4["tokenizer"],
                eos_token_id=original_v4["eos_token_id"], training=TRAIN_V7,
                episodes=copy.deepcopy(original_v4["episodes"]),
                qa_events=make_schedule_v7(original_v4["episodes"]))
    plan["content_sha256"] = digest(plan)
    validate_v7_plan(plan)
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
                          geometry=TRAIN_V7["geometry"],
                          variants=sorted(TRAIN_V7["variants"]), out=args.out)))


if __name__ == "__main__":
    main()
