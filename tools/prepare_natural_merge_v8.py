#!/usr/bin/env python3
"""Build the v8 training plan: trigger-20 geometry + HotpotQA supervision.

Sealed-plan rule: TRAIN_V4/V7 validators deep-compare against their frozen
records, so v8 is a NEW schema (`natural_merge_v8_training_plan/1`). The v7
records are never mutated.

User-registered changes over v7:
1. recurrent events fire before writes 20 and 24 (stream trigger 20 blocks:
   the student stream compacts at used >= 20, so events land where evaluation
   compacts); recurrent QA prefixes follow (0..19, 0..23);
2. the recurrent student/teacher streams run trigger_blocks=20 (the sealed v7
   geometry 20/13->9 and the supervised 13->9 merge window stay unchanged);
3. per-token knowledge protection (Selection-correct positions only);
4. official HotpotQA train episodes are appended as training supervision only.

QA rotation follows the sealed derivation rule (`make_schedule_v7`): a
context's QAs are split over all of its event appearances in order. Because
the recurrent event positions move, the QA schedule must be re-derived, not
copied. Single-merge windows stay sealed (19 contexts, starts [0,20]/[1,20]).
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_public_retention import digest, sha  # noqa: E402

SCHEMA = "natural_merge_v8_training_plan/1"
# Module-level training record for checkpoint validation
# (`natural_merge_runtime.training_for` deep-compares a v8 checkpoint's
# embedded training dict against this). Set by build_training(); kept frozen
# after `freeze_v8_training` so validation uses the exact sealed bytes.
TRAIN_V8 = None
RECURRENT_EVENTS = [20, 24]
TRIGGER_BLOCKS = 20
SINGLE_STARTS = [[0, 20], [1, 20]]
SINGLE_WINDOW_CONTEXTS = 19


def make_schedule_v8(episodes):
    """v7 rotation rule with v8 event positions."""
    events = []
    for rnd in range(4):
        for ep in episodes:
            if rnd < 2:
                for start in SINGLE_STARTS[rnd]:
                    events.append(dict(round=rnd, stream=ep["id"], stage="single_merge",
                                       window_start=start,
                                       source_indices=list(range(start, start + SINGLE_WINDOW_CONTEXTS))))
            else:
                for before in RECURRENT_EVENTS:
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


def ensure_train_v8():
    """Deterministically rebuild and cache TRAIN_V8 (runtime validation entry).

    Reads the sealed v7 plan and the hotpot episodes from the repo data
    directory; the rebuild is verified against the on-disk v8 plan
    content_sha256 so a drifted corpus fails loudly instead of validating a
    mismatched checkpoint.
    """
    global TRAIN_V8
    if TRAIN_V8 is not None:
        return TRAIN_V8
    plan_path = ROOT / "data" / "natural_merge_v8" / "train.json"
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != SCHEMA:
        raise ValueError("v8 plan file missing or wrong schema")
    v7_plan = json.loads((ROOT / "data" / "natural_merge_v7" / "train.json").read_text())
    hotpot = json.loads((ROOT / "data" / "natural_merge_v8" / "hotpot_train_episodes.json").read_text())
    rebuilt = build_plan(v7_plan, hotpot)
    if rebuilt["content_sha256"] != plan["content_sha256"]:
        raise ValueError("v8 plan rebuild does not match the sealed plan file")
    TRAIN_V8 = rebuilt["training"]
    return TRAIN_V8


def freeze_v8_training(training):
    """Pin the module-level TRAIN_V8 to the exact record baked into checkpoints."""
    global TRAIN_V8
    TRAIN_V8 = training
    return TRAIN_V8


def build_training(v7_training):
    training = copy.deepcopy(v7_training)
    training.update(dict(
        schema=SCHEMA,
        recurrent_event_before_writes=RECURRENT_EVENTS,
        trigger_blocks=TRIGGER_BLOCKS,
        token_protection_weight=.5,
        token_protection_margin=.30,
        token_kl_weight=.25,
        # 16 episodes (8 sealed + 8 hotpot) x 4 rounds: 64 single + 64 recurrent.
        planned_updates=128,
        single_merge_updates=64,
        recurrent_updates=64,
        streams=16,
        curriculum="rounds 0/1: two isolated 19-context windows (sealed v4 accumulation,"
                   " supervised 13->9 merge); rounds 2/3: 39-context recurrent streams compacted"
                   " at the 20-block trigger (events before writes 20 and 24)",
        recovery_weight=training["recovery_weight"],
        objective="context-balanced targeted recovery content CE (ce_full<=2.5 and gap>0.25 ->"
                  " 1+3*min(gap,2), else 0.25) + eligible Full content KL + protection content hinge"
                  " + first-token Selection KL anchor + per-token Selection-correct degraded-token"
                  " protection (hinge margin .30 @ weight .5, Selection KL @ weight .25; Selection-"
                  "wrong positions carry no imitation) + separate EOS CE",
        qa_schedule="complete original QA coverage + official HotpotQA train multi-hop supervision"
                    " across sealed per-context event occurrences; context-balanced losses;"
                    " questions are training supervision only and never pre-fed to writer/selection",
    ))
    return training


def validate_v8_plan(plan):
    from prepare_natural_merge_v7 import TRAIN_V7, make_schedule_v7, validate_v7_plan
    if plan.get("schema") != SCHEMA:
        raise ValueError("v8 schema required")
    if plan["training"] == TRAIN_V7:
        raise ValueError("v8 training record must differ from sealed TRAIN_V7")
    if TRAIN_V7["recurrent_event_before_writes"] != [14, 18]:
        raise ValueError("sealed TRAIN_V7 was mutated")
    training = plan["training"]
    if training["geometry"] != dict(budget_blocks=20, window_blocks=13, target_blocks=9):
        raise ValueError("v8 merge geometry stays 20/13->9")
    if training["trigger_blocks"] != TRIGGER_BLOCKS:
        raise ValueError("v8 trigger must be 20 blocks")
    if training["recurrent_event_before_writes"] != RECURRENT_EVENTS:
        raise ValueError("v8 recurrent events must be [20, 24]")
    if training["single_window_contexts"] != 19 or training["single_window_starts"] != SINGLE_STARTS:
        raise ValueError("single-merge curriculum stays sealed")
    if training["planned_updates"] != 128 or training["streams"] != 16:
        raise ValueError("v8 runs 16 episodes x 4 rounds = 128 updates")
    for key in ("token_protection_weight", "token_protection_margin", "token_kl_weight"):
        if key not in training:
            raise ValueError(f"missing v8 token-protection constant: {key}")
    if plan["qa_events"] != make_schedule_v8(plan["episodes"]):
        raise ValueError("v8 QA event schedule changed")
    old = plan["original_v4_plan"]
    # Sealed v4 single-window event positions/windows verbatim, restricted to
    # the sealed streams (hotpot episodes add their own single-window events).
    def without_qa_ids(event):
        return {k: v for k, v in event.items() if k != "qa_ids"}
    sealed_streams = {e["id"] for e in old["episodes"]}
    if ([without_qa_ids(e) for e in plan["qa_events"]
         if e["stage"] == "single_merge" and e["stream"] in sealed_streams]
            != [without_qa_ids(e) for e in old["qa_events"] if e["stage"] == "single_merge"]):
        raise ValueError("sealed v4 single-window events changed")
    # Full original-QA coverage over the sealed corpus episodes.
    sealed_v4_ids = {q["id"] for e in old["episodes"] for c in e["contexts"] for q in c["qas"]}
    seen = {i for e in plan["qa_events"] for i in e["qa_ids"]}
    if not sealed_v4_ids.issubset(seen):
        raise ValueError("incomplete sealed original QA training coverage")
    hotpot_ids = {q["id"] for e in plan["hotpot_episodes"] for c in e["contexts"] for q in c["qas"]}
    if sealed_v4_ids & hotpot_ids:
        raise ValueError("hotpot QA ids must be disjoint from the sealed corpus ids")
    if not hotpot_ids.issubset(seen):
        raise ValueError("incomplete hotpot QA training coverage")
    # hotpot episodes appear only in the recurrent rounds (supervision while
    # streaming 39-context episodes), spliced as episodes hotpot-N.
    hotpot_streams = {e["id"] for e in plan["hotpot_episodes"]}
    covered_hotpot_streams = {e["stream"] for e in plan["qa_events"] if e["stream"].startswith("hotpot-")}
    if hotpot_streams != covered_hotpot_streams:
        raise ValueError("hotpot episodes must all receive full curriculum coverage")
    for event in plan["qa_events"]:
        if event["stage"] == "single_merge" and not event["stream"].startswith("hotpot-"):
            if len(event["source_indices"]) != 19:
                raise ValueError("single_merge event must cover 19 contexts")
        if event["stage"] == "recurrent":
            if len(event["source_indices"]) != event["before_write"]:
                raise ValueError("recurrent QA prefix must end at the event")
            if event["before_write"] not in RECURRENT_EVENTS:
                raise ValueError("recurrent event at unexpected write position")
    # Hotpot QA provenance: supervision-only fields intact.
    for ep in plan["episodes"]:
        for c in ep["contexts"]:
            for q in c["qas"]:
                if q["context_sha256"] != c["context_sha256"]:
                    raise ValueError("QA/context provenance hash mismatch")
    for ep in plan["hotpot_episodes"]:
        for c in ep["contexts"]:
            for q in c["qas"]:
                if not q["id"].startswith("hotpot-"):
                    raise ValueError("hotpot QA id prefix missing")
                if len(q["prompt_ids"]) + len(q["answer_ids"]) > 512:
                    raise ValueError("hotpot QA exceeds reader input budget")
                if q["answer_ids"] != q["content_ids"] + [plan["eos_token_id"]]:
                    raise ValueError("hotpot answer/EOS encoding mismatch")


def build_plan(v7_plan, hotpot_payload):
    from prepare_natural_merge_v4 import validate_v4_plan
    original_v4 = v7_plan["original_v4_plan"]
    validate_v4_plan(original_v4)
    if v7_plan["content_sha256"] != digest(
            {k: v for k, v in v7_plan.items() if k != "content_sha256"}):
        raise ValueError("sealed v7 plan file integrity failure")
    sealed_episodes = original_v4["episodes"]
    hotpot_episodes = hotpot_payload["episodes"]
    all_episodes = copy.deepcopy(sealed_episodes) + copy.deepcopy(hotpot_episodes)
    training = freeze_v8_training(build_training(v7_plan["training"]))
    plan = dict(schema=SCHEMA,
                original_v7_plan=v7_plan,
                original_v7_file_content_sha256=v7_plan["content_sha256"],
                original_v4_plan=original_v4,
                base_plan=v7_plan["base_plan"], tokenizer=v7_plan["tokenizer"],
                eos_token_id=v7_plan["eos_token_id"], training=training,
                episodes=all_episodes,
                hotpot_episodes=hotpot_episodes,
                hotpot_provenance=hotpot_payload["provenance"],
                hotpot_parquet_sha256=hotpot_payload["parquet_sha256"],
                qa_events=make_schedule_v8(all_episodes))
    plan["content_sha256"] = digest(plan)
    validate_v8_plan(plan)
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v7-train-plan", required=True)
    p.add_argument("--hotpot-episodes", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    v7_plan = json.loads(Path(args.v7_train_plan).read_text())
    hotpot_payload = json.loads(Path(args.hotpot_episodes).read_text())
    plan = build_plan(v7_plan, hotpot_payload)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(plan, sort_keys=True) + "\n")
    print(json.dumps(dict(schema=plan["schema"], out=args.out,
                          content_sha256=plan["content_sha256"],
                          episodes=len(plan["episodes"]),
                          qa_events=len(plan["qa_events"]),
                          recurrent_event_before_writes=plan["training"]["recurrent_event_before_writes"],
                          trigger_blocks=plan["training"]["trigger_blocks"])))


if __name__ == "__main__":
    main()
