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

from io_utils import digest  # noqa: E402

SCHEMA = "natural_merge_v9_training_plan/1"
TRAIN_V9 = None
RECURRENT_EVENTS = [20, 24]
TRIGGER_BLOCKS = 20
SINGLE_STARTS = [[0, 20], [1, 20]]
SINGLE_WINDOW_CONTEXTS = 19
ARCHITECTURE = "grouped_gated_sparse_native_v9"
# 40 of 256 slots per carrier are bridges (bridge_fraction .156) and the core 216
# are hard-masked to exactly zero residual, so a whole-block budget of .08 caps the
# bridge amplitude at .08/sqrt(.156) ~ .20 and makes bridge_gamma_max .60 dead code.
# .24/sqrt(.156) ~ .608 >= .60, so the bridge cap becomes the active bound while the
# core slots stay untouched either way.
GLOBAL_RMS_BUDGET = .24
BRIDGE_BLOCKS = 3
NGRAM_KERNEL = (.25, .5, .25)
NATIVE_INIT = 0.


def make_schedule_v8(episodes):
    """Event schedule of the sealed curriculum: 2 rounds of isolated windows, then
    recurrent events before writes 20 and 24, with the QAs of every context split
    over all of its event appearances in order."""
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


def event_qas(plan, rnd, episode, stage, position):
    """The supervised QAs of exactly one training event."""
    key = "window_start" if stage == "single_merge" else "before_write"
    matches = [e for e in plan["qa_events"] if e["round"] == rnd and e["stream"] == episode["id"]
               and e["stage"] == stage and e[key] == position]
    if len(matches) != 1:
        raise ValueError("training event missing or duplicated")
    qas = {q["id"]: q for c in episode["contexts"] for q in c["qas"]}
    return [qas[i] for i in matches[0]["qa_ids"]]


def check_answer_encoding(plan, tokenizer):
    """The live tokenizer must reproduce the sealed answer/EOS boundary exactly."""
    if tokenizer.eos_token_id != plan["eos_token_id"]:
        raise ValueError("actual EOS differs from sealed plan")
    for ep in plan["episodes"]:
        for c in ep["contexts"]:
            for q in c["qas"]:
                ids = tokenizer(" " + q["answer"], add_special_tokens=False).input_ids
                if q["answer_ids"] != ids + [tokenizer.eos_token_id] or q["content_ids"] != ids:
                    raise ValueError("content boundary/EOS encoding mismatch")


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


def validate_paper_schedule(plan, training):
    """Structural contract of the 50-block event schedule.

    The sealed curriculum is regenerated by `make_schedule_v8` and must match it
    exactly; the paper stream supervises the merge provenance instead, so it is
    checked directly: one event per (round, stream, stage, position), every
    supervised QA belongs to that stream, and each recurrent event carries both
    the window provenance the merge may drop and the tail it never touches.
    """
    episodes = {e["id"]: e for e in plan["episodes"]}
    events = plan["qa_events"]
    counts = training.get("curriculum_updates") or {}
    if len(events) != training["planned_updates"]:
        raise ValueError("event count differs from the planned update budget")
    if counts:
        observed = [sum(1 for e in events if e["stage"] == stage) for stage in counts]
        if observed != list(counts.values()):
            raise ValueError("event stages differ from the declared curriculum")
    qa_by_stream = {sid: {q["id"] for c in e["contexts"] for q in c["qas"]}
                    for sid, e in episodes.items()}
    keys = set()
    for event in events:
        stream = event["stream"]
        if stream not in episodes:
            raise ValueError(f"event for unknown stream {stream!r}")
        position = (event.get("before_write") if event["stage"].startswith("recurrent")
                    else event.get("window_start"))
        key = (event["round"], stream, event["stage"], position)
        if key in keys:
            raise ValueError(f"duplicated training event {key}")
        keys.add(key)
        ids = event["qa_ids"]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(f"event {key} has empty or duplicated QAs")
        if not set(ids).issubset(qa_by_stream[stream]):
            raise ValueError(f"event {key} supervises QAs from another stream")
        if event["stage"].startswith("recurrent"):
            if position not in training["recurrent_event_before_writes"]:
                raise ValueError(f"event {key} is not at a declared compaction")
            if not event.get("source_indices") or not event.get("tail_indices"):
                raise ValueError(f"event {key} lacks window/tail provenance")
        elif position + training["single_window_contexts"] > len(episodes[stream]["contexts"]):
            raise ValueError(f"event {key} reads past the stream")
    return True


def validate_v9_plan(plan, base=None):
    if plan.get("schema") != SCHEMA:
        raise ValueError("v9 schema required")
    training = plan["training"]
    # Two geometries are legitimate: the sealed v9 20/13->9 training set and the
    # paper's 50-block pool, where the stream first fills 50 blocks and every
    # 19->9 merge releases 10. The update budget scales with the event count
    # (4 isolated windows + 8 compactions per episode over 2 rounds each).
    geometries = {
        (20, 13, 9): dict(trigger=TRIGGER_BLOCKS, events=RECURRENT_EVENTS, updates=128),
        # Paper stream: 246 writes, so the trigger fires at write 50 and then every
        # 19->9 merge releases 10 -- 20 compactions (writes 50..240). 4 isolated
        # windows + 2 rounds x 20 compactions over 16 streams = 704 updates.
        (50, 19, 9): dict(trigger=50, events=[50 + 10*i for i in range(20)],
                          updates=16*(4 + 2*20)),
    }
    key = (training["geometry"]["budget_blocks"], training["geometry"]["window_blocks"],
           training["geometry"]["target_blocks"])
    if key not in geometries:
        raise ValueError("unsupported merge geometry (sealed 20/13->9 or paper 50/19->9)")
    expected = geometries[key]
    if training["trigger_blocks"] != expected["trigger"]:
        raise ValueError("trigger must be 20 (sealed) or 50 (paper)")
    if training["recurrent_event_before_writes"] != expected["events"]:
        raise ValueError("recurrent events must match the sealed or paper schedule")
    if training["single_window_contexts"] != SINGLE_WINDOW_CONTEXTS:
        raise ValueError("single-merge curriculum stays sealed")
    if training["single_window_starts"] != SINGLE_STARTS:
        raise ValueError("single-merge window starts stay sealed")
    if training["planned_updates"] != expected["updates"] or training["streams"] != 16:
        raise ValueError("planned updates must be 128 (sealed) or 704 (paper) over 16 streams")
    if (training["token_protection_weight"], training["token_kl_weight"]) != (0., 0.):
        raise ValueError("v9 disables the per-token protection term")
    if training["global_rms_budget"] != GLOBAL_RMS_BUDGET:
        raise ValueError("v9 budget must be the bridge-scaled .24")
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
    if key == (50, 19, 9):
        validate_paper_schedule(plan, training)
    else:
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


def sealed_training(content_sha256=None):
    """Training record of the sealed plan file a checkpoint names.

    The plan files are data artifacts and are not shipped in git. When the named
    file is absent (or its content hash does not match) the caller falls back to
    the record embedded in the checkpoint, so validation never silently compares
    against a different generation's record.
    """
    candidates = [ROOT / "data" / "natural_merge_v9" / "train.json"]
    block_dir = ROOT / "data" / "natural_merge_v9_50block"
    if block_dir.is_dir():
        candidates += sorted(block_dir.glob("train*.json"))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            plan = json.loads(path.read_text())
        except ValueError:
            continue
        if plan.get("schema") != SCHEMA:
            continue
        if content_sha256 is not None and plan.get("content_sha256") != content_sha256:
            continue
        return plan["training"]
    return None


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
