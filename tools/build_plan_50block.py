#!/usr/bin/env python3
"""Rebuild the natural-merge plan for the 50-block paper geometry.

The sealed v9 plan has 39 writes per episode with geometry 20/13/9 and a trigger at
20 blocks, so with trigger 50 it would never merge. This transform keeps every
context record verbatim -- including the pre-tokenised QA supervision, so
`check_answer_encoding` still holds -- and rebuilds streams long enough for the
student to really perform 8 x (19 -> 9) merges.

Supervision follows the MERGE, not the raw input order: a compaction merges the
oldest `window` resident blocks, and after the first merge those blocks are a mix
of summary blocks (each carrying many originals) and raw blocks. `pool_schedule()`
replays the pool so every event is supervised with the evidence it actually
summarises, and each event records that provenance in `source_indices` so the
trainer can build a realisable local raw-evidence teacher.

Every other top-level plan key (base_plan, hotpot_episodes, provenance, tokenizer,
split audit) is carried over untouched: the objective functions read them.
"""
import hashlib
import json
from pathlib import Path

SRC = Path("data/natural_merge_v9/train.json")
OUT = Path("data/natural_merge_v9_50block/train.json")
EPISODES = 16
CONTEXTS = 125                 # 64k tokens at 512-token writes
WINDOW, TARGET, BUDGET, TRIGGER = 19, 9, 50, 50
QA_CAP = 20                    # sealed events supervise ~20 QAs; keep the backward loop bounded


def pool_schedule(n_writes, budget=BUDGET, window=WINDOW, target=TARGET):
    """write index -> sorted ORIGINAL write indices summarised by that window.

    blocks[0] is the oldest resident block; each entry lists the original writes
    its content came from, and a merged block inherits the union of the window it
    replaced. At the first trigger the window is writes 0..18; afterwards it is 9
    summary blocks (originals 0..k) plus the raw blocks that followed.
    """
    blocks, events = [], {}
    for index in range(n_writes):
        if len(blocks) >= budget:                      # trigger: compact before this write
            provenance = sorted({j for block in blocks[:window] for j in block})
            events[index] = provenance
            blocks = [list(provenance) for _ in range(target)] + blocks[window:]
        blocks.append([index])
    return events


def pick(provenance, cap=QA_CAP):
    """Oldest and newest halves of the provenance.

    A merge keeps `target` of the `window` blocks by importance and drops the
    rest, so sampling both ends keeps the recovery cohort (dropped evidence) and
    the preservation cohort (retained evidence) reachable in every event.
    """
    if len(provenance) <= cap:
        return list(provenance)
    head = provenance[: cap // 2]
    tail = provenance[-(cap - cap // 2):]
    return list(dict.fromkeys(head + tail))


def main():
    src = json.loads(SRC.read_text())
    pool, tokens = {}, {}
    for part in (src["episodes"], src["base_plan"]["episodes"]):
        for ep in part:
            for ctx in ep["contexts"]:
                if ctx["qas"]:                          # supervision needs at least one QA
                    pool.setdefault(ctx["context_sha256"], ctx)
            for write in ep["writes"]:
                tokens.setdefault(write["context_sha256"], write["token_ids"])
    shas = sorted(s for s in pool if s in tokens)
    if len(shas) < 60:
        raise ValueError(f"only {len(shas)} reusable contexts")

    episodes = []
    for k in range(EPISODES):
        contexts = []
        for j in range(CONTEXTS):
            sha = shas[(k * 37 + j * 7) % len(shas)]
            if contexts and sha == contexts[-1]["context_sha256"]:   # no adjacent repeats
                sha = shas[(shas.index(sha) + 1) % len(shas)]
            contexts.append(pool[sha])
        writes = [dict(context_sha256=c["context_sha256"],
                       raw_tokens=len(tokens[c["context_sha256"]]), role="related",
                       source_index=j, token_ids=tokens[c["context_sha256"]])
                  for j, c in enumerate(contexts)]
        episodes.append(dict(id=src["episodes"][k]["id"], contexts=contexts, writes=writes))

    schedule = pool_schedule(CONTEXTS)
    event_writes = sorted(schedule)
    expected_events = [BUDGET + step * (WINDOW - TARGET) for step in range(8)]
    if event_writes != expected_events:
        raise ValueError(f"unexpected compaction schedule {event_writes}")

    def qas_of(indices, episode):
        ids = []
        for index in indices:
            for qa in episode["contexts"][index]["qas"]:
                if qa["id"] not in ids:
                    ids.append(qa["id"])
        return ids[:QA_CAP]

    events = []
    for episode in episodes:
        for rnd, starts in ((0, (0, 20)), (1, (1, 20))):
            for start in starts:
                span = list(range(start, start + WINDOW))
                events.append(dict(qa_ids=qas_of(span, episode), round=rnd,
                                   stage="single_merge", stream=episode["id"],
                                   window_start=start, source_indices=span))
        for rnd in (2, 3):
            for write_index in event_writes:
                provenance = schedule[write_index]
                events.append(dict(qa_ids=qas_of(pick(provenance), episode), round=rnd,
                                   stage="recurrent", stream=episode["id"],
                                   before_write=write_index, source_indices=provenance))

    counts = dict(single_merge=EPISODES * 4, recurrent=EPISODES * 2 * len(event_writes))
    training = dict(src["training"])
    training.update(
        geometry=dict(budget_blocks=BUDGET, window_blocks=WINDOW, target_blocks=TARGET),
        trigger_blocks=TRIGGER, recurrent_event_before_writes=event_writes,
        single_window_contexts=WINDOW, single_window_starts=[[0, 20], [1, 20]],
        streams=EPISODES, planned_updates=sum(counts.values()),
        curriculum_updates=counts, teacher="local_raw_evidence",
        curriculum=(f"rounds 0/1: {counts['single_merge']} isolated {WINDOW}-context windows; "
                    f"rounds 2/3: {CONTEXTS}-write recurrent streams compacted at the "
                    f"{TRIGGER}-block trigger, {len(event_writes)} events per stream, "
                    f"supervised on the merge window's provenance"))

    plan = dict(src)        # base_plan / hotpot_episodes / provenance stay untouched
    plan.update(episodes=episodes, qa_events=events, training=training,
                content_sha256=hashlib.sha256(
                    json.dumps(episodes, sort_keys=True).encode()).hexdigest())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(plan))
    print(json.dumps(dict(episodes=len(episodes), writes=len(episodes[0]["writes"]),
                          qa_events=len(events), curriculum=counts, events=event_writes,
                          provenance_sizes={k: len(v) for k, v in list(schedule.items())[:3]},
                          keeps_hotpot="hotpot_episodes" in plan,
                          plan_bytes=OUT.stat().st_size)))


if __name__ == "__main__":
    main()
