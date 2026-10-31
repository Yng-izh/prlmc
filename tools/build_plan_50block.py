#!/usr/bin/env python3
"""Rebuild the natural-merge plan for the 50-block paper geometry.

The sealed v9 plan has 39 writes per episode with geometry 20/13/9 and a trigger at
20 blocks, so with trigger 50 it would never merge. This transform keeps every
context record verbatim -- including the pre-tokenised QA supervision, so
`check_answer_encoding` still holds -- and rebuilds streams long enough for the
student to really perform 8 x (19 -> 9) merges.

Every other top-level plan key (base_plan, hotpot_episodes, provenance, tokenizer,
split audit) is carried over untouched: the objective functions read them.
"""
import hashlib
import json
from pathlib import Path

SRC = Path("data/natural_merge_v9/train.json")
OUT = Path("data/natural_merge_v9_50block/train.json")
EPISODES = 16
CONTEXTS = 125                      # 64k tokens at 512-token writes
WINDOW, TARGET, BUDGET, TRIGGER = 19, 9, 50, 50
EVENT_WRITES = [BUDGET + step * (WINDOW - TARGET) for step in range(8)]   # 50..120


def main():
    src = json.loads(SRC.read_text())
    pool, tokens = {}, {}
    for plan_part in (src["episodes"], src["base_plan"]["episodes"]):
        for ep in plan_part:
            for ctx in ep["contexts"]:
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
            if contexts and sha == contexts[-1]["context_sha256"]:
                sha = shas[(shas.index(sha) + 1) % len(shas)]
            contexts.append(pool[sha])
        writes = [dict(context_sha256=c["context_sha256"],
                       raw_tokens=len(tokens[c["context_sha256"]]), role="related",
                       source_index=j, token_ids=tokens[c["context_sha256"]])
                  for j, c in enumerate(contexts)]
        episodes.append(dict(id=src["episodes"][k]["id"], contexts=contexts, writes=writes))

    def qa_ids(window):
        return [q["id"] for ctx in window for q in ctx["qas"]]

    events = []
    for ep in episodes:
        for rnd, start in ((0, 0), (1, 1)):
            window = ep["contexts"][start:start + WINDOW]
            events.append(dict(qa_ids=qa_ids(window), round=rnd, stage="single_merge",
                               stream=ep["id"], window_start=start,
                               source_indices=list(range(start, start + WINDOW))))
        for rnd in (2, 3):
            for write_index in EVENT_WRITES:
                low = max(0, write_index - WINDOW)
                window = ep["contexts"][low:write_index]
                events.append(dict(qa_ids=qa_ids(window), round=rnd, stage="recurrent",
                                   stream=ep["id"], before_write=write_index,
                                   source_indices=list(range(low, write_index))))

    training = dict(src["training"])
    training.update(
        geometry=dict(budget_blocks=BUDGET, window_blocks=WINDOW, target_blocks=TARGET),
        trigger_blocks=TRIGGER, recurrent_event_before_writes=EVENT_WRITES,
        single_window_contexts=WINDOW, single_window_starts=[[0, 20], [1, 20]],
        streams=EPISODES,
        curriculum=(f"rounds 0/1: two isolated {WINDOW}-context windows (supervised "
                    f"{WINDOW}->{TARGET} merge); rounds 2/3: {CONTEXTS}-write recurrent "
                    f"streams compacted at the {TRIGGER}-block trigger (8 events per stream)"))

    plan = dict(src)          # keep base_plan / hotpot_episodes / provenance untouched
    plan.update(episodes=episodes, qa_events=events, training=training,
                content_sha256=hashlib.sha256(
                    json.dumps(episodes, sort_keys=True).encode()).hexdigest())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(plan))
    print(json.dumps(dict(episodes=len(episodes), writes=len(episodes[0]["writes"]),
                          qa_events=len(events), geometry=training["geometry"],
                          trigger=training["trigger_blocks"],
                          events=training["recurrent_event_before_writes"],
                          keeps_hotpot="hotpot_episodes" in plan,
                          plan_bytes=OUT.stat().st_size)))


if __name__ == "__main__":
    main()
