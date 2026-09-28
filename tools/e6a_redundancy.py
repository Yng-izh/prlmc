"""Does every "dropped" probe also have a copy inside the kept blocks?

The mask diagnostics ended inconclusively for the mask, but decisively for the
probes: on the compacted pool, zeroing the dropped blocks changes nothing
(20/20) while zeroing the carriers destroys everything (0/20). So the carriers
answer these questions, which means selection never lost the information.

`probes_from_lines` only requires that the *newest* fact for a (slot, subject)
sits in a dropped block. FactConsolidation repeats subjects, so an older fact for
the same subject -- carrying the same answer -- can sit in a kept block, and the
reader answers from that copy. Such a probe cannot move between before and after
no matter how badly selection behaves.

This checks that text-only, with no GPU: for every candidate probe, whether any
line sharing its (slot, subject) lies in a kept block.
"""
import json
import re
import sys

sys.path.insert(0, "/share/latent/PRLM-C")
sys.path.insert(0, "/tmp")

import pandas as pd  # noqa: E402

PARQUET = "/share/latent/MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet"
ROW = 3
START = 17890
ANCHOR_LINES = 30
LINES_PER_CHUNK = 20
WINDOW_BLOCKS = 19
TARGET_BLOCKS = 9
DROPPED = list(range(10))
KEPT = list(range(10, 19))


def first_match(text):
    templates = [
        (r"^(.+) is a citizen of (.+)$", "What is the country of citizenship of %s?", "1"),
        (r"^The capital of (.+) is (.+)$", "What is the capital of %s?", "1"),
        (r"^(.+) is located in the continent of (.+)$", "What continent is %s located in?", "1"),
        (r"^(.+) is married to (.+)$", "Who is %s married to?", "1"),
        (r"^(.+) died in the city of (.+)$", "In what city did %s die?", "1"),
        (r"^(.+) was born in the city of (.+)$", "In what city was %s born?", "1"),
        (r"^(.+) was founded by (.+)$", "Who founded %s?", "1"),
        (r"^(.+) speaks the language of (.+)$", "What language does %s speak?", "1"),
        (r"^(.+) is affiliated with the religion of (.+)$", "What religion is %s affiliated with?", "1"),
        (r"^(.+) is associated with the sport of (.+)$", "What sport is %s associated with?", "1"),
        (r"^(.+) was performed by (.+)$", "Who performed %s?", "1"),
        (r"^(.+) was developed by (.+)$", "Who developed %s?", "1"),
        (r"^(.+) was created in the country of (.+)$", "In what country was %s created?", "1"),
        (r"^(.+) was written in the language of (.+)$", "In what language was %s written?", "1"),
        (r"^The author of (.+) is (.+)$", "Who is the author of %s?", "1"),
        (r"^(.+) author of (.+) is (.+)$", "Who is the author of %s?", "2"),
        (r"^The director of (.+) is (.+)$", "Who is the director of %s?", "1"),
        (r"^(.+) plays the position of (.+)$", "What position does %s play?", "1"),
        (r"^(.+) was founded in the city of (.+)$", "In what city was %s founded?", "1"),
        (r"^(.+) worked in the city of (.+)$", "In what city did %s work?", "1"),
        (r"^(.+) is employed by (.+)$", "Who is %s employed by?", "1"),
        (r"^(.+) is famous for (.+)$", "What is %s famous for?", "1"),
        (r"^(.+) was created by (.+)$", "Who created %s?", "1"),
    ]
    for pattern, question, group in templates:
        m = re.match(pattern, text)
        if m:
            slots = m.groups()
            if len(slots) >= int(group) + 1:
                return (pattern, slots[0], slots[int(group)], question)
    return None


def main():
    frame = pd.read_parquet(PARQUET)
    context = frame.iloc[ROW]["context"]
    lines = [
        re.sub(r"^[0-9]+\. ", "", l).strip()
        for l in context.split("\n") if re.match(r"^[0-9]+\. ", l)
    ]

    # Block -> line range, mirroring build_window with no gated-out chunks.
    offset = START + ANCHOR_LINES
    line_to_block = {}
    for block in range(WINDOW_BLOCKS):
        for line in range(offset + block * LINES_PER_CHUNK,
                          offset + (block + 1) * LINES_PER_CHUNK):
            line_to_block[line] = block

    window = set()
    for block in range(WINDOW_BLOCKS):
        for line in range(offset + block * LINES_PER_CHUNK,
                          offset + (block + 1) * LINES_PER_CHUNK):
            window.add(line)

    latest = {}
    occurrences = {}
    for index, text in enumerate(lines):
        found = first_match(text)
        if found is None:
            continue
        pattern, subject, answer, _q = found
        key = (pattern, subject)
        latest[key] = (index, answer)
        occurrences.setdefault(key, []).append((index, answer))

    candidates = []
    for index in sorted(window):
        block = line_to_block.get(index)
        if block not in DROPPED:
            continue
        found = first_match(lines[index])
        if found is None:
            continue
        pattern, subject, _answer, question = found
        key = (pattern, subject)
        owner, answer = latest.get(key, (None, None))
        if owner != index:
            continue
        if not (1 <= len(answer.split()) <= 5):
            continue
        copies = occurrences[key]
        in_kept = [
            (i, a) for i, a in copies
            if line_to_block.get(i) in KEPT
        ]
        same_answer = [c for c in in_kept if c[1] == answer]
        candidates.append({
            "line": index, "block": block, "question": question % subject,
            "answer": answer, "n_occurrences": len(copies),
            "copies_in_kept": in_kept,
            "copies_in_kept_same_answer": same_answer,
        })

    # Dedup by question, like probes_from_lines.
    seen = set()
    unique = []
    for c in candidates:
        if c["question"] in seen:
            continue
        seen.add(c["question"])
        unique.append(c)

    leaky = [c for c in unique if c["copies_in_kept_same_answer"]]
    clean = [c for c in unique if not c["copies_in_kept_same_answer"]]
    print("total candidate probes in dropped blocks: %d" % len(unique))
    print("with a same-answer copy in kept blocks:   %d" % len(leaky))
    print("with NO same-answer copy in kept blocks:  %d" % len(clean))
    print()
    print("--- leaky (cannot detect a bad selection) ---")
    for c in leaky[:10]:
        print("  line %d block %d  %s -> %r   kept copies: %s"
              % (c["line"], c["block"], c["question"], c["answer"],
                 [(i, a) for i, a in c["copies_in_kept_same_answer"]]))
    print()
    print("--- clean (a genuine test of the fusion) ---")
    for c in clean[:20]:
        print("  line %d block %d  %s -> %r"
              % (c["line"], c["block"], c["question"], c["answer"]))

    with open("/tmp/e6a_redundancy.json", "w") as handle:
        json.dump({"leaky": leaky, "clean": clean}, handle, indent=2)


if __name__ == "__main__":
    main()
