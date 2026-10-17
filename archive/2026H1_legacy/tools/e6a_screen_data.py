"""Offline screen for the hard-window E6a fixture. Text only: no model, no GPU.

Why this exists
---------------
The first E6a saturated (16/16 before, 16/16 after) because its probes were not
required to come from blocks that `select_blocks` actually drops. Fixing that
needs a window where selection provably loses answerable facts, and searching for
such a window by running the reader would be needlessly expensive.

Almost all of the search is text. A probe qualifies only if its answer's *latest*
matching fact is inside the window (otherwise any compaction keeps the answer) and
that fact sits in a block that selection is likely to drop. Both conditions are
decidable from the fact lines alone, and the block a line lands in is fixed
arithmetic: each write contributes exactly `num_tokens` = 256 slots = one block,
and chunks are `LINES_PER_CHUNK` lines, so line `i` is in block `(i - start) // 20`.

What this cannot decide
-----------------------
Whether a specific block is *actually* dropped. `_select_block_indices` mixes an
importance term (weight 0.4) with recency (0.3), and importance depends on the
written memory, so only the recency ordering is known here. The counts below are
therefore a *screen*: they rank candidate windows so the GPU confirmation can be
spent on the few that can plausibly show a gap.

Run:
    /share/latent/mllm_env/bin/python tools/e6a_screen_data.py --top 12
"""

import argparse
import json
import os
import re
import sys

PARQUET = "/share/latent/MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet"

ANCHOR_LINES = 30
LINES_PER_CHUNK = 20
WINDOW_BLOCKS = 19
TARGET_BLOCKS = 9
WINDOW_LINES = WINDOW_BLOCKS * LINES_PER_CHUNK   # 380

# Reuse the driver's templates so the screen selects exactly the probes the run
# would build. Importing it pulls torch, which is slow but keeps one definition.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e6a_overfit import first_match  # noqa: E402


def read_rows():
    import pandas as pd

    frame = pd.read_parquet(PARQUET)
    rows = []
    for i in range(len(frame)):
        context = frame.iloc[i]["context"]
        lines = [l for l in context.split("\n") if re.match(r"^[0-9]+\. ", l)]
        rows.append([re.sub(r"^[0-9]+\. ", "", l).strip() for l in lines])
    return rows


def qualifying(lines, start, latest):
    """Probes in this window whose supporting fact is the row's latest match.

    Returns (per_block_counts, probes). `probes` carries the block index so the
    GPU stage can cross-check it against the real dropped set.
    """
    per_block = [0] * WINDOW_BLOCKS
    probes = []
    seen = set()
    for index in range(start, start + WINDOW_LINES):
        block = (index - start) // LINES_PER_CHUNK
        found = first_match(lines[index])
        if found is None:
            continue
        slot, subject, _answer, question = found
        owner, answer = latest.get((slot, subject), (None, None))
        # Only the latest matching fact is the correct answer. An earlier line is
        # superseded, so probing it would mark a correct reader wrong.
        if owner != index:
            continue
        try:
            filled = question % subject
        except TypeError:
            continue
        if filled in seen or not (1 <= len(answer.split()) <= 5):
            continue
        seen.add(filled)
        per_block[block] += 1
        probes.append({"line": index, "block": block, "q": filled, "a": answer})
    return per_block, probes


def latest_matches(lines):
    latest = {}
    for index, text in enumerate(lines):
        found = first_match(text)
        if found is None:
            continue
        slot, subject, answer, _question = found
        latest[(slot, subject)] = (index, answer)
    return latest


def scan_row(row, lines):
    latest = latest_matches(lines)
    windows = []
    start = ANCHOR_LINES
    while start + WINDOW_LINES <= len(lines):
        per_block, probes = qualifying(lines, start, latest)
        if probes:
            windows.append({
                "row": row,
                "start": start,
                "per_block": per_block,
                "total": len(probes),
                # Selection keeps TARGET_BLOCKS = 9 of 19, so at least 10 blocks
                # go. Recency scores the oldest at 0.0 and the newest at 1.0, so
                # the head of the window is the part most likely to be dropped.
                "head10": sum(per_block[:10]),
                "head13": sum(per_block[:13]),
                "probes": probes,
            })
        start += WINDOW_LINES
    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--out", default="/tmp/e6a_screen_data.json")
    args = parser.parse_args()

    rows = read_rows()
    # Rows 4-7 repeat rows 0-3 byte for byte (same context lengths), so only the
    # first four carry distinct fact sets.
    distinct = []
    for i, lines in enumerate(rows[:4]):
        distinct.append((i, lines))

    all_windows = []
    for row, lines in distinct:
        found = scan_row(row, lines)
        all_windows.extend(found)
        print("[screen] row %d: %d lines, %d windows with probes" % (row, len(lines), len(found)), flush=True)

    if not all_windows:
        raise SystemExit("no window produced any qualifying probe")

    ranked = sorted(all_windows, key=lambda w: (w["head10"], w["total"]), reverse=True)

    print()
    print("window ranking (head10 = qualifying probes in the 10 oldest blocks)")
    print("%-4s %-7s %-7s %-7s  %s" % ("row", "start", "total", "head10", "per-block counts (oldest -> newest)"))
    for w in ranked[: args.top]:
        print("%-4d %-7d %-7d %-7d  %s" % (
            w["row"], w["start"], w["total"], w["head10"],
            " ".join("%2d" % c for c in w["per_block"]),
        ))

    best = ranked[0]
    print()
    print("best window: row %d, lines %d-%d" % (best["row"], best["start"], best["start"] + WINDOW_LINES - 1))
    print("  %d qualifying probes, %d in the oldest 10 blocks" % (best["total"], best["head10"]))
    print("  probes in the oldest 5 blocks: %d" % sum(best["per_block"][:5]))
    for p in best["probes"][: args.top]:
        print("    block %2d  %-58s -> %s" % (p["block"], p["q"], p["a"]))

    json.dump(
        {
            "ranking": [{k: v for k, v in w.items() if k != "probes"} for w in ranked],
            "best": best,
            "all": all_windows,
        },
        open(args.out, "w"),
        indent=2,
    )
    print()
    print("[screen] written to %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
