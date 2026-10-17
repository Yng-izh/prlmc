"""Read an observation run's stats and report the signal distributions.

The thresholds in prlmc_8b.yaml are placeholders. This turns the raw per-write
records from `prlmc_stats.jsonl` into the numbers a threshold should be set from:
quantiles of novelty, occupancy, effective rank and energy rank, plus the first
writes of each episode, which are the only observations of the pool before it is
full.

Two things the output is meant to settle:

  * The scale of `feature_novelty`. `tau_n = 0.50` is only meaningful if novelty
    is a ratio in [0, 1] (a projection-residual share of the candidate's energy).
    If the printed range is not within [0, 1], the threshold has to be rescaled
    rather than the write path changed.

  * Whether the trigger would ever fire. `occupancy_novelty_rank` requires
    occupancy above `tau_k` and novelty below `tau_n` for `trigger_consecutive`
    writes. Printing occupancy against the nominal 0.80 and novelty against the
    nominal 0.50 shows how much of the run each condition covers.

Usage:  python calibrate_obs.py <stats.jsonl> [more.jsonl ...]
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict

# The thresholds the main config currently ships with, printed alongside the
# measurements so the gap between guess and observation stays visible.
NOMINAL = {"tau_k": 0.80, "tau_n": 0.885, "tau_k_exit": 0.60, "tau_n_exit": 0.880}

QUANTILES = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)


def quantile(values, q):
    """Linear-interpolation quantile; avoids a numpy dependency here."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def load_writes(paths):
    """Pull every per-write record out of the drained stats lines."""
    writes = []
    episodes = defaultdict(list)
    lines = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except ValueError:
                    continue
                lines += 1
                key = (payload.get("sub_dataset"), payload.get("agent_name"))
                for record in payload.get("writes_since_last_query") or []:
                    record["_key"] = key
                    writes.append(record)
                    episodes[key].append(record)
    return writes, episodes, lines


def describe(name, values, indent="  "):
    if not values:
        print("%s%-18s (no data)" % (indent, name))
        return
    parts = []
    for q in QUANTILES:
        parts.append("%s=%.4g" % (
            {0.0: "min", 0.10: "p10", 0.25: "p25", 0.50: "p50",
             0.75: "p75", 0.90: "p90", 1.0: "max"}[q], quantile(values, q)))
    print("%s%-18s n=%-6d %s" % (indent, name, len(values), "  ".join(parts)))


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2

    writes, episodes, lines = load_writes(argv[1:])
    if not writes:
        print("no per-write records found in %d stats line(s)" % lines)
        print("check that the run used the config with stats_dir set and that the")
        print("observation build of layout.py (drain_history) is the one installed")
        return 1

    print("stats lines: %d   episodes: %d   writes: %d\n" % (lines, len(episodes), len(writes)))

    print("== all writes ==")
    for field in ("novelty", "gate", "occupancy", "effective_rank", "energy_rank",
                  "rank_utilisation", "used_slots", "tokens", "encode_seconds"):
        values = [r[field] for r in writes if isinstance(r.get(field), (int, float))]
        describe(field, values)

    print("\n== first 36 writes per episode (pool before it is near full) ==")
    early = []
    for records in episodes.values():
        early.extend(records[:36])
    for field in ("novelty", "occupancy", "effective_rank", "energy_rank"):
        describe(field, [r[field] for r in early if isinstance(r.get(field), (int, float))])

    print("\n== write path health ==")
    dropped = sum(1 for r in writes if r.get("dropped"))
    evicted = sum(1 for r in writes if (r.get("evicted_blocks") or 0) > 0)
    compacted = sum(1 for r in writes if r.get("compacted"))
    print("  dropped writes      %d" % dropped)
    print("  writes that evicted %d" % evicted)
    print("  writes that compacted %d" % compacted)
    if evicted:
        print("  NOTE: eviction means the pool filled and consolidation did not free")
        print("        space. This run has compact_mode=none so that is expected;")
        print("        it is the condition the real run must avoid.")

    print("\n== threshold coverage (placeholder values) ==")
    occ = [r["occupancy"] for r in writes if isinstance(r.get("occupancy"), (int, float))]
    nov = [r["novelty"] for r in writes if isinstance(r.get("novelty"), (int, float))]
    if occ:
        above = sum(1 for v in occ if v > NOMINAL["tau_k"])
        print("  occupancy > %.2f (tau_k): %d/%d = %.1f%%" % (
            NOMINAL["tau_k"], above, len(occ), 100.0 * above / len(occ)))
    if nov:
        above_n = sum(1 for v in nov if v >= NOMINAL["tau_n"])
        print("  novelty  >= %.2f (tau_n): %d/%d = %.1f%%" % (
            NOMINAL["tau_n"], above_n, len(nov), 100.0 * above_n / len(nov)))
    if occ and nov and len(occ) == len(nov):
        both = sum(1 for o, n in zip(occ, nov) if o >= NOMINAL["tau_k"] and n >= NOMINAL["tau_n"])
        print("  both conditions at once:  %d/%d = %.1f%%  (trigger needs %d in a row)" % (
            both, len(occ), 100.0 * both / len(occ), 5))

    print("\n== per episode ==")
    for (sub, agent), records in sorted(episodes.items(), key=lambda kv: str(kv[0])):
        final = records[-1]
        novs = [r["novelty"] for r in records if isinstance(r.get("novelty"), (int, float))]
        print("  %s / %s" % (sub, agent))
        print("    writes=%d  final_used=%s  final_occupancy=%s  final_effrank=%s" % (
            len(records), final.get("used_slots"), final.get("occupancy"),
            final.get("effective_rank")))
        if novs:
            print("    novelty min=%.4g p50=%.4g max=%.4g" % (
                min(novs), quantile(novs, 0.5), max(novs)))

    # The scale check is a hard failure of the placeholder, not a tuning hint:
    # tau_n is compared directly against feature_novelty's return value.
    nov_all = [r["novelty"] for r in writes if isinstance(r.get("novelty"), (int, float))]
    if nov_all and (min(nov_all) < -1e-6 or max(nov_all) > 1.0 + 1e-6):
        print("\nWARNING: novelty leaves [0, 1] (min=%.4g max=%.4g). tau_n as written"
              % (min(nov_all), max(nov_all)))
        print("         is not on the same scale as this signal; rescale before using it.")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
