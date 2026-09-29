"""E6b-mini: does one shared carrier-residual checkpoint generalise across hard windows?

The one question this answers, and the only one:

    E6a showed the compactor can recover dropped facts on a *single* hard window.
    Is it learning compression, or is it learning that window?

"hard window" keeps E6a's definition verbatim: the intact pool answers a probe, native
`select_blocks` loses it. Nothing here redefines hardness, and no ordinary window is
added to make the counts look better.

What is deliberately NOT changed
--------------------------------
Structure, loss, selector, trigger and gate are all frozen. This is the same
`CarrierResidualCompactor`, the same `_stage_weights`, the same `_select_block_indices`,
the same 19 -> 9 shape, the same `lr`, the same `gamma_max`. The only variable against
E6a is *how many windows* the shared checkpoint sees. That is what makes a negative
result readable: if held-out recovery is zero here, the cause is the data distribution,
not a simultaneous objective change.

Three design decisions worth stating, because each one is load-bearing
--------------------------------------------------------------------
1. **Stage A also runs across windows.** The spec fixes Stage B's loss; it says nothing
   about Stage A, which is the "do not break the reader" warm-up. Running it on one
   window would make it a single-window prefit and the "one shared checkpoint" claim
   would only hold for Stage B. Both stages iterate all training windows, splitting
   their step budget evenly.

2. **Held-out windows are scored with a larger probe set than training.** Evaluation is
   forward-only, so a wide probe set is cheap; training pays for every probe in every
   step. E6a's denominator was 3 lost probes, which quantises `recovery_rate` to
   0, 1/3, 2/3, 1 and cannot resolve "generalises" from "overfits". Train windows keep
   E6a's probe count so per-step cost is unchanged; held-out windows use a wider set so
   the denominator can actually carry a rate.

3. **Training and held-out windows come from different parquet rows.** The file has 8
   rows but only 4 distinct contexts: rows 0/4, 1/5, 2/6 and 3/7 are byte-identical.
   Splitting inside one row would make "held-out" a relabelling of the training text.
   The partitioner therefore puts held-out windows in rows that no training window uses,
   and additionally requires the *fact-line ranges* of any two selected windows in the
   same row to be disjoint.

Metrics
-------
Per window and aggregated:

    eligible  the pool was required: wrong with the pool zeroed, right intact
    lost      eligible and then lost by native selection      (the denominator)
    recovered lost and answered again by the trained carrier  (the numerator)

Reported separately for training windows and held-out windows. The training figure is
recorded as context; `verdict` is decided on the held-out figure alone.

Run on the host (screening + 32 + 192 training steps + two evaluation passes):
    /share/latent/mllm_env/bin/python /tmp/e6b_mini.py \
        --steps-a-per-window 4 --steps-b-per-window 24 \
        --train-windows 8 --eval-windows 2 --out /tmp/e6b_mini_report.json
"""

import argparse
import hashlib
import json
import os
import sys
import time

# The GPU pick must happen before torch is imported so CUDA_VISIBLE_DEVICES takes
# effect. This mirrors `e6a_overfit`; both tools share the same helper by import, but
# the `execv` has to be triggered from this module because the other one only does it
# under its own `__main__`.
TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = "/share/latent/PRLM-C"
for path in ("/tmp", TOOLS, REPO):
    if path not in sys.path:
        sys.path.insert(0, path)


def _reexec_on_free_gpu():
    import subprocess

    need_mb = int(os.environ.get("E6B_NEED_MB", os.environ.get("E6A_NEED_MB", "21000")))
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    free = []
    for line in out.strip().splitlines():
        index, used = [int(v) for v in line.split(",")]
        free.append((32768 - used, index))
    free.sort(reverse=True)
    if free[0][0] < need_mb:
        raise SystemExit(
            "no GPU with %d MB free; best is gpu%d with %d MB" % (need_mb, free[0][1], free[0][0])
        )
    print("[e6b] gpu%d, %d MB free" % (free[0][1], free[0][0]), flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(free[0][1])
    os.environ["E6A_GPU_SET"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


if __name__ == "__main__" and not os.environ.get("E6A_GPU_SET"):
    _reexec_on_free_gpu()

import torch  # noqa: E402

import e6a_overfit as E  # noqa: E402  (probe construction, scoring, window building)

from methods.prlmc.consolidate import CarrierResidualCompactor, select_block_compact  # noqa: E402
from methods.prlmc.config import PRLMCConfig  # noqa: E402
from methods.prlmc.layout import ProtectedResidualMemory  # noqa: E402
from methods.prlmc.probes import train_carrier_compactor  # noqa: E402

from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM  # noqa: E402

DEVICE = "cuda:0"


def context_hash(lines):
    """Short digest of a window's fact lines.

    The parquet has 8 rows but only 4 distinct contexts (0/4, 1/5, 2/6, 3/7 are
    byte-identical). Printing the digest makes a duplicate row visible in the log
    instead of silently inflating the apparent diversity of the training set.
    """
    return hashlib.md5("\n".join(lines).encode()).hexdigest()[:8]


# --- window construction -----------------------------------------------------

def release():
    """Return cached CUDA blocks to the driver.

    Not housekeeping, and not a guess. E6a trained 236 steps on one window and finished
    inside the same memory envelope E6b-mini died in, so the envelope is not the problem
    -- the allocator's block layout is. Each window here has a *different* probe set
    (probes are built from that window's fact lines), so the token lengths and therefore
    the activation shapes change at every window boundary, and blocks cached for one
    window stop fitting the next window's requests.

    The failure is arithmetic, not vague: PyTorch held 766.85 MiB of unallocated cache
    against a 3.12 GiB request with 3.12 GiB free. The cache was the entire shortfall.
    Clearing at window boundaries restores the steady state E6a ran in, where the
    allocator sees one shape family and never fragments.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def rebuild(manager, lines, start):
    """Re-anchor and re-write one 19-block window; return its coverage and line map.

    `build_window` calls `initialize_anchor`, which calls `reset`, so every rebuild
    starts from a zeroed pool. That is what makes it safe to hold several windows in
    one process: no window inherits pool state from the previous one.

    Returns `(covered, indices, line_to_block)`, or None when this start cannot fill a
    19-block window (the fact list ran out, or the write path compacted).
    """
    window_slots = E.WINDOW_BLOCKS * manager.block
    covered = E.write_until(manager, lines, window_slots, start)
    if covered is None or manager.used != window_slots:
        return None
    indices = E.probe_indices(covered, window_slots)
    if not indices:
        return None
    line_to_block = {}
    for block_index, (lo, hi, _before, _width) in enumerate(covered):
        for line in range(lo, hi):
            line_to_block[line] = block_index
    return covered, indices, line_to_block


def dropped_blocks_of(manager, window_slots):
    """The blocks native selection discards for the window currently in the pool."""
    begin = manager.total - manager.used
    source = manager._memory[:, begin : begin + window_slots, :].to(device=DEVICE)
    weights = manager._window_importance(begin, window_slots)
    dropped = E.dropped_block_indices(manager, source, weights)
    del source, weights
    torch.cuda.empty_cache()
    return dropped


def measure_window(manager, lines, row, start, compactor, n_probes, args, log_prefix="[e6b]"):
    """Build a window and measure every quantity the acceptance rule needs.

    One call = one fresh window state, so `no_memory`, `uncompacted` and `before` all
    describe the same pool. Returns a dict, or None when the window is unusable.
    """
    window_slots = E.WINDOW_BLOCKS * manager.block
    built = rebuild(manager, lines, start)
    if built is None:
        return None
    covered, indices, line_to_block = built

    dropped = dropped_blocks_of(manager, window_slots)
    probes = E.probes_from_lines(
        lines, indices, n_probes,
        dropped_blocks=set(dropped), line_to_block=line_to_block,
    )
    if len(probes) < args.min_probes:
        print("%s row %d start %-6d dropped %-2d probes %-3d (too few candidates)"
              % (log_prefix, row, start, len(dropped), len(probes)), flush=True)
        return None

    with torch.no_grad():
        no_mem = E.score_no_memory(manager, probes, args.max_probe_tokens)
        full = E.score_uncompacted(manager, probes, args.max_probe_tokens)
        # `candidate_pool` runs the compactor forward. Outside `no_grad` it builds a graph
        # that nothing ever frees; E6a leaked ~3.8 GB this way at the identity check and
        # the trainer then had no headroom. Measurement never needs gradients.
        pool, mask = E.candidate_pool(manager, compactor, DEVICE)
        before = E.score_probes(manager, probes, pool, args.max_probe_tokens, mask=mask)
    del pool, mask
    torch.cuda.empty_cache()

    # `before` is passed twice: this call exists for its `n_lost` (which depends only on
    # `full` and `before`), not for a recovery count. Recovery is computed against the
    # trained state in `evaluate_window`.
    gap = E.recovery_report(full, before, before, no_memory=no_mem)
    return {
        "row": row,
        "start": start,
        "context_hash": context_hash(lines),
        "dropped_blocks": dropped,
        "probes": probes,
        "line_set": set(indices),
        "n_probes": len(probes),
        "no_memory": no_mem,
        "uncompacted": full,
        "before": before,
        "n_eligible": gap["n_eligible"],
        "n_lost": gap["n_lost"],
        "lost_questions": gap["lost_questions"],
    }


def evaluate_window(manager, lines, entry, compactor, args):
    """Re-build a measured window and score it with the *trained* compactor.

    `no_memory`, `uncompacted` and `before` are taken from `entry` rather than
    re-measured. All three are deterministic functions of the window and the
    *pre-training* compactor: the rebuild re-anchors and re-writes the same fact lines
    from the same start, so the pool is bit-identical and re-running them would spend
    two passes per window to reproduce numbers that are already in hand. `before` in
    particular must stay the pre-training measurement, because it is the denominator
    `lost` is defined against.

    Only `after` is new, so only one scoring pass is needed here.
    """
    built = rebuild(manager, lines, entry["start"])
    if built is None:
        raise SystemExit("window row %d start %d no longer builds; pool state drifted"
                         % (entry["row"], entry["start"]))
    with torch.no_grad():
        pool, mask = E.candidate_pool(manager, compactor, DEVICE)
        after = E.score_probes(manager, entry["probes"], pool, args.max_probe_tokens, mask=mask)
    del pool, mask
    torch.cuda.empty_cache()
    recovery = E.recovery_report(
        entry["uncompacted"], entry["before"], after, no_memory=entry["no_memory"]
    )
    return {"recovery": recovery, "after": after}


# --- partitioning ------------------------------------------------------------

def overlaps(a, b):
    return bool(a & b)


def partition(hard, args, require_other_rows=False):
    """Split screened hard windows into training and held-out groups.

    Two constraints, both about making held-out mean something:

      * held-out rows must differ from every training row, because the parquet's 8 rows
        are only 4 distinct contexts (0/4, 1/5, 2/6, 3/7 are byte-identical) and a
        same-row split would evaluate on training text;
      * within a row, no two selected windows may share a fact line, so a held-out
        window cannot inherit a fact that a training window already taught.

    Training takes the row with the most hard windows, so the pool the checkpoint sees
    is as varied as the screen allows. Held-out is taken from other rows, best-ranked
    first.

    `require_other_rows=True` drops the fallback described below and returns an empty
    held-out list when no other row qualifies. The screen calls it that way: with the
    fallback enabled, screening would stop the moment one row had enough windows and
    silently produce a same-row held-out set. The final call uses the fallback, so a run
    that genuinely cannot find a second row still produces a (weaker, honestly labelled)
    result instead of nothing.
    """
    by_row = {}
    for entry in hard:
        by_row.setdefault(entry["row"], []).append(entry)

    train_row = max(by_row, key=lambda r: (len(by_row[r]), -r))
    train_rows = [train_row]

    train, used_lines = [], {}
    for entry in by_row[train_row]:
        if len(train) >= args.train_windows:
            break
        row = entry["row"]
        if overlaps(used_lines.setdefault(row, set()), entry["line_set"]):
            continue
        used_lines[row] |= entry["line_set"]
        train.append(entry)

    train_line_sets = [e["line_set"] for e in train]
    eval_rows = [r for r in sorted(by_row) if r not in train_rows]
    row_separated = True
    held = []
    for row in eval_rows:
        for entry in by_row[row]:
            if len(held) >= args.eval_windows:
                break
            if any(overlaps(entry["line_set"], s) for s in train_line_sets):
                continue
            held.append(entry)
        if len(held) >= args.eval_windows:
            break

    if len(held) < args.eval_windows and not require_other_rows:
        # No other row produced a hard window. A disjoint window from the training row
        # is still a valid held-out *fact* test, but it is weaker, and the report must
        # say so rather than quietly presenting it as an independent context.
        row_separated = False
        for entry in by_row[train_row]:
            if len(held) >= args.eval_windows:
                break
            if entry in train:
                continue
            if any(overlaps(entry["line_set"], s) for s in train_line_sets):
                continue
            held.append(entry)

    return {
        "train": train,
        "held": held,
        "train_row": train_row,
        "held_rows": sorted({e["row"] for e in held}),
        "row_separated": row_separated,
    }


def aggregate(entries):
    """Sum `lost` and `recovered` over windows.

    Careful, and deliberately not a mean of per-window rates: windows carry different
    numbers of lost probes, so averaging rates would weight a 1-lost window as heavily
    as a 5-lost one. The pooled rate is the quantity the acceptance rule is written in.
    """
    lost = recovered = eligible = 0
    per_window = []
    for entry in entries:
        rec = entry["recovery"]
        lost += rec["n_lost"]
        recovered += rec["n_recovered"]
        eligible += rec["n_eligible"]
        per_window.append({
            "row": entry["row"],
            "start": entry["start"],
            "n_probes": entry["n_probes"],
            "n_eligible": rec["n_eligible"],
            "n_lost": rec["n_lost"],
            "n_recovered": rec["n_recovered"],
            "recovery_rate": rec["recovery_rate"],
            "recovered_questions": rec["recovered_questions"],
        })
    return {
        "n_windows": len(entries),
        "n_eligible": eligible,
        "n_lost": lost,
        "n_recovered": recovered,
        "recovery_rate": (recovered / lost) if lost else float("nan"),
        "per_window": per_window,
    }


def verdict(train_agg, held_agg, row_separated):
    """The single decision this experiment produces.

    Thresholds are stated up front rather than chosen after seeing the numbers:

        held-out lost == 0                -> undecidable: the fixture cannot resolve it
        held-out rate == 0                -> window_overfit: it learned E6a's window
        held-out rate >= 0.5, recovered>=2 -> generalises
        anything else                     -> partial

    `partial` is a real outcome, not a cop-out: it is the case the next step has to be
    chosen for, which is why the train figure is reported next to it.
    """
    if held_agg["n_lost"] == 0:
        outcome = "undecidable"
        reason = "no held-out window lost a probe the pool was required for"
    elif held_agg["n_recovered"] == 0:
        outcome = "window_overfit"
        reason = "held-out recovered nothing while training recovered %d/%d" % (
            train_agg["n_recovered"], train_agg["n_lost"])
    elif held_agg["recovery_rate"] >= 0.5 and held_agg["n_recovered"] >= 2:
        outcome = "generalises"
        reason = "held-out recovery %d/%d" % (
            held_agg["n_recovered"], held_agg["n_lost"])
    else:
        outcome = "partial"
        reason = "held-out recovery %d/%d, below the 0.5 threshold" % (
            held_agg["n_recovered"], held_agg["n_lost"])
    if outcome != "undecidable" and not row_separated:
        reason += "; held-out shares the training row, so this is a weaker test"
    return {"outcome": outcome, "reason": reason,
            "thresholds": {"generalises_rate": 0.5, "generalises_min_recovered": 2}}


def train_window(manager, lines, entry, compactor, optimiser, stage, steps, args):
    """Train one window, retrying once on OOM with the state rolled back.

    The retry is not defensive padding. A window is 24 steps; losing the run to a
    transient allocation failure at hour two costs the whole experiment. What makes the
    retry safe is the rollback: an OOM can land mid-window, after some steps have already
    been applied to the shared optimiser, so a bare retry would train part of the window
    twice. Restoring both the compactor and the optimiser first makes the retry
    equivalent to the original attempt.

    Retried at most once, so a deterministic failure still surfaces as an error instead
    of looping.
    """
    import copy

    built = rebuild(manager, lines, entry["start"])
    if built is None:
        raise SystemExit("training window row %d start %d stopped building"
                         % (entry["row"], entry["start"]))
    del built

    for attempt in (1, 2):
        weights_state = copy.deepcopy(compactor.state_dict())
        optimiser_state = copy.deepcopy(optimiser.state_dict())
        release()
        try:
            trained = train_carrier_compactor(
                manager, entry["probes"], steps=steps, lr=args.lr, stage=stage,
                window_blocks=E.WINDOW_BLOCKS, target_blocks=E.TARGET_BLOCKS,
                max_probe_tokens=args.max_probe_tokens, compactor=compactor,
                optimiser=optimiser,
            )
            return trained
        except torch.OutOfMemoryError:
            if attempt == 2:
                raise
            print("[e6b]   OOM on stage %s window row %d start %d; rolling back and retrying"
                  % (stage, entry["row"], entry["start"]), flush=True)
            compactor.load_state_dict(weights_state)
            optimiser.load_state_dict(optimiser_state)
            release()
    raise AssertionError("unreachable")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps-a-per-window", type=int, default=4)
    parser.add_argument("--steps-b-per-window", type=int, default=24)
    parser.add_argument("--probes", type=int, default=20,
                        help="candidate probes per training window (E6a's count)")
    parser.add_argument("--eval-probes", type=int, default=60,
                        help="candidate probes per held-out window; evaluation is forward-only")
    parser.add_argument("--max-probe-tokens", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma-max", type=float, default=0.05)
    parser.add_argument("--train-windows", type=int, default=8)
    parser.add_argument("--eval-windows", type=int, default=2)
    parser.add_argument("--max-screen", type=int, default=66,
                        help="stop screening after this many candidates; the screen data "
                             "holds 66 and 10 hard windows may need most of them")
    parser.add_argument("--screen-budget-min", type=float, default=90.0,
                        help="wall-clock cap on screening; proceed with what was found")
    parser.add_argument("--min-probes", type=int, default=4)
    parser.add_argument("--min-lost", type=int, default=2)
    parser.add_argument("--rows", default=None,
                        help="restrict screening to these parquet rows, e.g. 3,2,1")
    parser.add_argument("--candidates", default="/tmp/e6a_screen_data.json")
    parser.add_argument("--out", default="/tmp/e6b_mini_report.json")
    parser.add_argument("--ckpt", default="/tmp/e6b_mini_ckpt.pt")
    parser.add_argument("--seed", type=int, default=None,
                        help="unset by default so the run matches E6a's unseeded behaviour")
    parser.add_argument("--screen-only", action="store_true")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    import yaml
    from transformers import AutoTokenizer

    raw = yaml.safe_load(open(E.CONFIG))
    config = dict(raw["prlmc"])
    config["device"] = DEVICE
    config["dtype"] = "float16"
    config["stats_dir"] = "./outputs/prlmc-e6b"
    config["carrier_gamma_max"] = args.gamma_max
    cfg = PRLMCConfig.from_dict(config)

    model = MemoryLLM.from_pretrained(E.MODEL, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = model.to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(E.MODEL)

    manager = ProtectedResidualMemory(model, tokenizer, cfg)
    window_slots = E.WINDOW_BLOCKS * manager.block

    compactor = CarrierResidualCompactor(
        num_layers=manager.num_layers,
        hidden=manager.hidden,
        block=manager.block,
        rank=int(cfg.carrier_rank),
        gamma_max=float(cfg.carrier_gamma_max),
    ).to(device=DEVICE, dtype=torch.float32)

    # Candidate windows, best-first from the offline text screen. That ranking only sees
    # how many answerable facts sit in the ten oldest blocks; whether the *reader* was
    # required for them is invisible offline, so every candidate still has to be
    # screened on the GPU.
    rows_cfg = None if not args.rows else {int(v) for v in args.rows.split(",")}
    candidates = []
    if args.candidates and os.path.exists(args.candidates):
        with open(args.candidates) as handle:
            ranking = json.load(handle)["ranking"]
        candidates = [(int(e["row"]), int(e["start"])) for e in ranking]
        print("[e6b] %d candidate windows from %s" % (len(candidates), args.candidates), flush=True)
    if not candidates:
        candidates = list(E.window_starts(argparse.Namespace(candidates=None, row=None, max_windows=64)))
        print("[e6b] using the built-in window list (%d)" % len(candidates), flush=True)
    if rows_cfg is not None:
        candidates = [c for c in candidates if c[0] in rows_cfg]
    if not candidates:
        raise SystemExit("[e6b] no candidate windows left after --rows filter")

    needed = args.train_windows + args.eval_windows
    cache = {}
    for row in sorted({r for r, _ in candidates}):
        cache[row] = E.read_fact_lines(row)

    # --- screen: keep only windows where selection provably loses required probes ---

    def enough(hard):
        # `require_other_rows=True` on purpose: the fallback would let screening stop as
        # soon as a single row had enough windows, producing a same-row "held-out" set
        # and defeating the point of the split.
        if not hard:
            return False
        part = partition(hard, args, require_other_rows=True)
        return len(part["train"]) >= args.train_windows and len(part["held"]) >= args.eval_windows

    hard = []
    started = time.time()
    budget = args.screen_budget_min * 60.0
    step_times = []
    for index, (row, start) in enumerate(candidates):
        if len(step_times) >= 3 and sum(step_times) / len(step_times) * (len(candidates) - index) > budget:
            print("[e6b] screen budget exhausted at candidate %d/%d"
                  % (index, len(candidates)), flush=True)
            break
        if index >= args.max_screen:
            print("[e6b] screen stopped at --max-screen %d" % args.max_screen, flush=True)
            break
        if enough(hard):
            break
        tick = time.time()
        try:
            entry = measure_window(manager, cache[row], row, start, compactor,
                                   args.probes, args, log_prefix="[e6b] screen")
        except (RuntimeError, ValueError) as exc:
            print("[e6b] screen row %d start %-6d failed: %s" % (row, start, exc), flush=True)
            torch.cuda.empty_cache()
            continue
        elapsed = time.time() - tick
        step_times.append(elapsed)
        if entry is None:
            continue
        print("[e6b] screen row %d start %-6d ctx %s dropped %-2d probes %-3d no-mem %-4s "
              "full %-4s select %-4s lost %d  (%.0fs)"
              % (row, start, entry["context_hash"], len(entry["dropped_blocks"]), entry["n_probes"],
                 "%d/%d" % (sum(bool(v) for v in entry["no_memory"]["correct"].values()),
                            entry["no_memory"]["n"]),
                 "%d/%d" % (entry["n_eligible"], entry["uncompacted"]["n"]),
                 "%d/%d" % (sum(bool(v) for v in entry["before"]["correct"].values()),
                            entry["before"]["n"]),
                 entry["n_lost"], elapsed), flush=True)
        if entry["n_lost"] < args.min_lost:
            continue
        hard.append(entry)
        print("[e6b]   hard window %d/%d" % (len(hard), needed), flush=True)

    print("[e6b] screened %d candidates in %.0fs, %d hard windows"
          % (len(step_times), time.time() - started, len(hard)), flush=True)

    parts = partition(hard, args)
    train_entries, held_entries = parts["train"], parts["held"]
    print("[e6b] train rows %s (%d windows), held rows %s (%d windows), row-separated %s"
          % ([parts["train_row"]], len(train_entries), parts["held_rows"],
             len(held_entries), parts["row_separated"]), flush=True)
    if len(train_entries) < args.train_windows:
        print("[e6b] WARNING: only %d training windows found (wanted %d)"
              % (len(train_entries), args.train_windows), flush=True)
    if not held_entries:
        raise SystemExit(
            "[e6b] no held-out window; the question this run exists to answer cannot be "
            "asked. Screen more candidates, or lower --min-lost."
        )

    # The identity check is the same one E6a makes: a zero-initialised residual must
    # reproduce `select_blocks` bitwise, or `before` is not the baseline's number.
    probe_entry = train_entries[0]
    built = rebuild(manager, cache[probe_entry["row"]], probe_entry["start"])
    begin = manager.total - manager.used
    source = manager._memory[:, begin : begin + window_slots, :].to(device=DEVICE)
    weights = manager._window_importance(begin, window_slots)
    reference = select_block_compact(
        source, E.TARGET_BLOCKS * manager.block, block=manager.block, weights=weights,
        goal_direction=manager.goal_direction, select_weights=manager.select_weights,
    )
    with torch.no_grad():
        got = compactor(
            source, weights=weights, target=E.TARGET_BLOCKS * manager.block, block=manager.block,
            goal_direction=manager.goal_direction, select_weights=manager.select_weights,
        )
    identity = bool(torch.equal(reference, got))
    print("[e6b] carrier_residual == select_blocks at init: %s (max |d| = %.3e)"
          % (identity, float((reference - got).abs().max())), flush=True)
    del got, reference, source, weights, built
    torch.cuda.empty_cache()

    report = {
        "design": {
            "structure_changed": False,
            "loss_changed": False,
            "selector_changed": False,
            "trigger": "not engaged",
            "gate": "not engaged",
            "window_blocks": E.WINDOW_BLOCKS,
            "target_blocks": E.TARGET_BLOCKS,
            "lr": args.lr,
            "gamma_max": float(cfg.carrier_gamma_max),
            "seed": args.seed,
            "steps_a_per_window": args.steps_a_per_window,
            "steps_b_per_window": args.steps_b_per_window,
            "probes_per_train_window": args.probes,
            "probes_per_held_window": args.eval_probes,
        },
        "identity_at_init": identity,
        "partition": {
            "train_row": parts["train_row"],
            "held_rows": parts["held_rows"],
            "row_separated": parts["row_separated"],
            "train_windows": [(e["row"], e["start"]) for e in train_entries],
            "held_windows": [(e["row"], e["start"]) for e in held_entries],
        },
        "screened": len(step_times),
        "hard_windows": len(hard),
    }

    if args.screen_only:
        json.dump(report, open(args.out, "w"), indent=2, default=str)
        print("[e6b] screen-only report written to %s" % args.out, flush=True)
        return

    # --- train ---

    # One optimiser for every window and both stages. AdamW's moments have to survive the
    # window switch, otherwise each window is a fresh fit and the checkpoint is not shared.
    optimiser = torch.optim.AdamW(compactor.parameters(), lr=args.lr)
    stages = []
    for stage, per_window in (("A", args.steps_a_per_window), ("B", args.steps_b_per_window)):
        if per_window <= 0:
            continue
        print("[e6b] stage %s: %d windows x %d steps" % (stage, len(train_entries), per_window), flush=True)
        for window_index, entry in enumerate(train_entries):
            trained = train_window(
                manager, cache[entry["row"]], entry, compactor, optimiser,
                stage, per_window, args,
            )
            state = trained["state_report"]
            stages.append({
                "stage": stage, "row": entry["row"], "start": entry["start"],
                "steps": per_window,
                "initial_loss": trained["initial_loss"], "final_loss": trained["final_loss"],
                "gamma": state["gamma"], "residual_ratio": state["residual_ratio"],
                "route_scale": state["route_scale"], "routing_entropy": state["routing_entropy"],
                "forward_calls": state["forward_calls"],
                "components": {k: [v[0], v[-1]] for k, v in trained["components"].items()},
            })
            print("[e6b]   stage %s window %d/%d row %d start %-6d loss %.4f -> %.4f "
                  "gamma %.6f residual %.6f route %.6f entropy %.6f"
                  % (stage, window_index + 1, len(train_entries), entry["row"], entry["start"],
                     trained["initial_loss"], trained["final_loss"], state["gamma"],
                     state["residual_ratio"], state["route_scale"], state["routing_entropy"]),
                  flush=True)
    report["stages"] = stages
    if stages:
        last = stages[-1]
        report["final_state"] = {
            "gamma": last["gamma"], "residual_ratio": last["residual_ratio"],
            "route_scale": last["route_scale"], "routing_entropy": last["routing_entropy"],
            "forward_calls": last["forward_calls"],
        }
    torch.save({"state_dict": compactor.state_dict()}, args.ckpt)

    # --- evaluate ---

    print("[e6b] evaluating %d training windows" % len(train_entries), flush=True)
    for window_index, entry in enumerate(train_entries):
        result = evaluate_window(manager, cache[entry["row"]], entry, compactor, args)
        entry["recovery"] = result["recovery"]
        entry["after"] = result["after"]
        print("[e6b]   train window %d/%d row %d start %-6d after %-4s recovery %d/%d"
              % (window_index + 1, len(train_entries), entry["row"], entry["start"],
                 "%d/%d" % (sum(bool(v) for v in result["after"]["correct"].values()),
                            result["after"]["n"]),
                 result["recovery"]["n_recovered"], result["recovery"]["n_lost"]), flush=True)

    # Held-out windows are re-measured with the wider probe set. Their hardness was
    # established at the training probe count; widening only sharpens the measurement.
    print("[e6b] measuring %d held-out windows at %d probes"
          % (len(held_entries), args.eval_probes), flush=True)
    held_measured = []
    for entry in held_entries:
        # `before` has to be re-measured on the wide probe set, because `lost` is defined
        # against the probe set it was measured with.
        widened = measure_window(
            manager, cache[entry["row"]], entry["row"], entry["start"], compactor,
            args.eval_probes,
            argparse.Namespace(min_probes=args.min_probes,
                               max_probe_tokens=args.max_probe_tokens),
            log_prefix="[e6b] held",
        )
        if widened is None:
            print("[e6b] held-out window row %d start %d unusable at %d probes"
                  % (entry["row"], entry["start"], args.eval_probes), flush=True)
            continue
        if widened["n_lost"] < 1:
            print("[e6b] held-out window row %d start %d loses nothing at %d probes"
                  % (entry["row"], entry["start"], args.eval_probes), flush=True)
            continue
        held_measured.append(widened)

    for entry in held_measured:
        result = evaluate_window(manager, cache[entry["row"]], entry, compactor, args)
        entry["recovery"] = result["recovery"]
        entry["after"] = result["after"]
        print("[e6b]   held window row %d start %-6d after %-4s recovery %d/%d"
              % (entry["row"], entry["start"],
                 "%d/%d" % (sum(bool(v) for v in result["after"]["correct"].values()),
                            result["after"]["n"]),
                 result["recovery"]["n_recovered"], result["recovery"]["n_lost"]), flush=True)

    train_agg = aggregate(train_entries)
    held_agg = aggregate(held_measured)
    report["train"] = train_agg
    report["held_out"] = held_agg
    report["verdict"] = verdict(train_agg, held_agg, parts["row_separated"])
    report["probes"] = {
        "train": [{"window": [e["row"], e["start"]],
                   "questions": [p.question for p in e["probes"]]} for e in train_entries],
        "held_out": [{"window": [e["row"], e["start"]],
                      "questions": [p.question for p in e["probes"]],
                      "answers": [p.answers[0] for p in e["probes"]]} for e in held_measured],
    }
    json.dump(report, open(args.out, "w"), indent=2, default=str)

    print("[e6b] train    recovery %d/%d = %.3f over %d windows"
          % (train_agg["n_recovered"], train_agg["n_lost"], train_agg["recovery_rate"],
             train_agg["n_windows"]), flush=True)
    print("[e6b] held-out recovery %d/%d = %.3f over %d windows"
          % (held_agg["n_recovered"], held_agg["n_lost"], held_agg["recovery_rate"],
             held_agg["n_windows"]), flush=True)
    print("[e6b] verdict: %s -- %s"
          % (report["verdict"]["outcome"], report["verdict"]["reason"]), flush=True)
    print("[e6b] checkpoint %s, report %s" % (args.ckpt, args.out), flush=True)


if __name__ == "__main__":
    main()
