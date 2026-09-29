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
    lost      eligible and then lost by the *initial* compactor  (the denominator)
    recovered lost and answered again by the trained carrier      (the numerator)

`lost` is measured against the compactor's **initial** state, which is bitwise identical
to `select_blocks` (the identity check asserts this). That keeps the denominator
comparable with the screening count, where the same quantity is what makes a window
"hard" in the first place.

Correction to the first version of this script
----------------------------------------------
The first run reported `verdict: window_overfit` on held-out 0/16. That verdict was
unmeasurable, and the code that produced it was wrong.

Held-out windows were re-measured at 60 probes *after* training, and `measure_window`
was handed the same `compactor` object that `train_window` had already updated in place.
So `before` and `after` were two evaluations of the same function:

  * the compactor forward is deterministic -- `attention_dropout` is 0.0, the pool path
    has no dropout, and `_select_block_indices` / `_greedy_diverse` contain no RNG;
  * `rebuild` re-anchors and re-writes the same fact lines from the same start, and
    `initialize_anchor` calls `reset`, so the pool is bit-identical;
  * `evaluate_window` reuses `entry["probes"]`, so the probe set is identical.

Identical weights, identical pool, identical probes => identical `correct` dicts =>
`recovered` is structurally zero. No amount of real recovery could have been observed.

The training side was never affected: its `before` came from the screen, measured before
any training, and `evaluate_window` reuses it. The two sides therefore ran under
different conventions, and only the held-out one was incapable of a positive result.

The fix is structural, not a patch to one call site. Three parts:

  * `main` snapshots `initial_state = copy.deepcopy(compactor.state_dict())` before any
    step runs. That snapshot is the single source of truth for every `before` in the run:
    screen, training and held-out alike.
  * `measure_window` takes `template` and `initial_state` and builds its own compactor
    via `before_compactor_for`, which deep-copies the template and *then* loads the
    initial weights. A caller that passes the trained module still gets an untrained
    compactor, so the bug is not merely fixed but made inexpressible.
  * `measure_window` also accepts `trained_compactor`, used only to run the trained
    weights through the same before-path a second time. Held-out windows compare that
    result against `after`: it must match bitwise, and that recorded equality is the
    standing evidence that the two scoring paths are deterministic. `correct_signature`
    hashes each pass's `correct` dict, and `after_matches_before` records whether a
    window's `after` collapsed onto its `before` -- the exact signature of the original
    defect, now reported per window instead of being invisible.

`measurement_integrity` collects the adjacent failure modes (weights never moved,
no window's answers ever changed, the two forwards disagreeing) and `verdict` refuses to
emit a threshold outcome at all when it fails. That is what stops a void measurement from
being reported as `window_overfit` again.

Reported separately for training windows and held-out windows. The training figure is
recorded as context; `verdict` is decided on the held-out figure alone.

Run on the host (screening + 32 + 192 training steps + two evaluation passes):
    /share/latent/mllm_env/bin/python /tmp/e6b_mini.py \
        --steps-a-per-window 4 --steps-b-per-window 24 \
        --train-windows 8 --eval-windows 2 --out /tmp/e6b_mini_report.json
"""

import argparse
import copy
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


def before_compactor_for(template, initial_state):
    """A fresh compactor carrying the *initial* weights.

    The name says exactly what the object is allowed to be, because the first version of
    this script got that wrong and the mistake was invisible: the held-out `before` was
    measured with the already-trained module, so `before` and `after` were the same
    function evaluated twice and `recovered` was structurally zero.

    `copy.deepcopy` plus an explicit `load_state_dict` means the returned module shares no
    storage with the trainer regardless of what `template` is. Passing the live, trained
    module as `template` still yields an untrained compactor, so the failure cannot be
    reintroduced by a caller that forgets which object it is holding.
    """
    clone = copy.deepcopy(template)
    clone.load_state_dict(copy.deepcopy(initial_state))
    clone.eval()
    return clone


def measure_window(manager, lines, row, start, template, initial_state, n_probes, args,
                   log_prefix="[e6b]", trained_compactor=None):
    """Build a window and measure every quantity the acceptance rule needs.

    One call = one fresh window state, so `no_memory`, `uncompacted` and `before` all
    describe the same pool.

    `before` is always produced by a compactor built from `initial_state`, never by
    `trained_compactor`. The two are separate parameters on purpose: `trained_compactor`
    is used only for the `trained_as_before` diagnostic, which exists to make the
    original defect visible in the report rather than merely fixed.

    Returns a dict, or None when the window is unusable.
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

    before_compactor = before_compactor_for(template, initial_state)
    with torch.no_grad():
        no_mem = E.score_no_memory(manager, probes, args.max_probe_tokens)
        full = E.score_uncompacted(manager, probes, args.max_probe_tokens)
        # `candidate_pool` runs the compactor forward. Outside `no_grad` it builds a graph
        # that nothing ever frees; E6a leaked ~3.8 GB this way at the identity check and
        # the trainer then had no headroom. Measurement never needs gradients.
        pool, mask = E.candidate_pool(manager, before_compactor, DEVICE)
        before = E.score_probes(manager, probes, pool, args.max_probe_tokens, mask=mask)
        # Released before the second forward, not after both. Each pool is
        # [32, 12800, 4096] fp16 = 3.12 GiB, so holding two at once would peak at
        # 6.24 GiB -- the same envelope that OOM'd this run's first version.
        del pool, mask
        torch.cuda.empty_cache()

        # Determinism evidence, not a measurement. Runs the *trained* compactor through
        # the same before-path, so in the fixed version it must equal `after` bitwise --
        # that equality is what proves the two scoring paths are deterministic, which is
        # the premise that made the original bug's `before == after` meaningful. On the
        # broken version `before` was this quantity, so it matched `after` because the
        # baseline had silently drifted onto the trained weights.
        trained_as_before = None
        if trained_compactor is not None:
            pool_t, mask_t = E.candidate_pool(manager, trained_compactor, DEVICE)
            trained_as_before = E.score_probes(
                manager, probes, pool_t, args.max_probe_tokens, mask=mask_t
            )
            del pool_t, mask_t
            torch.cuda.empty_cache()
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
        # The trained compactor's own `before`, kept only so the report can assert that
        # this path and `after` agree bitwise. See the module docstring.
        "trained_as_before": trained_as_before,
        "n_eligible": gap["n_eligible"],
        "n_lost": gap["n_lost"],
        "lost_questions": gap["lost_questions"],
    }


def correct_signature(scored):
    """A hashable fingerprint of which probes a scoring pass got right.

    Two scoring passes with the same weights, pool and probes must produce the same
    fingerprint. That equality is what makes the original defect detectable after the
    fact: `before` and `after` sharing a fingerprint meant they were the same function
    evaluated twice, so `recovered` could not be anything but zero.
    """
    if scored is None:
        return None
    return hashlib.md5(
        json.dumps(sorted((str(k), bool(v)) for k, v in scored["correct"].items())).encode()
    ).hexdigest()[:12]


def evaluate_window(manager, lines, entry, compactor, args):
    """Re-build a measured window and score it with the *trained* compactor.

    `no_memory`, `uncompacted` and `before` are taken from `entry` rather than
    re-measured. All three are deterministic functions of the window and the
    *pre-training* compactor: the rebuild re-anchors and re-writes the same fact lines
    from the same start, so the pool is bit-identical and re-running them would spend
    two passes per window to reproduce numbers that are already in hand. `before` in
    particular must stay the pre-training measurement, because it is the denominator
    `lost` is defined against.

    Only `after` is new, so only one scoring pass is needed here. The returned
    `same_as_before` flag is the standing check that the trained compactor does *not*
    reproduce the initial one: it must be False, or nothing was measured.
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
    return {
        "recovery": recovery,
        "after": after,
        "after_signature": correct_signature(after),
        "before_signature": correct_signature(entry["before"]),
        # True would mean `after` is the initial compactor's result, i.e. training had no
        # effect on this window -- or, in the original bug, that `before` was measured
        # with the trained weights and the two sides collapsed into one evaluation.
        "after_matches_before": (
            correct_signature(after) == correct_signature(entry["before"])
        ),
    }


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
            "signatures": entry.get("signatures"),
        })
    return {
        "n_windows": len(entries),
        "n_eligible": eligible,
        "n_lost": lost,
        "n_recovered": recovered,
        "recovery_rate": (recovered / lost) if lost else float("nan"),
        # How many windows found `after` bitwise identical to `before`. On the broken
        # version this was every window, because the two were the same evaluation.
        "n_after_matches_before": sum(
            1 for e in entries
            if (e.get("signatures") or {}).get("after_matches_before")
        ),
        "per_window": per_window,
    }


def measurement_integrity(initial_state, compactor, train_agg, held_agg, stages,
                          held_determinism):
    """Sanity checks on whether this run measured anything, plus one caveat flag.

    Be precise about what this can and cannot do, because overclaiming here would
    recreate the problem it exists to catch. The original defect is fixed
    *structurally*: `before` is built from `initial_state` via `before_compactor_for`,
    so it cannot carry trained weights. These checks are a net for adjacent failures,
    not the fix, and the broken run would in fact have passed the first three.

    1. **Training moved the weights.** If `state_dict` came back bitwise equal to the
       initial one, every step was skipped -- defect 6's NaN-gradient failure, where
       `GradScaler` discarded every update and the run still looked healthy.
    2. **Some window somewhere changed an answer.** If none did, the compactor is
       functionally identical to the one it started as and every `recovered` is zero
       for a reason unrelated to compression. Deliberately a disjunction over train and
       held-out: a single window where training changes no answer is a real measurement
       of nothing, and the pooled counts already account for it.
    3. **The scoring path is deterministic.** Held-out windows run the trained compactor
       through the before-path a second time; it must reproduce `after`. Two forwards
       disagreeing would invalidate every signature comparison in the report. This is the
       premise the original bug's collapse rested on, so it is checked, not assumed.

    One caveat that cannot be turned into a pass/fail, only reported:
    `held_all_unchanged` is True when every held-out window produced `after` identical to
    `before`. With the fix that is the signature of a genuine negative result -- training
    changed nothing on unseen windows -- but it is *also* what the original bug produced,
    so the two are not distinguishable from this flag alone. It is recorded, and
    `verdict` appends it as a caveat, rather than being silently folded into either
    outcome.
    """
    identical = all(
        torch.equal(initial_state[name], value)
        for name, value in compactor.state_dict().items()
    )
    moved_answers = (
        train_agg["n_windows"] - train_agg["n_after_matches_before"] > 0
        or held_agg["n_windows"] - held_agg["n_after_matches_before"] > 0
    )
    determinism_checked = [
        (e.get("signatures") or {}).get("trained_as_before_matches_after")
        for e in held_determinism
    ]
    determinism_checked = [v for v in determinism_checked if v is not None]
    held_all_unchanged = (
        held_agg["n_windows"] > 0
        and held_agg["n_after_matches_before"] == held_agg["n_windows"]
    )
    base = {
        "weights_identical": identical,
        "train_windows_changed": train_agg["n_windows"] - train_agg["n_after_matches_before"],
        "held_windows_changed": held_agg["n_windows"] - held_agg["n_after_matches_before"],
        "determinism_checked": len(determinism_checked),
        "held_all_unchanged": held_all_unchanged,
    }
    if not stages:
        return dict(base, ok=False, reason="no training stage ran; nothing was trained")
    if identical:
        return dict(base, ok=False,
                    reason="trained state_dict is bitwise equal to the initial one; "
                           "every update was discarded (NaN gradients?)")
    if not moved_answers:
        return dict(base, ok=False,
                    reason="no window anywhere changed a single answer; `after` is the "
                           "initial compactor, so `recovered` was not measured")
    if determinism_checked and not all(determinism_checked):
        return dict(base, ok=False,
                    reason="two forward passes of the trained compactor disagreed; the "
                           "scoring path is not deterministic and no signature "
                           "comparison is meaningful")
    return dict(base, ok=True,
                reason="weights changed and at least one window changed an answer")


def verdict(train_agg, held_agg, row_separated, integrity):
    """The single decision this experiment produces.

    Thresholds are stated up front rather than chosen after seeing the numbers:

        integrity failed                  -> unmeasured: report the failure, not a verdict
        held-out lost == 0                -> undecidable: the fixture cannot resolve it
        held-out rate == 0                -> window_overfit: it learned E6a's window
        held-out rate >= 0.5, recovered>=2 -> generalises
        anything else                     -> partial

    `unmeasured` comes first on purpose. The first version's `window_overfit` came from a
    comparison that could not produce a positive number, and the threshold table had no
    way to express that, so a broken measurement was reported as a finding.

    `held_all_unchanged` is appended as a caveat rather than folded into an outcome. It is
    the signature of a genuine negative result AND the signature of the original defect,
    and no flag can separate the two -- so the reader is told rather than reassured.
    """
    if not integrity["ok"]:
        return {"outcome": "unmeasured",
                "reason": "measurement integrity check failed: " + integrity["reason"],
                "thresholds": {"generalises_rate": 0.5, "generalises_min_recovered": 2}}
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
    if integrity.get("held_all_unchanged"):
        reason += ("; caveat: every held-out window produced `after` identical to "
                   "`before`, so this is either a real null result or the defect "
                   "resurfacing -- it cannot be told apart from this run alone")
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

    # The untrained weights, captured before any step runs. This is the single source of
    # truth for every `before` measurement in the run -- screen, training and held-out.
    # The first version had no such snapshot and measured the held-out `before` with the
    # already-trained module, which made `recovered` structurally zero. Keeping the
    # snapshot separate from `compactor` is what prevents that from being expressible.
    initial_state = copy.deepcopy(compactor.state_dict())

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
                                   initial_state, args.probes, args, log_prefix="[e6b] screen")
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
        entry["signatures"] = {
            "before": result["before_signature"],
            "after": result["after_signature"],
            "after_matches_before": result["after_matches_before"],
        }
        print("[e6b]   train window %d/%d row %d start %-6d after %-4s recovery %d/%d "
              "before_sig %s after_sig %s after==before %s"
              % (window_index + 1, len(train_entries), entry["row"], entry["start"],
                 "%d/%d" % (sum(bool(v) for v in result["after"]["correct"].values()),
                            result["after"]["n"]),
                 result["recovery"]["n_recovered"], result["recovery"]["n_lost"],
                 result["before_signature"], result["after_signature"],
                 result["after_matches_before"]), flush=True)

    # Held-out windows are re-measured with the wider probe set. Their hardness was
    # established at the training probe count; widening only sharpens the measurement.
    # `before` still comes from `initial_state`: the wider probe set changes which
    # questions are counted, never which weights define the baseline.
    print("[e6b] measuring %d held-out windows at %d probes"
          % (len(held_entries), args.eval_probes), flush=True)
    held_measured = []
    for entry in held_entries:
        # `before` has to be re-measured on the wide probe set, because `lost` is defined
        # against the probe set it was measured with.
        widened = measure_window(
            manager, cache[entry["row"]], entry["row"], entry["start"], compactor,
            initial_state, args.eval_probes,
            argparse.Namespace(min_probes=args.min_probes,
                               max_probe_tokens=args.max_probe_tokens),
            log_prefix="[e6b] held",
            trained_compactor=compactor,
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
        # `trained_as_before` came from a *separate* forward pass over the trained
        # compactor. It must reproduce `after` exactly; that is the determinism premise,
        # stated as a check rather than assumed. It is deliberately allowed to differ from
        # `before`, which is now the initial compactor's result.
        # Two equalities get printed and both matter, in opposite directions.
        # `trained_as_before == after` MUST hold: same weights, same pool, same probes,
        # two independent forwards. It is the determinism premise, now checked.
        # `after == before` must NOT normally hold: it would mean the trained compactor
        # answers exactly as the untrained one on this window, which is the pooled
        # `n_after_matches_before` the integrity check reads.
        trained_sig = correct_signature(entry.get("trained_as_before"))
        entry["signatures"] = {
            "before": result["before_signature"],
            "after": result["after_signature"],
            "after_matches_before": result["after_matches_before"],
            "trained_as_before": trained_sig,
            "trained_as_before_matches_after": trained_sig == result["after_signature"],
        }
        print("[e6b]   held window row %d start %-6d after %-4s recovery %d/%d "
              "before_sig %s after_sig %s trained_sig %s "
              "after==before %s trained==after %s"
              % (entry["row"], entry["start"],
                 "%d/%d" % (sum(bool(v) for v in result["after"]["correct"].values()),
                            result["after"]["n"]),
                 result["recovery"]["n_recovered"], result["recovery"]["n_lost"],
                 result["before_signature"], result["after_signature"], trained_sig,
                 result["after_matches_before"], trained_sig == result["after_signature"]),
              flush=True)

    train_agg = aggregate(train_entries)
    held_agg = aggregate(held_measured)
    integrity = measurement_integrity(
        initial_state, compactor, train_agg, held_agg, stages, held_measured
    )
    report["train"] = train_agg
    report["held_out"] = held_agg
    report["integrity"] = integrity
    report["verdict"] = verdict(train_agg, held_agg, parts["row_separated"], integrity)
    report["probes"] = {
        "train": [{"window": [e["row"], e["start"]],
                   "questions": [p.question for p in e["probes"]]} for e in train_entries],
        "held_out": [{"window": [e["row"], e["start"]],
                      "questions": [p.question for p in e["probes"]],
                      "answers": [p.answers[0] for p in e["probes"]]} for e in held_measured],
    }
    json.dump(report, open(args.out, "w"), indent=2, default=str)

    print("[e6b] train    recovery %d/%d = %.3f over %d windows (%d changed an answer)"
          % (train_agg["n_recovered"], train_agg["n_lost"], train_agg["recovery_rate"],
             train_agg["n_windows"],
             train_agg["n_windows"] - train_agg["n_after_matches_before"]), flush=True)
    print("[e6b] held-out recovery %d/%d = %.3f over %d windows (%d changed an answer)"
          % (held_agg["n_recovered"], held_agg["n_lost"], held_agg["recovery_rate"],
             held_agg["n_windows"],
             held_agg["n_windows"] - held_agg["n_after_matches_before"]), flush=True)
    print("[e6b] integrity: %s -- %s"
          % ("ok" if integrity["ok"] else "FAILED", integrity["reason"]), flush=True)
    print("[e6b] verdict: %s -- %s"
          % (report["verdict"]["outcome"], report["verdict"]["reason"]), flush=True)
    print("[e6b] checkpoint %s, report %s" % (args.ckpt, args.out), flush=True)


if __name__ == "__main__":
    main()
