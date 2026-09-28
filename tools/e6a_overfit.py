"""E6a: single-window overfit of the carrier-residual compactor.

The one question this answers: can the trained residual be made to beat native
block selection on the *same* compaction of the *same* window? It is the gate the
plan puts before any larger run -- "if one window cannot be overfit, do not run
any large experiment" -- so it is deliberately the smallest thing that can fail
informatively.

Why the probes are synthesised here
-----------------------------------
The benchmark's own questions in this file are multi-hop and paraphrased (every
one of the top-25 question patterns has count 1), so a fixed single-hop template
scores 0/0 against them. Rendering probes from the window's own fact lines is the
only way to get a gold answer that the pool actually contains.

Why the window is built through a live manager
----------------------------------------------
`CarrierResidualCompactor` is called with `goal_direction` and `select_weights`
supplied by the manager, and `_select_block_indices` is the same scorer the
`select_blocks` baseline uses. Building the window by writing real chunks keeps
every one of those inputs on the real code path, so the "before" number is the
baseline's number and not a re-derivation of it.

What is measured, in the plan's acceptance terms
------------------------------------------------
    uncompacted accuracy   upper bound: what the reader can do with no compaction
    select_blocks accuracy the baseline to beat (== carrier_residual at init)
    carrier_residual       after stage A, and after stage B
plus residual_ratio, gamma against its ceiling, and the per-component loss
trajectory (KD and reader-hidden must fall; carrier and stats must not run away).

Run on the host:
    /share/latent/mllm_env/bin/python /tmp/e6a_overfit.py --steps-a 16 --steps-b 24
"""

import argparse
import json
import os
import re
import subprocess
import sys
from contextlib import nullcontext

PARQUET = "/share/latent/MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet"
REPO = "/share/latent/PRLM-C"
MODEL = "/share/latent/MemoryLLM-8B"
CONFIG = os.path.join(REPO, "configs/agent_conf/prlmc_e6.yaml")

DEVICE = "cuda:0"
WINDOW_BLOCKS = 19          # the real E6 window: 4864 slots
TARGET_BLOCKS = 9           # 2304 slots, compact_ratio 0.5
ANCHOR_LINES = 30
# Every write contributes exactly `num_tokens` = 256 slots (the delta is the
# trailing slice of the forward), so one write is one block and 19 blocks needs 19
# writes. The context has 455 numbered fact lines, so chunks must be ~20 lines
# for that many writes to fit; 60 lines/chunk ran out of context at 8 blocks.
LINES_PER_CHUNK = 20

# The reader is only in answer format when it is asked the way inference asks.
# Rendered by the driver rather than imported so the prompt is visible here.
QUERY_TEMPLATE = (
    "Pretend you are a knowledge management system. Each fact in the knowledge pool "
    "is provided with a serial number at the beginning, and the newer fact has larger "
    "serial number. \n You need to solve the conflicts of facts in the knowledge pool "
    "by finding the newest fact with larger serial number. You need to answer a "
    "question based on this rule. You should give a very concise answer without saying "
    "other words for the question **only** from the knowledge pool you have memorized "
    "rather than the real facts in real world. \n\nFor example:\n\n [Knowledge Pool] "
    "\n\n Question: Based on the provided Knowledge Pool, what is the name of the "
    "current president of Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: "
    "Based on the provided Knowledge Pool, {question} \nAnswer:"
)


# --- GPU pick, before torch is imported so CUDA_VISIBLE_DEVICES takes effect ---

def pick_gpu(need_mb: int = None) -> int:
    # Overridable because the two modes have different footprints: a screen-only or
    # dry run holds no training graph (the model is 19.6 GB, so ~20 GB suffices),
    # while the training run peaks at 21.5-22.7 GB. Hard-coding the training figure
    # makes the cheap run wait for a card it does not need.
    if need_mb is None:
        need_mb = int(os.environ.get("E6A_NEED_MB", "21000"))
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
        raise SystemExit("no GPU with %d MB free; best is gpu%d with %d MB" % (need_mb, free[0][1], free[0][0]))
    print("[e6a] gpu%d, %d MB free" % (free[0][1], free[0][0]), flush=True)
    return free[0][1]


if __name__ == "__main__" and not os.environ.get("E6A_GPU_SET"):
    _gpu = pick_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu)
    os.environ["E6A_GPU_SET"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "methods", "memoryllm_modeling"))

from methods.prlmc.config import PRLMCConfig  # noqa: E402
from methods.prlmc.consolidate import CarrierResidualCompactor, select_block_compact  # noqa: E402
from methods.prlmc.layout import ProtectedResidualMemory  # noqa: E402
from methods.prlmc.probes import (  # noqa: E402
    Probe,
    active_mask_override,
    compressed_pool,
    differentiable_memory,
    train_carrier_compactor,
)


# --- fact lines -> probes ----------------------------------------------------

# Only templates whose question can be answered from the line alone. The regex is
# anchored, so a line matching none is simply not probed; nothing is guessed.
#
# A generic `^The (.+) of (.+) is (.+)$` template was removed: it asked "Who is the
# <relation> of <subject>?" for relations whose answer is not a person, producing
# unanswerable questions like "Who is the type of music that Dale Evans plays?".
TEMPLATES = [
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


def read_fact_lines(row=0):
    import pandas as pd

    frame = pd.read_parquet(PARQUET)
    context = frame.iloc[row]["context"]
    lines = [l for l in context.split("\n") if re.match(r"^[0-9]+\. ", l)]
    return [re.sub(r"^[0-9]+\. ", "", l).strip() for l in lines]


def query_stem(question):
    """The exact prompt the reader gets at inference for FactConsolidation.

    Probes must be scored through this, not a bare "Question: ... Answer:". The
    dataset's instruction block is what puts the reader in answer format; a bare
    prompt scored 0/16 even against the full uncompacted pool, so a before/after
    comparison on bare probes would report 0 vs 0 and prove nothing. The template
    carries the trailing "Answer:" that marks where `n_prefix` ends.
    """
    return QUERY_TEMPLATE.format(question=question)


def first_match(text):
    """The first template that matches `text`, or None.

    First-match is required, not cosmetic. The templates overlap: for
    "The capital of Great Britain is London.", both

        ^The capital of (.+) is (.+)$      -> "What is the capital of Great Britain?"
        ^The (.+) of (.+) is (.+)$         -> "Who is the capital of Great Britain?"

    apply. Rendering every match produced nonsense questions like "Who is the
    capital of Great Britain?", which no memory can answer, so a probe set built
    that way scores 0 even uncompacted. The specific templates are listed first and
    the generic `^The (.+) of (.+) is (.+)$` is last for exactly this reason.
    """
    stripped = text.rstrip(".")
    for slot, (pattern, question, groups) in enumerate(TEMPLATES):
        match = re.match(pattern, stripped)
        if not match:
            continue
        parts = match.groups()
        try:
            subject = tuple(parts[int(g) - 1] for g in groups.split(","))
        except (IndexError, TypeError):
            continue
        return slot, subject, parts[-1], question
    return None


def probes_from_lines(lines, indices, limit, dropped_blocks=None, line_to_block=None):
    """Render probes that the *compaction window* is responsible for answering.

    Four constraints, each of which would otherwise produce a probe that cannot
    detect a bad compaction:

    1. Conflict resolution. FactConsolidation's rule is "the newer fact has the
       larger serial number", so a question's correct answer is the *latest*
       matching fact, not the line it was rendered from. The pool contains real
       contradictions ("Chanel was founded by Coco Chanel" vs "... by Andy
       Warhol"), so rendering a probe from an early line would score a correct
       reader as wrong.
    2. The latest matching fact must lie inside the window. If a newer fact on the
       same subject sits in the retained region, the answer survives any
       compaction and the probe cannot move between before and after.
    3. The anchor region is excluded, since it is never compacted either.
    4. **The supporting fact must sit in a block that selection drops.** Without
       this the probe set saturates: the first E6a scored 16/16 both before and
       after training, because every answer happened to live in one of the nine
       surviving carriers, so no compaction could ever have lost it. A probe whose
       answer survives selection cannot measure whether the residual recovered
       anything -- it can only confirm that selection worked.

    Constraint 4 is what turns this from "can the residual re-encode facts it
    already kept" into "can the residual put a dropped block's information back
    into a surviving carrier". `line_to_block` maps an absolute fact-line index to
    its block; it is passed in rather than derived from `// LINES_PER_CHUNK`
    because a gated-out chunk leaves no block and would shift the arithmetic.
    """
    window = set(indices)
    # Latest matching line per (slot, subject), scanned in serial order.
    latest = {}
    for index, text in enumerate(lines):
        found = first_match(text)
        if found is None:
            continue
        slot, subject, answer, _question = found
        latest[(slot, subject)] = (index, answer)

    probes = []
    seen = set()
    for index in sorted(window):
        if dropped_blocks is not None and line_to_block is not None:
            if line_to_block.get(index) not in dropped_blocks:
                continue
        found = first_match(lines[index])
        if found is None:
            continue
        slot, subject, _answer, question = found
        owner, answer = latest.get((slot, subject), (None, None))
        if owner != index:
            continue
        try:
            filled = question % subject
        except TypeError:
            continue
        if filled in seen or not (1 <= len(answer.split()) <= 5):
            continue
        seen.add(filled)
        probes.append(
            Probe(
                question=filled,
                answers=[answer],
                split="train",
                prompt=query_stem(filled),
            )
        )
        if len(probes) >= limit:
            break
    return probes


# --- uncompacted reference ---------------------------------------------------

@torch.no_grad()
def score_uncompacted(manager, probes, max_probe_tokens):
    """Reader accuracy with no compaction at all: the upper bound for any operator."""
    return score_probes(manager, probes, None, max_probe_tokens)


@torch.no_grad()
def score_probes(manager, probes, pool, max_probe_tokens, mask=None):
    """Per-probe correctness, keyed by question.

    Returned per probe rather than as a scalar because the acceptance criterion is
    a set operation: recovery is counted over exactly those questions the
    uncompacted reader got right and selection then lost. A scalar accuracy
    cannot express that denominator.

    Tokenises one probe at a time instead of reusing `_probe_batches`, which drops
    probes it cannot use. Pairing results to probes positionally across a list
    that silently shrinks would attribute one question's correctness to another,
    and the per-question sets are the whole measurement.
    """
    device = manager.model.memory.device
    correct = {}
    loss = 0.0
    total = 0
    context = differentiable_memory(manager.model, pool) if pool is not None else nullcontext()
    guard = active_mask_override(manager.model, mask) if mask is not None else nullcontext()
    with context, guard:
        for probe in probes:
            stem = probe.prompt if probe.prompt else f"{probe.question}\nAnswer:"
            answer = probe.answers[0]
            encoded = manager.tokenizer(
                f"{stem} {answer}", return_tensors="pt", add_special_tokens=False
            )
            input_ids = encoded.input_ids[:, :max_probe_tokens].to(device)
            stem_len = len(manager.tokenizer(stem, add_special_tokens=False).input_ids)
            if stem_len >= max_probe_tokens:
                # Truncating the prompt does not shorten the question, it moves the
                # scored span into the middle of it. `QUERY_TEMPLATE` is ~163
                # tokens, so at `max_probe_tokens=64` the old
                # `n_prefix = min(stem_len, width - 1)` clamped to 63 and the
                # "answer" being scored was the instruction word following
                # "answer a". That token is predicted from the question text alone,
                # so accuracy was 20/20 for every pool -- including one whose every
                # slot the mask had blocked -- and `lost` was structurally zero.
                raise ValueError(
                    "score_probes: max_probe_tokens=%d truncates the %d-token prompt; "
                    "the scored span would be mid-instruction rather than the answer"
                    % (max_probe_tokens, stem_len)
                )
            n_prefix = stem_len
            if input_ids.shape[1] < 2 or n_prefix <= 0 or n_prefix >= input_ids.shape[1]:
                correct[probe.question] = False
                continue
            logits = manager.model(input_ids=input_ids, return_dict=True).logits
            span = logits[:, n_prefix - 1 : -1, :]
            labels = input_ids[:, n_prefix:]
            loss += float(F.cross_entropy(span.reshape(-1, span.shape[-1]), labels.reshape(-1)))
            correct[probe.question] = bool((span.argmax(dim=-1) == labels).all(dim=-1).item())
            total += 1
    return {
        "accuracy": (sum(bool(v) for v in correct.values()) / len(probes)) if probes else float("nan"),
        "gold_loss": loss / total if total else float("nan"),
        "n": len(probes),
        "correct": correct,
    }


def recovery_report(full, before, after, no_memory=None):
    """The metric the experiment exists to produce.

        eligible  questions the pool was *required* for: wrong with no memory,
                  right with the intact pool
        lost      eligible questions that selection then loses
        recovered lost questions the trained carrier answers again
        recovery_rate = recovered / lost

    `no_memory` is what makes `lost` mean anything. Without it the denominator
    counted questions the model answers from its pretrained weights -- which
    selection cannot destroy and a compactor therefore cannot recover. The first
    E6a run had `lost = 0` and reported `improved_over_select: False`, which
    measured nothing at all: `before` was already 20/20.

    The filter is applied to the numerator's population too, not just the
    denominator, so that an eligible-but-not-full question cannot be counted as
    recovered.
    """
    qs = list(full["correct"])
    if no_memory is None:
        eligible = [q for q in qs if full["correct"][q]]
    else:
        eligible = [
            q for q in qs
            if full["correct"][q] and not no_memory["correct"].get(q, False)
        ]
    eligible_set = set(eligible)
    lost = [q for q in eligible if not before["correct"].get(q, False)]
    recovered = [
        q for q in lost
        if after["correct"].get(q, False) and q in eligible_set
    ]
    return {
        "n_eligible": len(eligible),
        "n_full_correct": sum(bool(v) for v in full["correct"].values()),
        "n_lost": len(lost),
        "n_recovered": len(recovered),
        "recovery_rate": (len(recovered) / len(lost)) if lost else float("nan"),
        "lost_questions": lost,
        "recovered_questions": recovered,
    }


def candidate_pool(manager, compactor, device):
    """The compacted pool + mask, for scoring outside the trainer."""
    pool, mask = compressed_pool(
        manager, compactor, WINDOW_BLOCKS, TARGET_BLOCKS, device=device
    )
    return pool, mask


# --- window construction -----------------------------------------------------

def build_window(manager, lines, target_slots, start_line=None):
    """Write real fact chunks until the oldest `target_slots` are worth probing.

    `start_line` shifts the window deeper into the fact list so several disjoint
    19-block windows can be screened; None reproduces the original window.
    """
    offset = ANCHOR_LINES if start_line is None else start_line
    manager.initialize_anchor("\n".join(lines[offset : offset + ANCHOR_LINES]), max_tokens=512)
    offset += ANCHOR_LINES
    covered = []
    cumulative = 0
    writes = 0
    while cumulative < target_slots and offset < len(lines):
        chunk = lines[offset : offset + LINES_PER_CHUNK]
        if not chunk:
            break
        record = manager.write("\n".join(chunk), max_tokens=512)
        if record.get("dropped"):
            offset += LINES_PER_CHUNK
            continue
        width = int(record.get("tokens", 0))
        writes += 1
        if width:
            covered.append((offset, min(offset + LINES_PER_CHUNK, len(lines)), cumulative, width))
            cumulative += width
        if record.get("evicted_blocks") or record.get("compacted"):
            raise SystemExit("the window build compacted or evicted; E6a needs a pristine append-only window")
        offset += LINES_PER_CHUNK
    print("[e6a] %d writes, used=%d slots, window=%d slots" % (writes, manager.used, cumulative), flush=True)
    if manager.used != target_slots:
        # The row ran out of fact lines before the window filled. Callers must not
        # proceed: the compaction shape is 19 -> 9 by construction, and a shorter
        # pool makes `_compacted_pool`'s `used - window + target` arithmetic
        # negative -- which used to surface as a confusing over-wide-pool error.
        raise SystemExit(
            "window is %d slots short of the required %d; pick another start_line"
            % (target_slots - manager.used, target_slots)
        )
    return covered


def write_until(manager, lines, target_slots, start_line):
    """`build_window` for screening, but returns None instead of exiting."""
    try:
        return build_window(manager, lines, target_slots, start_line=start_line)
    except SystemExit:
        return None


def probe_indices(covered, window_slots):
    """Fact lines that lie entirely inside the compaction window."""
    indices = []
    for lo, hi, before, width in covered:
        if before + width <= window_slots:
            indices.extend(range(lo, hi))
    return indices


@torch.no_grad()
def score_no_memory(manager, probes, max_probe_tokens):
    """Accuracy with the pool's *contents* zeroed but its layout intact.

    The ablation is "this information is gone", not "the reader is broken", so the
    mask and geometry stay exactly as the full pool had them and only the values
    change. Passing a zeros tensor through `score_probes`' existing `pool`
    argument does this without touching storage: an in-place `zero_()` would have
    to be undone afterwards, and the pool is 3.36 GB, so the restore is the part
    that can silently fail.
    """
    zeros = torch.zeros_like(manager.model.memory)
    try:
        return score_probes(manager, probes, zeros, max_probe_tokens)
    finally:
        del zeros
        torch.cuda.empty_cache()


def dropped_block_indices(manager, source, weights):
    """The blocks `select_blocks` discards, from the real scorer.

    The recency term alone would only approximate this; importance (weight 0.4)
    also moves the ranking, so the dropped set is read off the operator that the
    run will actually compare against.
    """
    from methods.prlmc.consolidate import _select_block_indices

    kept = _select_block_indices(
        source,
        TARGET_BLOCKS * manager.block,
        manager.block,
        weights,
        manager.goal_direction,
        manager.select_weights,
    )
    b = WINDOW_BLOCKS
    return sorted(set(range(b)) - set(int(i) for i in kept.tolist()))


def select_window(manager, compactor, rows, window_starts, args):
    """Find a window where selection provably loses probes the pool was needed for.

    One line per candidate window:

        row start  dropped  probes  no-mem  full  select  lost

    `probes` counts candidates whose supporting fact sits in a dropped block;
    `no-mem` is accuracy with the pool zeroed; `full` is the intact pool; `select`
    is after selection. `lost` is the population the experiment needs: probes that
    the pool was *required* for (`full` and not `no-mem`) and that selection then
    lost (`not select`).

    All three measurements are necessary. The first screen checked only the middle
    two columns and accepted a window where `lost` was zero because the model knew
    the answers regardless -- a window that could not distinguish a working
    compactor from a broken one. Screening therefore continues until `lost` clears
    `--min-lost`; stopping at the first window with enough candidates is what
    produced the vacuous fixture.

    Every candidate is a fresh `initialize_anchor` + write cycle, so `rows` is
    re-read per candidate rather than cached across them.
    """
    results = []
    for row, start in window_starts:
        lines = rows[row]
        covered = write_until(manager, lines, WINDOW_BLOCKS * manager.block, start)
        if covered is None:
            continue
        window_slots = WINDOW_BLOCKS * manager.block
        # `build_window` stops when the fact list runs out, so a window near the
        # end of a row can cover fewer blocks than `WINDOW_BLOCKS`. The screen
        # must drop those candidates rather than request a 19-block window from a
        # shorter pool: `_compacted_pool` would then compute a negative `new_start`
        # and raise an over-wide-pool error that says nothing about the real cause.
        if manager.used != window_slots:
            print("[e6a] row %d start %-6d skipped: %d slots written, need %d"
                  % (row, start, manager.used, window_slots), flush=True)
            continue
        indices = probe_indices(covered, window_slots)
        if not indices:
            continue

        # Exact line -> block map. A gated-out chunk contributes no block, so this
        # cannot be `line // LINES_PER_CHUNK`.
        line_to_block = {}
        for block_index, (lo, hi, _before, _width) in enumerate(covered):
            for line in range(lo, hi):
                line_to_block[line] = block_index

        begin = manager.total - manager.used
        source = manager._memory[:, begin : begin + window_slots, :].to(device=DEVICE)
        weights = manager._window_importance(begin, window_slots)
        dropped = dropped_block_indices(manager, source, weights)
        # Verification that the fix is load-bearing, not cosmetic. Importance
        # carries weight 0.4 in the selector, so dropping it can pick a different
        # carrier set -- and that mismatch is exactly what made the earlier screen
        # report `lost 0` everywhere. Printed per candidate so a run that finds no
        # gap can be told apart from one where the two scorers never disagreed.
        unweighted = dropped_block_indices(manager, source, None)
        if set(dropped) != set(unweighted):
            print("[e6a]   weights change the dropped set: %s vs %s (unweighted)"
                  % (dropped, unweighted), flush=True)
        del source, weights
        torch.cuda.empty_cache()

        probes = probes_from_lines(
            lines, indices, args.probes,
            dropped_blocks=set(dropped), line_to_block=line_to_block,
        )
        if len(probes) < args.min_probes:
            print("[e6a] row %d start %-6d dropped %-2d  probes %-3d  (too few candidates)"
                  % (row, start, len(dropped), len(probes)), flush=True)
            continue

        with torch.no_grad():
            no_mem = score_no_memory(manager, probes, args.max_probe_tokens)
        full = score_uncompacted(manager, probes, args.max_probe_tokens)
        pool, mask = candidate_pool(manager, compactor, DEVICE)
        select = score_probes(manager, probes, pool, args.max_probe_tokens, mask=mask)
        del pool, mask
        torch.cuda.empty_cache()

        gap = recovery_report(full, select, select, no_memory=no_mem)
        print("[e6a] row %d start %-6d dropped %-2d  probes %-3d  no-mem %-4s  full %-4s  "
              "select %-4s  lost %d"
              % (row, start, len(dropped), len(probes),
                 "%d/%d" % (sum(bool(v) for v in no_mem["correct"].values()), no_mem["n"]),
                 "%d/%d" % (gap["n_eligible"], full["n"]),
                 "%d/%d" % (sum(bool(v) for v in select["correct"].values()), select["n"]),
                 gap["n_lost"]), flush=True)

        results.append({
            "row": row, "start": start, "dropped": dropped, "probes": probes,
            "n_probes": len(probes), "n_eligible": gap["n_eligible"],
            "n_lost": gap["n_lost"], "n_no_memory": sum(bool(v) for v in no_mem["correct"].values()),
            "window_lines": len(indices),
        })
        if gap["n_lost"] >= args.min_lost:
            break

    if not results:
        raise SystemExit("no window produced any probe whose fact selection drops")
    best = max(results, key=lambda r: r["n_lost"])
    if best["n_lost"] < args.min_lost:
        raise SystemExit(
            "no window loses >= %d probes that the pool was required for; best is row %d start %d "
            "with lost=%d (eligible=%d). Screening more windows would be the next step."
            % (args.min_lost, best["row"], best["start"], best["n_lost"], best["n_eligible"])
        )
    return best, results


# --- main --------------------------------------------------------------------

# Candidate (row, start) windows for the screen. Ordered so the cheapest readable
# rows come first, then deeper ones. Written by `tools/e6a_screen_data.py`, which
# ranks every disjoint 380-line window by how many qualifying probes sit in its
# ten oldest blocks -- the part selection is most likely to drop. Only the head of
# that ranking is listed here: the GPU screen is the expensive part and needs a
# handful of candidates, not all 66.
def window_starts(args):
    """Candidate (row, start) windows, best-first.

    Prefers the full offline ranking from `tools/e6a_screen_data.py` when it is
    available: that file ranks all 66 disjoint windows by how many qualifying
    probes sit in the ten oldest blocks, which is where selection is most likely to
    cut. The hardcoded list is only the head of that ranking and exists so the tool
    still runs without the screen output.

    Sweeping matters because the first-ranked window is not necessarily usable. The
    top entry (row 3 start 17890) had 20 probes in dropped blocks and still lost
    zero of them: the model answered every one from pretrained weights. Ranking by
    the text-level signal cannot see that, so the GPU screen has to walk down the
    list until it finds a window with a real gap.
    """
    fallback = [
        (3, 17890), (3, 17130), (3, 16750), (3, 14470), (3, 17510),
        (2, 3450), (3, 14850), (1, 1930), (3, 16370), (3, 15230),
        (2, 3830), (3, 15610), (3, 15990), (1, 1550),
    ]
    rows = fallback
    if args.candidates and os.path.exists(args.candidates):
        try:
            with open(args.candidates) as handle:
                ranking = json.load(handle)["ranking"]
            rows = [(int(entry["row"]), int(entry["start"])) for entry in ranking]
            print("[e6a] %d candidate windows from %s" % (len(rows), args.candidates), flush=True)
        except (KeyError, ValueError, OSError) as exc:
            print("[e6a] could not read %s (%s); using the built-in list"
                  % (args.candidates, exc), flush=True)
            rows = fallback
    if args.row is not None:
        rows = [r for r in rows if r[0] == args.row]
    return rows[: args.max_windows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps-a", type=int, default=16)
    parser.add_argument("--steps-b", type=int, default=24)
    parser.add_argument("--probes", type=int, default=20)
    parser.add_argument("--max-probe-tokens", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma-max", type=float, default=0.05)
    parser.add_argument("--out", default="/tmp/e6a_report.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--window", default=None, help="pin a window as ROW,START")
    parser.add_argument("--row", type=int, default=None, help="restrict the screen to one row")
    parser.add_argument("--max-windows", type=int, default=14)
    parser.add_argument("--candidates", default="/tmp/e6a_screen_data.json",
                        help="offline ranking from tools/e6a_screen_data.py")
    parser.add_argument("--min-probes", type=int, default=4,
                        help="candidate probes in dropped blocks needed to score a window")
    parser.add_argument("--min-lost", type=int, default=3,
                        help="probes the pool was required for that selection must lose")
    args = parser.parse_args()

    import yaml
    from transformers import AutoTokenizer

    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM

    raw = yaml.safe_load(open(CONFIG))
    config = dict(raw["prlmc"])
    config["device"] = DEVICE
    config["dtype"] = "float16"
    config["stats_dir"] = "./outputs/prlmc-e6a"
    config["carrier_gamma_max"] = args.gamma_max
    cfg = PRLMCConfig.from_dict(config)

    model = MemoryLLM.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = model.to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    manager = ProtectedResidualMemory(model, tokenizer, cfg)
    rows = {row: read_fact_lines(row) for row in set(r for r, _ in window_starts(args))}

    compactor = CarrierResidualCompactor(
        num_layers=manager.num_layers,
        hidden=manager.hidden,
        block=manager.block,
        rank=int(cfg.carrier_rank),
        gamma_max=float(cfg.carrier_gamma_max),
    ).to(device=DEVICE, dtype=torch.float32)

    window_slots = WINDOW_BLOCKS * manager.block
    if args.window:
        row, start = [int(v) for v in args.window.split(",")]
        rows[row] = read_fact_lines(row)
        best = {"row": row, "start": start, "n_lost": None}
        covered = build_window(manager, rows[row], window_slots, start_line=start)
        indices = probe_indices(covered, window_slots)
        line_to_block = {}
        for block_index, (lo, hi, _before, _width) in enumerate(covered):
            for line in range(lo, hi):
                line_to_block[line] = block_index
        begin = manager.total - manager.used
        source = manager._memory[:, begin : begin + window_slots, :].to(device=DEVICE)
        weights = manager._window_importance(begin, window_slots)
        dropped = dropped_block_indices(manager, source, weights)
        probes = probes_from_lines(
            rows[row], indices, args.probes,
            dropped_blocks=set(dropped), line_to_block=line_to_block,
        )
        print("[e6a] using pinned window row %d start %d" % (row, start), flush=True)
    else:
        best, _all = select_window(manager, compactor, rows, window_starts(args), args)
        probes = best["probes"]
        dropped = best["dropped"]
        print("[e6a] selected window row %d start %d: %d candidates, %d eligible, %d lost"
              % (best["row"], best["start"], best["n_probes"], best["n_eligible"],
                 best["n_lost"]), flush=True)
        # The screen rebuilt each candidate window to measure it; the winner's is
        # rebuilt here so the measurements below all read the same memory state.
        covered = build_window(manager, rows[best["row"]], window_slots, start_line=best["start"])
        indices = probe_indices(covered, window_slots)

    print("[e6a] dropped blocks: %s" % dropped, flush=True)
    for probe in probes:
        print("        %-60s -> %s" % (probe.question, probe.answers[0]))
    if len(probes) < args.min_probes:
        raise SystemExit(
            "only %d probes in dropped blocks; the gap this experiment needs is too small "
            "(use --window to pin another, or raise --probes)" % len(probes)
        )

    # Real-input version of the CPU structural test: a zero-initialised residual
    # must reproduce select_blocks bitwise, or the "before" number is not the
    # baseline's number. The weights come from the manager's own
    # `_window_importance`, which is exactly what `compact_with_report` feeds the
    # operator, so this compares the two paths on identical inputs.
    begin = manager.total - manager.used
    source = manager._memory[:, begin : begin + window_slots, :].to(device=DEVICE)
    weights = manager._window_importance(begin, window_slots)
    reference = select_block_compact(
        source, TARGET_BLOCKS * manager.block, block=manager.block,
        weights=weights, goal_direction=manager.goal_direction,
        select_weights=manager.select_weights,
    )
    # Under no_grad deliberately. At init the residual is identically zero, so the
    # comparison needs no graph -- and building one here leaked ~3.8 GB that stayed
    # resident into stage A: this call runs *before* the trainer, so its activations
    # were never freed, and the compactor's forward then had no headroom left.
    with torch.no_grad():
        got = compactor(
            source, weights=weights, target=TARGET_BLOCKS * manager.block, block=manager.block,
            goal_direction=manager.goal_direction, select_weights=manager.select_weights,
        )
    identity = bool(torch.equal(reference, got))
    print("[e6a] carrier_residual == select_blocks at init: %s (max |d| = %.3e)"
          % (identity, float((reference - got).abs().max())), flush=True)
    del got, reference, source, weights
    torch.cuda.empty_cache()

    report = {
        "window": {"row": best["row"], "start": best["start"]},
        "dropped_blocks": dropped,
        "window_slots": window_slots,
        "target_slots": TARGET_BLOCKS * manager.block,
        "used_slots": manager.used,
        "n_probes": len(probes),
        "gamma_max": float(cfg.carrier_gamma_max),
        "identity_at_init": identity,
        "probes": [{"q": p.question, "a": p.answers[0]} for p in probes],
    }

    device = manager.model.memory.device
    report["no_memory"] = score_no_memory(manager, probes, args.max_probe_tokens)
    report["uncompacted"] = score_uncompacted(manager, probes, args.max_probe_tokens)
    pool, mask = candidate_pool(manager, compactor, device)
    report["before"] = score_probes(manager, probes, pool, args.max_probe_tokens, mask=mask)
    del pool, mask
    torch.cuda.empty_cache()
    for label in ("no_memory", "uncompacted", "before"):
        entry = report[label]
        print("[e6a] %-11s acc %.3f (%d/%d)" % (
            label, entry["accuracy"],
            sum(bool(v) for v in entry["correct"].values()), entry["n"]), flush=True)

    gap = recovery_report(report["uncompacted"], report["before"], report["before"],
                          no_memory=report["no_memory"])
    print("[e6a] pool required for %d probes; selection loses %d of them"
          % (gap["n_eligible"], gap["n_lost"]), flush=True)
    for question in gap["lost_questions"]:
        print("        lost: %s" % question, flush=True)
    if gap["n_lost"] < args.min_lost:
        raise SystemExit(
            "selection loses only %d probes that the pool was required for (need >= %d; "
            "no-mem %d/%d, full %d/%d, select %d/%d); this fixture cannot test recovery"
            % (gap["n_lost"], args.min_lost,
               sum(bool(v) for v in report["no_memory"]["correct"].values()), report["no_memory"]["n"],
               gap["n_full_correct"], report["uncompacted"]["n"],
               sum(bool(v) for v in report["before"]["correct"].values()), report["before"]["n"])
        )

    if args.dry_run:
        json.dump(report, open(args.out, "w"), indent=2, default=str)
        print("[e6a] dry run written to %s" % args.out)
        return

    for stage, steps in (("A", args.steps_a), ("B", args.steps_b)):
        if steps <= 0:
            continue
        print("[e6a] training stage %s for %d steps" % (stage, steps), flush=True)
        trained = train_carrier_compactor(
            manager, probes, steps=steps, lr=args.lr, stage=stage,
            window_blocks=WINDOW_BLOCKS, target_blocks=TARGET_BLOCKS,
            max_probe_tokens=args.max_probe_tokens, compactor=compactor,
        )
        pool, mask = candidate_pool(manager, compactor, device)
        after = score_probes(manager, probes, pool, args.max_probe_tokens, mask=mask)
        del pool, mask
        torch.cuda.empty_cache()
        recovery = recovery_report(report["uncompacted"], report["before"], after,
                                   no_memory=report["no_memory"])
        report["stage_" + stage] = {"train": trained, "after": after, "recovery": recovery}
        print("[e6a] stage %s after acc %.3f, recovery %d/%d = %.3f"
              % (stage, after["accuracy"], recovery["n_recovered"], recovery["n_lost"],
                 recovery["recovery_rate"]), flush=True)

    final_stage = report.get("stage_B", report.get("stage_A"))
    final = final_stage["after"]
    report["acceptance"] = {
        "n_eligible": gap["n_eligible"],
        "n_lost": gap["n_lost"],
        "n_recovered": final_stage["recovery"]["n_recovered"],
        "recovery_rate": final_stage["recovery"]["recovery_rate"],
        "lost_questions": gap["lost_questions"],
        "recovered_questions": final_stage["recovery"]["recovered_questions"],
        "no_memory_accuracy": report["no_memory"]["accuracy"],
        "before_accuracy": report["before"]["accuracy"],
        "after_accuracy": final["accuracy"],
        "uncompacted_accuracy": report["uncompacted"]["accuracy"],
    }
    json.dump(report, open(args.out, "w"), indent=2, default=str)
    print("[e6a] acceptance: %s" % report["acceptance"], flush=True)
    print("[e6a] written to %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
