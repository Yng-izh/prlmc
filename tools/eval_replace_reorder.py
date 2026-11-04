#!/usr/bin/env python3
"""Experiment 2 -- how much does REPLACING carrier slots cost, and how much REORDERING?

Two things happen at once when V10 compresses 19 blocks into 9
-------------------------------------------------------------
Experiment 1 measured the joint effect of the V10 compression (Selection -> Graft)
but could not say whether the loss came from

  (i)  overwriting 40 of each carrier's own 256 slots with foreign content, or
  (ii) feeding the frozen reader the same 2,304 slot vectors in a DIFFERENT ORDER
       (V10 emits them in source-time order, so a block's 40 graft slots are
       interleaved with the carriers instead of sitting where the carrier's own
       slots used to be).

This driver separates them by building a third arm that is a pure permutation of
the V10 output, so B and C hold literally the same 2,304 vectors:

    Full       the 19 written blocks, nothing compressed          19 blocks / 4,864 slots
    A Selection  the 9 carrier blocks, untouched                   9 blocks / 2,304 slots
    B Replace   each carrier keeps its own 216 core slots AT THEIR ORIGINAL
                IN-BLOCK POSITIONS, and the 40 positions it vacated are filled
                with the 360 graft slots of the V10 compression  9 blocks / 2,304 slots
    C Graft     the V10 output exactly as the operator emits it, i.e. the same
                2,304 vectors in source-time order              9 blocks / 2,304 slots

  A -> B  the net effect of replacing slots (deleting 40 carrier slots and adding
          40 graft slots, jointly -- this single difference cannot separate the
          deletion loss from the graft gain)
  B -> C  the effect of the reorder ALONE, because the vector set is identical
  A -> C  the effect experiment 1 measured, re-run here as a cross-check

B is not a re-implementation of the compression: it is `graft_output.index_select(1, perm)`
with `perm` a verified bijection of `range(2304)`. That is what makes "same
vectors, different order" a fact of the code rather than a claim in a report:
`plan_inplace_replace` asserts `sorted(perm) == list(range(2304))` and the report
stores the permutation itself, so the layout can be rebuilt from arm C's output
after the run without touching a GPU.

Which 40 graft vectors land in which carrier
--------------------------------------------
The 360 graft slots are taken in source-time order and dealt out 40 at a time in
carrier order (carrier k gets graft slots 40k..40k+39). The rule is arbitrary but
fixed, monotone in document position, and it only decides which foreign vector
lands in which block -- never which vectors are in the memory. The report stores
the full permutation, the resulting per-carrier source-block counts and its
digest, so the choice is auditable instead of invisible, and a second assignment
rule can be compared against this one without re-running anything.

Protocol notes (all recorded in the report)
------------------------------------------
* Everything else is held at the experiment-1 setting: same 158 eligible rows,
  same frozen MemoryLLM-8B reader, one 19 -> 9 compression per sample, same 9
  carriers, same top-216 core selection, same 360 graft slots, `operator_mode`
  "native" (no condensation, no orthogonal delta, no core reinforcement), no
  training, greedy decoding.
* The write stream is the document prefix only, so block b covers exactly tokens
  `[512b, 512(b+1))` and the evidence mapping stays exact.
* `--verify-against` compares this run's Full/Selection/Graft answers with a
  finished experiment-1 report row by row. A 100% match on all three is the
  evidence that arm C really is the old implementation and not a lookalike.

Usage
-----
    # what would run, without loading the model
    python tools/eval_replace_reorder.py --dry-run \
        --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
        --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
        --model-path /dev/shm/s1d_reader_fp16 \
        --out reports/diagnostics/replace_reorder_hotpotqa.json

    # the experiment
    python tools/eval_replace_reorder.py \
        --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
        --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
        --model-path /dev/shm/s1d_reader_fp16 \
        --checkpoint runs/natural_merge_v9/full_v10.pt \
        --verify-against reports/diagnostics/graft_recovery_hotpotqa.json \
        --out reports/diagnostics/replace_reorder_hotpotqa.json

    # re-group a finished report without a GPU
    python tools/eval_replace_reorder.py --analyze reports/diagnostics/replace_reorder_hotpotqa.json
"""
import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_core as pe                                    # noqa: E402
import eval_graft_recovery as gr                          # noqa: E402
import native_eval as nv                                  # noqa: E402
from io_utils import atomic_json, digest, sha             # noqa: E402
from runtime import critical_manifest, install, load_runtime, reset  # noqa: E402

# The experiment-1 geometry is the only geometry this driver accepts.
BLOCK = gr.SLOTS_PER_BLOCK                                # 256 slots per block
WINDOW_BLOCKS = gr.WINDOW_BLOCKS                          # 19
TARGET_BLOCKS = gr.TARGET_BLOCKS                          # 9
CORE_SLOTS = gr.CORE_SLOTS                                # 216
GRAFT_SLOTS = gr.GRAFT_SLOTS                              # 40
TARGET_SLOTS = gr.TARGET_SLOTS                            # 2,304
CORE_SLOTS_TOTAL = gr.CORE_SLOTS_TOTAL                    # 1,944
GRAFT_SLOTS_TOTAL = gr.GRAFT_SLOTS_TOTAL                  # 360

# "Replace" is experiment 2's B: the V10 vectors at the carriers' original
# positions. "Graft" is the untouched V10 output, i.e. experiment 1's Graft.
FULL, SELECTION, REPLACE, GRAFT = "Full", "Selection", "Replace", "Graft"
ARMS = (FULL, SELECTION, REPLACE, GRAFT)

# Every pairwise comparison the experiment reports, as (treatment, baseline).
PAIRS = ((REPLACE, SELECTION), (GRAFT, REPLACE), (GRAFT, SELECTION))

SCREENING_CAVEAT = gr.SCREENING_CAVEAT


def _check_geometry():
    if (WINDOW_BLOCKS, TARGET_BLOCKS) != (19, 9):
        raise ValueError(f"this experiment is pinned to 19->9, got "
                         f"{WINDOW_BLOCKS}->{TARGET_BLOCKS}")
    if CORE_SLOTS + GRAFT_SLOTS != BLOCK:
        raise ValueError("core+graft slots must tile a native block")
    if (TARGET_SLOTS, CORE_SLOTS_TOTAL, GRAFT_SLOTS_TOTAL) != (2304, 1944, 360):
        raise ValueError("unexpected slot budget")


_check_geometry()


# ---------------------------------------------------------------------------
# arm B: the V10 vectors, put back where the carrier slots used to be
# ---------------------------------------------------------------------------
def plan_inplace_replace(telemetry, selected_blocks, window_blocks, block):
    """The permutation that puts the V10 slots back where the carrier slots were.

    Pure layout: no tensors, so the caller can plan once and materialise the
    memory only for the arm that needs it (holding both compressed windows
    resident costs another 0.6 GB for nothing). Returns a `layout` dict whose
    `permutation` is asserted to be a bijection of `range(2304)`: that list, not a
    digest of it, is what the report stores, because it is the only thing that
    makes "B and C hold the same vectors, in a different order" checkable after
    the run instead of taken on faith.

    The layout is read from `slot_source_positions`, the native window position
    of every output slot, which the operator publishes precisely so a slot can be
    put back where it came from. Deriving it from the values instead would need a
    quadratic match against every candidate slot.
    """
    if "slot_source_positions" not in telemetry:
        raise ValueError("the operator did not publish slot_source_positions; arm B "
                         "cannot be built without the in-block origin of each slot")
    positions = [int(position) for position in telemetry["slot_source_positions"]]
    total = len(positions)
    if total != TARGET_SLOTS:
        raise ValueError(f"expected {TARGET_SLOTS} output slots, got {total}")

    graft_offsets = [int(offset) for offset in telemetry["bridge_slot_offsets"]]
    if len(graft_offsets) != GRAFT_SLOTS_TOTAL:
        raise ValueError(f"expected {GRAFT_SLOTS_TOTAL} graft slots, got {len(graft_offsets)}")
    # `nonzero` on the output axis is ascending by construction, so this can only
    # fire if the operator's record ever stops being the chronological one.
    if graft_offsets != sorted(graft_offsets):
        raise ValueError("the graft offsets are not in chronological order")
    graft_set = set(graft_offsets)
    core_offsets = [index for index in range(total) if index not in graft_set]
    if len(core_offsets) != CORE_SLOTS_TOTAL:
        raise ValueError(f"expected {CORE_SLOTS_TOTAL} core slots, got {len(core_offsets)}")

    # The block provenance the operator already reports must agree with the
    # positions, otherwise one of the two records is lying and B would silently
    # move the wrong vectors.
    for index, block_id in enumerate(telemetry["slot_source_blocks"]):
        if positions[index] // block != int(block_id):
            raise ValueError(f"slot {index}: position {positions[index]} is block "
                             f"{positions[index] // block}, provenance says {block_id}")

    carriers = [int(block_id) for block_id in selected_blocks]
    if len(carriers) != TARGET_BLOCKS:
        raise ValueError(f"expected {TARGET_BLOCKS} carriers, got {len(carriers)}")
    if carriers != sorted(carriers):
        raise ValueError("the carrier set must be in window order for arm B to be well defined")

    dest = [-1] * total
    per_carrier, cursor = [], 0
    for slot, carrier in enumerate(carriers):
        core_local = []
        for index in range(cursor, cursor + CORE_SLOTS):
            origin = positions[core_offsets[index]]
            if origin // block != carrier:
                raise ValueError(f"core slot {index} claims block {origin // block}, "
                                 f"not its carrier {carrier}")
            core_local.append(origin % block)
        kept = set(core_local)
        if len(kept) != CORE_SLOTS:
            raise ValueError("a carrier reuses an in-block position for two core slots")
        if core_local != sorted(core_local):
            raise ValueError("core slots are not in ascending in-block order")
        vacated = [position for position in range(block) if position not in kept]
        if len(vacated) != GRAFT_SLOTS:
            raise ValueError(f"carrier {carrier} vacates {len(vacated)} positions, "
                             f"expected {GRAFT_SLOTS}")
        for offset, local in enumerate(core_local):
            dest[slot * block + local] = core_offsets[cursor + offset]
        for offset, local in enumerate(vacated):
            dest[slot * block + local] = graft_offsets[slot * GRAFT_SLOTS + offset]
        cursor += CORE_SLOTS
        received = Counter(positions[graft_offsets[slot * GRAFT_SLOTS + offset]] // block
                           for offset in range(GRAFT_SLOTS))
        per_carrier.append(dict(
            carrier=carrier, core_slots=len(core_local), graft_slots=len(vacated),
            graft_source_blocks={str(source): int(count)
                                 for source, count in sorted(received.items())}))
    if cursor != CORE_SLOTS_TOTAL:
        raise ValueError("the carriers did not consume every core slot")
    if any(target < 0 for target in dest):
        raise ValueError("arm B left an output position unwritten")
    if sorted(dest) != list(range(total)):
        raise ValueError("arm B is not a permutation of the V10 output: the two arms "
                         "would not hold the same slot vectors")

    source_counts = Counter()
    for carrier in per_carrier:
        for source, count in carrier["graft_source_blocks"].items():
            source_counts[int(source)] += int(count)
    return dict(
        permutation_is_bijection=True,
        permutation=dest,
        permutation_sha256=digest(dest),
        slots=total,
        per_carrier=per_carrier,
        graft_slots_by_source_block={str(key): int(source_counts.get(key, 0))
                                     for key in range(window_blocks)},
        note=("arm B is `C.index_select(1, permutation)`; `permutation` is stored in "
              "full so the layout can be rebuilt from arm C's output without re-running "
              "the GPU pipeline. The 360 graft slots are dealt out in source-time order, "
              "40 per carrier in carrier order, into each carrier's ascending vacated "
              "positions"))


def apply_inplace_replace(graft_output, layout):
    """Materialise arm B: the same 2,304 vectors as `graft_output`, permuted."""
    perm = torch.as_tensor(layout["permutation"], dtype=torch.long,
                           device=graft_output.device)
    return graft_output.index_select(1, perm)


def build_inplace_replace(graft_output, telemetry, selected_blocks, window_blocks, block):
    """Plan and materialise in one call. `memory[:, i] == graft_output[:, perm[i]]`."""
    layout = plan_inplace_replace(telemetry, selected_blocks, window_blocks, block)
    return apply_inplace_replace(graft_output, layout), layout


def install_output(manager, output, picked, window_blocks):
    """Install a compressed window; the same call `install_graft` makes."""
    window = window_blocks * manager.block
    selected = torch.as_tensor(picked, dtype=torch.long, device=output.device)
    install(manager, output, window, selected_blocks=selected, fused_offsets=None)
    return selected


# ---------------------------------------------------------------------------
# one sample
# ---------------------------------------------------------------------------
def run_sample(model, tokenizer, manager, merge, prepared, args):
    """Write 19 blocks once, then answer Full / Selection / Replace / Graft."""
    chunks = [prepared["write_ids"][index:index + gr.BLOCK_TOKENS]
              for index in range(0, len(prepared["write_ids"]), gr.BLOCK_TOKENS)]
    answers = prepared["answers"]
    fingerprints = {}

    with torch.no_grad():
        gr.write_window(manager, merge, chunks)
        state = gr.snapshot(model, manager)
        source, ages, picked = gr.select_carriers(manager, args.window_blocks,
                                                  args.target_blocks)
        frozen, pinned = gr.window_locks(manager, args.window_blocks)
        # ONE compression, computed on the pre-compression window. Arm C installs
        # it verbatim; arm B installs a permutation of it. Computing it once is
        # also what guarantees the two arms share the vectors by construction.
        graft_output, telemetry = gr.compress_graft(
            manager, merge, source, picked, args.window_blocks, args.target_blocks,
            frozen, pinned)
        # Arm B is planned here but materialised per readout: keeping a second
        # 0.60 GB compressed window resident for the whole sample costs ~1.2 GB of
        # peak GPU memory against experiment 1 for nothing.
        layout = plan_inplace_replace(telemetry, picked, args.window_blocks, manager.block)
    gr.restore(model, manager, state)

    def prepare(arm):
        def run():
            gr.restore(model, manager, state)
            if arm == SELECTION:
                gr.install_selection(manager, source, picked, args.window_blocks)
            elif arm == REPLACE:
                install_output(manager, apply_inplace_replace(graft_output, layout),
                               picked, args.window_blocks)
            elif arm == GRAFT:
                install_output(manager, graft_output, picked, args.window_blocks)
            fingerprints[arm] = gr.active_fingerprint(manager)
            fingerprints[arm + "/slots"] = int(manager.used)
        return run

    arms = {arm: gr.readout(model, tokenizer, prepared["query_ids"], answers,
                            args.f1_threshold, prepare(arm)) for arm in ARMS}
    gr.restore(model, manager, state)

    provenance = gr.graft_provenance(telemetry, args.window_blocks)
    if provenance["selected_blocks"] != picked:
        raise ValueError("the compression disagrees with the shared carrier set")
    if provenance["graft_slots"] != GRAFT_SLOTS_TOTAL:
        raise ValueError("the compression did not emit 360 graft slots")
    # Distinctness is a stride-13 mean, not a hash: it catches "the compression
    # did not run" and "arm B was not permuted", nothing finer.
    if len(set(fingerprints[arm] for arm in ARMS)) != len(ARMS):
        raise ValueError("two arms read the same pool: the compression did not run")
    if fingerprints[REPLACE] == fingerprints[GRAFT]:
        raise ValueError("Replace and Graft read the same pool: arm B did not permute")
    for arm in (SELECTION, REPLACE, GRAFT):
        if int(fingerprints[arm + "/slots"]) != TARGET_SLOTS:
            raise ValueError(f"arm {arm} does not hold the {TARGET_SLOTS}-slot budget")

    return dict(arms=arms, fingerprints=fingerprints, provenance=provenance,
                layout=layout, carrier_ages=[int(ages[block]) for block in picked])


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def mcnemar_exact(up, down):
    """Two-sided exact McNemar p-value for a paired 2x2 table.

    `up`/`down` are the discordant counts. Exact rather than chi-square because
    the discordant counts here are single digits and the asymptotic test is not
    trustworthy at that size.
    """
    total = int(up) + int(down)
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(0, min(up, down) + 1))
    return min(1.0, 2.0 * tail / 2 ** total)


def _accuracy(samples, arm, threshold):
    if not samples:
        return None
    return round(100.0 * sum(1.0 for sample in samples
                             if gr._correct(sample, arm, threshold)) / len(samples), 2)


def pair_summary(samples, treatment, baseline, threshold):
    """Treatment minus baseline, as accuracy, QA F1 and gold likelihood.

    `rescued` counts samples the treatment gets right and the baseline does not;
    `harmed` is the reverse. The McNemar p-value is on exactly those two counts,
    so a net gain of +6 built from 11 up and 5 down is correctly called weak.
    """
    pairs = [sample for sample in samples
             if treatment in sample.get("arms", {}) and baseline in sample.get("arms", {})]
    if not pairs:
        return dict(n=0, treatment=treatment, baseline=baseline)
    rescued = harmed = 0
    for sample in pairs:
        left = gr._correct(sample, treatment, threshold)
        right = gr._correct(sample, baseline, threshold)
        if left and not right:
            rescued += 1
        elif right and not left:
            harmed += 1
    f1 = gr._paired(pairs, treatment, baseline, "qa_f1")
    p_gold = gr._paired(pairs, treatment, baseline, "p_gold")
    return dict(
        n=len(pairs), treatment=treatment, baseline=baseline,
        accuracy_treatment=_accuracy(pairs, treatment, threshold),
        accuracy_baseline=_accuracy(pairs, baseline, threshold),
        rescued=rescued, harmed=harmed, net=rescued - harmed,
        mcnemar_p=mcnemar_exact(rescued, harmed),
        qa_f1_treatment=round(100.0 * gr._mean(
            [sample["arms"][treatment]["qa_f1"] for sample in pairs]), 2),
        qa_f1_baseline=round(100.0 * gr._mean(
            [sample["arms"][baseline]["qa_f1"] for sample in pairs]), 2),
        qa_f1_delta=round(100.0 * f1["mean"], 2),
        qa_f1_up=f1["up"], qa_f1_down=f1["down"], qa_f1_equal=f1["equal"],
        p_gold_treatment=gr._mean([sample["arms"][treatment].get("p_gold")
                                   for sample in pairs]),
        p_gold_baseline=gr._mean([sample["arms"][baseline].get("p_gold")
                                  for sample in pairs]),
        p_gold_delta=p_gold.get("mean"),
        p_gold_up=p_gold.get("up"), p_gold_down=p_gold.get("down"))


def _outcome(baseline, replace, graft):
    """What happened to one sample between the baseline, B and C.

    A generic 8-way map instead of ad-hoc counters, so the same vocabulary
    describes the protection group (baseline correct), the rescue group
    (baseline wrong) and the ungated set, and no case is silently dropped.
    "Unchanged" is split by which way it was unchanged: merging "right everywhere"
    with "wrong everywhere" would make the ungated table's largest bucket mean two
    opposite things.
    """
    key = (bool(baseline), bool(replace), bool(graft))
    return {
        (True, True, True): "unchanged_correct",
        (True, True, False): "broken_at_reorder",
        (True, False, True): "broken_at_replace_recovered_at_reorder",
        (True, False, False): "broken_at_replace",
        (False, True, True): "fixed_at_replace",
        (False, True, False): "fixed_at_replace_broken_at_reorder",
        (False, False, True): "fixed_at_reorder",
        (False, False, False): "unchanged_wrong",
    }[key]


def _stage_counts(samples, threshold):
    counts = Counter()
    for sample in samples:
        counts[_outcome(gr._correct(sample, SELECTION, threshold),
                        gr._correct(sample, REPLACE, threshold),
                        gr._correct(sample, GRAFT, threshold))] += 1
    return dict(sorted(counts.items()))


def _transitions(samples, threshold):
    """Per-sample (Selection, Replace, Graft) correctness and the resulting stage."""
    return [dict(
        id=sample["id"],
        selection=int(gr._correct(sample, SELECTION, threshold)),
        replace=int(gr._correct(sample, REPLACE, threshold)),
        graft=int(gr._correct(sample, GRAFT, threshold)),
        outcome=_outcome(gr._correct(sample, SELECTION, threshold),
                         gr._correct(sample, REPLACE, threshold),
                         gr._correct(sample, GRAFT, threshold)),
        qa_f1=dict(selection=round(sample["arms"][SELECTION]["qa_f1"], 4),
                   replace=round(sample["arms"][REPLACE]["qa_f1"], 4),
                   graft=round(sample["arms"][GRAFT]["qa_f1"], 4)),
    ) for sample in samples]


def analyse(samples, threshold, excluded=0):
    """The two-stage attribution, on the graded samples only."""
    graded = [sample for sample in samples if sample.get("status") == "graded"]
    for sample in graded:
        missing = [arm for arm in ARMS if arm not in sample.get("arms", {})]
        if missing:
            raise ValueError(f"sample {sample.get('id')} has no {missing} arm: this report "
                             f"was not produced by this driver, and every table below "
                             f"assumes all four arms exist")
    located = [sample for sample in graded if sample.get("evidence", {}).get("located")]

    def evidence_in_dropped(sample):
        return bool(set(sample["evidence"]["blocks"])
                    & set(sample["provenance"]["dropped_blocks"]))

    # The 13 rows experiment 1 found: Full and Selection both answer correctly, so
    # the compression lost nothing there. Any loss at B or C is damage.
    protect = [sample for sample in located
               if gr._correct(sample, FULL, threshold)
               and gr._correct(sample, SELECTION, threshold)]
    # The 6 rows where dropping the evidence block cost the answer.
    rescue = [sample for sample in located
              if gr._correct(sample, FULL, threshold)
              and not gr._correct(sample, SELECTION, threshold)
              and evidence_in_dropped(sample)]

    harmed = [sample for sample in protect if not gr._correct(sample, GRAFT, threshold)]
    harmed_stages = Counter(_outcome(True,
                                     gr._correct(sample, REPLACE, threshold),
                                     gr._correct(sample, GRAFT, threshold))
                            for sample in harmed)

    return dict(
        counts=dict(
            attempted=len(graded) + int(excluded), graded=len(graded),
            excluded=int(excluded), evidence_located=len(located),
            evidence_in_dropped=len([sample for sample in located
                                     if evidence_in_dropped(sample)]),
            selection_correct=len([sample for sample in graded
                                   if gr._correct(sample, SELECTION, threshold)]),
            replace_correct=len([sample for sample in graded
                                 if gr._correct(sample, REPLACE, threshold)]),
            graft_correct=len([sample for sample in graded
                               if gr._correct(sample, GRAFT, threshold)])),
        arms={arm: gr._arm_summary(graded, arm, threshold) for arm in ARMS},
        arms_evidence_located={arm: gr._arm_summary(located, arm, threshold)
                               for arm in ARMS},
        # The three questions, in order: does replacing cost, does reordering
        # cost, and does the pair reproduce experiment 1.
        pairs={f"{treatment}_minus_{baseline}": pair_summary(
                   graded, treatment, baseline, threshold)
               for treatment, baseline in PAIRS},
        pairs_evidence_located={f"{treatment}_minus_{baseline}": pair_summary(
                   located, treatment, baseline, threshold)
               for treatment, baseline in PAIRS},
        # THE key number: the protection group, split by the stage that broke it.
        protection_group=dict(
            definition=("Full correct and Selection correct, i.e. the compression "
                        "lost nothing before any slot was replaced"),
            n=len(protect),
            harmed_by_graft=len(harmed),
            harmed_by_replace=len([sample for sample in protect
                                   if not gr._correct(sample, REPLACE, threshold)]),
            harmed_rate=(len(harmed) / len(protect) if protect else None),
            stages=_stage_counts(protect, threshold),
            harmed_stages={stage: int(count) for stage, count in sorted(harmed_stages.items())},
            harmed_detail=[dict(id=sample["id"],
                                replace=int(gr._correct(sample, REPLACE, threshold)),
                                graft=int(gr._correct(sample, GRAFT, threshold)),
                                outcome=_outcome(True,
                                                 gr._correct(sample, REPLACE, threshold),
                                                 gr._correct(sample, GRAFT, threshold)),
                                qa_f1=dict(selection=round(sample["arms"][SELECTION]["qa_f1"], 4),
                                           replace=round(sample["arms"][REPLACE]["qa_f1"], 4),
                                           graft=round(sample["arms"][GRAFT]["qa_f1"], 4)))
                            for sample in harmed],
            transitions=_transitions(protect, threshold),
            note=("`broken_at_replace` means B was already wrong, so replacing the 40 "
                  "carrier slots did the damage; `broken_at_reorder` means B was right "
                  "and only the reorder lost the answer")),
        rescue_group=dict(
            definition=("Full correct, Selection wrong, and the dropped blocks hold "
                        "the supporting sentence"),
            n=len(rescue),
            rescued_by_graft=len([sample for sample in rescue
                                  if gr._correct(sample, GRAFT, threshold)]),
            stages=_stage_counts(rescue, threshold),
            transitions=_transitions(rescue, threshold)),
        ungated=dict(
            definition="every graded sample, no evidence filter",
            stages=_stage_counts(graded, threshold),
            transitions=_transitions(graded, threshold)),
        screening_caveat=SCREENING_CAVEAT,
        f1_threshold=threshold)


# ---------------------------------------------------------------------------
# cross-check against experiment 1
# ---------------------------------------------------------------------------
def load_reference(path):
    """Open the experiment-1 report before the GPU work, not after it.

    A typo or a moved file must fail in the first second, not at the finish line
    of a 40-minute run -- and not before the summary has been written either.
    """
    reference = json.loads(Path(path).read_text())
    table = {str(sample["id"]): sample for sample in reference.get("samples") or []}
    if not table:
        raise ValueError(f"--verify-against {path} holds no samples")
    missing = [arm for arm in (FULL, SELECTION, GRAFT)
               if not any(arm in sample.get("arms", {}) for sample in table.values())]
    if missing:
        raise ValueError(f"--verify-against {path} has no {missing} arms; it is not an "
                         f"experiment-1 report")
    return reference, table


def verify_against(path, reference, table, samples):
    """Compare this run with a finished experiment-1 report, row by row.

    Arm C is supposed to BE experiment 1's Graft arm. A 100% prediction match on
    all three shared arms is the only cheap evidence that it is, and a mismatch
    here is a bug in this driver, not a result.

    The row set is checked in BOTH directions: a run over a subset of the rows
    would otherwise report `different=0, missing=[]` and look like a perfect
    match, which is the one thing this cross-check must not be able to do.
    """
    this_run = {str(sample["id"]) for sample in samples}
    report = dict(
        reference=path, reference_schema=reference.get("schema"),
        reference_samples=len(table), compared=len(this_run),
        row_count_matches_reference=(len(this_run) == len(table)),
        reference_not_in_this_run=sorted(set(table) - this_run),
        this_run_not_in_reference=sorted(this_run - set(table)))
    passed = report["row_count_matches_reference"]
    for arm in (FULL, SELECTION, GRAFT):
        same = different = absent = 0
        differing = []
        for sample in samples:
            old = table.get(str(sample["id"]))
            if old is None or arm not in old.get("arms", {}):
                absent += 1
                continue
            if old["arms"][arm]["prediction"] == sample["arms"][arm]["prediction"]:
                same += 1
            else:
                different += 1
                differing.append(dict(
                    id=sample["id"],
                    old=old["arms"][arm]["prediction"],
                    new=sample["arms"][arm]["prediction"],
                    old_qa_f1=round(old["arms"][arm]["qa_f1"], 4),
                    new_qa_f1=round(sample["arms"][arm]["qa_f1"], 4)))
        report[arm] = dict(identical=same, different=different, not_in_reference=absent,
                           differing=differing[:20])
        passed = passed and different == 0 and absent == 0 and same == len(table)
    report["passed"] = passed
    return report


# ---------------------------------------------------------------------------
# CPU-only self-test: the permutation is the whole experiment, so it is checked
# ---------------------------------------------------------------------------
def _self_test():
    """Check arm B's permutation, the stage map and the McNemar value, on CPU.

    The permutation is the one thing here that cannot be eyeballed from a report:
    if it were wrong, B would silently hold different vectors and the B -> C
    comparison would measure nothing. This builds a synthetic 19 -> 9 layout with
    uniquely identifiable slot values and asserts every output position.
    """
    generator = torch.Generator().manual_seed(0)
    carriers = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    dropped = [index for index in range(WINDOW_BLOCKS) if index not in carriers]
    if len(carriers) != TARGET_BLOCKS or len(dropped) != WINDOW_BLOCKS - TARGET_BLOCKS:
        raise ValueError("self-test carrier set is wrong")
    core_local = {carrier: sorted(torch.randperm(BLOCK, generator=generator)[:CORE_SLOTS].tolist())
                  for carrier in carriers}
    per_dropped = GRAFT_SLOTS_TOTAL // len(dropped)
    graft_local = {block: sorted(torch.randperm(BLOCK, generator=generator)[:per_dropped].tolist())
                   for block in dropped}

    slots = [(carrier * BLOCK + position, "core", carrier)
             for carrier in carriers for position in core_local[carrier]]
    slots += [(block * BLOCK + position, "graft", block)
              for block in dropped for position in graft_local[block]]
    if len(slots) != TARGET_SLOTS:
        raise ValueError(f"self-test built {len(slots)} slots, expected {TARGET_SLOTS}")
    slots.sort(key=lambda item: item[0])
    telemetry = dict(
        bridge_slot_offsets=[index for index, item in enumerate(slots) if item[1] == "graft"],
        slot_source_blocks=[item[2] for item in slots],
        slot_source_positions=[item[0] for item in slots])
    # Each slot gets a unique scalar, so the permutation can be read back exactly.
    graft_output = torch.arange(TARGET_SLOTS, dtype=torch.float32).view(1, TARGET_SLOTS, 1)
    memory, layout = build_inplace_replace(graft_output, telemetry, carriers,
                                           WINDOW_BLOCKS, BLOCK)
    if tuple(memory.shape) != tuple(graft_output.shape):
        raise ValueError("arm B changed the memory shape")
    flat = [int(value) for value in memory[0, :, 0].tolist()]
    if sorted(flat) != list(range(TARGET_SLOTS)):
        raise ValueError("arm B is not a permutation of arm C")
    origin = {item[0]: index for index, item in enumerate(slots)}
    for slot, carrier in enumerate(carriers):
        for position in core_local[carrier]:
            if flat[slot * BLOCK + position] != origin[carrier * BLOCK + position]:
                raise ValueError(f"carrier {carrier}: core slot {position} was moved")
        vacated = [position for position in range(BLOCK) if position not in set(core_local[carrier])]
        for offset, position in enumerate(vacated):
            want = telemetry["bridge_slot_offsets"][slot * GRAFT_SLOTS + offset]
            if flat[slot * BLOCK + position] != want:
                raise ValueError(f"carrier {carrier}: vacated slot {position} got the "
                                 f"wrong graft vector")
    if sum(layout["graft_slots_by_source_block"].values()) != GRAFT_SLOTS_TOTAL:
        raise ValueError("arm B lost graft slots")
    if len(layout["per_carrier"]) != TARGET_BLOCKS:
        raise ValueError("arm B lost a carrier")

    # The guards that would otherwise fail silently on a real sample.
    for name, broken in (
            ("unsorted graft offsets",
             dict(telemetry, bridge_slot_offsets=list(reversed(telemetry["bridge_slot_offsets"])))),
            ("positions that disagree with the provenance",
             dict(telemetry, slot_source_positions=[0] * TARGET_SLOTS))):
        try:
            build_inplace_replace(graft_output, broken, carriers, WINDOW_BLOCKS, BLOCK)
        except ValueError:
            pass
        else:
            raise ValueError(f"arm B accepted a broken layout: {name}")

    # Experiment 1's ungated table was 11 rescued / 5 harmed, and this is the
    # value the report quoted for it: a regression test on the statistic.
    p_value = mcnemar_exact(11, 5)
    if abs(p_value - 0.2101) > 0.0005:
        raise ValueError(f"McNemar(11, 5) = {p_value}, expected 0.2101")
    if mcnemar_exact(0, 0) != 1.0:
        raise ValueError("McNemar(0, 0) must be 1.0")
    if abs(mcnemar_exact(3, 3) - 1.0) > 1e-12:
        raise ValueError("a balanced table must not be significant")

    stages = {
        (True, True, True): "unchanged_correct",
        (True, True, False): "broken_at_reorder",
        (True, False, False): "broken_at_replace",
        (True, False, True): "broken_at_replace_recovered_at_reorder",
        (False, True, True): "fixed_at_replace",
        (False, True, False): "fixed_at_replace_broken_at_reorder",
        (False, False, True): "fixed_at_reorder",
        (False, False, False): "unchanged_wrong"}
    if len(set(stages.values())) != 8:
        raise ValueError("the stage vocabulary must name all eight outcomes distinctly")
    for key, want in stages.items():
        if _outcome(*key) != want:
            raise ValueError(f"_outcome{key} = {_outcome(*key)}, expected {want}")

    return dict(self_test="ok", carriers=len(carriers), dropped=len(dropped),
                slots=TARGET_SLOTS, graft_slots=GRAFT_SLOTS_TOTAL,
                permutation_sha256=layout["permutation_sha256"],
                mcnemar_11_5=round(p_value, 6))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def build_protocol(args, task):
    return dict(
        experiment="2 replacement and reorder damage",
        task=task, data_dir=args.data_dir,
        arms=list(ARMS),
        arm_definitions=dict(
            Full="the 19 written blocks, nothing compressed",
            Selection="the 9 carrier blocks, untouched",
            Replace=("each carrier keeps its own 216 core slots at their original "
                     "in-block positions; the 40 vacated positions hold the V10 graft "
                     "slots"),
            Graft="the V10 output exactly as the operator emits it (source-time order)"),
        window_blocks=args.window_blocks, target_blocks=args.target_blocks,
        block_tokens=gr.BLOCK_TOKENS, slots_per_block=BLOCK,
        write_tokens=args.window_blocks * gr.BLOCK_TOKENS,
        full_slots=args.window_blocks * BLOCK,
        target_slots=args.target_blocks * BLOCK,
        core_slots_total=args.target_blocks * CORE_SLOTS,
        graft_slots_total=args.target_blocks * GRAFT_SLOTS,
        compressions_per_sample=1,
        write_stream="document prefix only (the leading template instruction is dropped)",
        query=gr.QUERY_SUFFIX,
        operator_mode="native",
        operator_note=("carrier and graft slots are copied verbatim; no same-block "
                       "condensation, no orthogonal context delta, no core reinforcement"),
        carrier_selection=("select_window_carrier_blocks runs once on the "
                           "pre-compression window; Selection, Replace and Graft share it"),
        replace_construction=("arm B is `C.index_select(1, perm)` with `perm` a verified "
                              "bijection of range(2304), so B and C hold the same 2,304 "
                              "vectors and differ only in order"),
        graft_assignment=("the 360 graft slots are dealt out in source-time order, 40 per "
                          "carrier in carrier order, into each carrier's ascending vacated "
                          "positions"),
        attribution_note=("A -> B is the JOINT net effect of deleting 40 carrier slots and "
                          "adding 40 graft slots; that single difference cannot separate the "
                          "deletion loss from the graft gain. B -> C is the reorder alone, "
                          "because the vector set is identical"),
        decoding="greedy, first non-blank line, max_new_tokens=%d" % gr.MAX_NEW,
        scoring=("official LongBench HotpotQA QA F1 (max over golds); EM; "
                 "gold-answer log-likelihood"),
        significance="two-sided exact McNemar on the discordant pairs",
        f1_threshold=args.f1_threshold,
        supporting_facts=args.supporting_facts_glob,
        seed=args.seed, training_performed=False,
        screening_caveat=SCREENING_CAVEAT)


def dry_run(args, facts, tokenizer):
    rows, prepared, excluded = gr.prepare_rows(args, tokenizer, facts)
    located = [sample for sample in prepared if sample["evidence"]["located"]]
    histogram = Counter(block for sample in located for block in sample["evidence"]["blocks"])
    plan = dict(
        dry_run=True, out=args.out,
        protocol=build_protocol(args, args.task),
        counts=dict(rows=len(rows), eligible=len(prepared), excluded=len(excluded),
                    supporting_facts_joined=len(
                        [sample for sample in prepared
                         if sample["evidence"]["supporting_sentences"]]),
                    sentence_located_in_context=len(
                        [sample for sample in prepared
                         if sample["evidence"]["supporting_sentences_located"]]),
                    evidence_in_window=len(located),
                    sentence_found_but_outside_window=len(
                        [sample for sample in prepared
                         if sample["evidence"]["supporting_sentences_located"]
                         and not sample["evidence"]["located"]])),
        evidence_block_histogram={str(block): histogram[block] for block in sorted(histogram)},
        excluded_examples=excluded[:5],
        eligible_ids=[sample["id"] for sample in prepared],
        arms=list(ARMS),
        readouts_per_sample=len(ARMS),
        note=("arm B is built from arm C's output, so the two share their 2,304 slot "
              "vectors by construction; the dry run does not need a GPU"))
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return plan


def evaluate(args):
    facts = gr.load_supporting_facts(args.supporting_facts_glob)
    if not facts:
        print("WARNING: no supporting facts; the protection/rescue grouping will be "
              "empty. Pass --supporting-facts-glob for the actual experiment.",
              file=sys.stderr)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    if args.dry_run:
        return dry_run(args, facts, tokenizer)
    if not args.checkpoint or not Path(args.checkpoint).is_file():
        raise ValueError("the Replace/Graft arms need the trained operator: pass --checkpoint")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("output exists; refuse overwrite")
    reference = table = None
    if args.verify_against:
        reference, table = load_reference(args.verify_against)

    rows, prepared, excluded = gr.prepare_rows(args, tokenizer, facts)
    if not prepared:
        raise ValueError("no eligible sample: every context is shorter than the window")
    if args.seed != 0:
        raise ValueError("the fixed evaluation seed is 0")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.cuda_device))
    nv.seed_everything(args.seed)
    pe.install_generation_compatibility()

    checkpoint_training = torch.load(args.checkpoint, map_location="cpu").get("training")
    model, tokenizer, manager, merge, initial, cfg, saved = load_runtime(
        args.model_path, checkpoint=args.checkpoint, train=False,
        architecture=None, budget_blocks=gr.BUDGET_BLOCKS,
        strict_code_hash=args.strict_code_hash, training=checkpoint_training)
    tokenizer.pad_token = tokenizer.eos_token
    if int(model.num_blocks) != gr.BUDGET_BLOCKS or int(model.num_tokens) != BLOCK:
        raise ValueError("expected the native 50-block / 256-slot MemoryLLM reader")
    merge.operator_mode = "native"       # experiment 2: native slots only

    # `manifold_graft.py` is NOT in CRITICAL_MERGE_FILES, so `--strict-code-hash`
    # cannot see a change to it. This run added a telemetry field there, so the
    # comparison is recorded explicitly instead of being assumed: a False here
    # means "the operator file changed since the checkpoint was written", and the
    # only thing that vouches for the numerics is `--verify-against`.
    operator_file = "methods/prlmc/manifold_graft.py"
    operator_now = sha(ROOT / operator_file)
    operator_then = ((saved or {}).get("source_manifest") or {}).get(operator_file)
    if operator_then != operator_now:
        print(f"WARNING: {operator_file} differs from the checkpoint's manifest "
              f"(ckpt {str(operator_then)[:8]} != code {operator_now[:8]}); the change "
              f"must be non-numeric, and --verify-against is what checks that",
              file=sys.stderr)

    report = dict(
        schema="replace_reorder_results/1", status="running",
        protocol=dict(build_protocol(args, args.task),
                      operator_mode_applied=merge.operator_mode),
        identity=dict(
            model_path=args.model_path, checkpoint=args.checkpoint,
            checkpoint_sha256=sha(args.checkpoint),
            checkpoint_code_hash_verified=saved.get("code_hash_verified") if saved else None,
            data_dir=args.data_dir,
            data_sha256=sha(Path(args.data_dir) / (args.task + ".jsonl")),
            supporting_facts_glob=args.supporting_facts_glob,
            entrypoint_sha256=sha(__file__),
            compression_entrypoint_sha256=sha(Path(gr.__file__)),
            operator_sha256=operator_now,
            operator_sha256_in_checkpoint=operator_then,
            operator_matches_checkpoint=(operator_then == operator_now),
            critical_manifest=critical_manifest(),
            reader_frozen=True),
        excluded=excluded, samples=[], training_performed=False,
        optimizer_created=False,
        full_split=(args.max_samples is None and args.sample_ids is None))
    atomic_json(output, report)

    started = time.monotonic()
    for position, sample in enumerate(prepared):
        sample_started = time.monotonic()
        nv.seed_everything(args.seed)
        reset(manager, initial)
        with torch.no_grad():
            manager._tile_free_region()
            manager._sync_active_mask()
        result = run_sample(model, tokenizer, manager, merge, sample, args)
        record = dict(sample_index=sample["sample_index"], id=sample["id"],
                      question=sample["question"], answers=sample["answers"],
                      context_sha256=sample["context_sha256"],
                      source_row_sha256=sample["source_row_sha256"],
                      context_tokens=sample["context_tokens"],
                      available_blocks=sample["available_blocks"],
                      evidence=sample["evidence"], status="graded",
                      seconds=time.monotonic() - sample_started, **result)
        report["samples"].append(record)
        report["progress"] = dict(sample=position + 1, total=len(prepared))
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output, report)
        print(json.dumps(dict(sample=position + 1, total=len(prepared), id=sample["id"],
                              selection=result["arms"][SELECTION]["correct"],
                              replace=result["arms"][REPLACE]["correct"],
                              graft=result["arms"][GRAFT]["correct"],
                              f1=round(result["arms"][SELECTION]["qa_f1"], 3),
                              seconds=round(record["seconds"], 1))), flush=True)

    report["summary"] = analyse(report["samples"], args.f1_threshold,
                                excluded=len(report["excluded"]))
    if args.verify_against:
        report["verification_against_experiment_1"] = verify_against(
            args.verify_against, reference, table, report["samples"])
    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["status"] = "complete"
    atomic_json(output, report)
    print(json.dumps(report["summary"], indent=1, ensure_ascii=False))
    verification = report.get("verification_against_experiment_1")
    if verification and not verification["passed"]:
        # Written to disk first: the report is still valid evidence for B vs C,
        # it just is not the experiment-1 reproduction it claims to be.
        print("WARNING: this run does NOT reproduce experiment 1 row for row; "
              f"{json.dumps({arm: verification[arm]['different'] for arm in (FULL, SELECTION, GRAFT)})} "
              "differing predictions. Treat arm C as a different implementation.",
              file=sys.stderr)


def analyze_existing(path, threshold=None):
    report = json.loads(Path(path).read_text())
    if threshold is None:
        threshold = report.get("protocol", {}).get("f1_threshold", 0.5)
    summary = analyse(report.get("samples") or [], threshold,
                      excluded=len(report.get("excluded") or []))
    # The stored per-arm `correct` flags are derived from the same QA F1, so a
    # moved threshold has to move them too: otherwise the summary and the records
    # it was computed from disagree inside one file.
    for sample in report.get("samples") or []:
        for record in (sample.get("arms") or {}).values():
            if "qa_f1" in record:
                record["correct"] = int(record["qa_f1"] >= threshold)
                record["f1_threshold"] = threshold
    report["summary"] = summary
    report.setdefault("protocol", {})["f1_threshold"] = threshold
    atomic_json(Path(path), report)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None,
                   help="directory holding <task>.jsonl (LongBench v1 test files)")
    p.add_argument("--task", default="hotpotqa", choices=("hotpotqa",))
    p.add_argument("--supporting-facts-glob", dest="supporting_facts_glob", default=None,
                   help="HotpotQA dev parquet glob providing supporting_facts; without it "
                        "the evidence location (and therefore the grouping) is unknown")
    p.add_argument("--model-path", default=None,
                   help="fp16 MemoryLLM reader directory (default: the eval_core constant)")
    p.add_argument("--checkpoint", default=None,
                   help="trained V10 operator, e.g. runs/natural_merge_v9/full_v10.pt")
    p.add_argument("--out", default=None)
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--window-blocks", dest="window_blocks", type=int, default=WINDOW_BLOCKS)
    p.add_argument("--target-blocks", dest="target_blocks", type=int, default=TARGET_BLOCKS)
    p.add_argument("--f1-threshold", dest="f1_threshold", type=float, default=None,
                   help="QA F1 at or above this counts as a correct answer (0.5 for a run); "
                        "the stage attribution is defined on this binary, and --analyze "
                        "recomputes it at the value passed here")
    p.add_argument("--sample-ids", dest="sample_ids", default=None,
                   help="JSON list of LongBench row ids; evaluates exactly these rows")
    p.add_argument("--max-samples", dest="max_samples", type=int, default=None)
    p.add_argument("--verify-against", dest="verify_against", default=None,
                   help="a finished experiment-1 report; the run then records a row-by-row "
                        "prediction comparison for Full/Selection/Graft")
    p.add_argument("--strict-code-hash", dest="strict_code_hash", default="true",
                   choices=("true", "false"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="resolve the sample set and evidence blocks, print the plan, "
                        "load no model and write no file")
    p.add_argument("--analyze", default=None,
                   help="re-group an existing report without inference")
    p.add_argument("--self-test", dest="self_test", action="store_true",
                   help="run the CPU-only permutation, stage-map and McNemar checks "
                        "and exit; no data, model or GPU needed")
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        print(json.dumps(_self_test(), indent=2, ensure_ascii=False))
        return
    if args.analyze:
        analyze_existing(args.analyze, args.f1_threshold)
        return
    args.strict_code_hash = args.strict_code_hash == "true"
    if args.model_path is None:
        args.model_path = pe.MODEL_PATH
    if not args.data_dir:
        raise SystemExit("--data-dir is required")
    if (args.window_blocks, args.target_blocks) != (WINDOW_BLOCKS, TARGET_BLOCKS):
        raise SystemExit(f"this experiment is pinned to {WINDOW_BLOCKS}->{TARGET_BLOCKS}; "
                         f"got {args.window_blocks}->{args.target_blocks}")
    if args.out is None and not args.dry_run:
        raise SystemExit("--out is required unless --dry-run")
    if args.f1_threshold is None:
        args.f1_threshold = 0.5
    evaluate(args)


if __name__ == "__main__":
    main()
