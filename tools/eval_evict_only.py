#!/usr/bin/env python3
"""Experiment 3 -- Evict-Only: is the Replace damage the slot DELETION or the INSERTION?

What experiment 2 left open
---------------------------
Replacing each carrier's 40 slots broke 4 of the 13 protection-group questions, but
a replacement does two things at once:

  (i)  it REMOVES the carrier's own Bottom-40 slots from the memory, and
  (ii) it ADDS the 360 graft slots back into the positions it vacated.

This driver adds the single arm that separates them:

    A Selection   the 9 carrier blocks, untouched                       2,304 readable slots
    B Evict       A's memory minus the same 360 slots, nothing added    1,944 readable slots
    C Replace     B's removal, then the 360 graft slots in the vacated
                  positions                                             2,304 readable slots

  A -> B  the deletion loss ALONE   (no foreign vector anywhere in the memory)
  B -> C  the insertion effect ALONE (same carriers, same positions, vectors added)
  A -> C  experiment 2's "replace" number, re-measured here as a cross-check

Every question it answers is a per-sample pattern of (A, B, C) correctness:

    A=1 B=0 C=0   deleting the 40 slots was already enough to break it
    A=1 B=1 C=0   deletion was harmless; INSERTING the foreign slots broke it
    A=1 B=0 C=1   deletion hurt, but the graft slots restored answerability
    A=0 B=0 C=1   only the inserted slots provide a net gain
    ... all eight patterns are named, none is dropped (see `OUTCOMES`).

Arm B is arm A's tensor, byte for byte
--------------------------------------
The evicted slots are NOT zeroed, NOT re-packed and NOT replaced by filler: the
carrier's own vectors stay exactly where they were and the reader is simply told
not to attend to them, through `memory_active_mask` -- the Phase-2 mechanism the
manager already publishes, which puts `finfo.min` on the attention score of every
inactive pool slot. Two consequences make this a single-factor experiment:

  * arm B's installed window is compared against arm A's with `torch.equal` on
    every sample -- same shape, same dtype, same bits -- so the two pools are
    measured to be identical and the mask is the only difference between A and B
    (the stride-13 mean fingerprint is recorded as telemetry, NOT as this proof:
    it is an average over the pool, and experiment 2 found all four of its arms
    within 1e-4 of each other);
  * each carrier keeps its 216 surviving slots at their ORIGINAL in-block
    positions, so "B deleted 40 slots" does not smuggle in a re-packing.

The mask is non-contiguous (40 holes inside each of 9 blocks), and the manager's
own `_sync_active_mask` recomputes the mask whenever `used` moves, so the mask is
applied AFTER the install in every readout pass and the active-slot count is
re-checked after generation (`mask_after_readout` in each sample, asserted to be
1,944). What is being verified is not "a mask was set" but "the reader saw 1,944
active slots during the answer".

Capacity caveat (recorded in the report)
----------------------------------------
Arm B reads 1,944 slots where A and C read 2,304. That is deliberate: this is a
mechanism diagnostic that isolates WHERE the damage happens, not an equal-capacity
comparison, and no number here should be read as "B is a better system". The only
claim B can support is a negative one -- if B is already wrong, the foreign slots
are innocent.

Fixed conditions (everything else is experiment 2's setting)
------------------------------------------------------------
Same 158 eligible HotpotQA rows, frozen MemoryLLM-8B reader, one 19 -> 9
compression per sample, the same 9 carriers from `select_window_carrier_blocks`,
the same top-216 core selection, the same 360 graft slots, `operator_mode`
"native" (no condensation, no orthogonal delta, no core reinforcement), no
training, greedy decoding, official LongBench QA F1 with the 0.5 cut, plus the
gold-answer log-likelihood. The write stream is the document prefix only, so
block b covers exactly tokens [512b, 512(b+1)).

Usage
-----
    # what would run, without loading the model
    python tools/eval_evict_only.py --dry-run \
        --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
        --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
        --model-path /dev/shm/s1d_reader_fp16 \
        --out reports/diagnostics/evict_only_hotpotqa.json

    # the experiment
    python tools/eval_evict_only.py \
        --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
        --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
        --model-path /dev/shm/s1d_reader_fp16 \
        --checkpoint runs/natural_merge_v9/full_v10.pt \
        --verify-against reports/diagnostics/graft_span_hotpotqa.json \
        --out reports/diagnostics/evict_only_hotpotqa.json

    # re-group a finished report without a GPU
    python tools/eval_evict_only.py --analyze reports/diagnostics/evict_only_hotpotqa.json

    # the CPU-only checks (layout, mask arithmetic, outcome map, McNemar)
    python tools/eval_evict_only.py --self-test
"""
import argparse
import json
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
import eval_replace_reorder as er                          # noqa: E402
import native_eval as nv                                  # noqa: E402
from io_utils import atomic_json, digest, sha             # noqa: E402
from runtime import critical_manifest, load_runtime, reset  # noqa: E402

# The experiment-2 geometry is the only geometry this driver accepts.
BLOCK = gr.SLOTS_PER_BLOCK                                # 256 slots per block
WINDOW_BLOCKS = gr.WINDOW_BLOCKS                          # 19
TARGET_BLOCKS = gr.TARGET_BLOCKS                          # 9
CORE_SLOTS = gr.CORE_SLOTS                                # 216
GRAFT_SLOTS = gr.GRAFT_SLOTS                              # 40
TARGET_SLOTS = gr.TARGET_SLOTS                            # 2,304
CORE_SLOTS_TOTAL = gr.CORE_SLOTS_TOTAL                    # 1,944
GRAFT_SLOTS_TOTAL = gr.GRAFT_SLOTS_TOTAL                  # 360

# "Evict" is this experiment's B: arm A's memory with the 360 replaceable slots
# removed from the reader's attention. "Replace" is experiment 2's B, re-measured
# here so the decomposition is paired inside one run.
FULL, SELECTION, EVICT, REPLACE = "Full", "Selection", "Evict", "Replace"
ARMS = (FULL, SELECTION, EVICT, REPLACE)

# Every pairwise comparison the experiment reports, as (treatment, baseline).
PAIRS = ((EVICT, SELECTION), (REPLACE, EVICT), (REPLACE, SELECTION))

SCREENING_CAVEAT = gr.SCREENING_CAVEAT

CAPACITY_CAVEAT = (
    f"arm Evict reads {CORE_SLOTS_TOTAL:,} slots where Selection and Replace read "
    f"{TARGET_SLOTS:,}. That is the point -- it removes the foreign vectors without "
    f"adding anything -- but it means A -> B is a mechanism diagnostic, not an "
    f"equal-capacity comparison. The only inference B licenses is negative: if a "
    f"sample is already wrong at B, the inserted slots are innocent")

MASK_NOTE = (
    "`memory_active_mask` puts `finfo.min` on the attention score of every inactive "
    "pool slot, so a masked slot is unreadable rather than merely unused. The mask is "
    "non-contiguous (40 holes inside each of 9 blocks) and the manager recomputes it "
    "whenever `used` moves, so it is re-applied after every install and the active "
    "count is checked again after generation")

# The 8 patterns of (A, B, C) correctness. Named rather than counted inline, so the
# same vocabulary describes the protection group, the rescue group and the ungated
# set, and no case can be silently dropped.
OUTCOMES = {
    (True, True, True): "unchanged_correct",
    (True, False, False): "broken_at_evict",
    (True, True, False): "broken_at_insert",
    (True, False, True): "broken_at_evict_recovered_at_insert",
    (False, True, True): "fixed_at_evict",
    (False, True, False): "fixed_at_evict_broken_at_insert",
    (False, False, True): "fixed_at_insert",
    (False, False, False): "unchanged_wrong",
}

# What each pattern licenses, in the decision table this experiment exists for.
OUTCOME_VOCABULARY = {
    "unchanged_correct": dict(
        pattern="A=1 B=1 C=1",
        meaning="right everywhere: the compression and the replacement both cost nothing"),
    "broken_at_evict": dict(
        pattern="A=1 B=0 C=0",
        meaning="DELETING the carrier's 40 slots was already enough to lose the answer; "
                "the foreign slots cannot be blamed"),
    "broken_at_insert": dict(
        pattern="A=1 B=1 C=0",
        meaning="deletion was harmless and INSERTING the 360 graft slots lost the "
                "answer: foreign content or its position assignment is the suspect"),
    "broken_at_evict_recovered_at_insert": dict(
        pattern="A=1 B=0 C=1",
        meaning="deletion hurt but the inserted slots restored answerability"),
    "fixed_at_evict": dict(
        pattern="A=0 B=1 C=1",
        meaning="wrong with all 256 slots, right once the noisy 40 were removed"),
    "fixed_at_evict_broken_at_insert": dict(
        pattern="A=0 B=1 C=0",
        meaning="removal fixed it, insertion broke it again"),
    "fixed_at_insert": dict(
        pattern="A=0 B=0 C=1",
        meaning="only the inserted slots provide a net gain"),
    "unchanged_wrong": dict(
        pattern="A=0 B=0 C=0",
        meaning="wrong at every stage: nothing measurable to lose"),
}


def _check_geometry():
    if (WINDOW_BLOCKS, TARGET_BLOCKS) != (19, 9):
        raise ValueError(f"this experiment is pinned to 19->9, got "
                         f"{WINDOW_BLOCKS}->{TARGET_BLOCKS}")
    if CORE_SLOTS + GRAFT_SLOTS != BLOCK:
        raise ValueError("core+graft slots must tile a native block")
    if (TARGET_SLOTS, CORE_SLOTS_TOTAL, GRAFT_SLOTS_TOTAL) != (2304, 1944, 360):
        raise ValueError("unexpected slot budget")
    # Experiments 2's arms are referenced by name, so a rename there must fail here
    # rather than silently compare against a key nobody writes any more.
    if not {FULL, SELECTION, REPLACE}.issubset(set(er.ARMS)):
        raise ValueError("eval_replace_reorder no longer defines Full/Selection/Replace")


_check_geometry()


# ---------------------------------------------------------------------------
# arm B: which slots to evict, and how to evict them
# ---------------------------------------------------------------------------
def evicted_window_positions(layout):
    """The 360 positions of the compressed window that hold a graft vector in arm C.

    Arm B must remove exactly these. Any other set would mean A -> B and B -> C
    decompose two different operations and the attribution tables would be
    measuring nothing. The set is re-derived from arm C's own permutation instead
    of from a second, parallel rule, so the two arms cannot drift apart.

    Positions are compressed-window offsets in [0, 2304): carrier `k` occupies
    `[256k, 256(k+1))`, so a position's block is its carrier, and the 40 holes of
    each carrier are its vacated in-block positions.
    """
    permutation = [int(source) for source in layout["permutation"]]
    if len(permutation) != TARGET_SLOTS:
        raise ValueError(f"expected a {TARGET_SLOTS}-slot permutation, "
                         f"got {len(permutation)}")
    graft = set()
    for entry in layout["per_carrier"]:
        offsets = [int(offset) for offset in entry["graft_offsets"]]
        if len(offsets) != GRAFT_SLOTS:
            raise ValueError(f"carrier {entry['carrier']} holds {len(offsets)} graft "
                             f"vectors, expected {GRAFT_SLOTS}")
        graft.update(offsets)
    if len(graft) != GRAFT_SLOTS_TOTAL:
        raise ValueError(f"the carriers hold {len(graft)} distinct graft vectors, "
                         f"expected {GRAFT_SLOTS_TOTAL}")

    evicted = [position for position, source in enumerate(permutation) if source in graft]
    core = [position for position, source in enumerate(permutation) if source not in graft]
    if len(evicted) != GRAFT_SLOTS_TOTAL:
        raise ValueError(f"arm C puts graft vectors in {len(evicted)} positions, "
                         f"expected {GRAFT_SLOTS_TOTAL}")
    if len(core) != CORE_SLOTS_TOTAL:
        raise ValueError(f"arm C keeps {len(core)} carrier positions, "
                         f"expected {CORE_SLOTS_TOTAL}")
    if sorted(evicted + core) != list(range(TARGET_SLOTS)):
        raise ValueError("the evicted and kept positions do not partition the window")
    per_carrier = Counter(position // BLOCK for position in evicted)
    if sorted(per_carrier) != list(range(TARGET_BLOCKS)):
        raise ValueError(f"the eviction skips a carrier: {sorted(per_carrier)}")
    if any(count != GRAFT_SLOTS for count in per_carrier.values()):
        raise ValueError(f"the eviction is not {GRAFT_SLOTS} slots per carrier: "
                         f"{sorted(per_carrier.items())}")
    contiguous = evicted == list(range(evicted[0], evicted[0] + len(evicted)))
    return dict(
        positions=evicted,
        core_positions=core,
        per_carrier={str(carrier): int(count)
                     for carrier, count in sorted(per_carrier.items())},
        contiguous=bool(contiguous),
        digest=digest(evicted))


def install_evict(manager, model, source, picked, eviction, window_blocks):
    """Arm B: install exactly what arm A installs, then unread 360 of its slots.

    `gr.install_selection` is called with the same arguments as arm A, so the two
    arms hold the same bytes by construction; the caller then asserts the pools are
    float-identical. Nothing is zero-filled: the evicted slots still hold the
    carrier's own vectors and are simply removed from the reader's attention, which
    is the difference between "these slots are gone" and "these slots are zeros the
    model still looks at".

    The mask is applied AFTER the install because `install` calls
    `_tile_free_region`, which calls `_sync_active_mask`, which would overwrite any
    mask set earlier.
    """
    gr.install_selection(manager, source, picked, window_blocks)
    mask = model.memory_active_mask
    used = int(manager.used)
    before = int(mask.sum())
    if before != used:
        raise ValueError(f"the installed window holds {before} active slots but the "
                         f"manager reports {used}: the eviction would be measured "
                         f"against the wrong baseline")
    offset = int(manager.total - used)
    index = torch.as_tensor([offset + position for position in eviction["positions"]],
                            dtype=torch.long, device=mask.device)
    if int(index.min()) < 0 or int(index.max()) >= int(manager.total):
        raise ValueError("the eviction address is outside the pool")
    if not bool(mask.index_select(0, index).all()):
        raise ValueError("an evicted slot was already inactive: the window layout "
                         "disagrees with the mask the install published")
    mask[index] = False
    after = int(mask.sum())
    if before - after != len(eviction["positions"]):
        raise ValueError(f"the mask lost {before - after} slots, expected "
                         f"{len(eviction['positions'])}")
    if after != CORE_SLOTS_TOTAL:
        raise ValueError(f"arm Evict ended with {after} active slots, expected "
                         f"{CORE_SLOTS_TOTAL}; the pool carries content this driver "
                         f"does not know about")
    return dict(offset=offset, active_before=before, active_after=after,
                mask_syncs=int(manager._active_mask_count))


def installed_window(manager):
    """The exact bytes of the compressed window currently resident in the pool.

    Returned as a view, not a copy: the caller decides whether it needs to keep it.
    `install` places the candidate at `[total - used, total - used + 2304)`, so this
    is the region two compressed arms are supposed to agree (or disagree) on.
    """
    if int(manager.used) != TARGET_SLOTS:
        raise ValueError(f"the pool holds {int(manager.used)} slots, expected the "
                         f"{TARGET_SLOTS}-slot compressed window")
    start = int(manager.total - manager.used)
    return manager._memory[:, start:start + TARGET_SLOTS]


# ---------------------------------------------------------------------------
# one sample
# ---------------------------------------------------------------------------
def run_sample(model, tokenizer, manager, merge, prepared, args):
    """Write 19 blocks once, then answer Full / Selection / Evict / Replace."""
    chunks = [prepared["write_ids"][index:index + gr.BLOCK_TOKENS]
              for index in range(0, len(prepared["write_ids"]), gr.BLOCK_TOKENS)]
    answers = prepared["answers"]
    fingerprints, readouts = {}, {}

    with torch.no_grad():
        gr.write_window(manager, merge, chunks)
        state = gr.snapshot(model, manager)
        source, ages, picked = gr.select_carriers(manager, args.window_blocks,
                                                  args.target_blocks)
        frozen, pinned = gr.window_locks(manager, args.window_blocks)
        # ONE compression: arm C installs it through the exp-2 permutation, and the
        # same permutation says which positions arm B must evict.
        graft_output, telemetry = gr.compress_graft(
            manager, merge, source, picked, args.window_blocks, args.target_blocks,
            frozen, pinned)
        layout = er.plan_inplace_replace(telemetry, picked, args.window_blocks,
                                         manager.block)
        eviction = evicted_window_positions(layout)
    gr.restore(model, manager, state)

    # Arm A's installed window is kept so arm B can be compared against it element by
    # element. A clone is 604 MB of the 40 GB card and it is what turns "A and B hold
    # the same memory" from an argument about the code into a measurement of the pool.
    captured = {}

    def prepare(arm):
        def run():
            gr.restore(model, manager, state)
            if arm == SELECTION:
                gr.install_selection(manager, source, picked, args.window_blocks)
            elif arm == EVICT:
                info = install_evict(manager, model, source, picked, eviction,
                                     args.window_blocks)
                readouts[arm] = info
            elif arm == REPLACE:
                er.install_output(manager, er.apply_inplace_replace(graft_output, layout),
                                  picked, args.window_blocks)
            if arm == SELECTION and "window" not in captured:
                captured["window"] = installed_window(manager).clone()
            elif arm == EVICT:
                # The exact check: same shape, same dtype, same bits. The stride-13
                # fingerprint recorded below is telemetry only -- it is a mean over
                # the pool, so two DIFFERENT memories can share it (experiment 2
                # measured all four of its arms within 1e-4 of each other), and using
                # it as the gate would be a coincidence test dressed up as a proof.
                window = installed_window(manager)
                if not torch.equal(captured["window"], window):
                    mismatch = int((captured["window"] != window).sum())
                    raise ValueError(
                        f"arm Evict no longer holds arm Selection's memory: "
                        f"{mismatch} of {window.numel()} slot values differ, so the "
                        f"eviction arm is not a mask-only change to arm A")
                readouts[arm]["tensor_equal_to_selection"] = True
                readouts[arm]["compared_slot_values"] = int(window.numel())
            fingerprints[arm] = gr.active_fingerprint(manager)
            fingerprints[arm + "/slots"] = int(manager.used)
            fingerprints[arm + "/active"] = int(model.memory_active_mask.sum())
        return run

    arms = {}
    for arm in ARMS:
        arms[arm] = gr.readout(model, tokenizer, prepared["query_ids"], answers,
                               args.f1_threshold, prepare(arm))
        # The mask has to survive generation, not just the install: `readout` ends
        # with the arm's prepared state in place, so this is the count the reader
        # actually attended to on the last pass. It has to be read per arm -- the
        # loop's last iteration is Replace, which is not the arm being checked.
        fingerprints[arm + "/active_after"] = int(model.memory_active_mask.sum())
    gr.restore(model, manager, state)

    provenance = gr.graft_provenance(telemetry, args.window_blocks)
    if provenance["selected_blocks"] != picked:
        raise ValueError("the compression disagrees with the shared carrier set")
    if provenance["graft_slots"] != GRAFT_SLOTS_TOTAL:
        raise ValueError("the compression did not emit 360 graft slots")
    # The single-factor guarantee, checked rather than assumed. `torch.equal` on the
    # installed window is the authority; `fingerprints` is a mean and is kept for the
    # record only (see the comment in `prepare`).
    if not readouts.get(EVICT, {}).get("tensor_equal_to_selection"):
        raise ValueError("arm Evict was never compared against arm Selection")
    if fingerprints[EVICT + "/active_after"] != CORE_SLOTS_TOTAL:
        raise ValueError(f"arm Evict read {fingerprints[EVICT + '/active_after']} active "
                         f"slots after generation, expected {CORE_SLOTS_TOTAL}: the "
                         f"manager overwrote the eviction mask during the readout")
    if readouts.get(EVICT, {}).get("active_after") != CORE_SLOTS_TOTAL:
        raise ValueError("the eviction mask was not applied on the last prepare pass")
    for arm in (SELECTION, REPLACE):
        if fingerprints[arm + "/slots"] != TARGET_SLOTS:
            raise ValueError(f"arm {arm} does not hold the {TARGET_SLOTS}-slot budget")
        if fingerprints[arm + "/active_after"] != TARGET_SLOTS:
            raise ValueError(f"arm {arm} read {fingerprints[arm + '/active_after']} "
                             f"active slots, expected {TARGET_SLOTS}")
    if int(fingerprints[FULL + "/slots"]) != args.window_blocks * manager.block:
        raise ValueError("the Full arm did not keep the whole window")

    return dict(arms=arms, fingerprints=fingerprints, provenance=provenance,
                layout=dict(permutation_sha256=layout["permutation_sha256"],
                            permutation_is_bijection=layout["permutation_is_bijection"],
                            per_carrier=layout["per_carrier"],
                            graft_slots_by_source_block=layout["graft_slots_by_source_block"],
                            note=layout["note"]),
                eviction=dict(positions=eviction["positions"],
                              core_slots=len(eviction["core_positions"]),
                              per_carrier=eviction["per_carrier"],
                              contiguous=eviction["contiguous"],
                              digest=eviction["digest"],
                              mask=readouts.get(EVICT)),
                carrier_ages=[int(ages[block]) for block in picked])


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def _outcome(selection, evict, replace):
    """What happened to one sample between A, B and C.

    A named 8-way map instead of ad-hoc counters, so the protection group (A
    right), the rescue group (A wrong) and the ungated set are described by the
    same vocabulary and no case is silently dropped. "Unchanged" is split by
    direction: merging "right everywhere" with "wrong everywhere" would make the
    largest bucket of the ungated table mean two opposite things.
    """
    return OUTCOMES[(bool(selection), bool(evict), bool(replace))]


def _states(sample, threshold):
    return (gr._correct(sample, SELECTION, threshold),
            gr._correct(sample, EVICT, threshold),
            gr._correct(sample, REPLACE, threshold))


def _stage_counts(samples, threshold):
    counts = Counter()
    for sample in samples:
        counts[_outcome(*_states(sample, threshold))] += 1
    return dict(sorted(counts.items()))


def _transitions(samples, threshold):
    """Per-sample (A, B, C) correctness, the resulting pattern and the F1s."""
    rows = []
    for sample in samples:
        selection, evict, replace = _states(sample, threshold)
        rows.append(dict(
            id=sample["id"], selection=int(selection), evict=int(evict),
            replace=int(replace), outcome=_outcome(selection, evict, replace),
            qa_f1=dict(selection=round(sample["arms"][SELECTION]["qa_f1"], 4),
                       evict=round(sample["arms"][EVICT]["qa_f1"], 4),
                       replace=round(sample["arms"][REPLACE]["qa_f1"], 4))))
    return rows


def _decisive(samples, threshold):
    """The four patterns the decision table is built on, counted."""
    counts = Counter(_outcome(*_states(sample, threshold)) for sample in samples)
    return dict(
        n=len(samples),
        deletion_breaks=counts["broken_at_evict"],
        insertion_breaks=counts["broken_at_insert"],
        deletion_breaks_insertion_recovers=counts["broken_at_evict_recovered_at_insert"],
        insertion_is_the_only_gain=counts["fixed_at_insert"],
        deletion_fixes=counts["fixed_at_evict"] + counts["fixed_at_evict_broken_at_insert"],
        note=("`deletion_breaks` says the 40 carrier slots that were removed carried the "
              "answer; `insertion_breaks` says the 360 graft slots actively did damage"))


def analyse(samples, threshold, excluded=0, reference_table=None):
    """The two-stage attribution, on the graded samples only."""
    graded = [sample for sample in samples if sample.get("status") == "graded"]
    for sample in graded:
        missing = [arm for arm in ARMS if arm not in sample.get("arms", {})]
        if missing:
            raise ValueError(f"sample {sample.get('id')} has no {missing} arm: this "
                             f"report was not produced by this driver, and every table "
                             f"below assumes all four arms exist")
        if "eviction" not in sample:
            raise ValueError(f"sample {sample.get('id')} carries no eviction record: "
                             f"arm B's mask cannot be audited after the fact")
    located = [sample for sample in graded if sample.get("evidence", {}).get("located")]

    def evidence_in_dropped(sample):
        return bool(set(sample["evidence"]["blocks"])
                    & set(sample["provenance"]["dropped_blocks"]))

    dropped = [sample for sample in located if evidence_in_dropped(sample)]
    kept = [sample for sample in located if not evidence_in_dropped(sample)]

    # The same 13 rows experiment 1 found, by the same rule, so the new arm is
    # compared with the old report on identical questions.
    protect = [sample for sample in located
               if gr._correct(sample, FULL, threshold)
               and gr._correct(sample, SELECTION, threshold)]
    # The 6 rows where dropping the evidence block cost the answer.
    rescue = [sample for sample in located
              if gr._correct(sample, FULL, threshold)
              and not gr._correct(sample, SELECTION, threshold)
              and evidence_in_dropped(sample)]

    broken_at_evict = [sample for sample in protect
                       if not gr._correct(sample, EVICT, threshold)]
    broken_at_insert = [sample for sample in protect
                        if gr._correct(sample, EVICT, threshold)
                        and not gr._correct(sample, REPLACE, threshold)]
    harmed_detail = [dict(id=sample["id"],
                          selection=int(gr._correct(sample, SELECTION, threshold)),
                          evict=int(gr._correct(sample, EVICT, threshold)),
                          replace=int(gr._correct(sample, REPLACE, threshold)),
                          outcome=_outcome(*_states(sample, threshold)),
                          qa_f1=dict(selection=round(sample["arms"][SELECTION]["qa_f1"], 4),
                                     evict=round(sample["arms"][EVICT]["qa_f1"], 4),
                                     replace=round(sample["arms"][REPLACE]["qa_f1"], 4)),
                          p_gold=dict(selection=sample["arms"][SELECTION].get("p_gold"),
                                      evict=sample["arms"][EVICT].get("p_gold"),
                                      replace=sample["arms"][REPLACE].get("p_gold")))
                     for sample in protect
                     if not gr._correct(sample, REPLACE, threshold)]

    protection = dict(
        definition=("Full correct and Selection correct, restricted to samples whose "
                    "evidence block is inside the written window: the compression lost "
                    "nothing there, so any loss at B or C is damage"),
        n=len(protect),
        ids=[sample["id"] for sample in protect],
        harmed_by_evict=len(broken_at_evict),
        harmed_by_insert=len(broken_at_insert),
        harmed_by_replace=len([sample for sample in protect
                               if not gr._correct(sample, REPLACE, threshold)]),
        harmed_rate=(len([sample for sample in protect
                          if not gr._correct(sample, REPLACE, threshold)]) / len(protect)
                     if protect else None),
        stages=_stage_counts(protect, threshold),
        decisive=_decisive(protect, threshold),
        transitions=_transitions(protect, threshold),
        # The rows that were damaged at SOME point, with all three answers side by
        # side: this is the table that says which stage did it.
        harmed_detail=harmed_detail,
        verdict=_verdict(broken_at_evict, broken_at_insert, protect))
    if reference_table:
        reference_protect = _reference_protection(reference_table, threshold)
        if reference_protect is not None:
            protection["reference_ids"] = sorted(reference_protect)
            protection["matches_reference"] = (
                sorted(protection["ids"]) == sorted(reference_protect))
            protection["ids_missing_from_this_run"] = sorted(
                reference_protect - set(protection["ids"]))
            protection["ids_not_in_reference"] = sorted(
                set(protection["ids"]) - reference_protect)

    return dict(
        counts=dict(
            attempted=len(graded) + int(excluded), graded=len(graded),
            excluded=int(excluded), evidence_located=len(located),
            evidence_in_dropped=len(dropped),
            full_correct=len([sample for sample in graded
                              if gr._correct(sample, FULL, threshold)]),
            selection_correct=len([sample for sample in graded
                                   if gr._correct(sample, SELECTION, threshold)]),
            evict_correct=len([sample for sample in graded
                               if gr._correct(sample, EVICT, threshold)]),
            replace_correct=len([sample for sample in graded
                                 if gr._correct(sample, REPLACE, threshold)])),
        arms={arm: gr._arm_summary(graded, arm, threshold) for arm in ARMS},
        arms_evidence_located={arm: gr._arm_summary(located, arm, threshold)
                               for arm in ARMS},
        arms_by_evidence=dict(
            dropped={arm: gr._arm_summary(dropped, arm, threshold) for arm in ARMS},
            kept={arm: gr._arm_summary(kept, arm, threshold) for arm in ARMS}),
        # The three questions, in order: does deleting cost, does inserting cost or
        # pay, and does the pair reproduce experiment 2's replace number.
        pairs={f"{treatment}_minus_{baseline}": er.pair_summary(
                   graded, treatment, baseline, threshold)
               for treatment, baseline in PAIRS},
        pairs_evidence_located={f"{treatment}_minus_{baseline}": er.pair_summary(
                   located, treatment, baseline, threshold)
               for treatment, baseline in PAIRS},
        pairs_kept={f"{treatment}_minus_{baseline}": er.pair_summary(
                        kept, treatment, baseline, threshold)
                    for treatment, baseline in PAIRS},
        protection_group=protection,
        rescue_group=dict(
            definition=("Full correct, Selection wrong, and the dropped blocks hold "
                        "the supporting sentences"),
            n=len(rescue),
            fixed_at_evict=len([sample for sample in rescue
                                if gr._correct(sample, EVICT, threshold)]),
            fixed_at_replace=len([sample for sample in rescue
                                  if gr._correct(sample, REPLACE, threshold)]),
            stages=_stage_counts(rescue, threshold),
            transitions=_transitions(rescue, threshold)),
        ungated=dict(definition="every graded sample, no evidence filter",
                     stages=_stage_counts(graded, threshold),
                     transitions=_transitions(graded, threshold)),
        vocabulary=OUTCOME_VOCABULARY,
        capacity_caveat=CAPACITY_CAVEAT,
        screening_caveat=SCREENING_CAVEAT,
        f1_threshold=threshold)


def _verdict(broken_at_evict, broken_at_insert, protect):
    """One sentence saying where the protection-group damage comes from."""
    if not protect:
        return "no protection-group sample: nothing to attribute"
    if not broken_at_evict and not broken_at_insert:
        return "no protection-group sample was damaged by either stage"
    if broken_at_evict and not broken_at_insert:
        return ("every damaged sample was already wrong once the 40 carrier slots were "
                "removed: the deletion is the damage channel, the inserted slots are "
                "innocent")
    if broken_at_insert and not broken_at_evict:
        return ("every damaged sample was still right after the deletion and only went "
                "wrong once the 360 graft slots were inserted: the insertion is the "
                "damage channel")
    return (f"both stages damage different samples ({len(broken_at_evict)} at the "
            f"deletion, {len(broken_at_insert)} at the insertion): the two mechanisms "
            f"need to be weighted separately")


def _reference_protection(table, threshold):
    """The protection-group ids of a finished experiment-2 report, if it has one."""
    if not table:
        return None
    rows = [sample for sample in table.values()
            if sample.get("status") == "graded" and sample.get("evidence", {}).get("located")
            and FULL in sample.get("arms", {}) and SELECTION in sample.get("arms", {})
            and gr._correct(sample, FULL, threshold)
            and gr._correct(sample, SELECTION, threshold)]
    return {str(sample["id"]) for sample in rows}


# ---------------------------------------------------------------------------
# cross-check against experiment 2
# ---------------------------------------------------------------------------
def load_reference(path):
    """Open the experiment-2 report before the GPU work, not after it.

    A typo or a moved file must fail in the first second, not at the finish line of
    a 45-minute run.
    """
    reference = json.loads(Path(path).read_text())
    table = {str(sample["id"]): sample for sample in reference.get("samples") or []}
    if not table:
        raise ValueError(f"--verify-against {path} holds no samples")
    missing = [arm for arm in (FULL, SELECTION, REPLACE)
               if not any(arm in sample.get("arms", {}) for sample in table.values())]
    if missing:
        raise ValueError(f"--verify-against {path} has no {missing} arms; it is not an "
                         f"experiment-2 report")
    return reference, table


def verify_against(path, reference, table, samples):
    """Compare this run with a finished experiment-2 report, row by row.

    Arms Full, Selection and Replace are supposed to be the same measurement as
    experiment 2: same code path, same carriers, same snapshots. A row-for-row
    prediction match is what makes the NEW arm comparable to numbers measured in
    a different process, and a mismatch is a bug in this driver, not a result.

    The row set is checked in BOTH directions: a run over a subset of the rows
    would otherwise report `different=0` and look like a perfect match.
    """
    this_run = {str(sample["id"]) for sample in samples}
    report = dict(
        reference=path, reference_schema=reference.get("schema"),
        reference_samples=len(table), compared=len(this_run),
        row_count_matches_reference=(len(this_run) == len(table)),
        reference_not_in_this_run=sorted(set(table) - this_run),
        this_run_not_in_reference=sorted(this_run - set(table)))
    passed = report["row_count_matches_reference"]
    for arm in (FULL, SELECTION, REPLACE):
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
# CPU-only self-test: the eviction set and the mask arithmetic
# ---------------------------------------------------------------------------
class _StubMask(torch.nn.Module):
    """Just enough of the reader to drive `install_evict`'s mask arithmetic."""

    def __init__(self, total):
        super().__init__()
        self.register_buffer("memory_active_mask", torch.ones(total, dtype=torch.bool),
                             persistent=False)


class _StubManager:
    """A manager whose pool is `total` wide with `used` slots installed."""

    def __init__(self, total, used, block):
        self.total, self.used, self.block = total, used, block
        self._active_mask_count = 0
        self.model = _StubMask(total)


def _self_test():
    """Check the eviction set, the mask arithmetic and the outcome map, on CPU.

    The eviction set is the whole experiment: if it were wrong, arm B would delete
    slots that arm C keeps (or keep slots arm C replaces), and A -> B / B -> C
    would stop decomposing the same operation. This builds a synthetic 19 -> 9
    layout with uniquely identifiable slot values and asserts the set, the
    partition, the per-carrier count and the mask arithmetic.
    """
    generator = torch.Generator().manual_seed(0)
    carriers = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    dropped = [index for index in range(WINDOW_BLOCKS) if index not in carriers]
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
    graft_output = torch.arange(TARGET_SLOTS, dtype=torch.float32).view(1, TARGET_SLOTS, 1)
    replaced, layout = er.build_inplace_replace(graft_output, telemetry, carriers,
                                                WINDOW_BLOCKS, BLOCK)

    eviction = evicted_window_positions(layout)
    if len(eviction["positions"]) != GRAFT_SLOTS_TOTAL:
        raise ValueError("the eviction set is not 360 slots")
    if len(eviction["core_positions"]) != CORE_SLOTS_TOTAL:
        raise ValueError("the kept set is not 1,944 slots")
    if eviction["contiguous"]:
        raise ValueError("the eviction set must be scattered, not a contiguous tail")
    if sorted(eviction["per_carrier"].values()) != [GRAFT_SLOTS] * TARGET_BLOCKS:
        raise ValueError("the eviction set is not 40 slots per carrier")
    # The evicted position of each carrier must be one the carrier actually vacated,
    # and the value arm C puts there must be a graft vector.
    flat = [int(value) for value in replaced[0, :, 0].tolist()]
    core_values = {index for index, item in enumerate(slots) if item[1] == "core"}
    for position in eviction["positions"]:
        if flat[position] in core_values:
            raise ValueError(f"arm C keeps a carrier vector at {position}, but arm B "
                             f"would evict it")
    for position in eviction["core_positions"]:
        if flat[position] not in core_values:
            raise ValueError(f"arm C puts a graft vector at {position}, but arm B "
                             f"would keep it")
        carrier = position // BLOCK
        if position % BLOCK not in set(core_local[carriers[carrier]]):
            raise ValueError(f"position {position} is not a surviving carrier slot")
    if sorted(eviction["positions"] + eviction["core_positions"]) != list(range(TARGET_SLOTS)):
        raise ValueError("evict + keep is not a partition of the window")

    # Mask arithmetic on a stub pool: 2304 active becomes 1944, and the driver
    # refuses a pool whose active count it does not recognise.
    total = 50 * BLOCK
    manager = _StubManager(total, 9 * BLOCK, BLOCK)
    # Model what `_sync_active_mask` publishes for a 9-block install: the distinct
    # suffix is active, the rest of the pool is not.
    manager.model.memory_active_mask.zero_()
    manager.model.memory_active_mask[total - 9 * BLOCK:] = True
    # `install_evict` needs a real manager to install a window; its mask arithmetic is
    # exercised here directly, with the same expressions and the same assertions.
    mask = manager.model.memory_active_mask
    offset = manager.total - manager.used
    if int(mask.sum()) != manager.used:
        raise ValueError("the stub pool does not start fully active")
    index = torch.as_tensor([offset + position for position in eviction["positions"]],
                            dtype=torch.long)
    mask[index] = False
    if int(mask.sum()) != CORE_SLOTS_TOTAL:
        raise ValueError(f"the mask arithmetic leaves {int(mask.sum())} slots active, "
                         f"expected {CORE_SLOTS_TOTAL}")
    if not bool(mask[total - 9 * BLOCK:total - 9 * BLOCK + BLOCK].any()):
        raise ValueError("the eviction emptied a whole block: the mask is not per-slot")

    # A pool carrying an unknown extra block must be refused, not silently
    # measured: the expected active count is a constant, not a guess.
    other = _StubManager(total, 10 * BLOCK, BLOCK)
    mask = other.model.memory_active_mask
    mask.zero_()
    mask[total - 10 * BLOCK:] = True
    index = torch.as_tensor([(other.total - other.used) + position
                             for position in eviction["positions"]], dtype=torch.long)
    mask[index] = False
    if int(mask.sum()) == CORE_SLOTS_TOTAL:
        raise ValueError("a pool with an extra block must not land on 1,944 active slots")

    # The outcome map must name all eight patterns distinctly and round-trip.
    if len(set(OUTCOMES.values())) != 8:
        raise ValueError("the outcome vocabulary must name all eight patterns")
    for key, name in OUTCOMES.items():
        if _outcome(*key) != name:
            raise ValueError(f"_outcome{key} = {_outcome(*key)}, expected {name}")
        if key not in [(True, True, True), (True, False, False), (True, True, False),
                       (True, False, True), (False, True, True), (False, True, False),
                       (False, False, True), (False, False, False)]:
            raise ValueError("unexpected outcome key")
    if sorted(OUTCOME_VOCABULARY) != sorted(OUTCOMES.values()):
        raise ValueError("every outcome needs a documented meaning")
    # The documented pattern must be the pattern the map actually produces: a
    # vocabulary that drifts from the code would mislabel the decision table.
    for key, name in OUTCOMES.items():
        documented = OUTCOME_VOCABULARY[name]["pattern"]
        derived = "A=%d B=%d C=%d" % tuple(int(flag) for flag in key)
        if documented != derived:
            raise ValueError(f"{name}: documented pattern {documented!r} but the map "
                             f"produces {derived!r}")

    # Experiment 2 quoted p = 0.424 for Replace - Selection (9 up, 5 down) and this
    # is the value the report carried: a regression test on the statistic.
    if abs(er.mcnemar_exact(9, 5) - 0.4240) > 0.0005:
        raise ValueError(f"McNemar(9, 5) = {er.mcnemar_exact(9, 5)}, expected 0.4240")
    if abs(er.mcnemar_exact(11, 5) - 0.2101) > 0.0005:
        raise ValueError(f"McNemar(11, 5) = {er.mcnemar_exact(11, 5)}, expected 0.2101")
    if er.mcnemar_exact(0, 0) != 1.0:
        raise ValueError("McNemar(0, 0) must be 1.0")

    verdict = _verdict([1, 2, 3], [], [0] * 13)
    if "deletion is the damage channel" not in verdict:
        raise ValueError(f"the verdict for evict-only damage reads wrongly: {verdict}")
    verdict = _verdict([], [1, 2], [0] * 13)
    if "insertion is the damage channel" not in verdict:
        raise ValueError(f"the verdict for insert-only damage reads wrongly: {verdict}")
    verdict = _verdict([1], [2], [0] * 13)
    if "both stages" not in verdict:
        raise ValueError(f"the verdict for mixed damage reads wrongly: {verdict}")

    return dict(self_test="ok", carriers=len(carriers), dropped=len(dropped),
                evicted=len(eviction["positions"]),
                kept=len(eviction["core_positions"]),
                per_carrier=GRAFT_SLOTS, contiguous=eviction["contiguous"],
                eviction_sha256=eviction["digest"],
                permutation_sha256=layout["permutation_sha256"],
                mcnemar_9_5=round(er.mcnemar_exact(9, 5), 6))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def build_protocol(args, task):
    return dict(
        experiment="3 evict-only: deletion loss vs insertion effect",
        task=task, data_dir=args.data_dir,
        arms=list(ARMS),
        arm_definitions=dict(
            Full="the 19 written blocks, nothing compressed",
            Selection=f"the 9 carrier blocks, untouched ({TARGET_SLOTS:,} slots)",
            Evict=(f"the same {TARGET_SLOTS:,} vectors as Selection, with the "
                   f"{GRAFT_SLOTS_TOTAL} slots that Replace overwrites removed from the "
                   f"reader's attention ({CORE_SLOTS_TOTAL:,} readable slots); nothing "
                   f"inserted, nothing zero-filled, nothing re-packed"),
            Replace=(f"each carrier keeps its own {CORE_SLOTS} core slots at their "
                     f"original in-block positions; the {GRAFT_SLOTS} vacated positions "
                     f"hold the V10 graft slots ({TARGET_SLOTS:,} slots)")),
        window_blocks=args.window_blocks, target_blocks=args.target_blocks,
        block_tokens=gr.BLOCK_TOKENS, slots_per_block=BLOCK,
        write_tokens=args.window_blocks * gr.BLOCK_TOKENS,
        full_slots=args.window_blocks * BLOCK,
        target_slots=args.target_blocks * BLOCK,
        core_slots_total=args.target_blocks * CORE_SLOTS,
        graft_slots_total=args.target_blocks * GRAFT_SLOTS,
        evict_slots=TARGET_BLOCKS * GRAFT_SLOTS,
        evict_construction=("arm Evict installs exactly what arm Selection installs and "
                            "then clears `memory_active_mask` at the 360 positions where "
                            "arm Replace puts a graft vector; the identity of the two "
                            "pools is measured per sample with `torch.equal` on the "
                            "installed window (`eviction.mask."
                            "tensor_equal_to_selection`), not with the fingerprint"),
        mask_mechanism=MASK_NOTE,
        capacity_caveat=CAPACITY_CAVEAT,
        compressions_per_sample=1,
        write_stream="document prefix only (the leading template instruction is dropped)",
        query=gr.QUERY_SUFFIX,
        operator_mode="native",
        operator_note=("carrier and graft slots are copied verbatim; no same-block "
                       "condensation, no orthogonal context delta, no core reinforcement"),
        carrier_selection=("select_window_carrier_blocks runs once on the "
                           "pre-compression window; Selection, Evict and Replace share "
                           "it, so the eviction set is a function of Replace's own "
                           "permutation"),
        attribution_note=("A -> B isolates the deletion (40 carrier slots removed, "
                          "nothing inserted); B -> C isolates the insertion (same "
                          "carriers, same positions, the 360 graft vectors added and "
                          "unmasked); A -> C re-measures experiment 2's replace number "
                          "in the same process"),
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
        note=("arm Evict adds no readout machinery: it is arm Selection plus an "
              "attention mask, so the only new cost is the 158 extra readouts; the "
              "dry run does not need a GPU"))
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
        raise ValueError("the Replace arm needs the trained operator: pass --checkpoint")

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
    if not getattr(manager, "active_memory_mask", False):
        raise ValueError("this experiment masks slots; it needs the managed reader with "
                         "active_memory_mask enabled, or arm Evict would be the identity")
    merge.operator_mode = "native"       # experiment 3: native slots only

    operator_file = "methods/prlmc/manifold_graft.py"
    operator_now = sha(ROOT / operator_file)
    operator_then = ((saved or {}).get("source_manifest") or {}).get(operator_file)
    if operator_then != operator_now:
        print(f"WARNING: {operator_file} differs from the checkpoint's manifest "
              f"(ckpt {str(operator_then)[:8]} != code {operator_now[:8]}); the change "
              f"must be non-numeric, and --verify-against is what checks that",
              file=sys.stderr)

    report = dict(
        schema="evict_only_results/1", status="running",
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
            replace_entrypoint_sha256=sha(Path(er.__file__)),
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
                              evict=result["arms"][EVICT]["correct"],
                              replace=result["arms"][REPLACE]["correct"],
                              evict_tensor_equals_selection=bool(
                                  result["eviction"]["mask"]["tensor_equal_to_selection"]),
                              seconds=round(record["seconds"], 1))), flush=True)

    report["summary"] = analyse(report["samples"], args.f1_threshold,
                                excluded=len(report["excluded"]),
                                reference_table=table)
    if args.verify_against:
        report["verification_against_experiment_2"] = verify_against(
            args.verify_against, reference, table, report["samples"])
    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["status"] = "complete"
    atomic_json(output, report)
    print(json.dumps(report["summary"], indent=1, ensure_ascii=False))
    verification = report.get("verification_against_experiment_2")
    if verification and not verification["passed"]:
        print("WARNING: this run does NOT reproduce experiment 2 row for row; "
              f"{json.dumps({arm: verification[arm]['different'] for arm in (FULL, SELECTION, REPLACE)})} "
              "differing predictions. Then arm Replace is a different implementation and "
              "B -> C is not comparable with the old report.", file=sys.stderr)


def analyze_existing(path, threshold=None):
    report = json.loads(Path(path).read_text())
    if threshold is None:
        threshold = report.get("protocol", {}).get("f1_threshold", 0.5)
    summary = analyse(report.get("samples") or [], threshold,
                      excluded=len(report.get("excluded") or []))
    # The stored per-arm `correct` flags are derived from the same QA F1, so a moved
    # threshold has to move them too: otherwise the summary and the records it was
    # computed from disagree inside one file.
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
                   help="a finished experiment-2 report; the run then records a row-by-row "
                        "prediction comparison for Full/Selection/Replace")
    p.add_argument("--strict-code-hash", dest="strict_code_hash", default="true",
                   choices=("true", "false"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="resolve the sample set and evidence blocks, print the plan, "
                        "load no model and write no file")
    p.add_argument("--analyze", default=None,
                   help="re-group an existing report without inference")
    p.add_argument("--self-test", dest="self_test", action="store_true",
                   help="run the CPU-only eviction-set, mask-arithmetic and outcome-map "
                        "checks and exit; no data, model or GPU needed")
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
