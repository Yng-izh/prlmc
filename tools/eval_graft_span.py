#!/usr/bin/env python3
"""Experiment 1A -- Span-4 source selection: does grouping the graft SOURCE help?

What changes and what does not
-------------------------------
V10 picks the 360 graft slots by ranking every dropped-block slot on its own
leverage and taking the global top 360. Those 360 slots land wherever the ranking
puts them, so the graft side of the memory is a scatter of isolated slots. This
experiment changes ONLY the source side:

    Graft   V10 as trained: top-360 individual slots from the 10 dropped blocks
    Span    the same 10 dropped blocks, scored as NON-OVERLAPPING 4-slot spans
            (mean leverage of the 4 slots), top-90 spans = the same 360 slots,
            arriving as 90 runs of 4 consecutive in-block positions

Everything else is frozen, exactly as specified: same 158 HotpotQA rows, frozen
MemoryLLM-8B reader, the same 9 carriers, the original single-point top-216 core
selection per carrier, 360 graft slots, 2,304 output slots, 100% native copies
(no condensation, no orthogonal delta, no core reinforcement), the existing
global chronological output order, and no training.

Why the comparison is paired
----------------------------
Both compressions are computed from ONE snapshot of the pre-compression window and
both are installed through the same `install`, so `Span - Graft` differs only in
which source slots were picked. The driver checks that structurally on every
sample: the core slot positions must be identical between the two telemetries
(only the source side moved), and the two graft position sets must actually
differ (otherwise the arm is the point arm and the experiment measures nothing).

The three numbers this experiment exists for
--------------------------------------------
1. the 16 questions the compression changes at all (`Selection` and `Graft`
   disagree): does Span rescue more of them, and does it harm fewer?
2. the 40 questions whose supporting evidence sits in a DROPPED block: does the
   span selection read more of that evidence?
3. overall rescued-minus-harmed: is the net bigger than the point rule's?

Usage
-----
    python tools/eval_graft_span.py --dry-run --data-dir ... --task hotpotqa \\
        --supporting-facts-glob '...' --model-path /dev/shm/s1d_reader_fp16

    python tools/eval_graft_span.py --self-test          # CPU only

    python tools/eval_graft_span.py --data-dir ... --task hotpotqa \\
        --supporting-facts-glob '...' --model-path /dev/shm/s1d_reader_fp16 \\
        --checkpoint runs/natural_merge_v9/full_v10.pt \\
        --verify-against reports/diagnostics/replace_reorder_hotpotqa.json \\
        --out reports/diagnostics/graft_span_hotpotqa.json

    python tools/eval_graft_span.py --analyze reports/diagnostics/graft_span_hotpotqa.json
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

BLOCK = gr.SLOTS_PER_BLOCK                                # 256 slots per block
WINDOW_BLOCKS = gr.WINDOW_BLOCKS                          # 19
TARGET_BLOCKS = gr.TARGET_BLOCKS                          # 9
CORE_SLOTS = gr.CORE_SLOTS                                # 216
GRAFT_SLOTS = gr.GRAFT_SLOTS                              # 40
TARGET_SLOTS = gr.TARGET_SLOTS                            # 2,304
CORE_SLOTS_TOTAL = gr.CORE_SLOTS_TOTAL                    # 1,944
GRAFT_SLOTS_TOTAL = gr.GRAFT_SLOTS_TOTAL                  # 360
SPAN = 4                                                  # slots per span
SPANS_TOTAL = GRAFT_SLOTS_TOTAL // SPAN                   # 90

FULL, SELECTION, GRAFT, SPAN_ARM = "Full", "Selection", "Graft", "Span"
ARMS = (FULL, SELECTION, GRAFT, SPAN_ARM)
# The primary comparison first: Span against the trained point rule. The other two
# are context, and `Graft - Selection` is the reproduction check.
PAIRS = ((SPAN_ARM, GRAFT), (SPAN_ARM, SELECTION), (GRAFT, SELECTION))

SCREENING_CAVEAT = gr.SCREENING_CAVEAT

# (Selection, Graft, Span) correctness, all eight patterns named.
OUTCOMES = {
    (True, True, True): "unchanged_correct",
    (True, True, False): "span_loses",
    (True, False, True): "span_recovers",
    (True, False, False): "both_lose",
    (False, True, True): "point_only_fix",
    (False, True, False): "point_fix_span_lose",
    (False, False, True): "span_only_fix",
    (False, False, False): "unchanged_wrong",
}
OUTCOME_VOCABULARY = {
    "unchanged_correct": dict(pattern="A=1 P=1 S=1", meaning="right everywhere"),
    "span_loses": dict(pattern="A=1 P=1 S=0",
                       meaning="the point rule was right, the span rule loses it"),
    "span_recovers": dict(pattern="A=1 P=0 S=1",
                          meaning="the point rule broke it, the span rule does not"),
    "both_lose": dict(pattern="A=1 P=0 S=0",
                      meaning="both rules lose the same question"),
    "point_only_fix": dict(pattern="A=0 P=1 S=1",
                           meaning="the point rule fixes it and the span rule keeps it"),
    "point_fix_span_lose": dict(pattern="A=0 P=1 S=0",
                                meaning="the point rule fixes it, the span rule does not"),
    "span_only_fix": dict(pattern="A=0 P=0 S=1",
                          meaning="only the span rule fixes it"),
    "unchanged_wrong": dict(pattern="A=0 P=0 S=0",
                            meaning="wrong under both rules"),
}


def _check_geometry():
    if (WINDOW_BLOCKS, TARGET_BLOCKS) != (19, 9):
        raise ValueError(f"this experiment is pinned to 19->9, got "
                         f"{WINDOW_BLOCKS}->{TARGET_BLOCKS}")
    if CORE_SLOTS + GRAFT_SLOTS != BLOCK:
        raise ValueError("core+graft slots must tile a native block")
    if BLOCK % SPAN or GRAFT_SLOTS_TOTAL % SPAN:
        raise ValueError(f"a {SPAN}-slot span must tile both a block and the budget")
    if GRAFT_SLOTS_TOTAL // SPAN != SPANS_TOTAL:
        raise ValueError("unexpected span budget")
    if not {FULL, SELECTION, GRAFT}.issubset(set(er.ARMS)):
        raise ValueError("eval_replace_reorder no longer defines Full/Selection/Graft")


_check_geometry()


# ---------------------------------------------------------------------------
# the two source rules, checked from the telemetry
# ---------------------------------------------------------------------------
def graft_positions(telemetry, total=GRAFT_SLOTS_TOTAL):
    """Window positions of the graft slots, in the operator's output order."""
    offsets = [int(offset) for offset in telemetry["bridge_slot_offsets"]]
    positions = [int(position) for position in telemetry["slot_source_positions"]]
    if len(offsets) != total:
        raise ValueError(f"the operator emitted {len(offsets)} graft slots, expected "
                         f"{total}")
    return [positions[offset] for offset in offsets]


def core_positions(telemetry):
    """Window positions of the 1,944 carrier slots (everything not grafted)."""
    graft = set(int(offset) for offset in telemetry["bridge_slot_offsets"])
    positions = [int(position) for position in telemetry["slot_source_positions"]]
    return [position for index, position in enumerate(positions) if index not in graft]


def span_structure(positions, block=BLOCK, span=SPAN):
    """How many complete runs of `span` consecutive in-block positions are here.

    A span rule that picks 90 spans must produce 90 complete runs and nothing
    else. A partial run would mean the operator selected slots it did not score as
    a group, which is the one thing that would make the arm a different experiment.
    """
    runs = {}
    for position in positions:
        key = (position // block, (position % block) // span)
        runs.setdefault(key, []).append(position % block)
    for (block_id, group), members in runs.items():
        wanted = [group * span + index for index in range(span)]
        if sorted(members) != wanted:
            raise ValueError(f"block {block_id} span {group} is incomplete: {sorted(members)}")
    return len(runs)


def check_source_rules(point, span, dropped, picked):
    """The two compressions must differ in the source side and nowhere else."""
    if point["selected_blocks"] != picked or span["selected_blocks"] != picked:
        raise ValueError("a compression disagrees with the shared carrier set")
    for name, telemetry in (("point", point), ("span", span)):
        if telemetry["grafted_slots"] != GRAFT_SLOTS_TOTAL:
            raise ValueError(f"{name}: {telemetry['grafted_slots']} graft slots, expected "
                             f"{GRAFT_SLOTS_TOTAL}")
    if point["graft_source"] != "point" or point["graft_span"] is not None:
        raise ValueError(f"the point arm is not the trained rule: {point['graft_source']}")
    if span["graft_source"] != "span4" or span["graft_span"] != SPAN:
        raise ValueError(f"the span arm is not a {SPAN}-slot span rule: "
                         f"{span['graft_source']}/{span['graft_span']}")
    if span["graft_spans"] != SPANS_TOTAL:
        raise ValueError(f"the span arm selected {span['graft_spans']} spans, expected "
                         f"{SPANS_TOTAL}")
    if sorted(core_positions(point)) != sorted(core_positions(span)):
        raise ValueError("the two rules disagree about the CORE slots: this is supposed "
                         "to be a source-side experiment only")
    point_graft = set(graft_positions(point))
    span_graft = set(graft_positions(span))
    if len(point_graft) != GRAFT_SLOTS_TOTAL or len(span_graft) != GRAFT_SLOTS_TOTAL:
        raise ValueError("a rule selected the same slot twice")
    if point_graft == span_graft:
        raise ValueError("the span rule picked exactly the point rule's slots: the "
                         "arm would measure nothing")
    runs = span_structure(sorted(span_graft))
    if runs != SPANS_TOTAL:
        raise ValueError(f"the span rule produced {runs} complete runs, expected "
                         f"{SPANS_TOTAL}")
    dropped_set = set(int(block) for block in dropped)
    for position in list(point_graft) + list(span_graft):
        if position // BLOCK not in dropped_set:
            raise ValueError(f"a graft slot came from block {position // BLOCK}, which "
                             f"was not dropped")
    return dict(
        point_graft_positions=sorted(point_graft),
        span_graft_positions=sorted(span_graft),
        shared_graft_slots=len(point_graft & span_graft),
        span_runs=runs,
        point_graft_digest=digest(sorted(point_graft)),
        span_graft_digest=digest(sorted(span_graft)))


# ---------------------------------------------------------------------------
# one sample
# ---------------------------------------------------------------------------
def run_sample(model, tokenizer, manager, merge, prepared, args):
    """Write 19 blocks once, compress twice (point, span), answer four arms."""
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
        # Both compressions read the SAME pre-compression window; the operator does
        # not mutate the manager, so the second call sees exactly what the first did.
        merge.graft_source = "point"
        point_output, point_telemetry = gr.compress_graft(
            manager, merge, source, picked, args.window_blocks, args.target_blocks,
            frozen, pinned)
        merge.graft_source = "span4"
        merge.graft_span = SPAN
        span_output, span_telemetry = gr.compress_graft(
            manager, merge, source, picked, args.window_blocks, args.target_blocks,
            frozen, pinned)
        merge.graft_source = "point"      # leave the operator as the checkpoint had it
    gr.restore(model, manager, state)

    dropped = [block for block in range(args.window_blocks) if block not in set(picked)]
    source_check = check_source_rules(point_telemetry, span_telemetry, dropped, picked)

    def prepare(arm):
        def run():
            gr.restore(model, manager, state)
            if arm == SELECTION:
                gr.install_selection(manager, source, picked, args.window_blocks)
            elif arm == GRAFT:
                er.install_output(manager, point_output, picked, args.window_blocks)
            elif arm == SPAN_ARM:
                er.install_output(manager, span_output, picked, args.window_blocks)
            fingerprints[arm] = gr.active_fingerprint(manager)
            fingerprints[arm + "/slots"] = int(manager.used)
        return run

    arms = {arm: gr.readout(model, tokenizer, prepared["query_ids"], answers,
                            args.f1_threshold, prepare(arm)) for arm in ARMS}
    gr.restore(model, manager, state)

    if int(fingerprints[FULL + "/slots"]) != args.window_blocks * manager.block:
        raise ValueError("the Full arm did not keep the whole window")
    for arm in (SELECTION, GRAFT, SPAN_ARM):
        if int(fingerprints[arm + "/slots"]) != TARGET_SLOTS:
            raise ValueError(f"arm {arm} does not hold the {TARGET_SLOTS}-slot budget")

    return dict(arms=arms, fingerprints=fingerprints,
                provenance=gr.graft_provenance(point_telemetry, args.window_blocks),
                source_check=source_check,
                span_telemetry=dict(graft_source=span_telemetry["graft_source"],
                                    graft_span=span_telemetry["graft_span"],
                                    graft_spans=span_telemetry["graft_spans"]),
                carrier_ages=[int(ages[block]) for block in picked])


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def _outcome(selection, graft, span):
    return OUTCOMES[(bool(selection), bool(graft), bool(span))]


def _states(sample, threshold):
    return (gr._correct(sample, SELECTION, threshold),
            gr._correct(sample, GRAFT, threshold),
            gr._correct(sample, SPAN_ARM, threshold))


def _stage_counts(samples, threshold):
    counts = Counter()
    for sample in samples:
        counts[_outcome(*_states(sample, threshold))] += 1
    return dict(sorted(counts.items()))


def _transitions(samples, threshold):
    rows = []
    for sample in samples:
        selection, graft, span = _states(sample, threshold)
        rows.append(dict(
            id=sample["id"], selection=int(selection), graft=int(graft),
            span=int(span), outcome=_outcome(selection, graft, span),
            qa_f1=dict(selection=round(sample["arms"][SELECTION]["qa_f1"], 4),
                       graft=round(sample["arms"][GRAFT]["qa_f1"], 4),
                       span=round(sample["arms"][SPAN_ARM]["qa_f1"], 4))))
    return rows


def _evidence_in_dropped(sample):
    return bool(set(sample["evidence"]["blocks"])
                & set(sample["provenance"]["dropped_blocks"]))


def analyse(samples, threshold, excluded=0):
    graded = [sample for sample in samples if sample.get("status") == "graded"]
    for sample in graded:
        missing = [arm for arm in ARMS if arm not in sample.get("arms", {})]
        if missing:
            raise ValueError(f"sample {sample.get('id')} has no {missing} arm: this "
                             f"report was not produced by this driver")
        if "source_check" not in sample:
            raise ValueError(f"sample {sample.get('id')} carries no source check: the "
                             f"two rules cannot be told apart afterwards")
    located = [sample for sample in graded if sample.get("evidence", {}).get("located")]
    dropped = [sample for sample in located if _evidence_in_dropped(sample)]
    kept = [sample for sample in located if not _evidence_in_dropped(sample)]

    # (1) The 16 questions the compression changes at all: Selection and the trained
    # point rule disagree. Experiment 1/2 measured 11 rescued and 5 harmed here.
    changed = [sample for sample in graded
               if gr._correct(sample, SELECTION, threshold)
               != gr._correct(sample, GRAFT, threshold)]
    rescued_point = [sample for sample in changed
                     if not gr._correct(sample, SELECTION, threshold)
                     and gr._correct(sample, GRAFT, threshold)]
    harmed_point = [sample for sample in changed
                    if gr._correct(sample, SELECTION, threshold)
                    and not gr._correct(sample, GRAFT, threshold)]
    rescued_span = [sample for sample in changed
                    if not gr._correct(sample, SELECTION, threshold)
                    and gr._correct(sample, SPAN_ARM, threshold)]
    harmed_span = [sample for sample in changed
                   if gr._correct(sample, SELECTION, threshold)
                   and not gr._correct(sample, SPAN_ARM, threshold)]

    protect = [sample for sample in located
               if gr._correct(sample, FULL, threshold)
               and gr._correct(sample, SELECTION, threshold)]

    overall_span = er.pair_summary(graded, SPAN_ARM, SELECTION, threshold)
    overall_point = er.pair_summary(graded, GRAFT, SELECTION, threshold)

    return dict(
        counts=dict(
            attempted=len(graded) + int(excluded), graded=len(graded),
            excluded=int(excluded), evidence_located=len(located),
            evidence_in_dropped=len(dropped),
            changed_by_compression=len(changed),
            full_correct=len([s for s in graded if gr._correct(s, FULL, threshold)]),
            selection_correct=len([s for s in graded
                                   if gr._correct(s, SELECTION, threshold)]),
            graft_correct=len([s for s in graded if gr._correct(s, GRAFT, threshold)]),
            span_correct=len([s for s in graded
                              if gr._correct(s, SPAN_ARM, threshold)])),
        arms={arm: gr._arm_summary(graded, arm, threshold) for arm in ARMS},
        arms_evidence_located={arm: gr._arm_summary(located, arm, threshold)
                               for arm in ARMS},
        arms_by_evidence=dict(
            dropped={arm: gr._arm_summary(dropped, arm, threshold) for arm in ARMS},
            kept={arm: gr._arm_summary(kept, arm, threshold) for arm in ARMS}),
        pairs={f"{treatment}_minus_{baseline}": er.pair_summary(
                   graded, treatment, baseline, threshold)
               for treatment, baseline in PAIRS},
        pairs_dropped={f"{treatment}_minus_{baseline}": er.pair_summary(
                           dropped, treatment, baseline, threshold)
                       for treatment, baseline in PAIRS},
        pairs_kept={f"{treatment}_minus_{baseline}": er.pair_summary(
                        kept, treatment, baseline, threshold)
                    for treatment, baseline in PAIRS},
        # (1) the 16 questions the compression changes at all
        changed_group=dict(
            definition=("Selection and the trained point rule disagree on correctness, "
                        "i.e. every question the compression changes in either "
                        "direction (11 rescued + 5 harmed in experiments 1 and 2)"),
            n=len(changed),
            rescued_by_point=len(rescued_point), harmed_by_point=len(harmed_point),
            rescued_by_span=len(rescued_span), harmed_by_span=len(harmed_span),
            net_point=len(rescued_point) - len(harmed_point),
            net_span=len(rescued_span) - len(harmed_span),
            stages=_stage_counts(changed, threshold),
            transitions=_transitions(changed, threshold)),
        # (2) the 40 questions whose evidence block was dropped
        dropped_evidence_group=dict(
            definition=("the supporting sentence sits in a block the compression "
                        "dropped, so only the graft slots can carry it"),
            n=len(dropped),
            arms={arm: gr._arm_summary(dropped, arm, threshold) for arm in ARMS},
            span_minus_point=er.pair_summary(dropped, SPAN_ARM, GRAFT, threshold),
            span_minus_selection=er.pair_summary(dropped, SPAN_ARM, SELECTION, threshold),
            point_minus_selection=er.pair_summary(dropped, GRAFT, SELECTION, threshold),
            stages=_stage_counts(dropped, threshold),
            transitions=_transitions(dropped, threshold)),
        # (3) is the overall rescued-minus-harmed bigger than the point rule's?
        net=dict(
            point=dict(rescued=overall_point.get("rescued"),
                       harmed=overall_point.get("harmed"),
                       net=overall_point.get("net"),
                       mcnemar_p=overall_point.get("mcnemar_p")),
            span=dict(rescued=overall_span.get("rescued"),
                      harmed=overall_span.get("harmed"),
                      net=overall_span.get("net"),
                      mcnemar_p=overall_span.get("mcnemar_p")),
            net_delta=(None if None in (overall_point.get("net"), overall_span.get("net"))
                       else overall_span["net"] - overall_point["net"]),
            note=("rescued/harmed are counted against Selection, so `net` is exactly the "
                  "quantity experiment 1 reported as +6 for the trained rule")),
        protection_group=dict(
            definition="Full correct and Selection correct (the 13 rows of experiment 1)",
            n=len(protect),
            harmed_by_point=len([s for s in protect
                                 if not gr._correct(s, GRAFT, threshold)]),
            harmed_by_span=len([s for s in protect
                                if not gr._correct(s, SPAN_ARM, threshold)]),
            stages=_stage_counts(protect, threshold),
            transitions=_transitions(protect, threshold)),
        ungated=dict(definition="every graded sample, no evidence filter",
                     stages=_stage_counts(graded, threshold),
                     transitions=_transitions(graded, threshold)),
        vocabulary=OUTCOME_VOCABULARY,
        screening_caveat=SCREENING_CAVEAT,
        f1_threshold=threshold)


# ---------------------------------------------------------------------------
# cross-check against the finished point-rule run
# ---------------------------------------------------------------------------
def load_reference(path):
    reference = json.loads(Path(path).read_text())
    table = {str(sample["id"]): sample for sample in reference.get("samples") or []}
    if not table:
        raise ValueError(f"--verify-against {path} holds no samples")
    missing = [arm for arm in (FULL, SELECTION, GRAFT)
               if not any(arm in sample.get("arms", {}) for sample in table.values())]
    if missing:
        raise ValueError(f"--verify-against {path} has no {missing} arms; it is not an "
                         f"experiment-1/2 report")
    return reference, table


def verify_against(path, reference, table, samples):
    """The point arm must reproduce the finished run row for row.

    The operator gained a source-selection branch, so "the trained rule is
    untouched" is a claim that has to be measured, not asserted: if `Graft` does
    not match the earlier report exactly, the comparison in this experiment is
    between two different implementations.
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
                differing.append(dict(id=sample["id"],
                                      old=old["arms"][arm]["prediction"],
                                      new=sample["arms"][arm]["prediction"]))
        report[arm] = dict(identical=same, different=different, not_in_reference=absent,
                           differing=differing[:20])
        passed = passed and different == 0 and absent == 0 and same == len(table)
    report["passed"] = passed
    return report


# ---------------------------------------------------------------------------
# CPU-only self-test
# ---------------------------------------------------------------------------
def _self_test():
    """Drive the real operator at tiny geometry and check the span rule.

    The source rule is the whole experiment: if a span were only partially
    selected, or the span arm silently degenerated into the point arm, the run
    would report a difference that is not the difference being tested.
    """
    from methods.prlmc.manifold_graft import ManifoldGraftedSlotMerge

    def build():
        return ManifoldGraftedSlotMerge(
            num_layers=2, hidden_size=8, num_groups=1, reference_layer=0,
            block_size=16, core_slots_per_block=12, graft_slots_per_block=4)

    torch.manual_seed(0)
    layers, hidden, block, blocks = 2, 8, 16, 5
    window = torch.randn(layers, blocks * block, hidden)
    carriers = [0, 4]
    dropped = [index for index in range(blocks) if index not in carriers]
    picks = {}
    for mode in ("point", "span4"):
        merge = build()
        merge.operator_mode = "native"
        merge.graft_source = mode
        output, _, telemetry = merge(window, carriers)
        if tuple(output.shape) != (layers, len(carriers) * block, hidden):
            raise ValueError(f"{mode}: unexpected output shape {tuple(output.shape)}")
        positions = graft_positions(telemetry, total=len(carriers) * 4)
        if len(positions) != len(carriers) * 4:
            raise ValueError(f"{mode}: {len(positions)} graft slots")
        if len(set(positions)) != len(positions):
            raise ValueError(f"{mode}: a slot was selected twice")
        if any(position // block not in dropped for position in positions):
            raise ValueError(f"{mode}: a graft slot came from a carrier block")
        picks[mode] = set(positions)
        if mode == "span4":
            if telemetry["graft_span"] != SPAN:
                raise ValueError("the span arm did not record its span width")
            if span_structure(sorted(positions), block=block, span=SPAN) != len(positions) // SPAN:
                raise ValueError("the span arm did not produce complete runs")
    if picks["point"] == picks["span4"]:
        raise ValueError("the two rules selected the same slots on the self-test window")
    if span_structure(sorted(picks["span4"]), block=block, span=SPAN) != 2:
        raise ValueError("the span arm produced the wrong number of runs")

    # A span that does not tile the block, or a budget that does not divide, must
    # fail loudly instead of silently falling back to the point rule.
    for broken in (3, 5):
        merge = build()
        merge.operator_mode = "native"
        merge.graft_source = "span4"
        merge.graft_span = broken
        try:
            merge(window, carriers)
        except ValueError:
            pass
        else:
            raise ValueError(f"the operator accepted a {broken}-slot span")
    merge = build()
    merge.operator_mode = "native"
    merge.graft_source = "nonsense"
    try:
        merge(window, carriers)
    except ValueError:
        pass
    else:
        raise ValueError("the operator accepted an unknown graft_source")

    if len(set(OUTCOMES.values())) != 8:
        raise ValueError("the outcome vocabulary must name all eight patterns")
    for key, name in OUTCOMES.items():
        if _outcome(*key) != name:
            raise ValueError(f"_outcome{key} = {_outcome(*key)}, expected {name}")
        documented = OUTCOME_VOCABULARY[name]["pattern"]
        derived = "A=%d P=%d S=%d" % tuple(int(flag) for flag in key)
        if documented != derived:
            raise ValueError(f"{name}: documented {documented!r}, derived {derived!r}")
    if abs(er.mcnemar_exact(11, 5) - 0.2101) > 0.0005:
        raise ValueError(f"McNemar(11, 5) = {er.mcnemar_exact(11, 5)}, expected 0.2101")
    if abs(er.mcnemar_exact(9, 5) - 0.4240) > 0.0005:
        raise ValueError(f"McNemar(9, 5) = {er.mcnemar_exact(9, 5)}, expected 0.4240")

    return dict(self_test="ok", block=block, span=SPAN, carriers=len(carriers),
                graft_slots=len(picks["point"]),
                point_digest=digest(sorted(picks["point"])),
                span_digest=digest(sorted(picks["span4"])),
                shared_slots=len(picks["point"] & picks["span4"]),
                mcnemar_11_5=round(er.mcnemar_exact(11, 5), 6))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def build_protocol(args, task):
    return dict(
        experiment="1A span-4 source selection",
        task=task, data_dir=args.data_dir,
        arms=list(ARMS),
        arm_definitions=dict(
            Full="the 19 written blocks, nothing compressed",
            Selection="the 9 carrier blocks, untouched",
            Graft=(f"V10 as trained: the top {GRAFT_SLOTS_TOTAL} individual slots of the "
                   f"{args.window_blocks - args.target_blocks} dropped blocks by leverage"),
            Span=(f"the same dropped blocks scored as non-overlapping {SPAN}-slot spans "
                  f"(mean leverage), top {SPANS_TOTAL} spans = the same "
                  f"{GRAFT_SLOTS_TOTAL} slots in {SPANS_TOTAL} runs of {SPAN}")),
        window_blocks=args.window_blocks, target_blocks=args.target_blocks,
        block_tokens=gr.BLOCK_TOKENS, slots_per_block=BLOCK,
        write_tokens=args.window_blocks * gr.BLOCK_TOKENS,
        full_slots=args.window_blocks * BLOCK,
        target_slots=args.target_blocks * BLOCK,
        core_slots_total=args.target_blocks * CORE_SLOTS,
        graft_slots_total=args.target_blocks * GRAFT_SLOTS,
        span=SPAN, spans_total=SPANS_TOTAL,
        source_selection=("Graft = global top-360 individual slots by leverage (V10's "
                          "trained rule, unchanged); Span = the same budget spent on the "
                          f"top {SPANS_TOTAL} non-overlapping {SPAN}-slot spans ranked by "
                          f"mean leverage"),
        unchanged_conditions=dict(
            data="the same 158 HotpotQA rows as experiments 1 and 2",
            reader="frozen MemoryLLM-8B", carriers="the same 9, unchanged",
            core_selection="original single-point top-216 per carrier",
            source_capacity=GRAFT_SLOTS_TOTAL, output_capacity=TARGET_SLOTS,
            output_vectors="100% native copies", output_order="the existing global "
            "chronological order", training="none"),
        operator_mode="native",
        operator_note=("carrier and graft slots are copied verbatim; no same-block "
                       "condensation, no orthogonal context delta, no core reinforcement"),
        operator_change=("methods/prlmc/manifold_graft.py gained a `graft_source` branch; "
                         "the default 'point' path is the trained rule, and the Graft arm "
                         "is verified row for row against a finished run"),
        pairing_note=("both compressions read one snapshot of the pre-compression window "
                      "and are installed through the same call, so Span - Graft differs "
                      "only in which source slots were picked; the driver checks on every "
                      "sample that the core positions are identical and the graft sets "
                      "differ"),
        compressions_per_sample=2,
        write_stream="document prefix only (the leading template instruction is dropped)",
        query=gr.QUERY_SUFFIX,
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
        arms=list(ARMS), readouts_per_sample=len(ARMS),
        compressions_per_sample=2,
        note=("the two compressions share one snapshot and one install path; the dry run "
              "does not need a GPU"))
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return plan


def evaluate(args):
    facts = gr.load_supporting_facts(args.supporting_facts_glob)
    if not facts:
        print("WARNING: no supporting facts; the evidence grouping will be empty. Pass "
              "--supporting-facts-glob for the actual experiment.", file=sys.stderr)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    if args.dry_run:
        return dry_run(args, facts, tokenizer)
    if not args.checkpoint or not Path(args.checkpoint).is_file():
        raise ValueError("this experiment needs the trained operator: pass --checkpoint")

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
    merge.operator_mode = "native"
    if not hasattr(merge, "graft_source"):
        raise ValueError("the loaded operator has no graft_source: manifold_graft.py is "
                         "not the version this driver needs")

    operator_file = "methods/prlmc/manifold_graft.py"
    operator_now = sha(ROOT / operator_file)
    operator_then = ((saved or {}).get("source_manifest") or {}).get(operator_file)
    if operator_then != operator_now:
        print(f"WARNING: {operator_file} differs from the checkpoint's manifest "
              f"(ckpt {str(operator_then)[:8]} != code {operator_now[:8]}). The change is "
              f"supposed to be the added graft_source branch with 'point' as its default; "
              f"--verify-against is what checks that the trained rule still matches.",
              file=sys.stderr)

    report = dict(
        schema="graft_span_results/1", status="running",
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
                              graft=result["arms"][GRAFT]["correct"],
                              span=result["arms"][SPAN_ARM]["correct"],
                              shared_graft_slots=result["source_check"]["shared_graft_slots"],
                              seconds=round(record["seconds"], 1))), flush=True)

    report["summary"] = analyse(report["samples"], args.f1_threshold,
                                excluded=len(report["excluded"]))
    if args.verify_against:
        report["verification_against_point_rule"] = verify_against(
            args.verify_against, reference, table, report["samples"])
    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["status"] = "complete"
    atomic_json(output, report)
    print(json.dumps(report["summary"], indent=1, ensure_ascii=False))
    verification = report.get("verification_against_point_rule")
    if verification and not verification["passed"]:
        print("WARNING: the point arm does NOT reproduce the reference run; the two "
              "source rules are then not comparable. "
              f"{json.dumps({arm: verification[arm]['different'] for arm in (FULL, SELECTION, GRAFT)})}",
              file=sys.stderr)


def analyze_existing(path, threshold=None):
    report = json.loads(Path(path).read_text())
    if threshold is None:
        threshold = report.get("protocol", {}).get("f1_threshold", 0.5)
    summary = analyse(report.get("samples") or [], threshold,
                      excluded=len(report.get("excluded") or []))
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
    p.add_argument("--data-dir", default=None)
    p.add_argument("--task", default="hotpotqa", choices=("hotpotqa",))
    p.add_argument("--supporting-facts-glob", dest="supporting_facts_glob", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--window-blocks", dest="window_blocks", type=int, default=WINDOW_BLOCKS)
    p.add_argument("--target-blocks", dest="target_blocks", type=int, default=TARGET_BLOCKS)
    p.add_argument("--f1-threshold", dest="f1_threshold", type=float, default=None)
    p.add_argument("--sample-ids", dest="sample_ids", default=None)
    p.add_argument("--max-samples", dest="max_samples", type=int, default=None)
    p.add_argument("--verify-against", dest="verify_against", default=None,
                   help="a finished experiment-1/2 report; the Graft arm is compared with "
                        "it row by row, because the operator gained a branch")
    p.add_argument("--strict-code-hash", dest="strict_code_hash", default="true",
                   choices=("true", "false"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--analyze", default=None)
    p.add_argument("--self-test", dest="self_test", action="store_true")
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
        raise SystemExit(f"this experiment is pinned to {WINDOW_BLOCKS}->{TARGET_BLOCKS}")
    if args.out is None and not args.dry_run:
        raise SystemExit("--out is required unless --dry-run")
    if args.f1_threshold is None:
        args.f1_threshold = 0.5
    evaluate(args)


if __name__ == "__main__":
    main()
