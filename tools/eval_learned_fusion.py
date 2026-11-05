#!/usr/bin/env python3
"""Experiment 1B -- can a learned many-to-one fusion beat plain deletion?

The one difference between the two arms
---------------------------------------
Both arms keep the V10 working point exactly as experiments 1, 2 and 1A left it:
the same 158 HotpotQA rows, the frozen MemoryLLM-8B reader, the same 9 carriers,
the same top-216 core selection per carrier, the same 360 graft slots picked by
the trained single-point leverage rule, the same 2,304 output slots and the same
global chronological output order.

    Graft    delete 40 of a carrier's 256 slots, refill those 40 positions with
             slots copied verbatim from the ten dropped blocks      (V10)
    Fusion   FIRST fuse those 40 doomed slots into that carrier's own 216 kept
             slots with a small residual module, then delete and graft exactly
             as the Graft arm does

The module is trained offline (`tools/train_core_fusion.py`) on a disjoint
LongBench task with answer-free reader-readout distillation. Nothing is trained
during this evaluation.

Why the comparison is paired, and how that is checked
-----------------------------------------------------
Both windows are computed from ONE snapshot of the pre-compression window, by one
operator call, and the fused window is produced from that same output. The driver
then checks, on every sample:

* the graft slots of the fused window are BYTE-IDENTICAL to the unfused one
  (`torch.equal` on the 360 columns) -- the fusion only writes at core offsets;
* the core positions, the graft positions and the 360-slot quota are the same;
* the core slots did change, otherwise the arm measures nothing;
* the fusion used exactly 40 doomed slots per carrier, and none of them is in
  the output.

The four numbers this experiment exists for
-------------------------------------------
1. is the knowledge damage smaller than the Graft arm's -- accuracy and QA F1
   against the uncompressed window, on every graded row;
2. does it still hold after a SECOND compression round, on the 88 rows long
   enough to carry one (the round-2 window is round 1's 9 blocks plus 10 fresh
   native blocks, compressed 19 -> 9 again; each arm continues from its OWN
   round-1 memory);
3. is overall accuracy above the Graft arm's (paired, exact McNemar);
4. did the mechanism do what it was trained to do -- the reader-readout distance
   to the uncompressed window, measured at the four distillation taps (trained)
   and at four HELD-OUT layers (4/12/20/28, never optimised). A module that
   closes the trained-tap gap and not the held-out one has learned a narrow
   proxy; one that closes neither has not learned the task at all.

Usage
-----
    python tools/eval_learned_fusion.py --self-test          # CPU only

    python tools/eval_learned_fusion.py --dry-run --data-dir ... --task hotpotqa \\
        --supporting-facts-glob '...' --model-path /dev/shm/s1d_reader_fp16

    python tools/eval_learned_fusion.py --data-dir ... --task hotpotqa \\
        --supporting-facts-glob '...' --model-path /dev/shm/s1d_reader_fp16 \\
        --checkpoint runs/natural_merge_v9/full_v10.pt \\
        --fusion runs/core_fusion/fusion_rank32.pt \\
        --verify-against reports/diagnostics/graft_span_hotpotqa.json \\
        --out reports/diagnostics/learned_fusion_hotpotqa.json

    python tools/eval_learned_fusion.py --analyze reports/diagnostics/learned_fusion_hotpotqa.json
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
from methods.prlmc.core_fusion import (                   # noqa: E402
    LearnedCoreFusion, SCHEMA as FUSION_SCHEMA, fuse_output, prompt_hidden_states,
    readout_distance, split_core_and_deleted)
from tools.v9_geometry_50block import DISTILL_LAYERS      # noqa: E402

BLOCK = gr.SLOTS_PER_BLOCK                                # 256 slots per block
WINDOW_BLOCKS = gr.WINDOW_BLOCKS                          # 19
TARGET_BLOCKS = gr.TARGET_BLOCKS                          # 9
CORE_SLOTS = gr.CORE_SLOTS                                # 216
GRAFT_SLOTS = gr.GRAFT_SLOTS                              # 40
TARGET_SLOTS = gr.TARGET_SLOTS                            # 2,304
CORE_SLOTS_TOTAL = gr.CORE_SLOTS_TOTAL                    # 1,944
GRAFT_SLOTS_TOTAL = gr.GRAFT_SLOTS_TOTAL                  # 360
DELETED_SLOTS_TOTAL = TARGET_BLOCKS * (BLOCK - CORE_SLOTS)  # 360

FULL, SELECTION, GRAFT, FUSION = "Full", "Selection", "Graft", "Fusion"
GRAFT2, FUSION2 = "Graft2", "Fusion2"
ROUND1_ARMS = (FULL, SELECTION, GRAFT, FUSION)
ARMS = ROUND1_ARMS + (GRAFT2, FUSION2)
# The primary comparison first; `Graft - Selection` is the reproduction check
# against experiments 1, 2 and 1A.
PAIRS = ((FUSION, GRAFT), (FUSION, SELECTION), (GRAFT, SELECTION),
         (FUSION2, GRAFT2), (FUSION2, FUSION), (GRAFT2, GRAFT))

TAPS = tuple(int(layer) for layer in DISTILL_LAYERS)      # (8, 16, 24, 31)
# Never optimised: if the fusion only closes the gap at the taps it was trained
# on, it learned the probe and not the reader.
HELD_OUT_LAYERS = (4, 12, 20, 28)
TAP_LAYERS = TAPS + HELD_OUT_LAYERS
# `--per-layer` reports the readout distance at every reader layer instead of at
# the eight probe layers. It exists because a checkpoint distilled at all 32
# layers moved some layers and destroyed others, and an eight-layer mean cannot
# show that.
ALL_LAYERS = tuple(range(32))

SCREENING_CAVEAT = gr.SCREENING_CAVEAT

# (Selection, Graft, Fusion) correctness, all eight patterns named.
OUTCOMES = {
    (True, True, True): "unchanged_correct",
    (True, True, False): "fusion_loses",
    (True, False, True): "fusion_recovers",
    (True, False, False): "both_lose",
    (False, True, True): "graft_only_fix",
    (False, True, False): "graft_fix_fusion_lose",
    (False, False, True): "fusion_only_fix",
    (False, False, False): "unchanged_wrong",
}
OUTCOME_VOCABULARY = {
    "unchanged_correct": dict(pattern="A=1 G=1 F=1", meaning="right everywhere"),
    "fusion_loses": dict(pattern="A=1 G=1 F=0",
                         meaning="deletion was harmless, the fusion breaks it"),
    "fusion_recovers": dict(pattern="A=1 G=0 F=1",
                            meaning="deletion broke it and the fusion does not"),
    "both_lose": dict(pattern="A=1 G=0 F=0",
                      meaning="both arms lose the same question"),
    "graft_only_fix": dict(pattern="A=0 G=1 F=1",
                           meaning="the graft fixes it and the fusion keeps it"),
    "graft_fix_fusion_lose": dict(pattern="A=0 G=1 F=0",
                                  meaning="the graft fixes it, the fusion loses it"),
    "fusion_only_fix": dict(pattern="A=0 G=0 F=1",
                            meaning="only the fusion arm gets it right"),
    "unchanged_wrong": dict(pattern="A=0 G=0 F=0",
                            meaning="wrong under both arms"),
}


def _check_geometry():
    if (WINDOW_BLOCKS, TARGET_BLOCKS) != (19, 9):
        raise ValueError(f"this experiment is pinned to 19->9, got "
                         f"{WINDOW_BLOCKS}->{TARGET_BLOCKS}")
    if CORE_SLOTS + GRAFT_SLOTS != BLOCK:
        raise ValueError("core+graft slots must tile a native block")
    if DELETED_SLOTS_TOTAL != GRAFT_SLOTS_TOTAL:
        raise ValueError("the deleted and grafted slot counts must match: the arm "
                         "only fuses what the graft replaces")
    if set(TAPS) & set(HELD_OUT_LAYERS):
        raise ValueError("a held-out tap is also a training tap")
    if not {FULL, SELECTION, GRAFT}.issubset(set(er.ARMS)):
        raise ValueError("eval_replace_reorder no longer defines Full/Selection/Graft")


_check_geometry()


# ---------------------------------------------------------------------------
# window preparation
# ---------------------------------------------------------------------------
def attach_round2_blocks(prepared, rows, tokenizer, args):
    """Give every multi-round row its second round's native blocks.

    `gr.prepare_sample` truncates the write stream to the 19-block window, so the
    driver cannot take "the next ten blocks" from it. They are re-tokenized from
    the same row here, and the prefix is checked against `write_ids` so the
    re-tokenization is proven identical to the one the window was built from
    rather than assumed to be.
    """
    need = args.window_blocks + args.round2_blocks
    window_tokens = args.window_blocks * gr.BLOCK_TOKENS
    attached = 0
    for sample in prepared:
        if sample["available_blocks"] < need:
            continue
        row = rows[sample["sample_index"]]
        ids = tokenizer(row["context"], add_special_tokens=False)["input_ids"]
        if list(ids[:window_tokens]) != list(sample["write_ids"]):
            raise ValueError(f"{sample['id']}: the re-tokenized prefix differs from "
                             f"the write stream the window was built from")
        extra = list(ids[window_tokens:need * gr.BLOCK_TOKENS])
        if len(extra) != args.round2_blocks * gr.BLOCK_TOKENS:
            raise ValueError(f"{sample['id']}: {len(extra)} extra tokens, expected "
                             f"{args.round2_blocks * gr.BLOCK_TOKENS}")
        sample["round2_ids"] = extra
        attached += 1
    return attached


def write_blocks(manager, merge, chunks):
    """Append native blocks through the plain (non-compacting) writer."""
    from stream_eval import StreamingMemory
    stream = StreamingMemory(manager, merge, "Full", budget_blocks=gr.BUDGET_BLOCKS)
    for index, chunk in enumerate(chunks):
        record = dict(role="write", token_ids=chunk, token_sha256=digest(chunk))
        if stream.write(record, index) is None:
            raise ValueError("the writer refused a block: the window does not fit the pool")
    return stream


def compress_window(manager, merge, fusion, window_blocks, target_blocks):
    """One compression, its fusion, and the structure both are checked against."""
    source, ages, picked = gr.select_carriers(manager, window_blocks, target_blocks)
    frozen, pinned = gr.window_locks(manager, window_blocks)
    with torch.no_grad():
        output, telemetry = gr.compress_graft(manager, merge, source, picked,
                                              window_blocks, target_blocks,
                                              frozen, pinned)
        parts = split_core_and_deleted(telemetry["slot_source_positions"],
                                       telemetry["bridge_slot_offsets"], picked,
                                       manager.block, CORE_SLOTS,
                                       device=source.device)
        fused, info = fuse_output(output, source, parts, fusion)
    return dict(source=source, ages=ages, picked=picked, output=output, fused=fused,
                parts=parts, telemetry=telemetry, fusion_info=info)


def check_fusion(round_result, name, block=BLOCK):
    """The fused window must differ from the unfused one at the core and nowhere else."""
    telemetry = round_result["telemetry"]
    output, fused = round_result["output"], round_result["fused"]
    parts = round_result["parts"]
    device = output.device
    graft = torch.as_tensor(telemetry["bridge_slot_offsets"], dtype=torch.long,
                            device=device)
    core = parts["core_offsets"]
    if int(graft.numel()) != GRAFT_SLOTS_TOTAL:
        raise ValueError(f"{name}: {int(graft.numel())} graft slots, expected "
                         f"{GRAFT_SLOTS_TOTAL}")
    if int(core.numel()) != CORE_SLOTS_TOTAL:
        raise ValueError(f"{name}: {int(core.numel())} core slots, expected "
                         f"{CORE_SLOTS_TOTAL}")
    if len(parts["deleted_positions"]) != DELETED_SLOTS_TOTAL:
        raise ValueError(f"{name}: the fusion was aimed at "
                         f"{len(parts['deleted_positions'])} slots, expected "
                         f"{DELETED_SLOTS_TOTAL}")
    if telemetry["selected_blocks"] != sorted(int(b) for b in round_result["picked"]):
        raise ValueError(f"{name}: the compression disagrees with the carrier set")
    if not torch.equal(output.index_select(1, graft), fused.index_select(1, graft)):
        raise ValueError(f"{name}: the fusion changed a GRAFT slot; the two arms no "
                         f"longer differ only in the core")
    if torch.equal(output.index_select(1, core), fused.index_select(1, core)):
        raise ValueError(f"{name}: the fusion left every core slot unchanged, so the "
                         f"arm measures nothing")
    if not torch.isfinite(fused).all():
        raise ValueError(f"{name}: the fused window is not finite")
    survivors = set(int(position) for position in telemetry["slot_source_positions"])
    for position in parts["deleted_positions"]:
        if position in survivors:
            raise ValueError(f"{name}: window position {position} is both fused and "
                             f"kept")
    kept = Counter(position // block for position in parts["core_positions"])
    if set(kept.values()) != {CORE_SLOTS}:
        raise ValueError(f"{name}: carriers kept an unequal number of slots: {kept}")
    gone = Counter(position // block for position in parts["deleted_positions"])
    if set(gone.values()) != {BLOCK - CORE_SLOTS}:
        raise ValueError(f"{name}: carriers deleted an unequal number of slots: {gone}")
    return dict(
        graft_slots=int(graft.numel()), core_slots=int(core.numel()),
        deleted_slots=int(len(parts["deleted_positions"])),
        deleted_per_carrier=sorted(set(gone.values())),
        graft_bit_identical=True,
        core_changed=True,
        relative_delta=round(round_result["fusion_info"]["relative_delta"], 6),
        gate=round(round_result["fusion_info"]["gate"], 6))


def readout_gap(model, source, compressed, query_hidden, layers):
    """Reader-readout distance from the uncompressed window, at `layers`."""
    pre = {layer: source[layer] for layer in layers}
    post = {layer: compressed[layer] for layer in layers}
    with torch.no_grad():
        return float(readout_distance(model, pre, post, query_hidden, layers))


def per_layer_gap(model, source, compressed, query_hidden, layers):
    """`readout_gap` one layer at a time: {layer: distance}."""
    gaps = {}
    for layer in layers:
        gaps[int(layer)] = readout_gap(model, source, compressed, query_hidden, [layer])
    return gaps


def selection_window(source, picked):
    """The Selection arm's window: the carrier blocks kept whole, nothing else."""
    blocks = source.reshape(source.shape[0], source.shape[1] // BLOCK, BLOCK,
                            source.shape[2])
    selected = torch.as_tensor(picked, dtype=torch.long, device=source.device)
    return blocks.index_select(1, selected).flatten(1, 2)


def readout_once(model, tokenizer, query_ids, answers, threshold, prepare):
    """`gr.readout` with a single preparation.

    `gr.readout` re-applies the arm before the gold-likelihood pass as well, which
    is free for an arm whose preparation is the `install` of an already-computed
    window. A round-2 preparation has to replay ten native writes and a second
    compression, so it is done once here; generation runs under `isolated_query`,
    which restores the pool, so the likelihood pass sees exactly the memory the
    answer was generated from.
    """
    prepare()
    prediction, output_ids = pe.generate(model, tokenizer, query_ids, gr.MAX_NEW)
    gold = gr.answer_logprob(model, tokenizer, query_ids, answers[0])
    record = dict(prediction=prediction, output_tokens=output_ids,
                  qa_f1=float(nv.score(prediction, answers)),
                  qa_f1_truncated=float(nv.score(nv.truncated_prediction(prediction),
                                                 answers)),
                  em=int(any(gr.normalise(prediction) == gr.normalise(answer)
                             for answer in answers)),
                  correct=int(nv.score(prediction, answers) >= threshold),
                  f1_threshold=threshold)
    if gold is not None:
        record.update(gold)
    return record


# ---------------------------------------------------------------------------
# one sample
# ---------------------------------------------------------------------------
def run_sample(model, tokenizer, manager, merge, fusion, prepared, args):
    """Write the window once, compress once, then answer every arm."""
    chunks = [prepared["write_ids"][index:index + gr.BLOCK_TOKENS]
              for index in range(0, len(prepared["write_ids"]), gr.BLOCK_TOKENS)]
    answers = prepared["answers"]
    fingerprints, gaps = {}, {}
    multi_round = "round2_ids" in prepared
    if multi_round != (prepared["available_blocks"]
                       >= args.window_blocks + args.round2_blocks):
        raise ValueError("the round-2 write stream is missing for an eligible row")

    probe = ALL_LAYERS if getattr(args, "per_layer", False) else TAP_LAYERS
    with torch.no_grad():
        write_blocks(manager, merge, chunks[:args.window_blocks])
        state = gr.snapshot(model, manager)
        query_hidden = prompt_hidden_states(model, prepared["query_ids"], probe)
        round1 = compress_window(manager, merge, fusion, args.window_blocks,
                                 args.target_blocks)
    structure1 = check_fusion(round1, "round 1")

    selection = selection_window(round1["source"], round1["picked"])
    gaps[SELECTION] = readout_gap(model, round1["source"], selection, query_hidden, TAPS)
    gaps[GRAFT] = readout_gap(model, round1["source"], round1["output"], query_hidden, TAPS)
    gaps[FUSION] = readout_gap(model, round1["source"], round1["fused"], query_hidden, TAPS)
    gaps[GRAFT + "/heldout"] = readout_gap(model, round1["source"], round1["output"],
                                           query_hidden, HELD_OUT_LAYERS)
    gaps[FUSION + "/heldout"] = readout_gap(model, round1["source"], round1["fused"],
                                            query_hidden, HELD_OUT_LAYERS)
    if getattr(args, "per_layer", False):
        gaps[GRAFT + "/perlayer"] = per_layer_gap(model, round1["source"], round1["output"],
                                                  query_hidden, ALL_LAYERS)
        gaps[FUSION + "/perlayer"] = per_layer_gap(model, round1["source"], round1["fused"],
                                                   query_hidden, ALL_LAYERS)
    del selection
    gr.restore(model, manager, state)

    def prepare(arm):
        def run():
            gr.restore(model, manager, state)
            if arm == SELECTION:
                gr.install_selection(manager, round1["source"], round1["picked"],
                                     args.window_blocks)
            elif arm == GRAFT:
                er.install_output(manager, round1["output"], round1["picked"],
                                  args.window_blocks)
            elif arm == FUSION:
                er.install_output(manager, round1["fused"], round1["picked"],
                                  args.window_blocks)
            fingerprints[arm] = gr.active_fingerprint(manager)
            fingerprints[arm + "/slots"] = int(manager.used)
        return run

    arms = {arm: gr.readout(model, tokenizer, prepared["query_ids"], answers,
                            args.f1_threshold, prepare(arm))
            for arm in ROUND1_ARMS}
    gr.restore(model, manager, state)

    structure = {SELECTION: None, GRAFT: structure1, FUSION: structure1}
    if multi_round:
        extra = [prepared["round2_ids"][index:index + gr.BLOCK_TOKENS]
                 for index in range(0, len(prepared["round2_ids"]), gr.BLOCK_TOKENS)]
        if len(extra) != args.round2_blocks:
            raise ValueError(f"{len(extra)} round-2 blocks, expected "
                             f"{args.round2_blocks}")
        for arm, source_arm, round2_arm in ((GRAFT, "output", GRAFT2),
                                            (FUSION, "fused", FUSION2)):
            def prepare_round2(arm=arm, source_arm=source_arm):
                def run():
                    gr.restore(model, manager, state)
                    er.install_output(manager, round1[source_arm], round1["picked"],
                                      args.window_blocks)
                    with torch.no_grad():
                        write_blocks(manager, merge, extra)
                        round2 = compress_window(manager, merge, fusion,
                                                 args.window_blocks, args.target_blocks)
                    structure[round2_arm] = check_fusion(round2, f"round 2 ({arm})")
                    er.install_output(manager, round2[source_arm], round2["picked"],
                                      args.window_blocks)
                    fingerprints[round2_arm] = gr.active_fingerprint(manager)
                    fingerprints[round2_arm + "/slots"] = int(manager.used)
                return run
            arms[round2_arm] = readout_once(model, tokenizer, prepared["query_ids"],
                                            answers, args.f1_threshold,
                                            prepare_round2())
        gr.restore(model, manager, state)

    if int(fingerprints[FULL + "/slots"]) != args.window_blocks * manager.block:
        raise ValueError("the Full arm did not keep the whole window")
    for arm in (SELECTION, GRAFT, FUSION, GRAFT2, FUSION2):
        if arm in fingerprints and int(fingerprints[arm + "/slots"]) != TARGET_SLOTS:
            raise ValueError(f"arm {arm} does not hold the {TARGET_SLOTS}-slot budget")
    if fingerprints[SELECTION] == fingerprints[GRAFT]:
        raise ValueError("Selection and Graft read the same pool: no slot was grafted")
    if fingerprints[GRAFT] == fingerprints[FUSION]:
        raise ValueError("Graft and Fusion read the same pool: the fusion did not run")
    if multi_round and fingerprints[GRAFT2] == fingerprints[FUSION2]:
        raise ValueError("the two round-2 arms read the same pool")

    provenance = gr.graft_provenance(round1["telemetry"], args.window_blocks)
    if provenance["selected_blocks"] != sorted(round1["picked"]):
        raise ValueError("graft provenance disagrees with the shared carrier set")
    return dict(arms=arms, fingerprints=fingerprints, gaps=gaps,
                provenance=provenance, structure=structure,
                multi_round=multi_round,
                carrier_ages=[int(round1["ages"][block]) for block in round1["picked"]])


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def _outcome(selection, graft, fusion):
    return OUTCOMES[(bool(selection), bool(graft), bool(fusion))]


def _states(sample, threshold):
    return (gr._correct(sample, SELECTION, threshold),
            gr._correct(sample, GRAFT, threshold),
            gr._correct(sample, FUSION, threshold))


def _stage_counts(samples, threshold):
    counts = Counter()
    for sample in samples:
        counts[_outcome(*_states(sample, threshold))] += 1
    return dict(sorted(counts.items()))


def _transitions(samples, threshold):
    rows = []
    for sample in samples:
        selection, graft, fusion = _states(sample, threshold)
        rows.append(dict(
            id=sample["id"], selection=int(selection), graft=int(graft),
            fusion=int(fusion), outcome=_outcome(selection, graft, fusion),
            qa_f1=dict(selection=round(sample["arms"][SELECTION]["qa_f1"], 4),
                       graft=round(sample["arms"][GRAFT]["qa_f1"], 4),
                       fusion=round(sample["arms"][FUSION]["qa_f1"], 4))))
    return rows


def _evidence_in_dropped(sample):
    return bool(set(sample["evidence"]["blocks"])
                & set(sample["provenance"]["dropped_blocks"]))


def _gap_summary(samples, key):
    values = [sample["gaps"][key] for sample in samples
              if sample.get("gaps", {}).get(key) is not None]
    if not values:
        return dict(n=0)
    return dict(n=len(values), mean=round(float(sum(values) / len(values)), 5),
                median=round(sorted(values)[len(values) // 2], 5))


def _gap_removed(samples, treatment, baseline):
    rows = [sample for sample in samples
            if sample.get("gaps", {}).get(treatment) is not None
            and sample.get("gaps", {}).get(baseline) is not None
            and sample["gaps"][baseline] > 0]
    if not rows:
        return dict(n=0)
    removed = [1.0 - sample["gaps"][treatment] / sample["gaps"][baseline]
               for sample in rows]
    return dict(n=len(rows), mean_removed=round(float(sum(removed) / len(removed)), 4),
                up=int(sum(1 for value in removed if value > 0)),
                down=int(sum(1 for value in removed if value < 0)))


def _per_layer_summary(samples, distill_layers=None):
    """Per-layer `fusion - graft` on the readout distance, when `--per-layer` ran.

    The eight-layer mean hides a module that trades one group of layers for
    another; this table is what shows it. Negative change means the fusion moved
    that layer's readout closer to the uncompressed window.
    """
    rows = [sample for sample in samples
            if sample.get("gaps", {}).get(GRAFT + "/perlayer")
            and sample.get("gaps", {}).get(FUSION + "/perlayer")]
    if not rows:
        return None
    supervised = set(int(layer) for layer in (distill_layers or TAPS))
    every_layer = len(supervised) >= 32
    layers = sorted(rows[0]["gaps"][GRAFT + "/perlayer"])
    table = []
    for layer in layers:
        graft, fusion, change = [], [], []
        for sample in rows:
            g = sample["gaps"][GRAFT + "/perlayer"].get(layer)
            f = sample["gaps"][FUSION + "/perlayer"].get(layer)
            if g is None or f is None or g <= 0:
                continue
            graft.append(g)
            fusion.append(f)
            change.append(1.0 - f / g)
        if not change:
            continue
        table.append(dict(
            layer=int(layer), supervised=bool(int(layer) in supervised),
            graft=round(float(sum(graft) / len(graft)), 5),
            fusion=round(float(sum(fusion) / len(fusion)), 5),
            removed=round(float(sum(change) / len(change)), 4)))
    if not table:
        return None
    supervised_rows = [row["removed"] for row in table if row["supervised"]]
    other = [row["removed"] for row in table if not row["supervised"]]
    mean = lambda values: (round(float(sum(values) / len(values)), 4) if values else None)
    best = max(table, key=lambda row: row["removed"])
    worst = min(table, key=lambda row: row["removed"])
    return dict(
        definition=("per-layer mean readout distance, graft vs fusion, over the same "
                    "rows; `removed` is the fraction of the graft distance the fusion "
                    "takes away, so positive is closer to the uncompressed window"),
        n=len(rows),
        mean_removed=mean([row["removed"] for row in table]),
        supervised_mean_removed=(None if every_layer else mean(supervised_rows)),
        other_mean_removed=(None if every_layer else mean(other)),
        note=("this checkpoint was distilled at every layer, so the supervised / other "
              "split is empty: read the per-layer rows instead"
              if every_layer else
              "supervised layers are the ones the loss covered; the rest were never "
              "optimised and are the honest test of whether the readout moved"),
        improved_layers=[row["layer"] for row in table if row["removed"] > 0],
        best_layer=dict(layer=best["layer"], removed=best["removed"]),
        worst_layer=dict(layer=worst["layer"], removed=worst["removed"]),
        layers=table)


def _damage(samples, arm, threshold):
    """Accuracy lost against the uncompressed window, on the rows Full gets right."""
    protect = [sample for sample in samples if gr._correct(sample, FULL, threshold)]
    if not protect:
        return dict(n=0)
    lost = len([sample for sample in protect if not gr._correct(sample, arm, threshold)])
    return dict(n=len(protect), lost=lost,
                kept=len(protect) - lost,
                accuracy_on_protect=round(100.0 * (len(protect) - lost) / len(protect), 2))


def analyse(samples, threshold, excluded=0, round2_blocks=10, distill_layers=None):
    graded = [sample for sample in samples if sample.get("status") == "graded"]
    for sample in graded:
        missing = [arm for arm in ROUND1_ARMS if arm not in sample.get("arms", {})]
        if missing:
            raise ValueError(f"sample {sample.get('id')} has no {missing} arm: this "
                             f"report was not produced by this driver")
        if not sample.get("structure", {}).get(GRAFT):
            raise ValueError(f"sample {sample.get('id')} carries no fusion structure "
                             f"check: the two arms cannot be told apart afterwards")
    located = [sample for sample in graded if sample.get("evidence", {}).get("located")]
    dropped = [sample for sample in located if _evidence_in_dropped(sample)]
    kept = [sample for sample in located if not _evidence_in_dropped(sample)]
    multi = [sample for sample in graded if sample.get("multi_round")]
    present = [arm for arm in ARMS
               if any(arm in sample.get("arms", {}) for sample in graded)]
    trained_everywhere = bool(distill_layers) and len(set(distill_layers)) >= 32

    changed = [sample for sample in graded
               if gr._correct(sample, SELECTION, threshold)
               != gr._correct(sample, GRAFT, threshold)]
    protect = [sample for sample in located
               if gr._correct(sample, FULL, threshold)
               and gr._correct(sample, SELECTION, threshold)]

    def rescued_harmed(rows, treatment):
        rescued = [sample for sample in rows
                   if not gr._correct(sample, SELECTION, threshold)
                   and gr._correct(sample, treatment, threshold)]
        harmed = [sample for sample in rows
                  if gr._correct(sample, SELECTION, threshold)
                  and not gr._correct(sample, treatment, threshold)]
        return len(rescued), len(harmed)

    graft_rescued, graft_harmed = rescued_harmed(changed, GRAFT)
    fusion_rescued, fusion_harmed = rescued_harmed(changed, FUSION)

    return dict(
        counts=dict(
            attempted=len(graded) + int(excluded), graded=len(graded),
            excluded=int(excluded), evidence_located=len(located),
            evidence_in_dropped=len(dropped), multi_round=len(multi),
            changed_by_compression=len(changed),
            full_correct=len([s for s in graded if gr._correct(s, FULL, threshold)]),
            selection_correct=len([s for s in graded
                                   if gr._correct(s, SELECTION, threshold)]),
            graft_correct=len([s for s in graded if gr._correct(s, GRAFT, threshold)]),
            fusion_correct=len([s for s in graded if gr._correct(s, FUSION, threshold)]),
            graft2_correct=len([s for s in multi if gr._correct(s, GRAFT2, threshold)]),
            fusion2_correct=len([s for s in multi
                                 if gr._correct(s, FUSION2, threshold)])),
        arms={arm: gr._arm_summary(graded, arm, threshold) for arm in present},
        arms_multi_round={arm: gr._arm_summary(multi, arm, threshold) for arm in present},
        pairs={f"{treatment}_minus_{baseline}": er.pair_summary(
                   graded, treatment, baseline, threshold)
               for treatment, baseline in PAIRS},
        pairs_dropped={f"{treatment}_minus_{baseline}": er.pair_summary(
                           dropped, treatment, baseline, threshold)
                       for treatment, baseline in PAIRS},
        pairs_kept={f"{treatment}_minus_{baseline}": er.pair_summary(
                        kept, treatment, baseline, threshold)
                    for treatment, baseline in PAIRS},
        pairs_multi_round={f"{treatment}_minus_{baseline}": er.pair_summary(
                               multi, treatment, baseline, threshold)
                           for treatment, baseline in PAIRS},
        # (1) is the knowledge damage smaller?
        damage=dict(
            definition=("rows the UNCOMPRESSED window answers correctly; `lost` is how "
                        "many of them the arm no longer answers"),
            full=gr._arm_summary(graded, FULL, threshold),
            by_arm={arm: _damage(graded, arm, threshold)
                    for arm in (SELECTION, GRAFT, FUSION)},
            damage_delta=(_damage(graded, FUSION, threshold).get("lost", 0)
                          - _damage(graded, GRAFT, threshold).get("lost", 0))),
        # (2) does it survive a second round?
        multi_round_group=dict(
            definition=(f"rows with at least {WINDOW_BLOCKS + round2_blocks} native "
                        f"blocks, so round 2 can compress round 1's {TARGET_BLOCKS} "
                        f"blocks plus {round2_blocks} fresh ones"),
            n=len(multi),
            arms={arm: gr._arm_summary(multi, arm, threshold) for arm in present},
            fusion2_minus_graft2=er.pair_summary(multi, FUSION2, GRAFT2, threshold),
            fusion2_minus_fusion=er.pair_summary(multi, FUSION2, FUSION, threshold),
            graft2_minus_graft=er.pair_summary(multi, GRAFT2, GRAFT, threshold),
            damage={arm: _damage(multi, arm, threshold)
                    for arm in (GRAFT, FUSION, GRAFT2, FUSION2)}),
        # (3) the 16 questions the compression changes at all
        changed_group=dict(
            definition=("Selection and the trained graft disagree on correctness, i.e. "
                        "every question the compression changes in either direction"),
            n=len(changed),
            rescued_by_graft=graft_rescued, harmed_by_graft=graft_harmed,
            rescued_by_fusion=fusion_rescued, harmed_by_fusion=fusion_harmed,
            net_graft=graft_rescued - graft_harmed,
            net_fusion=fusion_rescued - fusion_harmed,
            stages=_stage_counts(changed, threshold),
            transitions=_transitions(changed, threshold)),
        # (4) did the mechanism do what it was trained to do
        readout=dict(
            definition=("mean reader-readout distance from the uncompressed window at "
                        "the last 8 question tokens; lower is closer"),
            trained_taps=list(TAPS), held_out_layers=list(HELD_OUT_LAYERS),
            distill_layers=(list(distill_layers) if distill_layers else None),
            by_arm={arm: _gap_summary(graded, arm)
                    for arm in (SELECTION, GRAFT, FUSION)},
            held_out=({arm: _gap_summary(graded, arm + "/heldout")
                       for arm in (GRAFT, FUSION)} if not trained_everywhere else None),
            fusion_removes=_gap_removed(graded, FUSION, GRAFT),
            fusion_removes_held_out=(_gap_removed(graded, FUSION + "/heldout",
                                                  GRAFT + "/heldout")
                                     if not trained_everywhere else None),
            per_layer=_per_layer_summary(graded, distill_layers),
            note=("this checkpoint was distilled at EVERY reader layer, so no layer is "
                  "held out and the four 'held-out' layers above are trained too: the "
                  "gap is a trained-layer number, and the honest test of whether the "
                  "knowledge moved is the accuracy, not this distance"
                  if trained_everywhere else
                  "the trained taps are the loss the module was optimised on; the "
                  "held-out layers were never optimised and are the honest test of "
                  "whether the readout moved or only the probe")),
        dropped_evidence_group=dict(
            definition=("the supporting sentence sits in a block the compression "
                        "dropped, so only the graft slots can carry it"),
            n=len(dropped),
            arms={arm: gr._arm_summary(dropped, arm, threshold) for arm in present},
            fusion_minus_graft=er.pair_summary(dropped, FUSION, GRAFT, threshold),
            stages=_stage_counts(dropped, threshold)),
        protection_group=dict(
            definition="Full correct and Selection correct",
            n=len(protect),
            harmed_by_graft=len([s for s in protect
                                 if not gr._correct(s, GRAFT, threshold)]),
            harmed_by_fusion=len([s for s in protect
                                  if not gr._correct(s, FUSION, threshold)]),
            stages=_stage_counts(protect, threshold),
            transitions=_transitions(protect, threshold)),
        ungated=dict(definition="every graded sample, no evidence filter",
                     stages=_stage_counts(graded, threshold),
                     transitions=_transitions(graded, threshold)),
        vocabulary=OUTCOME_VOCABULARY,
        screening_caveat=SCREENING_CAVEAT,
        f1_threshold=threshold)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def build_protocol(args, task, fusion_config):
    return dict(
        experiment="1B learned many-to-one core fusion before the graft",
        task=task, data_dir=args.data_dir,
        arms=list(ARMS),
        window_blocks=args.window_blocks, target_blocks=args.target_blocks,
        round2_blocks=args.round2_blocks,
        block_tokens=gr.BLOCK_TOKENS, slots_per_block=BLOCK,
        full_slots=args.window_blocks * BLOCK,
        target_slots=args.target_blocks * BLOCK,
        core_slots_total=args.target_blocks * CORE_SLOTS,
        graft_slots_total=args.target_blocks * GRAFT_SLOTS,
        deleted_slots_total=args.target_blocks * (BLOCK - CORE_SLOTS),
        compressions_per_sample=1,
        round2_compressions_per_sample=2,
        round2_definition=("round 2 installs the arm's OWN round-1 window, appends "
                           f"{args.round2_blocks} fresh native blocks and compresses "
                           "19 -> 9 again; it runs only on rows with at least "
                           f"{args.window_blocks + args.round2_blocks} blocks"),
        unchanged_conditions=dict(
            data="the same 158 HotpotQA rows as experiments 1, 2 and 1A",
            reader="frozen MemoryLLM-8B", carriers="the same 9, unchanged",
            core_selection="original single-point top-216 per carrier",
            source_selection="V10's trained single-point leverage rule, unchanged",
            source_capacity=GRAFT_SLOTS_TOTAL, output_capacity=TARGET_SLOTS,
            output_vectors="native copies on both sides; the fused core is the only "
                           "non-native vector",
            output_order="the existing global chronological order"),
        only_difference=("the 40 slots a carrier deletes are fused into that carrier's "
                         "own 216 kept slots before the graft; the graft slots of the "
                         "fused window are byte-identical to the unfused one"),
        fusion=dict(schema=FUSION_SCHEMA, **fusion_config,
                    checkpoint=args.fusion,
                    checkpoint_sha256=(sha(args.fusion) if args.fusion
                                       and Path(args.fusion).is_file() else None),
                    source="the 40 slots the SAME carrier block deletes",
                    training="offline, answer-free reader-readout distillation on a "
                             "disjoint LongBench task; nothing trained here"),
        operator_mode="native",
        per_layer_probe=bool(getattr(args, "per_layer", False)),
        operator_note=("carrier and graft slots are copied verbatim; no same-block "
                       "condensation, no orthogonal context delta, no core reinforcement"),
        pairing_note=("both windows come from ONE snapshot and one operator call; the "
                      "driver checks on every sample that the graft columns are "
                      "bit-identical, the positions and quota are unchanged, and the "
                      "core slots did change"),
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


def load_fusion(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != FUSION_SCHEMA + "/checkpoint":
        raise ValueError(f"{path} is not a {FUSION_SCHEMA} checkpoint")
    config = dict(payload["config"])
    module = LearnedCoreFusion(**config)
    module.load_state_dict(payload["state_dict"], strict=True)
    return module.to(device).eval().requires_grad_(False), config, payload.get("training")


def supervised_layers(training):
    """The reader layers the checkpoint's distillation loss actually covered.

    The trainer records them as `taps`. This decides whether the "held-out
    layers" in the summary are honest: a checkpoint distilled at all 32 layers
    holds nothing out, so those numbers are trained numbers and the only honest
    test left is the accuracy.
    """
    payload = training or {}
    layers = payload.get("distill_layers") or payload.get("taps") or []
    layers = [int(layer) for layer in layers]
    return layers or None


def load_reference(path):
    reference = json.loads(Path(path).read_text())
    table = {str(sample["id"]): sample for sample in reference.get("samples") or []}
    if not table:
        raise ValueError(f"--verify-against {path} holds no samples")
    missing = [arm for arm in (FULL, SELECTION, GRAFT)
               if not any(arm in sample.get("arms", {}) for sample in table.values())]
    if missing:
        raise ValueError(f"--verify-against {path} has no {missing} arms; it is not an "
                         f"experiment-1/1A report")
    return reference, table


def verify_against(path, reference, table, samples):
    """The three untouched arms must reproduce the finished run row for row.

    The fusion is supposed to be the only difference, so `Graft`, `Selection` and
    `Full` have to come out exactly as they did in the run this one is compared
    against. If they do not, the comparison is between two implementations and
    not between two arms.
    """
    this_run = {str(sample["id"]) for sample in samples}
    report = dict(reference=path, reference_schema=reference.get("schema"),
                  reference_samples=len(table), compared=len(this_run),
                  row_count_matches_reference=(len(this_run) == len(table)),
                  reference_not_in_this_run=sorted(set(table) - this_run),
                  this_run_not_in_reference=sorted(this_run - set(table)))
    # A subset run (`--sample-ids`) compares fewer rows by construction; what has
    # to hold is that every row it DID compare matches, and that no row here is
    # missing from the reference.
    passed = not report["this_run_not_in_reference"]
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
        passed = passed and different == 0 and absent == 0 and same == len(this_run)
    report["passed"] = passed
    return report


def dry_run(args, facts, tokenizer):
    rows, prepared, excluded = gr.prepare_rows(args, tokenizer, facts)
    located = [sample for sample in prepared if sample["evidence"]["located"]]
    multi = [sample for sample in prepared
             if sample["available_blocks"] >= args.window_blocks + args.round2_blocks]
    histogram = Counter(block for sample in located for block in sample["evidence"]["blocks"])
    fusion_config = dict(hidden_size=4096, rank=args.fusion_rank,
                         max_ratio=args.fusion_max_ratio, gate_init=-2.0)
    plan = dict(
        dry_run=True, out=args.out,
        protocol=build_protocol(args, args.task, fusion_config),
        counts=dict(rows=len(rows), eligible=len(prepared), excluded=len(excluded),
                    multi_round=len(multi),
                    evidence_in_window=len(located),
                    sentence_found_but_outside_window=len(
                        [sample for sample in prepared
                         if sample["evidence"]["supporting_sentences_located"]
                         and not sample["evidence"]["located"]])),
        evidence_block_histogram={str(block): histogram[block] for block in sorted(histogram)},
        excluded_examples=excluded[:5],
        eligible_ids=[sample["id"] for sample in prepared],
        readouts_per_sample=dict(round1=len(ROUND1_ARMS),
                                 with_round2=len(ARMS),
                                 without_round2=len(ROUND1_ARMS)),
        note=("the dry run does not need a GPU or the fusion checkpoint; the arm "
              "structure is checked on the CPU by --self-test"))
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
    if not args.fusion or not Path(args.fusion).is_file():
        raise ValueError("this experiment needs the trained fusion module: pass --fusion")

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
    multi_round_rows = attach_round2_blocks(prepared, rows, tokenizer, args)
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
    fusion, fusion_config, fusion_training = load_fusion(args.fusion, model.memory.device)
    if fusion_config.get("hidden_size") != int(manager.hidden):
        raise ValueError(f"the fusion module was built for hidden "
                         f"{fusion_config.get('hidden_size')}, the reader is "
                         f"{manager.hidden}")

    operator_file = "methods/prlmc/manifold_graft.py"
    operator_now = sha(ROOT / operator_file)
    operator_then = ((saved or {}).get("source_manifest") or {}).get(operator_file)
    if operator_then != operator_now:
        print(f"WARNING: {operator_file} differs from the checkpoint's manifest "
              f"(ckpt {str(operator_then)[:8]} != code {operator_now[:8]}). The change "
              f"is experiment 1A's source-selection branch with 'point' as its default; "
              f"--verify-against is what checks the trained rule still matches.",
              file=sys.stderr)

    report = dict(
        schema="learned_fusion_results/1", status="running",
        counts=dict(eligible=len(prepared), multi_round=multi_round_rows),
        protocol=dict(build_protocol(args, args.task, fusion_config),
                      operator_mode_applied=merge.operator_mode,
                      fusion_training=fusion_training),
        identity=dict(
            model_path=args.model_path, checkpoint=args.checkpoint,
            checkpoint_sha256=sha(args.checkpoint),
            checkpoint_code_hash_verified=saved.get("code_hash_verified") if saved else None,
            fusion=args.fusion, fusion_sha256=sha(args.fusion),
            data_dir=args.data_dir,
            data_sha256=sha(Path(args.data_dir) / (args.task + ".jsonl")),
            supporting_facts_glob=args.supporting_facts_glob,
            entrypoint_sha256=sha(__file__),
            compression_entrypoint_sha256=sha(Path(gr.__file__)),
            fusion_module_sha256=sha(ROOT / "methods/prlmc/core_fusion.py"),
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
        result = run_sample(model, tokenizer, manager, merge, fusion, sample, args)
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
                              fusion=result["arms"][FUSION]["correct"],
                              graft2=(result["arms"].get(GRAFT2) or {}).get("correct"),
                              fusion2=(result["arms"].get(FUSION2) or {}).get("correct"),
                              gap_removed=round(
                                  result["gaps"][FUSION] / result["gaps"][GRAFT], 4)
                              if result["gaps"].get(GRAFT) else None,
                              delta=result["structure"][GRAFT]["relative_delta"],
                              seconds=round(record["seconds"], 1))), flush=True)

    report["summary"] = analyse(report["samples"], args.f1_threshold,
                                excluded=len(report["excluded"]),
                                round2_blocks=args.round2_blocks,
                                distill_layers=supervised_layers(fusion_training))
    if args.verify_against:
        report["verification_against_reference"] = verify_against(
            args.verify_against, reference, table, report["samples"])
    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["status"] = "complete"
    atomic_json(output, report)
    print(json.dumps(report["summary"], indent=1, ensure_ascii=False))
    verification = report.get("verification_against_reference")
    if verification and not verification["passed"]:
        print("WARNING: the untouched arms do NOT reproduce the reference run; the two "
              "arms are then not comparable. "
              f"{json.dumps({arm: verification[arm]['different'] for arm in (FULL, SELECTION, GRAFT)})}",
              file=sys.stderr)


def analyze_existing(path, threshold=None):
    report = json.loads(Path(path).read_text())
    protocol = report.get("protocol") or {}
    if threshold is None:
        threshold = protocol.get("f1_threshold", 0.5)
    summary = analyse(report.get("samples") or [], threshold,
                      excluded=len(report.get("excluded") or []),
                      round2_blocks=int(protocol.get("round2_blocks", 10)),
                      distill_layers=supervised_layers(protocol.get("fusion_training")))
    for sample in report.get("samples") or []:
        for record in (sample.get("arms") or {}).values():
            if "qa_f1" in record:
                record["correct"] = int(record["qa_f1"] >= threshold)
                record["f1_threshold"] = threshold
    report["summary"] = summary
    report.setdefault("protocol", {})["f1_threshold"] = threshold
    atomic_json(Path(path), report)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


# ---------------------------------------------------------------------------
# CPU-only self-test
# ---------------------------------------------------------------------------
def _stub_reader(hidden=8, n_heads=2, n_kv=1, head_dim=4, layers=4):
    """A reader-shaped module: the two attributes `readout_distance` touches."""
    from types import SimpleNamespace
    import torch.nn as nn

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(hidden, n_kv * head_dim, bias=False)
            self.v_proj = nn.Linear(hidden, n_kv * head_dim, bias=False)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = nn.LayerNorm(hidden)
            self.self_attn = Attention()

    class Body(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block() for _ in range(layers)])

    class Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_attention_heads=n_heads,
                                          num_key_value_heads=n_kv,
                                          hidden_size=hidden, head_dim=head_dim)
            self.model = Body()

    return Stub()


def _self_test():
    """Drive the real operator and the real fusion at tiny geometry.

    Every claim this experiment makes about "only the core moved" is a property
    of `fuse_output`, so it is checked here against the real V10 operator rather
    than asserted in a docstring.
    """
    from methods.prlmc.manifold_graft import ManifoldGraftedSlotMerge

    torch.manual_seed(0)
    layers, hidden, block, blocks = 2, 8, 16, 5
    core, graft = 12, 4
    carriers = [0, 4]
    window = torch.randn(layers, blocks * block, hidden)
    merge = ManifoldGraftedSlotMerge(num_layers=layers, hidden_size=hidden,
                                     num_groups=1, reference_layer=0,
                                     block_size=block, core_slots_per_block=core,
                                     graft_slots_per_block=graft)
    merge.operator_mode = "native"
    output, _, telemetry = merge(window, carriers)
    parts = split_core_and_deleted(telemetry["slot_source_positions"],
                                   telemetry["bridge_slot_offsets"], carriers, block,
                                   core)
    if len(parts["core_positions"]) != len(carriers) * core:
        raise ValueError("the split lost core slots")
    if len(parts["deleted_positions"]) != len(carriers) * (block - core):
        raise ValueError("the split lost deleted slots")
    if set(parts["deleted_positions"]) & set(telemetry["slot_source_positions"]):
        raise ValueError("a deleted slot is still in the output")

    # A zero output projection must reproduce the unfused window EXACTLY.
    fusion = LearnedCoreFusion(hidden_size=hidden, rank=4)
    identity, _ = fuse_output(output, window, parts, fusion)
    if not torch.equal(identity, output):
        raise ValueError("the zero-initialised fusion is not the identity")

    with torch.no_grad():
        fusion.out_proj.normal_(0.0, 0.5)
    fused, info = fuse_output(output, window, parts, fusion)
    graft_offsets = torch.as_tensor(telemetry["bridge_slot_offsets"], dtype=torch.long)
    if not torch.equal(fused.index_select(1, graft_offsets),
                       output.index_select(1, graft_offsets)):
        raise ValueError("the fusion changed a graft slot")
    if torch.equal(fused.index_select(1, parts["core_offsets"]),
                   output.index_select(1, parts["core_offsets"])):
        raise ValueError("the fusion changed no core slot")
    base_rms = output.float().square().mean(-1).sqrt()
    fused_rms = fused.float().square().mean(-1).sqrt()
    if not torch.allclose(base_rms, fused_rms, rtol=2e-3, atol=2e-3):
        raise ValueError("the fusion did not preserve the per-slot RMS")
    if not 0.0 < info["relative_delta"] < 1.0:
        raise ValueError(f"implausible residual size {info['relative_delta']}")

    # A core slot with no doomed slot of its own block must fail, not silently
    # attend across blocks.
    broken = dict(parts)
    broken["core_block_ids"] = torch.full_like(parts["core_block_ids"], 99)
    try:
        fuse_output(output, window, broken, fusion)
    except ValueError:
        pass
    else:
        raise ValueError("the fusion attended across carrier blocks")

    # The readout distance must be the sealed objective's, not a lookalike.
    import tools.objective as objective
    stub = _stub_reader()
    taps = (0, 2)
    pre = {layer: torch.randn(20, 8) for layer in taps}
    post = {layer: torch.randn(12, 8) for layer in taps}
    query = {layer: torch.randn(1, 6, 8) for layer in taps}
    with torch.no_grad():
        mine = float(readout_distance(stub, pre, post, query, taps))
    saved_taps = objective.DISTILL_LAYERS
    objective.DISTILL_LAYERS = taps
    try:
        with torch.no_grad():
            theirs = float(objective.compute_local_window_readout_loss(
                stub, pre, post, query, slot_stride=1))
    finally:
        objective.DISTILL_LAYERS = saved_taps
    if abs(mine - theirs) > 1e-6:
        raise ValueError(f"readout_distance {mine} != the sealed objective {theirs}")
    with torch.no_grad():
        identical = float(readout_distance(stub, pre, pre, query, taps))
    if abs(identical) > 1e-6:
        raise ValueError(f"a window's distance to itself is {identical}, expected 0")

    if len(set(OUTCOMES.values())) != 8:
        raise ValueError("the outcome vocabulary must name all eight patterns")
    for key, name in OUTCOMES.items():
        if _outcome(*key) != name:
            raise ValueError(f"_outcome{key} = {_outcome(*key)}, expected {name}")
        documented = OUTCOME_VOCABULARY[name]["pattern"]
        derived = "A=%d G=%d F=%d" % tuple(int(flag) for flag in key)
        if documented != derived:
            raise ValueError(f"{name}: documented {documented!r}, derived {derived!r}")
    if abs(er.mcnemar_exact(11, 5) - 0.2101) > 0.0005:
        raise ValueError(f"McNemar(11, 5) = {er.mcnemar_exact(11, 5)}, expected 0.2101")
    if abs(er.mcnemar_exact(9, 5) - 0.4240) > 0.0005:
        raise ValueError(f"McNemar(9, 5) = {er.mcnemar_exact(9, 5)}, expected 0.4240")

    return dict(self_test="ok", block=block, core=core, graft=graft,
                core_slots=len(parts["core_positions"]),
                deleted_slots=len(parts["deleted_positions"]),
                identity_at_init=True, graft_bit_identical=True,
                rms_preserved=True, cross_block_rejected=True,
                readout_matches_objective=True,
                relative_delta=round(info["relative_delta"], 5),
                mcnemar_11_5=round(er.mcnemar_exact(11, 5), 6))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--task", default="hotpotqa", choices=("hotpotqa",))
    p.add_argument("--supporting-facts-glob", dest="supporting_facts_glob", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--checkpoint", default=None,
                   help="the frozen V10 operator, e.g. runs/natural_merge_v9/full_v10.pt")
    p.add_argument("--fusion", default=None,
                   help="the trained fusion module, e.g. runs/core_fusion/fusion_rank32.pt")
    p.add_argument("--out", default=None)
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--window-blocks", dest="window_blocks", type=int, default=WINDOW_BLOCKS)
    p.add_argument("--target-blocks", dest="target_blocks", type=int, default=TARGET_BLOCKS)
    p.add_argument("--round2-blocks", dest="round2_blocks", type=int, default=10,
                   help="fresh native blocks appended for the second compression round")
    p.add_argument("--fusion-rank", dest="fusion_rank", type=int, default=32,
                   help="dry-run only: the rank the checkpoint is expected to carry")
    p.add_argument("--fusion-max-ratio", dest="fusion_max_ratio", type=float, default=0.5,
                   help="dry-run only: the residual bound the checkpoint is expected "
                        "to carry")
    p.add_argument("--f1-threshold", dest="f1_threshold", type=float, default=None)
    p.add_argument("--sample-ids", dest="sample_ids", default=None)
    p.add_argument("--max-samples", dest="max_samples", type=int, default=None)
    p.add_argument("--verify-against", dest="verify_against", default=None,
                   help="a finished experiment-1/1A report; Full, Selection and Graft "
                        "must reproduce it row for row")
    p.add_argument("--per-layer", dest="per_layer", action="store_true",
                   help="report the readout distance at all 32 reader layers, not only "
                        "at the eight probe layers (used to show which layers a "
                        "checkpoint moved and which it destroyed)")
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
