#!/usr/bin/env python3
"""Round 3, experiment 2: with the anchor calibration frozen, can the candidates add?

Experiment 1 left one mechanism standing: the 360 anchors' own representation
adjustment (56/158), with the 2,200 candidate slots deleted and their replacement
by another document's slots changing no answer. This driver asks the remaining
question -- not "does the candidate PATH matter" but "does the candidate CONTENT
carry knowledge the calibrated anchors do not have":

    Full               19 blocks, uncompressed                    reference ceiling
    Selection          9 carrier blocks, verbatim                 capacity baseline
    Graft              V10, 360 native anchors                    the arm to beat
    AnchorOnly         + the frozen AnchorResidual                the baseline now
    AnchorFusion       + a trainable sparse top-4 branch on the block's leftovers
    AnchorFusionDonor  + the same branch on ANOTHER document's leftovers

`AnchorFusion` is trained by the same script, on the same data, for the same
number of updates, with the same loss, and its branch has the same 524,289
parameters as every other module in this project. The only thing that changes
between it and `AnchorOnly` is whether the branch may read a candidate; the only
thing that changes between it and `AnchorFusionDonor` is WHOSE candidates it reads.

What would count as success
---------------------------
Both conditions, not either:

1. `AnchorFusion` recovers knowledge `AnchorOnly` does not have: a net gain over
   the 56/158 baseline, concentrated on the rows whose evidence the compression
   dropped (the rows where the missing slots could matter at all).
2. `AnchorFusionDonor` does not: the same branch, the same residual machinery,
   the same update count, reading slots that belong to an unrelated document,
   must lose that gain.

Condition 1 alone is not enough -- a branch that moves the window more, or that
produces a larger residual, is not evidence of knowledge. Condition 2 alone is not
enough either -- a donor arm that loses accuracy is consistent with the branch
being harmful in general. What the pair rules out is the explanation that the gain
is a generic perturbation of the anchors rather than content.

The fusion mode
---------------
The branch writes its residual one of two ways, and the checkpoint records which:

    plain      `apply_delta(base, delta)` -- the residual moves all 4,096 dimensions
               the reader uses for both its keys and its values (experiment 3)
    key-safe   the residual is confined to the null space of the layer's key map
               and the slot's L2 norm is held, so the reader's keys stay put while
               its values move (`methods/prlmc/key_safe_projection.py`)

`--fusion-mode` overrides the checkpoint's mode for one legitimate case -- projecting
an already-trained plain checkpoint without retraining -- and the override is
recorded, never inferred.

The integrity checks
--------------------
* the frozen stage inside the fused path must be the AnchorOnly window, bit for
  bit, on every row (not "the same module": the same tensor);
* the frozen anchor weights in the fusion checkpoint must equal the ones in
  `--anchor`, which must equal the ones the training run froze;
* the four old arms must reproduce the finished anchor-only report row for row;
* the donor of a row is never the row itself, and the donor arm's window must
  actually differ from the fusion arm's.

Usage
-----
    python tools/eval_anchor_fusion.py --self-test        # no GPU

    python tools/eval_anchor_fusion.py \
        --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
        --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
        --model-path /dev/shm/s1d_reader_fp16 \
        --checkpoint runs/natural_merge_v9/full_v10.pt \
        --anchor runs/anchor_residual/anchor_only_rank64.pt \
        --fusion runs/anchor_fusion/fusion_top4_rank32.pt \
        --verify-against reports/anchor_only/eval_anchor_only_hotpotqa.json \
        --cuda-device 0 \
        --out reports/anchor_fusion/eval_anchor_fusion_hotpotqa.json
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_counterfactual_candidates as cf               # noqa: E402
import eval_graft_recovery as gr                          # noqa: E402
import eval_learned_fusion as ef                          # noqa: E402
import eval_replace_reorder as er                         # noqa: E402
import eval_source_compaction as es                       # noqa: E402
import native_eval as nv                                  # noqa: E402
from io_utils import atomic_json, sha                     # noqa: E402
from runtime import load_runtime, reset, state_digest     # noqa: E402
from methods.prlmc.core_fusion import prompt_hidden_states  # noqa: E402
from methods.prlmc.anchor_residual import (               # noqa: E402
    SCHEMA as ANCHOR_SCHEMA, AnchorResidual,
    compact_window as compact_window_anchor)
from methods.prlmc.anchor_fusion import (                 # noqa: E402
    SCHEMA as FUSION_SCHEMA, FrozenAnchorFusion, SparseCandidateFusion,
    compact_window as compact_window_fusion, frozen_stage)
import methods.prlmc.key_safe_projection as ks              # noqa: E402
from methods.prlmc.source_compaction import split_source_anchors  # noqa: E402

BLOCK_TOKENS = gr.BLOCK_TOKENS
TARGET_SLOTS = gr.TARGET_SLOTS
FULL, SELECTION, GRAFT = ef.FULL, ef.SELECTION, ef.GRAFT
ANCHOR_ONLY = "AnchorOnly"
ANCHOR_FUSION = "AnchorFusion"
DONOR = "AnchorFusionDonor"
ARMS = (FULL, SELECTION, GRAFT, ANCHOR_ONLY, ANCHOR_FUSION, DONOR)
SCHEMA_RESULTS = "anchor_fusion_results/1"


# ---------------------------------------------------------------------------
# window preparation
# ---------------------------------------------------------------------------
def compress_window(manager, merge, anchor_module, fusion_module, args):
    """One write, one compression, and the three windows built from it."""
    source, ages, picked = gr.select_carriers(manager, args.window_blocks,
                                              args.target_blocks)
    frozen, pinned = gr.window_locks(manager, args.window_blocks)
    with torch.no_grad():
        output, telemetry = gr.compress_graft(manager, merge, source, picked,
                                              args.window_blocks, args.target_blocks,
                                              frozen, pinned)
        parts = split_source_anchors(telemetry, manager.block, args.window_blocks,
                                     device=source.device)
        anchor, anchor_info = compact_window_anchor(output, parts, anchor_module)
        fused, fusion_info = compact_window_fusion(output, source, parts, fusion_module)
    return dict(source=source, ages=ages, picked=picked, output=output,
                fused=fused, parts=parts, telemetry=telemetry,
                anchor_window=anchor, anchor_info=anchor_info,
                fusion_info=fusion_info)


def key_preservation(model, round1, module):
    """Measured once per run, on the real RMSNorm and the real k_proj / v_proj.

    `plain` is the same branch with the projection switched off -- the operator
    experiment 3 evaluated -- so the report carries the ratio between the two key
    perturbations rather than an absolute number with no reference point.
    """
    offsets = round1["parts"]["anchor_offsets"]
    base = round1["anchor_window"].index_select(1, offsets)
    fused = round1["fused"].index_select(1, offsets)
    loose_module = FrozenAnchorFusion(module.anchor, module.fusion, key_bases=None,
                                      key_safe=False)
    with torch.no_grad():
        loose_window, _ = compact_window_fusion(round1["output"], round1["source"],
                                                round1["parts"], loose_module)
    loose = loose_window.index_select(1, offsets)
    safe = ks.measured_key_error(model, base, fused)
    plain = ks.measured_key_error(model, base, loose)
    values = ks.measured_value_change(model, base, fused)
    return dict(
        definition=("relative change of the reader's keys (RMSNorm then k_proj, the "
                    "real modules) and of its values (v_proj) when the branch writes "
                    "into the 360 graft slots"),
        key_safe=safe, plain=plain, value=values,
        key_error_ratio=float(plain["stored"] / max(safe["stored"], 1e-12)))


def check_fusion(result):
    """Graft slots only, core untouched, and the frozen stage is the AnchorOnly arm."""
    check = es.check_compaction(dict(result, fused=result["fused"],
                                     compaction_info=result["fusion_info"]))
    if result["fusion_info"].get("fusion_mode") == "key-safe":
        # Both halves of the constraint, on every row: the arithmetic must leave the
        # merged slot exactly on the null-space shell, whatever fp16 does afterwards.
        leak = float(result["fusion_info"].get("row_leak", 0.0))
        if leak > 1e-4:
            raise ValueError(f"the key-safe merge leaked {leak:.2e} of the change into "
                             f"the key's row space; the null space is wrong")
        preserved = float(result["fusion_info"].get("norm_preserved", 0.0))
        if abs(preserved - 1.0) > 1e-5:
            raise ValueError(f"the key-safe merge changed the slot norm by "
                             f"{preserved - 1.0:.2e}; the RMSNorm denominator moved")
    offsets = result["parts"]["anchor_offsets"]
    # The baseline under the new branch must be the AnchorOnly window itself, not a
    # re-implementation of it: compared as tensors, on every row.
    if not torch.equal(frozen_stage(result["output"], result["parts"],
                                    result["module"]),
                       result["anchor_window"].index_select(1, offsets)):
        raise ValueError("the frozen stage inside the fused path is not the AnchorOnly "
                         "window: the branch is not being added to the baseline it "
                         "claims")
    if torch.equal(result["fused"].index_select(1, offsets),
                   result["anchor_window"].index_select(1, offsets)):
        raise ValueError("the fusion branch changed nothing: this row would compare "
                         "AnchorOnly against itself")
    if abs(float(result["fusion_info"]["rms_preserved"]) - 1.0) > 1e-3:
        raise ValueError("the fusion branch changed the window's RMS: the slots are no "
                         "longer in the norm range the frozen reader expects")
    return check


def donor_window(module, round1, donor, seed, sample_id):
    """The fused window when the branch reads another document's leftovers."""
    if donor["id"] == sample_id:
        raise ValueError("the donor is the sample itself: this row would measure "
                         "nothing")
    parts = round1["parts"]
    mapping = cf.donor_map(parts, donor, seed)
    donor_values = donor["slots"].index_select(1, mapping.to(donor["slots"].device))
    real_values = round1["source"].index_select(1, parts["candidate_positions"])
    if torch.equal(donor_values, real_values):
        raise ValueError("the donor supplied the sample's own candidate slots")
    hybrid = round1["source"].index_copy(1, parts["candidate_positions"], donor_values)
    with torch.no_grad():
        window, info = compact_window_fusion(round1["output"], hybrid, parts, module)
    ratio = float(donor_values.float().square().mean().sqrt()
                  / real_values.float().square().mean().sqrt().clamp_min(1e-12))
    return window, dict(info, donor_id=donor["id"], candidate_rms_ratio=ratio,
                        mapping_sha256=hashlib.sha256(
                            mapping.cpu().numpy().tobytes()).hexdigest())


def check_donor(round1, window, info, sample_id):
    """Graft slots only, and they must differ from the fusion arm's."""
    check = es.check_compaction(dict(round1, fused=window,
                                     compaction_info=info))
    offsets = round1["parts"]["anchor_offsets"]
    if torch.equal(window.index_select(1, offsets),
                   round1["fused"].index_select(1, offsets)):
        raise ValueError("the donor arm reproduced the fusion window exactly: the "
                         "candidate content never reached the branch")
    if info["donor_id"] == sample_id:
        raise ValueError("the donor is the sample itself")
    return check


def build_donor(manager, merge, prepared, args):
    """One document's window, compressed once; only its candidate slots are kept."""
    chunks = [prepared["write_ids"][index:index + BLOCK_TOKENS]
              for index in range(0, len(prepared["write_ids"]), BLOCK_TOKENS)]
    with torch.no_grad():
        es.write_blocks(manager, merge, chunks[:args.window_blocks])
        source, _, picked = gr.select_carriers(manager, args.window_blocks,
                                               args.target_blocks)
        frozen, pinned = gr.window_locks(manager, args.window_blocks)
        _, telemetry = gr.compress_graft(manager, merge, source, picked,
                                         args.window_blocks, args.target_blocks,
                                         frozen, pinned)
        parts = split_source_anchors(telemetry, manager.block, args.window_blocks,
                                     device=source.device)
        slots = source.index_select(1, parts["candidate_positions"])
    return dict(id=prepared["id"], slots=slots, blocks=parts["candidate_blocks"],
                dropped_blocks=parts["dropped_blocks"])


# ---------------------------------------------------------------------------
# one sample
# ---------------------------------------------------------------------------
def run_sample(model, tokenizer, manager, merge, anchor_module, fusion_module,
               prepared, donor, args, position):
    """Write the window once, compress once, then answer all six arms."""
    chunks = [prepared["write_ids"][index:index + BLOCK_TOKENS]
              for index in range(0, len(prepared["write_ids"]), BLOCK_TOKENS)]
    answers = prepared["answers"]
    fingerprints, gaps = {}, {}
    with torch.no_grad():
        es.write_blocks(manager, merge, chunks[:args.window_blocks])
        state = gr.snapshot(model, manager)
        query_hidden = prompt_hidden_states(model, prepared["query_ids"], ef.TAP_LAYERS)
        round1 = compress_window(manager, merge, anchor_module, fusion_module, args)
        round1["module"] = fusion_module
        donor_arm, donor_info = donor_window(fusion_module, round1, donor,
                                             args.seed * 100003 + position,
                                             prepared["id"])
    # `check_compaction` reads `compaction_info`; the fused window's info is what this
    # run has, and it is the only compacted window in the dict it is given.
    check = es.check_compaction(dict(round1, compaction_info=round1["fusion_info"]))
    fusion_check = check_fusion(round1)
    donor_check = check_donor(round1, donor_arm, donor_info, prepared["id"])
    # Once per run, on the first row: this is the check the whole round exists for,
    # and it needs the reader's own modules rather than a re-derivation.
    preservation = (key_preservation(model, round1, fusion_module)
                    if position == 0 and fusion_module.fusion_mode == "key-safe"
                    else None)
    # Keyed by arm because the shared analysis reads `structure[Graft]`; the same
    # check covers every compacted arm -- each changes the graft slots and leaves
    # the core alone.
    structure = {SELECTION: None, GRAFT: check, ANCHOR_ONLY: check,
                 ANCHOR_FUSION: fusion_check, DONOR: donor_check}

    selection = ef.selection_window(round1["source"], round1["picked"])
    gaps[SELECTION] = es.readout_gap(model, round1["source"], selection,
                                     query_hidden, ef.TAPS)
    for arm, window in ((GRAFT, round1["output"]), (ANCHOR_ONLY, round1["anchor_window"]),
                        (ANCHOR_FUSION, round1["fused"]), (DONOR, donor_arm)):
        gaps[arm] = es.readout_gap(model, round1["source"], window, query_hidden, ef.TAPS)
        gaps[arm + "/heldout"] = es.readout_gap(model, round1["source"], window,
                                                query_hidden, ef.HELD_OUT_LAYERS)
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
            elif arm == ANCHOR_ONLY:
                er.install_output(manager, round1["anchor_window"], round1["picked"],
                                  args.window_blocks)
            elif arm == ANCHOR_FUSION:
                er.install_output(manager, round1["fused"], round1["picked"],
                                  args.window_blocks)
            elif arm == DONOR:
                er.install_output(manager, donor_arm, round1["picked"],
                                  args.window_blocks)
            fingerprints[arm] = gr.active_fingerprint(manager)
            fingerprints[arm + "/slots"] = int(manager.used)
        return run

    arms = {arm: gr.readout(model, tokenizer, prepared["query_ids"], answers,
                            args.f1_threshold, prepare(arm))
            for arm in ARMS}
    gr.restore(model, manager, state)

    if int(fingerprints[FULL + "/slots"]) != args.window_blocks * manager.block:
        raise ValueError("the Full arm did not keep the whole window")
    for arm in (SELECTION, GRAFT, ANCHOR_ONLY, ANCHOR_FUSION, DONOR):
        if int(fingerprints[arm + "/slots"]) != TARGET_SLOTS:
            raise ValueError(f"arm {arm} does not hold the {TARGET_SLOTS}-slot budget")
    for left, right in ((SELECTION, GRAFT), (GRAFT, ANCHOR_ONLY),
                        (ANCHOR_ONLY, ANCHOR_FUSION), (ANCHOR_FUSION, DONOR)):
        if fingerprints[left] == fingerprints[right]:
            raise ValueError(f"{left} and {right} read the same pool: the manipulation "
                             f"did not reach the memory")

    provenance = gr.graft_provenance(round1["telemetry"], args.window_blocks)
    if provenance["selected_blocks"] != sorted(round1["picked"]):
        raise ValueError("graft provenance disagrees with the shared carrier set")
    record = dict(arms=arms, fingerprints=fingerprints, gaps=gaps,
                  provenance=provenance, structure=structure, multi_round=False,
                  anchor=dict(**round1["anchor_info"], check=check),
                  fusion=dict(**round1["fusion_info"], check=fusion_check),
                  donor=dict(**donor_info, check=donor_check),
                  key_preservation=preservation,
                  carrier_ages=[int(round1["ages"][block]) for block in round1["picked"]])
    next_donor = dict(id=prepared["id"],
                      slots=round1["source"].index_select(
                          1, round1["parts"]["candidate_positions"]),
                      blocks=round1["parts"]["candidate_blocks"],
                      dropped_blocks=round1["parts"]["dropped_blocks"])
    return record, next_donor


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def _stats(values):
    values = [float(value) for value in values]
    if not values:
        return dict(n=0)
    return dict(n=len(values), mean=round(sum(values) / len(values), 6),
                median=round(sorted(values)[len(values) // 2], 6))


def fusion_summary(samples, threshold):
    """The block the driver exists for: does the branch add knowledge, and is it content?"""
    graded = [sample for sample in samples if sample.get("status") == "graded"]
    located = [sample for sample in graded if sample.get("evidence", {}).get("located")]
    dropped = [sample for sample in located if ef._evidence_in_dropped(sample)]
    kept = [sample for sample in located if not ef._evidence_in_dropped(sample)]

    def comparisons(rows):
        return {
            f"{ANCHOR_FUSION}_minus_{ANCHOR_ONLY}":
                er.pair_summary(rows, ANCHOR_FUSION, ANCHOR_ONLY, threshold),
            f"{ANCHOR_FUSION}_minus_{DONOR}":
                er.pair_summary(rows, ANCHOR_FUSION, DONOR, threshold),
            f"{ANCHOR_FUSION}_minus_{GRAFT}":
                er.pair_summary(rows, ANCHOR_FUSION, GRAFT, threshold),
            f"{ANCHOR_ONLY}_minus_{GRAFT}":
                er.pair_summary(rows, ANCHOR_ONLY, GRAFT, threshold),
            f"{DONOR}_minus_{ANCHOR_ONLY}":
                er.pair_summary(rows, DONOR, ANCHOR_ONLY, threshold),
        }

    def group(rows):
        return dict(n=len(rows),
                    arms={arm: gr._arm_summary(rows, arm, threshold) for arm in ARMS},
                    pairs=comparisons(rows))

    dropped_group = group(dropped)
    # The two conditions of the driver, read off the rows where a missing slot could
    # possibly matter: net recovery over AnchorOnly, and no such recovery from slots
    # that belong to another document.
    recovery = dict(
        definition=("on the rows whose evidence the compression dropped, the net number "
                    "of questions the fusion branch turns from wrong to right, against "
                    "the frozen AnchorOnly baseline and against the donor arm"),
        n=dropped_group["n"],
        versus_anchor_only=dropped_group["pairs"][f"{ANCHOR_FUSION}_minus_{ANCHOR_ONLY}"],
        versus_donor=dropped_group["pairs"][f"{ANCHOR_FUSION}_minus_{DONOR}"])

    return dict(
        definition=("the frozen AnchorOnly baseline plus a trainable sparse top-4 branch "
                    "on the same block's leftovers, against the same branch reading "
                    "another document's leftovers"),
        n=len(graded),
        arms={arm: gr._arm_summary(graded, arm, threshold) for arm in ARMS},
        pairs=comparisons(graded),
        knowledge_recovery=recovery,
        dropped_evidence_group=dropped_group,
        kept_evidence_group=group(kept),
        damage={arm: ef._damage(graded, arm, threshold) for arm in ARMS},
        residual=dict(
            definition="RMS of the injected residual over RMS of the slot it is added to",
            fusion=_stats([sample["fusion"]["relative_delta"] for sample in graded]),
            donor=_stats([sample["donor"]["relative_delta"] for sample in graded]),
            frozen_stage=_stats([sample["anchor"]["relative_delta"]
                                 for sample in graded]),
            anchor_relative=_stats([sample["fusion"]["anchor_relative_delta"]
                                    for sample in graded])),
        selection=dict(
            definition="how the top-4 mix, over all 360 anchors and all 32 layers",
            max_weight=_stats([sample["fusion"]["max_weight"] for sample in graded]),
            entropy=_stats([sample["fusion"]["entropy"] for sample in graded]),
            novelty_ratio=_stats([sample["fusion"]["novelty_ratio"]
                                  for sample in graded]),
            top_k=sorted({int(sample["fusion"]["top_k"]) for sample in graded})),
        key_safe=dict(
            definition=("what the key-safe constraint cost and what it kept: the share "
                        "of the branch's residual the projection removed, how far the "
                        "null component moved, and whether the slot norm held"),
            row_space_fraction=_stats([sample["fusion"]["row_space_fraction"]
                                       for sample in graded
                                       if "row_space_fraction" in sample["fusion"]]),
            null_shift=_stats([sample["fusion"]["null_shift"] for sample in graded
                               if "null_shift" in sample["fusion"]]),
            norm_preserved=_stats([sample["fusion"]["norm_preserved"]
                                   for sample in graded
                                   if "norm_preserved" in sample["fusion"]]),
            modes=sorted({sample["fusion"].get("fusion_mode", "plain")
                          for sample in graded})),
        readout=dict(
            definition="mean reader-readout distance from the uncompressed window",
            by_arm={arm: ef._gap_summary(graded, arm)
                    for arm in (GRAFT, ANCHOR_ONLY, ANCHOR_FUSION, DONOR)},
            held_out={arm: ef._gap_summary(graded, arm + "/heldout")
                      for arm in (GRAFT, ANCHOR_ONLY, ANCHOR_FUSION, DONOR)}))


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------
def self_test():
    """The structural promises of the fused path, on a synthetic window, on the CPU.

    No reader and no checkpoint: which slots move, which must not, whether the fused
    path's frozen stage really is the AnchorOnly window, whether a zero output
    projection is the identity there, and whether the branch can see a candidate at
    all -- the parts that can be wrong while every GPU number still looks fine.
    """
    block, core, graft_per_block = 16, 12, 4
    window_blocks, target_blocks = 5, 3
    selected, dropped = [0, 2, 4], [1, 3]
    layers, hidden = 2, 8
    torch.manual_seed(0)
    window = torch.randn(layers, window_blocks * block, hidden)
    core_positions = [carrier * block + offset
                      for carrier in selected for offset in range(core)]
    offsets, full_positions = [], [None] * (target_blocks * block)
    for slot in range(target_blocks):
        for offset in range(core):
            full_positions[slot * block + offset] = core_positions[slot * core + offset]
        for offset in range(graft_per_block):
            source_block = dropped[(slot + offset) % len(dropped)]
            offsets.append(slot * block + core + offset)
            full_positions[slot * block + core + offset] = (
                source_block * block + 6 + offset)
    telemetry = dict(bridge_slot_offsets=offsets,
                     slot_source_positions=full_positions,
                     selected_blocks=selected)
    parts = split_source_anchors(telemetry, block, window_blocks)
    output = torch.randn(layers, target_blocks * block, hidden)

    anchor_module = AnchorResidual(hidden_size=hidden, rank=4)
    with torch.no_grad():
        anchor_module.up.normal_(std=0.05)
    anchor_window, anchor_info = compact_window_anchor(output, parts, anchor_module)

    fusion = SparseCandidateFusion(hidden_size=hidden, rank=2, top_k=4)
    module = FrozenAnchorFusion(anchor_module, fusion).eval()
    if any(parameter.requires_grad for parameter in module.anchor.parameters()):
        raise ValueError("the anchor module is still trainable after wrapping")

    fused, info = compact_window_fusion(output, window, parts, module)
    if not torch.equal(fused, anchor_window):
        raise ValueError("a zero output projection did not reproduce the AnchorOnly "
                         "window")
    if not torch.equal(frozen_stage(output, parts, module),
                       anchor_window.index_select(1, parts["anchor_offsets"])):
        raise ValueError("the frozen stage is not the AnchorOnly window")
    if int(info["candidate_slots"]) != int(parts["candidate_positions"].numel()):
        raise ValueError("the fusion module reports the wrong candidate count")
    if int(info["top_k"]) != 4:
        raise ValueError("the fusion module reports the wrong top_k")

    with torch.no_grad():
        fusion.fuse_up.normal_(std=0.05)
    fused, info = compact_window_fusion(output, window, parts, module)
    keep = torch.ones(output.shape[1], dtype=torch.bool)
    keep[parts["anchor_offsets"]] = False
    if not torch.equal(output[:, keep], fused[:, keep]):
        raise ValueError("the fusion branch changed a core slot")
    if torch.equal(anchor_window.index_select(1, parts["anchor_offsets"]),
                   fused.index_select(1, parts["anchor_offsets"])):
        raise ValueError("the fusion branch changed nothing")
    if not torch.isfinite(fused).all():
        raise ValueError("the fused window is not finite")
    if abs(float(info["rms_preserved"]) - 1.0) > 1e-3:
        raise ValueError("the fused window does not preserve the slot RMS")

    # The residual is bounded by a vector norm: every coordinate of a slot shares
    # one scale, so the residual's direction is exactly the raw direction.
    anchors = anchor_window.index_select(1, parts["anchor_offsets"])
    base = frozen_stage(output, parts, module)
    delta, stats = fusion(base, window.index_select(1, parts["candidate_positions"]),
                          parts["same_block"])
    bound = (torch.sigmoid(fusion.gate_logit.float()) * fusion.max_ratio
             * base.float().square().mean(dim=-1, keepdim=True).sqrt())
    measured = delta.float().square().mean(dim=-1, keepdim=True).sqrt()
    if bool((measured > bound + 1e-6).any()):
        raise ValueError("the residual exceeded its vector-norm bound")

    # A donor candidate set must move the window, and must not be the same window.
    donor_window_values = window.index_select(1, parts["candidate_positions"]).roll(1, 0)
    donor_hybrid = window.index_copy(1, parts["candidate_positions"], donor_window_values)
    with torch.no_grad():
        donor_fused, donor_info = compact_window_fusion(output, donor_hybrid, parts,
                                                        module)
    if torch.equal(donor_fused, fused):
        raise ValueError("replacing every candidate left the fused window unchanged")
    if int(donor_info["candidate_slots"]) != int(info["candidate_slots"]):
        raise ValueError("the donor arm changed the candidate count")

    print(json.dumps(dict(
        self_test="ok", anchors=int(parts["anchor_offsets"].numel()),
        candidate_slots=int(info["candidate_slots"]), top_k=int(info["top_k"]),
        identity_at_init=True, frozen_stage_is_anchor_only=True,
        core_bit_identical=True, graft_changed=True,
        rms_preserved=round(float(info["rms_preserved"]), 6),
        max_weight=round(float(stats["max_weight"]), 4),
        entropy=round(float(stats["entropy"]), 4),
        novelty_ratio=round(float(stats["novelty_ratio"]), 4),
        residual_within_bound=True, donor_moves_window=True), indent=2))
    return 0


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def build_protocol(args, task, anchor_config, fusion_config):
    """The protocol block, in the same shape as experiment 2's, minus the compaction."""
    return dict(
        experiment="3-2: with the anchor calibration frozen, do the candidates add?",
        treatment=("the 360 anchors are calibrated by the FROZEN AnchorResidual and then "
                   "adjusted by a trainable branch that reads the top-4 leftovers of "
                   "their own source block at native width"),
        task=task, data_dir=args.data_dir, arms=list(ARMS),
        window_blocks=args.window_blocks, target_blocks=args.target_blocks,
        block_tokens=BLOCK_TOKENS, slots_per_block=es.BLOCK,
        full_slots=args.window_blocks * es.BLOCK, target_slots=TARGET_SLOTS,
        core_slots_total=es.CORE_SLOTS_TOTAL, graft_slots_total=es.GRAFT_SLOTS_TOTAL,
        dropped_blocks=args.window_blocks - args.target_blocks,
        dropped_slots=(args.window_blocks - args.target_blocks) * es.BLOCK,
        compressions_per_sample=1,
        unchanged_conditions=dict(
            data="the same 158 HotpotQA rows as every earlier experiment",
            reader="frozen MemoryLLM-8B", carriers="the same 9, unchanged",
            core_selection="original single-point top-216 per carrier",
            source_selection="V10's trained single-point leverage rule, unchanged",
            source_capacity=es.GRAFT_SLOTS_TOTAL, output_capacity=TARGET_SLOTS,
            output_order="the existing global chronological order",
            core_vectors="native copies on both sides, byte-identical"),
        only_difference=("the new branch reads the top-4 leftovers of each anchor's own "
                         "source block and adds a bounded residual to the calibrated "
                         "anchor; the frozen stage is byte-identical to AnchorOnly"),
        compaction=None,
        anchor_only=dict(
            schema=ANCHOR_SCHEMA, **anchor_config, checkpoint=args.anchor,
            checkpoint_sha256=(sha(args.anchor) if args.anchor
                               and Path(args.anchor).is_file() else None),
            input="the 360 native anchors and nothing else",
            role="the frozen baseline the new branch is added to, and the arm to beat"),
        anchor_fusion=dict(
            schema=FUSION_SCHEMA, **fusion_config, checkpoint=args.fusion,
            checkpoint_sha256=(sha(args.fusion) if args.fusion
                               and Path(args.fusion).is_file() else None),
            frozen="the AnchorResidual from --anchor, digest-checked before and after "
                   "training and again when this report is written",
            trainable="the sparse top-4 candidate branch, 524,289 parameters",
            fusion_mode=("read from the checkpoint config; an override is recorded "
                         "in identity.fusion_mode_override"),
            selection="top-4 of each anchor's own block's leftovers by a bilinear score",
            width="the four winners are mixed at their native 4,096 dimensions; only "
                  "the mixture is compressed",
            bound="vector norm: delta = raw * bound / (bound + rms(raw)), so the "
                  "direction of the residual is untouched and no coordinate saturates",
            training="offline, answer cross-entropy on a disjoint LongBench task, 384 "
                     "updates, nothing trained here",
            unchanged=dict(
                anchors="the 360 native anchors, byte-identical",
                training="the same script, data, updates, learning rate, loss, bounds",
                parameters="equal by construction: 4 x 32 x 4096 + 1 = 524,289",
                core_slots="byte-identical to the grafted window",
                carriers="the same 9", reader="frozen MemoryLLM-8B",
                data="the same 158 HotpotQA rows", decoding="greedy, unchanged"),
            only_difference="whether the branch may read a candidate, and whose"),
        donor=dict(
            arm=DONOR,
            rule=("the donor of a row is the previous eligible row in evaluation order "
                  "(row 0 borrows row 1's window); the k-th dropped block of the sample "
                  "maps to the k-th dropped block of the donor"),
            why=("a gain that survives the donor swap is not knowledge: it is a generic "
                 "perturbation of the anchors")),
        success=("both, not either: AnchorFusion beats AnchorOnly on the rows whose "
                 "evidence the compression dropped, and AnchorFusionDonor does not"),
        operator_mode="native",
        per_layer_probe=False,
        round2="not implemented: the single compression is run first",
        write_stream="document prefix only, 19 native blocks",
        query=gr.QUERY_SUFFIX,
        scoring="official HotpotQA QA F1, F1 >= threshold is correct, gold likelihood "
                "recorded alongside",
        f1_threshold=args.f1_threshold, seed=args.seed,
        screening_caveat=gr.SCREENING_CAVEAT,
        fusion_training=None,
        pairing_note=("all six arms run in one process on one pre-compression window; "
                      "the four old arms must reproduce the anchor-only report before "
                      "the new arms are read"))


def verify_anchor_reference(args, reference, table, samples):
    """Full, Selection, Graft AND AnchorOnly must reproduce the anchor-only report."""
    verdict = ef.verify_against(args.verify_against, reference, table, samples)
    same = different = 0
    differing = []
    for sample in samples:
        old = table.get(str(sample["id"]))
        if old is None or ANCHOR_ONLY not in old.get("arms", {}):
            continue
        if old["arms"][ANCHOR_ONLY]["prediction"] == sample["arms"][ANCHOR_ONLY]["prediction"]:
            same += 1
        else:
            different += 1
            differing.append(dict(id=sample["id"],
                                  old=old["arms"][ANCHOR_ONLY]["prediction"],
                                  new=sample["arms"][ANCHOR_ONLY]["prediction"]))
    verdict[ANCHOR_ONLY] = dict(identical=same, different=different,
                                differing=differing[:20])
    verdict["passed"] = verdict["passed"] and different == 0 and same == len(samples)
    return verdict


def load_anchor_module(path, hidden):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != ANCHOR_SCHEMA + "/checkpoint":
        raise ValueError(f"{path} is not a {ANCHOR_SCHEMA} checkpoint")
    config = dict(payload["config"])
    module = AnchorResidual(**config)
    module.load_state_dict(payload["state_dict"], strict=True)
    if int(config.get("hidden_size", -1)) != int(hidden):
        raise ValueError(f"{path} was built for hidden {config.get('hidden_size')}, "
                         f"the reader is {hidden}")
    return module.eval().requires_grad_(False), config


def load_fusion_module(path, anchor, anchor_path, hidden, reader, mode_override=None):
    """Rebuild the trained branch in the mode it was trained in.

    The mode is part of the checkpoint config, not a flag: a branch trained with the
    plain merge and evaluated with the key-safe one is a different operator, and the
    only way to tell them apart afterwards is to have recorded it. `mode_override`
    exists for the one legitimate case -- projecting an already-trained plain
    checkpoint without retraining -- and is reported as an override, never silently.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != FUSION_SCHEMA + "/checkpoint":
        raise ValueError(f"{path} is not a {FUSION_SCHEMA} checkpoint")
    config = dict(payload["config"])
    trained_mode = config.pop("fusion_mode", "plain")
    mode = trained_mode if mode_override is None else mode_override
    recorded = (payload.get("training") or {}).get("anchor_checkpoint_sha256")
    if recorded and recorded != sha(anchor_path):
        raise ValueError(f"{path} was trained on top of anchor checkpoint {recorded}, "
                         f"but --anchor is {sha(anchor_path)}: the frozen baseline in "
                         f"the checkpoint is not the one being evaluated")
    if int(config.get("hidden_size", -1)) != int(hidden):
        raise ValueError(f"{path} was built for hidden {config.get('hidden_size')}, "
                         f"the reader is {hidden}")

    key_bases = key_report = key_digest = None
    if mode == "key-safe":
        # Rebuilt from the reader being evaluated, then matched against the digest
        # the training run recorded: a basis built from different weights (or from a
        # different LoRA adapter) annihilates the wrong keys and every later number
        # would be meaningless.
        key_bases, key_report = ks.build_key_bases(reader)
        key_digest = ks.basis_digest(key_bases)
        recorded_basis = (payload.get("training") or {}).get("key_basis_sha256")
        if recorded_basis and recorded_basis != key_digest:
            raise ValueError(
                f"the key basis rebuilt from this reader is {key_digest}, the "
                f"checkpoint recorded {recorded_basis}: the reader weights or the "
                f"active LoRA adapter are not the ones the branch was trained "
                f"against")

    module = FrozenAnchorFusion(anchor, SparseCandidateFusion(**config),
                                key_bases=key_bases, key_safe=(mode == "key-safe"))
    module.load_state_dict(payload["state_dict"], strict=True)
    meta = dict(trained_mode=trained_mode, mode=mode,
                override=(mode != trained_mode), key_basis=key_report,
                key_basis_sha256=key_digest)
    return (module.to(next(iter(module.parameters())).device).eval()
            .requires_grad_(False), config, meta)


def evaluate(args):
    facts = gr.load_supporting_facts(args.supporting_facts_glob)
    if not facts:
        print("WARNING: no supporting facts; the evidence grouping will be empty.",
              file=sys.stderr)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    if args.dry_run:
        rows, prepared, excluded = gr.prepare_rows(args, tokenizer, facts)
        print(json.dumps(dict(
            dry_run=True, out=args.out,
            counts=dict(rows=len(rows), eligible=len(prepared), excluded=len(excluded)),
            donors=dict(first=prepared[1]["id"] if len(prepared) > 1 else None,
                        then="each row donates to the next"),
            readouts_per_sample=len(ARMS)), indent=2, ensure_ascii=False))
        return None
    for flag, value, what in (("--checkpoint", args.checkpoint, "the frozen operator"),
                              ("--anchor", args.anchor, "the frozen AnchorResidual"),
                              ("--fusion", args.fusion, "the trained fusion branch")):
        if not value or not Path(value).is_file():
            raise ValueError(f"this experiment needs {what}: pass {flag}")

    output = Path(args.out)
    if output.exists():
        raise ValueError(f"{output} exists; delete it first or choose another --out")

    nv.seed_everything(args.seed)
    checkpoint_training = torch.load(args.checkpoint, map_location="cpu").get("training")
    model, tokenizer, manager, merge, initial, cfg, saved = load_runtime(
        args.model_path, checkpoint=args.checkpoint, train=False, architecture=None,
        budget_blocks=gr.BUDGET_BLOCKS, strict_code_hash=args.strict_code_hash,
        training=checkpoint_training)
    merge.operator_mode = "native"

    anchor_module, anchor_config = load_anchor_module(args.anchor, manager.hidden)
    anchor_module = anchor_module.to(model.memory.device)
    device = model.memory.device
    reference_digest = state_digest(anchor_module)
    override = None if args.fusion_mode == "from-checkpoint" else args.fusion_mode
    fusion_module, fusion_config, fusion_meta = load_fusion_module(
        args.fusion, anchor_module, args.anchor, manager.hidden, model, override)
    fusion_module = fusion_module.to(device)
    # The frozen weights inside the fusion checkpoint must be the ones in --anchor:
    # otherwise the branch is being added to a baseline nobody evaluated.
    if state_digest(fusion_module.anchor) != reference_digest:
        raise ValueError("the frozen anchor weights inside the fusion checkpoint differ "
                         "from --anchor: the baseline under the branch is not the one "
                         "this report evaluates")
    # Both halves are frozen here; the split is by ROLE, not by `requires_grad` --
    # the branch was trainable in the run that produced the checkpoint and the anchor
    # never was. Sizes are equal by construction (524,289 each).
    parameters = dict(
        frozen_anchor=int(sum(p.numel() for p in fusion_module.anchor.parameters())),
        fusion_branch=int(sum(p.numel() for p in fusion_module.fusion.parameters())),
        trainable_at_inference=0)

    rows, prepared, excluded = gr.prepare_rows(args, tokenizer, facts)
    if len(prepared) < 2:
        raise ValueError("a counterfactual needs at least two rows: the donor of a row "
                         "is another document")
    report = dict(
        schema=SCHEMA_RESULTS, status="running",
        counts=dict(rows=len(rows), eligible=len(prepared), excluded=len(excluded)),
        protocol=build_protocol(args, args.task, anchor_config, fusion_config),
        identity=dict(
            model_path=args.model_path, checkpoint=args.checkpoint,
            checkpoint_sha256=sha(args.checkpoint),
            checkpoint_code_hash_verified=(saved or {}).get("code_hash_verified"),
            anchor=args.anchor, anchor_sha256=sha(args.anchor),
            anchor_state_digest=reference_digest,
            fusion=args.fusion, fusion_sha256=sha(args.fusion),
            fusion_state_digest=state_digest(fusion_module),
            fusion_mode=fusion_meta["mode"],
            fusion_mode_trained=fusion_meta["trained_mode"],
            fusion_mode_override=fusion_meta["override"],
            key_basis=fusion_meta["key_basis"],
            key_basis_sha256=fusion_meta["key_basis_sha256"],
            parameters=parameters,
            data_dir=args.data_dir,
            data_sha256=sha(Path(args.data_dir) / (args.task + ".jsonl")),
            supporting_facts_glob=args.supporting_facts_glob,
            entrypoint_sha256=sha(__file__),
            anchor_module_sha256=sha(ROOT / "methods/prlmc/anchor_residual.py"),
            fusion_module_sha256=sha(ROOT / "methods/prlmc/anchor_fusion.py"),
            reader_frozen=True),
        excluded=excluded, samples=[], training_performed=False,
        optimizer_created=False,
        full_split=(args.max_samples is None and args.sample_ids is None))
    atomic_json(output, report)

    reference, table = ((None, None) if not args.verify_against
                        else ef.load_reference(args.verify_against))
    started = time.monotonic()
    donor = None
    for position, sample in enumerate(prepared):
        sample_started = time.monotonic()
        nv.seed_everything(args.seed)
        reset(manager, initial)
        if donor is None:
            # Row 0 has no predecessor: borrow row 1's window, then rebuild row 0's.
            donor = build_donor(manager, merge, prepared[1], args)
            reset(manager, initial)
            nv.seed_everything(args.seed)
        result, donor = run_sample(model, tokenizer, manager, merge, anchor_module,
                                   fusion_module, sample, donor, args, position)
        record = dict(id=sample["id"], status="graded",
                      available_blocks=sample["available_blocks"],
                      evidence=sample["evidence"],
                      seconds=time.monotonic() - sample_started, **result)
        report["samples"].append(record)
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output, report)
        print(json.dumps(dict(
            sample=position + 1, total=len(prepared), id=sample["id"],
            donor=record["donor"]["donor_id"],
            graft=result["arms"][GRAFT]["correct"],
            anchor_only=result["arms"][ANCHOR_ONLY]["correct"],
            fusion=result["arms"][ANCHOR_FUSION]["correct"],
            donor_correct=result["arms"][DONOR]["correct"],
            delta_fusion=record["fusion"]["relative_delta"],
            delta_donor=record["donor"]["relative_delta"],
            seconds=round(record["seconds"], 1))), flush=True)

    first = report["samples"][0] if report["samples"] else {}
    if first.get("key_preservation"):
        report["identity"]["key_preservation"] = first["key_preservation"]
    report["summary"] = fusion_summary(report["samples"], args.f1_threshold)
    if args.verify_against:
        report["verification_against_reference"] = verify_anchor_reference(
            args, reference, table, report["samples"])
    report["elapsed_seconds"] = time.monotonic() - started
    report["status"] = "complete"
    report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    atomic_json(output, report)
    print(json.dumps(dict(status="complete", out=str(output),
                          elapsed_seconds=round(report["elapsed_seconds"], 1),
                          arms={arm: report["summary"]["arms"][arm].get("correct")
                                for arm in ARMS},
                          pairs={key: value for key, value in
                                 report["summary"]["pairs"].items()}),
                     indent=1, ensure_ascii=False))
    return report


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--task", default="hotpotqa", choices=("hotpotqa",))
    p.add_argument("--supporting-facts-glob", dest="supporting_facts_glob", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--checkpoint", default=None,
                   help="the frozen V10 operator (runs/natural_merge_v9/full_v10.pt)")
    p.add_argument("--anchor", default=None,
                   help="the trained AnchorResidual checkpoint (the frozen baseline)")
    p.add_argument("--fusion", default=None,
                   help="the trained sparse fusion checkpoint (the new arms)")
    p.add_argument("--fusion-mode", dest="fusion_mode", default="from-checkpoint",
                   choices=("from-checkpoint", "plain", "key-safe"),
                   help="rebuild the branch in this mode instead of the one the "
                        "checkpoint records; an override is reported in "
                        "identity.fusion_mode_override rather than applied silently")
    p.add_argument("--out", default=None)
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--window-blocks", dest="window_blocks", type=int,
                   default=gr.WINDOW_BLOCKS)
    p.add_argument("--target-blocks", dest="target_blocks", type=int,
                   default=gr.TARGET_BLOCKS)
    p.add_argument("--f1-threshold", dest="f1_threshold", type=float, default=0.5)
    p.add_argument("--sample-ids", dest="sample_ids", default=None)
    p.add_argument("--max-samples", dest="max_samples", type=int, default=None)
    p.add_argument("--verify-against", dest="verify_against", default=None,
                   help="the finished anchor-only report; Full, Selection, Graft and "
                        "AnchorOnly must reproduce it row for row")
    p.add_argument("--strict-code-hash", dest="strict_code_hash", default="true",
                   choices=("true", "false"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--self-test", dest="self_test", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    if args.dry_run and not args.data_dir:
        raise SystemExit("--data-dir is required")
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
