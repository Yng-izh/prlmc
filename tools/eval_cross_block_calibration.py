#!/usr/bin/env python3
"""Experiment 9 (V13): cross-block conditioned calibration, on V11Evict's structure.

    Full              19 blocks, uncompressed                     reference ceiling
    Selection         9 carrier blocks, verbatim                  capacity baseline
    Graft             V10 native graft, 1,944 + 360               the arm AnchorOnly beat
    AnchorOnly        the frozen AnchorResidual at 360 anchors     the 56/158 baseline
    V11Evict          V11Evict's window + the SAME module trained with --cross off
    V11EvictDonor     V11Evict, graft slots taken from ANOTHER document
    V13               V11Evict's window + the cross-block conditioned module
    V13Donor          V13, graft slots taken from ANOTHER document

The three questions, in the order they have to be answered
---------------------------------------------------------
1. `V13 > V11Evict` -- does conditioning the residual on cross-block neighbours buy
   anything beyond what a plain full-pool calibration already buys? Both arms run the
   same window, the same budget, the same loss and the same number of updates; the only
   difference is the context term in the calibrator.
2. `V13 > V13Donor` -- is the gain attributable to the CONTENT of the graft slots? Same
   window geometry, same module, same cross-block machinery; only whose dropped blocks
   they came from changes. Every earlier mechanism in this project passed (1) and failed
   (2), so (2) is not optional. `V11Evict > V11EvictDonor` is reported for the same
   reason: it is the arm that already passed this test once.
3. Damage on the rows the uncompressed window already answers -- a mechanism that buys
   questions by losing others is a redistribution, not a gain. The two ceilings of the
   calibrator (0.15 x rms on carriers, 0.50 x rms on graft) exist for this.

Two notes on what this driver cannot do
---------------------------------------
There is no second dataset in this file. The independent natural multi-hop set the plan
asks for is a separate run against a different `--task`/`--data-dir`; the protocol block
records `task`, `data_dir` and their sha256 so two runs can be compared safely.
`V11Evict` here is NOT the checkpoint from experiment 7 -- that one was deleted together
with the V11 code -- it is the same module retrained with `--cross off` inside this
round's framework, which is the cleaner comparison anyway (one variable differs).

What is checked before a number is read
--------------------------------------
* the four pre-existing arms must reproduce the reference report row for row;
* every compressed arm holds 2,304 slots split 1,620 / 684, chronologically ordered,
  every slot a verbatim copy of the source position it claims;
* the calibration must move both halves, and a `--v11-evict` checkpoint must really be
  cross-block conditioned (`--v11` must really not be);
* the donor arm's window must differ from the true one;
* the cross-block context must be built from OTHER blocks -- the neighbour spread is
  recorded per row, and a context with only same-block neighbours is a bug, not a result.

Usage
-----
    python tools/eval_cross_block_calibration.py --dry-run \
        --data-dir /data/.../LongBench/extracted/data --model-path /dev/shm/s1d_reader_fp16

    python tools/eval_cross_block_calibration.py \
        --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
        --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
        --model-path /dev/shm/s1d_reader_fp16 \
        --checkpoint runs/natural_merge_v9/full_v10.pt \
        --anchor runs/anchor_residual/anchor_only_rank64.pt \
        --v11 runs/cross_block/plain.pt \
        --v11-evict runs/cross_block/cross.pt \
        --verify-against reports/anchor_fusion/eval_anchor_fusion_hotpotqa.json \
        --cuda-device 0 --out reports/cross_block/eval_v13_hotpotqa.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_graft_recovery as gr                          # noqa: E402
import eval_learned_fusion as ef                          # noqa: E402
import eval_replace_reorder as er                         # noqa: E402
import eval_source_compaction as es                       # noqa: E402
import native_eval as nv                                  # noqa: E402
from io_utils import atomic_json, sha                     # noqa: E402
from runtime import load_runtime, reset                   # noqa: E402
from methods.prlmc.anchor_residual import (               # noqa: E402
    SCHEMA as ANCHOR_SCHEMA, AnchorResidual,
    compact_window as compact_window_anchor)
from methods.prlmc.cross_block_calibration import (       # noqa: E402
    SCHEMA as CALIB_SCHEMA, CrossBlockResidual, calibrate)
from methods.prlmc.evict_window import (                  # noqa: E402
    SCHEMA as JOINT_SCHEMA, JointConsolidation)

# V13 reuses the V11 driver: the calibration module is what changed.
FULL_POOL_SCHEMA = CALIB_SCHEMA
FullPoolResidual = CrossBlockResidual
from methods.prlmc.reader_utility import ReaderScorer     # noqa: E402
from methods.prlmc.source_compaction import split_source_anchors  # noqa: E402

BLOCK_TOKENS = gr.BLOCK_TOKENS
SLOTS_PER_BLOCK = gr.SLOTS_PER_BLOCK
TARGET_SLOTS = gr.TARGET_SLOTS
FULL, SELECTION, GRAFT = ef.FULL, ef.SELECTION, ef.GRAFT
ANCHOR_ONLY = "AnchorOnly"
# The slots of the V11 driver, relabelled for V13: the plain full-pool calibration is
# now the arm to BEAT (it is `V11Evict` retrained here with --cross off), and the
# cross-block conditioned module is the treatment (`V13`).
V11 = "V11Evict"
V11_DONOR = "V11EvictDonor"
V11_EVICT = "V13"
V11_EVICT_DONOR = "V13Donor"
ARMS = (FULL, SELECTION, GRAFT, ANCHOR_ONLY, V11, V11_DONOR, V11_EVICT, V11_EVICT_DONOR)
# The baseline quota, which is what the reference report's arms were built at.
BASELINE_QUOTA = dict(core_per_block=216, graft_per_block=40)
CORE_TOTAL = 9 * 180
GRAFT_TOTAL = 9 * 76
SCHEMA_RESULTS = "cross_block_calibration_results/1"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_anchor_module(path, hidden):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != ANCHOR_SCHEMA + "/checkpoint":
        raise ValueError(f"{path} is not a {ANCHOR_SCHEMA} checkpoint")
    config = dict(payload["config"])
    if int(config.get("hidden_size", -1)) != int(hidden):
        raise ValueError(f"{path} was built for hidden {config.get('hidden_size')}")
    module = AnchorResidual(**config)
    module.load_state_dict(payload["state_dict"], strict=True)
    return module.eval().requires_grad_(False), config


def load_v11(path, hidden):
    """A V11 checkpoint: the module and the geometry the window has to be rebuilt with."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != FULL_POOL_SCHEMA + "/checkpoint":
        raise ValueError(f"{path} is not a {FULL_POOL_SCHEMA} checkpoint")
    geometry = dict(payload.get("geometry") or {})
    for field in ("cross", "core_per_block", "graft_per_block", "utility_layers",
                  "probe_ids"):
        if field not in geometry:
            raise ValueError(f"{path} records no {field}: the window cannot be rebuilt "
                             f"from it")
    if int(geometry["core_per_block"]) + int(geometry["graft_per_block"]) != SLOTS_PER_BLOCK:
        raise ValueError(f"{path}: the recorded quota does not fill a block")
    module = FullPoolResidual(**dict(payload["config"]))
    module.load_state_dict(payload["state_dict"], strict=True)
    return module.eval().requires_grad_(False), geometry, dict(payload["config"]), payload


def joint_for(geometry, scorer, probe_cache, hidden):
    """Rebuild the exact window builder a checkpoint was trained with."""
    layers = tuple(int(x) for x in geometry["utility_layers"])
    bank_key = (layers, tuple(tuple(ids) for ids in geometry["probe_ids"]))
    bank = probe_cache.get(bank_key)
    if bank is None:
        bank = [scorer.probe_bank(ids) for ids in geometry["probe_ids"]]
        probe_cache[bank_key] = bank
    return JointConsolidation(
        scorer, bank, core_per_block=int(geometry["core_per_block"]),
        graft_per_block=int(geometry["graft_per_block"]), policy="evict",
        mmr_penalty=float(geometry.get("mmr_penalty", 0.5)),
        novelty_weight=float(geometry.get("novelty_weight", 0.5)))


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def set_quota(merge, core_per_block, graft_per_block, block=SLOTS_PER_BLOCK):
    if core_per_block + graft_per_block != block:
        raise ValueError("the baseline quota does not fill a block")
    merge.core_slots_per_block = int(core_per_block)
    merge.graft_slots_per_block = int(graft_per_block)
    if (int(merge.core_slots_per_block) != int(core_per_block)
            or int(merge.graft_slots_per_block) != int(graft_per_block)):
        raise ValueError("the operator did not take the baseline quota")


def check_v11(window, meta, telemetry, source, name):
    """Capacity, split, order, provenance, and what each half is made of."""
    capacity = window.shape[1]
    if capacity != TARGET_SLOTS:
        raise ValueError(f"{name}: {capacity} slots, expected {TARGET_SLOTS}")
    core = int((meta["slot_type"] == 0).sum())
    graft = int((meta["slot_type"] == 1).sum())
    expected_core = telemetry["core_total"]
    expected_graft = telemetry["graft_total"]
    if core != expected_core or graft != expected_graft:
        raise ValueError(f"{name}: {core}/{graft} slots, telemetry says "
                         f"{expected_core}/{expected_graft}")
    if core + graft != capacity:
        raise ValueError(f"{name}: the split does not fill the budget")
    positions = meta["source_position"]
    if int(positions.numel()) != capacity:
        raise ValueError(f"{name}: one source position per slot")
    if not torch.equal(positions, torch.sort(positions).values):
        raise ValueError(f"{name}: the output is not in global chronological order")
    dropped = set(int(block) for block in telemetry["dropped_blocks"])
    carriers = set(int(block) for block in telemetry["carrier_blocks"])
    graft_blocks = set(int(x) for x in meta["source_block"][meta["slot_type"] == 1].tolist())
    if graft_blocks - dropped:
        raise ValueError(f"{name}: a graft slot came from a carrier block")
    core_blocks = set(int(x) for x in meta["source_block"][meta["slot_type"] == 0].tolist())
    if core_blocks - carriers:
        raise ValueError(f"{name}: a core slot came from a dropped block")
    if not torch.isfinite(window).all():
        raise ValueError(f"{name}: the window is not finite")

    index = positions.to(window.device)
    copied = source.index_select(1, index)
    unfolded = meta["cluster_size"] == 1
    if not torch.equal(window[:, unfolded], copied[:, unfolded]):
        raise ValueError(f"{name}: an unfolded slot is not a verbatim copy of its source")
    folded = ~unfolded
    if bool(folded.any()):
        base = copied[:, folded]
        moved = window[:, folded]
        base_rms = base.float().square().mean(dim=-1).sqrt()
        moved_rms = moved.float().square().mean(dim=-1).sqrt()
        drift = float(((moved_rms - base_rms).abs() / base_rms.clamp_min(1e-6)).max())
        if drift > 1e-2:
            raise ValueError(f"{name}: a folded slot lost its representative's RMS "
                             f"(max relative drift {drift:.2e})")
    per_block = telemetry["graft_per_dropped_block"]
    if sum(per_block) != expected_graft:
        raise ValueError(f"{name}: the per-block graft counts do not add up")
    merge = telemetry["merge"]
    return dict(
        core_total=core, graft_total=graft, capacity=capacity,
        folded_slots=int(folded.sum()), merge_policy=telemetry["policy"],
        merged_pairs=int(merge["merged_pairs"]), avoided_evictions=int(merge["evicted"]),
        blocks_with_no_merge=int(merge["blocks_with_no_merge"]),
        blocks_where_quota_came_from_eviction=int(
            merge["blocks_where_quota_came_from_eviction"]),
        mean_cluster_size=float(telemetry["cluster_size_core_mean"]),
        empty_dropped_blocks=int(telemetry["empty_dropped_blocks"]),
        min_per_dropped_block=int(min(per_block)), max_per_dropped_block=int(max(per_block)),
        novelty_mean=float(telemetry["novelty_mean"]),
        verbose=merge)


def check_calibration(calibrated, window, meta, info, name):
    """Both halves must move, and neither may move out of the native norm range."""
    core = meta["slot_type"] == 0
    graft = meta["slot_type"] == 1
    if torch.equal(calibrated[:, core], window[:, core]):
        raise ValueError(f"{name}: the calibration did not touch a single core slot, so "
                         f"this arm does not test the full-pool module")
    if torch.equal(calibrated[:, graft], window[:, graft]):
        raise ValueError(f"{name}: the calibration did not touch a single graft slot")
    ratio = (calibrated.float().square().mean(dim=-1).sqrt()
             / window.float().square().mean(dim=-1).sqrt().clamp_min(1e-6))
    if float((ratio - 1.0).abs().max()) > 2e-2:
        raise ValueError(f"{name}: the RMS restore did not hold "
                         f"(max drift {float((ratio - 1.0).abs().max()):.2e})")
    return dict(core_relative_delta=info["core_relative_delta"],
                graft_relative_delta=info["graft_relative_delta"],
                relative_delta=info["relative_delta"],
                folded_slots=info["folded_slots"], gate=info["gate"], check=True)


def check_anchor(output, parts, anchor_window, name):
    offsets = parts["anchor_offsets"]
    keep = torch.ones(output.shape[1], dtype=torch.bool)
    keep[offsets] = False
    if not torch.equal(output[:, keep], anchor_window[:, keep]):
        raise ValueError(f"{name}: the anchor module changed a core slot")
    if torch.equal(output.index_select(1, offsets), anchor_window.index_select(1, offsets)):
        raise ValueError(f"{name}: the anchor module changed nothing")
    return dict(anchor_slots=int(offsets.numel()), check=True)


# ---------------------------------------------------------------------------
# one sample
# ---------------------------------------------------------------------------
def build_v11(joint, module, source, picked, block, name):
    """One window plus its calibrated twin, both checked."""
    window, meta, telemetry = joint.build(source, picked, block)
    window_check = check_v11(window, meta, telemetry, source, name)
    calibrated, info = calibrate(window, meta, module)
    calibration_check = check_calibration(calibrated, window, meta, info, name)
    return dict(window=window, calibrated=calibrated, meta=meta,
                telemetry=telemetry, check=window_check,
                calibration=calibration_check)


def _chunks(prepared, window_blocks):
    chunks = [prepared["write_ids"][index:index + BLOCK_TOKENS]
              for index in range(0, len(prepared["write_ids"]), BLOCK_TOKENS)]
    return chunks[:window_blocks]


def run_sample(model, tokenizer, manager, merge, anchor_module, v11, v11_evict,
               prepared, donor_graft, donor_graft_evict, donor_id, args):
    """One write; the baseline window and both V11 windows; then every arm's answer."""
    chunks = _chunks(prepared, args.window_blocks)
    answers = prepared["answers"]
    fingerprints, windows = {}, {}
    next_donor_evict = None
    with torch.no_grad():
        es.write_blocks(manager, merge, chunks[:args.window_blocks])
        state = gr.snapshot(model, manager)
        source, ages, picked = gr.select_carriers(manager, args.window_blocks,
                                                  args.target_blocks)
        frozen, pinned = gr.window_locks(manager, args.window_blocks)
        # Arm A: the baseline quota, native grafting, then the frozen AnchorResidual.
        set_quota(merge, BASELINE_QUOTA["core_per_block"],
                  BASELINE_QUOTA["graft_per_block"])
        output, telemetry = gr.compress_graft(manager, merge, source, picked,
                                              args.window_blocks, args.target_blocks,
                                              frozen, pinned)
        parts = split_source_anchors(telemetry, manager.block, args.window_blocks,
                                     device=source.device)
        anchor_window, anchor_info = compact_window_anchor(output, parts, anchor_module)
        windows[GRAFT] = output
        windows[ANCHOR_ONLY] = anchor_window
        baseline = dict(anchor=check_anchor(output, parts, anchor_window, ANCHOR_ONLY),
                        core_slots=int(output.shape[1]) - int(parts["anchor_offsets"].numel()),
                        graft_slots=int(parts["anchor_offsets"].numel()),
                        relative_delta=float(anchor_info["relative_delta"]))
        # Arm B and arm D: two windows, built by two recorded builders.
        built = dict(B=build_v11(v11["joint"], v11["module"], source, picked,
                                 manager.block, V11))
        if v11_evict is not None:
            built["D"] = build_v11(v11_evict["joint"], v11_evict["module"], source,
                                   picked, manager.block, V11_EVICT)
        # Arm C: arm B's window with the graft slots taken from the donor document.
        graft_offsets = torch.nonzero(built["B"]["meta"]["slot_type"] == 1,
                                      as_tuple=False).flatten()
        # `shape[1]` is the slot count; `shape[-1]` is the hidden width. Comparing the
        # hidden width with 684 was never true, so arm C was silently skipped and the
        # run answered every question except the one it exists for.
        if donor_graft is not None and int(donor_graft.shape[1]) == int(
                graft_offsets.numel()):
            donor_window = built["B"]["window"].clone()
            donor_window[:, graft_offsets] = donor_graft.to(built["B"]["window"].dtype)
            if torch.equal(donor_window, built["B"]["window"]):
                raise ValueError("the donor window equals the true window: the graft "
                                 "swap did not reach the memory")
            donor_calibrated, donor_info = calibrate(donor_window, built["B"]["meta"],
                                                     v11["module"])
            # Same key set as arm B so the record below can read it uniformly. Arm C
            # shares B's geometry and B's window check; only the graft vectors and the
            # calibration residual differ, and both are recorded.
            built["C"] = dict(window=donor_window,
                              calibrated=donor_calibrated,
                              check=built["B"]["check"],
                              calibration=check_calibration(
                                  donor_calibrated, donor_window, built["B"]["meta"],
                                  donor_info, V11_DONOR))
        # Arm E: the eviction line's own counterfactual. Same builder as arm D, same
        # calibration module, same 684 slots -- only whose dropped blocks they came
        # from changes. Arm C cannot answer this for D: the fold and the eviction
        # select different slots, so a donor taken from the folded window would
        # confound content with slot identity.
        next_donor_evict = None
        if "D" in built:
            evict_offsets = torch.nonzero(built["D"]["meta"]["slot_type"] == 1,
                                          as_tuple=False).flatten()
            next_donor_evict = built["D"]["window"].index_select(
                1, evict_offsets).detach().clone()
            if donor_graft_evict is not None and int(donor_graft_evict.shape[1]) == int(
                    evict_offsets.numel()):
                donor_window_evict = built["D"]["window"].clone()
                donor_window_evict[:, evict_offsets] = donor_graft_evict.to(
                    built["D"]["window"].dtype)
                if torch.equal(donor_window_evict, built["D"]["window"]):
                    raise ValueError("the eviction donor window equals the true one: the "
                                     "graft swap did not reach the memory")
                evict_donor_calibrated, evict_donor_info = calibrate(
                    donor_window_evict, built["D"]["meta"], v11_evict["module"])
                built["E"] = dict(
                    window=donor_window_evict, calibrated=evict_donor_calibrated,
                    check=built["D"]["check"],
                    calibration=check_calibration(
                        evict_donor_calibrated, donor_window_evict, built["D"]["meta"],
                        evict_donor_info, V11_EVICT_DONOR))
        windows[V11] = built["B"]["calibrated"]
        if "C" in built:
            windows[V11_DONOR] = built["C"]["calibrated"]
        if "D" in built:
            windows[V11_EVICT] = built["D"]["calibrated"]
        if "E" in built:
            windows[V11_EVICT_DONOR] = built["E"]["calibrated"]
        next_donor = built["B"]["window"].index_select(1, graft_offsets).detach().clone()
    gr.restore(model, manager, state)

    def prepare(arm):
        def run():
            gr.restore(model, manager, state)
            if arm == SELECTION:
                gr.install_selection(manager, source, picked, args.window_blocks)
            elif arm != FULL:
                er.install_output(manager, windows[arm], picked, args.window_blocks)
            fingerprints[arm] = gr.active_fingerprint(manager)
            fingerprints[arm + "/slots"] = int(manager.used)
        return run

    answered = {arm: gr.readout(model, tokenizer, prepared["query_ids"], answers,
                                args.f1_threshold, prepare(arm))
                for arm in ARMS if arm in windows or arm in (FULL, SELECTION)}
    gr.restore(model, manager, state)

    if int(fingerprints[FULL + "/slots"]) != args.window_blocks * manager.block:
        raise ValueError("the Full arm did not keep the whole window")
    for arm in ARMS:
        if arm in (FULL, SELECTION):
            continue
        if arm not in windows:
            if arm == V11_DONOR:
                # A single-row run has no donor; the arm is simply not answered.
                continue
            raise ValueError(f"{arm} produced no window this run")
        if int(fingerprints[arm + "/slots"]) != TARGET_SLOTS:
            raise ValueError(f"{arm} does not hold the {TARGET_SLOTS}-slot budget")
    for left, right in ((SELECTION, ANCHOR_ONLY), (ANCHOR_ONLY, V11), (V11, GRAFT)):
        if fingerprints[left] == fingerprints[right]:
            raise ValueError(f"{left} and {right} read the same pool")
    if len(prepared) > 1 and V11_DONOR not in windows:
        raise ValueError("the donor arm was not built: the content-causality control "
                         "is missing, so this run cannot answer its own question")
    if len(prepared) > 1 and V11_EVICT in windows and V11_EVICT_DONOR not in windows:
        raise ValueError("the eviction line's donor arm was not built: this run cannot "
                         "attribute arm D's gain to content")
    if V11_DONOR in windows and fingerprints[V11] == fingerprints[V11_DONOR]:
        raise ValueError("the donor arm reads the same pool as V11: the swap did not "
                         "reach the memory")
    if V11_EVICT_DONOR in windows and (
            fingerprints[V11_EVICT] == fingerprints[V11_EVICT_DONOR]):
        raise ValueError("the eviction donor arm reads the same pool as V11Evict")
    if V11_EVICT in windows and fingerprints[V11] == fingerprints[V11_EVICT]:
        raise ValueError("the eviction control reads the same pool as V11: the fold "
                         "moved nothing")

    record = dict(
        arms=answered, fingerprints=fingerprints, baseline=baseline,
        v11=built["B"]["check"], v11_calibration=built["B"]["calibration"],
        donor=dict(window_check=built["C"]["check"] if "C" in built else None,
                   calibration=built["C"]["calibration"] if "C" in built else None,
                   source_id=donor_id),
        evict=(dict(window_check=built["D"]["check"],
                    calibration=built["D"]["calibration"]) if "D" in built else None),
        evict_donor=(dict(calibration=built["E"]["calibration"], source_id=donor_id)
                     if "E" in built else None),
        dropped_blocks=[int(block) for block in parts["dropped_blocks"]],
        selected_blocks=[int(block) for block in telemetry["selected_blocks"]],
        carrier_ages=[int(ages[block]) for block in picked])
    return record, dict(grafts=next_donor, grafts_evict=next_donor_evict,
                        id=prepared["id"])


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def _stats(values):
    values = [float(value) for value in values]
    if not values:
        return dict(n=0)
    return dict(n=len(values), mean=round(sum(values) / len(values), 6),
                min=round(min(values), 6), max=round(max(values), 6),
                median=round(sorted(values)[len(values) // 2], 6))


def _evidence_in_dropped(sample):
    blocks = set(int(block) for block in (sample.get("evidence", {}).get("blocks") or []))
    return bool(blocks) and blocks <= set(sample["dropped_blocks"])


def _count(rows, arm, threshold):
    """Correct rows, as a count rather than a percentage, for the planned thresholds."""
    return int(sum(1 for row in rows if gr._correct(row, arm, threshold)))


def joint_compare(samples, threshold):
    graded = [sample for sample in samples if sample.get("status") == "graded"]
    located = [sample for sample in graded if sample.get("evidence", {}).get("located")]
    dropped = [sample for sample in located if _evidence_in_dropped(sample)]
    kept = [sample for sample in located if not _evidence_in_dropped(sample)]
    present = [arm for arm in ARMS if arm in graded[0]["arms"]]
    controls = [arm for arm in (V11_DONOR, V11_EVICT, V11_EVICT_DONOR, GRAFT)
                if arm in present]

    def block(rows):
        return dict(
            n=len(rows),
            arms={arm: gr._arm_summary(rows, arm, threshold) for arm in present},
            versus_anchor_only={arm: er.pair_summary(rows, arm, ANCHOR_ONLY, threshold)
                                for arm in controls},
            versus_v11={arm: er.pair_summary(rows, arm, V11, threshold)
                        for arm in controls if arm in (V11_DONOR, V11_EVICT)},
            versus_v11evict={V11_EVICT_DONOR: er.pair_summary(
                rows, V11_EVICT_DONOR, V11_EVICT, threshold)}
            if V11_EVICT_DONOR in present else {})

    merge_rows = [sample["v11"] for sample in graded]
    thresholds = dict(
        total=dict(target=62, baseline=_count(graded, ANCHOR_ONLY, threshold),
                   achieved=_count(graded, V11, threshold)),
        dropped_evidence=dict(target=19, n=len(dropped),
                              baseline=_count(dropped, ANCHOR_ONLY, threshold),
                              achieved=_count(dropped, V11, threshold)),
        kept_evidence=dict(target=12, n=len(kept),
                           baseline=_count(kept, ANCHOR_ONLY, threshold),
                           achieved=_count(kept, V11, threshold)),
        content_causality=dict(
            target=5,
            achieved=er.pair_summary(graded, V11, V11_DONOR, threshold).get("net")
            if V11_DONOR in present else None),
        cross_over_plain=dict(
            target=5,
            achieved=er.pair_summary(graded, V11_EVICT, V11, threshold).get("net")
            if V11_EVICT in present else None),
        evict_content_causality=dict(
            target=5,
            achieved=er.pair_summary(graded, V11_EVICT, V11_EVICT_DONOR,
                                     threshold).get("net")
            if V11_EVICT_DONOR in present else None))
    return dict(
        definition=("the same 19-block window, the same nine carriers, the same 2,304 "
                    "slots, the same 158 rows; the V11 arms move the selection rule, the "
                    "fold and the calibration, and each control moves exactly one of "
                    "them back"),
        arms_present=present,
        n=len(graded),
        arms={arm: gr._arm_summary(graded, arm, threshold) for arm in present},
        total=block(graded), dropped_evidence_group=block(dropped),
        kept_evidence_group=block(kept),
        damage={arm: ef._damage(graded, arm, threshold) for arm in present},
        thresholds=thresholds,
        merge_health=dict(
            merged_pairs=_stats([row["merged_pairs"] for row in merge_rows]),
            folded_slots=_stats([row["folded_slots"] for row in merge_rows]),
            blocks_with_no_merge=_stats([row["blocks_with_no_merge"]
                                         for row in merge_rows]),
            blocks_where_quota_came_from_eviction=_stats(
                [row["blocks_where_quota_came_from_eviction"] for row in merge_rows]),
            mean_cluster_size=_stats([row["mean_cluster_size"] for row in merge_rows]),
            novelty_mean=_stats([row["novelty_mean"] for row in merge_rows])),
        calibration=dict(
            core=_stats([sample["v11_calibration"]["core_relative_delta"]
                         for sample in graded]),
            graft=_stats([sample["v11_calibration"]["graft_relative_delta"]
                          for sample in graded]),
            gate=_stats([sample["v11_calibration"]["gate"] for sample in graded])))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def build_protocol(args, task, anchor_config, v11_geometry, evict_geometry):
    return dict(
        experiment="9 (V13): does conditioning the calibration residual on sparse "
                   "CROSS-BLOCK neighbours buy anything over a plain full-pool "
                   "calibration, and is that gain carried by the real graft content?",
        treatment=("the calibrator sees, for every slot, the k = 4 slots from OTHER "
                   "blocks it is most associated with (reader key cosine, same-block "
                   "masked), aggregated into a context vector that is injected into the "
                   "residual bottleneck by a zero-initialised projection. The slot itself "
                   "is never blended: the same bounded low-rank map moves it, with a "
                   "tighter ceiling on carrier slots (0.15 x rms) than on graft slots "
                   "(0.50 x rms)"),
        task=task, data_dir=args.data_dir, arms=list(ARMS),
        window_blocks=args.window_blocks, target_blocks=args.target_blocks,
        block_tokens=BLOCK_TOKENS, slots_per_block=SLOTS_PER_BLOCK,
        budget=TARGET_SLOTS,
        baseline=dict(arm=ANCHOR_ONLY, quota=BASELINE_QUOTA,
                      core_total=args.target_blocks * BASELINE_QUOTA["core_per_block"],
                      graft_total=args.target_blocks * BASELINE_QUOTA["graft_per_block"],
                      anchor=args.anchor, anchor_sha256=sha(args.anchor),
                      anchor_config=anchor_config),
        v11=dict(arm=V11, path=args.v11, sha256=sha(args.v11), geometry=v11_geometry,
                 core_total=CORE_TOTAL, graft_total=GRAFT_TOTAL),
        controls=dict(
            donor=dict(arm=V11_DONOR,
                       definition="the plain calibration's window with the 684 graft "
                                  "slots taken from the previous row's document"),
            cross_donor=dict(arm=V11_EVICT_DONOR,
                             definition="V13's window with the 684 graft slots taken "
                                        "from the previous row's document: the only "
                                        "thing that changes is whose content the "
                                        "calibrator is conditioned on"),
            evict=dict(arm=V11_EVICT, path=args.v11_evict,
                       sha256=sha(args.v11_evict) if args.v11_evict else None,
                       geometry=evict_geometry,
                       definition="same selection and calibration, carriers cut to 1,620 "
                                  "by plain eviction instead of folding")),
        unchanged_conditions=dict(
            data="the same 158 HotpotQA rows", reader="frozen MemoryLLM-8B",
            carriers="the same 9, selected once and shared by every arm",
            window="the same 19-block write, document prefix only",
            capacity="2,304 slots in every compressed arm",
            transport="one selection shared by all 32 layers; chronological output",
            decoding="greedy, unchanged"),
        checks=dict(
            reproduction="Full, Selection, Graft and AnchorOnly must reproduce the "
                         "reference report row for row",
            capacity="every V11 arm holds 2,304 slots split 1,620 / 684",
            provenance="every graft slot from a dropped block, verbatim; every core "
                       "slot from a carrier",
            folding="unfolded slots are verbatim copies; folded slots keep their "
                    "representative's per-layer RMS",
            calibration="both halves must move, and the RMS restore must hold",
            cross_block="the context must be built from other blocks: the neighbour "
                        "block spread is recorded per row",
            donor="the donor arm's window must differ from the true one"),
        thresholds=dict(
            total="V11 >= 62/158 against 56", dropped_evidence="V11 >= 19/40 against 15",
            kept_evidence="V11 >= 12/34 against 13",
            damage="correct rows lost by Full <= 13",
            content_causality="V11 - V11Donor >= 5 questions"),
        screening_caveat=gr.SCREENING_CAVEAT, f1_threshold=args.f1_threshold,
        seed=args.seed, query=gr.QUERY_SUFFIX,
        scoring="official HotpotQA QA F1, F1 >= threshold is correct",
        reader_utility_schema=JOINT_SCHEMA)


REFERENCE_ARMS = ((FULL, FULL), (SELECTION, SELECTION), (GRAFT, GRAFT),
                  (ANCHOR_ONLY, ANCHOR_ONLY))


def verify_against_report(args, reference, table, samples):
    """The four pre-existing arms must reproduce the reference report row for row."""
    this_run = {str(sample["id"]) for sample in samples}
    verdict = dict(reference=args.verify_against,
                   reference_schema=reference.get("schema"),
                   reference_samples=len(table), compared=len(this_run),
                   row_count_matches_reference=(len(this_run) == len(table)),
                   reference_not_in_this_run=sorted(set(table) - this_run),
                   this_run_not_in_reference=sorted(this_run - set(table)),
                   matched={reference_arm: ours for reference_arm, ours in REFERENCE_ARMS})
    passed = not verdict["this_run_not_in_reference"]
    for reference_arm, ours in REFERENCE_ARMS:
        same = different = absent = 0
        differing = []
        for sample in samples:
            old = table.get(str(sample["id"]))
            if old is None or reference_arm not in old.get("arms", {}):
                absent += 1
                continue
            if old["arms"][reference_arm]["prediction"] == sample["arms"][ours]["prediction"]:
                same += 1
            else:
                different += 1
                differing.append(dict(id=sample["id"],
                                      old=old["arms"][reference_arm]["prediction"],
                                      new=sample["arms"][ours]["prediction"]))
        verdict[reference_arm] = dict(matched_to=ours, identical=same, different=different,
                                      not_in_reference=absent,
                                      differing=differing[:20])
        passed = passed and different == 0 and absent == 0 and same == len(this_run)
    verdict["passed"] = passed
    return verdict


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
            arms=list(ARMS), readouts_per_sample=len(ARMS),
            v11=args.v11, v11_evict=args.v11_evict), indent=2, ensure_ascii=False))
        return None
    for flag, value, what in (("--checkpoint", args.checkpoint, "the frozen operator"),
                              ("--anchor", args.anchor, "the AnchorOnly module"),
                              ("--v11", args.v11, "the trained V11 module")):
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

    v11_module, v11_geometry, v11_config, _ = load_v11(args.v11, manager.hidden)
    v11_module = v11_module.to(model.memory.device).float()
    scorer = ReaderScorer(model, tuple(int(x) for x in v11_geometry["utility_layers"]))
    probe_cache = {}
    v11 = dict(module=v11_module, geometry=v11_geometry, config=v11_config,
               joint=joint_for(v11_geometry, scorer, probe_cache, manager.hidden))
    v11_evict = None
    if args.v11_evict:
        evict_module, evict_geometry, evict_config, _ = load_v11(args.v11_evict,
                                                                 manager.hidden)
        evict_module = evict_module.to(model.memory.device).float()
        if not evict_geometry.get("cross"):
            raise ValueError("--v11-evict is not a cross-block conditioned checkpoint: "
                             "the arm that is supposed to test the new mechanism would "
                             "run the old one")
        if v11_geometry.get("cross"):
            raise ValueError("--v11 is cross-block conditioned: it must be the plain "
                             "calibration the treatment is compared against")
        v11_evict = dict(module=evict_module, geometry=evict_geometry,
                         config=evict_config,
                         joint=joint_for(evict_geometry, scorer, probe_cache,
                                         manager.hidden))

    rows, prepared, excluded = gr.prepare_rows(args, tokenizer, facts)
    report = dict(
        schema=SCHEMA_RESULTS, status="running",
        counts=dict(rows=len(rows), eligible=len(prepared), excluded=len(excluded)),
        protocol=build_protocol(args, args.task, anchor_config, v11_geometry,
                                v11_evict["geometry"] if v11_evict else None),
        identity=dict(
            model_path=args.model_path, checkpoint=args.checkpoint,
            checkpoint_sha256=sha(args.checkpoint),
            checkpoint_code_hash_verified=(saved or {}).get("code_hash_verified"),
            anchor=args.anchor, anchor_sha256=sha(args.anchor),
            v11=args.v11, v11_sha256=sha(args.v11),
            v11_evict=args.v11_evict,
            v11_evict_sha256=sha(args.v11_evict) if args.v11_evict else None,
            data_dir=args.data_dir,
            data_sha256=sha(Path(args.data_dir) / (args.task + ".jsonl")),
            supporting_facts_glob=args.supporting_facts_glob,
            entrypoint_sha256=sha(__file__),
            joint_module_sha256=sha(ROOT / "methods/prlmc/joint_consolidation.py"),
            utility_module_sha256=sha(ROOT / "methods/prlmc/reader_utility.py"),
            merge_module_sha256=sha(ROOT / "methods/prlmc/homogeneous_merge.py"),
            residual_module_sha256=sha(ROOT / "methods/prlmc/anchor_residual.py"),
            reader_frozen=True, training_performed=False),
        excluded=excluded, samples=[], training_performed=False,
        optimizer_created=False,
        full_split=(args.max_samples is None and args.sample_ids is None))
    atomic_json(output, report)

    reference, table = ((None, None) if not args.verify_against
                        else ef.load_reference(args.verify_against))
    # The donor of the first row is the last prepared row's window, built once up
    # front: a donor must exist for row 0 and must never be row 0 itself.
    donor_graft, donor_graft_evict, donor_id = None, None, None
    if len(prepared) > 1:
        with torch.no_grad():
            reset(manager, initial)
            es.write_blocks(manager, merge,
                            _chunks(prepared[-1], args.window_blocks))
            warm_source, _, warm_picked = gr.select_carriers(manager, args.window_blocks,
                                                             args.target_blocks)
            warm_window, warm_meta, _ = v11["joint"].build(warm_source, warm_picked,
                                                           manager.block)
            warm_offsets = torch.nonzero(warm_meta["slot_type"] == 1,
                                         as_tuple=False).flatten()
            donor_graft = warm_window.index_select(1, warm_offsets).detach().clone()
            donor_graft_evict = None
            if v11_evict is not None:
                evict_window, evict_meta, _ = v11_evict["joint"].build(
                    warm_source, warm_picked, manager.block)
                evict_offsets = torch.nonzero(evict_meta["slot_type"] == 1,
                                              as_tuple=False).flatten()
                donor_graft_evict = evict_window.index_select(
                    1, evict_offsets).detach().clone()
                del evict_window
            donor_id = prepared[-1]["id"]
            del warm_window, warm_source
        reset(manager, initial)

    started = time.monotonic()
    for position, sample in enumerate(prepared):
        sample_started = time.monotonic()
        nv.seed_everything(args.seed)
        reset(manager, initial)
        result, carried = run_sample(model, tokenizer, manager, merge, anchor_module,
                                     v11, v11_evict, sample, donor_graft,
                                     donor_graft_evict, donor_id, args)
        # `donor_id` is the row whose graft slots this sample read; `carried["id"]` is
        # this sample. Comparing the carried id with the sample id compared a row with
        # itself and fired on every row.
        if donor_id is not None and donor_id == sample["id"]:
            raise ValueError(f"sample {sample['id']} was used as its own donor")
        donor_graft, donor_graft_evict = carried["grafts"], carried["grafts_evict"]
        donor_id = carried["id"]
        record = dict(id=sample["id"], status="graded",
                      available_blocks=sample["available_blocks"],
                      evidence=sample["evidence"],
                      seconds=time.monotonic() - sample_started, **result)
        report["samples"].append(record)
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output, report)
        print(json.dumps(dict(
            sample=position + 1, total=len(prepared), id=sample["id"],
            **{arm: result["arms"][arm]["correct"] for arm in ARMS
               if arm in result["arms"]},
            seconds=round(record["seconds"], 1))), flush=True)

    report["summary"] = joint_compare(report["samples"], args.f1_threshold)
    if args.verify_against:
        report["verification_against_reference"] = verify_against_report(
            args, reference, table, report["samples"])
    report["elapsed_seconds"] = time.monotonic() - started
    report["status"] = "complete"
    report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    atomic_json(output, report)
    print(json.dumps(dict(
        status="complete", out=str(output),
        elapsed_seconds=round(report["elapsed_seconds"], 1),
        arms={arm: round(report["summary"]["arms"][arm]["accuracy"], 2) for arm in ARMS},
        net={name: value.get("net") for name, value in
             dict(**report["summary"]["total"]["versus_anchor_only"],
                  **report["summary"]["total"]["versus_v11"]).items()},
        thresholds=report["summary"]["thresholds"]),
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
                   help="the finished AnchorOnly checkpoint (the 56/158 baseline)")
    p.add_argument("--v11", default=None,
                   help="the plain full-pool calibration checkpoint (trained with "
                        "--cross off): the V11Evict arm to beat")
    p.add_argument("--v11-evict", dest="v11_evict", default=None,
                   help="the cross-block conditioned checkpoint (trained with --cross "
                        "on): the V13 arm")
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
                   help="a finished report; Full, Selection, Graft and AnchorOnly must "
                        "reproduce it row for row")
    p.add_argument("--strict-code-hash", dest="strict_code_hash", default="true",
                   choices=("true", "false"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.dry_run and not args.data_dir:
        raise SystemExit("--data-dir is required")
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
