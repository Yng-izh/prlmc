#!/usr/bin/env python3
"""Experiment 10 (V14): does the SOURCE of the neighbours buy anything?

    Full              19 blocks, uncompressed                     reference ceiling
    Selection         9 carrier blocks, verbatim                  capacity baseline
    Graft             V10 native graft, 1,944 + 360               the arm AnchorOnly beat
    AnchorOnly        the frozen AnchorResidual at 360 anchors     the 56/158 baseline
    V11Evict          V11Evict's window + the SAME module trained with --cross off
    V11EvictDonor     V11Evict, graft slots taken from ANOTHER document
    V13               the cross-block conditioned module, k = 4 kind-blind neighbours
    V13Donor          V13, graft slots taken from ANOTHER document
    V14               the SAME window as V13, 4 neighbours = 2 of the query's own
                      core/graft kind + 2 from the other side
    V14Donor          V14, graft slots taken from ANOTHER document

    V15               the layered sink-gated relation module on the SAME window as V13
    V15Donor          V15 on the donor window
    V16               the band-aligned, margin-gated module on the SAME window as V13
    V16Donor          V16 on the donor window
    V13Permuted<seed> inference-time attribution arm (one per `--v13-permute-seeds`
                      seed): V13's own window, weights and self-calibration branch, with
                      the precomputed cross-block context rows shuffled inside
                      (source_block, slot_type). No retraining, no re-selection.

V13 and V14 read the IDENTICAL window: `built["D"]`'s window is calibrated twice, by two
modules -- same 2,304 slots, same 1,620 / 684 split, same graft content, same rank,
ceilings, regulariser, loss, updates, cross learning rate and initialisation. The only
thing that differs is which slots each context row is built from, so `V14 - V13` measures
the neighbour rule and nothing else.

The questions, in the order they have to be answered
----------------------------------------------------
1. `V14 > V13` -- is the complement side worth anything over the kind-blind top-k that
   already reaches 67/158? The plan's bar is +4 to +5 questions net; below that the
   change is noise and the neighbour source was not the bottleneck.
2. `V14 > V13Donor` and `V14 > V14Donor` -- the gain still has to come from the CONTENT of
   the graft slots. Every mechanism in this project that passed (1) and failed (2) moved
   answers without carrying knowledge. `V13 > V11Evict` and `V11Evict > V11EvictDonor`
   are reported unchanged, so this run re-answers the earlier tests too.
3. Damage on the rows the uncompressed window already answers must not grow: a mechanism
   that buys questions by losing others is a redistribution, not a gain.

The permutation arms answer a different question: does V13's gain depend on WHICH slot
receives WHICH neighbour context, or would any context of the same provenance do? The
context is permuted AFTER it is computed, inside (source_block, slot_type) groups, so the
marginal distribution -- neighbour count, similarity range, block spread, context RMS --
is preserved and only the pairing is destroyed. Recorded for them: total correct, how many
of the rows V13 newly rescued over the plain calibration come back wrong, the damage on
Full's 47 protected rows, and the fraction of slots whose context actually moved (a
permutation that moved nothing is refused, not reported as "no effect").

The composition V13 never reported is now recorded per row, for every cross arm:
`neighbour_type_cross_fraction` (share of the selected neighbours from the other side of
the core/graft divide), `core_to_graft_neighbour_fraction`,
`graft_to_core_neighbour_fraction`, and `cross_weight_mass_mean` (how much of the context
weight the complement side actually received -- a complement arm whose cross mass sits
near zero has silently degenerated into "top-2 of my own kind").

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
* `--v14` must be a complement-mode checkpoint and `--v11-evict` a top-k one, with the
  same neighbour count, or the two arms would not differ in the neighbour source alone;
* the donor arm's window must differ from the true one;
* V14's calibrated window must differ from V13's: a complement rule that changes no slot
  changed nothing, and must not be reported as "no gain";
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
        --v11-evict runs/cross_block/cross_lr.pt \
        --v14 runs/cross_block/complement.pt \
        --verify-against reports/anchor_fusion/eval_anchor_fusion_hotpotqa.json \
        --cuda-device 0 --out reports/cross_block/eval_v14_hotpotqa.json
"""
import argparse
import copy
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
from methods.prlmc import band_relation as br             # noqa: E402
from methods.prlmc import hierarchical_relation as hr       # noqa: E402
from methods.prlmc.cross_block_calibration import (       # noqa: E402
    SCHEMA as CALIB_SCHEMA, CrossBlockResidual, calibrate, cross_block_context,
    permute_context, relevance_graft_caps, slot_relevance)
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
# V14: the same window, the same modules, the same training budget -- only the neighbour
# rule differs (2 of the query's own kind + 2 from the other side of the core/graft
# divide, instead of the 4 most similar cross-block slots whatever kind they are).
V14 = "V14"
V14_DONOR = "V14Donor"
# V15: the same window again, calibrated by the layered sink-gated relation module.
V15 = "V15"
V15_DONOR = "V15Donor"
# V16: band-aligned injection + post-projection margin gate, on V13's window again.
V16 = "V16"
V16_DONOR = "V16Donor"
# The V13 line's own successor: same module class/schema, different context rule
# (span-value neighbours, in-block diffusion, CSLS seats) and a tighter graft ceiling.
# Both checkpoints load through the same loader; each rebuilds its rule from its config.
V13_PLUS = "V13Plus"
V13_PLUS_DONOR = "V13PlusDonor"
ARMS = (FULL, SELECTION, GRAFT, ANCHOR_ONLY, V11, V11_DONOR, V11_EVICT, V11_EVICT_DONOR,
        V13_PLUS, V13_PLUS_DONOR, V14, V14_DONOR, V15, V15_DONOR, V16, V16_DONOR)
# Inference-time attribution arms for V13's cross-block correspondence: same window, same
# module, same self-calibration branch, context rows shuffled inside (source_block,
# slot_type). They are named after their seed and appended to `ARMS` at run time.
PERMUTE_PREFIX = "V13Permuted"
# Attribution arms for the neighbour CHOICE: V13's weights, window, projection and
# neighbour count are untouched; only the rule that picks the four slots changes.
NEIGHBOR_ARMS = {"random": "V13RandNeighbor", "bottom": "V13BottomNeighbor"}


def cap_arm(scale):
    """Arm name for a uniform inference-time graft ceiling: 0.38 -> V13Cap038."""
    return f"V13Cap{int(round(float(scale) * 100)):03d}"


def parse_scales(raw):
    if not raw:
        return []
    return [float(part) for part in str(raw).replace(" ", "").split(",") if part]


def permute_arm(seed):
    return f"{PERMUTE_PREFIX}{int(seed)}"


def parse_seeds(raw):
    """`--v13-permute-seeds 0,1,2` -> [0, 1, 2]; empty means the arms are not built."""
    if not raw:
        return []
    seeds = [int(part) for part in str(raw).replace(" ", "").split(",") if part != ""]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--v13-permute-seeds repeats a seed: {raw}")
    return seeds
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
    config = dict(payload["config"])
    if "use_cross" not in config:
        # This round's checkpoints were written before the field was recorded. The state
        # dict says it unambiguously: only a cross-block module carries `cross_proj`.
        config["use_cross"] = any(str(key).startswith("cross_proj")
                                  for key in payload["state_dict"])
    module = FullPoolResidual(**config)
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
                folded_slots=info["folded_slots"], gate=info["gate"],
                # Without these three the report cannot tell a mechanism that was
                # trained and did nothing from one that was never applied.
                use_cross=info.get("use_cross"),
                neighbor_mode=info.get("neighbor_mode"),
                k_neighbors=info.get("k_neighbors"),
                cross_proj_norm=info.get("cross_proj_norm"),
                cross_context=info.get("cross_context"),
                evicted_context=info.get("evicted_context"),
                relation=info.get("relation"),
                # V16's band diagnostics. Without these `_band_gate` finds no rows and
                # reports {n: 0} for every band -- an empty table that looks like "no
                # gate" rather than "the number was never carried out of the module".
                gate_bands=info.get("gate_bands"),
                band_proj_norm=info.get("band_proj_norm"),
                layer_proj_norm=info.get("layer_proj_norm"),
                cross_injection_rms=info.get("cross_injection_rms"),
                shallow_cross_leak=info.get("shallow_cross_leak"), check=True)


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
    calibrated, info = calibrate(window, meta, module, source=source)
    calibration_check = check_calibration(calibrated, window, meta, info, name)
    return dict(window=window, calibrated=calibrated, meta=meta,
                telemetry=telemetry, check=window_check,
                calibration=calibration_check)


def _chunks(prepared, window_blocks):
    chunks = [prepared["write_ids"][index:index + BLOCK_TOKENS]
              for index in range(0, len(prepared["write_ids"]), BLOCK_TOKENS)]
    return chunks[:window_blocks]


def run_sample(model, tokenizer, manager, merge, anchor_module, v11, v11_evict,
               prepared, donor_graft, donor_graft_evict, donor_id, args, v14=None,
               v15=None, v16=None, v13_plus=None):
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
                    donor_window_evict, built["D"]["meta"], v11_evict["module"],
                    source=source)
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
        # Arms for the V13 successor: the SAME window as D, calibrated by the module
        # that carries span-value / diffusion / CSLS in its own config.
        if v13_plus is not None:
            if "D" not in built:
                raise ValueError("--v13-plus needs the cross-block window it shares")
            plus_window, plus_info = calibrate(built["D"]["window"], built["D"]["meta"],
                                               v13_plus["module"], source=source)
            built["L"] = dict(window=built["D"]["window"], calibrated=plus_window,
                              check=built["D"]["check"],
                              calibration=check_calibration(
                                  plus_window, built["D"]["window"], built["D"]["meta"],
                                  plus_info, V13_PLUS))
            windows[V13_PLUS] = built["L"]["calibrated"]
            if "E" in built:
                plus_donor, plus_donor_info = calibrate(built["E"]["window"],
                                                        built["D"]["meta"],
                                                        v13_plus["module"],
                                                        source=source)
                built["M"] = dict(window=built["E"]["window"], calibrated=plus_donor,
                                  check=built["D"]["check"],
                                  calibration=check_calibration(
                                      plus_donor, built["E"]["window"],
                                      built["D"]["meta"], plus_donor_info,
                                      V13_PLUS_DONOR))
                windows[V13_PLUS_DONOR] = built["M"]["calibrated"]
        # Arms H and I: V15 on the SAME windows as D and E. Same slots, same graft
        # content, same budget; the relation rule and the module are what differ.
        if v15 is not None:
            if "D" not in built:
                raise ValueError("--v15 needs the cross-block window it has to share")
            relation_calibrated, relation_info = hr.calibrate(built["D"]["window"],
                                                              built["D"]["meta"],
                                                              v15["module"])
            built["H"] = dict(window=built["D"]["window"],
                              calibrated=relation_calibrated,
                              check=built["D"]["check"],
                              calibration=check_calibration(
                                  relation_calibrated, built["D"]["window"],
                                  built["D"]["meta"], relation_info, V15))
            windows[V15] = built["H"]["calibrated"]
            if "E" in built:
                relation_donor, relation_donor_info = hr.calibrate(
                    built["E"]["window"], built["D"]["meta"], v15["module"])
                built["I"] = dict(window=built["E"]["window"],
                                  calibrated=relation_donor,
                                  check=built["D"]["check"],
                                  calibration=check_calibration(
                                      relation_donor, built["E"]["window"],
                                      built["D"]["meta"], relation_donor_info,
                                      V15_DONOR))
                windows[V15_DONOR] = built["I"]["calibrated"]
        # Arms J and K: V16 on the SAME window as V13 and V15. Band routing and the
        # margin gate are the only things that differ.
        if v16 is not None:
            if "D" not in built:
                raise ValueError("--v16 needs the cross-block window it has to share")
            band_calibrated, band_info = br.calibrate(built["D"]["window"],
                                                      built["D"]["meta"],
                                                      v16["module"])
            built["J"] = dict(window=built["D"]["window"], calibrated=band_calibrated,
                              check=built["D"]["check"],
                              calibration=check_calibration(
                                  band_calibrated, built["D"]["window"],
                                  built["D"]["meta"], band_info, V16))
            windows[V16] = built["J"]["calibrated"]
            if "E" in built:
                band_donor, band_donor_info = br.calibrate(built["E"]["window"],
                                                           built["D"]["meta"],
                                                           v16["module"])
                built["K"] = dict(window=built["E"]["window"], calibrated=band_donor,
                                  check=built["D"]["check"],
                                  calibration=check_calibration(
                                      band_donor, built["E"]["window"],
                                      built["D"]["meta"], band_donor_info, V16_DONOR))
                windows[V16_DONOR] = built["K"]["calibrated"]
        # Inference-time graft-cap arms: V13's weights and window, only the bound the
        # graft residual may spend changes. A uniform sweep first; then the
        # relevance-proportional cap, which gives the graft slots that really bridge to
        # the retained blocks the full 0.50 and squeezes the quota-filler slots to 0.25.
        cap_arms = {}
        if getattr(args, "graft_scales", None) or getattr(args, "relevance_cap", False):
            if "D" not in built or v11_evict is None:
                raise ValueError("the graft-cap arms need the V13 window and module")
            relevance = None
            if getattr(args, "relevance_cap", False):
                relevance = slot_relevance(built["D"]["window"], built["D"]["meta"],
                                           k=v11_evict["module"].k_neighbors)
            for scale in getattr(args, "graft_scales", []):
                arm = cap_arm(scale)
                capped = copy.deepcopy(v11_evict["module"])
                capped.set_graft_cap(float(scale))
                capped_window, capped_info = calibrate(built["D"]["window"],
                                                       built["D"]["meta"], capped)
                built[arm] = dict(window=built["D"]["window"], calibrated=capped_window,
                                  check=built["D"]["check"],
                                  calibration=check_calibration(
                                      capped_window, built["D"]["window"],
                                      built["D"]["meta"], capped_info, arm))
                cap_arms[arm] = dict(kind="uniform", graft_cap=float(scale))
                windows[arm] = built[arm]["calibrated"]
                del capped, capped_window
            if relevance is not None:
                arm = "V13RelevanceCap"
                capped = copy.deepcopy(v11_evict["module"])
                caps = relevance_graft_caps(relevance, built["D"]["meta"]["slot_type"])
                capped.set_graft_cap(caps)
                capped_window, capped_info = calibrate(built["D"]["window"],
                                                       built["D"]["meta"], capped)
                built[arm] = dict(window=built["D"]["window"], calibrated=capped_window,
                                  check=built["D"]["check"],
                                  calibration=check_calibration(
                                      capped_window, built["D"]["window"],
                                      built["D"]["meta"], capped_info, arm))
                graft = (built["D"]["meta"]["slot_type"] > 0.5)
                cap_arms[arm] = dict(kind="relevance",
                                     graft_cap_min=float(caps[graft].min()),
                                     graft_cap_max=float(caps[graft].max()),
                                     graft_cap_mean=float(caps[graft].mean()),
                                     relevance_mean=float(relevance[graft].mean()),
                                     relevance_std=float(relevance[graft].std()))
                windows[arm] = built[arm]["calibrated"]
                del capped, capped_window, caps
        # Neighbour-choice arms: V13's window, weights and projection with the four
        # neighbours picked at random / from the least similar end instead of the top.
        neighbor_arms = {}
        modes = [str(mode) for mode in (getattr(args, "v13_neighbor_modes", None) or [])]
        if modes:
            if "D" not in built or v11_evict is None:
                raise ValueError("the neighbour arms need the V13 window and module")
            for mode in modes:
                if mode not in NEIGHBOR_ARMS:
                    raise ValueError(f"unknown neighbour mode {mode!r}")
                arm = NEIGHBOR_ARMS[mode]
                swapped, swap_report = cross_block_context(
                    built["D"]["window"], built["D"]["meta"],
                    k=v11_evict["module"].k_neighbors, return_report=True,
                    selection=mode, seed=args.seed,
                    span_value=v11_evict["module"].span_value,
                    context_diffusion=v11_evict["module"].context_diffusion,
                    csls=v11_evict["module"].csls)
                swapped_window, swapped_info = calibrate(
                    built["D"]["window"], built["D"]["meta"], v11_evict["module"],
                    context=swapped)
                swapped_info["neighbour_swap"] = swap_report
                built[arm] = dict(window=built["D"]["window"],
                                  calibrated=swapped_window,
                                  check=built["D"]["check"],
                                  calibration=check_calibration(
                                      swapped_window, built["D"]["window"],
                                      built["D"]["meta"], swapped_info, arm))
                neighbor_arms[arm] = swap_report
                windows[arm] = built[arm]["calibrated"]
                del swapped, swapped_window
        # Permutation arms: the SAME window, the SAME trained module and the SAME
        # self-calibration branch as arm D; only the mapping slot -> context row is
        # shuffled, inside (source_block, slot_type) groups so the batch a slot draws
        # from is untouched. Nothing is retrained and nothing is re-selected.
        permuted_arms = {}
        seeds = list(getattr(args, "permute_seeds", []) or [])
        if seeds:
            if "D" not in built or v11_evict is None:
                raise ValueError("the permutation arms need the V13 window and module")
            base_context, _ = v11_evict["module"].neighbours(built["D"]["window"],
                                                             built["D"]["meta"],
                                                             return_report=True)
            for seed in seeds:
                arm = permute_arm(seed)
                moved, move_info = permute_context(base_context, built["D"]["meta"], seed)
                if torch.equal(moved, base_context):
                    raise ValueError(f"{arm}: the permutation left every context row in "
                                     f"place, so this arm is arm D under another name")
                arm_window, arm_info = calibrate(built["D"]["window"], built["D"]["meta"],
                                                 v11_evict["module"], context=moved)
                built[arm] = dict(window=built["D"]["window"], calibrated=arm_window,
                                  check=built["D"]["check"],
                                  calibration=check_calibration(
                                      arm_window, built["D"]["window"], built["D"]["meta"],
                                      arm_info, arm))
                permuted_arms[arm] = move_info["moved_fraction"]
                windows[arm] = built[arm]["calibrated"]
                del moved, arm_window
            del base_context
        # Arms F and G: the SAME windows as D and E, calibrated by the complement-mode
        # module. Both modules see the identical slot vectors, so the difference between
        # V13 and V14 is the neighbour rule and nothing else -- not the window, not the
        # graft content, not the number of parameters that moved.
        if v14 is not None:
            if "D" not in built:
                raise ValueError("--v14 needs the cross-block window it has to share")
            v14_calibrated, v14_info = calibrate(built["D"]["window"], built["D"]["meta"],
                                                 v14["module"])
            built["F"] = dict(window=built["D"]["window"], calibrated=v14_calibrated,
                              check=built["D"]["check"],
                              calibration=check_calibration(
                                  v14_calibrated, built["D"]["window"],
                                  built["D"]["meta"], v14_info, V14))
            windows[V14] = built["F"]["calibrated"]
            if "E" in built:
                v14_donor, v14_donor_info = calibrate(built["E"]["window"],
                                                      built["D"]["meta"], v14["module"])
                built["G"] = dict(window=built["E"]["window"],
                                  calibrated=v14_donor,
                                  check=built["D"]["check"],
                                  calibration=check_calibration(
                                      v14_donor, built["E"]["window"], built["D"]["meta"],
                                      v14_donor_info, V14_DONOR))
                windows[V14_DONOR] = built["G"]["calibrated"]
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

    # The permutation arms are named after their seed and are not in ARMS; every arm that
    # has a window must still be read out, or the arm silently disappears from the table.
    ordered = list(ARMS) + [arm for arm in windows if arm not in ARMS]
    answered = {arm: gr.readout(model, tokenizer, prepared["query_ids"], answers,
                                args.f1_threshold, prepare(arm))
                for arm in ordered if arm in windows or arm in (FULL, SELECTION)}
    gr.restore(model, manager, state)

    if int(fingerprints[FULL + "/slots"]) != args.window_blocks * manager.block:
        raise ValueError("the Full arm did not keep the whole window")
    for arm in ARMS:
        if arm in (FULL, SELECTION):
            continue
        if arm not in windows:
            if (arm in (V11_DONOR, V14, V14_DONOR, V15, V15_DONOR, V16, V16_DONOR,
                        V13_PLUS, V13_PLUS_DONOR)
                    or arm.startswith(PERMUTE_PREFIX)):
                # A run without the donor document (or without --v14) has no such arm;
                # it is simply not answered, and `arms_present` records that.
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
    if V14 in windows and fingerprints[V14] == fingerprints[V11_EVICT]:
        raise ValueError("V14 reads the same pool as V13: the complement neighbour rule "
                         "changed not one slot, so this run cannot speak about it")
    if len(prepared) > 1 and V14 in windows and V14_DONOR not in windows:
        raise ValueError("V14's donor arm was not built: the complement arm's content "
                         "causality cannot be answered by V13's donor")
    if V14_DONOR in windows and fingerprints[V14] == fingerprints[V14_DONOR]:
        raise ValueError("V14's donor arm reads the same pool as V14: the swap did not "
                         "reach the memory")
    if V13_PLUS in windows and fingerprints[V13_PLUS] == fingerprints[V11_EVICT]:
        raise ValueError("V13Plus reads the same pool as V13: the new context rule "
                         "changed nothing")
    if V16 in windows and fingerprints[V16] == fingerprints[V11_EVICT]:
        raise ValueError("V16 reads the same pool as V13: the band injection changed "
                         "nothing, so this run cannot speak about it")
    if len(prepared) > 1 and V16 in windows and V16_DONOR not in windows:
        raise ValueError("V16's donor arm was not built: its content causality cannot be "
                         "answered by V13's donor")
    if V15 in windows and fingerprints[V15] == fingerprints[V11_EVICT]:
        raise ValueError("V15 reads the same pool as V13: the relation module changed "
                         "nothing, so this run cannot speak about it")
    if len(prepared) > 1 and V15 in windows and V15_DONOR not in windows:
        raise ValueError("V15's donor arm was not built: its content causality cannot be "
                         "answered by V13's donor")
    for arm in cap_arms:
        if fingerprints[V11_EVICT] == fingerprints[arm]:
            raise ValueError(f"{arm} reads the same pool as V13: the cap changed nothing")
    for arm in neighbor_arms:
        if fingerprints[V11_EVICT] == fingerprints[arm]:
            raise ValueError(f"{arm} reads the same pool as V13: the neighbour swap "
                             f"changed nothing")
    for arm in permuted_arms:
        if fingerprints[V11_EVICT] == fingerprints[arm]:
            raise ValueError(f"{arm} reads the same pool as V13: the permutation changed "
                             f"no answer, which is not the same as it changing no slot")
        if int(fingerprints[arm + "/slots"]) != TARGET_SLOTS:
            raise ValueError(f"{arm} does not hold the {TARGET_SLOTS}-slot budget")

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
        v14=(dict(window_check=built["F"]["check"],
                  calibration=built["F"]["calibration"]) if "F" in built else None),
        v14_donor=(dict(calibration=built["G"]["calibration"], source_id=donor_id)
                   if "G" in built else None),
        v15=(dict(window_check=built["H"]["check"],
                  calibration=built["H"]["calibration"]) if "H" in built else None),
        v15_donor=(dict(calibration=built["I"]["calibration"], source_id=donor_id)
                   if "I" in built else None),
        v13_plus=(dict(window_check=built["L"]["check"],
                       calibration=built["L"]["calibration"]) if "L" in built else None),
        v13_plus_donor=(dict(calibration=built["M"]["calibration"], source_id=donor_id)
                        if "M" in built else None),
        v16=(dict(window_check=built["J"]["check"],
                  calibration=built["J"]["calibration"]) if "J" in built else None),
        v16_donor=(dict(calibration=built["K"]["calibration"], source_id=donor_id)
                   if "K" in built else None),
        graft_cap={arm: dict(info=info, calibration=built[arm]["calibration"])
                   for arm, info in cap_arms.items()} or None,
        neighbor_swap={arm: dict(calibration=built[arm]["calibration"],
                                 report=report)
                       for arm, report in neighbor_arms.items()} or None,
        permutation={arm: dict(calibration=built[arm]["calibration"],
                               moved_fraction=moved)
                     for arm, moved in permuted_arms.items()} or None,
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


def _neighbour_composition(samples, key):
    """Per-row neighbour diagnostics of one arm's calibration record, aggregated.

    This is the block V13 never had: without it, "the top-4 neighbours are all carriers"
    is a hypothesis about the mechanism, not a measurement. `cross_weight_mass_mean` is
    the one that catches a complement arm which selected two of each kind but let the
    own-kind side keep all the weight.
    """
    reports = [(sample.get(key) or {}).get("calibration") or {} for sample in samples]
    reports = [row.get("cross_context") for row in reports]
    reports = [report for report in reports if report]
    if not reports:
        return None
    fields = ("mode", "k", "neighbour_similarity_mean", "neighbour_block_spread_mean",
              "context_rms", "neighbour_graft_fraction", "neighbour_type_cross_fraction",
              "core_to_graft_neighbour_fraction", "graft_to_core_neighbour_fraction",
              "cross_weight_mass_mean", "own_kind_weight_mass_mean")
    summary = {field: _stats([report[field] for report in reports if field in report])
               for field in fields if field not in ("mode", "k")}
    summary["mode"] = sorted({str(report.get("mode")) for report in reports})
    summary["k"] = sorted({int(report.get("k")) for report in reports if "k" in report})
    summary["n"] = len(reports)
    return summary


def permutation_report(graded, threshold, seeds):
    """The three numbers the permutation test exists to produce, per seed.

    `rescued_by_v13`  rows V13 answers and the plain calibration does not;
    `then_lost`       of those, the rows the permuted arm answers WRONG -- the direct
                      question "does V13's gain survive when the pairing is destroyed";
    `damage`          rows Full answers that the permuted arm does not, which is where a
                      redistribution would show up.
    """
    paired = [sample for sample in graded
              if V11 in sample.get("arms", {}) and V11_EVICT in sample.get("arms", {})]

    def correct(sample, arm):
        return gr._correct(sample, arm, threshold)

    rescued = [sample for sample in paired if correct(sample, V11_EVICT)
               and not correct(sample, V11)]
    arms = {}
    for seed in seeds:
        arm = permute_arm(seed)
        rows = [sample for sample in paired if arm in sample.get("arms", {})]
        if not rows:
            continue
        moved = [sample["permutation"][arm]["moved_fraction"] for sample in rows
                 if sample.get("permutation") and arm in sample["permutation"]]
        arms[arm] = dict(
            seed=int(seed),
            correct=int(sum(1 for sample in rows if correct(sample, arm))),
            accuracy=round(100.0 * sum(1 for sample in rows if correct(sample, arm))
                           / len(rows), 2),
            versus_v13=er.pair_summary(rows, arm, V11_EVICT, threshold),
            versus_plain=er.pair_summary(rows, arm, V11, threshold),
            rescued_by_v13=len(rescued),
            rescued_by_v13_then_lost=int(sum(1 for sample in rescued
                                             if not correct(sample, arm))),
            changed_answers=int(sum(1 for sample in rows
                                    if sample["arms"][arm]["prediction"]
                                    != sample["arms"][V11_EVICT]["prediction"])),
            damage=ef._damage(rows, arm, threshold),
            context_moved_fraction=_stats(moved),
            calibration=dict(
                core_relative_delta=_stats([
                    sample["permutation"][arm]["calibration"]["core_relative_delta"]
                    for sample in rows]),
                graft_relative_delta=_stats([
                    sample["permutation"][arm]["calibration"]["graft_relative_delta"]
                    for sample in rows])))
    return dict(
        definition=("inference-time only: the trained V13 module, its weights, the window "
                    "and the self-calibration branch are unchanged; the precomputed "
                    "cross-block context rows are shuffled inside (source_block, "
                    "slot_type) groups, so every slot draws from the same batch of "
                    "contexts as before and only WHICH row it receives changes"),
        seeds=[int(seed) for seed in seeds], arms=arms,
        original=dict(arm=V11_EVICT,
                      correct=int(sum(1 for sample in paired if correct(sample, V11_EVICT))),
                      rescued_by_v13=len(rescued), damage=ef._damage(
                          paired, V11_EVICT, threshold)),
        plain=dict(arm=V11,
                   correct=int(sum(1 for sample in paired if correct(sample, V11)))),
        reading=("a stable drop of 4-6 questions means the exact pairing carries "
                 "information; 0-2 means this run has no evidence that it does. Neither "
                 "is a significance test, and neither shows that the model performs "
                 "cross-block multi-hop reasoning"))


def _band_gate(samples):
    """V16's per-band gate distribution and the injected RMS, aggregated over rows."""
    rows = [((sample.get("v16") or {}).get("calibration") or {}) for sample in samples]
    rows = [row for row in rows if row]
    if not rows:
        return None
    bands = sorted({band for row in rows for band in (row.get("gate_bands") or {})})
    summary = {}
    for band in bands:
        entry = {}
        for field in ("gate_core_mean", "gate_core_std", "gate_graft_mean",
                      "gate_graft_std"):
            entry[field] = _stats([row["gate_bands"][band][field] for row in rows
                                   if band in row.get("gate_bands", {})
                                   and field in row["gate_bands"][band]])
        summary[band] = entry
    return dict(
        n=len(rows), bands=summary,
        band_proj_norm={band: _stats([row["band_proj_norm"][band] for row in rows
                                      if row.get("band_proj_norm")])
                        for band in range(len(rows[0].get("band_proj_norm") or []))},
        layer_proj_norm=_stats([row["layer_proj_norm"] for row in rows
                                if row.get("layer_proj_norm") is not None]),
        cross_injection_rms=_stats([row["cross_injection_rms"] for row in rows
                                    if row.get("cross_injection_rms") is not None]),
        shallow_cross_leak=_stats([row["shallow_cross_leak"] for row in rows
                                   if row.get("shallow_cross_leak") is not None]),
        core_relative_delta=_stats([row["core_relative_delta"] for row in rows
                                    if row.get("core_relative_delta") is not None]),
        graft_relative_delta=_stats([row["graft_relative_delta"] for row in rows
                                     if row.get("graft_relative_delta") is not None]),
        note="gate is omega*eta applied after the projection; shallow_cross_leak is the "
             "largest cross-term value on memory layers 0..9 and must be exactly 0")


def _relation_gate(samples):
    """Per-layer sink occupancy and retained mass of V15, aggregated over rows.

    This is the number the design turns on: a sink that never closes leaves the retained
    mass at 1 and the gate is decorative, whatever the loss did.
    """
    reports = [((sample.get("v15") or {}).get("calibration") or {}).get("relation")
               for sample in samples]
    reports = [report for report in reports if report]
    if not reports:
        return None
    layers = sorted({layer for report in reports
                     for layer in (report.get("per_layer") or {})})
    fields = ("retained_mass", "sink_mass", "neighbour_similarity_mean", "feature_rms")
    return dict(
        n=len(reports),
        retained_mass_mean=_stats([report["retained_mass_mean"] for report in reports
                                   if "retained_mass_mean" in report]),
        sink_mass_mean=_stats([report["sink_mass_mean"] for report in reports
                               if "sink_mass_mean" in report]),
        tau=reports[0].get("tau"), sink_similarity=reports[0].get("sink_similarity"),
        per_layer={layer: {field: _stats(
            [report["per_layer"][layer][field] for report in reports
             if layer in (report.get("per_layer") or {})
             and field in report["per_layer"][layer]])
            for field in fields} for layer in layers},
        note="retained mass is the share of the context weight the sink left for "
             "injection (<= 1 by construction); a value near 1 on every layer means the "
             "gate is decorative and the run is V13 with extra parameters")


def joint_compare(samples, threshold):
    graded = [sample for sample in samples if sample.get("status") == "graded"]
    located = [sample for sample in graded if sample.get("evidence", {}).get("located")]
    dropped = [sample for sample in located if _evidence_in_dropped(sample)]
    kept = [sample for sample in located if not _evidence_in_dropped(sample)]
    present = [arm for arm in ARMS if arm in graded[0]["arms"]]
    # The permutation arms are named after their seed, so they are not in the static
    # ARMS tuple; they must still be summarised, damaged and paired like every other arm.
    extra = sorted({arm for sample in graded for arm in sample["arms"]} - set(ARMS))
    present += [arm for arm in extra if arm in graded[0]["arms"]]
    controls = [arm for arm in (V11_DONOR, V11_EVICT, V11_EVICT_DONOR, V14, V14_DONOR,
                                V15, V15_DONOR, V16, V16_DONOR, V13_PLUS,
                                V13_PLUS_DONOR, GRAFT) + tuple(extra)
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
            if V11_EVICT_DONOR in present else {},
            # The one comparison this round exists for: the same window, the same
            # modules' training budget, only the neighbour source differs.
            versus_v13={arm: er.pair_summary(rows, arm, V11_EVICT, threshold)
                        for arm in (V14, V15, V16, V13_PLUS) if arm in present},
            versus_v14={V14_DONOR: er.pair_summary(rows, V14_DONOR, V14, threshold)}
            if V14_DONOR in present else {},
            versus_v15={V15_DONOR: er.pair_summary(rows, V15_DONOR, V15, threshold)}
            if V15_DONOR in present else {},
            versus_v16={V16_DONOR: er.pair_summary(rows, V16_DONOR, V16, threshold)}
            if V16_DONOR in present else {},
            versus_v13_plus={V13_PLUS_DONOR: er.pair_summary(rows, V13_PLUS_DONOR,
                                                             V13_PLUS, threshold)}
            if V13_PLUS_DONOR in present else {})

    merge_rows = [sample["v11"] for sample in graded]
    thresholds = dict(
        total=dict(target=62, baseline=_count(graded, ANCHOR_ONLY, threshold),
                   achieved=_count(graded, V11, threshold),
                   # `achieved` above keeps the V11 driver's meaning (the plain
                   # calibration); the treatment counts live here so a V13/V14 run does
                   # not silently report 61 as its own result.
                   achieved_by_arm={arm: _count(graded, arm, threshold)
                                    for arm in (V11, V11_EVICT, V14) if arm in present}),
        dropped_evidence=dict(target=19, n=len(dropped),
                              baseline=_count(dropped, ANCHOR_ONLY, threshold),
                              achieved=_count(dropped, V11, threshold),
                              achieved_by_arm={arm: _count(dropped, arm, threshold)
                                               for arm in (V11, V11_EVICT, V14)
                                               if arm in present}),
        kept_evidence=dict(target=12, n=len(kept),
                           baseline=_count(kept, ANCHOR_ONLY, threshold),
                           achieved=_count(kept, V11, threshold),
                           achieved_by_arm={arm: _count(kept, arm, threshold)
                                            for arm in (V11, V11_EVICT, V14)
                                            if arm in present}),
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
            if V11_EVICT_DONOR in present else None),
        # V14's own three bars, stated before the run: complement over top-k, content
        # causality, and a donor that must not win.
        complement_over_topk=dict(
            target=4,
            achieved=er.pair_summary(graded, V14, V11_EVICT, threshold).get("net")
            if V14 in present else None),
        v14_content_causality=dict(
            target=5,
            achieved=er.pair_summary(graded, V14, V14_DONOR, threshold).get("net")
            if V14_DONOR in present else None),
        # The V15 targets, stated before the run: +6 questions over V13 (73/158) and
        # damage below 8 of Full's 47.
        v15_over_v13=dict(
            target=6,
            achieved=er.pair_summary(graded, V15, V11_EVICT, threshold).get("net")
            if V15 in present else None),
        v15_content_causality=dict(
            target=5,
            achieved=er.pair_summary(graded, V15, V15_DONOR, threshold).get("net")
            if V15_DONOR in present else None),
        v15_damage=dict(
            target=8,
            achieved=(ef._damage(graded, V15, threshold).get("lost")
                      if V15 in present else None)),
        # V16's targets, stated before the run: +5 questions over V13 (72/158), damage
        # down to 8-9 of Full's 47, and content causality kept.
        v16_over_v13=dict(
            target=5,
            achieved=er.pair_summary(graded, V16, V11_EVICT, threshold).get("net")
            if V16 in present else None),
        v16_content_causality=dict(
            target=5,
            achieved=er.pair_summary(graded, V16, V16_DONOR, threshold).get("net")
            if V16_DONOR in present else None),
        v13_plus_over_v13=dict(
            target=5,
            achieved=er.pair_summary(graded, V13_PLUS, V11_EVICT, threshold).get("net")
            if V13_PLUS in present else None),
        v13_plus_content_causality=dict(
            target=5,
            achieved=er.pair_summary(graded, V13_PLUS, V13_PLUS_DONOR,
                                     threshold).get("net")
            if V13_PLUS_DONOR in present else None),
        v13_plus_damage=dict(
            target=9,
            achieved=(ef._damage(graded, V13_PLUS, threshold).get("lost")
                      if V13_PLUS in present else None)),
        v16_damage=dict(
            target=9,
            achieved=(ef._damage(graded, V16, threshold).get("lost")
                      if V16 in present else None)))

    return dict(
        definition=("the same 19-block window, the same nine carriers, the same 2,304 "
                    "slots, the same 158 rows; each arm moves one variable: the "
                    "selection rule, the capacity split, the calibration, the neighbour "
                    "source, or whose content the graft slots hold"),
        arms_present=present,
        n=len(graded),
        arms={arm: gr._arm_summary(graded, arm, threshold) for arm in present},
        total=block(graded), dropped_evidence_group=block(dropped),
        kept_evidence_group=block(kept),
        damage={arm: ef._damage(graded, arm, threshold) for arm in present},
        permutation=permutation_report(
            graded, threshold,
            [int(arm[len(PERMUTE_PREFIX):]) for arm in present
             if arm.startswith(PERMUTE_PREFIX)]) if any(
                arm.startswith(PERMUTE_PREFIX) for arm in present) else None,
        relation_gate=_relation_gate(graded),
        band_gate=_band_gate(graded),
        neighbour_composition=dict(
            topk=_neighbour_composition(graded, "evict"),
            complement=_neighbour_composition(graded, "v14"),
            note="topk is V13's kind-blind top-4; complement is V14's 2 + 2. "
                 "neighbour_type_cross_fraction is the share of selected neighbours "
                 "from the other side of the core/graft divide, and cross_weight_mass "
                 "is the share of context weight they received"),
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
def build_protocol(args, task, anchor_config, v11_geometry, evict_geometry,
                   v14_geometry=None, permute_seeds=None, v15_geometry=None,
                   v16_geometry=None):
    return dict(
        experiment="10 (V14): does the SOURCE of the cross-block neighbours buy anything? "
                   "Same window, same budget, same training as V13; only the rule that "
                   "picks the 4 neighbours changes -- kind-blind top-k versus 2 of the "
                   "query's own core/graft kind + 2 from the other side.",
        treatment=("every slot is calibrated against a sparse set of OTHER blocks' slots: "
                   "V13 takes the k = 4 most associated by reader-key cosine; V14 takes "
                   "k_same = 2 of the query's own kind and k_cross = 2 from the other "
                   "side of the core/graft divide, softmaxed per group with equal group "
                   "mass. The context vector is injected into the residual bottleneck by "
                   "a zero-initialised projection; the slot itself is never blended, and "
                   "the same bounded low-rank map moves it, with a tighter ceiling on "
                   "carrier slots (0.15 x rms) than on graft slots (0.50 x rms)"),
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
        band_relation=dict(
            arm=V16, path=args.v16, sha256=sha(args.v16) if args.v16 else None,
            geometry=v16_geometry,
            treatment=("one cross-block top-4 candidate set chosen from the five-layer "
                       "CONSENSUS similarity (keys centred over slots, L2-normalised); "
                       "each band then softmaxes its own similarity over those four and "
                       "aggregates its own raw values; memory layer l reads band "
                       "BAND_OF_LAYER[l] only (L12->L10-14, L16->L15-18, L20->L19-22, "
                       "L24->L23-26, L28->L27-31, L0-9 self branch only), through "
                       "band_proj[band] + layer_proj[layer], both zero-initialised; the "
                       "injection is gated AFTER the projection by "
                       "omega = sigmoid(kappa[type,band]*(margin - tau[type,band])) and "
                       "eta = sigmoid(eta_logit[type,layer]), with nothing normalising "
                       "the product afterwards"),
            targets=dict(questions="V16 >= 72/158 (+5 over V13)",
                         damage="8-9 of Full's 47 lost",
                         content="V16 - V16Donor keeps a clear margin")),
        relation=dict(
            arm=V15, path=args.v15, sha256=sha(args.v15) if args.v15 else None,
            geometry=v15_geometry,
            treatment=("five INDEPENDENT top-4 relations, one per utility layer "
                       "(12/16/20/24/28), each softmaxed with a per-layer sink logit that "
                       "is dropped afterwards so the retained mass can be below 1; the "
                       "feature is the neighbours minus the slot (sum_j w_ij v_j - m_i "
                       "v_i); it enters as a FiLM modulation, b * (1 + tanh(W_s c)) + "
                       "tanh(W_h c), on an RMS-normalised sum of five per-layer "
                       "projections; memory layers 0..9 keep the self branch only"),
            targets=dict(questions="V15 >= 73/158 (+6 over V13)",
                         damage="at most 8 of Full's 47 lost")),
        permutation=dict(
            arms=[permute_arm(seed) for seed in (permute_seeds or [])],
            seeds=list(permute_seeds or []),
            treatment=("inference time only, no retraining: V13's precomputed "
                       "cross-block context rows are shuffled inside (source_block, "
                       "slot_type) groups, so each slot draws from the same batch of "
                       "neighbours it could draw from before and only the slot -> row "
                       "pairing changes"),
            not_done=("random top-4 neighbours, which would change the marginal "
                      "distribution of the context as well as the pairing")),
        controls=dict(
            donor=dict(arm=V11_DONOR,
                       definition="the plain calibration's window with the 684 graft "
                                  "slots taken from the previous row's document"),
            cross_donor=dict(arm=V11_EVICT_DONOR,
                             definition="V13's window with the 684 graft slots taken "
                                        "from the previous row's document: the only "
                                        "thing that changes is whose content the "
                                        "calibrator is conditioned on"),
            complement=dict(arm=V14, path=args.v14,
                            sha256=sha(args.v14) if args.v14 else None,
                            geometry=v14_geometry,
                            definition="V13's window calibrated by the complement-mode "
                                       "module: identical slots, identical graft "
                                       "content, only the neighbour source differs"),
            complement_donor=dict(arm=V14_DONOR,
                                  definition="V14 on the donor window: the complement "
                                             "arm's own content-causality control"),
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
            donor="the donor arm's window must differ from the true one",
            neighbour_source="--v14 must be complement mode and --v11-evict top-k mode "
                             "with the same k, so the two arms differ in the neighbour "
                             "source alone; V14's calibrated window must differ from "
                             "V13's",
            permutation="the permuted arms must change the cross-block context (a "
                        "permutation that moves nothing is refused), must differ from "
                        "V13 in answers or the arm is refused, and must still hold the "
                        "2,304-slot budget; the self-calibration branch and every weight "
                        "stay untouched"),
        thresholds=dict(
            total="V11 >= 62/158 against 56", dropped_evidence="V11 >= 19/40 against 15",
            kept_evidence="V11 >= 12/34 against 13",
            damage="correct rows lost by Full <= 13",
            content_causality="V11 - V11Donor >= 5 questions",
            complement_over_topk="V14 - V13 >= 4 questions net",
            v14_content_causality="V14 - V14Donor >= 5 questions"),
        screening_caveat=gr.SCREENING_CAVEAT, f1_threshold=args.f1_threshold,
        seed=args.seed, query=gr.QUERY_SUFFIX,
        scoring="official HotpotQA QA F1, F1 >= threshold is correct",
        reader_utility_schema=JOINT_SCHEMA)


def _net_summary(total):
    """Every paired comparison of the run as `arm -> net`, duplicates collapsed."""
    net = {}
    for name in ("versus_anchor_only", "versus_v11", "versus_v11evict", "versus_v13",
                 "versus_v14", "versus_v15", "versus_v16", "versus_v13_plus"):
        for arm, value in (total.get(name) or {}).items():
            net[arm] = value.get("net")
    return net


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
            arms=list(ARMS) + [permute_arm(seed) for seed in args.permute_seeds],
            readouts_per_sample=len(ARMS) + len(args.permute_seeds),
            v11=args.v11, v11_evict=args.v11_evict, v14=args.v14, v15=args.v15,
            v16=args.v16, v13_plus=args.v13_plus,
            permute_seeds=args.permute_seeds), indent=2,
            ensure_ascii=False))
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
    args.graft_scales = parse_scales(args.v13_graft_scales)
    args.relevance_cap = str(args.v13_relevance_cap or "").lower() in ("on", "true", "1")
    if (args.graft_scales or args.relevance_cap) and v11_evict is None:
        raise ValueError("--v13-graft-scales / --v13-relevance-cap need --v11-evict: the "
                         "ceiling belongs to the V13 checkpoint")
    args.v13_neighbor_modes = [mode.strip() for mode
                               in str(args.v13_neighbor_modes or "").split(",")
                               if mode.strip()]
    for mode in args.v13_neighbor_modes:
        if mode not in NEIGHBOR_ARMS:
            raise ValueError(f"--v13-neighbor-modes takes {sorted(NEIGHBOR_ARMS)}, "
                             f"got {mode!r}")
        if v11_evict is None:
            raise ValueError("--v13-neighbor-modes needs --v11-evict: the neighbours "
                             "belong to the V13 checkpoint")
    args.permute_seeds = parse_seeds(args.v13_permute_seeds)
    if args.permute_seeds:
        if v11_evict is None:
            raise ValueError("--v13-permute-seeds needs --v11-evict: the correspondence "
                             "that is being permuted belongs to the V13 checkpoint")
        if v11_evict["module"].neighbor_mode != "topk":
            raise ValueError("the permutation test is defined on V13's kind-blind top-k "
                             "context; this checkpoint carries neighbour mode "
                             f"{v11_evict['module'].neighbor_mode!r}")
    v13_plus = None
    if args.v13_plus:
        if v11_evict is None:
            raise ValueError("--v13-plus needs --v11-evict: that is the arm to beat")
        plus_module, plus_geometry, plus_config, _ = load_v11(args.v13_plus,
                                                              manager.hidden)
        plus_module = plus_module.to(model.memory.device).float()
        if not plus_geometry.get("cross"):
            raise ValueError("--v13-plus is not a cross-block conditioned checkpoint")
        for field in ("core_per_block", "graft_per_block"):
            if plus_geometry.get(field) != v11_evict["geometry"].get(field):
                raise ValueError(f"--v13-plus and --v11-evict record different {field}: "
                                 f"they cannot share one window")
        v13_plus = dict(module=plus_module, geometry=plus_geometry,
                        config=plus_config)
    v16 = None
    if args.v16:
        if v11_evict is None:
            raise ValueError("--v16 needs --v11-evict: V13 is the arm V16 has to beat")
        v16_module, v16_config, v16_geometry = br.load_band(args.v16, manager.hidden)
        v16_module = v16_module.to(model.memory.device).float().eval()
        v16_module.requires_grad_(False)
        for field in ("core_per_block", "graft_per_block"):
            if v16_geometry.get(field) != v11_evict["geometry"].get(field):
                raise ValueError(f"--v16 and --v11-evict record different {field}: they "
                                 f"cannot share one window")
        if int(v16_config.get("rank", -1)) != int(v11_evict["config"].get("rank", -1)):
            raise ValueError("--v16 and --v11-evict were built with different ranks")
        v16 = dict(module=v16_module, config=v16_config, geometry=v16_geometry)
    v15 = None
    if args.v15:
        if v11_evict is None:
            raise ValueError("--v15 needs --v11-evict: V13 is the arm V15 has to beat")
        v15_module, v15_config, v15_geometry = hr.load_relation(args.v15, manager.hidden)
        v15_module = v15_module.to(model.memory.device).float().eval()
        v15_module.requires_grad_(False)
        if not torch.isfinite(v15_module.sink_similarity).all():
            raise ValueError("--v15 carries no measured sink threshold: the checkpoint "
                             "was written before init_sink, and the gate it is supposed "
                             "to test would be undefined")
        for field in ("core_per_block", "graft_per_block"):
            if v15_geometry.get(field) != v11_evict["geometry"].get(field):
                raise ValueError(f"--v15 and --v11-evict record different {field}: they "
                                 f"cannot share one window")
        v15 = dict(module=v15_module, config=v15_config, geometry=v15_geometry)
    v14 = None
    if args.v14:
        if v11_evict is None:
            raise ValueError("--v14 needs --v11-evict: the top-k arm is the baseline the "
                             "complement arm is measured against")
        v14_module, v14_geometry, v14_config, _ = load_v11(args.v14, manager.hidden)
        v14_module = v14_module.to(model.memory.device).float()
        if not v14_geometry.get("cross"):
            raise ValueError("--v14 is not a cross-block conditioned checkpoint")
        if v14_module.neighbor_mode != "complement":
            raise ValueError(f"--v14 carries neighbour mode "
                             f"{v14_module.neighbor_mode!r}: a top-k checkpoint cannot "
                             f"test the complement rule")
        if v11_evict["module"].neighbor_mode != "topk":
            raise ValueError("--v11-evict carries neighbour mode "
                             f"{v11_evict['module'].neighbor_mode!r}: V13 and V14 must "
                             f"differ in the neighbour rule, so the reference arm has to "
                             f"be the kind-blind top-k one")
        if int(v14_module.k_neighbors) != int(v11_evict["module"].k_neighbors):
            raise ValueError(f"the two arms read {v14_module.k_neighbors} and "
                             f"{v11_evict['module'].k_neighbors} neighbours: the "
                             f"comparison has to hold k fixed")
        for field in ("core_per_block", "graft_per_block", "utility_layers"):
            if v14_geometry.get(field) != v11_evict["geometry"].get(field):
                raise ValueError(f"--v14 and --v11-evict record different {field}: they "
                                 f"cannot share one window")
        v14 = dict(module=v14_module, geometry=v14_geometry, config=v14_config)

    rows, prepared, excluded = gr.prepare_rows(args, tokenizer, facts)
    report = dict(
        schema=SCHEMA_RESULTS, status="running",
        counts=dict(rows=len(rows), eligible=len(prepared), excluded=len(excluded)),
        protocol=build_protocol(args, args.task, anchor_config, v11_geometry,
                                v11_evict["geometry"] if v11_evict else None,
                                v14["geometry"] if v14 else None,
                                permute_seeds=args.permute_seeds,
                                v15_geometry=(v15["geometry"] if v15 else None),
                                v16_geometry=(v16["geometry"] if v16 else None)),
        identity=dict(
            model_path=args.model_path, checkpoint=args.checkpoint,
            checkpoint_sha256=sha(args.checkpoint),
            checkpoint_code_hash_verified=(saved or {}).get("code_hash_verified"),
            anchor=args.anchor, anchor_sha256=sha(args.anchor),
            v11=args.v11, v11_sha256=sha(args.v11),
            v11_evict=args.v11_evict,
            v11_evict_sha256=sha(args.v11_evict) if args.v11_evict else None,
            v14=args.v14,
            v14_sha256=sha(args.v14) if args.v14 else None,
            v15=args.v15,
            v15_sha256=sha(args.v15) if args.v15 else None,
            v16=args.v16,
            v16_sha256=sha(args.v16) if args.v16 else None,
            data_dir=args.data_dir,
            data_sha256=sha(Path(args.data_dir) / (args.task + ".jsonl")),
            supporting_facts_glob=args.supporting_facts_glob,
            entrypoint_sha256=sha(__file__),
            window_module_sha256=sha(ROOT / "methods/prlmc/evict_window.py"),
            calibration_module_sha256=sha(ROOT / "methods/prlmc/cross_block_calibration.py"),
            utility_module_sha256=sha(ROOT / "methods/prlmc/reader_utility.py"),
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
                                     donor_graft_evict, donor_id, args, v14, v15, v16,
                                     v13_plus)
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
        arms={arm: round(summary["accuracy"], 2)
              for arm, summary in report["summary"]["arms"].items()},
        # Merging these with `dict(**a, **b)` crashed on V15's run: an arm that appears
        # in two comparison blocks (V11EvictDonor is in both versus_anchor_only and
        # versus_v11evict) is a duplicate keyword, and the whole run died AFTER writing
        # status=complete. Update the dict instead of unpacking it.
        net=_net_summary(report["summary"]["total"]),
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
                        "on --neighbor-mode topk): the V13 arm")
    p.add_argument("--v13-plus", dest="v13_plus", default=None,
                   help="a CrossBlockResidual checkpoint trained with span-value / "
                        "context-diffusion / csls (the V13 successor); calibrates the "
                        "SAME window as --v11-evict")
    p.add_argument("--v16", default=None,
                   help="the band-aligned, margin-gated module (trained with "
                        "--architecture v16): calibrates the SAME window as --v11-evict")
    p.add_argument("--v15", default=None,
                   help="the layered sink-gated relation module (trained with "
                        "--architecture v15): calibrates the SAME window as --v11-evict")
    p.add_argument("--v13-graft-scales", dest="v13_graft_scales", default=None,
                   help="inference-time graft ceiling sweep on the V13 checkpoint, e.g. "
                        "0.38,0.40,0.42: weights and window untouched, only the bound the "
                        "graft residual may spend changes")
    p.add_argument("--v13-relevance-cap", dest="v13_relevance_cap", default=None,
                   help="inference-time relevance-proportional graft cap: "
                        "0.25 + 0.25 * rank_norm(mean Top-4 cross-block similarity)")
    p.add_argument("--v13-neighbor-modes", dest="v13_neighbor_modes", default=None,
                   help="attribution arms for V13's neighbour choice, comma separated "
                        "from {random,bottom}: same weights, same window, same projection, "
                        "same number of neighbours, only WHICH cross-block slots are "
                        "picked changes (random = uniform over legal cross-block slots, "
                        "bottom = the least similar). Builds V13RandNeighbor / "
                        "V13BottomNeighbor")
    p.add_argument("--v13-permute-seeds", dest="v13_permute_seeds", default=None,
                   help="inference-time attribution test for V13's cross-block "
                        "correspondence: a comma separated seed list (e.g. 0,1,2) builds "
                        "one arm per seed that calibrates V13's window with the "
                        "precomputed context rows shuffled inside (source_block, "
                        "slot_type). Nothing is retrained")
    p.add_argument("--v14", default=None,
                   help="the complement-mode checkpoint (trained with --cross on "
                        "--neighbor-mode complement --k-same 2 --k-cross 2): the V14 arm, "
                        "which calibrates the SAME window as --v11-evict")
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
