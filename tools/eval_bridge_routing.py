#!/usr/bin/env python3
"""Experiment 8 (V12): bridging route + dynamic capacity, native transport only.

    Full            19 blocks, uncompressed                      reference ceiling
    Selection       9 carrier blocks, verbatim                   capacity baseline
    Graft           V10 native graft, 1,944 + 360                the arm AnchorOnly beat
    AnchorOnly      frozen AnchorResidual at 360 anchors         arm A  (the 56/158 baseline)
    V11NativeGraft  V11's fixed quota, native, no calibration    arm B
    Bridge          140 protected/carrier + 1,044 contested       arm C

The one question this round is allowed to ask
---------------------------------------------
Does a routing rule that (i) gives every carrier a floor instead of a fixed quota and
(ii) lets the whole window compete for the rest, put more of the evidence that only
lives in a DROPPED block back into the 2,304-slot window -- without costing the
questions the carriers already answer?

Net gain is `rescued - harmed` against each baseline, row by row, and the three
subsets are the ones every round has used: rows whose gold evidence lies entirely in
the dropped blocks, rows whose evidence survived in a carrier, and the rows the
uncompressed window already answers. A routing gain that only shows up as damage on
the third group is a redistribution, not a gain.

What arm B is, exactly
----------------------
V11's eviction window (`V11Evict`, 61/158) was 1,620 core + 684 graft slots built
from the reader's own scores and then passed through a trained calibration module.
This round removes the calibration -- the question is whether SELECTION still buys
anything -- so arm B is that window's selection rebuilt over native slots with no
trained parameters at all: per carrier, the top 180 by reader utility verbatim; then
684 slots from the ten dropped blocks by utility x novelty with MMR. Arm C differs
only in the allocation (140 floor + 1,044 contested globally) and in adding the
bridging term to the ranking. Everything else -- the nine carriers, the write, the
budget, the reader, the order -- is shared.

What this round does NOT settle
-------------------------------
There is no donor arm here, by instruction: three arms, no probes, no training. So a
positive result attributes to the whole routing bundle (floor + global competition +
bridging term), not to its content, and the report records that limit next to the
numbers instead of leaving it for the reader to notice.

What is checked before a number is read
---------------------------------------
* the four pre-existing arms must reproduce the reference report row for row
  (`--verify-against`);
* every compressed arm holds exactly 2,304 slots in global chronological order;
* every installed slot is a byte-for-byte copy of the source slot it claims to be --
  this round transports native vectors and nothing else, so the check is an equality,
  not a tolerance;
* arm C's floor is exactly `9 x 140`, every protected slot comes from its own carrier
  block, and `admitted_from_dropped + admitted_from_carriers == 1,044`;
* arm B and arm C never install the same pool.

Usage
-----
    python tools/eval_bridge_routing.py --dry-run \
        --data-dir /data/.../LongBench/extracted/data --model-path /dev/shm/s1d_reader_fp16

    python tools/eval_bridge_routing.py \
        --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
        --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
        --model-path /dev/shm/s1d_reader_fp16 \
        --checkpoint runs/natural_merge_v9/full_v10.pt \
        --anchor runs/anchor_residual/anchor_only_rank64.pt \
        --verify-against reports/anchor_fusion/eval_anchor_fusion_hotpotqa.json \
        --cuda-device 0 --out reports/bridge_routing/eval_v12_hotpotqa.json
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
from io_utils import atomic_json, digest, sha            # noqa: E402
from runtime import load_runtime, reset                   # noqa: E402
from methods.prlmc.anchor_residual import (               # noqa: E402
    SCHEMA as ANCHOR_SCHEMA, AnchorResidual,
    compact_window as compact_window_anchor)
from methods.prlmc.bridge_routing import (                # noqa: E402
    DEFAULT_BRIDGE_WEIGHT, DEFAULT_SEEDS_PER_BLOCK, SCHEMA as ROUTER_SCHEMA,
    BridgeRouter, FixedQuotaRouter)
from methods.prlmc.reader_utility import ReaderScorer, UTILITY_LAYERS  # noqa: E402
from methods.prlmc.source_compaction import split_source_anchors  # noqa: E402
from train_core_fusion import load_documents              # noqa: E402

BLOCK_TOKENS = gr.BLOCK_TOKENS
SLOTS_PER_BLOCK = gr.SLOTS_PER_BLOCK
TARGET_SLOTS = gr.TARGET_SLOTS
FULL, SELECTION, GRAFT = ef.FULL, ef.SELECTION, ef.GRAFT
ANCHOR_ONLY = "AnchorOnly"
V11_NATIVE = "V11NativeGraft"
BRIDGE = "Bridge"
ARMS = (FULL, SELECTION, GRAFT, ANCHOR_ONLY, V11_NATIVE, BRIDGE)
# The baseline quota, which is what the reference report's arms were built at.
BASELINE_QUOTA = dict(core_per_block=216, graft_per_block=40)
SCHEMA_RESULTS = "bridge_routing_results/1"


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


def build_probes(args, tokenizer):
    """The fixed query bank the utility score is computed against. No test data.

    Same recipe as every reader-aware round so far: the first `--probe-count`
    questions of a TRAINING task (`musique`, the same corpus the operator was trained
    on), encoded once before the first test row and reused for all 158. Nothing from
    HotpotQA enters this, and no reference answer does either.
    """
    if args.probe_text:
        ids = tokenizer(args.probe_text, add_special_tokens=False)["input_ids"]
        return [list(ids)], dict(kind="text", text=args.probe_text, count=1)
    pool, _ = load_documents(tokenizer, args.data_dir, args.probe_task, None)
    wanted = min(int(args.probe_count), len(pool))
    if wanted <= 0:
        raise ValueError("the reader-utility score needs at least one probe question")
    bank = [list(document["query_ids"]) for document in pool[:wanted]]
    return bank, dict(kind="training_questions", task=args.probe_task, count=len(bank),
                      documents=[document["id"] for document in pool[:wanted]])


def set_quota(merge, core_per_block, graft_per_block, block=SLOTS_PER_BLOCK):
    if core_per_block + graft_per_block != block:
        raise ValueError("the baseline quota does not fill a block")
    merge.core_slots_per_block = int(core_per_block)
    merge.graft_slots_per_block = int(graft_per_block)
    if (int(merge.core_slots_per_block) != int(core_per_block)
            or int(merge.graft_slots_per_block) != int(graft_per_block)):
        raise ValueError("the operator did not take the baseline quota")


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def digest_ids(ids):
    """One digest over the whole probe bank, so the report pins what was scored."""
    return digest([int(token) for sequence in ids for token in sequence])


def check_window(window, meta, telemetry, source, name):
    """Capacity, order, native provenance, and the allocation the arm claims."""
    capacity = int(window.shape[1])
    if capacity != TARGET_SLOTS:
        raise ValueError(f"{name}: {capacity} slots, expected {TARGET_SLOTS}")
    positions = meta["source_position"]
    if int(positions.numel()) != capacity:
        raise ValueError(f"{name}: one source position per slot")
    if int(torch.unique(positions).numel()) != capacity:
        raise ValueError(f"{name}: two slots claim the same source position")
    if not torch.equal(positions, torch.sort(positions).values):
        raise ValueError(f"{name}: the output is not in global chronological order")
    copied = source.index_select(1, positions.to(source.device))
    if not torch.equal(window, copied):
        raise ValueError(f"{name}: a slot is not a verbatim copy of its source, so the "
                         f"transport is not native")
    if not bool(torch.isfinite(window).all()):
        raise ValueError(f"{name}: the window is not finite")

    carriers = set(int(block) for block in telemetry["carrier_blocks"])
    dropped = set(int(block) for block in telemetry["dropped_blocks"])
    protected = meta["protected"]
    graft = meta["slot_type"] == 1
    if set(int(x) for x in meta["source_block"][protected].tolist()) - carriers:
        raise ValueError(f"{name}: a protected slot came from a dropped block")
    if set(int(x) for x in meta["source_block"][graft].tolist()) - dropped:
        raise ValueError(f"{name}: a graft slot came from a carrier block")
    if int(protected.sum()) != int(telemetry["protected_total"]):
        raise ValueError(f"{name}: {int(protected.sum())} protected slots, telemetry says "
                         f"{telemetry['protected_total']}")
    admitted = int((~protected).sum())
    if admitted != int(telemetry["contested_total"]):
        raise ValueError(f"{name}: {admitted} admitted slots, telemetry says "
                         f"{telemetry['contested_total']}")
    if int(graft.sum()) != int(telemetry["admitted_from_dropped"]):
        raise ValueError(f"{name}: {int(graft.sum())} dropped-block slots, telemetry says "
                         f"{telemetry['admitted_from_dropped']}")
    if (int(telemetry["admitted_from_dropped"])
            + int(telemetry["admitted_from_carriers"]) != admitted):
        raise ValueError(f"{name}: the admitted split does not add up")
    if sum(telemetry["admitted_per_dropped_block"]) != int(graft.sum()):
        raise ValueError(f"{name}: the per-block admitted counts do not add up")
    return dict(
        total=capacity, protected=int(protected.sum()), admitted=admitted,
        admitted_from_dropped=int(graft.sum()),
        admitted_from_carriers=admitted - int(graft.sum()),
        admitted_share_from_dropped=round(100.0 * int(graft.sum()) / max(admitted, 1), 2),
        protected_per_carrier_block=[int((meta["source_block"][protected] == block).sum())
                                     for block in telemetry["carrier_blocks"]],
        admitted_per_carrier_block=list(telemetry["admitted_per_carrier_block"]),
        admitted_per_dropped_block=list(telemetry["admitted_per_dropped_block"]),
        empty_dropped_blocks=int(telemetry["empty_dropped_blocks"]),
        min_per_dropped_block=int(telemetry["min_per_dropped_block"]),
        max_per_dropped_block=int(telemetry["max_per_dropped_block"]),
        bridge_available=bool(telemetry["bridge_available"]),
        bridge_utility_correlation=telemetry.get("bridge_utility_correlation"),
        bridge_admitted_mean=telemetry.get("bridge_admitted_mean"),
        bridge_rejected_mean=telemetry.get("bridge_rejected_mean"),
        check=True)


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
def _chunks(prepared, window_blocks):
    chunks = [prepared["write_ids"][index:index + BLOCK_TOKENS]
              for index in range(0, len(prepared["write_ids"]), BLOCK_TOKENS)]
    return chunks[:window_blocks]


def run_sample(model, tokenizer, manager, merge, anchor_module, fixed, bridge, prepared,
               args):
    """One write, four windows, six answers."""
    chunks = _chunks(prepared, args.window_blocks)
    answers = prepared["answers"]
    fingerprints, windows, built = {}, {}, {}
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
        # Arms B and C: two windows from ONE reader reading of the same source.
        reading = fixed.read(source)
        for name, router in ((V11_NATIVE, fixed), (BRIDGE, bridge)):
            window, meta, info = router.build(source, picked, manager.block,
                                              reading=reading)
            built[name] = dict(check=check_window(window, meta, info, source, name),
                               telemetry=dict(policy=info["policy"],
                                              routing=info["routing"],
                                              protected_per_block=info["protected_per_block"],
                                              contested_total=info["contested_total"],
                                              bridge_weight=info["bridge_weight"],
                                              seeds_total=info["seeds_total"],
                                              seeds_per_block=info["seeds_per_block"],
                                              mmr_penalty=info["mmr_penalty"],
                                              novelty_weight=info["novelty_weight"],
                                              novelty_mean=round(info["novelty_mean"], 6)))
            windows[name] = window
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
            raise ValueError(f"{arm} produced no window this run")
        if int(fingerprints[arm + "/slots"]) != TARGET_SLOTS:
            raise ValueError(f"{arm} does not hold the {TARGET_SLOTS}-slot budget")
    for left, right in ((SELECTION, ANCHOR_ONLY), (ANCHOR_ONLY, V11_NATIVE),
                        (V11_NATIVE, BRIDGE)):
        if fingerprints[left] == fingerprints[right]:
            raise ValueError(f"{left} and {right} read the same pool")

    record = dict(
        arms=answered, fingerprints=fingerprints, baseline=baseline,
        windows={V11_NATIVE: built[V11_NATIVE]["check"], BRIDGE: built[BRIDGE]["check"]},
        router={arm: built[arm]["telemetry"] for arm in (V11_NATIVE, BRIDGE)},
        dropped_blocks=[int(block) for block in parts["dropped_blocks"]],
        selected_blocks=[int(block) for block in telemetry["selected_blocks"]],
        carrier_ages=[int(ages[block]) for block in picked])
    return record


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def _stats(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return dict(n=0)
    ordered = sorted(values)
    return dict(n=len(values), mean=round(sum(values) / len(values), 6),
                min=round(ordered[0], 6), max=round(ordered[-1], 6),
                median=round(ordered[len(ordered) // 2], 6))


def _evidence_in_dropped(sample):
    blocks = set(int(block) for block in (sample.get("evidence", {}).get("blocks") or []))
    return bool(blocks) and blocks <= set(sample["dropped_blocks"])


def _count(rows, arm, threshold):
    return int(sum(1 for row in rows if gr._correct(row, arm, threshold)))


def _decide(total, damage, baseline_damage):
    """The user's three-branch rule, applied mechanically instead of by eye.

    A net gain that is bought with more damage than the baseline does not count, and
    a net gain of fewer than five questions is not distinguishable from noise at
    n = 158 -- the same bar the last four rounds used.
    """
    vs_fixed = total["versus_fixed"]
    vs_anchor = total["versus_anchor_only"]
    if vs_fixed["net"] >= 5 and damage["lost"] <= baseline_damage:
        return dict(branch="proceed",
                    why="beats the existing transport by >= 5 questions without costing "
                        "more protected questions than the baseline does: the routing "
                        "carries its own weight, add the micro-span next")
    if vs_anchor["rescued"] > vs_anchor["harmed"]:
        return dict(branch="fix_the_floor",
                    why="rescues more than it breaks but does not convert that into a net "
                        "gain: the 140-slot floor is the thing to revisit before adding "
                        "any further mechanism")
    return dict(branch="stop",
                why="the routing rule does not move the window: check whether the bridging "
                    "score agrees with the reader at all before stacking anything on it")


def bridge_compare(samples, threshold):
    graded = [sample for sample in samples if sample.get("status") == "graded"]
    located = [sample for sample in graded if sample.get("evidence", {}).get("located")]
    dropped = [sample for sample in located if _evidence_in_dropped(sample)]
    kept = [sample for sample in located if not _evidence_in_dropped(sample)]
    present = [arm for arm in ARMS if arm in graded[0]["arms"]]

    def block(rows):
        return dict(
            n=len(rows),
            arms={arm: gr._arm_summary(rows, arm, threshold) for arm in present},
            versus_anchor_only={arm: er.pair_summary(rows, arm, ANCHOR_ONLY, threshold)
                                for arm in (BRIDGE, V11_NATIVE, GRAFT)},
            versus_fixed={arm: er.pair_summary(rows, arm, V11_NATIVE, threshold)
                          for arm in (BRIDGE,)})

    total = block(graded)
    damage = {arm: ef._damage(graded, arm, threshold) for arm in present}
    baseline_damage = int(damage[ANCHOR_ONLY]["lost"])
    thresholds = dict(
        total=dict(target=62, baseline=_count(graded, ANCHOR_ONLY, threshold),
                   achieved=_count(graded, BRIDGE, threshold)),
        versus_fixed=dict(target=5,
                          achieved=total["versus_fixed"][BRIDGE].get("net")),
        dropped_evidence=dict(target=19, n=len(dropped),
                              baseline=_count(dropped, ANCHOR_ONLY, threshold),
                              achieved=_count(dropped, BRIDGE, threshold)),
        kept_evidence=dict(target=12, n=len(kept),
                           baseline=_count(kept, ANCHOR_ONLY, threshold),
                           achieved=_count(kept, BRIDGE, threshold)),
        damage=dict(target=baseline_damage, full_rows=int(damage[FULL]["n"]),
                    baseline_lost=baseline_damage, achieved=int(damage[BRIDGE]["lost"])),
        rescue_over_damage=dict(
            rescued=int(total["versus_anchor_only"][BRIDGE]["rescued"]),
            harmed=int(total["versus_anchor_only"][BRIDGE]["harmed"]),
            rule="rescued must exceed harmed; a +8/-10 swap is not a gain"))
    bridge_rows = [sample["windows"][BRIDGE] for sample in graded]
    fixed_rows = [sample["windows"][V11_NATIVE] for sample in graded]
    routers = {arm: [sample["router"][arm] for sample in graded]
               for arm in (V11_NATIVE, BRIDGE)}
    return dict(
        definition=("the same 19-block write, the same nine carriers, the same 2,304 "
                    "slots and the same 158 rows; arm B keeps V11's fixed quota "
                    "(180 + 76 per carrier), arm C keeps 140 per carrier and lets the "
                    "whole window compete for the remaining 1,044. No parameter is "
                    "trained in either arm"),
        arms_present=present, n=len(graded),
        arms={arm: gr._arm_summary(graded, arm, threshold) for arm in present},
        total=total, dropped_evidence_group=block(dropped),
        kept_evidence_group=block(kept), damage=damage, thresholds=thresholds,
        decision=_decide(total, damage[BRIDGE], baseline_damage),
        allocation=dict(
            bridge=dict(
                protected_total=_stats([row["protected"] for row in bridge_rows]),
                admitted=_stats([row["admitted"] for row in bridge_rows]),
                admitted_from_dropped=_stats(
                    [row["admitted_from_dropped"] for row in bridge_rows]),
                admitted_share_from_dropped=_stats(
                    [row["admitted_share_from_dropped"] for row in bridge_rows]),
                empty_dropped_blocks=_stats(
                    [row["empty_dropped_blocks"] for row in bridge_rows]),
                min_per_dropped_block=_stats(
                    [row["min_per_dropped_block"] for row in bridge_rows]),
                max_per_dropped_block=_stats(
                    [row["max_per_dropped_block"] for row in bridge_rows]),
                admitted_per_carrier_block=_stats(
                    [min(row["admitted_per_carrier_block"]) for row in bridge_rows]),
                bridge_utility_correlation=_stats(
                    [row["bridge_utility_correlation"] for row in bridge_rows]),
                bridge_admitted_mean=_stats(
                    [row["bridge_admitted_mean"] for row in bridge_rows]),
                bridge_rejected_mean=_stats(
                    [row["bridge_rejected_mean"] for row in bridge_rows])),
            fixed=dict(
                protected_total=_stats([row["protected"] for row in fixed_rows]),
                admitted_from_dropped=_stats(
                    [row["admitted_from_dropped"] for row in fixed_rows]),
                empty_dropped_blocks=_stats(
                    [row["empty_dropped_blocks"] for row in fixed_rows]),
                min_per_dropped_block=_stats(
                    [row["min_per_dropped_block"] for row in fixed_rows]),
                max_per_dropped_block=_stats(
                    [row["max_per_dropped_block"] for row in fixed_rows])),
            settings=dict(
                fixed_policy=routers[V11_NATIVE][0]["policy"],
                bridge_policy=routers[BRIDGE][0]["policy"],
                bridge_weight=routers[BRIDGE][0]["bridge_weight"],
                seeds_per_block=routers[BRIDGE][0]["seeds_per_block"],
                mmr_penalty=routers[BRIDGE][0]["mmr_penalty"],
                novelty_weight=routers[BRIDGE][0]["novelty_weight"],
                novelty_mean=_stats([row["novelty_mean"] for row in routers[BRIDGE]]))))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def build_protocol(args, task, anchor_config, fixed_config, bridge_config):
    return dict(
        experiment="8: does a cross-block bridging route with a dynamic capacity "
                   "allocation recover more dropped-block evidence than V11's fixed "
                   "quota, without costing the carriers' answers?",
        treatment=("arm C replaces arm B's fixed 180 + 76 per-block split with a "
                   "140-slot floor per carrier and one global competition for the "
                   "remaining 1,044 slots, ranked by reader utility PLUS a cross-block "
                   "bridging mass computed from the frozen reader's own Q/K projections "
                   "(memory slots as proxy queries), de-duplicated with MMR"),
        task=task, data_dir=args.data_dir, arms=list(ARMS),
        window_blocks=args.window_blocks, target_blocks=args.target_blocks,
        block_tokens=BLOCK_TOKENS, slots_per_block=SLOTS_PER_BLOCK,
        budget=TARGET_SLOTS,
        baseline=dict(arm=ANCHOR_ONLY, quota=BASELINE_QUOTA,
                      core_total=args.target_blocks * BASELINE_QUOTA["core_per_block"],
                      graft_total=args.target_blocks * BASELINE_QUOTA["graft_per_block"],
                      anchor=args.anchor, anchor_sha256=sha(args.anchor),
                      anchor_config=anchor_config),
        fixed=dict(arm=V11_NATIVE, config=fixed_config,
                   definition="V11's quota (180 core + 76 graft per carrier) over native "
                              "slots, reader-utility ranking, no trained parameters"),
        bridge=dict(arm=BRIDGE, config=bridge_config,
                    definition="140 protected per carrier + 1,044 contested globally, "
                               "reader utility + cross-block bridging mass, MMR"),
        unchanged_conditions=dict(
            data="the same 158 HotpotQA rows",
            reader="frozen MemoryLLM-8B, frozen writer",
            carriers="the same 9, selected once and shared by every arm",
            window="the same 19-block write, document prefix only",
            capacity="2,304 slots in every compressed arm",
            transport="one selection shared by all 32 layers; chronological output; every "
                      "installed slot verbatim",
            training="none: no module is trained, loaded or fine-tuned in this round",
            decoding="greedy, unchanged"),
        checks=dict(
            reproduction="Full, Selection, Graft and AnchorOnly must reproduce the "
                         "reference report row for row",
            capacity="every compressed arm holds exactly 2,304 slots",
            order="global chronological order, unique source positions",
            native="every installed slot is bit-identical to the source slot it claims "
                   "to be (equality, not a tolerance)",
            floor="arm C protects exactly 9 x 140 slots, each from its own carrier block",
            split="admitted_from_dropped + admitted_from_carriers == 1,044"),
        thresholds=dict(
            total="Bridge >= 62/158 against AnchorOnly's 56",
            versus_fixed="Bridge - V11NativeGraft >= 5 questions",
            dropped_evidence="Bridge >= 19/40", kept_evidence="Bridge >= 12/34",
            damage="Bridge loses <= the baseline's 13 protected rows",
            rescue_over_damage="rescued > harmed against AnchorOnly"),
        known_limits=("no donor arm and no probe this round: a positive result attributes "
                      "to the whole routing bundle (floor + global competition + bridging "
                      "term), not to its content, and single-seed n=158 with a screening "
                      "F1 threshold is not a significance claim"),
        screening_caveat=gr.SCREENING_CAVEAT, f1_threshold=args.f1_threshold,
        seed=args.seed, query=gr.QUERY_SUFFIX,
        scoring="official HotpotQA QA F1, F1 >= threshold is correct",
        router_schema=ROUTER_SCHEMA, utility_layers=list(UTILITY_LAYERS))


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
                                      not_in_reference=absent, differing=differing[:20])
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
            dry_run=True, out=args.out, arms=list(ARMS),
            counts=dict(rows=len(rows), eligible=len(prepared), excluded=len(excluded)),
            readouts_per_sample=len(ARMS), bridge_weight=args.bridge_weight,
            probes=dict(task=args.probe_task, count=args.probe_count,
                        text=args.probe_text),
            protected_per_block=args.protected_per_block,
            seeds_per_block=args.seeds_per_block), indent=2, ensure_ascii=False))
        return None
    for flag, value, what in (("--checkpoint", args.checkpoint, "the frozen operator"),
                              ("--anchor", args.anchor, "the AnchorOnly module")):
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

    layers = tuple(int(x) for x in args.utility_layers)
    scorer = ReaderScorer(model, layers)
    probe_ids, probe_sources = build_probes(args, tokenizer)
    probes = [scorer.probe_bank(ids) for ids in probe_ids]
    fixed = FixedQuotaRouter(scorer, probes, core_per_block=args.core_per_block,
                             graft_per_block=args.graft_per_block,
                             mmr_penalty=args.mmr_penalty,
                             novelty_weight=args.novelty_weight)
    bridge = BridgeRouter(scorer, probes, protected_per_block=args.protected_per_block,
                          seeds_per_block=args.seeds_per_block,
                          bridge_weight=args.bridge_weight,
                          mmr_penalty=args.mmr_penalty,
                          novelty_weight=args.novelty_weight)
    fixed_config = dict(core_per_block=args.core_per_block,
                        graft_per_block=args.graft_per_block,
                        mmr_penalty=args.mmr_penalty,
                        novelty_weight=args.novelty_weight)
    bridge_config = dict(protected_per_block=args.protected_per_block,
                         seeds_per_block=args.seeds_per_block,
                         bridge_weight=args.bridge_weight,
                         mmr_penalty=args.mmr_penalty,
                         novelty_weight=args.novelty_weight,
                         utility_layers=list(layers),
                         probe=dict(sources=probe_sources, count=len(probes),
                                    sha256=digest_ids(probe_ids)))

    rows, prepared, excluded = gr.prepare_rows(args, tokenizer, facts)
    report = dict(
        schema=SCHEMA_RESULTS, status="running",
        counts=dict(rows=len(rows), eligible=len(prepared), excluded=len(excluded)),
        protocol=build_protocol(args, args.task, anchor_config, fixed_config,
                                bridge_config),
        identity=dict(
            model_path=args.model_path, checkpoint=args.checkpoint,
            checkpoint_sha256=sha(args.checkpoint),
            checkpoint_code_hash_verified=(saved or {}).get("code_hash_verified"),
            anchor=args.anchor, anchor_sha256=sha(args.anchor),
            data_dir=args.data_dir,
            data_sha256=sha(Path(args.data_dir) / (args.task + ".jsonl")),
            supporting_facts_glob=args.supporting_facts_glob,
            entrypoint_sha256=sha(__file__),
            router_module_sha256=sha(ROOT / "methods/prlmc/bridge_routing.py"),
            utility_module_sha256=sha(ROOT / "methods/prlmc/reader_utility.py"),
            residual_module_sha256=sha(ROOT / "methods/prlmc/anchor_residual.py"),
            reader_frozen=True, training_performed=False),
        excluded=excluded, samples=[], training_performed=False,
        optimizer_created=False,
        full_split=(args.max_samples is None and args.sample_ids is None))
    atomic_json(output, report)

    reference, table = ((None, None) if not args.verify_against
                        else ef.load_reference(args.verify_against))
    started = time.monotonic()
    for position, sample in enumerate(prepared):
        sample_started = time.monotonic()
        nv.seed_everything(args.seed)
        reset(manager, initial)
        result = run_sample(model, tokenizer, manager, merge, anchor_module, fixed,
                            bridge, sample, args)
        record = dict(id=sample["id"], status="graded",
                      available_blocks=sample["available_blocks"],
                      evidence=sample["evidence"],
                      seconds=time.monotonic() - sample_started, **result)
        report["samples"].append(record)
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output, report)
        print(json.dumps(dict(
            sample=position + 1, total=len(prepared), id=sample["id"],
            **{arm: result["arms"][arm]["correct"] for arm in ARMS},
            admitted_from_dropped=result["windows"][BRIDGE]["admitted_from_dropped"],
            seconds=round(record["seconds"], 1))), flush=True)

    report["summary"] = bridge_compare(report["samples"], args.f1_threshold)
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
        versus_anchor_only={
            arm: report["summary"]["total"]["versus_anchor_only"][arm].get("net")
            for arm in (BRIDGE, V11_NATIVE, GRAFT)},
        versus_fixed=report["summary"]["total"]["versus_fixed"][BRIDGE].get("net"),
        decision=report["summary"]["decision"],
        thresholds=report["summary"]["thresholds"]), indent=1, ensure_ascii=False))
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
    p.add_argument("--out", default=None)
    p.add_argument("--protected-per-block", dest="protected_per_block", type=int,
                   default=140, help="arm C's per-carrier floor (9 x 140 = 1,260)")
    p.add_argument("--seeds-per-block", dest="seeds_per_block", type=int,
                   default=DEFAULT_SEEDS_PER_BLOCK,
                   help="bridging query seeds drawn from each carrier's best slots")
    p.add_argument("--bridge-weight", dest="bridge_weight", type=float,
                   default=DEFAULT_BRIDGE_WEIGHT,
                   help="convex weight of the bridging term; 0 reproduces arm B's "
                        "ranking at arm C's allocation")
    p.add_argument("--core-per-block", dest="core_per_block", type=int, default=180,
                   help="arm B's fixed core quota (V11's 180)")
    p.add_argument("--graft-per-block", dest="graft_per_block", type=int, default=76,
                   help="arm B's fixed graft quota (V11's 76)")
    p.add_argument("--mmr-penalty", dest="mmr_penalty", type=float, default=0.5)
    p.add_argument("--novelty-weight", dest="novelty_weight", type=float, default=0.5)
    p.add_argument("--utility-layers", dest="utility_layers", type=int, nargs="+",
                   default=list(UTILITY_LAYERS))
    p.add_argument("--probe-task", dest="probe_task", default="musique",
                   help="training task the fixed query bank is drawn from; must not be "
                        "the evaluation task")
    p.add_argument("--probe-count", dest="probe_count", type=int, default=16)
    p.add_argument("--probe-text", dest="probe_text", default=None,
                   help="substitute one fixed sentence for the query bank")
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
