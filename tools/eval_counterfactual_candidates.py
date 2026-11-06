#!/usr/bin/env python3
"""Experiment 2, counterfactual ablation: are the 2,200 candidate slots load-bearing?

Experiment 2 gained +10.13 points by letting each of the 360 graft slots read the
unselected leftovers of its own source block. This driver asks the one question
that gain has not answered yet: does the gain need *those* slots, or would any
2,200 latent vectors of the same shape have done?

Nothing is retrained and nothing about the architecture changes. The trained
module, the 360 native anchors, the carriers, the positions, the quota, the core
slots, the reader and the decoding are all exactly experiment 2's. The single
manipulation is the CONTENT of the 2,200 candidate slots: instead of the sample's
own dropped blocks they are filled from ANOTHER document's dropped blocks, which
breaks every association between what the anchor reads and the question asked.

    Full            19 blocks, uncompressed                    reference ceiling
    Selection       9 carrier blocks, verbatim                 capacity baseline
    Graft           V10, 360 native anchors                    the arm to beat
    Fusion          + the trained module on its own leftovers  experiment 2, re-run
    Counterfactual  + the same module on another document's leftovers

All five arms run inside one process on one pre-compression window, so every
comparison is paired row for row and the four old arms double as a reproduction
check (`--verify-against` compares them with the finished experiment-2 report).

What each outcome would mean
----------------------------
* Counterfactual ~ Graft   the gain needs the same document's knowledge
* Counterfactual ~ Fusion  the gain is a generic perturbation of the anchors,
                           not knowledge: experiment 2's reading would be wrong
* in between               the knowledge helps, but part of the effect is the
                           residual itself

The donor rule
--------------
The donor of a row is the previous eligible row in evaluation order (row 0
borrows row 1's window), so every row reads a different, unrelated document and
no row reads itself. Both windows drop exactly ten blocks, so the k-th dropped
block of the sample maps to the k-th dropped block of the donor: an anchor still
reads leftovers of "its" block, they simply belong to another document. Per-block
counts differ between documents because the leverage rule does not pick the same
number of anchors from every block, so the walk over the donor's leftovers is a
seeded permutation, cycled if the donor has fewer leftovers than the sample needs.
The mapping digest is stored per row; the rule is deterministic and rebuildable.

Usage
-----
    python tools/eval_counterfactual_candidates.py --self-test        # no GPU

    python tools/eval_counterfactual_candidates.py --dry-run \
        --data-dir /data/.../LongBench/extracted/data

    python tools/eval_counterfactual_candidates.py \
        --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
        --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
        --model-path /dev/shm/s1d_reader_fp16 \
        --checkpoint runs/natural_merge_v9/full_v10.pt \
        --compaction runs/source_compaction/compaction_rank32.pt \
        --verify-against reports/exp2_source_compaction/eval_trained_hotpotqa.json \
        --cuda-device 0 \
        --out reports/exp2_source_compaction/eval_counterfactual_hotpotqa.json
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

import eval_graft_recovery as gr                          # noqa: E402
import eval_learned_fusion as ef                          # noqa: E402
import eval_replace_reorder as er                         # noqa: E402
import eval_source_compaction as es                       # noqa: E402
import native_eval as nv                                  # noqa: E402
from io_utils import atomic_json, sha                     # noqa: E402
from runtime import load_runtime, reset                   # noqa: E402
from methods.prlmc.core_fusion import prompt_hidden_states  # noqa: E402
from methods.prlmc.source_compaction import (             # noqa: E402
    SCHEMA, SourceCompaction, compact_window, split_source_anchors)

BLOCK = gr.SLOTS_PER_BLOCK
BLOCK_TOKENS = gr.BLOCK_TOKENS
GRAFT_SLOTS_TOTAL = gr.GRAFT_SLOTS_TOTAL
TARGET_SLOTS = gr.TARGET_SLOTS
FULL, SELECTION, GRAFT, FUSION = ef.FULL, ef.SELECTION, ef.GRAFT, ef.FUSION
COUNTERFACTUAL = "Counterfactual"
ARMS = (FULL, SELECTION, GRAFT, FUSION, COUNTERFACTUAL)
SCHEMA_RESULTS = "source_compaction_counterfactual_results/1"


# ---------------------------------------------------------------------------
# the donor
# ---------------------------------------------------------------------------
def donor_map(parts, donor, seed):
    """Row indices into the donor's candidate pool, one per candidate position.

    Both windows drop exactly `window_blocks - target_blocks` blocks, so the k-th
    dropped block of the sample maps to the k-th dropped block of the donor. The
    counts per block differ between documents (the leverage rule picks a
    different number of anchors from every block), so the donor's leftovers are
    walked in a seeded permutation and wrapped when the sample needs more than the
    donor has: every candidate position is filled by a slot that really is a
    leftover of the donor's k-th dropped block, and nothing is left unfilled.

    The bookkeeping runs on the CPU: it is 2,200 integers, and a device-side
    permutation would need a device-side generator for no benefit.
    """
    blocks = parts["candidate_blocks"].cpu()
    donor_blocks = donor["blocks"].cpu()
    mapping = torch.full((int(blocks.numel()),), -1, dtype=torch.long)
    generator = torch.Generator().manual_seed(int(seed))
    for current_block, donor_block in zip(parts["dropped_blocks"], donor["dropped_blocks"]):
        current = torch.nonzero(blocks == int(current_block)).flatten()
        pool = torch.nonzero(donor_blocks == int(donor_block)).flatten()
        if current.numel() == 0 or pool.numel() == 0:
            raise ValueError(f"dropped block {current_block} has no candidate slot on "
                             f"one side: the counterfactual cannot be filled")
        order = pool[torch.randperm(int(pool.numel()), generator=generator)]
        mapping[current] = order[torch.arange(int(current.numel())) % int(order.numel())]
    if bool((mapping < 0).any()):
        raise ValueError("a candidate position was left without a donor slot")
    return mapping.to(parts["candidate_positions"].device)


def counterfactual_window(module, round1, donor, seed, sample_id):
    """The compacted window when the candidates come from another document."""
    if donor["id"] == sample_id:
        raise ValueError("the donor is the sample itself: this row would measure nothing")
    parts = round1["parts"]
    mapping = donor_map(parts, donor, seed)
    donor_values = donor["slots"].index_select(1, mapping.to(donor["slots"].device))
    real_values = round1["source"].index_select(1, parts["candidate_positions"])
    if torch.equal(donor_values, real_values):
        raise ValueError("the donor supplied the sample's own candidate slots")
    hybrid = round1["source"].index_copy(1, parts["candidate_positions"], donor_values)
    window, info = compact_window(round1["output"], hybrid, parts, module)
    ratio = float(donor_values.float().square().mean().sqrt()
                  / real_values.float().square().mean().sqrt().clamp_min(1e-12))
    return window, dict(info, donor_id=donor["id"], candidate_rms_ratio=ratio,
                        mapping_sha256=hashlib.sha256(
                            mapping.cpu().numpy().tobytes()).hexdigest())


def check_counterfactual(round1, counterfactual, info, sample_id):
    """Graft slots only, and they must actually change -- checked, not assumed."""
    check = es.check_compaction(dict(round1, fused=counterfactual, compaction_info=info))
    anchors = round1["parts"]["anchor_offsets"]
    if torch.equal(counterfactual.index_select(1, anchors),
                   round1["fused"].index_select(1, anchors)):
        raise ValueError("the counterfactual reproduced the Fusion window exactly")
    if info["donor_id"] == sample_id:
        raise ValueError("the donor is the sample itself")
    return check


def build_donor(manager, merge, module, prepared, args):
    """One document's window, compressed once; only its candidate slots are kept.

    `compress_window` is experiment 2's own one-shot path, so the donor's
    leftovers are the slots experiment 2 would have read on that document. The
    compaction it performs is thrown away: only the candidates are needed.
    """
    chunks = [prepared["write_ids"][index:index + BLOCK_TOKENS]
              for index in range(0, len(prepared["write_ids"]), BLOCK_TOKENS)]
    with torch.no_grad():
        es.write_blocks(manager, merge, chunks[:args.window_blocks])
        result = es.compress_window(manager, merge, module, args.window_blocks,
                                    args.target_blocks)
        slots = result["source"].index_select(1, result["parts"]["candidate_positions"])
    return dict(id=prepared["id"], slots=slots,
                blocks=result["parts"]["candidate_blocks"],
                dropped_blocks=result["parts"]["dropped_blocks"])


# ---------------------------------------------------------------------------
# one sample
# ---------------------------------------------------------------------------
def run_sample(model, tokenizer, manager, merge, module, prepared, donor, args, position):
    """Write the window once, compress once, then answer all five arms."""
    chunks = [prepared["write_ids"][index:index + BLOCK_TOKENS]
              for index in range(0, len(prepared["write_ids"]), BLOCK_TOKENS)]
    answers = prepared["answers"]
    fingerprints, gaps = {}, {}
    with torch.no_grad():
        es.write_blocks(manager, merge, chunks[:args.window_blocks])
        state = gr.snapshot(model, manager)
        query_hidden = prompt_hidden_states(model, prepared["query_ids"], ef.TAP_LAYERS)
        round1 = es.compress_window(manager, merge, module, args.window_blocks,
                                    args.target_blocks)
        counterfactual, counterfactual_info = counterfactual_window(
            module, round1, donor, args.seed * 100003 + position, prepared["id"])
    check = es.check_compaction(round1)
    counterfactual_check = check_counterfactual(round1, counterfactual,
                                                counterfactual_info, prepared["id"])
    # Keyed by arm because the shared analysis reads `structure[Graft]`; the same
    # check covers both compacted arms -- each changes the graft slots and leaves
    # the core alone.
    structure = {SELECTION: None, GRAFT: check, FUSION: check}

    selection = ef.selection_window(round1["source"], round1["picked"])
    gaps[SELECTION] = es.readout_gap(model, round1["source"], selection,
                                     query_hidden, ef.TAPS)
    for arm, window in ((GRAFT, round1["output"]), (FUSION, round1["fused"]),
                        (COUNTERFACTUAL, counterfactual)):
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
            elif arm == FUSION:
                er.install_output(manager, round1["fused"], round1["picked"],
                                  args.window_blocks)
            elif arm == COUNTERFACTUAL:
                er.install_output(manager, counterfactual, round1["picked"],
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
    for arm in (SELECTION, GRAFT, FUSION, COUNTERFACTUAL):
        if int(fingerprints[arm + "/slots"]) != TARGET_SLOTS:
            raise ValueError(f"arm {arm} does not hold the {TARGET_SLOTS}-slot budget")
    for left, right in ((SELECTION, GRAFT), (GRAFT, FUSION), (FUSION, COUNTERFACTUAL)):
        if fingerprints[left] == fingerprints[right]:
            raise ValueError(f"{left} and {right} read the same pool: the manipulation "
                             f"did not reach the memory")

    provenance = gr.graft_provenance(round1["telemetry"], args.window_blocks)
    if provenance["selected_blocks"] != sorted(round1["picked"]):
        raise ValueError("graft provenance disagrees with the shared carrier set")
    record = dict(arms=arms, fingerprints=fingerprints, gaps=gaps,
                  provenance=provenance, structure=structure, multi_round=False,
                  counterfactual=dict(**counterfactual_info, check=counterfactual_check),
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


def counterfactual_summary(samples, threshold):
    """The block the whole driver exists for: Counterfactual against Fusion and Graft."""
    graded = [sample for sample in samples if sample.get("status") == "graded"]
    located = [sample for sample in graded if sample.get("evidence", {}).get("located")]
    dropped = [sample for sample in located if ef._evidence_in_dropped(sample)]
    kept = [sample for sample in located if not ef._evidence_in_dropped(sample)]

    def comparisons(rows):
        return {
            f"{COUNTERFACTUAL}_minus_{FUSION}":
                er.pair_summary(rows, COUNTERFACTUAL, FUSION, threshold),
            f"{COUNTERFACTUAL}_minus_{GRAFT}":
                er.pair_summary(rows, COUNTERFACTUAL, GRAFT, threshold),
            f"{FUSION}_minus_{GRAFT}": er.pair_summary(rows, FUSION, GRAFT, threshold),
        }

    def group(rows):
        return dict(n=len(rows),
                    arms={arm: gr._arm_summary(rows, arm, threshold) for arm in ARMS},
                    pairs=comparisons(rows))

    return dict(
        definition=("Fusion with its 2,200 candidate slots replaced, per row, by the "
                    "leftovers of another document's dropped blocks; anchors, module, "
                    "carriers, quota, core slots and reader unchanged"),
        n=len(graded),
        arms={arm: gr._arm_summary(graded, arm, threshold) for arm in ARMS},
        pairs=comparisons(graded),
        dropped_evidence_group=group(dropped),
        kept_evidence_group=group(kept),
        damage={arm: ef._damage(graded, arm, threshold)
                for arm in (GRAFT, FUSION, COUNTERFACTUAL)},
        residual=dict(
            definition="RMS of the injected residual over RMS of the anchor",
            fusion=_stats([sample["structure"][FUSION]["relative_delta"]
                           for sample in graded]),
            counterfactual=_stats([sample["counterfactual"]["relative_delta"]
                                   for sample in graded]),
            candidate_rms_ratio=_stats([sample["counterfactual"]["candidate_rms_ratio"]
                                        for sample in graded])),
        readout=dict(
            definition="mean reader-readout distance from the uncompressed window",
            by_arm={arm: ef._gap_summary(graded, arm)
                    for arm in (GRAFT, FUSION, COUNTERFACTUAL)},
            held_out={arm: ef._gap_summary(graded, arm + "/heldout")
                      for arm in (GRAFT, FUSION, COUNTERFACTUAL)},
            counterfactual_removes=ef._gap_removed(graded, COUNTERFACTUAL, GRAFT)),
        donors=dict(rule="the previous eligible row; row 0 borrows row 1",
                    distinct=len({sample["counterfactual"]["donor_id"]
                                  for sample in graded}),
                    self_donor=len([sample for sample in graded
                                    if sample["counterfactual"]["donor_id"]
                                    == sample["id"]])),
    )


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _synthetic(seed, block=16, core=12, graft_per_block=4, window_blocks=5,
               target_blocks=3, selected=(0, 2, 4), layers=2, hidden=8):
    """A window, its compaction geometry and an operator output, on the CPU."""
    dropped = [index for index in range(window_blocks) if index not in selected]
    torch.manual_seed(seed)
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
            full_positions[slot * block + core + offset] = source_block * block + 6 + offset
    telemetry = dict(bridge_slot_offsets=offsets, slot_source_positions=full_positions,
                     selected_blocks=list(selected))
    parts = split_source_anchors(telemetry, block, window_blocks)
    return window, parts, torch.randn(layers, target_blocks * block, hidden)


def self_test():
    """The donor mapping and the counterfactual path, checked without a GPU.

    Two synthetic documents stand in for two real ones. What is checked here is
    exactly what a plausible-looking GPU number would not show: that the mapping
    fills every candidate position from the donor's matching block, that it is
    deterministic, and that a zero output projection still reproduces the grafted
    window through the hybrid memory.
    """
    window, parts, output = _synthetic(0)
    donor_window, donor_parts, _ = _synthetic(1)
    donor = dict(id="donor",
                 slots=donor_window.index_select(1, donor_parts["candidate_positions"]),
                 blocks=donor_parts["candidate_blocks"],
                 dropped_blocks=donor_parts["dropped_blocks"])
    mapping = donor_map(parts, donor, seed=7)
    if int(mapping.numel()) != int(parts["candidate_positions"].numel()):
        raise ValueError("the mapping does not cover every candidate position")
    if int(mapping.min()) < 0 or int(mapping.max()) >= int(donor["slots"].shape[1]):
        raise ValueError("a mapped row falls outside the donor pool")
    for current_block, donor_block in zip(parts["dropped_blocks"], donor["dropped_blocks"]):
        current = torch.nonzero(parts["candidate_blocks"] == current_block).flatten()
        rows = donor["blocks"][mapping[current]]
        if set(rows.tolist()) != {int(donor_block)}:
            raise ValueError(f"a candidate of block {current_block} reads a slot that is "
                             f"not a leftover of the donor's block {donor_block}")
    if not torch.equal(mapping, donor_map(parts, donor, seed=7)):
        raise ValueError("the mapping is not deterministic")
    if torch.equal(mapping, donor_map(parts, donor, seed=8)):
        raise ValueError("the mapping ignores its seed")
    # A donor with fewer leftovers than the sample needs must wrap, not fail: on
    # real documents the leverage rule does not leave the same count per block.
    thin = dict(id="thin", slots=torch.randn(2, 5, 8),
                blocks=torch.tensor([1, 1, 1, 3, 3]), dropped_blocks=[1, 3])
    thin_map = donor_map(parts, thin, seed=7)
    if int(thin_map.max()) >= 5 or int(thin_map.min()) < 0:
        raise ValueError("the thin donor was read out of range")
    for block, pool in ((1, {0, 1, 2}), (3, {3, 4})):
        current = torch.nonzero(parts["candidate_blocks"] == block).flatten()
        if set(thin_map[current].tolist()) - pool:
            raise ValueError("the thin donor was read from the wrong block")
        if set(thin_map[current].tolist()) != pool:
            raise ValueError("the thin donor's leftovers were not all used")
    hybrid = window.index_copy(1, parts["candidate_positions"],
                               donor["slots"].index_select(1, mapping))
    if torch.equal(hybrid.index_select(1, parts["candidate_positions"]),
                   window.index_select(1, parts["candidate_positions"])):
        raise ValueError("the hybrid window kept the sample's own candidates")

    module = SourceCompaction(hidden_size=8, rank=4).eval().requires_grad_(False)
    counterfactual, info = compact_window(output, hybrid, parts, module)
    if not torch.equal(counterfactual, output):
        raise ValueError("a zero output projection did not reproduce the grafted window")
    with torch.no_grad():
        module.out_proj.normal_(std=0.05)
    counterfactual, info = compact_window(output, hybrid, parts, module)
    fused, _ = compact_window(output, window, parts, module)
    keep = torch.ones(output.shape[1], dtype=torch.bool)
    keep[parts["anchor_offsets"]] = False
    if not torch.equal(counterfactual[:, keep], output[:, keep]):
        raise ValueError("the counterfactual changed a core slot")
    if torch.equal(counterfactual, fused):
        raise ValueError("the donor candidates produced the same window as the sample's")
    if not torch.isfinite(counterfactual).all():
        raise ValueError("the counterfactual window is not finite")
    print(json.dumps(dict(
        self_test="ok", anchors=int(parts["anchor_offsets"].numel()),
        candidates=int(parts["candidate_positions"].numel()),
        dropped_blocks=list(parts["dropped_blocks"]),
        candidates_per_block=parts["candidates_per_block"],
        donor_candidates=int(donor["slots"].shape[1]),
        identity_at_init=True, core_bit_identical=True,
        differs_from_fusion=True, mapping_deterministic=True,
        relative_delta=round(info["relative_delta"], 5)), indent=2))
    return 0


def build_protocol(args, task, module_config):
    protocol = es.build_protocol(args, task, module_config)
    protocol.update(
        experiment=("2-counterfactual: are the 2,200 candidate slots load-bearing?"),
        treatment=("the 360 native anchors and the trained module are unchanged; the "
                   "2,200 candidate slots each anchor reads are replaced, per row, by "
                   "the leftovers of ANOTHER document's dropped blocks"),
        arms=list(ARMS),
        counterfactual=dict(
            donor="the previous eligible row in evaluation order; row 0 borrows row 1 "
                  "(a different document in every case, checked per row)",
            mapping="the k-th dropped block of the sample maps to the k-th dropped "
                    "block of the donor; the donor's leftovers are walked in a seeded "
                    "permutation, cycled when the donor has fewer leftovers than the "
                    "sample needs",
            seed_rule="torch.Generator().manual_seed(seed * 100003 + row_index)",
            unchanged=dict(
                anchors="the 360 native anchors, byte-identical",
                module="the experiment-2 checkpoint, eval mode, no training",
                core_slots="byte-identical to the grafted window",
                carriers="the same 9", reader="frozen MemoryLLM-8B",
                data="the same 158 HotpotQA rows", decoding="greedy, unchanged"),
            only_difference="the content of the 2,200 candidate slots",
            pairing_note=("all five arms run in one process on one pre-compression "
                          "window, so every comparison is paired row for row and the "
                          "four old arms double as a reproduction check"),
            reading=("Counterfactual close to Graft means the gain needs the same "
                     "document's knowledge; close to Fusion means it is a generic "
                     "perturbation of the anchors rather than knowledge")),
        compaction=dict(protocol["compaction"],
                        candidates="the donor document's unselected dropped-block slots, "
                                   "same shape and same per-block grouping"))
    return protocol


def verify_reference(args, reference, table, samples):
    """Full, Selection, Graft AND Fusion must reproduce the experiment-2 report.

    `ef.verify_against` checks the three untouched arms; Fusion is added here
    because it is the arm the counterfactual is subtracted from. If it did not
    reproduce the earlier run, the counterfactual would be compared against a
    different baseline.
    """
    verdict = ef.verify_against(args.verify_against, reference, table, samples)
    same = different = 0
    differing = []
    for sample in samples:
        old = table.get(str(sample["id"]))
        if old is None or FUSION not in old.get("arms", {}):
            continue
        if old["arms"][FUSION]["prediction"] == sample["arms"][FUSION]["prediction"]:
            same += 1
        else:
            different += 1
            differing.append(dict(id=sample["id"],
                                  old=old["arms"][FUSION]["prediction"],
                                  new=sample["arms"][FUSION]["prediction"]))
    verdict[FUSION] = dict(identical=same, different=different, differing=differing[:20])
    verdict["passed"] = verdict["passed"] and different == 0 and same == len(samples)
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
            donors=dict(first=prepared[1]["id"] if len(prepared) > 1 else None,
                        then="each row donates to the next"),
            readouts_per_sample=len(ARMS)), indent=2, ensure_ascii=False))
        return None
    if not args.checkpoint or not Path(args.checkpoint).is_file():
        raise ValueError("this experiment needs the frozen operator: pass --checkpoint")
    if not args.compaction or not Path(args.compaction).is_file():
        raise ValueError("this experiment needs the trained module: pass --compaction")

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
    payload = torch.load(args.compaction, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA + "/checkpoint":
        raise ValueError(f"{args.compaction} is not a {SCHEMA} checkpoint")
    module_config = dict(payload["config"])
    module = SourceCompaction(**module_config)
    module.load_state_dict(payload["state_dict"], strict=True)
    module = module.to(model.memory.device).eval().requires_grad_(False)
    if module_config.get("hidden_size") != int(manager.hidden):
        raise ValueError(f"the module was built for hidden "
                         f"{module_config.get('hidden_size')}, the reader is "
                         f"{manager.hidden}")

    rows, prepared, excluded = gr.prepare_rows(args, tokenizer, facts)
    if len(prepared) < 2:
        raise ValueError("a counterfactual needs at least two rows: the donor of a row "
                         "is another document")
    report = dict(
        schema=SCHEMA_RESULTS, status="running",
        counts=dict(rows=len(rows), eligible=len(prepared), excluded=len(excluded)),
        protocol=build_protocol(args, args.task, module_config),
        identity=dict(
            model_path=args.model_path, checkpoint=args.checkpoint,
            checkpoint_sha256=sha(args.checkpoint),
            checkpoint_code_hash_verified=(saved or {}).get("code_hash_verified"),
            compaction=args.compaction, compaction_sha256=sha(args.compaction),
            data_dir=args.data_dir,
            data_sha256=sha(Path(args.data_dir) / (args.task + ".jsonl")),
            supporting_facts_glob=args.supporting_facts_glob,
            entrypoint_sha256=sha(__file__),
            compaction_module_sha256=sha(ROOT / "methods/prlmc/source_compaction.py"),
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
            donor = build_donor(manager, merge, module, prepared[1], args)
            reset(manager, initial)
            nv.seed_everything(args.seed)
        result, donor = run_sample(model, tokenizer, manager, merge, module, sample,
                                   donor, args, position)
        record = dict(id=sample["id"], status="graded",
                      available_blocks=sample["available_blocks"],
                      evidence=sample["evidence"],
                      seconds=time.monotonic() - sample_started, **result)
        report["samples"].append(record)
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output, report)
        print(json.dumps(dict(
            sample=position + 1, total=len(prepared), id=sample["id"],
            donor=record["counterfactual"]["donor_id"],
            graft=result["arms"][GRAFT]["correct"],
            fusion=result["arms"][FUSION]["correct"],
            counterfactual=result["arms"][COUNTERFACTUAL]["correct"],
            delta_fusion=result["structure"][FUSION]["relative_delta"],
            delta_counterfactual=record["counterfactual"]["relative_delta"],
            seconds=round(record["seconds"], 1))), flush=True)

    report["summary"] = ef.analyse(report["samples"], args.f1_threshold,
                                   excluded=len(report["excluded"]))
    report["summary"]["counterfactual"] = counterfactual_summary(report["samples"],
                                                                 args.f1_threshold)
    if args.verify_against:
        report["verification_against_reference"] = verify_reference(
            args, reference, table, report["samples"])
    report["elapsed_seconds"] = time.monotonic() - started
    report["status"] = "complete"
    report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    atomic_json(output, report)
    print(json.dumps(dict(status="complete", out=str(output),
                          elapsed_seconds=round(report["elapsed_seconds"], 1),
                          pairs={key: value for key, value in
                                 report["summary"]["counterfactual"]["pairs"].items()}),
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
    p.add_argument("--compaction", default=None,
                   help="the trained SourceCompaction checkpoint")
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
                   help="the finished experiment-2 report; Full, Selection, Graft and "
                        "Fusion must reproduce it row for row")
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
