#!/usr/bin/env python3
"""Round 5: is the 1944 / 360 split the problem?

Every negative result in this project shares one property: the thing being added
had to be squeezed into 360 native slots, and the fixes for it (attention over the
leftovers, a sparse top-4 read, a frozen calibration, a key-safe write, a
key/value swap) all failed to add knowledge. None of them questioned the split
itself. 84.4% of the 2,304 slots are handed to the nine carrier blocks before any
question is asked, and only 360 verbatim slots of the ten dropped blocks survive.

This driver changes nothing but that ratio, and asks the question directly: with a
frozen reader and no fusion module at all, does giving the dropped blocks more
native slots recover more of what they lost?

    config   core / block   graft / block   core total   graft total   graft share
    A        216            40              1944         360           15.6%
    B        160            96              1440         864           37.5%
    C        144           112              1296         1008           43.75%

Two arms per config, and nothing else changes:

    Graft_x        native V10 grafting at that quota
    AnchorOnly_x   the same window, with the already-trained AnchorResidual applied

`AnchorOnly` is NOT retrained. The one trained module is loaded once and applied to
whatever anchor count the config produces -- 360, 864 or 1008. That is a transfer,
not a fit, and it is the second question this driver answers: does the calibration
learned at 360 anchors survive at a different quota?

Everything else is held: the same 19-block window, the same nine carriers (selected
once and shared by every arm of every config), the same 9 output blocks, the same
2,304-slot budget, the same frozen reader, the same 158 rows, greedy decoding.
Only `merge.core_slots_per_block` / `merge.graft_slots_per_block` move, which the
operator reads at call time (its only invariant is core + graft = 256 per block).

Why the old evaluator's checks are not reused
---------------------------------------------
`eval_graft_recovery` and `eval_source_compaction` assert 1,944 core and 360 graft
by name, and `graft_provenance` refuses anything else. Those assertions are the
reason the old drivers cannot express this experiment at all, so this one checks
the same properties against the QUOTA OF THE RUN instead of against constants:
total capacity, core/graft counts, core slots being verbatim copies of their source
positions, the anchors coming only from dropped blocks, and the output staying in
global chronological order. It also records how many graft slots each dropped block
actually received, and counts the blocks that got none -- a global leverage ranking
over 2,560 candidates does not have to spread.

Reading the result
------------------
The first 158-row run is a screen, not evidence:

* AnchorOnly_B or _C at least ~5 questions above AnchorOnly_A (56), with the new
  rescues concentrated on the rows whose evidence the compression dropped, and with
  damage on already-correct rows not growing -> capacity reallocation is worth
  building on, and the next step is a larger natural set.
* 0-2 questions, or a decline, at both B and C -> simply handing out more native
  slots does not produce a signal either, and static quota tuning should stop. Only
  then is a reader-aware, per-layer selection mechanism the right thing to build.

Config A is a reproduction check before anything else is read: its four arms must
match the finished anchor-only report row for row, because A is that run's quota.

Usage
-----
    python tools/eval_budget_reallocation.py --dry-run \
        --data-dir /data/.../LongBench/extracted/data --model-path /dev/shm/s1d_reader_fp16

    python tools/eval_budget_reallocation.py \
        --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
        --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
        --model-path /dev/shm/s1d_reader_fp16 \
        --checkpoint runs/natural_merge_v9/full_v10.pt \
        --anchor runs/anchor_residual/anchor_only_rank64.pt \
        --verify-against reports/anchor_only/eval_anchor_only_hotpotqa.json \
        --cuda-device 0 \
        --out reports/budget_reallocation/eval_budget_hotpotqa.json
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
from methods.prlmc.source_compaction import split_source_anchors  # noqa: E402

BLOCK_TOKENS = gr.BLOCK_TOKENS
SLOTS_PER_BLOCK = gr.SLOTS_PER_BLOCK

# The sweep. `graft_per_block` also fixes the total: 9 output blocks x per-block.
CONFIGS = (
    dict(name="A", core_per_block=216, graft_per_block=40),
    dict(name="B", core_per_block=160, graft_per_block=96),
    dict(name="C", core_per_block=144, graft_per_block=112),
)
FULL, SELECTION = ef.FULL, ef.SELECTION
SCHEMA_RESULTS = "budget_reallocation_results/1"


def arms_of(config_name):
    """The two arms of one quota, in report order."""
    return (f"Graft_{config_name}", f"AnchorOnly_{config_name}")


def all_arms():
    return (FULL, SELECTION) + tuple(arm for config in CONFIGS
                                     for arm in arms_of(config["name"]))


def totals(config, target_blocks):
    core = target_blocks * config["core_per_block"]
    graft = target_blocks * config["graft_per_block"]
    return core, graft


# ---------------------------------------------------------------------------
# the one thing this experiment moves
# ---------------------------------------------------------------------------
def set_quota(merge, config, block=SLOTS_PER_BLOCK):
    """Point the frozen operator at a different core/graft split.

    `ManifoldGraftedSlotMerge` reads both counts inside its forward and asserts only
    that they sum to the block size in `__init__`, so rebinding them between calls is
    the whole change -- there is no cached index or mask to rebuild.
    """
    if config["core_per_block"] + config["graft_per_block"] != block:
        raise ValueError(f"{config['name']}: {config['core_per_block']} core + "
                         f"{config['graft_per_block']} graft does not fill a "
                         f"{block}-slot block")
    if config["core_per_block"] <= 0 or config["graft_per_block"] <= 0:
        raise ValueError(f"{config['name']}: a quota of zero is not a reallocation")
    merge.core_slots_per_block = int(config["core_per_block"])
    merge.graft_slots_per_block = int(config["graft_per_block"])
    if (int(merge.core_slots_per_block) != int(config["core_per_block"])
            or int(merge.graft_slots_per_block) != int(config["graft_per_block"])):
        raise ValueError("the operator did not take the new quota")


# ---------------------------------------------------------------------------
# checks, against the quota of THIS run instead of against constants
# ---------------------------------------------------------------------------
def check_window(config, output, telemetry, parts, source, manager, args):
    """Capacity, split, provenance, order and the verbatim-core property."""
    expected_core, expected_graft = totals(config, args.target_blocks)
    capacity = args.target_blocks * manager.block
    if expected_core + expected_graft != capacity:
        raise ValueError("the quota does not fill the 2,304-slot budget")

    if output.shape[1] != capacity:
        raise ValueError(f"{config['name']}: {output.shape[1]} output slots, expected "
                         f"{capacity}")
    offsets = int(parts["anchor_offsets"].numel())
    if offsets != expected_graft:
        raise ValueError(f"{config['name']}: {offsets} graft slots, expected "
                         f"{expected_graft}")
    keep = torch.ones(capacity, dtype=torch.bool)
    keep[parts["anchor_offsets"]] = False
    core_positions = torch.nonzero(keep).flatten()
    if int(core_positions.numel()) != expected_core:
        raise ValueError(f"{config['name']}: {int(core_positions.numel())} core slots, "
                         f"expected {expected_core}")

    positions = [int(value) for value in telemetry["slot_source_positions"]]
    if len(positions) != capacity:
        raise ValueError(f"{config['name']}: {len(positions)} source positions for "
                         f"{capacity} output slots")
    if positions != sorted(positions):
        raise ValueError(f"{config['name']}: the output is not in global chronological "
                         f"order")
    dropped = set(int(block) for block in parts["dropped_blocks"])
    if set(int(block) for block in parts["anchor_blocks"].tolist()) - dropped:
        raise ValueError(f"{config['name']}: an anchor came from a block this run kept")

    # The core slots must be verbatim copies of the window positions they claim.
    index = torch.as_tensor(positions, dtype=torch.long, device=output.device)
    copied = source.index_select(1, index.index_select(0, core_positions.to(output.device)))
    if not torch.equal(output.index_select(1, core_positions.to(output.device)), copied):
        raise ValueError(f"{config['name']}: a core slot is not a verbatim copy of its "
                         f"source position")
    if not torch.isfinite(output).all():
        raise ValueError(f"{config['name']}: the compressed window is not finite")

    per_block = [int(parts["anchors_per_block"][str(block)]) for block in parts["dropped_blocks"]]
    return dict(
        core_total=expected_core, graft_total=expected_graft,
        capacity=capacity, anchor_offsets=offsets,
        candidate_slots=int(parts["candidate_positions"].numel()),
        graft_share=round(100.0 * expected_graft / capacity, 2),
        per_dropped_block=per_block,
        empty_dropped_blocks=int(sum(1 for count in per_block if count == 0)),
        min_per_dropped_block=int(min(per_block)), max_per_dropped_block=int(max(per_block)),
        core_verbatim=True, chronological=True)


def check_anchor(config, output, parts, anchor_window, info):
    """The calibration may touch the anchors and nothing else, at any quota."""
    offsets = parts["anchor_offsets"]
    keep = torch.ones(output.shape[1], dtype=torch.bool)
    keep[offsets] = False
    if not torch.equal(output[:, keep], anchor_window[:, keep]):
        raise ValueError(f"{config['name']}: the anchor module changed a core slot")
    if torch.equal(output.index_select(1, offsets), anchor_window.index_select(1, offsets)):
        raise ValueError(f"{config['name']}: the anchor module changed nothing, so this "
                         f"arm would be a copy of Graft")
    if int(info["candidate_slots"]) != 0:
        raise ValueError(f"{config['name']}: the anchor module reports candidate slots")
    if abs(float(info["relative_delta"])) <= 0.0:
        raise ValueError(f"{config['name']}: a zero residual at a non-zero quota")
    return dict(anchor_slots=int(offsets.numel()), relative_delta=float(info["relative_delta"]),
                gate=float(info["gate"]), check=True)


# ---------------------------------------------------------------------------
# one sample
# ---------------------------------------------------------------------------
def run_sample(model, tokenizer, manager, merge, anchor_module, prepared, args):
    """One write, then one compression per config, then every arm's answer."""
    chunks = [prepared["write_ids"][index:index + BLOCK_TOKENS]
              for index in range(0, len(prepared["write_ids"]), BLOCK_TOKENS)]
    answers = prepared["answers"]
    windows, configs, fingerprints = {}, {}, {}
    with torch.no_grad():
        es.write_blocks(manager, merge, chunks[:args.window_blocks])
        state = gr.snapshot(model, manager)
        source, ages, picked = gr.select_carriers(manager, args.window_blocks,
                                                  args.target_blocks)
        frozen, pinned = gr.window_locks(manager, args.window_blocks)
        for config in CONFIGS:
            set_quota(merge, config)
            output, telemetry = gr.compress_graft(manager, merge, source, picked,
                                                  args.window_blocks, args.target_blocks,
                                                  frozen, pinned)
            parts = split_source_anchors(telemetry, manager.block, args.window_blocks,
                                         device=source.device)
            window_check = check_window(config, output, telemetry, parts, source,
                                        manager, args)
            anchor_window, anchor_info = compact_window_anchor(output, parts, anchor_module)
            configs[config["name"]] = dict(
                core_per_block=config["core_per_block"],
                graft_per_block=config["graft_per_block"], **window_check,
                anchor=check_anchor(config, output, parts, anchor_window, anchor_info),
                dropped_blocks=[int(block) for block in parts["dropped_blocks"]],
                selected_blocks=[int(block) for block in telemetry["selected_blocks"]])
            graft_arm, anchor_arm = arms_of(config["name"])
            windows[graft_arm] = output
            windows[anchor_arm] = anchor_window
        set_quota(merge, CONFIGS[0])
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

    arms = {arm: gr.readout(model, tokenizer, prepared["query_ids"], answers,
                            args.f1_threshold, prepare(arm))
            for arm in all_arms()}
    gr.restore(model, manager, state)

    capacity = args.target_blocks * manager.block
    if int(fingerprints[FULL + "/slots"]) != args.window_blocks * manager.block:
        raise ValueError("the Full arm did not keep the whole window")
    if int(fingerprints[SELECTION + "/slots"]) != capacity:
        raise ValueError("the Selection arm does not hold the whole 2,304-slot budget")
    for config in CONFIGS:
        for arm in arms_of(config["name"]):
            if int(fingerprints[arm + "/slots"]) != capacity:
                raise ValueError(f"{arm} does not hold the {capacity}-slot budget")
        graft_arm, anchor_arm = arms_of(config["name"])
        if fingerprints[graft_arm] == fingerprints[anchor_arm]:
            raise ValueError(f"{graft_arm} and {anchor_arm} read the same pool")
        if fingerprints[graft_arm] == fingerprints[SELECTION]:
            raise ValueError(f"{graft_arm} reads the same pool as Selection")
    # Different quotas have to produce different windows, or the sweep moved nothing.
    quotas = [fingerprints[arms_of(config["name"])[0]] for config in CONFIGS]
    if len(set(quotas)) != len(quotas):
        raise ValueError("two configs produced the same window: the quota did not reach "
                         "the operator")

    dropped = [tuple(configs[config["name"]]["dropped_blocks"]) for config in CONFIGS]
    if len(set(dropped)) != 1:
        raise ValueError("the configs disagree on which blocks were dropped: the carrier "
                         "set is supposed to be shared")
    return dict(arms=arms, fingerprints=fingerprints, configs=configs,
                dropped_blocks=list(dropped[0]),
                selected_blocks=list(configs[CONFIGS[0]["name"]]["selected_blocks"]),
                carrier_ages=[int(ages[block]) for block in picked])


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


def budget_summary(samples, threshold):
    """Per config, and the comparison that decides the direction."""
    graded = [sample for sample in samples if sample.get("status") == "graded"]
    located = [sample for sample in graded if sample.get("evidence", {}).get("located")]
    dropped = [sample for sample in located if _evidence_in_dropped(sample)]
    kept = [sample for sample in located if not _evidence_in_dropped(sample)]

    def arms_and_pairs(rows):
        out = {}
        for config in CONFIGS:
            name = config["name"]
            graft_arm, anchor_arm = arms_of(name)
            out[name] = dict(
                arms={graft_arm: gr._arm_summary(rows, graft_arm, threshold),
                      anchor_arm: gr._arm_summary(rows, anchor_arm, threshold)},
                pairs={f"{anchor_arm}_minus_{graft_arm}":
                       er.pair_summary(rows, anchor_arm, graft_arm, threshold)})
        return out

    baseline = arms_of(CONFIGS[0]["name"])[1]
    versus = {}
    for config in CONFIGS[1:]:
        name = config["name"]
        graft_arm, anchor_arm = arms_of(name)
        versus[name] = dict(
            anchor_only=er.pair_summary(graded, anchor_arm, baseline, threshold),
            graft=er.pair_summary(graded, graft_arm, arms_of(CONFIGS[0]["name"])[0],
                                  threshold))
        versus[name]["dropped_evidence_group"] = dict(
            n=len(dropped),
            anchor_only=er.pair_summary(dropped, anchor_arm, baseline, threshold),
            graft=er.pair_summary(dropped, graft_arm, arms_of(CONFIGS[0]["name"])[0],
                                  threshold))
        versus[name]["kept_evidence_group"] = dict(
            n=len(kept),
            anchor_only=er.pair_summary(kept, anchor_arm, baseline, threshold),
            graft=er.pair_summary(kept, graft_arm, arms_of(CONFIGS[0]["name"])[0],
                                  threshold))

    allocation = {}
    for config in CONFIGS:
        name = config["name"]
        rows = [sample["configs"][name] for sample in graded]
        allocation[name] = dict(
            core_total=rows[0]["core_total"], graft_total=rows[0]["graft_total"],
            graft_share=rows[0]["graft_share"],
            candidate_slots=_stats([row["candidate_slots"] for row in rows]),
            empty_dropped_blocks_total=int(sum(row["empty_dropped_blocks"] for row in rows)),
            min_per_dropped_block=_stats([row["min_per_dropped_block"] for row in rows]),
            max_per_dropped_block=_stats([row["max_per_dropped_block"] for row in rows]),
            residual=_stats([row["anchor"]["relative_delta"] for row in rows]),
            gate=_stats([row["anchor"]["gate"] for row in rows]))

    return dict(
        definition=("the same 19-block window, the same nine carriers and the same 2,304 "
                    "slots; only the core/graft split inside each output block moves, and "
                    "the dropped blocks get the space the carriers give up"),
        n=len(graded), arms={arm: gr._arm_summary(graded, arm, threshold)
                             for arm in all_arms()},
        configs=arms_and_pairs(graded),
        versus_A=versus,
        dropped_evidence_group=dict(n=len(dropped), **arms_and_pairs(dropped)),
        kept_evidence_group=dict(n=len(kept), **arms_and_pairs(kept)),
        allocation=allocation,
        damage={arm: ef._damage(graded, arm, threshold) for arm in all_arms()})


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def build_protocol(args, task, anchor_config):
    return dict(
        experiment="5: does the 1944/360 core-graft split cost the dropped blocks?",
        treatment=("only `merge.core_slots_per_block` / `merge.graft_slots_per_block` "
                   "change; the window, the carriers, the 2,304-slot budget, the frozen "
                   "reader and the decoding do not"),
        task=task, data_dir=args.data_dir, arms=list(all_arms()),
        sweep=[dict(name=config["name"],
                    core_per_block=config["core_per_block"],
                    graft_per_block=config["graft_per_block"],
                    core_total=totals(config, args.target_blocks)[0],
                    graft_total=totals(config, args.target_blocks)[1],
                    graft_share=round(100.0 * totals(config, args.target_blocks)[1]
                                      / (args.target_blocks * SLOTS_PER_BLOCK), 2))
               for config in CONFIGS],
        window_blocks=args.window_blocks, target_blocks=args.target_blocks,
        block_tokens=BLOCK_TOKENS, slots_per_block=SLOTS_PER_BLOCK,
        full_slots=args.window_blocks * SLOTS_PER_BLOCK,
        target_slots=args.target_blocks * SLOTS_PER_BLOCK,
        unchanged_conditions=dict(
            data="the same 158 HotpotQA rows", reader="frozen MemoryLLM-8B",
            carriers="the same 9, selected once and shared by every arm of every config",
            window="the same 19-block write, document prefix only",
            capacity="2,304 slots in every config",
            operator_mode="native grafting, no fusion module in the Graft arms",
            decoding="greedy, unchanged"),
        anchor_only=dict(
            schema=ANCHOR_SCHEMA, **anchor_config, checkpoint=args.anchor,
            checkpoint_sha256=(sha(args.anchor) if args.anchor
                               and Path(args.anchor).is_file() else None),
            retrained=False,
            note=("the module was fitted at 360 anchors and is applied unchanged to 864 "
                  "and 1,008: this is a transfer, and whether it survives is the second "
                  "question the sweep answers")),
        checks=dict(
            capacity="every config fills 2,304 slots",
            split="graft anchors == 9 x graft_per_block, core == 9 x core_per_block",
            provenance="every anchor comes from a dropped block",
            verbatim_core="each core slot equals its claimed source position",
            order="the output stays in global chronological order",
            allocation="graft slots per dropped block, and how many blocks got none"),
        screening=("the first 158-row run is a screen: +5 questions for AnchorOnly_B or "
                   "_C over AnchorOnly_A (56) with the rescues in the dropped-evidence "
                   "group would justify adaptive budget allocation; 0-2 or a decline "
                   "means static quota tuning stops here"),
        reproduction=("config A is the anchor-only run's quota, so its four arms must "
                      "reproduce that report row for row before anything else is read"),
        screening_caveat=gr.SCREENING_CAVEAT, f1_threshold=args.f1_threshold,
        seed=args.seed, query=gr.QUERY_SUFFIX,
        scoring="official HotpotQA QA F1, F1 >= threshold is correct")


# The reference report calls config A's arms by the names they had there.
REFERENCE_ARMS = ((FULL, FULL), (SELECTION, SELECTION),
                  (gr.GRAFT if hasattr(gr, "GRAFT") else "Graft", arms_of("A")[0]),
                  ("AnchorOnly", arms_of("A")[1]))


def verify_against_report(args, reference, table, samples):
    """Full, Selection and config A's two arms must reproduce the anchor-only report.

    Config A IS that run's quota, so this is the check that the sweep's window
    construction is still experiment 3's before any of B or C is read. The arms are
    matched by name rather than by position: the reference calls them `Graft` and
    `AnchorOnly`, this driver calls them `Graft_A` and `AnchorOnly_A`.
    """
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
        verdict[reference_arm] = dict(
            matched_to=ours, identical=same, different=different,
            not_in_reference=absent, differing=differing[:20])
        passed = passed and different == 0 and absent == 0 and same == len(this_run)
    verdict["passed"] = passed
    return verdict


def load_anchor_module(path, hidden):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != ANCHOR_SCHEMA + "/checkpoint":
        raise ValueError(f"{path} is not a {ANCHOR_SCHEMA} checkpoint")
    config = dict(payload["config"])
    module = AnchorResidual(**config)
    module.load_state_dict(payload["state_dict"], strict=True)
    if int(config.get("hidden_size", -1)) != int(hidden):
        raise ValueError(f"{path} was built for hidden {config.get('hidden_size')}, the "
                         f"reader is {hidden}")
    return module.eval().requires_grad_(False), config


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
            sweep=[dict(name=config["name"],
                        core_per_block=config["core_per_block"],
                        graft_per_block=config["graft_per_block"],
                        core_total=totals(config, args.target_blocks)[0],
                        graft_total=totals(config, args.target_blocks)[1])
                   for config in CONFIGS],
            readouts_per_sample=len(all_arms())), indent=2, ensure_ascii=False))
        return None
    for flag, value, what in (("--checkpoint", args.checkpoint, "the frozen operator"),
                              ("--anchor", args.anchor, "the trained AnchorResidual")):
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
    if int(merge.core_slots_per_block) + int(merge.graft_slots_per_block) != manager.block:
        raise ValueError("the loaded operator does not split a block in two")
    anchor_module, anchor_config = load_anchor_module(args.anchor, manager.hidden)
    anchor_module = anchor_module.to(model.memory.device)
    if int(sum(parameter.numel() for parameter in anchor_module.parameters())) != 524289:
        raise ValueError("the anchor checkpoint is not the 524,289-parameter module this "
                         "sweep is supposed to transfer")

    rows, prepared, excluded = gr.prepare_rows(args, tokenizer, facts)
    report = dict(
        schema=SCHEMA_RESULTS, status="running",
        counts=dict(rows=len(rows), eligible=len(prepared), excluded=len(excluded)),
        protocol=build_protocol(args, args.task, anchor_config),
        identity=dict(
            model_path=args.model_path, checkpoint=args.checkpoint,
            checkpoint_sha256=sha(args.checkpoint),
            checkpoint_code_hash_verified=(saved or {}).get("code_hash_verified"),
            anchor=args.anchor, anchor_sha256=sha(args.anchor),
            data_dir=args.data_dir,
            data_sha256=sha(Path(args.data_dir) / (args.task + ".jsonl")),
            supporting_facts_glob=args.supporting_facts_glob,
            entrypoint_sha256=sha(__file__),
            anchor_module_sha256=sha(ROOT / "methods/prlmc/anchor_residual.py"),
            operator_quota_after_load=dict(
                core_per_block=int(merge.core_slots_per_block),
                graft_per_block=int(merge.graft_slots_per_block)),
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
        result = run_sample(model, tokenizer, manager, merge, anchor_module, sample, args)
        record = dict(id=sample["id"], status="graded",
                      available_blocks=sample["available_blocks"],
                      evidence=sample["evidence"],
                      seconds=time.monotonic() - sample_started, **result)
        report["samples"].append(record)
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output, report)
        print(json.dumps(dict(
            sample=position + 1, total=len(prepared), id=sample["id"],
            **{arm: result["arms"][arm]["correct"] for arm in all_arms()},
            seconds=round(record["seconds"], 1))), flush=True)

    report["summary"] = budget_summary(report["samples"], args.f1_threshold)
    if args.verify_against:
        report["verification_against_reference"] = verify_against_report(
            args, reference, table, report["samples"])
    report["elapsed_seconds"] = time.monotonic() - started
    report["status"] = "complete"
    report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    atomic_json(output, report)
    print(json.dumps(dict(status="complete", out=str(output),
                          elapsed_seconds=round(report["elapsed_seconds"], 1),
                          arms={arm: round(report["summary"]["arms"][arm]["accuracy"], 2)
                                for arm in all_arms()},
                          versus_A={name: value["anchor_only"]["net"]
                                    for name, value in report["summary"]["versus_A"].items()}),
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
                   help="the trained AnchorResidual, applied unchanged at every quota")
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
                   help="the finished anchor-only report; Full, Selection and config A's "
                        "two arms must reproduce it row for row")
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
