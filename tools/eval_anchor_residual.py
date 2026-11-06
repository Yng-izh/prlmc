#!/usr/bin/env python3
"""Round 3, experiment 1: how much of the +10.13 points needs a candidate slot?

The counterfactual ablation replaced the 2,200 candidate slots with another
document's and every one of the 158 answers stayed put. That rules out the
IDENTITY of the candidates; it does not rule out the candidate PATH. This driver
answers the remaining question by deleting the path:

    Full            19 blocks, uncompressed                    reference ceiling
    Selection       9 carrier blocks, verbatim                 capacity baseline
    Graft           V10, 360 native anchors                    the arm to beat
    Fusion          + the experiment-2 module on its block's leftovers
    AnchorOnly      + a module that sees the anchor and nothing else

`AnchorOnly` is trained by the same script, on the same data, for the same number
of updates, with the same loss and the same residual bounds, and its rank is
chosen so that it has exactly the same parameter count as the experiment-2 module
(524,289). The two arms therefore differ in one thing: whether the residual may
read a candidate at all.

Reading the result
------------------
* AnchorOnly near Fusion   the gain is a representation adjustment of the anchors
* AnchorOnly near Graft    the candidate path does contribute something -- not the
                           slots' identity, but a statistic of the block

The four old arms are re-run in the same process, so every comparison is paired
row for row and `--verify-against` can prove that this run reproduces experiment 2
before the new arm is read at all.

Usage
-----
    python tools/eval_anchor_residual.py --self-test        # no GPU

    python tools/eval_anchor_residual.py \
        --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
        --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
        --model-path /dev/shm/s1d_reader_fp16 \
        --checkpoint runs/natural_merge_v9/full_v10.pt \
        --compaction runs/source_compaction/compaction_rank32.pt \
        --anchor runs/anchor_residual/anchor_only_rank64.pt \
        --verify-against reports/exp2_source_compaction/eval_trained_hotpotqa.json \
        --cuda-device 0 \
        --out reports/exp3_anchor_residual/eval_anchor_only_hotpotqa.json
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

import eval_counterfactual_candidates as cf               # noqa: E402
import eval_graft_recovery as gr                          # noqa: E402
import eval_learned_fusion as ef                          # noqa: E402
import eval_replace_reorder as er                         # noqa: E402
import eval_source_compaction as es                       # noqa: E402
import native_eval as nv                                  # noqa: E402
from io_utils import atomic_json, sha                     # noqa: E402
from runtime import load_runtime, reset                   # noqa: E402
from methods.prlmc.core_fusion import prompt_hidden_states  # noqa: E402
from methods.prlmc.anchor_residual import (               # noqa: E402
    SCHEMA as ANCHOR_SCHEMA, AnchorResidual,
    compact_window as compact_window_anchor)
from methods.prlmc.source_compaction import (             # noqa: E402
    SCHEMA, SourceCompaction, compact_window, split_source_anchors)

BLOCK_TOKENS = gr.BLOCK_TOKENS
TARGET_SLOTS = gr.TARGET_SLOTS
FULL, SELECTION, GRAFT, FUSION = ef.FULL, ef.SELECTION, ef.GRAFT, ef.FUSION
ANCHOR_ONLY = "AnchorOnly"
ARMS = (FULL, SELECTION, GRAFT, FUSION, ANCHOR_ONLY)
SCHEMA_RESULTS = "anchor_residual_results/1"


# ---------------------------------------------------------------------------
# window preparation
# ---------------------------------------------------------------------------
def compress_window(manager, merge, source_module, anchor_module, args):
    """One write, one compression, and the two compacted windows built from it."""
    source, ages, picked = gr.select_carriers(manager, args.window_blocks,
                                              args.target_blocks)
    frozen, pinned = gr.window_locks(manager, args.window_blocks)
    with torch.no_grad():
        output, telemetry = gr.compress_graft(manager, merge, source, picked,
                                              args.window_blocks, args.target_blocks,
                                              frozen, pinned)
        parts = split_source_anchors(telemetry, manager.block, args.window_blocks,
                                     device=source.device)
        fused, info = compact_window(output, source, parts, source_module)
        anchor, anchor_info = compact_window_anchor(output, parts, anchor_module)
    return dict(source=source, ages=ages, picked=picked, output=output,
                fused=fused, parts=parts, telemetry=telemetry,
                compaction_info=info, anchor_window=anchor,
                anchor_info=anchor_info)


def check_anchor(result):
    """The anchor-only window must change the graft slots only, and must change them."""
    check = es.check_compaction(dict(result, fused=result["anchor_window"],
                                     compaction_info=result["anchor_info"]))
    anchors = result["parts"]["anchor_offsets"]
    if int(result["anchor_info"]["candidate_slots"]) != 0:
        raise ValueError("the anchor-only module reports candidate slots: the path "
                         "that was supposed to be deleted is still there")
    if torch.equal(result["anchor_window"].index_select(1, anchors),
                   result["fused"].index_select(1, anchors)):
        raise ValueError("anchor-only reproduced the Fusion window exactly: the two "
                         "modules are the same module")
    if torch.equal(result["anchor_window"].index_select(1, anchors),
                   result["output"].index_select(1, anchors)):
        raise ValueError("anchor-only left the graft slots at the grafted values")
    return check


# ---------------------------------------------------------------------------
# one sample
# ---------------------------------------------------------------------------
def run_sample(model, tokenizer, manager, merge, source_module, anchor_module,
               prepared, args):
    """Write the window once, compress once, then answer all five arms."""
    chunks = [prepared["write_ids"][index:index + BLOCK_TOKENS]
              for index in range(0, len(prepared["write_ids"]), BLOCK_TOKENS)]
    answers = prepared["answers"]
    fingerprints, gaps = {}, {}
    with torch.no_grad():
        es.write_blocks(manager, merge, chunks[:args.window_blocks])
        state = gr.snapshot(model, manager)
        query_hidden = prompt_hidden_states(model, prepared["query_ids"], ef.TAP_LAYERS)
        round1 = compress_window(manager, merge, source_module, anchor_module, args)
    check = es.check_compaction(round1)
    anchor_check = check_anchor(round1)
    structure = {SELECTION: None, GRAFT: check, FUSION: check}

    selection = ef.selection_window(round1["source"], round1["picked"])
    gaps[SELECTION] = es.readout_gap(model, round1["source"], selection,
                                     query_hidden, ef.TAPS)
    for arm, window in ((GRAFT, round1["output"]), (FUSION, round1["fused"]),
                        (ANCHOR_ONLY, round1["anchor_window"])):
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
            elif arm == ANCHOR_ONLY:
                er.install_output(manager, round1["anchor_window"], round1["picked"],
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
    for arm in (SELECTION, GRAFT, FUSION, ANCHOR_ONLY):
        if int(fingerprints[arm + "/slots"]) != TARGET_SLOTS:
            raise ValueError(f"arm {arm} does not hold the {TARGET_SLOTS}-slot budget")
    for left, right in ((SELECTION, GRAFT), (GRAFT, FUSION), (FUSION, ANCHOR_ONLY)):
        if fingerprints[left] == fingerprints[right]:
            raise ValueError(f"{left} and {right} read the same pool: the manipulation "
                             f"did not reach the memory")

    provenance = gr.graft_provenance(round1["telemetry"], args.window_blocks)
    if provenance["selected_blocks"] != sorted(round1["picked"]):
        raise ValueError("graft provenance disagrees with the shared carrier set")
    return dict(arms=arms, fingerprints=fingerprints, gaps=gaps,
                provenance=provenance, structure=structure, multi_round=False,
                anchor=dict(**round1["anchor_info"], check=anchor_check),
                carrier_ages=[int(round1["ages"][block]) for block in round1["picked"]])


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def _stats(values):
    values = [float(value) for value in values]
    if not values:
        return dict(n=0)
    return dict(n=len(values), mean=round(sum(values) / len(values), 6),
                median=round(sorted(values)[len(values) // 2], 6))


def anchor_summary(samples, threshold):
    """The block the driver exists for: AnchorOnly against Fusion and against Graft."""
    graded = [sample for sample in samples if sample.get("status") == "graded"]
    located = [sample for sample in graded if sample.get("evidence", {}).get("located")]
    dropped = [sample for sample in located if ef._evidence_in_dropped(sample)]
    kept = [sample for sample in located if not ef._evidence_in_dropped(sample)]

    def comparisons(rows):
        return {
            f"{ANCHOR_ONLY}_minus_{FUSION}":
                er.pair_summary(rows, ANCHOR_ONLY, FUSION, threshold),
            f"{ANCHOR_ONLY}_minus_{GRAFT}":
                er.pair_summary(rows, ANCHOR_ONLY, GRAFT, threshold),
            f"{FUSION}_minus_{GRAFT}": er.pair_summary(rows, FUSION, GRAFT, threshold),
        }

    def group(rows):
        return dict(n=len(rows),
                    arms={arm: gr._arm_summary(rows, arm, threshold) for arm in ARMS},
                    pairs=comparisons(rows))

    return dict(
        definition=("the same 360 anchors, the same training and the same parameter "
                    "count, with the 2,200-slot candidate path deleted instead of "
                    "replaced"),
        n=len(graded),
        arms={arm: gr._arm_summary(graded, arm, threshold) for arm in ARMS},
        pairs=comparisons(graded),
        dropped_evidence_group=group(dropped),
        kept_evidence_group=group(kept),
        damage={arm: ef._damage(graded, arm, threshold)
                for arm in (GRAFT, FUSION, ANCHOR_ONLY)},
        residual=dict(
            definition="RMS of the injected residual over RMS of the anchor",
            fusion=_stats([sample["structure"][FUSION]["relative_delta"]
                           for sample in graded]),
            anchor_only=_stats([sample["anchor"]["relative_delta"]
                                for sample in graded]),
            candidate_slots=_stats([sample["anchor"]["candidate_slots"]
                                    for sample in graded])),
        readout=dict(
            definition="mean reader-readout distance from the uncompressed window",
            by_arm={arm: ef._gap_summary(graded, arm)
                    for arm in (GRAFT, FUSION, ANCHOR_ONLY)},
            held_out={arm: ef._gap_summary(graded, arm + "/heldout")
                      for arm in (GRAFT, FUSION, ANCHOR_ONLY)},
            anchor_removes=ef._gap_removed(graded, ANCHOR_ONLY, GRAFT)),
    )


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def self_test():
    """The anchor-only path's structural promises, on a synthetic window, on the CPU.

    No reader and no checkpoint: which slots move, which must not, whether a zero
    output projection is the identity, and whether the module can see a candidate
    at all -- the part that can be wrong while every GPU number still looks fine.
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
    module = AnchorResidual(hidden_size=hidden, rank=4).eval().requires_grad_(False)
    compacted, info = compact_window_anchor(output, parts, module)
    if not torch.equal(compacted, output):
        raise ValueError("a zero output projection did not reproduce the grafted window")
    if int(info["candidate_slots"]) != 0:
        raise ValueError("the anchor-only module reports a candidate slot")
    with torch.no_grad():
        module.up.normal_(std=0.05)
    compacted, info = compact_window_anchor(output, parts, module)
    keep = torch.ones(output.shape[1], dtype=torch.bool)
    keep[parts["anchor_offsets"]] = False
    if not torch.equal(output[:, keep], compacted[:, keep]):
        raise ValueError("the anchor-only residual changed a core slot")
    if torch.equal(output[:, parts["anchor_offsets"]],
                   compacted[:, parts["anchor_offsets"]]):
        raise ValueError("the anchor-only residual changed nothing")
    if not torch.isfinite(compacted).all():
        raise ValueError("the anchor-only window is not finite")
    source_module = SourceCompaction(hidden_size=hidden, rank=2).eval().requires_grad_(False)
    with torch.no_grad():
        source_module.out_proj.normal_(std=0.05)
    fused, _ = compact_window(output, window, parts, source_module)
    if torch.equal(fused, compacted):
        raise ValueError("the two modules produced the same window: this run would "
                         "compare a module against itself")
    print(json.dumps(dict(
        self_test="ok", anchors=int(parts["anchor_offsets"].numel()),
        candidate_slots=int(info["candidate_slots"]),
        identity_at_init=True, core_bit_identical=True, graft_changed=True,
        forward_takes=("anchors",), relative_delta=round(info["relative_delta"], 5)),
        indent=2))
    return 0


def build_protocol(args, task, source_config, anchor_config):
    protocol = es.build_protocol(args, task, source_config)
    protocol.update(
        experiment="3-1: how much of the gain needs a candidate slot?",
        treatment=("the 360 anchors are adjusted by a module that sees the anchor "
                   "alone; the 2,200-slot candidate path is deleted, not replaced"),
        arms=list(ARMS),
        anchor_only=dict(
            schema=ANCHOR_SCHEMA, **anchor_config,
            checkpoint=args.anchor,
            checkpoint_sha256=(sha(args.anchor) if args.anchor
                               and Path(args.anchor).is_file() else None),
            input="the 360 native anchors and nothing else (`forward(anchors)`)",
            candidates="none: the module has no candidate argument to pass one to",
            unchanged=dict(
                anchors="the 360 native anchors, byte-identical",
                training="the same script, data, updates, learning rate, loss, bounds",
                parameters="equal by construction: 4 x 32 x 4096 + 1 against "
                           "2 x 64 x 4096 + 1, both 524,289",
                core_slots="byte-identical to the grafted window",
                carriers="the same 9", reader="frozen MemoryLLM-8B",
                data="the same 158 HotpotQA rows", decoding="greedy, unchanged"),
            only_difference="whether the residual may read a candidate slot",
            reading=("AnchorOnly close to Fusion means the gain is a representation "
                     "adjustment of the anchors; close to Graft means the candidate "
                     "path contributes something beyond the slots' identity")),
        pairing_note=("all five arms run in one process on one pre-compression window; "
                      "the four old arms must reproduce the experiment-2 report before "
                      "the new arm is read"),
        compaction=dict(protocol["compaction"],
                        candidates="every unselected slot of the anchor's source block"))
    return protocol


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
            readouts_per_sample=len(ARMS)), indent=2, ensure_ascii=False))
        return None
    for flag, value, what in (("--checkpoint", args.checkpoint, "the frozen operator"),
                              ("--compaction", args.compaction, "the experiment-2 module"),
                              ("--anchor", args.anchor, "the anchor-only module")):
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

    def load_module(path, schema, factory):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema") != schema + "/checkpoint":
            raise ValueError(f"{path} is not a {schema} checkpoint")
        config = dict(payload["config"])
        module = factory(**config)
        module.load_state_dict(payload["state_dict"], strict=True)
        if config.get("hidden_size") != int(manager.hidden):
            raise ValueError(f"{path} was built for hidden {config.get('hidden_size')}, "
                             f"the reader is {manager.hidden}")
        return module.to(model.memory.device).eval().requires_grad_(False), config

    source_module, source_config = load_module(args.compaction, SCHEMA, SourceCompaction)
    anchor_module, anchor_config = load_module(args.anchor, ANCHOR_SCHEMA, AnchorResidual)
    parameters = dict(source=sum(p.numel() for p in source_module.parameters()),
                      anchor=sum(p.numel() for p in anchor_module.parameters()))
    if parameters["source"] != parameters["anchor"]:
        raise ValueError(f"the two modules are not the same size: {parameters}: the "
                         f"comparison would confound capacity with architecture")

    rows, prepared, excluded = gr.prepare_rows(args, tokenizer, facts)
    report = dict(
        schema=SCHEMA_RESULTS, status="running",
        counts=dict(rows=len(rows), eligible=len(prepared), excluded=len(excluded)),
        protocol=build_protocol(args, args.task, source_config, anchor_config),
        identity=dict(
            model_path=args.model_path, checkpoint=args.checkpoint,
            checkpoint_sha256=sha(args.checkpoint),
            checkpoint_code_hash_verified=(saved or {}).get("code_hash_verified"),
            compaction=args.compaction, compaction_sha256=sha(args.compaction),
            anchor=args.anchor, anchor_sha256=sha(args.anchor),
            parameters=parameters,
            data_dir=args.data_dir,
            data_sha256=sha(Path(args.data_dir) / (args.task + ".jsonl")),
            supporting_facts_glob=args.supporting_facts_glob,
            entrypoint_sha256=sha(__file__),
            compaction_module_sha256=sha(ROOT / "methods/prlmc/source_compaction.py"),
            anchor_module_sha256=sha(ROOT / "methods/prlmc/anchor_residual.py"),
            reader_frozen=True),
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
        result = run_sample(model, tokenizer, manager, merge, source_module,
                            anchor_module, sample, args)
        record = dict(id=sample["id"], status="graded",
                      available_blocks=sample["available_blocks"],
                      evidence=sample["evidence"],
                      seconds=time.monotonic() - sample_started, **result)
        report["samples"].append(record)
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output, report)
        print(json.dumps(dict(
            sample=position + 1, total=len(prepared), id=sample["id"],
            graft=result["arms"][GRAFT]["correct"],
            fusion=result["arms"][FUSION]["correct"],
            anchor_only=result["arms"][ANCHOR_ONLY]["correct"],
            delta_fusion=result["structure"][FUSION]["relative_delta"],
            delta_anchor=record["anchor"]["relative_delta"],
            seconds=round(record["seconds"], 1))), flush=True)

    report["summary"] = ef.analyse(report["samples"], args.f1_threshold,
                                   excluded=len(report["excluded"]))
    report["summary"]["anchor_only"] = anchor_summary(report["samples"],
                                                      args.f1_threshold)
    if args.verify_against:
        report["verification_against_reference"] = cf.verify_reference(
            args, reference, table, report["samples"])
    report["elapsed_seconds"] = time.monotonic() - started
    report["status"] = "complete"
    report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    atomic_json(output, report)
    print(json.dumps(dict(status="complete", out=str(output),
                          elapsed_seconds=round(report["elapsed_seconds"], 1),
                          pairs={key: value for key, value in
                                 report["summary"]["anchor_only"]["pairs"].items()}),
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
                   help="the trained SourceCompaction checkpoint (the Fusion arm)")
    p.add_argument("--anchor", default=None,
                   help="the trained AnchorResidual checkpoint (the AnchorOnly arm)")
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
