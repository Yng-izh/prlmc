#!/usr/bin/env python3
"""Experiment 2 evaluation: source compaction vs native grafting, HotpotQA.

Four readouts per row, all on the same pre-compression memory and the same 9
carriers:

    Full        19 blocks, uncompressed                  reference ceiling
    Selection   9 carrier blocks, verbatim               capacity baseline
    Graft       V10: 1,944 core + 360 copied graft slots the arm to beat
    Fusion      same window, but each of the 360 graft slots is a native anchor
                plus a learned residual from the unselected slots of its own
                source block

`Fusion` is the arm name `eval_learned_fusion` already analyses, so the grouping,
pairing and McNemar code is imported rather than re-typed; `protocol.treatment`
records what the treatment really is. Only the first compression is implemented:
experiment 2 runs the single compression first and recurses only if that is
positive, so `multi_round_group` is empty by construction rather than by accident.

Usage
-----
    python tools/eval_source_compaction.py --self-test        # no GPU

    python tools/eval_source_compaction.py --dry-run \
        --data-dir /data/.../LongBench/extracted/data

    python tools/eval_source_compaction.py \
        --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
        --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
        --model-path /dev/shm/s1d_reader_fp16 \
        --checkpoint runs/natural_merge_v9/full_v10.pt \
        --compaction runs/source_compaction/compaction_rank32.pt \
        --verify-against reports/diagnostics/graft_span_hotpotqa.json \
        --cuda-device 0 --out reports/diagnostics/source_compaction_hotpotqa.json
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
import native_eval as nv                                  # noqa: E402
from io_utils import atomic_json, digest, sha             # noqa: E402
from runtime import load_runtime, reset                   # noqa: E402
from methods.prlmc.core_fusion import prompt_hidden_states, readout_distance  # noqa: E402
from methods.prlmc.source_compaction import (             # noqa: E402
    SCHEMA, SourceCompaction, compact_window, split_source_anchors)

BLOCK = gr.SLOTS_PER_BLOCK
BLOCK_TOKENS = gr.BLOCK_TOKENS
CORE_SLOTS_TOTAL = gr.CORE_SLOTS_TOTAL
GRAFT_SLOTS_TOTAL = gr.GRAFT_SLOTS_TOTAL
WINDOW_BLOCKS = gr.WINDOW_BLOCKS
TARGET_BLOCKS = gr.TARGET_BLOCKS
TARGET_SLOTS = gr.TARGET_SLOTS
FULL, SELECTION, GRAFT, FUSION = ef.FULL, ef.SELECTION, ef.GRAFT, ef.FUSION
ROUND1_ARMS = (FULL, SELECTION, GRAFT, FUSION)
SCHEMA_RESULTS = "source_compaction_results/1"


# ---------------------------------------------------------------------------
# window preparation
# ---------------------------------------------------------------------------
def write_blocks(manager, merge, chunks):
    """Append `len(chunks)` native blocks through the plain (non-compacting) writer."""
    from stream_eval import StreamingMemory
    stream = StreamingMemory(manager, merge, "Full", budget_blocks=gr.BUDGET_BLOCKS)
    for index, chunk in enumerate(chunks):
        record = dict(role="write", token_ids=chunk, token_sha256=digest(chunk))
        if stream.write(record, index) is None:
            raise ValueError("the writer refused a block: the window does not fit the pool")
    return stream


def compress_window(manager, merge, module, window_blocks, target_blocks):
    """One compression, its compaction, and the structure both are checked against."""
    source, ages, picked = gr.select_carriers(manager, window_blocks, target_blocks)
    frozen, pinned = gr.window_locks(manager, window_blocks)
    with torch.no_grad():
        output, telemetry = gr.compress_graft(manager, merge, source, picked,
                                              window_blocks, target_blocks,
                                              frozen, pinned)
        parts = split_source_anchors(telemetry, manager.block, window_blocks,
                                     device=source.device)
        compacted, info = compact_window(output, source, parts, module)
    return dict(source=source, ages=ages, picked=picked, output=output,
                fused=compacted, parts=parts, telemetry=telemetry,
                compaction_info=info)


def check_compaction(result):
    """The compacted window must differ from the grafted one at the graft slots only."""
    telemetry = result["telemetry"]
    output, compacted = result["output"], result["fused"]
    parts = result["parts"]
    device = output.device
    anchors = parts["anchor_offsets"]
    if int(anchors.numel()) != GRAFT_SLOTS_TOTAL:
        raise ValueError(f"{int(anchors.numel())} anchors, expected {GRAFT_SLOTS_TOTAL}")
    if int(parts["candidate_positions"].numel()) <= int(anchors.numel()):
        raise ValueError("fewer candidate slots than anchors: nothing was compacted")
    keep = torch.ones(output.shape[1], dtype=torch.bool, device=device)
    keep[anchors] = False
    core = torch.nonzero(keep).flatten()
    if int(core.numel()) != CORE_SLOTS_TOTAL:
        raise ValueError(f"{int(core.numel())} core slots, expected {CORE_SLOTS_TOTAL}")
    if telemetry["selected_blocks"] != sorted(int(b) for b in result["picked"]):
        raise ValueError("the compression disagrees with the carrier set")
    if not torch.equal(output.index_select(1, core), compacted.index_select(1, core)):
        raise ValueError("the compaction changed a CORE slot; the two arms no longer "
                         "differ only in the graft slots")
    if torch.equal(output.index_select(1, anchors), compacted.index_select(1, anchors)):
        raise ValueError("the compaction left every graft slot unchanged, so the arm "
                         "measures nothing")
    if not torch.isfinite(compacted).all():
        raise ValueError("the compacted window is not finite")
    dropped = set(int(b) for b in parts["dropped_blocks"])
    if set(int(b) for b in parts["anchor_blocks"].tolist()) - dropped:
        raise ValueError("an anchor came from a block the compaction kept")
    return dict(
        core_slots=int(core.numel()), graft_slots=int(anchors.numel()),
        candidate_slots=int(parts["candidate_positions"].numel()),
        dropped_blocks=len(dropped),
        anchors_per_block=sorted(set(parts["anchors_per_block"].values())),
        candidates_per_block=sorted(set(parts["candidates_per_block"].values())),
        core_bit_identical=True, graft_changed=True,
        relative_delta=round(result["compaction_info"]["relative_delta"], 6),
        gate=round(result["compaction_info"]["gate"], 6))


def readout_gap(model, source, compressed, query_hidden, layers):
    pre = {layer: source[layer] for layer in layers}
    post = {layer: compressed[layer] for layer in layers}
    with torch.no_grad():
        return float(readout_distance(model, pre, post, query_hidden, layers))


# ---------------------------------------------------------------------------
# one sample
# ---------------------------------------------------------------------------
def run_sample(model, tokenizer, manager, merge, module, prepared, args):
    """Write the window once, compress once, then answer all four arms."""
    chunks = [prepared["write_ids"][index:index + BLOCK_TOKENS]
              for index in range(0, len(prepared["write_ids"]), BLOCK_TOKENS)]
    answers = prepared["answers"]
    fingerprints, gaps = {}, {}
    probe = ef.ALL_LAYERS if getattr(args, "per_layer", False) else ef.TAP_LAYERS
    with torch.no_grad():
        write_blocks(manager, merge, chunks[:args.window_blocks])
        state = gr.snapshot(model, manager)
        query_hidden = prompt_hidden_states(model, prepared["query_ids"], probe)
        round1 = compress_window(manager, merge, module, args.window_blocks,
                                 args.target_blocks)
    # Keyed by arm because the shared analysis reads `structure[Graft]`; the same
    # check covers both arms here -- the compaction changes the graft slots and
    # leaves the core alone, the mirror image of experiment 1B.
    check = check_compaction(round1)
    structure = {SELECTION: None, GRAFT: check, FUSION: check}

    selection = ef.selection_window(round1["source"], round1["picked"])
    gaps[SELECTION] = readout_gap(model, round1["source"], selection,
                                  query_hidden, ef.TAPS)
    gaps[GRAFT] = readout_gap(model, round1["source"], round1["output"],
                              query_hidden, ef.TAPS)
    gaps[FUSION] = readout_gap(model, round1["source"], round1["fused"],
                               query_hidden, ef.TAPS)
    gaps[GRAFT + "/heldout"] = readout_gap(model, round1["source"], round1["output"],
                                           query_hidden, ef.HELD_OUT_LAYERS)
    gaps[FUSION + "/heldout"] = readout_gap(model, round1["source"], round1["fused"],
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
            fingerprints[arm] = gr.active_fingerprint(manager)
            fingerprints[arm + "/slots"] = int(manager.used)
        return run

    arms = {arm: gr.readout(model, tokenizer, prepared["query_ids"], answers,
                            args.f1_threshold, prepare(arm))
            for arm in ROUND1_ARMS}
    gr.restore(model, manager, state)

    if int(fingerprints[FULL + "/slots"]) != args.window_blocks * manager.block:
        raise ValueError("the Full arm did not keep the whole window")
    for arm in (SELECTION, GRAFT, FUSION):
        if int(fingerprints[arm + "/slots"]) != TARGET_SLOTS:
            raise ValueError(f"arm {arm} does not hold the {TARGET_SLOTS}-slot budget")
    if fingerprints[SELECTION] == fingerprints[GRAFT]:
        raise ValueError("Selection and Graft read the same pool: no slot was grafted")
    if fingerprints[GRAFT] == fingerprints[FUSION]:
        raise ValueError("Graft and Fusion read the same pool: the compaction did not run")

    provenance = gr.graft_provenance(round1["telemetry"], args.window_blocks)
    if provenance["selected_blocks"] != sorted(round1["picked"]):
        raise ValueError("graft provenance disagrees with the shared carrier set")
    return dict(arms=arms, fingerprints=fingerprints, gaps=gaps,
                provenance=provenance, structure=structure, multi_round=False,
                carrier_ages=[int(round1["ages"][block]) for block in round1["picked"]])


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def self_test():
    """Check the module's structural promises on a synthetic window, on the CPU.

    No reader and no checkpoint: this is the part that can be wrong while every GPU
    number still looks plausible -- which slots are the anchors, which slots may an
    anchor read, and whether a zero output projection really is the identity.
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
    if parts["anchor_offsets"].tolist() != offsets:
        raise ValueError("the anchors are not the graft slots")
    anchors = [full_positions[offset] for offset in offsets]
    if set(parts["candidate_positions"].tolist()) & set(anchors):
        raise ValueError("an anchor is also its own candidate")
    if set(int(p) // block for p in parts["candidate_positions"].tolist()) - set(dropped):
        raise ValueError("a candidate came from a block that was not dropped")
    output = torch.randn(layers, target_blocks * block, hidden)
    module = SourceCompaction(hidden_size=hidden, rank=4).eval().requires_grad_(False)
    compacted, info = compact_window(output, window, parts, module)
    if not torch.equal(compacted, output):
        raise ValueError("a zero output projection did not reproduce the grafted window")
    with torch.no_grad():
        module.out_proj.normal_(std=0.05)
    compacted, info = compact_window(output, window, parts, module)
    keep = torch.ones(output.shape[1], dtype=torch.bool)
    keep[parts["anchor_offsets"]] = False
    if not torch.equal(output[:, keep], compacted[:, keep]):
        raise ValueError("the compaction changed a core slot")
    if torch.equal(output[:, parts["anchor_offsets"]],
                   compacted[:, parts["anchor_offsets"]]):
        raise ValueError("the compaction changed nothing after out_proj left zero")
    if not torch.isfinite(compacted).all():
        raise ValueError("the compacted window is not finite")
    print(json.dumps(dict(
        self_test="ok", block=block, core=core, graft_per_block=graft_per_block,
        anchors=int(parts["anchor_offsets"].numel()),
        candidates=int(parts["candidate_positions"].numel()),
        anchors_per_block=parts["anchors_per_block"],
        candidates_per_block=parts["candidates_per_block"],
        identity_at_init=True, core_bit_identical=True, graft_changed=True,
        relative_delta=round(info["relative_delta"], 5)), indent=2))
    return 0


def build_protocol(args, task, module_config):
    return dict(
        experiment="2 source compaction: many-to-one knowledge into the 360 graft slots",
        treatment="the 360 graft slots are a native anchor plus a learned residual "
                  "from the unselected slots of the anchor's own source block",
        task=task, data_dir=args.data_dir, arms=list(ROUND1_ARMS),
        window_blocks=args.window_blocks, target_blocks=args.target_blocks,
        block_tokens=BLOCK_TOKENS, slots_per_block=BLOCK,
        full_slots=args.window_blocks * BLOCK, target_slots=TARGET_SLOTS,
        core_slots_total=CORE_SLOTS_TOTAL, graft_slots_total=GRAFT_SLOTS_TOTAL,
        dropped_blocks=args.window_blocks - args.target_blocks,
        dropped_slots=(args.window_blocks - args.target_blocks) * BLOCK,
        compressions_per_sample=1,
        unchanged_conditions=dict(
            data="the same 158 HotpotQA rows as experiments 1, 2, 3, 1A and 1B",
            reader="frozen MemoryLLM-8B", carriers="the same 9, unchanged",
            core_selection="original single-point top-216 per carrier",
            source_selection="V10's trained single-point leverage rule, unchanged",
            source_capacity=GRAFT_SLOTS_TOTAL, output_capacity=TARGET_SLOTS,
            output_order="the existing global chronological order",
            core_vectors="native copies on both sides; the core slots are the only "
                         "ones experiment 1B changed, and here they are byte-identical"),
        only_difference=("each of the 360 graft slots is its native anchor plus a "
                         "bounded residual pulled from the unselected slots of the "
                         "anchor's own source block"),
        compaction=dict(schema=SCHEMA, **module_config,
                        checkpoint=args.compaction,
                        checkpoint_sha256=(sha(args.compaction) if args.compaction
                                           and Path(args.compaction).is_file() else None),
                        candidates="every unselected slot of the anchor's source block",
                        training="offline, answer cross-entropy on a disjoint LongBench "
                                 "task; nothing trained here"),
        operator_mode="native",
        per_layer_probe=bool(getattr(args, "per_layer", False)),
        round2="not implemented: the single compression is run first",
        write_stream="document prefix only, 19 native blocks",
        query=gr.QUERY_SUFFIX, scoring="official HotpotQA QA F1, F1 >= threshold is "
                                       "correct, gold likelihood recorded alongside",
        f1_threshold=args.f1_threshold, seed=args.seed,
        screening_caveat=gr.SCREENING_CAVEAT,
        fusion_training=None)


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
            dry_run=True, out=args.out, counts=dict(rows=len(rows),
                                                    eligible=len(prepared),
                                                    excluded=len(excluded)),
            eligible_ids=[sample["id"] for sample in prepared],
            readouts_per_sample=len(ROUND1_ARMS)), indent=2, ensure_ascii=False))
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
    for position, sample in enumerate(prepared):
        sample_started = time.monotonic()
        nv.seed_everything(args.seed)
        reset(manager, initial)
        result = run_sample(model, tokenizer, manager, merge, module, sample, args)
        record = dict(id=sample["id"], status="graded",
                      available_blocks=sample["available_blocks"],
                      evidence=sample["evidence"],
                      seconds=time.monotonic() - sample_started, **result)
        report["samples"].append(record)
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output, report)
        print(json.dumps(dict(
            sample=position + 1, total=len(prepared), id=sample["id"],
            selection=result["arms"][SELECTION]["correct"],
            graft=result["arms"][GRAFT]["correct"],
            fusion=result["arms"][FUSION]["correct"],
            gap_removed=round(result["gaps"][FUSION] / result["gaps"][GRAFT], 4)
            if result["gaps"].get(GRAFT) else None,
            delta=result["structure"][GRAFT]["relative_delta"],
            seconds=round(record["seconds"], 1))), flush=True)

    report["summary"] = ef.analyse(report["samples"], args.f1_threshold,
                                   excluded=len(report["excluded"]))
    if args.verify_against:
        report["verification_against_reference"] = ef.verify_against(
            args.verify_against, reference, table, report["samples"])
    report["elapsed_seconds"] = time.monotonic() - started
    report["status"] = "complete"
    report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    atomic_json(output, report)
    print(json.dumps(dict(status="complete", out=str(output),
                          elapsed_seconds=round(report["elapsed_seconds"], 1),
                          pairs={key: value for key, value in
                                 report["summary"]["pairs"].items()}), indent=1,
                     ensure_ascii=False))
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
    p.add_argument("--window-blocks", dest="window_blocks", type=int, default=WINDOW_BLOCKS)
    p.add_argument("--target-blocks", dest="target_blocks", type=int, default=TARGET_BLOCKS)
    p.add_argument("--f1-threshold", dest="f1_threshold", type=float, default=0.5)
    p.add_argument("--sample-ids", dest="sample_ids", default=None)
    p.add_argument("--max-samples", dest="max_samples", type=int, default=None)
    p.add_argument("--per-layer", dest="per_layer", action="store_true",
                   help="also record the readout distance at all 32 reader layers")
    p.add_argument("--verify-against", dest="verify_against", default=None,
                   help="a finished experiment-1/1A report; Full, Selection and Graft "
                        "must reproduce it row for row")
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
