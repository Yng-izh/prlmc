#!/usr/bin/env python3
"""Diagnostic: how much of V13's 91 wrong rows is retrieval, and how much is the reader.

Two questions, both answered without training anything:

1. RETRIEVAL. Of the rows V13 answers wrong, how many have their supporting-fact blocks
   among the 10 DROPPED blocks and no evidence slot surviving into the window at all?
   Those rows are lost by the slot selection, not by the calibration.

2. CEILING. How many of the 158 rows does the frozen reader fail even when it is handed
   ONLY the golden blocks -- the raw blocks that contain the supporting facts, no
   compression, no calibration? That number is the hard ceiling this base model has on
   this row set, and no amount of memory engineering can cross it.

Both are measured on the same window V13 reads (`--v11-evict`'s geometry), so the pieces
are comparable: a row counted in (1) is a selection miss; a row counted in (2) is a
reader miss; the remainder is where a better calibration could still win.

    python tools/diagnose_v13_oracle.py \
        --data-dir ... --task hotpotqa --supporting-facts-glob '...' \
        --model-path /dev/shm/s1d_reader_fp16 \
        --checkpoint runs/natural_merge_v9/full_v10.pt \
        --anchor runs/anchor_residual/anchor_only_rank64.pt \
        --v11-evict runs/cross_block/cross_lr.pt \
        --v13-report reports/cross_block/eval_v16.json \
        --cuda-device 0 --out reports/cross_block/diagnose_v13_oracle.json
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

import eval_cross_block_calibration as ev                      # noqa: E402
import eval_graft_recovery as gr                               # noqa: E402
import eval_source_compaction as es                            # noqa: E402
import native_eval as nv                                       # noqa: E402
from io_utils import atomic_json, sha                          # noqa: E402
from runtime import load_runtime, reset                        # noqa: E402
from methods.prlmc.reader_utility import ReaderScorer          # noqa: E402

SCHEMA = "v13_oracle_diagnosis/1"
BLOCK_TOKENS = gr.BLOCK_TOKENS


def chunks_of(prepared, blocks, limit):
    """The prepared row's token chunks, restricted to `blocks`, in block order."""
    chunks = [prepared["write_ids"][index:index + BLOCK_TOKENS]
              for index in range(0, len(prepared["write_ids"]), BLOCK_TOKENS)]
    out = []
    for block in sorted(set(blocks)):
        if 0 <= int(block) < min(limit, len(chunks)):
            out.append(chunks[int(block)])
    return out


def blank_chunk(tokenizer, length):
    """A chunk of `length` neutral tokens -- same shape, no content."""
    ids = tokenizer(" .", add_special_tokens=False)["input_ids"]
    token = int(ids[-1])
    return [token] * int(length)


def golden_answer(model, tokenizer, manager, merge, initial, prepared, blocks, args):
    """Greedy answer with the golden evidence ALONE inside the normal 19-block pool.

    The pool keeps its usual shape and occupancy: all 19 blocks are written, the golden
    blocks verbatim, every other block filled with the same number of neutral tokens.
    Writing only the two golden blocks instead (the first version of this tool) changes
    the pool's size and the reader's regime, and it scored 18/158 -- below Full's 47 and
    below V13's 67, i.e. it was measuring the regime change, not the reader's ability.
    """
    chunks = [prepared["write_ids"][index:index + BLOCK_TOKENS]
              for index in range(0, len(prepared["write_ids"]), BLOCK_TOKENS)]
    chunks = chunks[:args.window_blocks]
    # Positive control: with `--control-full` every block counts as golden, so no block
    # is blanked and this function must reproduce the Full arm row for row. If it does
    # not, the harness -- not the ceiling -- is what the 18/158 measured.
    wanted = (set(range(len(chunks))) if args.control_full
              else set(int(block) for block in blocks))
    if not wanted or not wanted <= set(range(len(chunks))):
        return dict(available=False, correct=False, qa_f1=0.0, prediction="",
                    blocks=[], slots=0, blanked=0)
    stream, blanked = [], 0
    for index, chunk in enumerate(chunks):
        if index in wanted:
            stream.append(chunk)
        else:
            stream.append(blank_chunk(tokenizer, len(chunk)))
            blanked += 1
    with torch.no_grad():
        reset(manager, initial)
        es.write_blocks(manager, merge, stream)
        answer = gr.readout(model, tokenizer, prepared["query_ids"], prepared["answers"],
                            args.f1_threshold, lambda: None)
    return dict(available=True, correct=bool(answer["qa_f1"] >= args.f1_threshold),
                qa_f1=float(answer["qa_f1"]), prediction=answer["prediction"],
                blocks=sorted(wanted), blanked_blocks=blanked,
                slots=int(manager.used))


def evidence_survival(meta, blocks):
    """How many slots of the evidence blocks survived, split by core / graft.

    `meta["source_block"]` carries the true block id, so this asks the direct question:
    is any slot OF THAT BLOCK in the window? (Deriving the block from `source_position`
    would be wrong -- that index addresses the carrier pool, not the 19-block window.)
    """
    source = meta["source_block"]
    mask = torch.zeros_like(source, dtype=torch.bool)
    for block_id in set(int(block) for block in blocks):
        mask |= source == block_id
    if not bool(mask.any()):
        return dict(total=0, core=0, graft=0)
    return dict(total=int(mask.sum()),
                core=int((mask & (meta["slot_type"] == 0)).sum()),
                graft=int((mask & (meta["slot_type"] == 1)).sum()))


def evaluate(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    facts = gr.load_supporting_facts(args.supporting_facts_glob)
    nv.seed_everything(args.seed)
    checkpoint_training = torch.load(args.checkpoint, map_location="cpu").get("training")
    model, tokenizer, manager, merge, initial, cfg, saved = load_runtime(
        args.model_path, checkpoint=args.checkpoint, train=False, architecture=None,
        budget_blocks=gr.BUDGET_BLOCKS, strict_code_hash=args.strict_code_hash,
        training=checkpoint_training)
    merge.operator_mode = "native"
    module, geometry, config, _ = ev.load_v11(args.v11_evict, manager.hidden)
    module = module.to(model.memory.device).float()
    scorer = ReaderScorer(model, tuple(int(x) for x in geometry["utility_layers"]))
    joint = ev.joint_for(geometry, scorer, {}, manager.hidden)
    rows, prepared, excluded = gr.prepare_rows(args, tokenizer, facts)
    if args.max_samples:
        prepared = prepared[:args.max_samples]
    original = {}
    if args.v13_report:
        payload = json.load(open(args.v13_report))
        for sample in payload["samples"]:
            arms = sample.get("arms", {}).get(ev.V11_EVICT)
            full = sample.get("arms", {}).get(ev.FULL)
            if arms:
                original[str(sample["id"])] = dict(
                    accuracy_correct=bool(arms["qa_f1"] >= args.f1_threshold),
                    qa_f1=float(arms["qa_f1"]), prediction=arms["prediction"],
                    full_correct=(bool(full["qa_f1"] >= args.f1_threshold)
                                  if full else None))

    report = dict(schema=SCHEMA, status="running", counts=dict(
        rows=len(rows), eligible=len(prepared), excluded=len(excluded)),
        identity=dict(model_path=args.model_path, checkpoint=args.checkpoint,
                      checkpoint_sha256=sha(args.checkpoint),
                      v11_evict=args.v11_evict, v11_evict_sha256=sha(args.v11_evict),
                      v13_report=args.v13_report,
                      geometry_core=geometry["core_per_block"],
                      geometry_graft=geometry["graft_per_block"],
                      entrypoint_sha256=sha(__file__)), samples=[])
    output = Path(args.out)
    if output.exists():
        raise ValueError(f"{output} exists; delete it first")
    atomic_json(output, report)
    started = time.monotonic()
    for position, prepared_row in enumerate(prepared):
        sample_started = time.monotonic()
        nv.seed_everything(args.seed)
        with torch.no_grad():
            reset(manager, initial)
            es.write_blocks(manager, merge,
                            [prepared_row["write_ids"][index:index + BLOCK_TOKENS]
                             for index in range(0, len(prepared_row["write_ids"]),
                                                BLOCK_TOKENS)][:args.window_blocks])
            source, ages, picked = gr.select_carriers(manager, args.window_blocks,
                                                      args.target_blocks)
            window, meta, telemetry = joint.build(source, picked, manager.block)
        dropped = [int(block) for block in telemetry["dropped_blocks"]]
        blocks = [int(block) for block in (prepared_row.get("evidence", {}).get("blocks")
                                           or [])]
        inside = [block for block in blocks if block < args.window_blocks]
        outside = [block for block in blocks if block >= args.window_blocks]
        survival = evidence_survival(meta, inside)
        golden = golden_answer(model, tokenizer, manager, merge, initial, prepared_row,
                               inside, args)
        reference = original.get(str(prepared_row["id"]))
        record = dict(
            id=prepared_row["id"], evidence_blocks=blocks,
            evidence_blocks_outside_window=outside,
            dropped_blocks=dropped,
            evidence_in_dropped=bool(inside) and set(inside) <= set(dropped),
            evidence_slots=survival,
            evidence_lost=bool(inside) and survival["total"] == 0,
            golden=golden,
            v13=reference,
            seconds=time.monotonic() - sample_started)
        report["samples"].append(record)
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output, report)
        print(json.dumps(dict(sample=position + 1, total=len(prepared),
                              id=prepared_row["id"],
                              golden=golden["correct"],
                              slots=survival["total"],
                              in_dropped=record["evidence_in_dropped"],
                              seconds=round(record["seconds"], 1))), flush=True)

    graded = [sample for sample in report["samples"] if sample["v13"]]
    wrong = [sample for sample in graded if not sample["v13"]["accuracy_correct"]]
    right = [sample for sample in graded if sample["v13"]["accuracy_correct"]]
    golden_fail = [sample for sample in report["samples"] if not sample["golden"]["correct"]]
    selection_loss = [sample for sample in wrong if sample["evidence_lost"]]
    both = [sample for sample in wrong
            if sample["evidence_lost"] and not sample["golden"]["correct"]]
    report["summary"] = dict(
        n=len(report["samples"]),
        v13_rows=len(graded), v13_correct=len(right), v13_wrong=len(wrong),
        v13_accuracy=(round(100.0 * len(right) / max(len(graded), 1), 2)),
        golden_ceiling_correct=len(report["samples"]) - len(golden_fail),
        golden_ceiling_accuracy=round(
            100.0 * (len(report["samples"]) - len(golden_fail))
            / max(len(report["samples"]), 1), 2),
        reader_fails_on_golden=len(golden_fail),
        wrong_and_evidence_in_dropped=len([s for s in wrong
                                           if s["evidence_in_dropped"]]),
        wrong_and_evidence_lost=len(selection_loss),
        wrong_and_reader_fails_on_golden=len([s for s in wrong
                                             if not s["golden"]["correct"]]),
        wrong_lost_and_reader_fails=len(both),
        wrong_explained=len([s for s in wrong
                             if s["evidence_lost"] or not s["golden"]["correct"]]),
        golden_over_full=len([s for s in report["samples"]
                              if s["golden"]["correct"]
                              and s.get("v13") and s["v13"].get("full_correct") is False]),
        full_correct_rows=len([s for s in report["samples"]
                               if s.get("v13") and s["v13"].get("full_correct")]),
        blanked_blocks=[s["golden"].get("blanked_blocks") for s in report["samples"][:1]],
        definition=("`golden_ceiling_accuracy` is the reader's ceiling with the golden "
                    "evidence alone in a normal 19-block pool (non-golden blocks filled "
                    "with the same number of neutral tokens). `full_correct_rows` is the "
                    "same reader given all 19 blocks raw, read from the finished report. "
                    "`wrong_and_evidence_lost`: V13 answers wrong AND no supporting-fact "
                    "slot survived into the 2,304-slot window -- lost by selection. "
                    "`wrong_and_reader_fails_on_golden`: V13 answers wrong AND the frozen "
                    "reader also fails when given only the golden blocks -- lost by the "
                    "reader itself. `golden_ceiling_accuracy` is that reader ceiling over "
                    "all rows."),
        per_row=[dict(id=sample["id"], v13=bool(sample["v13"]["accuracy_correct"]),
                      golden=bool(sample["golden"]["correct"]),
                      evidence_lost=bool(sample["evidence_lost"]),
                      in_dropped=bool(sample["evidence_in_dropped"]))
                 for sample in report["samples"]])
    report["status"] = "complete"
    report["elapsed_seconds"] = time.monotonic() - started
    atomic_json(output, report)
    print(json.dumps({k: v for k, v in report["summary"].items()
                      if k not in ("definition", "per_row")}, indent=1), flush=True)
    return report


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--task", default="hotpotqa", choices=("hotpotqa",))
    p.add_argument("--supporting-facts-glob", dest="supporting_facts_glob", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--anchor", default=None)
    p.add_argument("--v11-evict", dest="v11_evict", default=None,
                   help="the V13 checkpoint whose window is being diagnosed")
    p.add_argument("--v13-report", dest="v13_report", default=None,
                   help="a finished report holding the V13 arm; its per-row correctness "
                        "decides which rows are 'wrong'")
    p.add_argument("--out", default=None)
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--window-blocks", dest="window_blocks", type=int,
                   default=gr.WINDOW_BLOCKS)
    p.add_argument("--target-blocks", dest="target_blocks", type=int,
                   default=gr.TARGET_BLOCKS)
    p.add_argument("--f1-threshold", dest="f1_threshold", type=float, default=0.5)
    p.add_argument("--control-full", dest="control_full", action="store_true",
                   help="positive control: treat all 19 blocks as golden (nothing is "
                        "blanked) and check the harness reproduces the Full arm")
    p.add_argument("--sample-ids", dest="sample_ids", default=None,
                   help="comma separated row ids (gr.prepare_rows honours it)")
    p.add_argument("--max-samples", dest="max_samples", type=int, default=None)
    p.add_argument("--strict-code-hash", dest="strict_code_hash", default="true",
                   choices=("true", "false"))
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.data_dir or not args.model_path or not args.v11_evict:
        raise SystemExit("--data-dir, --model-path and --v11-evict are required")
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
