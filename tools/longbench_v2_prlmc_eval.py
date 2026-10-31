#!/usr/bin/env python3
"""PRLM-C (SharedMerge / Selection / Consolidate / Full) on LongBench v2.

Adapter layer only. Everything in the method path is imported from the v1
entrypoint so there is exactly one implementation of the method:

    tools/longbench_prlmc_eval.py   load_runtime, MergeStream, reset, stream
                                    geometry handshake, run_sample, generate
    tools/longbench_v2_eval.py      official data.json loading, prompt template,
                                    128k middle truncation, answer extraction

What this file adds is the LongBench v2 wiring: length groups, the official
four-choice scoring and a report in the same schema shape as the other
entrypoints.

Budget caveat (printed on every run): the v9 checkpoint is trained and
validated at the 20-block working budget (19 -> 9 geometry), while the main
table's MemoryLLM-8B row measures the native 50-block pool (12,800 slots).
Numbers from this entrypoint are NOT directly comparable to that row.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import longbench_prlmc_eval as pe            # noqa: E402
import longbench_v2_eval as lbv2             # noqa: E402
from longbench_memoryllm_eval import atomic_json, digest, seed_everything, sha  # noqa: E402

MAX_NEW = 128   # native main-table row generates 128 tokens
TOTAL_CHOICES = (8192, 16384, 32768, 128000)
VALIDATED_TOTALS = (8192, 16384)
NATIVE_BUDGET = 50          # native 50-block / 12,800-slot capacity
BUDGET_CAVEAT = (
    "budget caveat: this run uses the PRLM-C working budget "
    "(--budget-blocks), not the native 50-block pool of the main-table "
    "MemoryLLM-8B row; do not compare the two directly")


def build_input(tokenizer, row, total_tokens, tail_tokens, mode):
    prompt = lbv2.build_prompt(row, row["context"])
    ids = tokenizer(prompt, add_special_tokens=False, truncation=False).input_ids
    original = len(ids)
    if len(ids) > total_tokens:
        ids = lbv2.truncate_ids(ids, total_tokens, mode)
    tail = min(tail_tokens, max(1, len(ids) // 2))
    return prompt, ids, ids[:-tail], ids[-tail:], original


def evaluate(args):
    if args.total_tokens not in TOTAL_CHOICES:
        raise ValueError(f"--total-tokens must be one of {TOTAL_CHOICES}")
    if args.total_tokens not in VALIDATED_TOTALS:
        print(f"warning: --total-tokens {args.total_tokens} is outside the "
              f"validated {VALIDATED_TOTALS} range; smoke only", file=sys.stderr)
    print(BUDGET_CAVEAT, file=sys.stderr)
    if args.seed != 0:
        raise ValueError("the fixed evaluation seed is 0")
    if args.arm == "SharedMerge" and not (args.checkpoint and Path(args.checkpoint).is_file()):
        raise ValueError("--arm SharedMerge needs an existing --checkpoint")
    if args.arm != "SharedMerge" and args.checkpoint is not None:
        raise ValueError(f"--arm {args.arm} does not load a checkpoint")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("output exists; refuse overwrite")

    rows = lbv2.load_rows(args.data_json, args.length)
    if args.max_samples is not None:
        rows = rows[:args.max_samples]

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.cuda_device))
    seed_everything(args.seed)
    pe.install_generation_compatibility()

    checkpoint_training = (torch.load(args.checkpoint, map_location="cpu").get("training")
                           if args.checkpoint else None)
    model, tokenizer, manager, merge, initial, cfg, saved = pe.load_runtime(
        args.model_path, checkpoint=args.checkpoint, train=False,
        architecture=(None if args.checkpoint else "grouped_gated_sparse_native_v9"),
        budget_blocks=args.budget_blocks, strict_code_hash=args.strict_code_hash,
        fusion_mode=args.fusion_mode, training=checkpoint_training)
    tokenizer.pad_token = tokenizer.eos_token
    lora_state = None
    if args.reader_lora:
        from reader_lora_train import load_lora_state, set_lora_enabled
        lora_state = torch.load(args.reader_lora, map_location="cpu")
        load_lora_state(model, lora_state)
        set_lora_enabled(model, True)
    if int(model.num_blocks) != 50 or int(model.num_tokens) != 256:
        raise ValueError("expected the native 50-block MemoryLLM reader")
    if args.budget_blocks == NATIVE_BUDGET:
        # Native 50-block capacity: the pre-registered budget_geometry table only
        # covers 8/10/12/20, so the 19 -> 9 geometry is supplied explicitly and the
        # trigger is the moment a write would exceed the 50-block pool.
        window = args.window_blocks if args.window_blocks is not None else 19
        target = args.target_blocks if args.target_blocks is not None else 9
        geometry = (window, target)
        stream_arguments = dict(budget_blocks=args.budget_blocks,
                                window_blocks=window, target_blocks=target)
        if args.trigger_blocks is not None:
            stream_arguments["trigger_blocks"] = args.trigger_blocks
        # local geometry check: the shared helper only whitelists the 8/10/12/20
        # training budgets, while the native 50-block pool is the full physical pool
        physical = manager.total // manager.block
        if not (0 < target < window <= args.budget_blocks <= physical):
            raise ValueError(f"invalid 50-block geometry window={window} "
                             f"target={target} physical={physical}")
    else:
        stream_arguments, geometry = pe.stream_kwargs(args)
        if geometry is not None:
            pe.validate_geometry(args.budget_blocks, geometry[0], geometry[1],
                                 manager.total // manager.block)
    capacity_blocks = (pe.PHYSICAL_BLOCKS if geometry is None
                       else max(args.budget_blocks, geometry[0]))

    report = dict(
        schema="longbench_v2_prlmc_results/1", status="running",
        protocol=dict(
            benchmark="LongBench v2", data_json=args.data_json, length_group=args.length,
            prompt="official gpt4 template (tools/longbench_v2_eval.py build_prompt)",
            total_input_tokens=args.total_tokens, tail_tokens=args.tail_tokens,
            doc_truncation=args.doc_truncation, merge_arm=args.arm,
            budget_blocks=args.budget_blocks,
            geometry=list(geometry) if geometry else None,
            fusion_mode=args.fusion_mode,
            do_sample=False, num_beams=1, seed=args.seed, max_new_tokens=MAX_NEW,
            scoring="official four-choice extraction, accuracy",
            caveat=BUDGET_CAVEAT),
        identity=dict(checkpoint=args.checkpoint,
                      checkpoint_sha256=sha(args.checkpoint) if args.checkpoint else None,
                      model_path=args.model_path, data_json_sha256=sha(args.data_json),
                      entrypoint_sha256=sha(__file__)),
        samples=[], training_performed=False, optimizer_created=False,
        full_test_split=args.max_samples is None, max_samples=args.max_samples)
    atomic_json(output, report)

    started = time.monotonic()
    for index, row in enumerate(rows):
        sample_started = time.monotonic()
        seed_everything(args.seed)
        pe.reset(manager, initial)
        with torch.no_grad():
            manager._tile_free_region()
            manager._sync_active_mask()
        if lora_state is not None:
            set_lora_enabled(model, False)
        prompt, ids, write_ids, query_ids, original = build_input(
            tokenizer, row, args.total_tokens, args.tail_tokens, args.doc_truncation)
        writes = [write_ids[i:i + 512] for i in range(0, len(write_ids), 512)]
        merge_module = pe.build_consolidate_merge(merge, args.arm, manager)
        stream = pe.MergeStream(manager, merge_module, args.arm, **stream_arguments)
        events = pe.run_sample(stream, writes)
        if lora_state is not None:
            set_lora_enabled(model, True)
        prediction, output_ids = pe.generate(model, tokenizer, query_ids, MAX_NEW)
        letter = lbv2.extract_answer(prediction)
        active = int(model.memory_active_mask.sum())
        if active != manager.used or active > capacity_blocks * model.num_tokens:
            raise ValueError("active budget violation after writes")
        report["samples"].append(dict(
            task="longbench_v2", sample_index=index, id=row.get("_id"),
            length=row.get("length"), difficulty=row.get("difficulty"),
            domain=row.get("domain"), arm=args.arm,
            source_row_sha256=digest(row), context_sha256=digest(row["context"]),
            original_prompt_tokens=original, input_tokens=len(ids),
            prefix_tokens=len(write_ids), query_tokens=len(query_ids),
            write_count=len(writes), compression_count=len(events),
            peak_active_slots=stream.peak, final_active_slots=active,
            seconds=time.monotonic() - sample_started,
            prediction=prediction, output_tokens=output_ids,
            extracted_letter=letter, answer=row["answer"],
            correct=int(letter == row["answer"])))
        report["progress"] = dict(sample=index + 1, total=len(rows))
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output, report)
        print(json.dumps(dict(**report["progress"], letter=letter, answer=row["answer"],
                              correct=report["samples"][-1]["correct"],
                              compressions=len(events),
                              seconds=report["samples"][-1]["seconds"])), flush=True)

    samples = report["samples"]
    summary = dict(longbench_v2=dict(
        n=len(samples),
        accuracy=100.0 * float(np.mean([s["correct"] for s in samples])),
        compression_samples=int(sum(1 for s in samples if s["compression_count"] > 0)),
        mean_active_slots=float(np.mean([s["final_active_slots"] for s in samples]))))
    for group in ("short", "medium", "long"):
        picked = [s for s in samples if s["length"] == group]
        if picked:
            summary[f"longbench_v2_{group}"] = dict(
                n=len(picked), accuracy=100.0 * float(np.mean([s["correct"] for s in picked])))
    report["summary"] = summary
    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["status"] = "complete"
    atomic_json(output, report)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-json", required=True, help="LongBench v2 data.json")
    p.add_argument("--model-path", default=None,
                   help="fp16 MemoryLLM reader directory (default: the v1 entrypoint constant)")
    p.add_argument("--out", required=True)
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--length", default="medium", choices=("all", "short", "medium", "long"))
    p.add_argument("--total-tokens", type=int, default=128000, choices=TOTAL_CHOICES)
    p.add_argument("--tail-tokens", type=int, default=2048, choices=(512, 1024, 2048))
    p.add_argument("--doc-truncation", default="middle", choices=("middle", "tail"))
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--reader-lora", default=None,
                   help="frozen Reader LoRA payload from reader_lora_train.py")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--arm", default="SharedMerge",
                   choices=("SharedMerge", "Selection", "Consolidate", "Full"))
    p.add_argument("--budget-blocks", dest="budget_blocks", type=int, default=20,
                   choices=(8, 10, 12, 20, 50))
    p.add_argument("--window-blocks", dest="window_blocks", type=int, default=None)
    p.add_argument("--target-blocks", dest="target_blocks", type=int, default=None)
    p.add_argument("--trigger-blocks", dest="trigger_blocks", type=int, default=None)
    p.add_argument("--fusion-mode", dest="fusion_mode", default="sealed",
                   choices=("sealed", "block_ortho"))
    p.add_argument("--strict-code-hash", dest="strict_code_hash", default="true",
                   choices=("true", "false"))
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    args.strict_code_hash = args.strict_code_hash == "true"
    if args.model_path is None:
        args.model_path = pe.MODEL_PATH
    evaluate(args)


if __name__ == "__main__":
    main()
