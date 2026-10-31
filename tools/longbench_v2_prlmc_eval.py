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
import collections
import hashlib
import json
import os
import subprocess
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


CACHE_SCHEMA = "native_write_state/1"


def write_state_key(args, ids, geometry, lora_sha):
    """Identity of a completed write phase.

    The 246 writes are deterministic given the input tokens, the frozen native
    write protocol, the merge arm, its weights and the geometry -- compressions
    happen *inside* the write stream, so the arm is part of the identity and a
    cache entry can never be shared across arms.
    """
    payload = dict(schema=CACHE_SCHEMA, protocol="tail2048_native_write_v1",
                   writes_sha256=hashlib.sha256(
                       np.asarray(ids, dtype=np.int64).tobytes()).hexdigest(),
                   arm=args.arm, budget_blocks=args.budget_blocks,
                   geometry=list(geometry) if geometry else None,
                   trigger_blocks=args.trigger_blocks, fusion_mode=args.fusion_mode,
                   checkpoint=sha(args.checkpoint) if args.checkpoint else None,
                   reader_lora=lora_sha, model_path=args.model_path,
                   entrypoint=sha(__file__),
                   runtime=sha(ROOT / "tools" / "natural_merge_runtime.py"))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    return value


def snapshot_state(model, manager):
    """Everything the write phase mutates, on the CPU.

    Block ages are recorded for diagnostics but not restored: they only feed the
    write path, which a cache hit replaces wholesale.
    """
    return dict(memory=_to_cpu(model.memory),
                active_mask=_to_cpu(model.memory_active_mask),
                initialized=_to_cpu(model.initialized),
                used=int(manager.used), n_consolidated=int(manager.n_consolidated),
                goal_direction=_to_cpu(manager.goal_direction),
                active_mask_used=_to_cpu(manager._active_mask_used),
                block_age_summary=manager.block_age_summary())


def restore_state(model, manager, state):
    with torch.no_grad():
        model.memory.copy_(state["memory"].to(model.memory.device))
        model.memory_active_mask.copy_(state["active_mask"].to(model.memory_active_mask.device))
        model.initialized.copy_(state["initialized"].to(model.initialized.device))
    manager.used = state["used"]
    manager.n_consolidated = state["n_consolidated"]
    manager.goal_direction = _to_cpu(state["goal_direction"]) if state["goal_direction"] is not None else None
    if state["goal_direction"] is not None and torch.is_tensor(manager.goal_direction):
        manager.goal_direction = manager.goal_direction.to(model.memory.device)
    manager._active_mask_used = state["active_mask_used"]
    if torch.is_tensor(manager._active_mask_used):
        manager._active_mask_used = manager._active_mask_used.to(model.memory.device)


def remember(cache, key, value, slots):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max(1, slots):
        cache.popitem(last=False)


def summarize(samples):
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
    return summary


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
    if args.group_by_context:
        # Same-context questions back to back: the write phase is identical for
        # them, so the cache turns the second..nth read into a pure read.
        rows = sorted(rows, key=lambda r: (digest(r["context"]), str(r.get("_id"))))
    if args.num_shards > 1:
        chunk = -(-len(rows) // args.num_shards)
        rows = rows[args.shard_index * chunk:(args.shard_index + 1) * chunk]
        if not rows:
            raise ValueError("empty shard; lower --num-shards or drop --max-samples")

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
            write_cache_dir=args.write_cache_dir,
            group_by_context=args.group_by_context,
            shard=([args.shard_index, args.num_shards] if args.num_shards > 1 else None),
            caveat=BUDGET_CAVEAT),
        identity=dict(checkpoint=args.checkpoint,
                      checkpoint_sha256=sha(args.checkpoint) if args.checkpoint else None,
                      model_path=args.model_path, data_json_sha256=sha(args.data_json),
                      entrypoint_sha256=sha(__file__)),
        samples=[], training_performed=False, optimizer_created=False,
        full_test_split=args.max_samples is None, max_samples=args.max_samples)
    atomic_json(output, report)

    started = time.monotonic()
    cache_dir = Path(args.write_cache_dir) if args.write_cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    memory_cache = collections.OrderedDict()
    lora_sha = sha(args.reader_lora) if args.reader_lora else None
    write_cache_hits = write_cache_misses = 0
    for index, row in enumerate(rows):
        sample_started = time.monotonic()
        seed_everything(args.seed)
        pe.reset(manager, initial)
        with torch.no_grad():
            manager._tile_free_region()
            manager._sync_active_mask()
        prompt, ids, write_ids, query_ids, original = build_input(
            tokenizer, row, args.total_tokens, args.tail_tokens, args.doc_truncation)
        key, key_payload = write_state_key(args, ids, geometry, lora_sha)
        path = (cache_dir / f"{key}.pt") if cache_dir else None
        cached, cache_source = None, None
        if path is not None and path.is_file():
            cached = torch.load(path, map_location="cpu")
            cache_source = "disk"
        elif key in memory_cache:
            cached = memory_cache[key]
            cache_source = "memory"
        if cached is not None:
            restore_state(model, manager, cached["state"])
            events, peak = cached["events"], cached["peak_slots"]
            write_count = cached["write_count"]
            write_cache_hits += 1
        else:
            if lora_state is not None:
                set_lora_enabled(model, False)   # writes stay native
            writes = [write_ids[i:i + 512] for i in range(0, len(write_ids), 512)]
            with torch.inference_mode():
                merge_module = pe.build_consolidate_merge(merge, args.arm, manager)
                stream = pe.MergeStream(manager, merge_module, args.arm, **stream_arguments)
                events = pe.run_sample(stream, writes)
            peak, write_count = stream.peak, len(writes)
            write_cache_misses += 1
            entry = dict(key=key_payload, state=snapshot_state(model, manager),
                         events=events, peak_slots=peak, write_count=write_count)
            remember(memory_cache, key, entry, args.write_cache_slots)
            if path is not None:
                torch.save(entry, path)
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
            write_count=write_count, compression_count=len(events),
            peak_active_slots=peak, final_active_slots=active,
            write_cache_hit=int(cache_source is not None), write_cache_source=cache_source,
            write_cache_key=key,
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

    summary = summarize(report["samples"])
    summary["write_cache"] = dict(hits=write_cache_hits, misses=write_cache_misses,
                                  slots=args.write_cache_slots, dir=args.write_cache_dir)
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
    p.add_argument("--write-cache-dir", dest="write_cache_dir", default=None,
                   help="persist completed write states here (3.35 GB per entry); the key "
                        "carries protocol, tokens, arm, geometry, checkpoint, LoRA and code sha")
    p.add_argument("--write-cache-slots", dest="write_cache_slots", type=int, default=1,
                   help="in-process write-state cache entries kept on the CPU")
    p.add_argument("--group-by-context", dest="group_by_context", action="store_true",
                   help="order samples by context so same-context questions reuse the "
                        "write phase; per-sample results are independent of order")
    p.add_argument("--num-shards", dest="num_shards", type=int, default=1,
                   help="split the sample list across N processes, one per GPU")
    p.add_argument("--shard-index", dest="shard_index", type=int, default=None,
                   help="internal: index of the shard this process evaluates")
    return p


def run_shards(args):
    """One process per GPU, then merge. Sharding is sample-level, so every sample
    is computed by exactly one process with the single-GPU code path."""
    argv = sys.argv[1:]
    if "--out" not in argv:
        raise ValueError("sharded runs need an explicit --out")
    cut = argv.index("--out")
    base = argv[:cut] + argv[cut + 2:]
    children = []
    for shard in range(args.num_shards):
        out = Path(f"{args.out}.shard{shard}")
        if out.exists():
            raise ValueError(f"shard output exists; refuse overwrite: {out}")
        command = [sys.executable, str(Path(__file__).resolve())] + base + [
            "--num-shards", str(args.num_shards), "--shard-index", str(shard),
            "--cuda-device", "0", "--out", str(out)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard))
        children.append((subprocess.Popen(command, env=env), out))
    failed = [(proc.wait(), out) for proc, out in children]
    if any(code != 0 for code, _ in failed):
        raise SystemExit(f"shard failure: {failed}")
    samples, elapsed = [], 0.0
    for _, out in failed:
        shard_report = json.loads(out.read_text())
        samples.extend(shard_report["samples"])
        elapsed = max(elapsed, shard_report.get("elapsed_seconds", 0.0))
    samples.sort(key=lambda s: s["sample_index"])
    report = dict(schema="longbench_v2_prlmc_results/1", status="complete",
                  protocol=dict(arm=args.arm, budget_blocks=args.budget_blocks,
                                total_input_tokens=args.total_tokens,
                                geometry=list(args.window_blocks and (args.window_blocks, args.target_blocks))
                                if args.window_blocks and args.target_blocks else None,
                                shards=args.num_shards,
                                shard_reports=[str(out) for _, out in failed]),
                  identity=dict(checkpoint=args.checkpoint,
                                data_json_sha256=sha(args.data_json)),
                  samples=samples, elapsed_seconds=elapsed,
                  full_test_split=args.max_samples is None, max_samples=args.max_samples,
                  training_performed=False, optimizer_created=False)
    report["summary"] = summarize(samples)
    atomic_json(Path(args.out), report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=1))


def main():
    args = build_parser().parse_args()
    args.strict_code_hash = args.strict_code_hash == "true"
    if args.model_path is None:
        args.model_path = pe.MODEL_PATH
    if args.num_shards > 1 and args.shard_index is None:
        run_shards(args)
        return
    if args.shard_index is None:
        args.shard_index = 0
    evaluate(args)


if __name__ == "__main__":
    main()
