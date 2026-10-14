#!/usr/bin/env python3
"""Frozen E1/E2/E3 original natural-data generation, fixed carriers and budget."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_natural_merge import validate_plan
from prepare_public_retention import sha, tokenizer_manifest
from natural_merge_runtime import (MergeStream, install, native_capacity, native_write,
    evaluate_questions, load_runtime, reset, source_manifest, state_digest, costs,
    tensor_digest, model_identity, E6D_SHA)

E1_ARMS = ("Full", "Selection", "SharedMerge", "CarrierOnlyFusion")
RETENTION_ARMS = ("Full", "Selection", "Recent", "SharedMerge", "NativeBudget", "NativeDefault")


def validate_checkpoint_scope(plan, saved, diagnostic=False):
    contexts = {c["context_sha256"] for e in plan["episodes"] for c in e["contexts"]}
    articles = {c["article"] for e in plan["episodes"] for c in e["contexts"]}
    distractors = {w.get("context_sha256") for e in plan["episodes"] for w in e["writes"]}
    distractor_articles = {w.get("article") for e in plan["episodes"] for w in e["writes"]}
    distractor_articles.update(a for e in plan["episodes"] for w in e["writes"] for a in w.get("source_articles",[]))
    if diagnostic:
        if plan["partition"] != "development" or plan["experiment"] != "E1":
            raise ValueError("legacy E6d is allowed only for E1 development diagnosis")
        return
    if saved.get("schema") != "natural_merge_checkpoint/1" or saved.get("evaluation_eligible") is not True:
        raise ValueError("eligible natural shared checkpoint required")
    if contexts.union(distractors).intersection(saved["train_context_hashes"]):
        raise ValueError("training/evaluation context leakage")
    if articles.union(distractor_articles).intersection(saved["train_articles"]):
        raise ValueError("training/evaluation article leakage")
    if saved["tokenizer"] != plan["tokenizer"] or saved["protocol"] != plan["protocol"]:
        raise ValueError("checkpoint data/protocol mismatch")


def validate_freeze(freeze, plan, checkpoint, model_path, suite):
    from prepare_public_retention import digest
    identity = dict(freeze); expected = identity.pop("content_sha256", None)
    if expected != digest(identity) or freeze.get("schema") != "natural_merge_freeze/1":
        raise ValueError("freeze integrity failure")
    if (freeze["checkpoint_sha256"] != sha(checkpoint) or freeze["sources"] != source_manifest()
            or freeze["model"] != model_identity(model_path) or freeze["suite_sha256"] != sha(suite)):
        raise ValueError("frozen checkpoint/model/code/suite changed")
    if plan["content_sha256"] not in freeze["allowed_plan_content_sha256"]:
        raise ValueError("evaluation plan not in pre-frozen suite")


def e1_episode(manager, merge, tokenizer, episode, progress):
    import torch
    from methods.prlmc.consolidate import _select_block_indices
    qas = [q for c in episode["contexts"] for q in c["qas"]]
    stream = MergeStream(manager, merge, "Full")
    for i, write in enumerate(episode["writes"]):
        if stream.write(write, i) is None:
            raise ValueError("19 natural blocks do not fit Full reference")
    before = manager._memory.detach().cpu().clone()
    used = manager.used
    source = manager._memory[:, manager.total-used:].contiguous()
    source_hash = tensor_digest(source)
    weights = manager._window_importance(manager.total-used, used)
    indices = _select_block_indices(source, 9*manager.block, manager.block, weights, None, manager.select_weights)
    selected = indices.cpu().tolist()
    row = dict(id=episode["id"], source_memory_sha256=source_hash, carrier_indices=selected,
               original_contexts=[c["context_sha256"] for c in episode["contexts"]], arms={})
    for arm in E1_ARMS:
        manager._memory.copy_(before.to(manager._memory.device))
        manager.used = used; manager._active_mask_used = None; manager._tile_free_region()
        if arm != "Full":
            # This source always belongs to the same unmodified premerge state.
            if arm == "Selection":
                candidate = source.reshape(manager.num_layers, 19, manager.block, manager.hidden).index_select(1, indices).flatten(1,2)
                fusion = None
            else:
                allow = torch.ones(19, dtype=torch.bool, device=source.device)
                if arm == "CarrierOnlyFusion":
                    allow.zero_(); allow[indices] = True
                candidate = merge(source, target=9*manager.block, carrier_indices=indices, source_allow_mask=allow)
                fusion = dict(merge.last_fusion)
            install(manager, candidate, used)
        else:
            fusion = None
        row["arms"][arm] = dict(status="complete", active_slots=manager.used,
            source_memory_sha256=source_hash, carrier_indices=selected if arm!="Full" else list(range(19)),
            fusion=fusion, predictions=evaluate_questions(manager.model, tokenizer, qas))
        progress(row, arm)
    return row


def retention_episode(manager, merge, tokenizer, initial, episode, progress):
    qas = [q for c in episode["contexts"] for q in c["qas"]]
    row = dict(id=episode["id"], original_contexts=[c["context_sha256"] for c in episode["contexts"]], arms={})
    schedule = None
    for arm in ("SharedMerge", "Selection", "Recent", "NativeBudget", "NativeDefault", "Full"):
        reset(manager, initial)
        record = dict(status="running", trace=[], compressions=[],
            comparison="native system pipeline; capacity/layout/rotary offsets differ" if arm.startswith("Native")
                       else "same managed writer/reader/generation")
        row["arms"][arm] = record
        def hook(phase, index, selected):
            record["compressions"].append(dict(phase=phase, before_write=index,
                active_slots=manager.used, carrier_indices=selected.cpu().tolist(),
                predictions=evaluate_questions(manager.model, tokenizer, qas),
                measurement="isolated compression; no intervening write"))
        stream = None if arm.startswith("Native") else MergeStream(manager, merge, arm, hook, carrier_schedule=schedule if arm=="Selection" else None)
        context = native_capacity(manager.model, 20) if arm=="NativeBudget" else nullcontext()
        with context:
            record["physical_slots"]=manager.model.memory.shape[1]
            record["physical_pool_bytes"]=manager.model.memory.numel()*manager.model.memory.element_size()
            record["effective_capacity_slots"]=(manager.model.memory.shape[1] if arm.startswith("Native") or arm=="Full" else 20*manager.block)
            for i, write in enumerate(episode["writes"]):
                entry = native_write(manager.model, write, i) if stream is None else stream.write(write, i)
                if entry is None:
                    record.update(status="capacity_exceeded", failed_before_write=i,
                        reason="Full physical capacity exceeded; no truncation or final Full reference")
                    break
                entry["predictions"] = evaluate_questions(manager.model, tokenizer, qas)
                record["trace"].append(entry)
                progress(row, arm)
            else:
                record["status"] = "complete"
        record["peak_active_slots"] = max(t["active_slots"] for t in record["trace"])
        record["merge_events"] = stream.events if stream else []
        if arm == "SharedMerge":
            schedule = [event["selected_blocks"] for event in stream.events]
    return row


def validate_e2_gate(result, plan, current_checkpoint, freeze):
    from verify_natural_merge import verify
    verify(result, plan)
    if (plan["experiment"]!="E2" or plan["partition"]!="formal"
            or result["checkpoint_sha256"]!=sha(current_checkpoint) or result["diagnostic_only"]
            or result["plan_content_sha256"] not in freeze["allowed_plan_content_sha256"]
            or result["sharding"]["count"]!=1 or len(result["episodes"])!=len(plan["episodes"])):
        raise ValueError("complete frozen formal E2 evidence required before E3")
    summary=result["summary"]
    if not summary["conclusion_allowed"] or summary["shared_vs_selection"]["ci95"][0]<=0:
        raise ValueError("E2 progression gate failed: context-cluster CI does not support positive Shared-Selection")


def run(args):
    plan = json.loads(Path(args.plan).read_text()); validate_plan(plan)
    if plan["partition"] == "train":
        raise ValueError("training plan cannot be used for evaluation")
    if tokenizer_manifest(args.model_path) != plan["tokenizer"]:
        raise ValueError("evaluation tokenizer mismatch")
    if Path(args.out).exists():
        raise ValueError("evaluation output exists")
    if args.legacy_diagnostic and sha(args.checkpoint) != E6D_SHA:
        raise ValueError("legacy diagnosis requires the archived E6d hash")
    if plan["partition"] in ("formal", "transfer") and (not args.freeze or not args.suite):
        raise ValueError("formal/transfer evaluation requires frozen suite and checkpoint")
    if not 0 <= args.shard_index < args.shard_count <= len(plan["episodes"]):
        raise ValueError("invalid episode shard")
    if args.freeze:
        validate_freeze(json.loads(Path(args.freeze).read_text()), plan, args.checkpoint, args.model_path, args.suite)
    if plan["experiment"]=="E3":
        if not args.e2_result or not args.e2_plan or not args.freeze:
            raise ValueError("E3 requires a successful complete formal E2 result; no transfer-based tuning")
        e2=json.loads(Path(args.e2_result).read_text());e2plan=json.loads(Path(args.e2_plan).read_text())
        if e2["plan_sha256"]!=sha(args.e2_plan) or e2["freeze_sha256"]!=sha(args.freeze):
            raise ValueError("E2 result identity differs from frozen method")
        validate_e2_gate(e2,e2plan,args.checkpoint,json.loads(Path(args.freeze).read_text()))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    os.environ["S0B_GPU_SET"] = "1"
    import torch
    from native_qa_compare import atomic_json
    from verify_natural_merge import summarize, verify
    model, tokenizer, manager, merge, initial, cfg, saved = load_runtime(args.model_path, args.checkpoint)
    validate_checkpoint_scope(plan, saved, args.legacy_diagnostic)
    if not args.legacy_diagnostic and saved["model"] != model_identity(args.model_path):
        raise ValueError("trained reader checkpoint differs from evaluation reader")
    frozen = state_digest(merge)
    if saved.get("final_state_sha256", frozen) != frozen and not args.legacy_diagnostic:
        raise ValueError("checkpoint state hash mismatch")
    episode_indices = [i for i in range(len(plan["episodes"])) if i % args.shard_count == args.shard_index]
    report = dict(schema="natural_merge_results/1", status="running", experiment=plan["experiment"],
        architecture=saved.get("architecture", "block_v1"), variant=saved.get("variant"),
        partition=plan["partition"], diagnostic_only=args.legacy_diagnostic, protocol=plan["protocol"],
        plan_sha256=sha(args.plan), plan_content_sha256=plan["content_sha256"],
        checkpoint_sha256=sha(args.checkpoint), sources=source_manifest(), model=model_identity(args.model_path),
        suite_sha256=sha(args.suite) if args.suite else None,
        freeze_sha256=sha(args.freeze) if args.freeze else None,
        resources=costs(manager, merge), manager_config=cfg,
        runtime=dict(torch=torch.__version__,transformers=__import__("transformers").__version__,
            cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),physical_device=args.cuda_device),
        training_performed=False,
        optimizer_created=False, shared_state_sha256_before=frozen, episodes=[],
        sharding=dict(index=args.shard_index, count=args.shard_count, episode_indices=episode_indices))
    atomic_json(args.out, report)
    begun = time.monotonic()
    try:
        for i in episode_indices:
            reset(manager, initial)
            def progress(row, arm):
                report["progress"] = dict(episode=i, arm=arm,
                    write=len(row["arms"][arm].get("trace", [])))
                report["current_episode"] = row
                atomic_json(args.out, report)
                print(json.dumps(report["progress"]), flush=True)
            with torch.no_grad():
                row = (e1_episode(manager, merge, tokenizer, plan["episodes"][i], progress) if plan["experiment"]=="E1"
                       else retention_episode(manager, merge, tokenizer, initial, plan["episodes"][i], progress))
            row["episode_index"] = i
            report["episodes"].append(row)
            report.pop("current_episode", None)
            if state_digest(merge) != frozen or any(p.requires_grad for p in merge.parameters()) or any(p.requires_grad for p in model.parameters()):
                raise ValueError("frozen evaluation parameter invariant failed")
            atomic_json(args.out, report)
        report.update(status="complete", shared_state_sha256_after=state_digest(merge),
            summary=summarize(report), elapsed_seconds=time.monotonic()-begun,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        certificate = verify(report,plan,tokenizer)
        report["audit"] = dict(status=certificate["status"], verified=certificate["verified"])
    except Exception as exc:
        report.update(status="failed", failure=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        atomic_json(args.out, report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-path", default="/dev/shm/s1d_reader_fp16")
    p.add_argument("--cuda-device", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--legacy-diagnostic", action="store_true")
    p.add_argument("--freeze")
    p.add_argument("--suite")
    p.add_argument("--e2-result")
    p.add_argument("--e2-plan")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    run(p.parse_args())

if __name__ == "__main__":
    main()
