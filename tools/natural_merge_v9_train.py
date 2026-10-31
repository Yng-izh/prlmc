#!/usr/bin/env python3
"""Fresh DnativeV9 128-update training on the v9 plan.

Cloned from the v7 trainer with the user-registered v8 differences only:

1. the plan is `natural_merge_v8_training_plan/1` (sealed TRAIN_V7 untouched,
   separate checkpoint generation);
2. recurrent student/teacher streams run trigger_blocks=20, so the streaming
   compaction fires at used >= 20 blocks and the recurrent events (before
   writes 20 and 24) land exactly where the evaluation stream compacts;
3. the objective imports the v8 per-token knowledge protection
   (`natural_merge_v8_objective`); Selection-correct answer positions carry a
   degraded-token hinge + Selection KL, Selection-wrong positions carry none;
4. HotpotQA train episodes ride the same recurrent curriculum as the sealed
   corpus episodes (questions are supervision only; they are never written
   into memory).
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_natural_merge_v9 import validate_v9_plan  # noqa: E402
from prepare_natural_merge_v4 import event_qas  # noqa: E402
from prepare_public_retention import sha, tokenizer_manifest  # noqa: E402
from natural_merge_runtime import (MergeStream, install, load_runtime, reset,
    source_manifest, state_digest, costs, model_identity)  # noqa: E402
from natural_merge_v4_objective import NativeOrigins, make_optimizer  # noqa: E402
import natural_merge_v9_objective as objective_v9  # noqa: E402
from reader_lora_train import (AdapterHandle, attach_lora, load_lora_state,  # noqa: E402
                               lora_named_state, set_lora_enabled)


def run(args):
    plan = json.loads(Path(args.plan).read_text()); validate_v9_plan(plan)
    training = plan["training"]
    if args.variant not in training["variants"]:
        raise ValueError("unknown variant")
    if tokenizer_manifest(args.model_path) != plan["tokenizer"]:
        raise ValueError("tokenizer mismatch")
    if Path(args.out).exists() or Path(args.checkpoint_out).exists():
        raise ValueError("training output exists")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device); os.environ["S0B_GPU_SET"] = "1"
    # Fragmentation on a shared 40-GiB A100 is the difference between finishing
    # at 25.8 GiB and OOMing at the first recurrent update.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    # Card preflight. The free-memory gate used to be a hardcoded 28500/30500 MiB
    # inside the launcher shell script, which silently assumed a dedicated
    # 32-GiB V100; on this host the card is a 40-GiB A100 shared with other
    # tenants, so ask the driver what is actually free and refuse before the
    # reader is loaded instead of dying hours later.
    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable: no device for --cuda-device %d" % args.cuda_device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_mib, total_mib = free_bytes // 2**20, total_bytes // 2**20
    card = dict(name=torch.cuda.get_device_name(0),
                compute_capability=list(torch.cuda.get_device_capability(0)),
                total_mib=int(total_mib), free_mib_at_start=int(free_mib),
                min_free_mib=int(args.min_free_mib),
                alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
                reader_weight_dtype="float16")
    if free_mib < args.min_free_mib:
        raise ValueError("insufficient free VRAM on %s: %d MiB free < %d MiB required "
                         "(measured peak is ~25800 MiB; free the card first)"
                         % (card["name"], free_mib, args.min_free_mib))
    from native_qa_compare import atomic_json
    from methods.prlmc.probes import _checkpoint_func
    variant = training["variants"][args.variant]
    geometry = training["geometry"]
    model, tokenizer, manager, merge, initial, cfg, _ = load_runtime(
        args.model_path, train=True, architecture=variant["architecture"],
        budget_blocks=geometry["budget_blocks"],
        window_blocks=geometry["window_blocks"],
        target_blocks=geometry["target_blocks"])
    if float(getattr(model.config, "attention_dropout", 0.)) != 0.:
        raise ValueError("reader checkpointing requires deterministic zero dropout")
    # Recompute-versus-residency trade. Checkpointing keeps the 20-QA backward
    # loop inside ~26 GiB on a 32-GiB V100, but it pays a second reader forward
    # per backward; on a card with 10+ GiB of headroom that recompute is pure
    # lost time, and the gradient is bit-identical either way because the
    # recomputation is deterministic.
    card["reader_checkpointing"] = not args.no_reader_checkpointing
    if not args.no_reader_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.model.gradient_checkpointing = True
        model.model._gradient_checkpointing_func = _checkpoint_func()
    model.train()  # Reader body stays frozen; activates gradient checkpointing.
    # Joint reader adapter. Only the deltas train, and they are switched on only
    # around the supervised QA read: every memory write (teacher and student) runs
    # the native reader, so the stored representation is bit-identical to the one
    # the evaluation stream produces.
    adapter = None
    adapter_before = None
    if args.joint_reader_lora:
        attached = attach_lora(model, last_layers=args.lora_layers,
                               r=args.lora_rank, alpha=args.lora_alpha)
        if args.reader_lora_init:
            load_lora_state(model, torch.load(args.reader_lora_init, map_location="cpu"))
        adapter = AdapterHandle(model)
        set_lora_enabled(model, False)
        adapter_before = adapter.digest()
        card["reader_adapter"] = dict(modules=len(attached), rank=args.lora_rank,
                                      alpha=args.lora_alpha, layers=args.lora_layers,
                                      lr=args.lora_lr, targets=list(attached[:2]),
                                      init=args.reader_lora_init)
    from prepare_natural_merge_v4 import check_answer_encoding
    check_answer_encoding(plan, tokenizer)
    optimizer = make_optimizer(merge, plan=training)
    if adapter is not None:
        optimizer.add_param_group(dict(params=adapter.parameters(), lr=args.lora_lr,
                                       weight_decay=0., name="reader_adapter",
                                       parameter_names=adapter.names))
    scaler = torch.amp.GradScaler("cuda", init_scale=128.)
    episodes = plan["episodes"]
    events_by_stream = {}
    for event in plan["qa_events"]:
        events_by_stream.setdefault(event["stream"], []).append(event)
    base = plan["base_plan"]; begun = time.monotonic()
    train_event = objective_v9.train_event
    full_targets = objective_v9.full_targets
    provenance = {(e["stream"], e["round"], e["before_write"]): e["source_indices"]
                  for e in plan["qa_events"] if e["stage"] == "recurrent"}
    evidence_budget = geometry["budget_blocks"]
    recurrent_targets = None  # teacher references built per episode below

    def build_recurrent_references(episode, round_index):
        """Local raw-evidence teacher targets at each recurrent event.

        The old teacher replayed every write through an uncompressed stream, which
        cannot hold 125 writes in a 50-block pool. The teacher now writes the raw
        contexts that the supervised window actually summarises (the plan's
        provenance record), truncated to the pool capacity, so every reference is a
        realisable no-compression state over the same evidence. The Selection
        per-token reference still comes from `full_targets`.
        """
        result = {}
        for i in training["recurrent_event_before_writes"]:
            qas = event_qas(plan, round_index, episode, "recurrent", i)
            keep = provenance[(episode["id"], round_index, i)][-evidence_budget:]
            reset(manager, initial)
            stream = MergeStream(manager, merge, "Full",
                                 budget_blocks=geometry["budget_blocks"],
                                 window_blocks=geometry["window_blocks"],
                                 target_blocks=geometry["target_blocks"],
                                 trigger_blocks=training["trigger_blocks"])
            with torch.no_grad():
                for position, write_index in enumerate(keep):
                    if stream.write(episode["writes"][write_index], position) is None:
                        raise ValueError("local evidence teacher exceeds pool capacity")
                result[i] = full_targets(manager, qas)
        if sorted(result) != training["recurrent_event_before_writes"]:
            raise ValueError("incomplete teacher events")
        return result

    report = dict(schema="natural_merge_v9_training/1", status="running", variant=args.variant,
                  architecture=variant["architecture"], training=training, protocol=base["protocol"],
                  objective=training["objective"], geometry=dict(geometry),
                  trigger_blocks=training["trigger_blocks"],
                  plan_sha256=sha(args.plan), train_content_sha256=plan["content_sha256"],
                  dataset=base["dataset"], split_audit=base["split_audit"], sources=source_manifest(),
                  model=model_identity(args.model_path), manager_config=cfg, resources=costs(manager, merge),
                  runtime=dict(torch=torch.__version__, transformers=__import__("transformers").__version__,
                               cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                               physical_device=args.cuda_device, card=card),
                  reader_weights_frozen=True, shared_state_sha256_before=state_digest(merge), updates=[],
                  smoke=args.smoke_updates is not None, evaluation_eligible=False,
                  initialization="fresh identity residual, effective gate .02, fixed seed0; no warm start",
                  optimizer_groups=[{k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups])
    atomic_json(args.out, report)
    maximum = args.smoke_updates or training["planned_updates"]
    try:
        for round_index in range(4):
            for episode in episodes:
                if round_index < 2:
                    length = training["single_window_contexts"]
                    for start in training["single_window_starts"][round_index]:
                        reset(manager, initial)
                        # Accumulation stream: the sealed 19-context window must
                        # be fully resident before the supervised event; the
                        # supervised merge itself uses the 13->9 evaluation geometry.
                        stream = MergeStream(manager, merge, "Full")
                        origins = NativeOrigins()
                        with torch.no_grad():
                            contexts = episode["contexts"][start:start + length]
                            writes = episode["writes"][start:start + length]
                            for i, write in enumerate(writes):
                                if stream.write(write, i) is None:
                                    raise ValueError("single-merge training exceeds pool")
                                origins.append(contexts[i]["qas"][0]["source_index"])
                        qas = event_qas(plan, round_index, episode, "single_merge", start)
                        targets = full_targets(manager, qas)
                        event = train_event(manager, merge, optimizer, qas, targets,
                                            args.variant, origins, scaler, plan=training,
                                            adapter=adapter)
                        event.update(stage="single_merge", window_start=start, before_write=len(writes))
                        record_event(report, event, round_index, episode, args.out, begun)
                        del targets
                        if len(report["updates"]) >= maximum: break
                else:
                    references = build_recurrent_references(episode, round_index)
                    reset(manager, initial)
                    # The student stream runs trigger_blocks=20: compaction at
                    # used >= 20 blocks, so the recurrent events before writes
                    # 20 and 24 land exactly where the evaluation stream compacts.
                    stream = MergeStream(manager, merge, "Full",
                                         budget_blocks=geometry["budget_blocks"],
                                         window_blocks=geometry["window_blocks"],
                                         target_blocks=geometry["target_blocks"],
                                         trigger_blocks=training["trigger_blocks"])
                    origins = NativeOrigins()
                    for i, write in enumerate(episode["writes"]):
                        if i in training["recurrent_event_before_writes"]:
                            if manager.used < training["trigger_blocks"] * manager.block:
                                raise ValueError("recurrent event is not at the 20-block trigger")
                            qas = event_qas(plan, round_index, episode, "recurrent", i)
                            event = train_event(manager, merge, optimizer, qas, references[i],
                                                args.variant, origins, scaler, plan=training,
                                                adapter=adapter)
                            event.update(stage="recurrent", before_write=i)
                            record_event(report, event, round_index, episode, args.out, begun)
                            if len(report["updates"]) >= maximum: break
                        with torch.no_grad():
                            if stream.write(write, i) is None:
                                raise ValueError("student training exceeds pool")
                            origins.append(episode["contexts"][i]["qas"][0]["source_index"])
                    del references
                if len(report["updates"]) >= maximum: break
            if len(report["updates"]) >= maximum: break
        if len(report["updates"]) != maximum:
            raise ValueError("training schedule incomplete")
        adapter_ids = {id(p) for p in adapter.parameters()} if adapter else set()
        if any((p.requires_grad or p.grad is not None) and id(p) not in adapter_ids
               for p in model.parameters()):
            raise ValueError("reader body (non-adapter parameters) must stay frozen")
        final = state_digest(merge)
        if final == report["shared_state_sha256_before"]:
            raise ValueError("training performed no parameter update")
        seen = sorted({q["context_sha256"] for u in report["updates"] for q in u["qas"]})
        expected = sorted({c["context_sha256"] for e in plan["episodes"] for c in e["contexts"]})
        eligible = args.smoke_updates is None and len(report["updates"]) == training["planned_updates"]
        counts = training.get("curriculum_updates") or dict(single_merge=64, recurrent=64)
        expected_stages = (["single_merge"] * counts["single_merge"]
                           + ["recurrent"] * counts["recurrent"])
        if eligible and [u["stage"] for u in report["updates"]] != expected_stages:
            raise ValueError("single/recurrent training curriculum changed")
        covered_qas = sorted({q["id"] for u in report["updates"] for q in u["qas"]})
        expected_qas = sorted({q["id"] for e in plan["episodes"] for c in e["contexts"] for q in c["qas"]})
        if eligible and counts["recurrent"] == 64:      # sealed full-coverage contract
            if seen != expected:
                raise ValueError("some train contexts never entered compression supervision")
            if covered_qas != expected_qas:
                raise ValueError("not every original training QA was supervised")
        checkpoint = dict(schema="natural_merge_checkpoint/1", architecture=variant["architecture"],
                          variant=args.variant, state_dict={k: v.detach().cpu() for k, v in merge.state_dict().items()},
                          final_state_sha256=final, effective_gamma_max=.05, rank=32, evaluation_eligible=eligible,
                          module_configuration=variant, global_rms_budget=training["global_rms_budget"],
                          training=training, objective=training["objective"], protocol=base["protocol"],
                          geometry=dict(geometry), recovery_gate=dict(training["recovery_gate"]),
                          train_plan_sha256=sha(args.plan), train_content_sha256=plan["content_sha256"],
                          training_sources=source_manifest(), source_manifest=source_manifest(),
                          tokenizer=plan["tokenizer"], model=report["model"],
                          train_context_hashes=expected,
                          train_articles=sorted({c["article"] for e in plan["episodes"] for c in e["contexts"]}))
        Path(args.checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.checkpoint_out)
        compressions = sum(1 for u in report["updates"]
                           if u.get("window_blocks", 0) > u.get("target_blocks", 0))
        if eligible and compressions == 0:
            raise ValueError("no real 19->9 compression happened during training")
        adapter_metrics = None
        if adapter is not None:
            after = adapter.digest()
            if after == adapter_before:
                raise ValueError("reader adapter performed no parameter update")
            Path(args.lora_out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(lora_named_state(model), args.lora_out)
            adapter_metrics = dict(modules=len(adapter.names),
                                   state_digest_before=adapter_before,
                                   state_digest_after=after,
                                   events_with_grad=sum(1 for u in report["updates"]
                                                        if u.get("reader_adapter")),
                                   payload=args.lora_out)
        report.update(compressions=compressions, reader_adapter=adapter_metrics)
        report.update(status="complete", evaluation_eligible=eligible, checkpoint_sha256=sha(args.checkpoint_out),
                      shared_state_sha256_after=final, supervised_contexts=seen, supervised_qa_ids=covered_qas,
                      elapsed_seconds=time.monotonic() - begun,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as exc:
        report.update(status="failed", failure=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        atomic_json(args.out, report)


def record_event(report, event, round_index, episode, out, begun):
    from native_qa_compare import atomic_json
    event.update(update=len(report["updates"]) + 1, round=round_index, stream=episode["id"],
                 elapsed_seconds=time.monotonic() - begun)
    report["updates"].append(event); atomic_json(out, report)
    print(json.dumps({k: event[k] for k in ("update", "stage", "stream", "gold_ce",
                                            "weighted_gold_ce", "teacher_kl", "token_protection")}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--variant", required=True, choices=["DnativeV9"])
    p.add_argument("--model-path", default="/dev/shm/s1d_reader_fp16")
    p.add_argument("--cuda-device", type=int, required=True)
    p.add_argument("--min-free-mib", type=int, default=29000,
                   help="refuse to start unless this much VRAM is free (peak is ~25800 MiB)")
    p.add_argument("--no-reader-checkpointing", action="store_true",
                   help="keep reader activations resident instead of recomputing them "
                        "(faster, needs more VRAM; identical gradients)")
    p.add_argument("--out", required=True); p.add_argument("--checkpoint-out", required=True)
    p.add_argument("--smoke-updates", type=int)
    p.add_argument("--joint-reader-lora", action="store_true",
                   help="train the reader adapter and the merge module on one loss")
    p.add_argument("--reader-lora-init", default=None,
                   help="optional warm start for the adapter (reader_lora_train payload)")
    p.add_argument("--lora-out", default="runs/natural_merge_v9/reader_lora_joint.pt")
    p.add_argument("--lora-layers", type=int, default=8)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-lr", type=float, default=1e-4)
    args = p.parse_args()
    if args.smoke_updates is not None:
        planned = json.loads(Path(args.plan).read_text())["training"]["planned_updates"]
        if not 1 <= args.smoke_updates <= planned:
            p.error("smoke updates exceed the planned update budget")
    run(args)


if __name__ == "__main__":
    main()
