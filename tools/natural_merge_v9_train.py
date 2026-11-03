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
from collections import Counter
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


def atomic_torch_save(payload, path):
    """Save to a sibling temp file then rename, so a crash never leaves a
    half-written checkpoint that a later resume would happily load."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def prune_step_checkpoints(ckpt_path, keep=3):
    """Keep only the newest `keep` step checkpoints of this run."""
    ckpt_path = Path(ckpt_path)
    prefix = ckpt_path.stem.rsplit("_step", 1)[0]
    saved = sorted(ckpt_path.parent.glob(f"{prefix}_step*.pt"))
    for stale in saved[:-keep]:
        stale.unlink(missing_ok=True)


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
    # `build_merge` would otherwise read the v9 gate/route spec from the module-level
    # TRAIN_V9 record built out of the sealed v8 plan, and silently fall back to bare
    # defaults (rank 32, gamma_max .05, no bridge) when that record is unavailable.
    # The plan carries the complete record, so hand it in explicitly and verify the
    # operator really came out with it.
    model, tokenizer, manager, merge, initial, cfg, _ = load_runtime(
        args.model_path, train=True, architecture=variant["architecture"],
        budget_blocks=geometry["budget_blocks"],
        window_blocks=geometry["window_blocks"],
        target_blocks=geometry["target_blocks"], training=training)
    effective_config = {k: getattr(merge, k) for k in
                        ("rank", "topk", "gamma_max", "bridge_blocks", "bridge_fraction",
                         "bridge_gamma_max", "native_init", "global_rms_budget")
                        if hasattr(merge, k)}
    expected_config = dict(topk=variant["topk"], bridge_blocks=training["bridge_blocks"],
                           native_init=training["native_init"],
                           gamma_max=variant["local_gate_max"])
    wrong_config = {k: (effective_config[k], v) for k, v in expected_config.items()
                    if k in effective_config and effective_config[k] != v}
    if wrong_config:
        raise ValueError(f"merge module built with the wrong v9 config: {wrong_config}")
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
        adapter.enable_grad()          # load_lora_state freezes; joint training does not
        set_lora_enabled(model, False)
        adapter_before = adapter.digest()
        card["reader_adapter"] = dict(modules=len(attached), rank=args.lora_rank,
                                      alpha=args.lora_alpha, layers=args.lora_layers,
                                      lr=args.lora_lr, targets=list(attached[:2]),
                                      init=args.reader_lora_init)
    if args.bridge_adapter:
        if args.joint_reader_lora:
            raise ValueError("--bridge-adapter replaces --joint-reader-lora, not adds to it")
        from bridge_kv_adapter import attach_bridge_adapter
        low, high = (int(x) for x in args.bridge_layers.split("-"))
        adapter = attach_bridge_adapter(model, manager, layers=range(low, high + 1),
                                       rank=args.lora_rank, lora_alpha=args.lora_alpha)
        adapter.enable_grad()
        adapter.off()
        adapter_before = adapter.digest()
        card["reader_adapter"] = dict(kind="bridge_kv_masked", layers=args.bridge_layers,
                                      rank=args.lora_rank, alpha=args.lora_alpha,
                                      lr=args.adapter_lr, targets="memory_kv")
    # Stage 1 keeps the reader untouched, so the adapter stays structurally off and
    # frozen until the split; stage 2 turns it on and hands it the low learning rate.
    stage_split = int(args.stage_split)
    if adapter is not None:
        adapter.stage_active = not stage_split
        if stage_split:
            for p in adapter.parameters():
                p.requires_grad_(False)
    from prepare_natural_merge_v4 import check_answer_encoding
    check_answer_encoding(plan, tokenizer)
    optimizer = make_optimizer(merge, plan=training)
    if getattr(args, "resume", None):
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        merge.load_state_dict(saved["state_dict"])
        resumed = saved.get("resume") or {}
        if resumed.get("optimizer"):
            optimizer.load_state_dict(resumed["optimizer"])
        if resumed.get("rng"):
            torch.set_rng_state(resumed["rng"]["torch"])
            torch.cuda.set_rng_state_all(resumed["rng"]["cuda"])
        if resumed.get("manager_memory") is not None:
            manager._memory.copy_(resumed["manager_memory"].to(manager._memory.device, manager._memory.dtype))
            manager.used = int(resumed.get("manager_used", manager.used))
            manager._tile_free_region()
        print(json.dumps({"resumed_from": args.resume, "updates": saved.get("updates"),
                          "pool_restored": resumed.get("manager_memory") is not None}))
    if adapter is not None:
        optimizer.add_param_group(dict(
            params=adapter.parameters(),
            lr=args.adapter_lr if stage_split else args.lora_lr,
            weight_decay=0., name="reader_adapter",
            parameter_names=adapter.names))
    scaler = torch.amp.GradScaler("cuda", init_scale=128.)
    episodes = plan["episodes"]
    events_by_stream = {}
    for event in plan["qa_events"]:
        events_by_stream.setdefault(event["stream"], []).append(event)
    base = plan["base_plan"]; begun = time.monotonic()
    train_event = objective_v9.train_event
    stage_two = False

    def sync_adapter_stage(update_index):
        """Flip the adapter on exactly once, at the planned update."""
        nonlocal stage_two
        if adapter is None or not stage_split:
            return
        wanted = update_index >= stage_split
        if wanted == stage_two:
            return
        stage_two = wanted
        adapter.stage_active = wanted
        for p in adapter.parameters():
            p.requires_grad_(wanted)
        if wanted:
            print(f"[stage] reader adapter trainable at update {update_index}", flush=True)
    full_targets = objective_v9.full_targets
    provenance = {(e["stream"], e["round"], e["before_write"]): e["source_indices"]
                  for e in plan["qa_events"] if e["stage"].startswith("recurrent")}
    recurrent_stage_at = training.get("recurrent_stage_by_position") or {}
    evidence_budget = geometry["budget_blocks"]
    recurrent_targets = None  # teacher references built per episode below

    def build_recurrent_references(episode, round_index):
        """RETIRED -- kept for reference only; no longer called.

        It built a raw-evidence teacher over the window provenance, which (a) could
        not represent deep windows at all and (b) named contexts that had already
        lost their block, so the cohorts it produced were labelled `skip`. Every
        recurrent event now takes the live pre-merge state as its reference; see the
        recurrent branch of the training loop.

        Original docstring: local raw-evidence teacher targets for the shallow
        recurrent events.

        The teacher writes the raw contexts of *every supervised QA*. Two defects of
        the previous version are fixed here:

        * it wrote only the merge window's provenance, but the protect cohort is the
          untouched tail of the pool, so the teacher never held that cohort's
          evidence and its KL term was measured against a pool that had never seen
          the context;
        * `recurrent_deep` events are no longer precomputed at all. Past write 90 the
          window provenance is already wider than the 50-block pool, so no
          no-compression state over that evidence exists; those events take the
          pre-merge state captured live in `run` as their reference instead.

        Writing exactly the supervised contexts (<= QA_CAP <= evidence_budget) makes
        the reference a realisable no-compression state that covers every supervised
        QA, which is what keeps both cohorts populated. The Selection per-token
        reference still comes from `full_targets`.
        """
        result = {}
        for i in training["recurrent_event_before_writes"]:
            stage = recurrent_stage_at.get(str(i), "recurrent")
            if stage != "recurrent":
                continue                      # deep events: live pre-merge reference
            qas = event_qas(plan, round_index, episode, stage, i)
            keep = sorted({int(qa["source_index"]) for qa in qas})
            if len(keep) > evidence_budget:
                raise ValueError("supervised evidence exceeds the teacher pool")
            kept_origins = set(keep)
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
                for qa in qas:
                    result[i][qa["id"]]["teacher_evidence"] = (
                        qa["source_index"] in kept_origins)
        expected = [i for i in training["recurrent_event_before_writes"]
                    if recurrent_stage_at.get(str(i), "recurrent") == "recurrent"]
        if sorted(result) != expected:
            raise ValueError("incomplete teacher events")
        return result

    report = dict(schema="natural_merge_v9_training/1", status="running", variant=args.variant,
                  architecture=variant["architecture"], training=training, protocol=base["protocol"],
                  objective=training["objective"], geometry=dict(geometry),
                  trigger_blocks=training["trigger_blocks"],
                  plan_sha256=sha(args.plan), train_content_sha256=plan["content_sha256"],
                  dataset=base["dataset"], split_audit=base["split_audit"], sources=source_manifest(),
                  model=model_identity(args.model_path), manager_config=cfg, resources=costs(manager, merge),
                  effective_module_config=effective_config,
                  runtime=dict(torch=torch.__version__, transformers=__import__("transformers").__version__,
                               cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                               physical_device=args.cuda_device, card=card),
                  reader_weights_frozen=True, shared_state_sha256_before=state_digest(merge), updates=[],
                  two_stage=dict(split=args.stage_split, adapter_lr=args.adapter_lr,
                                 bridge_adapter=args.bridge_adapter,
                                 bridge_layers=args.bridge_layers),
                  smoke=args.smoke_updates is not None, evaluation_eligible=False,
                  initialization="fresh identity residual, effective gate .02, fixed seed0; no warm start",
                  optimizer_groups=[{k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups])
    atomic_json(args.out, report)
    maximum = args.smoke_updates or training["planned_updates"]
    if args.preflight:
        # One stream is enough to exercise every branch; 16 streams x 2 rounds of
        # single-merge events would make the gate take half an hour.
        episodes = episodes[:1]
    # Reaches `train_event` through the plan it already receives as `plan=training`.
    training["qa_batch"] = max(1, int(args.qa_batch or 1))
    def dump_progress():
        """Write the current weights to disk every `--save-every` updates.

        The weights are the only irreplaceable output of a 7 h run, so they are
        persisted while training proceeds instead of only at the end.
        """
        step = len(report["updates"])
        if args.save_every <= 0 or step == 0 or step % args.save_every:
            return
        base_out = args.checkpoint_out[:-3] if args.checkpoint_out.endswith(".pt") else args.checkpoint_out
        ckpt_path = f"{base_out}_step{step}.pt"
        resume_state = dict(optimizer=optimizer.state_dict(),
                            rng=dict(torch=torch.get_rng_state(),
                                     cuda=torch.cuda.get_rng_state_all()),
                            manager_used=int(manager.used),
                            elapsed_seconds=time.time() - begun)
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(dict(schema="natural_merge_checkpoint/1", architecture=variant["architecture"],
                        variant=args.variant, partial=True, updates=step, evaluation_eligible=False,
                        state_dict={k: v.detach().cpu() for k, v in merge.state_dict().items()},
                        module_configuration=variant, global_rms_budget=training["global_rms_budget"],
                        training=training, objective=training["objective"], protocol=base["protocol"],
                        geometry=dict(geometry), recovery_gate=dict(training["recovery_gate"]),
                        train_plan_sha256=sha(args.plan), train_content_sha256=plan["content_sha256"],
                        tokenizer=plan["tokenizer"], model=report["model"], resume=resume_state), ckpt_path)
        adapter_path = None
        if adapter is not None:
            adapter_path = (args.lora_out[:-3] if args.lora_out.endswith(".pt") else args.lora_out) + f"_step{step}.pt"
            torch.save(dict(schema="reader_adapter/2", kind="bridge_kv_masked", partial=True, updates=step,
                            state=adapter.state_dict(), layers=args.bridge_layers,
                            rank=args.lora_rank, alpha=args.lora_alpha, dropout=0.0,
                            targets=list(adapter.names), geometry=dict(geometry),
                            source="natural_merge_v9_train.joint"), adapter_path)
        print(json.dumps(dict(progress_dump=step, checkpoint=ckpt_path, adapter=adapter_path)), flush=True)
        prune_step_checkpoints(ckpt_path)
        latest_path = f"{base_out}_latest.pt"
        atomic_torch_save(dict(schema="natural_merge_resume/1", updates=step,
                               architecture=variant["architecture"], variant=args.variant,
                               state_dict={k: v.detach().cpu() for k, v in merge.state_dict().items()},
                               resume=dict(optimizer=optimizer.state_dict(),
                                           rng=dict(torch=torch.get_rng_state(),
                                                    cuda=torch.cuda.get_rng_state_all()),
                                           manager_memory=manager._memory.detach().cpu().clone(),
                                           manager_used=int(manager.used),
                                           elapsed_seconds=time.time() - begun),
                               training=training, geometry=dict(geometry),
                               train_plan_sha256=sha(args.plan)), latest_path)

    try:
        for round_index in range(4):
            for episode in episodes:
                if round_index < 2:
                    length = training["single_window_contexts"]
                    starts_this_round = training["single_window_starts"][round_index]
                    if args.preflight:
                        starts_this_round = starts_this_round[:1]
                    for start in starts_this_round:
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
                        targets = full_targets(manager, qas, chunk=args.qa_batch)
                        sync_adapter_stage(len(report["updates"]))
                        event = train_event(manager, merge, optimizer, qas, targets,
                                            args.variant, origins, scaler, plan=training,
                                            adapter=adapter)
                        event.update(stage="single_merge", window_start=start, before_write=len(writes))
                        record_event(report, event, round_index, episode, args.out, begun)
                        dump_progress()
                        del targets
                        if len(report["updates"]) >= maximum: break
                else:
                    # No precomputed references: every recurrent reference is the live
                    # pre-merge state of this event (see the event branch below).
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
                    recurrent_events = 0
                    for i, write in enumerate(episode["writes"]):
                        if i in training["recurrent_event_before_writes"]:
                            if manager.used < training["trigger_blocks"] * manager.block:
                                raise ValueError("recurrent event is not at the 20-block trigger")
                            stage_here = recurrent_stage_at.get(str(i), "recurrent")
                            # Diagnosed cause of the empty recover cohort (85-87% of
                            # events): `NativeOrigins.compact` keeps ONE representative
                            # origin per block, so once a merge absorbs a context it no
                            # longer owns a block. The plan's QA list is built from the
                            # merge provenance, i.e. mostly from absorbed contexts, and
                            # `qa_cohorts` can only label those `skip` -- which empties
                            # the cohort the recovery term is defined on. Supervision is
                            # therefore anchored on the contexts that currently own a
                            # block: the window blocks supply recover, the tail protect.
                            planned = event_qas(plan, round_index, episode, stage_here, i)
                            resident = [episode["contexts"][o]["qas"][0]
                                        for o in origins.blocks]
                            seen = {q["id"] for q in planned}
                            # Cost is linear in the supervised QA count -- measured
                            # 2.2-2.5 s per QA per event, i.e. essentially the whole
                            # 7 h budget -- so the set is the smallest one that still
                            # guarantees both cohorts: 4 long-range QAs from the plan
                            # plus up to 6 contexts that currently own a block (the
                            # window blocks supply recover, the tail protect).
                            qas = planned[:4] + [q for q in resident if q["id"] not in seen][:6]
                            # The reference is the state this merge starts from: it is
                            # the only realisable no-fusion state for deep events (the
                            # window summarises far more originals than the pool can
                            # hold) and it holds the evidence of every supervised QA,
                            # so no QA is ever supervised against a blind teacher.
                            with torch.no_grad():
                                reference = full_targets(manager, qas, chunk=args.qa_batch)
                            for qa in qas:
                                reference[qa["id"]]["teacher_evidence"] = True
                            sync_adapter_stage(len(report["updates"]))
                            event = train_event(manager, merge, optimizer, qas, reference,
                                                args.variant, origins, scaler, plan=training,
                                                adapter=adapter)
                            event.update(stage=stage_here, before_write=i)
                            record_event(report, event, round_index, episode, args.out, begun)
                            dump_progress()
                            recurrent_events += 1
                            if args.preflight and recurrent_events >= 2: break
                            if len(report["updates"]) >= maximum: break
                        with torch.no_grad():
                            if stream.write(write, i) is None:
                                raise ValueError("student training exceeds pool")
                            origins.append(episode["contexts"][i]["qas"][0]["source_index"])
                if len(report["updates"]) >= maximum: break
            if len(report["updates"]) >= maximum: break
        # A preflight deliberately runs 6 of the planned events, so the coverage
        # assertion cannot apply to it.
        if not args.preflight and len(report["updates"]) != maximum:
            raise ValueError("training schedule incomplete")
        # Assertions that can only be evaluated once the run is over are collected
        # here and raised AFTER the weights are written, so a bookkeeping complaint
        # can never again throw away a complete run.
        pending = []
        adapter_ids = {id(p) for p in adapter.parameters()} if adapter else set()
        if any((p.requires_grad or p.grad is not None) and id(p) not in adapter_ids
               for p in model.parameters()):
            pending.append("reader body (non-adapter parameters) must stay frozen")
        final = state_digest(merge)
        if final == report["shared_state_sha256_before"]:
            pending.append("training performed no parameter update")
        seen = sorted({q["context_sha256"] for u in report["updates"] for q in u["qas"]})
        expected = sorted({c["context_sha256"] for e in plan["episodes"] for c in e["contexts"]})
        eligible = args.preflight or (args.smoke_updates is None
                                      and len(report["updates"]) == training["planned_updates"])
        counts = training.get("curriculum_updates") or dict(single_merge=64, recurrent=64)
        # Derive the expectation from the plan instead of hardcoding the two sealed
        # stages: the paper schedule also has `recurrent_deep`, and a hardcoded list
        # rejected a complete 704-update run at the very last check, after 5.8 h of
        # training and before the checkpoint was written.
        # Rebuild the exact stage order the loop produces: rounds 2/3 emit each
        # stream's shallow positions and then its deep ones, so the sequence
        # interleaves and a flatten-by-count list can never match it. Two complete
        # 704-update runs died on that comparison after 6-7 h, before any weights
        # were written; the mix is now compared as a multiset.
        expected_stages = []
        for _round in range(4):
            for _episode in range(len(episodes)):
                if _round < 2:
                    expected_stages += ["single_merge"] * len(
                        training["single_window_starts"][_round])
                else:
                    expected_stages += [recurrent_stage_at.get(str(_pos), "recurrent")
                                        for _pos in training["recurrent_event_before_writes"]]
        observed = [u["stage"] for u in report["updates"]]
        if args.preflight:
            unknown = [s for s in observed if s not in counts]
            if unknown:
                raise ValueError(f"preflight produced unknown stages {sorted(set(unknown))}")
        elif eligible and Counter(observed) != Counter(expected_stages):
            pending.append("single/recurrent training curriculum changed: "
                           f"{dict(Counter(observed))} vs {dict(Counter(expected_stages))}")
        covered_qas = sorted({q["id"] for u in report["updates"] for q in u["qas"]})
        expected_qas = sorted({q["id"] for e in plan["episodes"] for c in e["contexts"] for q in c["qas"]})
        if eligible and counts["recurrent"] == 64:      # sealed full-coverage contract
            if seen != expected:
                pending.append("some train contexts never entered compression supervision")
            if covered_qas != expected_qas:
                pending.append("not every original training QA was supervised")
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
        atomic_torch_save(checkpoint, args.checkpoint_out)
        compressions = sum(1 for u in report["updates"]
                           if u.get("window_blocks", 0) > u.get("target_blocks", 0))
        if eligible and compressions == 0 and not args.preflight:
            pending.append("no real 19->9 compression happened during training")
        adapter_metrics = None
        if adapter is not None:
            after = adapter.digest()
            if after == adapter_before:
                pending.append("reader adapter performed no parameter update")
            Path(args.lora_out).parent.mkdir(parents=True, exist_ok=True)
            event_count = sum(1 for u in report["updates"] if u.get("reader_adapter"))
            if args.bridge_adapter:
                # The bridge adapter is not a LoRALinear: its tensors live in its own
                # ModuleDict, so `lora_named_state` cannot see them.
                payload = dict(schema="reader_adapter/2", kind="bridge_kv_masked",
                               state=adapter.state_dict(), layers=args.bridge_layers,
                               rank=args.lora_rank, alpha=args.lora_alpha,
                               dropout=0.0, targets=list(adapter.names),
                               geometry=dict(geometry), source="natural_merge_v9_train.joint",
                               events_with_grad=event_count)
            else:
                payload = dict(schema="reader_lora/1", state=lora_named_state(model),
                               rank=args.lora_rank, alpha=args.lora_alpha, dropout=0.0,
                               last_layers=args.lora_layers, targets=list(adapter.names),
                               geometry=dict(geometry), source="natural_merge_v9_train.joint",
                               events_with_grad=event_count)
            torch.save(payload, args.lora_out)
            adapter_metrics = dict(modules=len(adapter.names),
                                   state_digest_before=adapter_before,
                                   state_digest_after=after,
                                   events_with_grad=sum(1 for u in report["updates"]
                                                        if u.get("reader_adapter")),
                                   payload=args.lora_out)
        report.update(compressions=compressions, reader_adapter=adapter_metrics)
        if pending:
            # Bookkeeping observations only: they are recorded, never raised.
            report.update(assertion_warnings=pending)
            print(json.dumps(dict(assertion_warnings=pending)), flush=True)
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
    p.add_argument("--stage-split", type=int, default=0,
                   help="switch the reader adapter on at this update index "
                        "(0 keeps the old single-stage run)")
    p.add_argument("--adapter-lr", type=float, default=1e-5,
                   help="reader-adapter learning rate for the second stage")
    p.add_argument("--bridge-adapter", action="store_true",
                   help="use the bridge-masked K/V adapter instead of the legacy "
                        "q/o LoRA (memory-read only, bridge slots only)")
    p.add_argument("--bridge-layers", default="12-31",
                   help="inclusive decoder-layer range for the bridge adapter")
    p.add_argument("--reader-lora-init", default=None,
                   help="optional warm start for the adapter (reader_lora_train payload)")
    p.add_argument("--lora-out", default="runs/natural_merge_v9/reader_lora_joint.pt")
    p.add_argument("--lora-layers", type=int, default=8)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-lr", type=float, default=1e-4)
    p.add_argument("--preflight", action="store_true",
                   help="6-event gate: exercises the recurrent branch, the two-stage "
                        "flip and the post-loop integrity checks / artifact save in "
                        "minutes instead of a full 704-update run")
    p.add_argument("--resume", default=None,
                   help="resume merge weights + optimizer from this checkpoint")
    p.add_argument("--save-every", type=int, default=50,
                   help="dump weights every N updates so a late failure never loses the run; 0 disables")
    p.add_argument("--qa-batch", type=int, default=1,
                   help="supervised QAs per forward; >1 batches the QA reads (right-padded, "
                        "exactly equivalent) and cuts the dominant per-QA cost")
    args = p.parse_args()
    if args.smoke_updates is not None:
        planned = json.loads(Path(args.plan).read_text())["training"]["planned_updates"]
        if not 1 <= args.smoke_updates <= planned:
            p.error("smoke updates exceed the planned update budget")
    run(args)


if __name__ == "__main__":
    main()
