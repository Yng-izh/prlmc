#!/usr/bin/env python3
"""Fresh DsupV7/DnativeV7 64-update training (v7 plan; 13->9 merge geometry).

Cloned from the v6 trainer with three v7 differences:

1. the plan is `natural_merge_v7_training_plan/1` and the variants are
   DsupV7/DnativeV7 (TRAIN_V7 is never TRAIN_V4/TRAIN_V6, so v6 checkpoints and
   v7 checkpoints cannot be loaded across generations);
2. the supervised merge uses the 13->9 geometry from the plan, and the recurrent
   student stream runs the same 13/9 streaming geometry the evaluation stream
   runs (events fire at the `used >= min(budget, window+1)` trigger);
3. the checkpoint records `source_manifest`, so t2's strict source-hash check
   can verify it instead of refusing it as a pre-manifest artifact.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_natural_merge_v7 import TRAIN_V7, validate_v7_plan, check_answer_encoding
from prepare_natural_merge_v4 import event_qas
from prepare_public_retention import sha, tokenizer_manifest
from natural_merge_runtime import (MergeStream, install, load_runtime, reset,
    source_manifest, state_digest, costs, model_identity)
from natural_merge_v4_objective import train_event, NativeOrigins, make_optimizer, full_targets, recurrent_targets

GEOMETRY = TRAIN_V7["geometry"]


def run(args):
    plan = json.loads(Path(args.plan).read_text()); validate_v7_plan(plan)
    if args.variant not in TRAIN_V7["variants"]: raise ValueError("unknown variant")
    if tokenizer_manifest(args.model_path) != plan["tokenizer"]: raise ValueError("tokenizer mismatch")
    if Path(args.out).exists() or Path(args.checkpoint_out).exists(): raise ValueError("training output exists")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device); os.environ["S0B_GPU_SET"] = "1"
    import torch
    from native_qa_compare import atomic_json
    from methods.prlmc.probes import _checkpoint_func
    variant = TRAIN_V7["variants"][args.variant]
    model,tokenizer,manager,merge,initial,cfg,_ = load_runtime(args.model_path,train=True,
        architecture=variant["architecture"], budget_blocks=GEOMETRY["budget_blocks"],
        window_blocks=GEOMETRY["window_blocks"], target_blocks=GEOMETRY["target_blocks"])
    if float(getattr(model.config,"attention_dropout",0.)) != 0.:
        raise ValueError("reader checkpointing requires deterministic zero dropout")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.model.gradient_checkpointing=True; model.model._gradient_checkpointing_func=_checkpoint_func()
    model.train()  # Reader parameters stay frozen; activates gradient checkpointing.
    check_answer_encoding(plan,tokenizer)
    optimizer=make_optimizer(merge, plan=TRAIN_V7)
    scaler=torch.amp.GradScaler("cuda",init_scale=128.)
    base=plan["base_plan"]; begun=time.monotonic()
    report=dict(schema="natural_merge_v7_training/1",status="running",variant=args.variant,
        architecture=variant["architecture"],training=TRAIN_V7,protocol=base["protocol"],
        objective=TRAIN_V7["objective"],geometry=dict(GEOMETRY),
        recovery_gate=dict(TRAIN_V7["recovery_gate"]),
        plan_sha256=sha(args.plan),train_content_sha256=plan["content_sha256"],
        dataset=base["dataset"],split_audit=base["split_audit"],sources=source_manifest(),
        model=model_identity(args.model_path),manager_config=cfg,resources=costs(manager,merge),
        runtime=dict(torch=torch.__version__,transformers=__import__("transformers").__version__,
            cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),physical_device=args.cuda_device),
        reader_weights_frozen=True,shared_state_sha256_before=state_digest(merge),updates=[],
        smoke=args.smoke_updates is not None,evaluation_eligible=False,
        initialization="fresh identity residual, effective gate .02, fixed seed0; no warm start",
        optimizer_groups=[{k:v for k,v in g.items() if k != "params"} for g in optimizer.param_groups])
    atomic_json(args.out,report); maximum=args.smoke_updates or TRAIN_V7["planned_updates"]
    try:
        for round_index in range(4):
            for episode in plan["episodes"]:
                if round_index < 2:
                    tasks=[]
                    length=TRAIN_V7["single_window_contexts"]
                    for start in TRAIN_V7["single_window_starts"][round_index]:
                        tasks.append((start,episode["contexts"][start:start+length],episode["writes"][start:start+length]))
                    for start,contexts,writes in tasks:
                        reset(manager,initial)
                        # Accumulation stream: the sealed 19-context v4 window has
                        # to be fully resident before the supervised event, so the
                        # write path runs the pre-registered 19/9 table geometry
                        # (it cannot compact inside 19 writes). The supervised
                        # merge itself uses the v7 13->9 geometry, which is what
                        # the evaluation stream runs.
                        stream=MergeStream(manager,merge,"Full")
                        origins=NativeOrigins()
                        with torch.no_grad():
                            for i,write in enumerate(writes):
                                if stream.write(write,i) is None: raise ValueError("single-merge training exceeds pool")
                                origins.append(contexts[i]["qas"][0]["source_index"])
                        qas=event_qas(plan,round_index,episode,"single_merge",start); targets=full_targets(manager,qas)
                        event=train_event(manager,merge,optimizer,qas,targets,args.variant,origins,scaler,plan=TRAIN_V7)
                        event.update(stage="single_merge",window_start=start,before_write=len(writes))
                        record_event(report,event,round_index,episode,args.out,begun)
                        del targets
                        if len(report["updates"]) >= maximum: break
                else:
                    references=recurrent_targets(manager,merge,initial,episode,round_index,plan,TRAIN_V7)
                    reset(manager,initial)
                    # The student stream runs the v7 13/9 geometry, so the
                    # recurrent events land exactly where the evaluation stream
                    # would compact: used >= min(20, 13+1) = 14 blocks.
                    stream=MergeStream(manager,merge,"Full",
                        budget_blocks=GEOMETRY["budget_blocks"],
                        window_blocks=GEOMETRY["window_blocks"],
                        target_blocks=GEOMETRY["target_blocks"])
                    origins=NativeOrigins()
                    for i,write in enumerate(episode["writes"]):
                        if i in TRAIN_V7["recurrent_event_before_writes"]:
                            if manager.used < (GEOMETRY["window_blocks"]+1)*manager.block:
                                raise ValueError("recurrent event is not at the v7 13/9 streaming trigger")
                            qas=event_qas(plan,round_index,episode,"recurrent",i)
                            event=train_event(manager,merge,optimizer,qas,references[i],args.variant,origins,scaler,plan=TRAIN_V7)
                            event.update(stage="recurrent",before_write=i)
                            record_event(report,event,round_index,episode,args.out,begun)
                            if len(report["updates"]) >= maximum: break
                        with torch.no_grad():
                            if stream.write(write,i) is None: raise ValueError("student training exceeds pool")
                            origins.append(episode["contexts"][i]["qas"][0]["source_index"])
                    del references
                if len(report["updates"]) >= maximum: break
            if len(report["updates"]) >= maximum: break
        if len(report["updates"]) != maximum: raise ValueError("training schedule incomplete")
        if any(p.requires_grad or p.grad is not None for p in model.parameters()): raise ValueError("reader not frozen")
        final=state_digest(merge)
        if final==report["shared_state_sha256_before"]: raise ValueError("training performed no parameter update")
        seen=sorted({q["context_sha256"] for u in report["updates"] for q in u["qas"]})
        expected=sorted({c["context_sha256"] for e in plan["episodes"] for c in e["contexts"]})
        eligible=args.smoke_updates is None and len(report["updates"])==64
        if eligible and [u["stage"] for u in report["updates"]] != ["single_merge"]*32+["recurrent"]*32:
            raise ValueError("single/recurrent training curriculum changed")
        if eligible and seen!=expected: raise ValueError("some train contexts never entered compression supervision")
        covered_qas=sorted({q["id"] for u in report["updates"] for q in u["qas"]})
        expected_qas=sorted({q["id"] for e in plan["episodes"] for c in e["contexts"] for q in c["qas"]})
        if eligible and covered_qas!=expected_qas:raise ValueError("not every original training QA was supervised")
        checkpoint=dict(schema="natural_merge_checkpoint/1",architecture=variant["architecture"],
            variant=args.variant,state_dict={k:v.detach().cpu() for k,v in merge.state_dict().items()},
            final_state_sha256=final,effective_gamma_max=.05,rank=32,evaluation_eligible=eligible,
            module_configuration=variant, global_rms_budget=TRAIN_V7["global_rms_budget"],
            training=TRAIN_V7,objective=TRAIN_V7["objective"],protocol=base["protocol"],
            geometry=dict(GEOMETRY),recovery_gate=dict(TRAIN_V7["recovery_gate"]),
            train_plan_sha256=sha(args.plan),train_content_sha256=plan["content_sha256"],
            training_sources=source_manifest(),source_manifest=source_manifest(),
            tokenizer=plan["tokenizer"],model=report["model"],
            train_context_hashes=expected,train_articles=sorted({c["article"] for e in plan["episodes"] for c in e["contexts"]}))
        Path(args.checkpoint_out).parent.mkdir(parents=True,exist_ok=True); torch.save(checkpoint,args.checkpoint_out)
        report.update(status="complete",evaluation_eligible=eligible,checkpoint_sha256=sha(args.checkpoint_out),
            shared_state_sha256_after=final,supervised_contexts=seen, supervised_qa_ids=covered_qas, elapsed_seconds=time.monotonic()-begun,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as exc:
        report.update(status="failed",failure=dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally: atomic_json(args.out,report)


def record_event(report,event,round_index,episode,out,begun):
    from native_qa_compare import atomic_json
    event.update(update=len(report["updates"])+1,round=round_index,stream=episode["id"],
                 elapsed_seconds=time.monotonic()-begun)
    report["updates"].append(event); atomic_json(out,report)
    print(json.dumps({k:event[k] for k in ("update","stage","stream","gold_ce","weighted_gold_ce","teacher_kl")}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",required=True); p.add_argument("--variant",required=True,choices=["DsupV7","DnativeV7"])
    p.add_argument("--model-path",default="/dev/shm/s1d_reader_fp16")
    p.add_argument("--cuda-device",type=int,required=True)
    p.add_argument("--out",required=True); p.add_argument("--checkpoint-out",required=True)
    p.add_argument("--smoke-updates",type=int)
    args=p.parse_args()
    if args.smoke_updates is not None and not 1<=args.smoke_updates<=64: p.error("smoke updates must be in 1..64")
    run(args)


if __name__=="__main__": main()
