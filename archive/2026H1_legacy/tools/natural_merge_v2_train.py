#!/usr/bin/env python3
"""Common 64-update A/B/C curriculum; frozen reader and training-only Full teacher."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_natural_merge_v2 import TRAIN_V2, selected_qas, validate_v2_plan
from prepare_public_retention import sha, tokenizer_manifest
from natural_merge_runtime import (MergeStream, install, load_runtime, reset,
    source_manifest, state_digest, costs, model_identity)


def answer_logits(model, qa, device):
    import torch
    prefix, answer = qa["prompt_ids"], qa["answer_ids"]
    ids = torch.tensor([prefix + answer], dtype=torch.long, device=device)
    logits = model(input_ids=ids, use_cache=False, return_dict=True).logits
    return logits[0, len(prefix)-1:-1].float(), torch.tensor(answer, dtype=torch.long, device=device)


def full_targets(manager, qas):
    """No graph/optimizer. Teacher distribution stored on CPU, one QA at a time."""
    import torch
    import torch.nn.functional as F
    result = {}
    with torch.no_grad():
        for qa in qas:
            logits, gold = answer_logits(manager.model, qa, manager._memory.device)
            ce = float(F.cross_entropy(logits, gold))
            result[qa["id"]] = dict(gold_ce=ce,
                log_probs=F.log_softmax(logits, dim=-1).to(device="cpu", dtype=torch.float16),
                tf_token_accuracy=float((logits.argmax(-1) == gold).float().mean()),
                context_sha256=qa["context_sha256"],
                prompt_ids=list(qa["prompt_ids"]), answer_ids=list(qa["answer_ids"]))
    return result


def recovery_weights(full_ce, selection_ce):
    """Detached, bounded, mean-one training weights; evaluation is never consulted."""
    if len(full_ce) != len(selection_ce) or not full_ce:
        raise ValueError("nonempty paired training CEs required")
    raw = [1. + min(2., max(0., s - f)) for f, s in zip(full_ce, selection_ce)]
    average = sum(raw)/len(raw)
    return [w/average for w in raw]


def teacher_kl(student_logits, teacher_log_probs):
    import torch.nn.functional as F
    if student_logits.shape != teacher_log_probs.shape:
        raise ValueError("teacher/student answer positions or vocabulary differ")
    temperature = TRAIN_V2["teacher_temperature"]
    # CPU fp16 caching introduces rounding: renormalize before computing KL.
    teacher_log = F.log_softmax(teacher_log_probs.to(student_logits.device).float()/temperature, dim=-1)
    student_log = F.log_softmax(student_logits.float()/temperature, dim=-1)
    return F.kl_div(student_log, teacher_log, reduction="batchmean", log_target=True) * temperature**2


def realized_residual(candidate, carriers):
    """Measure the actual stored-dtype update, without a full FP32 pool copy."""
    import torch
    residual_sq = carrier_sq = changed = count = 0
    with torch.no_grad():
        for output, carrier in zip(candidate, carriers):
            residual_sq += float((output.float()-carrier.float()).square().sum())
            carrier_sq += float(carrier.float().square().sum())
            changed += int((output != carrier).sum())
            count += output.numel()
    return dict(stored_dtype=str(candidate.dtype),
        stored_residual_rms_ratio=(residual_sq/max(carrier_sq,1e-24))**.5,
        stored_changed_fraction=changed/count)


def train_event(manager, merge, optimizer, qas, targets, variant, scaler=None):
    import torch
    import torch.nn.functional as F
    from methods.prlmc.probes import (_pool_in_place, _compacted_mask,
        active_mask_override, differentiable_memory)
    from methods.prlmc.consolidate import _select_block_indices
    if variant not in TRAIN_V2["variants"] or not qas or len({q["id"] for q in qas}) != len(qas):
        raise ValueError("known variant and unique nonempty original QAs required")
    for qa in qas:
        teacher = targets.get(qa["id"], {})
        if (teacher.get("context_sha256") != qa["context_sha256"]
                or teacher.get("prompt_ids") != qa["prompt_ids"]
                or teacher.get("answer_ids") != qa["answer_ids"]):
            raise ValueError("Full training teacher provenance mismatch")
    m = manager; window, target = 19*m.block, 9*m.block
    if m.used < window:
        raise ValueError("actual compression required for supervision")
    start = m.total-m.used
    source = m._memory[:, start:start+window].contiguous()
    indices = _select_block_indices(source, target, m.block,
        m._window_importance(start, window), None, m.select_weights)
    kwargs = dict(target=target, block=m.block, carrier_indices=indices)
    carriers = source.reshape(m.num_layers,19,m.block,m.hidden).index_select(1,indices).flatten(1,2)
    mask = _compacted_mask(m, m.used-window+target, m._memory.device)
    full_ce = [targets[q["id"]]["gold_ce"] for q in qas]
    selection_ce = []
    # Every variant evaluates the same local Selection reference; A ignores its
    # weights/KD but records them, keeping diagnostics and reader compute matched.
    with torch.no_grad(), _pool_in_place(m,m._memory.device,carriers,window,target) as injection:
        with active_mask_override(m.model,mask):
            for qa in qas:
                injection.reinject()
                logits, gold = answer_logits(m.model,qa,m._memory.device)
                selection_ce.append(float(F.cross_entropy(logits,gold)))
    if not all(torch.isfinite(torch.tensor(full_ce+selection_ce))):
        raise ValueError("nonfinite training reference CE")
    weights = recovery_weights(full_ce,selection_ce) if variant != "A" else [1.]*len(qas)
    eligible = [variant != "A" and f <= TRAIN_V2["teacher_max_gold_ce"] for f in full_ce]
    eligible_count = sum(eligible)
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad(): value = merge(source,**kwargs)
    # The frozen reader's backward materializes a full physical-pool gradient.
    # Keep only the compacted leaf on the GPU during that phase; the original
    # window and selected carriers are not inputs to the reader graph. Restore
    # the exact detached source for the compactor VJP after all QAs finish.
    source_host = source.detach().cpu() if source.is_cuda else source.detach()
    del source, carriers
    leaf = value.detach().requires_grad_(True)
    rows = []; objective = weighted_ce = kd_mean = 0.
    with _pool_in_place(m,m._memory.device,leaf,window,target) as injection:
        with active_mask_override(m.model,mask):
            for i,qa in enumerate(qas):
                pool = injection.reinject()
                with differentiable_memory(m.model,pool):
                    logits,gold = answer_logits(m.model,qa,m._memory.device)
                ce = F.cross_entropy(logits,gold)
                kd = teacher_kl(logits,targets[qa["id"]]["log_probs"]) if eligible[i] else logits.new_zeros(())
                ce_term = weights[i]*ce/len(qas)
                kd_term = TRAIN_V2["teacher_weight"]*kd/max(eligible_count,1)
                loss = ce_term+kd_term
                if not torch.isfinite(loss): raise ValueError("nonfinite v2 objective")
                (scaler.scale(loss) if scaler else loss).backward()
                weighted_ce += float(ce_term.detach()); kd_mean += float(kd.detach())/max(eligible_count,1)
                objective += float(loss.detach())
                rows.append(dict(id=qa["id"],context_sha256=qa["context_sha256"],
                    full_ce=full_ce[i],selection_ce=selection_ce[i],shared_ce=float(ce.detach()),
                    weight=weights[i],teacher_eligible=eligible[i],teacher_kl=float(kd.detach()),
                    full_tf_token_accuracy=targets[qa["id"]]["tf_token_accuracy"],
                    shared_tf_token_accuracy=float((logits.detach().argmax(-1)==gold).float().mean())))
                del pool,logits,ce,kd,ce_term,kd_term,loss
    if leaf.grad is None or not torch.isfinite(leaf.grad).all():
        raise ValueError("missing/nonfinite frozen-reader memory gradient")
    source = source_host.to(m._memory.device)
    rebuilt = merge(source,**kwargs); torch.autograd.backward(rebuilt,leaf.grad)
    del rebuilt,leaf,value
    if scaler: scaler.unscale_(optimizer)
    gradients = {name:p.grad for name,p in merge.named_parameters() if p.grad is not None}
    if not gradients or any(not torch.isfinite(g).all() for g in gradients.values()):
        raise ValueError("missing/nonfinite v2 module gradient")
    gradient_norms = {name:float(g.norm()) for name,g in gradients.items()}
    norm = float(torch.nn.utils.clip_grad_norm_(merge.parameters(),TRAIN_V2["gradient_clip"]))
    if not torch.isfinite(torch.tensor(norm)):
        raise ValueError("nonfinite total module gradient norm")
    if scaler: scaler.step(optimizer); scaler.update()
    else: optimizer.step()
    with torch.no_grad():
        current = merge(source,**kwargs)
        carriers = source.reshape(m.num_layers,19,m.block,m.hidden).index_select(1,indices).flatten(1,2)
        stored = realized_residual(current,carriers)
        fusion = dict(merge.last_fusion)
        install(m,current,window)
    return dict(objective=objective,gold_ce=sum(r["shared_ce"] for r in rows)/len(rows),
        weighted_gold_ce=weighted_ce,teacher_kl=kd_mean,teacher_eligible_count=eligible_count,
        teacher_weight=0. if variant=="A" else TRAIN_V2["teacher_weight"],
        original_qa_count=len(qas),qas=rows,carrier_indices=indices.cpu().tolist(),
        gradient_norms_before_clip=gradient_norms,grad_norm_before_clip=norm,
        fusion=fusion,realized_residual=stored,
        selection_reference="same pre-event student history with only current window selected",
        qa_measurement="before optimizer update; installed residual measured after update")


def recurrent_targets(manager,merge,initial,episode,round_index):
    """Independent uncompressed writer history, never reused as student memory."""
    reset(manager,initial); stream = MergeStream(manager,merge,"Full")
    result = {}
    for i,write in enumerate(episode["writes"]):
        if i in TRAIN_V2["recurrent_event_before_writes"]:
            result[i] = full_targets(manager,selected_qas(episode["contexts"][:i],round_index))
        with __import__("torch").no_grad():
            if stream.write(write,i) is None: raise ValueError("uncompressed training teacher exceeds physical pool")
    if sorted(result) != TRAIN_V2["recurrent_event_before_writes"]:
        raise ValueError("incomplete Full training teacher schedule")
    return result


def run(args):
    plan = json.loads(Path(args.plan).read_text()); validate_v2_plan(plan)
    if args.variant not in TRAIN_V2["variants"]: raise ValueError("unknown variant")
    if tokenizer_manifest(args.model_path) != plan["tokenizer"]: raise ValueError("tokenizer mismatch")
    if Path(args.out).exists() or Path(args.checkpoint_out).exists(): raise ValueError("training output exists")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device); os.environ["S0B_GPU_SET"] = "1"
    import torch
    from native_qa_compare import atomic_json
    from methods.prlmc.probes import _checkpoint_func
    variant = TRAIN_V2["variants"][args.variant]
    model,tokenizer,manager,merge,initial,cfg,_ = load_runtime(args.model_path,train=True,
        architecture=variant["architecture"])
    if float(getattr(model.config,"attention_dropout",0.)) != 0.:
        raise ValueError("reader checkpointing requires deterministic zero dropout")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.model.gradient_checkpointing=True; model.model._gradient_checkpointing_func=_checkpoint_func()
    model.train()  # Reader parameters stay frozen; activates gradient checkpointing.
    optimizer=torch.optim.AdamW(merge.parameters(),lr=TRAIN_V2["learning_rate"],weight_decay=TRAIN_V2["weight_decay"])
    scaler=torch.amp.GradScaler("cuda",init_scale=128.)
    base=plan["base_plan"]; begun=time.monotonic()
    report=dict(schema="natural_merge_v2_training/1",status="running",variant=args.variant,
        architecture=variant["architecture"],training=TRAIN_V2,protocol=base["protocol"],
        objective=variant["objective"],plan_sha256=sha(args.plan),train_content_sha256=plan["content_sha256"],
        dataset=base["dataset"],split_audit=base["split_audit"],sources=source_manifest(),
        model=model_identity(args.model_path),manager_config=cfg,resources=costs(manager,merge),
        runtime=dict(torch=torch.__version__,transformers=__import__("transformers").__version__,
            cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),physical_device=args.cuda_device),
        reader_weights_frozen=True,shared_state_sha256_before=state_digest(merge),updates=[],
        smoke=args.smoke_updates is not None,evaluation_eligible=False,
        initialization="fresh identity residual, fixed seed0; no warm start")
    atomic_json(args.out,report); maximum=args.smoke_updates or TRAIN_V2["planned_updates"]
    try:
        for round_index in range(4):
            for episode in plan["episodes"]:
                if round_index < 2:
                    tasks=[]
                    for start in TRAIN_V2["single_window_starts"][round_index]:
                        tasks.append((start,episode["contexts"][start:start+19],episode["writes"][start:start+19]))
                    for start,contexts,writes in tasks:
                        reset(manager,initial); stream=MergeStream(manager,merge,"Full")
                        with torch.no_grad():
                            for i,write in enumerate(writes):
                                if stream.write(write,i) is None: raise ValueError("single-merge training exceeds pool")
                        qas=selected_qas(contexts,round_index); targets=full_targets(manager,qas)
                        event=train_event(manager,merge,optimizer,qas,targets,args.variant,scaler)
                        event.update(stage="single_merge",window_start=start,before_write=19)
                        record_event(report,event,round_index,episode,args.out,begun)
                        del targets
                        if len(report["updates"]) >= maximum: break
                else:
                    references=recurrent_targets(manager,merge,initial,episode,round_index)
                    reset(manager,initial); stream=MergeStream(manager,merge,"Full")
                    for i,write in enumerate(episode["writes"]):
                        if manager.used+manager.block > 20*manager.block:
                            qas=selected_qas(episode["contexts"][:i],round_index)
                            event=train_event(manager,merge,optimizer,qas,references[i],args.variant,scaler)
                            event.update(stage="recurrent",before_write=i)
                            record_event(report,event,round_index,episode,args.out,begun)
                            if len(report["updates"]) >= maximum: break
                        with torch.no_grad():
                            if stream.write(write,i) is None: raise ValueError("student training exceeds pool")
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
        checkpoint=dict(schema="natural_merge_checkpoint/1",architecture=variant["architecture"],
            variant=args.variant,state_dict={k:v.detach().cpu() for k,v in merge.state_dict().items()},
            final_state_sha256=final,effective_gamma_max=.05,rank=32,evaluation_eligible=eligible,
            training=TRAIN_V2,objective=variant["objective"],protocol=base["protocol"],
            train_plan_sha256=sha(args.plan),train_content_sha256=plan["content_sha256"],
            training_sources=source_manifest(),tokenizer=plan["tokenizer"],model=report["model"],
            train_context_hashes=expected,train_articles=sorted({c["article"] for e in plan["episodes"] for c in e["contexts"]}))
        Path(args.checkpoint_out).parent.mkdir(parents=True,exist_ok=True); torch.save(checkpoint,args.checkpoint_out)
        report.update(status="complete",evaluation_eligible=eligible,checkpoint_sha256=sha(args.checkpoint_out),
            shared_state_sha256_after=final,supervised_contexts=seen,elapsed_seconds=time.monotonic()-begun,
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
    p.add_argument("--plan",required=True); p.add_argument("--variant",required=True,choices=["A","B","C"])
    p.add_argument("--model-path",default="/dev/shm/s1d_reader_fp16")
    p.add_argument("--cuda-device",type=int,required=True)
    p.add_argument("--out",required=True); p.add_argument("--checkpoint-out",required=True)
    p.add_argument("--smoke-updates",type=int)
    args=p.parse_args()
    if args.smoke_updates is not None and not 1<=args.smoke_updates<=64: p.error("smoke updates must be in 1..64")
    run(args)


if __name__=="__main__": main()
