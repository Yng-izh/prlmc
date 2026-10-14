#!/usr/bin/env python3
"""One shared rank32 natural SQuAD model; online recurrent compression training."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_natural_merge import TRAIN_PROTOCOL, validate_plan
from prepare_public_retention import sha, tokenizer_manifest
from natural_merge_runtime import (MergeStream, install, load_runtime, reset,
    source_manifest, state_digest, costs, model_identity)


def train_event(manager, merge, optimizer, qas, scaler=None):
    """Fresh reader graph per QA; one optimizer shared across all streams/events.

    Rebuild the merge graph only after restoring displaced source slots. This
    matches the existing V100 gradient bridge without holding both graphs alive.
    Prior compressed states are detached; this is online/truncated training,
    not backpropagation through the whole write history.
    """
    import torch
    import torch.nn.functional as F
    from methods.prlmc.probes import (_pool_in_place, _compacted_mask,
        active_mask_override, differentiable_memory)
    from methods.prlmc.consolidate import _select_block_indices
    m = manager
    if not qas:
        raise ValueError("no written-context original QA supervision")
    window, target = 19*m.block, 9*m.block
    start = m.total-m.used
    source = m._memory[:, start:start+window].contiguous()
    weights = m._window_importance(start, window)
    indices = _select_block_indices(source, target, m.block, weights, None, m.select_weights)
    kwargs = dict(target=target, block=m.block, carrier_indices=indices)
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        value = merge(source, **kwargs)
    leaf = value.detach().requires_grad_(True)
    mask = _compacted_mask(m, m.used-window+target, m._memory.device)
    losses = []
    with _pool_in_place(m, m._memory.device, leaf, window, target) as injection:
        with active_mask_override(m.model, mask):
            for qa in qas:
                prefix, answer = qa["prompt_ids"], qa["answer_ids"]
                ids = torch.tensor([prefix+answer], dtype=torch.long, device=m._memory.device)
                pool = injection.reinject()
                with differentiable_memory(m.model, pool):
                    logits = m.model(input_ids=ids, use_cache=False, return_dict=True).logits
                gold = torch.tensor(answer, dtype=torch.long, device=ids.device)
                ce = F.cross_entropy(logits[0, len(prefix)-1:-1].float(), gold)
                loss = ce/len(qas)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite original-QA loss")
                (scaler.scale(loss) if scaler else loss).backward()
                losses.append(float(ce.detach()))
                del pool, logits, loss, ce, ids
    if leaf.grad is None or not torch.isfinite(leaf.grad).all():
        raise ValueError("missing/nonfinite frozen-reader memory gradient")
    rebuilt = merge(source, **kwargs)
    torch.autograd.backward(rebuilt, leaf.grad)
    del rebuilt, source, leaf
    if scaler:
        scaler.unscale_(optimizer)
    gradients = [p.grad for p in merge.parameters() if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
        raise ValueError("missing/nonfinite shared-merge gradient; refuse silent skipped update")
    norm = float(torch.nn.utils.clip_grad_norm_(merge.parameters(), TRAIN_PROTOCOL["gradient_clip"]))
    if scaler:
        scaler.step(optimizer); scaler.update()
    else:
        optimizer.step()
    # The next write uses the CURRENT updated module; earlier recurrent states
    # remain detached and are never replaced by future/full reference memories.
    with torch.no_grad():
        src = m._memory[:, start:start+window].contiguous()
        current = merge(src, **kwargs)
        install(m, current, window)
    return dict(gold_ce=sum(losses)/len(losses), original_qa_count=len(qas),
        grad_norm_before_clip=norm, carrier_indices=indices.cpu().tolist(),
        source_context_hashes=sorted({q["context_sha256"] for q in qas}),
        fusion=dict(merge.last_fusion), teacher_weight=0.)


def run(args):
    plan = json.loads(Path(args.plan).read_text()); validate_plan(plan)
    if plan["partition"] != "train" or plan["experiment"] != "train":
        raise ValueError("train entry accepts only the isolated natural training plan")
    if tokenizer_manifest(args.model_path) != plan["tokenizer"]:
        raise ValueError("training tokenizer mismatch")
    if Path(args.out).exists() or Path(args.checkpoint_out).exists():
        raise ValueError("training output exists; refuse overwriting run/checkpoint")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    os.environ["S0B_GPU_SET"] = "1"
    import torch
    from native_qa_compare import atomic_json
    from methods.prlmc.probes import _checkpoint_func
    model, tokenizer, manager, merge, initial, cfg, _ = load_runtime(args.model_path, train=True)
    if float(getattr(model.config, "attention_dropout", 0.)) != 0.:
        raise ValueError("reader train-mode checkpointing requires deterministic zero dropout")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.model.gradient_checkpointing = True
    model.model._gradient_checkpointing_func = _checkpoint_func()
    model.train()  # Only activates reader gradient checkpointing; weights frozen.
    optimizer = torch.optim.AdamW(merge.parameters(), lr=TRAIN_PROTOCOL["learning_rate"],
                                 weight_decay=TRAIN_PROTOCOL["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", init_scale=128.)
    report = dict(schema="natural_merge_training/1", status="running", training=TRAIN_PROTOCOL,
        protocol=plan["protocol"], plan_sha256=sha(args.plan), train_content_sha256=plan["content_sha256"],
        dataset=plan["dataset"], split_audit=plan["split_audit"], sources=source_manifest(),
        model=model_identity(args.model_path), manager_config=cfg, resources=costs(manager, merge),
        runtime=dict(torch=torch.__version__,transformers=__import__("transformers").__version__,
            cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),physical_device=args.cuda_device),
        shared_state_sha256_before=state_digest(merge), reader_weights_frozen=True,
        initialization="fresh identity residual, fixed seed0", updates=[],
        smoke=args.smoke_updates is not None, evaluation_eligible=False)
    atomic_json(args.out, report)
    begin = time.monotonic()
    max_updates = args.smoke_updates or TRAIN_PROTOCOL["planned_updates"]
    try:
        for round_index in range(TRAIN_PROTOCOL["rounds"]):
            for episode in plan["episodes"]:
                reset(manager, initial)
                written_qas = []
                stream = MergeStream(manager, merge, "Full")
                for i, write in enumerate(episode["writes"]):
                    if manager.used+manager.block > 20*manager.block:
                        event = train_event(manager, merge, optimizer, written_qas, scaler)
                        event.update(update=len(report["updates"])+1, round=round_index,
                                     stream=episode["id"], before_write=i,
                                     recurrent_source=bool(len(report["updates"]) and i>=30))
                        report["updates"].append(event)
                        atomic_json(args.out, report)
                        print(json.dumps({k: event[k] for k in ("update", "stream", "before_write", "gold_ce")}), flush=True)
                        if len(report["updates"]) >= max_updates:
                            break
                    if stream.write(write, i) is None:
                        raise ValueError("training stream exceeded physical pool")
                    written_qas.extend(episode["contexts"][i]["qas"])
                if len(report["updates"]) >= max_updates:
                    break
            if len(report["updates"]) >= max_updates:
                break
        if len(report["updates"]) != max_updates:
            raise ValueError("scheduled updates were not completed")
        if any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise ValueError("reader weights were not frozen")
        final = state_digest(merge)
        if final == report["shared_state_sha256_before"]:
            raise ValueError("training performed no parameter update")
        eligible = args.smoke_updates is None and len(report["updates"]) == 64
        checkpoint = dict(schema="natural_merge_checkpoint/1", state_dict={k: v.detach().cpu() for k,v in merge.state_dict().items()},
            final_state_sha256=final, effective_gamma_max=.05, rank=32, evaluation_eligible=eligible,
            training=TRAIN_PROTOCOL, protocol=plan["protocol"], train_plan_sha256=sha(args.plan),
            train_content_sha256=plan["content_sha256"], training_sources=source_manifest(),
            tokenizer=plan["tokenizer"], model=report["model"],
            train_context_hashes=sorted({c["context_sha256"] for e in plan["episodes"] for c in e["contexts"]}),
            train_articles=sorted({c["article"] for e in plan["episodes"] for c in e["contexts"]}))
        Path(args.checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.checkpoint_out)
        report.update(status="complete", evaluation_eligible=eligible,
            checkpoint_sha256=sha(args.checkpoint_out), shared_state_sha256_after=final,
            elapsed_seconds=time.monotonic()-begin, peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except Exception as exc:
        report.update(status="failed", failure=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        atomic_json(args.out, report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--model-path", default="/dev/shm/s1d_reader_fp16")
    p.add_argument("--cuda-device", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint-out", required=True)
    p.add_argument("--smoke-updates", type=int)
    a = p.parse_args()
    if a.smoke_updates is not None and not 1 <= a.smoke_updates <= 64:
        p.error("smoke updates must be in 1..64 and produce an ineligible checkpoint")
    run(a)

if __name__ == "__main__":
    main()
