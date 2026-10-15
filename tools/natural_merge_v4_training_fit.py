#!/usr/bin/env python3
"""Frozen generation on a fixed TRAIN window: fitting diagnosis, never a test score."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_natural_merge_v4 import validate_v4_plan,TRAIN_V4
from prepare_public_retention import sha,tokenizer_manifest
from natural_merge_runtime import (load_runtime,reset,state_digest,source_manifest,model_identity,costs,scores)
from natural_merge_eval import e1_episode
from prepare_natural_merge_v4 import check_answer_encoding
from verify_natural_merge import summarize,final_predictions


def verify_training_predictions(row,episode,tokenizer):
    """Validate originals/tokens/scores, without a development or formal certificate."""
    qas={q["id"]:q for c in episode["contexts"] for q in c["qas"]}
    for arm in ("Full","Selection","SharedMerge","CarrierOnlyFusion"):
        saved=row["arms"][arm]["predictions"]
        if len(saved)!=len(qas) or {p["id"] for p in saved}!=qas.keys():
            raise ValueError("training-fit original QA coverage differs")
        for p in saved:
            q=qas[p["id"]]
            if (len(p["token_ids"])>10
                    or tokenizer.decode(p["token_ids"],skip_special_tokens=False)!=p["prediction"]
                    or tokenizer.decode(p["token_ids"],skip_special_tokens=True)!=p["auxiliary_prediction"]
                    or p["context_sha256"]!=q["context_sha256"] or p["source_index"]!=q["source_index"]
                    or p["question"]!=q["question"] or p["original_answers"]!=q["answers"]
                    or p["scoring_target"]!=q["scoring_target"]
                    or p["scores"]!=scores(p["prediction"],q,p["auxiliary_prediction"])):
                raise ValueError("training-fit original QA/token/scoring mismatch")


def run(args):
    plan=json.loads(Path(args.plan).read_text());validate_v4_plan(plan)
    if Path(args.out).exists():raise ValueError("training-fit output exists")
    if tokenizer_manifest(args.model_path)!=plan["tokenizer"]:raise ValueError("tokenizer mismatch")
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda_device);os.environ["S0B_GPU_SET"]="1"
    import torch
    from native_qa_compare import atomic_json
    model,tokenizer,manager,merge,initial,cfg,saved=load_runtime(args.model_path,args.checkpoint)
    if (saved.get("variant") not in TRAIN_V4["variants"] or saved.get("training")!=TRAIN_V4
            or saved.get("train_plan_sha256")!=sha(args.plan)
            or saved.get("train_content_sha256")!=plan["content_sha256"]
            or saved.get("model")!=model_identity(args.model_path)):
        raise ValueError("training-fit checkpoint/plan/model mismatch")
    check_answer_encoding(plan,tokenizer)
    frozen=state_digest(merge)
    if saved["final_state_sha256"]!=frozen:raise ValueError("checkpoint state hash mismatch")
    # Fixed before any results; never select an easy/favorable training window.
    original=plan["episodes"][0]
    episode=dict(id="training-fit-first-stream-first19",
        contexts=original["contexts"][:19],writes=original["writes"][:19])
    report=dict(schema="natural_merge_v4_training_fit/1",status="running",experiment="E1",
        variant=saved["variant"],architecture=saved["architecture"],partition="train",
        diagnostic_only=True,training_fit_only=True,evaluation_eligible=False,
        interpretation="in-training fitting diagnosis; cannot support generalization or progression gates",
        plan_sha256=sha(args.plan),checkpoint_sha256=sha(args.checkpoint),protocol=plan["base_plan"]["protocol"],
        sources=source_manifest(),model=model_identity(args.model_path),resources=costs(manager,merge),
        shared_state_sha256_before=frozen,training_performed=False,optimizer_created=False,episodes=[],
        runtime=dict(physical_device=args.cuda_device,gpu=torch.cuda.get_device_name(0)))
    atomic_json(args.out,report);begun=time.monotonic()
    try:
        reset(manager,initial)
        def progress(row,arm):
            report["current_episode"]=row;report["progress"]=arm;atomic_json(args.out,report)
            print(json.dumps(dict(arm=arm,training_fit_only=True)),flush=True)
        with torch.no_grad():row=e1_episode(manager,merge,tokenizer,episode,progress)
        verify_training_predictions(row,episode,tokenizer)
        report["episodes"]=[row];report.pop("current_episode",None)
        if state_digest(merge)!=frozen or any(p.requires_grad for p in merge.parameters()) or any(p.requires_grad for p in model.parameters()):
            raise ValueError("training-fit frozen parameter invariant failed")
        evidence=summarize(report)
        full={p["id"]:p for p in final_predictions(report,"Full")}
        selection={p["id"]:p for p in final_predictions(report,"Selection")}
        shared={p["id"]:p for p in final_predictions(report,"SharedMerge")}
        lost=sorted(q for q in full if full[q]["scores"]["author_containment"] and not selection[q]["scores"]["author_containment"])
        report.update(status="complete",summary=evidence,full_correct_selection_wrong=lost,
            recovered_compression_losses=[q for q in lost if shared[q]["scores"]["author_containment"]],
            prediction_validation="original QA/token decoding/score checks passed; TRAIN diagnosis only",
            shared_state_sha256_after=frozen,elapsed_seconds=time.monotonic()-begun)
    except Exception as exc:
        report.update(status="failed",failure=dict(type=type(exc).__name__,message=str(exc)));raise
    finally:atomic_json(args.out,report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",required=True);p.add_argument("--checkpoint",required=True)
    p.add_argument("--cuda-device",type=int,required=True);p.add_argument("--out",required=True)
    p.add_argument("--model-path",default="/dev/shm/s1d_reader_fp16")
    run(p.parse_args())


if __name__=="__main__":main()
