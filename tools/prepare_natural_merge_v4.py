#!/usr/bin/env python3
"""Immutable Dsup/Dnative supervision: answer boundary, EOS, complete QA coverage."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_natural_merge_v3 import TRAIN_V3, validate_v3_plan
from prepare_public_retention import digest, sha, write_new, tokenizer_manifest

TRAIN_V4=copy.deepcopy(TRAIN_V3)
TRAIN_V4.update(answer_supervision="one leading ASCII space + original answer; separate true EOS CE",
    eos_weight=.25, teacher_scope="Full answer-content CE and KL only; EOS excluded from eligibility",
    qa_schedule="complete original QA coverage across sealed per-context event occurrences; context-balanced losses",
    objective="context-balanced recovery content CE + eligible Full content KL + protection content hinge"
              " + first-token Selection KL anchor + separate EOS CE",
    # The content hinge is one-sided and sits at zero for most events, so the
    # protect cohort receives no gradient exactly when the shared read starts to
    # drift. The anchor below is non-negative everywhere and keeps the first
    # answer token's distribution tied to the Selection reference at every event.
    protection_anchor_weight=.10,
    # v5 routing: the pretrained 4096-D representation dominates the score and
    # the learned 32-D projection is a bounded residual view, so a domain shift
    # cannot pull routing away from what the frozen reader calls similar.
    native_prior_weight=.60, route_mix_temperature=.04, balance_strength=.15,
    carrier_selector_layer=16,
    variants={
        "Dsup":dict(architecture="grouped_gated_sparse_slot_v5",groups=8,reference_layer=16,
                    routing="top8",topk=8,local_gate_max=.20,
                    routing_score="0.6*native 4096-D cosine + 0.4*learned rank-32 cosine, tau .04,"
                                  " column log-mean load balance .15",
                    gate="slotwise hidden z-score + 3 domain-agnostic geometry features"
                         " (top1 affinity, top-k sharpness, in-block redundancy); zero-initialized",
                    carrier_selector_layer=16),
        "Dnative":dict(architecture="grouped_gated_sparse_native_v5",groups=8,reference_layer=16,
                    routing="top8",topk=8,local_gate_max=.20,
                    routing_score="0.6*native 4096-D cosine + 0.4*learned rank-32 cosine, tau .04,"
                                  " column log-mean load balance .15",
                    gate="slotwise hidden z-score + 3 domain-agnostic geometry features"
                         " (top1 affinity, top-k sharpness, in-block redundancy); zero-initialized",
                    carrier_selector_layer=16,
                    native_path="top-2 sharpened full-width source average minus carrier,"
                                " routed-confidence scaled; one zero-initialized signed gain per layer group")})


def make_schedule(episodes):
    events=[]
    for rnd in range(4):
        for ep in episodes:
            if rnd<2:
                for start in TRAIN_V4["single_window_starts"][rnd]:
                    events.append(dict(round=rnd,stream=ep["id"],stage="single_merge",window_start=start,
                                       source_indices=list(range(start,start+19))))
            else:
                for before in TRAIN_V4["recurrent_event_before_writes"]:
                    events.append(dict(round=rnd,stream=ep["id"],stage="recurrent",before_write=before,
                                       source_indices=list(range(before))))
    appearances={}
    for i,event in enumerate(events):
        for source in event["source_indices"]:
            appearances.setdefault((event["stream"],source),[]).append(i)
    by_id={ep["id"]:ep for ep in episodes}
    for i,event in enumerate(events):
        ids=[]
        for source in event["source_indices"]:
            qas=by_id[event["stream"]]["contexts"][source]["qas"]
            occurrences=appearances[(event["stream"],source)]
            j=occurrences.index(i)
            chosen=qas[j::len(occurrences)] or [qas[j%len(qas)]]
            ids.extend(q["id"] for q in chosen)
        event["qa_ids"]=ids
    return events


def event_qas(plan,rnd,episode,stage,position):
    key="window_start" if stage=="single_merge" else "before_write"
    matches=[e for e in plan["qa_events"] if e["round"]==rnd and e["stream"]==episode["id"]
             and e["stage"]==stage and e[key]==position]
    if len(matches)!=1: raise ValueError("training event missing or duplicated")
    qas={q["id"]:q for c in episode["contexts"] for q in c["qas"]}
    return [qas[i] for i in matches[0]["qa_ids"]]


def check_answer_encoding(plan,tokenizer):
    if tokenizer.eos_token_id!=plan["eos_token_id"]:
        raise ValueError("actual EOS differs from sealed plan")
    for ep in plan["episodes"]:
        for c in ep["contexts"]:
            for q in c["qas"]:
                ids=tokenizer(" "+q["answer"],add_special_tokens=False).input_ids
                if q["answer_ids"]!=ids+[tokenizer.eos_token_id] or q["content_ids"]!=ids:
                    raise ValueError("content boundary/EOS encoding mismatch")


def validate_v4_plan(plan):
    identity=dict(plan); expected=identity.pop("content_sha256",None)
    if expected!=digest(identity) or plan.get("schema")!="natural_merge_v4_training_plan/1" or plan["training"]!=TRAIN_V4:
        raise ValueError("v4 plan integrity/protocol failure")
    old=plan["original_v3_plan"]; validate_v3_plan(old)
    if plan["base_plan"]!=old["base_plan"] or plan["tokenizer"]!=old["tokenizer"]:
        raise ValueError("base protocol/tokenizer changed")
    if not isinstance(plan["eos_token_id"],int) or plan["eos_token_id"]<0:
        raise ValueError("valid actual EOS required")
    if len(plan["episodes"])!=len(old["episodes"]):raise ValueError("training streams changed")
    excluded={"answer_ids","answer_sha256","training_answer_text","content_ids","content_sha256"}
    for previous,ep in zip(old["episodes"],plan["episodes"]):
        if {k:v for k,v in previous.items() if k!="contexts"}!={k:v for k,v in ep.items() if k!="contexts"}:
            raise ValueError("original write stream changed")
        if len(previous["contexts"])!=len(ep["contexts"]):raise ValueError("contexts omitted")
        for a,c in zip(previous["contexts"],ep["contexts"]):
            if {k:v for k,v in a.items() if k!="qas"}!={k:v for k,v in c.items() if k!="qas"}:
                raise ValueError("original context changed")
            if len(a["qas"])!=len(c["qas"]):raise ValueError("original QAs omitted")
            for p,q in zip(a["qas"],c["qas"]):
                if {k:v for k,v in p.items() if k not in excluded}!={k:v for k,v in q.items() if k not in excluded}:
                    raise ValueError("original question/answer/prompt changed")
                if (not q["content_ids"] or q["answer_ids"]!=q["content_ids"]+[plan["eos_token_id"]]
                    or q["training_answer_text"]!=" "+q["answer"]
                    or digest(q["answer_ids"])!=q["answer_sha256"]
                    or digest(q["content_ids"])!=q["content_sha256"]
                    or len(q["prompt_ids"])+len(q["answer_ids"])>512):
                    raise ValueError("answer content/EOS integrity failure")
    if plan["qa_events"]!=make_schedule(plan["episodes"]) or len(plan["qa_events"])!=64:
        raise ValueError("deterministic QA event schedule changed")
    all_ids={q["id"] for e in plan["episodes"] for c in e["contexts"] for q in c["qas"]}
    seen={i for e in plan["qa_events"] for i in e["qa_ids"]}
    if all_ids!=seen:raise ValueError("incomplete original QA training coverage")


def build_plan(original,tokenizer,source_sha):
    validate_v3_plan(original)
    if tokenizer.eos_token_id is None:raise ValueError("tokenizer has no actual EOS")
    episodes=copy.deepcopy(original["episodes"])
    for ep in episodes:
        for c in ep["contexts"]:
            for q in c["qas"]:
                q["training_answer_text"]=" "+q["answer"]
                q["content_ids"]=tokenizer(q["training_answer_text"],add_special_tokens=False).input_ids
                q["content_sha256"]=digest(q["content_ids"])
                q["answer_ids"]=q["content_ids"]+[tokenizer.eos_token_id]
                q["answer_sha256"]=digest(q["answer_ids"])
    plan=dict(schema="natural_merge_v4_training_plan/1",original_v3_plan=original,
              original_v3_file_sha256=source_sha,base_plan=original["base_plan"],
              tokenizer=original["tokenizer"],eos_token_id=tokenizer.eos_token_id,
              training=TRAIN_V4,episodes=episodes,qa_events=make_schedule(episodes))
    plan["content_sha256"]=digest(plan)
    validate_v4_plan(plan);check_answer_encoding(plan,tokenizer)
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v3-train-plan",required=True);p.add_argument("--out",required=True)
    p.add_argument("--model-path",default="/dev/shm/s1d_reader_fp16")
    args=p.parse_args();original=json.loads(Path(args.v3_train_plan).read_text())
    if tokenizer_manifest(args.model_path)!=original["tokenizer"]:raise ValueError("tokenizer changed")
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model_path,local_files_only=True)
    plan=build_plan(original,tokenizer,sha(args.v3_train_plan));write_new(args.out,plan)
    print(json.dumps(dict(out=args.out,events=len(plan["qa_events"]),
        qa_observations=sum(len(e["qa_ids"]) for e in plan["qa_events"]),
        unique_qas=len({i for e in plan["qa_events"] for i in e["qa_ids"]}),
        eos_token_id=plan["eos_token_id"],content_sha256=plan["content_sha256"])))

if __name__=="__main__":main()
