#!/usr/bin/env python3
"""Freeze one natural checkpoint after complete E1 DEVELOPMENT mechanism evidence."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_public_retention import digest,sha,write_new
from prepare_natural_merge import validate_plan
from verify_natural_merge import verify
from natural_merge_runtime import model_identity,source_manifest


def freeze(args):
    suite=json.loads(Path(args.suite).read_text())
    identity=dict(suite);expected=identity.pop("content_sha256",None)
    if expected!=digest(identity) or suite["schema"]!="natural_merge_suite/1":
        raise ValueError("sealed suite integrity failure")
    plan=json.loads(Path(args.development_plan).read_text());validate_plan(plan)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model_path,local_files_only=True)
    result=json.loads(Path(args.development_result).read_text());verify(result,plan,tokenizer)
    if result["plan_sha256"]!=sha(args.development_plan) or result["checkpoint_sha256"]!=sha(args.checkpoint):
        raise ValueError("development checkpoint/plan hash mismatch")
    if plan["partition"]!="development" or plan["experiment"]!="E1" or result["diagnostic_only"]:
        raise ValueError("natural-trained checkpoint E1 development comparison required")
    if result["sharding"]["count"]!=1 or len(result["episodes"])!=len(plan["episodes"]):
        raise ValueError("complete development coverage required")
    summary=result["summary"]
    if (summary["reader_reference"]=="unreadable_workload" or summary["reader_suitability"]!="minimum_multisource_readability" or summary["unselected_source_full_vs_masked"]["net"]<=0
            or summary["shared_vs_selection"]["net"]<=0):
        raise ValueError("E1 development mechanism gate failed; no readable positive noncarrier fusion evidence")
    current_model=model_identity(args.model_path);sources=source_manifest()
    if result["model"]!=current_model or result["sources"]!=sources:
        raise ValueError("development code/model changed before freeze")
    allowed=[]
    for name,entry in suite["plans"].items():
        path=Path(args.suite).parent/entry["path"]
        if sha(path)!=entry["sha256"]:
            raise ValueError("suite plan file changed")
        p=json.loads(path.read_text());validate_plan(p)
        if p["partition"] in ("formal","transfer"):
            allowed.append(p["content_sha256"])
        if name=="E1-development" and sha(path)!=sha(args.development_plan):
            raise ValueError("development plan is not from the sealed suite")
    payload=dict(schema="natural_merge_freeze/1",suite_sha256=sha(args.suite),checkpoint_sha256=sha(args.checkpoint),
        sources=sources,model=current_model,development_result_sha256=sha(args.development_result),
        development_plan_sha256=sha(args.development_plan),allowed_plan_content_sha256=allowed,
        gate="E1 development: readable Full, positive Shared-Selection and full-masked noncarrier QA",
        limitation="gate is a progression rule, not proof of statistically reliable generalization",
        transfer_policy="NaturalQA never used for checkpoint or protocol selection")
    payload["content_sha256"]=digest(payload)
    write_new(args.out,payload);print(json.dumps(payload,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("suite","checkpoint","development-plan","development-result","out"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--model-path",default="/dev/shm/s1d_reader_fp16")
    freeze(p.parse_args())

if __name__=="__main__":main()
