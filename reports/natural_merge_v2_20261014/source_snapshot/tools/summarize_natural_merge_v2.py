#!/usr/bin/env python3
"""Verify A/B/C E1 development results and report compression-loss recovery."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_public_retention import sha, write_new
from prepare_natural_merge_v2 import TRAIN_V2
from verify_natural_merge import verify, final_predictions, paired


def summarize_variants(results,plan):
    if set(results)!={"A","B","C"}:
        raise ValueError("complete A/B/C development results required")
    if plan["experiment"]!="E1" or plan["partition"]!="development":
        raise ValueError("v2 method comparison accepts only E1 development")
    reference=results["A"]
    summary=dict(schema="natural_merge_v2_comparison/1",status="complete",variants={},
        interpretation="one already-observed development stream; no independent generalization claim",
        formal_evaluation_started=False)
    for variant,result in results.items():
        verify(result,plan)
        if (result.get("variant")!=variant or result.get("architecture")!=TRAIN_V2["variants"][variant]["architecture"]
                or result["diagnostic_only"]):
            raise ValueError("variant/checkpoint architecture mismatch")
        for key in ("protocol","plan_content_sha256","model","sources"):
            if result[key]!=reference[key]: raise ValueError("A/B/C model/plan/code/protocol differs")
        for arm in ("Full","Selection"):
            if final_predictions(result,arm)!=final_predictions(reference,arm):
                raise ValueError("A/B/C precompression Full/Selection controls differ")
        for a,b in zip(result["episodes"],reference["episodes"]):
            if a["carrier_indices"]!=b["carrier_indices"] or a["source_memory_sha256"]!=b["source_memory_sha256"]:
                raise ValueError("A/B/C carrier/source memory differs")
        full={p["id"]:p for p in final_predictions(result,"Full")}
        selection={p["id"]:p for p in final_predictions(result,"Selection")}
        shared={p["id"]:p for p in final_predictions(result,"SharedMerge")}
        lost=sorted(q for q in full if full[q]["scores"]["author_containment"]
                    and not selection[q]["scores"]["author_containment"])
        recovered=[q for q in lost if shared[q]["scores"]["author_containment"]]
        evidence=result["summary"]
        gate=(evidence["reader_suitability"]=="minimum_multisource_readability"
              and evidence["shared_vs_selection"]["net"]>0
              and evidence["unselected_source_full_vs_masked"]["net"]>0)
        summary["variants"][variant]=dict(architecture=result["architecture"],
            checkpoint_sha256=result["checkpoint_sha256"],arms=evidence["arms"],
            shared_vs_selection=evidence["shared_vs_selection"],
            noncarrier_full_vs_masked=evidence["unselected_source_full_vs_masked"],
            full_correct_selection_wrong=lost,recovered_compression_losses=recovered,
            recovered_count=len(recovered),development_gate_passed=gate)
    summary["B_vs_A"]=paired(final_predictions(results["B"],"SharedMerge"),final_predictions(results["A"],"SharedMerge"))
    summary["C_vs_B"]=paired(final_predictions(results["C"],"SharedMerge"),final_predictions(results["B"],"SharedMerge"))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",required=True)
    for variant in ("a","b","c"): p.add_argument("--"+variant,required=True)
    p.add_argument("--out",required=True)
    args=p.parse_args(); plan=json.loads(Path(args.plan).read_text())
    paths={v:getattr(args,v.lower()) for v in ("A","B","C")}
    results={v:json.loads(Path(path).read_text()) for v,path in paths.items()}
    if any(r["plan_sha256"]!=sha(args.plan) for r in results.values()):
        raise ValueError("development plan file differs")
    summary=summarize_variants(results,plan)
    summary["result_files"]={v:dict(path=path,sha256=sha(path)) for v,path in paths.items()}
    write_new(args.out,summary); print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
