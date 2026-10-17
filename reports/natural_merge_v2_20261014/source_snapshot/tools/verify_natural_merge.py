#!/usr/bin/env python3
"""Re-score every original QA, verify matched carriers/budgets and paired curves."""
import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_public_retention import digest, sha, write_new
from prepare_natural_merge import validate_plan
from natural_merge_runtime import scores


def paired(left, right):
    a, b = {p["id"]: p for p in left}, {p["id"]: p for p in right}
    if len(a)!=len(left) or len(b)!=len(right) or a.keys()!=b.keys():
        raise ValueError("paired original QA IDs differ or repeat")
    gains = [q for q in a if a[q]["scores"]["author_containment"] and not b[q]["scores"]["author_containment"]]
    harms = [q for q in a if b[q]["scores"]["author_containment"] and not a[q]["scores"]["author_containment"]]
    clusters = {}
    for q in a:
        if a[q]["context_sha256"] != b[q]["context_sha256"]:
            raise ValueError("paired QA context identity mismatch")
        clusters.setdefault(a[q]["context_sha256"], []).append(
            int(a[q]["scores"]["author_containment"])-int(b[q]["scores"]["author_containment"]))
    if not clusters:
        return dict(n=0, gained=[], harmed=[], delta=None, ci95=None, n_contexts=0)
    values = list(clusters.values()); rng=random.Random(0); boot=[]
    for _ in range(2000):
        drawn=[values[rng.randrange(len(values))] for _ in values]
        boot.append(sum(sum(v) for v in drawn)/sum(len(v) for v in drawn))
    boot.sort()
    return dict(n=len(a), gained=gains, harmed=harms, net=len(gains)-len(harms),
        delta=(len(gains)-len(harms))/len(a), n_contexts=len(values), ci95=[boot[49],boot[1949]],
        uncertainty="context-cluster bootstrap, 2000 resamples, seed0; conditional on fixed distractor stream")


def accuracy(predictions):
    n=len(predictions)
    return dict(n=n, original_contexts=len({p["context_sha256"] for p in predictions}),
        correct=sum(p["scores"]["author_containment"] for p in predictions),
        author_containment=sum(p["scores"]["author_containment"] for p in predictions)/n if n else None,
        normalized_em=sum(p["scores"]["normalized_em"] for p in predictions)/n if n else None,
        normalized_f1=sum(p["scores"]["normalized_f1"] for p in predictions)/n if n else None,
        source_contexts_answered=len({p["context_sha256"] for p in predictions if p["scores"]["author_containment"]}))


def final_predictions(report, arm):
    result=[]
    for episode in report["episodes"]:
        row=episode["arms"][arm]
        if row["status"]!="complete":
            continue
        result.extend(row["predictions"] if report["experiment"]=="E1" else row["trace"][-1]["predictions"])
    return result


def summarize(report):
    if not report["episodes"]:
        raise ValueError("no completed natural episode")
    arms=list(report["episodes"][0]["arms"])
    full=final_predictions(report,"Full")
    ours=final_predictions(report,"SharedMerge"); selection=final_predictions(report,"Selection")
    comparison=paired(ours,selection)
    readable_reference = any(p["scores"]["author_containment"] for p in full)
    if report["experiment"]!="E1":
        # Full is unavailable after physical exhaustion, but its earlier intact
        # readouts still diagnose reader suitability. Never invent final Full.
        readable_reference = any(p["scores"]["author_containment"]
            for e in report["episodes"] for a in ("Full","NativeDefault")
            for t in e["arms"][a]["trace"] for p in t["predictions"])
    result=dict(arms={a:accuracy(final_predictions(report,a)) for a in arms},
        shared_vs_selection=comparison,
        full_capacity_exceeded=[e["id"] for e in report["episodes"] if e["arms"]["Full"]["status"]!="complete"],
        reader_reference="some_original_QAs_readable" if readable_reference else "unreadable_workload",
        conclusion="paired frozen-module evidence on original contexts; interpretation requires actual compression and reader validity")
    full_map={p["id"]:p for p in full}
    result["full_correct_retention_diagnostic"]={}
    for a in arms:
        eligible=[p for p in final_predictions(report,a) if p["id"] in full_map and full_map[p["id"]]["scores"]["author_containment"]]
        result["full_correct_retention_diagnostic"][a]=accuracy(eligible)
    if report["experiment"]=="E1":
        noncarrier_ours=[]; noncarrier_masked=[]
        for e in report["episodes"]:
            carriers=set(e["carrier_indices"])
            noncarrier_ours.extend(p for p in e["arms"]["SharedMerge"]["predictions"] if p["source_index"] not in carriers)
            noncarrier_masked.extend(p for p in e["arms"]["CarrierOnlyFusion"]["predictions"] if p["source_index"] not in carriers)
        result["unselected_source_full_vs_masked"]=paired(noncarrier_ours,noncarrier_masked)
        # Gates use all original QA. Full-correct subset remains diagnostic only.
        result["readable_source_contexts"]=accuracy(full)["source_contexts_answered"]
        result["reader_suitability"]=("weak_readability" if accuracy(full)["author_containment"]<.1 or result["readable_source_contexts"]<2
                                      else "minimum_multisource_readability")
        result["reader_suitability_rule"]="predeclared diagnostic minimum: >=10% all-QA containment and >=2 source contexts; not a performance target"
        result["fusion_behavior_observed"]=result["unselected_source_full_vs_masked"]["net"]>0
    else:
        result["retention_by_write"]={}
        result["isolated_compression_changes"]={}
        for a in arms:
            traces=[e["arms"][a]["trace"] for e in report["episodes"]]
            result["retention_by_write"][a]=[dict(write=i+1,**accuracy([p for t in traces if i<len(t) for p in t[i]["predictions"]]))
                for i in range(max(len(t) for t in traces))]
            changes=[]
            for e in report["episodes"]:
                hooks=e["arms"][a]["compressions"]
                for i in range(0,len(hooks),2):
                    before,after=hooks[i:i+2]
                    changes.append(dict(episode=e["id"],before_write=before["before_write"],
                        before=accuracy(before["predictions"]),after=accuracy(after["predictions"]),
                        paired=paired(after["predictions"],before["predictions"])))
            result["isolated_compression_changes"][a]=changes
        result["actual_shared_compressions"]=sum(len(e["arms"]["SharedMerge"]["merge_events"]) for e in report["episodes"])
        result["conclusion_allowed"]=bool(result["actual_shared_compressions"]) and result["reader_reference"]!="unreadable_workload"
    return result


def verify(report, plan, tokenizer=None):
    validate_plan(plan)
    if report.get("schema")!="natural_merge_results/1" or report["status"]!="complete":
        raise ValueError("complete natural merge result required")
    if report["protocol"]!=plan["protocol"] or report["plan_content_sha256"]!=plan["content_sha256"]:
        raise ValueError("report/plan identity mismatch")
    if report["training_performed"] or report["optimizer_created"] or report["shared_state_sha256_before"]!=report["shared_state_sha256_after"]:
        raise ValueError("frozen evaluation violated")
    indices=report["sharding"]["episode_indices"]
    shard=report["sharding"]
    if not 0<=shard["index"]<shard["count"]<=len(plan["episodes"]):
        raise ValueError("invalid report shard")
    if indices != [i for i in range(len(plan["episodes"])) if i % shard["count"]==shard["index"]]:
        raise ValueError("shard omitted or reassigned original episodes")
    if [e["episode_index"] for e in report["episodes"]]!=indices:
        raise ValueError("reported episode coverage mismatch")
    block=report["resources"]["effective_budget_slots"]//20
    if block<=0 or report["resources"]["effective_budget_slots"]!=20*block:
        raise ValueError("invalid matched budget")
    e1=plan["experiment"]=="E1"
    arms=("Full","Selection","SharedMerge","CarrierOnlyFusion") if e1 else ("Full","Selection","Recent","SharedMerge","NativeBudget","NativeDefault")
    def check_predictions(rows,qas):
        reference={q["id"]:q for q in qas}
        if len(rows)!=len(reference) or {r["id"] for r in rows}!=reference.keys():
            raise ValueError("original QA omitted or duplicated")
        for r in rows:
            q=reference[r["id"]]
            if not isinstance(r["token_ids"],list) or len(r["token_ids"])>10 or any(type(t) is not int or t<0 for t in r["token_ids"]):
                raise ValueError("generated tokens violate fixed generation limit")
            if not isinstance(r.get("auxiliary_prediction"),str):
                raise ValueError("auxiliary normalized-metric decoded prediction missing")
            if tokenizer is not None and tokenizer.decode(r["token_ids"],skip_special_tokens=True)!=r["auxiliary_prediction"]:
                raise ValueError("auxiliary prediction differs from generated tokens")
            if tokenizer is not None and tokenizer.decode(r["token_ids"],skip_special_tokens=False)!=r["prediction"]:
                raise ValueError("prediction differs from saved generated tokens")
            if (r["context_sha256"]!=q["context_sha256"] or r["source_index"]!=q["source_index"]
                or r["question"]!=q["question"] or r["original_answers"]!=q["answers"] or r["scoring_target"]!=q["scoring_target"]
                or r["scores"]!=scores(r["prediction"],q,r["auxiliary_prediction"])):
                raise ValueError("original QA/scoring changed")
    for entry in report["episodes"]:
        ep=plan["episodes"][entry["episode_index"]]
        qas=[q for c in ep["contexts"] for q in c["qas"]]
        if entry["id"]!=ep["id"] or set(entry["arms"])!=set(arms):
            raise ValueError("episode/arm set mismatch")
        if e1:
            indices=entry["carrier_indices"]
            if len(indices)!=9 or len(set(indices))!=9 or indices!=sorted(indices) or not set(indices).issubset(range(19)):
                raise ValueError("invalid fixed carriers")
            for a,row in entry["arms"].items():
                if row["status"]!="complete" or row["active_slots"]!=(19 if a=="Full" else 9)*block:
                    raise ValueError("E1 compression budget failed")
                if row["source_memory_sha256"]!=entry["source_memory_sha256"]:
                    raise ValueError("E1 arms did not share precompression memory")
                if a!="Full" and row["carrier_indices"]!=indices:
                    raise ValueError("E1 causal ablation reselected carriers")
                check_predictions(row["predictions"],qas)
            for a,allowed in (("SharedMerge",list(range(19))),("CarrierOnlyFusion",indices)):
                fusion=entry["arms"][a]["fusion"]
                if fusion["carrier_indices"]!=indices or fusion["allowed_source_indices"]!=allowed:
                    raise ValueError("causal source ablation mask mismatch")
        else:
            expected_compressions=[i for i in range(20,len(ep["writes"]),10)]
            for a,row in entry["arms"].items():
                failed=row["status"]=="capacity_exceeded"
                expected=min(len(ep["writes"]),report["resources"]["physical_slots"]//block) if a=="Full" else len(ep["writes"])
                if len(row["trace"])!=expected or (failed and (a!="Full" or expected==len(ep["writes"]))) or (not failed and row["status"]!="complete"):
                    raise ValueError("incomplete write stream or fabricated Full final reference")
                for i,r in enumerate(row["trace"]):
                    if r["index"]!=i or r["input_sha256"]!=ep["writes"][i]["token_sha256"]:
                        raise ValueError("writer input changed")
                    if a in ("Selection","Recent","SharedMerge","NativeBudget") and r["active_slots"]>20*block:
                        raise ValueError("matched capacity exceeded")
                    check_predictions(r["predictions"],qas)
                hooks=row["compressions"]
                if len(hooks)%2:
                    raise ValueError("missing isolated compression measurement")
                for j in range(0,len(hooks),2):
                    pre,post=hooks[j:j+2]
                    if pre["phase"]!="before" or post["phase"]!="after" or pre["before_write"]!=post["before_write"] or pre["active_slots"]-post["active_slots"]!=10*block:
                        raise ValueError("pre/post measurements are not the same isolated compression")
                    if pre["carrier_indices"]!=post["carrier_indices"]:
                        raise ValueError("compression carrier mismatch")
                    check_predictions(pre["predictions"],qas);check_predictions(post["predictions"],qas)
                if a in ("Selection","Recent","SharedMerge") and [h["before_write"] for h in hooks[::2]]!=expected_compressions:
                    raise ValueError("compression schedule omitted or moved")
                if a in ("Selection","Recent","SharedMerge") and len(row["merge_events"])*2!=len(hooks):
                    raise ValueError("compression trace incomplete")
            selected=[e["selected_blocks"] for e in entry["arms"]["Selection"]["merge_events"]]
            shared=[e["selected_blocks"] for e in entry["arms"]["SharedMerge"]["merge_events"]]
            # Subsequent content-dependent choices can differ after residual
            # writes. Fair Selection requires replaying OURS' carrier schedule.
            if selected!=shared:
                raise ValueError("Selection did not use the same OURS carrier schedule")
    expected=summarize(report)
    if report.get("summary")!=expected:
        raise ValueError("summary disagrees with independently rescored predictions")
    return dict(schema="natural_merge_certificate/1", status="passed", result_content_sha256=digest(report),
        plan_content_sha256=plan["content_sha256"], original_qa_count=sum(len(c["qas"]) for e in report["episodes"] for c in plan["episodes"][e["episode_index"]]["contexts"]),
        verified=["original QA and official primary rescoring", "fixed carrier schedule", "active budgets",
            "isolated compression hooks", "frozen state", "paired context bootstrap", "honest Full capacity"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result",required=True);p.add_argument("--plan",required=True);p.add_argument("--out",required=True)
    p.add_argument("--model-path",default="/dev/shm/s1d_reader_fp16")
    a=p.parse_args();r=json.loads(Path(a.result).read_text());plan=json.loads(Path(a.plan).read_text())
    from transformers import AutoTokenizer
    from prepare_public_retention import tokenizer_manifest
    if tokenizer_manifest(a.model_path)!=plan["tokenizer"]:
        raise ValueError("scoring tokenizer mismatch")
    tokenizer=AutoTokenizer.from_pretrained(a.model_path,local_files_only=True)
    if r["plan_sha256"]!=sha(a.plan):
        raise ValueError("plan file hash mismatch")
    certificate=verify(r,plan,tokenizer);certificate["result_sha256"]=sha(a.result)
    write_new(a.out,certificate);print(json.dumps(certificate,indent=2))

if __name__=="__main__":
    main()
