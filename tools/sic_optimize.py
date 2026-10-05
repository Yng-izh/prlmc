"""SIC optimization: frozen discovery/development/confirmation partitions."""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def fact_items(E, lines, start=30):
    groups = {}
    for i, text in enumerate(lines):
        match = E.first_match(text)
        if match:
            groups.setdefault((match[0], match[1]), []).append((i, match))
    result = []
    for key, items in groups.items():
        i, (_, subject, answer, template) = items[-1]
        if start+30 <= i < start+410 and 1 <= len(answer.split()) <= 5:
            result.append(dict(key=repr(key), line=i, block=(i-start-30)//20,
                               relation=key[0], question=template % subject,
                               answer=answer, stratum="current" if len(items)>1 else "singleton"))
    return result


def balanced(items, n, rng):
    """Round-robin relation types, independent of block location or scores."""
    buckets = {}
    for item in items:
        buckets.setdefault(item["relation"], []).append(item)
    for values in buckets.values():
        rng.shuffle(values)
    order = sorted(buckets)
    rng.shuffle(order)
    result = []
    while len(result) < n and any(buckets.values()):
        for block in order:
            if buckets[block] and len(result) < n:
                result.append(buckets[block].pop())
    return result


def prepare(E, args):
    old = json.loads(Path(args.previous).read_text())
    reserved = {x["key"] for p in old["plans"] for split in ("discovery", "heldout")
                for x in p[split]}
    plans, used = [], set(reserved)
    for previous in old["plans"]:
        row = previous["row"]
        lines = E.read_fact_lines(row)
        rng = random.Random(args.seed + row)
        items = [x for x in fact_items(E, lines, 30) if x["key"] not in used]
        confirmation = balanced(items, min(args.test_questions,len(items)), rng)
        if len(confirmation) < 5:
            raise ValueError("row %d: only %d unused confirmation keys" % (row,len(confirmation)))
        used.update(x["key"] for x in confirmation)
        plans.append(dict(row=row,start=30,context_sha256=previous["context_sha256"],
                          discovery=list(previous["discovery"]),legacy_discovery=previous["discovery"],
                          development=previous["heldout"],confirmation=confirmation))
    for p in plans:
        rng = random.Random(args.seed + p["row"] + 100)
        remaining = [x for x in fact_items(E,E.read_fact_lines(p["row"]),30) if x["key"] not in used]
        extra = balanced(remaining,max(0,args.probes-len(p["discovery"])),rng)
        p["discovery"].extend(extra)
        used.update(x["key"] for x in extra)
    all_items = [x for p in plans for split in ("discovery","development","confirmation") for x in p[split]]
    if len(all_items) != len({x["key"] for x in all_items}) or len(all_items) != len({x["question"] for x in all_items}):
        raise ValueError("partition overlap or duplicate question")
    if len({p["context_sha256"] for p in plans}) != 4:
        raise ValueError("expected four distinct contexts")
    plan = dict(seed=args.seed, previous_reserved_confirmation_keys=len(reserved), plans=plans,
                protocol=dict(window_blocks=19,target_blocks=9,start_line=30,
                              scoring="teacher_forced_answer_token_exact_match",
                              candidates=["raw","normalized","coverage","taylor","taylor_coverage"],
                              sampling="relation-balanced, no support-position or selected-block filter",
                              selection="development eligible lost, then all correct, then mean CE, then name",
                              development_is_previous_g2c_heldout=True,
                              confirmation_is_fresh_questions_not_new_contexts=True,
                              initial_improvement="confirmation eligible lost reduction >= 10%, positive total correct delta",
                              strong_gate="eligible lost reduction >= 30% and exact paired p < .05",
                              confirmation_opened_after_choice=True,b2_generation=False))
    save(args.plan, plan)
    print("PLAN", [(p["row"],len(p["discovery"]),len(p["development"]),len(p["confirmation"])) for p in plans], flush=True)


def stats(records, name, eligible=True):
    xs = [r for r in records if r["eligible"] or not eligible]
    b = sum(not r["arms"]["select"] and r["arms"][name] for r in xs)
    c = sum(r["arms"]["select"] and not r["arms"][name] for r in xs)
    n = b+c
    p = min(1.,2*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n) if n else 1.
    base = sum(not r["arms"]["select"] for r in xs)
    wrong = sum(not r["arms"][name] for r in xs)
    return dict(n=len(xs),select_wrong=base,wrong=wrong,rescued=b,harmed=c,
                net=b-c,reduction=1-wrong/base if base else None,exact_mcnemar_p=p)


def reduction_gate(base_wrong, candidate_wrong, percent):
    return base_wrong > 0 and 100*(base_wrong-candidate_wrong) >= percent*base_wrong


def validate_development_choice(choice, development):
    if choice.get("source_sha256") != development.get("source_sha256"):
        raise ValueError("choice/development source hashes differ")
    records, windows = development.get("records",[]), development.get("windows",[])
    if development.get("stage")!="develop" or len(records)!=200 or len(windows)!=4:
        raise ValueError("confirmation requires a complete development report")
    names = list(development["eligible"])
    for name in names:
        if development["eligible"][name]!=stats(records,name) or development["all"][name]!=stats(records,name,False):
            raise ValueError("development statistics do not match records")
    best = min((name for name in names if name!="select"),
               key=lambda name:(development["eligible"][name]["wrong"],
                                development["all"][name]["wrong"],
                                sum(w["gold_loss"][name] for w in windows),name))
    if choice["selected"]!=best or development["selected"]!=best:
        raise ValueError("frozen selection differs from development ranking")


def collect(manager, probes, E, S, torch):
    from methods.prlmc.sic import block_taylor_salience
    window_slots = E.WINDOW_BLOCKS * manager.block
    begin = manager.total-manager.used
    source = manager._memory[:,begin:begin+window_slots]
    pool = manager._memory.clone().requires_grad_(True)
    rows, taylor, losses = [], [], []
    started = time.monotonic()
    for i, probe in enumerate(probes):
        scalar = S.probe_loss_scalar(manager,probe,pool,None,256)
        if scalar is None:
            raise ValueError("invalid discovery probe")
        grad = torch.autograd.grad(scalar,pool)[0]
        window_grad = grad[:,begin:begin+window_slots]
        rows.append(torch.stack([S.block_salience(window_grad[:,j*manager.block:(j+1)*manager.block],manager.block)[0]
                                 for j in range(E.WINDOW_BLOCKS)]).cpu())
        taylor.append(block_taylor_salience(window_grad,source,manager.block).cpu())
        losses.append(float(scalar.detach()))
        del grad,window_grad,scalar
        torch.cuda.empty_cache()
        print("gradient",i+1,"/",len(probes),"seconds",round(time.monotonic()-started,1),flush=True)
    del pool
    torch.cuda.empty_cache()
    return torch.stack(rows),torch.stack(taylor),losses


def candidates(energy, taylor, native, torch):
    from methods.prlmc.sic import normalize_probe_scores, coverage_select, salience_select
    normalized = normalize_probe_scores(energy)
    direction = normalize_probe_scores(taylor)
    return dict(select=native,raw=salience_select(energy.sum(0),9),
                normalized=salience_select(normalized.sum(0),9),
                coverage=coverage_select(energy,9),
                taylor=salience_select(direction.sum(0),9),
                taylor_coverage=coverage_select(taylor,9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage",choices=["prepare","develop","confirm"],required=True)
    ap.add_argument("--plan",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--choice")
    ap.add_argument("--development-report")
    ap.add_argument("--frozen-kept",action="store_true")
    ap.add_argument("--previous",default="runs/g2c_report.json")
    ap.add_argument("--seed",type=int,default=20261009)
    ap.add_argument("--test-questions",type=int,default=25)
    ap.add_argument("--probes",type=int,default=38)
    args = ap.parse_args()
    if args.frozen_kept and args.stage != "confirm":
        ap.error("--frozen-kept is only valid for confirmation")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES","9")
    import torch
    import e6a_overfit as E
    if args.stage == "prepare":
        prepare(E,args)
        return
    import sic_probe as S
    import yaml
    from transformers import AutoTokenizer
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from methods.prlmc.probes import Probe
    from methods.prlmc.consolidate import _select_block_indices
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory
    plan_bytes = Path(args.plan).read_bytes()
    plan = json.loads(plan_bytes)
    chosen, development = None, None
    if args.stage == "confirm":
        if not args.choice or not args.development_report:
            raise ValueError("confirmation requires --choice and --development-report")
        choice = json.loads(Path(args.choice).read_text())
        development = json.loads(Path(args.development_report).read_text())
        digest = hashlib.sha256(plan_bytes).hexdigest()
        if choice["plan_sha256"] != digest or development["plan_sha256"] != digest:
            raise ValueError("choice/development plan hash differs")
        validate_development_choice(choice,development)
        chosen = choice["selected"]
        allowed = development["eligible"] if args.frozen_kept else plan["protocol"]["candidates"]
        if chosen not in allowed or development["selected"] != chosen:
            raise ValueError("confirmation selection differs from frozen development")
        if args.frozen_kept and "perrow_kept" not in choice:
            raise ValueError("frozen confirmation requires per-row kept sets")
    random.seed(plan["seed"])
    torch.manual_seed(plan["seed"])
    torch.cuda.manual_seed_all(plan["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    cfg = PRLMCConfig.from_dict(dict(yaml.safe_load(Path(E.CONFIG).read_text())["prlmc"],
                                    device="cuda:0",active_memory_mask=True))
    model = MemoryLLM.from_pretrained(E.MODEL,torch_dtype=torch.float16,
                                      attn_implementation="sdpa").to("cuda:0")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):
            module.p = 0.
        if hasattr(module,"attention_dropout"):
            module.attention_dropout = 0.
    manager = ProtectedResidualMemory(model,AutoTokenizer.from_pretrained(E.MODEL),cfg)
    def probes(items):
        result = []
        for x in items:
            stem = E.query_stem(x["question"])
            tokens = manager.tokenizer(stem+" "+x["answer"],add_special_tokens=False).input_ids
            if len(tokens) > 256:
                raise ValueError("answer tail would be truncated")
            result.append(Probe(question=x["question"],answers=[x["answer"]],prompt=stem))
        return result
    report = dict(stage=args.stage,plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
                  selected=chosen,frozen_kept_confirmation=args.frozen_kept,
                  records=[],windows=[],protocol=plan["protocol"],
                  runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,
                               gpu=torch.cuda.get_device_name(0),deterministic_warn_only=True,
                               cross_process_reproduction_tested=False))
    save(args.out,report)
    for p in plan["plans"]:
        row = p["row"]
        lines = E.read_fact_lines(row)
        if hashlib.sha256("\n".join(lines).encode()).hexdigest() != p["context_sha256"]:
            raise ValueError("context changed after plan freeze")
        model.eval()
        start = p["start"]
        covered = E.build_window(manager,lines,19*manager.block,start)
        if covered != [(start+30+20*j,start+50+20*j,256*j,256) for j in range(19)]:
            raise RuntimeError("unexpected written block layout")
        begin = manager.total-manager.used
        source = manager._memory[:,begin:begin+19*manager.block]
        native = _select_block_indices(source,9*manager.block,manager.block,
                                       manager._window_importance(begin,19*manager.block),
                                       manager.goal_direction,manager.select_weights).tolist()
        del source
        model.train()
        energy,taylor,losses = collect(manager,probes(p["discovery"]),E,S,torch)
        kept = candidates(energy,taylor,native,torch)
        if chosen:
            if args.frozen_kept:
                frozen = choice["perrow_kept"][str(row)]
                if len(frozen)!=9 or frozen!=sorted(set(frozen)) or any(type(j) is not int or not 0<=j<19 for j in frozen):
                    raise ValueError("invalid frozen kept set")
                if chosen in kept and kept[chosen] != frozen:
                    raise RuntimeError("recomputed candidate differs from frozen choice")
                kept[chosen] = frozen
            kept = {name:kept[name] for name in dict.fromkeys(["select","raw",chosen])}
            previous_window = next(w for w in development["windows"] if w["row"] == row)
            for name, blocks in kept.items():
                if blocks != previous_window["kept"][name]:
                    raise RuntimeError("cross-process kept set differs for row %d arm %s" % (row,name))
        model.eval()
        evaluation = p["development"] if args.stage == "develop" else p["confirmation"]
        qs = probes(evaluation)
        full = E.score_uncompacted(manager,qs,256)["correct"]
        no_mem = E.score_no_memory(manager,qs,256)["correct"]
        outcomes,ce,cache = {},{},{}
        for name,blocks in kept.items():
            key = tuple(blocks)
            if key not in cache:
                with torch.no_grad():
                    pool,mask = S.pool_from_blocks(manager,blocks)
                    first = E.score_probes(manager,qs,pool,256,mask)
                    second = E.score_probes(manager,qs,pool,256,mask)
                    if first["correct"] != second["correct"]:
                        raise RuntimeError("scoring changed on same pool")
                del pool,mask
                torch.cuda.empty_cache()
                cache[key] = first
            outcomes[name] = cache[key]["correct"]
            ce[name] = cache[key]["gold_loss"]
            print("row",row,name,"correct",sum(outcomes[name].values()),"loss",ce[name],"kept",blocks,flush=True)
        for x in evaluation:
            q = x["question"]
            report["records"].append(dict(x,row=row,full=full[q],no_memory=no_mem[q],
                                           eligible=full[q] and not no_mem[q],
                                           arms={name:answer[q] for name,answer in outcomes.items()}))
        report["windows"].append(dict(row=row,kept=kept,energy=energy.tolist(),
                                       taylor=taylor.tolist(),discovery_losses=losses,
                                       gold_loss=ce,repeat_equal=True))
        save(args.out,report)
    if not report["windows"]:
        raise ValueError("plan produced no windows")
    report["cross_process_kept_equal"] = bool(chosen)
    names = list(report["windows"][0]["kept"])
    report["eligible"] = {name:stats(report["records"],name) for name in names}
    report["all"] = {name:stats(report["records"],name,False) for name in names}
    report["per_context"] = {str(row):{name:stats([r for r in report["records"] if r["row"]==row],name)
                                       for name in names} for row in sorted({r["row"] for r in report["records"]})}
    if args.stage == "develop":
        choices = [name for name in names if name != "select"]
        best = min(choices,key=lambda name:(report["eligible"][name]["wrong"],
                                           report["all"][name]["wrong"],
                                           sum(w["gold_loss"][name] for w in report["windows"]),name))
        report["selected"] = best
        report["development_improved"] = report["eligible"][best]["net"]>0
        save(args.choice or str(Path(args.out).with_suffix(".choice.json")),
             dict(plan_sha256=report["plan_sha256"],selected=best,
                  development_improved=report["development_improved"],
                  eligible=report["eligible"],all=report["all"]))
    else:
        s = report["eligible"][chosen]
        report["initial_improvement"] = bool(reduction_gate(s["select_wrong"],s["wrong"],10)
                                               and report["all"][chosen]["net"]>0)
        report["strong_gate"] = bool(reduction_gate(s["select_wrong"],s["wrong"],30)
                                    and s["exact_mcnemar_p"]<.05)
    save(args.out,report)
    print("FINAL",json.dumps({k:report[k] for k in ("selected","eligible","all")}),flush=True)


if __name__ == "__main__":
    main()
