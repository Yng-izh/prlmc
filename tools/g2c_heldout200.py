"""Fixed 200 held-out facts; full-context gold, disjoint discovery, deterministic QA."""
import argparse
import hashlib
import json
import math
import os
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace


def samples(E, lines, row, seed, excluded):
    groups = {}
    for i, text in enumerate(lines):
        m = E.first_match(text)
        if m:
            groups.setdefault((m[0], m[1]), []).append((i, m))
    buckets = {"current": [], "singleton": []}
    for key, items in groups.items():
        i, (_, subject, answer, template) = items[-1]
        if repr(key) in excluded or not 60 <= i < 440 or not 1 <= len(answer.split()) <= 5:
            continue
        cls = "current" if len(items) > 1 else "singleton"
        buckets[cls].append(dict(line=i, key=repr(key), relation=key[0],
                                 question=template % subject, answer=answer, stratum=cls))
    rng = random.Random(seed + row)
    discovery, heldout = [], []
    for values in buckets.values():
        rng.shuffle(values)
        n = min(12, len(values) // 3)
        discovery.extend(values[:n])
        heldout.extend(values[n:n+min(25, len(values)-n)])
    used = {x["key"] for x in discovery + heldout}
    remaining = [x for values in buckets.values() for x in values if x["key"] not in used]
    rng.shuffle(remaining)
    heldout.extend(remaining[:50-len(heldout)])
    assert len(heldout) == len({x["question"] for x in heldout}) == 50
    assert not ({x["key"] for x in discovery} & {x["key"] for x in heldout})
    return discovery, heldout


def paired(records, subset):
    xs = [x for x in records if subset(x)]
    b = sum(not x["select"] and x["sic"] for x in xs)
    c = sum(x["select"] and not x["sic"] for x in xs)
    n = b+c
    p = min(1., 2*sum(math.comb(n, k) for k in range(min(b,c)+1))/2**n) if n else 1.
    sl = sum(not x["select"] for x in xs)
    si = sum(not x["sic"] for x in xs)
    return dict(n=len(xs), select_wrong=sl, sic_wrong=si, b=b, c=c,
                exact_mcnemar_p=p, reduction=1-si/sl if sl else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.dry_run:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True)
        free, index = max((int(b),int(a)) for a,b in
                          (line.split(",") for line in result.stdout.splitlines()))
        if free < 24000:
            raise RuntimeError("need at least 24000 MiB free")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
        print("GPU", index, "free MiB", free, flush=True)
    import torch
    import e6a_overfit as E
    import sic_probe as S
    from methods.prlmc.probes import Probe
    from methods.prlmc.consolidate import _select_block_indices
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory
    plans, seen, excluded = [], set(), set()
    for row in range(4):
        lines = E.read_fact_lines(row)
        digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
        assert digest not in seen
        seen.add(digest)
        disc, held = samples(E, lines, row, args.seed, excluded)
        excluded.update(x["key"] for x in disc + held)
        plans.append(dict(row=row, context_sha256=digest, discovery=disc, heldout=held))
    report = dict(design=dict(seed=args.seed, n_heldout=200, start=30,
                             scoring="teacher_forced_answer_token_exact_match",
                             eligibility="full_correct_and_no_memory_wrong",
                             primary="eligible lost reduction >= .30 and exact McNemar p < .05",
                             b2_generation=False), plans=plans, records=[], windows=[])
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    if args.dry_run:
        print([(p["row"],len(p["discovery"]),len(p["heldout"])) for p in plans])
        return
    import yaml
    from transformers import AutoTokenizer
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    torch.manual_seed(args.seed)
    cfg = PRLMCConfig.from_dict(dict(yaml.safe_load(Path(E.CONFIG).read_text())["prlmc"],
                                     device="cuda:0", active_memory_mask=True))
    model = MemoryLLM.from_pretrained(E.MODEL, torch_dtype=torch.float16,
                                      attn_implementation="sdpa").to("cuda:0")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    manager = ProtectedResidualMemory(model, AutoTokenizer.from_pretrained(E.MODEL), cfg)
    def probes(items):
        return [Probe(question=x["question"],answers=[x["answer"]],
                      prompt=E.query_stem(x["question"])) for x in items]
    for plan in plans:
        row = plan["row"]
        model.eval()
        covered = E.build_window(manager,E.read_fact_lines(row),19*manager.block,30)
        assert covered == [(60+20*j,80+20*j,256*j,256) for j in range(19)]
        start = manager.total-manager.used
        source = manager._memory[:,start:start+19*manager.block]
        selected = _select_block_indices(source,9*manager.block,manager.block,
                                          manager._window_importance(start,19*manager.block),
                                          manager.goal_direction,manager.select_weights).tolist()
        del source
        model.train()
        for module in model.modules():
            if isinstance(module,torch.nn.Dropout):
                module.p = 0.
            if hasattr(module,"attention_dropout"):
                module.attention_dropout = 0.
        scores,_ = S.harvest_gradients(manager,dict(probes=probes(plan["discovery"]),
                                                   kept_native=selected),
                                       SimpleNamespace(arm="sic",max_probe_tokens=256))
        sic = S.salience_select(scores,9)
        model.eval()
        held = probes(plan["heldout"])
        no_mem = E.score_no_memory(manager,held,256)["correct"]
        full = E.score_uncompacted(manager,held,256)["correct"]
        outcomes = {}
        for name,kept in (("select",selected),("sic",sic)):
            with torch.no_grad():
                pool,mask = S.pool_from_blocks(manager,kept)
                first = E.score_probes(manager,held,pool,256,mask)["correct"]
                second = E.score_probes(manager,held,pool,256,mask)["correct"]
                assert first == second,"non-deterministic scoring"
            del pool,mask
            torch.cuda.empty_cache()
            outcomes[name] = first
        for x in plan["heldout"]:
            q = x["question"]
            report["records"].append(dict(x,row=row,full=full[q],no_memory=no_mem[q],
                                          eligible=full[q] and not no_mem[q],
                                          select=outcomes["select"][q],sic=outcomes["sic"][q]))
        report["windows"].append(dict(row=row,select=selected,sic=sic,
                                       salience=scores.tolist(),repeat_equal=True))
        path.write_text(json.dumps(report,indent=2))
        print("row",row,paired([x for x in report["records"] if x["row"]==row],
                                lambda x:x["eligible"]),flush=True)
    report["all_questions"] = paired(report["records"],lambda x:True)
    report["eligible"] = paired(report["records"],lambda x:x["eligible"])
    report["strata"] = {cls:paired(report["records"],
                                    lambda x:x["eligible"] and x["stratum"]==cls)
                         for cls in ("current","singleton")}
    stat = report["eligible"]
    report["verdict"] = "PASS" if (stat["reduction"] is not None and
                                         stat["reduction"]>=.3 and stat["exact_mcnemar_p"]<.05) else "NOT_PASSED"
    path.write_text(json.dumps(report,indent=2))
    print("FINAL",report["eligible"],report["verdict"],flush=True)


if __name__ == "__main__":
    main()
