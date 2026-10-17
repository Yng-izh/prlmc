"""Entity-disjoint controlled stress data, not natural-context generalization."""
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from sic_data_inventory import templates

SEED = 20261012


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(",", ":")).encode()).hexdigest()


def extract(context, specs):
    result = []
    for line in context.splitlines():
        numbered = re.match(r"^(\d+)\.\s+(.*)$", line)
        if not numbered:
            continue
        text = numbered.group(2).strip()
        for relation, (pattern, question, groups) in enumerate(specs):
            match = re.match(pattern, text.rstrip("."))
            if match:
                subject = tuple(match.group(int(g)) for g in groups.split(","))
                if len(subject) != 1:
                    raise ValueError("controlled protocol requires a single subject")
                result.append(dict(relation=relation, subject=subject[0],
                                   answer=match.groups()[-1], question=question % subject,
                                   source_serial=int(numbered.group(1)), source_text=text,
                                   answer_span=list(match.span(len(match.groups())))))
                break
    return result


def balanced(items, count, rng):
    buckets = defaultdict(list)
    for item in items:
        buckets[item["relation"]].append(item)
    order = sorted(buckets)
    rng.shuffle(order)
    for values in buckets.values():
        rng.shuffle(values)
    chosen = []
    while len(chosen) < count:
        progressed = False
        for relation in order:
            if buckets[relation] and len(chosen) < count:
                chosen.append(buckets[relation].pop())
                progressed = True
        if not progressed:
            raise ValueError("insufficient items for balanced partition")
    return chosen


def generate(contexts, specs, episodes=24, dev=8, seed=SEED, source_hashes=None):
    if episodes < 1 or not 0 <= dev <= episodes or len(contexts) < 4:
        raise ValueError("invalid counts or missing source rows 0..3")
    rng = random.Random(seed)
    excluded = set()
    for context in contexts[:4]:
        numbered = [line for line in context.splitlines() if re.match(r"^\d+\.\s", line)]
        excluded.update(f["subject"] for f in extract("\n".join(numbered[:440]), specs))
    source = extract(contexts[3], specs)
    banks = defaultdict(set)
    by_subject = defaultdict(lambda: defaultdict(list))
    for fact in source:
        if not 1 <= len(fact["answer"].split()) <= 5:
            continue
        banks[fact["relation"]].add(fact["answer"])
        if fact["subject"] not in excluded:
            by_subject[fact["subject"]][fact["relation"]].append(fact)
    banks = {r: sorted(values) for r, values in banks.items()}
    candidates = {}
    for subject, relations in sorted(by_subject.items()):
        usable = [r for r in sorted(relations) if len(banks[r]) >= 3]
        if usable:
            candidates[subject] = relations[rng.choice(usable)]
    capacity = dict(source_template_facts=len(source),
                    source_unique_subjects=len({f["subject"] for f in source}),
                    excluded_subjects=len(excluded), available_subjects=len(candidates),
                    required_subjects=episodes*158, maximum_episodes=len(candidates)//158,
                    answer_bank_sizes={str(r): len(v) for r, v in sorted(banks.items())},
                    source_unique_fact_keys=len({(f["relation"], f["subject"]) for f in source}))
    result = dict(experiment="SIC-controlled-stress", seed=seed,
                  context_type="constructed_entity_disjoint_controlled_stress_not_natural_contexts",
                  status="insufficient_data", requested_episodes=episodes, requested_development=dev,
                  capacity=capacity, source_hashes=source_hashes or {},
                  answer_banks={str(r): values for r,values in sorted(banks.items())},
                  source_context_hashes=[hashlib.sha256(c.encode()).hexdigest() for c in contexts[:4]],
                  protocol=dict(anchor_lines=30, written_lines=380, block_lines=20,
                                probe_keys=128, splits={"discovery":32,"calibration":32,"evaluation":64},
                                max_versions=3, answer_words=[1,5],
                                source_serial_preserved_in_metadata=True,
                                episode_serials="monotonic after random version ordering; latest is largest serial",
                                support_position_used_for_selection=False, test_scored=False), episodes=[])
    if len(candidates) < episodes*158:
        result["feasible_adjustments"] = dict(maximum_episodes_at_fixed_shape=len(candidates)//158,
            additional_unique_subjects_required=episodes*158-len(candidates),
            action="provide more disjoint entities or explicitly revise counts; no automatic reduction")
        return result
    subjects = sorted(candidates)
    rng.shuffle(subjects)
    labels = ["development"]*dev + ["test"]*(episodes-dev)
    rng.shuffle(labels)
    for index in range(episodes):
        allocated = subjects[index*158:(index+1)*158]
        probe_subjects, anchor_subjects = allocated[:128], allocated[128:]
        keys = [rng.choice(candidates[s]) for s in probe_subjects]
        # Relation cycling precedes any line/block metadata construction.
        ordered = balanced(keys,128,rng)
        split_keys = {"discovery":ordered[:32],"calibration":ordered[32:64],"evaluation":ordered[64:]}
        two_versions = set(rng.sample(probe_subjects,4))
        versions = []
        for key in keys:
            count = 2 if key["subject"] in two_versions else 3
            for answer in rng.sample(banks[key["relation"]],count):
                versions.append(dict(rng.choice(candidates[key["subject"]]),answer=answer))
        rng.shuffle(versions)
        records = [dict(rng.choice(candidates[s])) for s in anchor_subjects] + versions
        lines, latest = [], {}
        for line, fact in enumerate(records):
            start,end = fact["answer_span"]
            text = fact["source_text"][:start]+fact["answer"]+fact["source_text"][end:]
            fact.update(serial=line+1,line=line,block=None if line<30 else (line-30)//20)
            lines.append("%d. %s" % (line+1,text))
            latest[(fact["relation"],fact["subject"])] = fact
        splits = {}
        for name,partition in split_keys.items():
            splits[name] = []
            for key in partition:
                fact = latest[(key["relation"],key["subject"])]
                splits[name].append(dict(key=[key["relation"],key["subject"]],relation=key["relation"],
                    subject=key["subject"],question=key["question"],answer=fact["answer"],
                    line=fact["line"],block=fact["block"],serial=fact["serial"],source_serial=fact["source_serial"]))
        episode = dict(episode=index,partition=labels[index],lines=lines,facts=records,splits=splits,
                       counts=dict(unique_probe_keys=128,unique_anchor_entities=30,
                                   anchor_lines=30,written_lines=380,total_lines=410),
                       context_sha256=hashlib.sha256("\n".join(lines).encode()).hexdigest())
        episode["episode_sha256"] = digest(episode)
        result["episodes"].append(episode)
    result["status"] = "completed"
    result["actual_counts"] = dict(episodes=episodes,development=dev,test=episodes-dev,
        probe_keys=episodes*128,anchor_entities=episodes*30,entity_count=episodes*158,lines=episodes*410)
    validate(result,excluded)
    result["plan_sha256"] = digest(result)
    return result


def validate(plan, excluded=()):
    if plan["status"] != "completed":
        raise ValueError("validation requires completed data")
    if "plan_sha256" in plan:
        copy = dict(plan)
        copy.pop("plan_sha256")
        if digest(copy)!=plan["plan_sha256"]:
            raise ValueError("plan hash mismatch")
    ids = [episode["episode"] for episode in plan["episodes"]]
    if ids!=list(range(plan["requested_episodes"])):
        raise ValueError("episode identity mismatch")
    entities,global_questions = set(),set()
    for episode in plan["episodes"]:
        if episode["partition"] not in ("development","test"):
            raise ValueError("unknown partition")
        if len(episode["lines"])!=410 or len(episode["facts"])!=410:
            raise ValueError("exactly 410 lines required")
        if hashlib.sha256("\n".join(episode["lines"]).encode()).hexdigest()!=episode["context_sha256"]:
            raise ValueError("context hash mismatch")
        copy = dict(episode)
        copy.pop("episode_sha256")
        if digest(copy)!=episode["episode_sha256"]:
            raise ValueError("episode hash mismatch")
        anchor = {f["subject"] for f in episode["facts"][:30]}
        counts = Counter((f["relation"],f["subject"]) for f in episode["facts"][30:])
        probe = {subject for _,subject in counts}
        if len(anchor)!=30 or len(probe)!=128 or len(counts)!=128 or anchor&probe:
            raise ValueError("anchor/probe entity collision")
        if sorted(counts.values())!=[2]*4+[3]*124:
            raise ValueError("version budget differs from 380")
        if (anchor|probe)&(entities|set(excluded)):
            raise ValueError("historical or cross-episode entity leakage")
        entities.update(anchor|probe)
        seen,latest,values = set(),{},defaultdict(set)
        for line,fact in enumerate(episode["facts"]):
            if fact["serial"]!=line+1 or fact["line"]!=line:
                raise ValueError("serial/line mismatch")
            expected_block = None if line<30 else (line-30)//20
            start,end = fact["answer_span"]
            expected_line = "%d. %s" % (line+1,fact["source_text"][:start]+fact["answer"]+fact["source_text"][end:])
            if fact["block"]!=expected_block or episode["lines"][line]!=expected_line:
                raise ValueError("rendered line/block mismatch")
            if not 1<=len(fact["answer"].split())<=5:
                raise ValueError("answer length outside 1..5 words")
            bank = plan["answer_banks"].get(str(fact["relation"]),[])
            if len(set(bank))<3 or fact["answer"] not in bank:
                raise ValueError("answer not in eligible source relation bank")
            key = (fact["relation"],fact["subject"])
            latest[key] = fact
            if line>=30:
                values[key].add(fact["answer"])
        if any(len(values[key])!=count for key,count in counts.items()):
            raise ValueError("versions must have distinct answers")
        for name,size in (("discovery",32),("calibration",32),("evaluation",64)):
            partition = episode["splits"][name]
            if len(partition)!=size:
                raise ValueError("split size mismatch")
            for question in partition:
                key = tuple(question["key"])
                if key not in counts or question["subject"] in seen or question["question"] in global_questions:
                    raise ValueError("fact/entity/question split overlap")
                gold = latest[key]
                fields = ("answer","line","block","serial","source_serial","relation","subject","question")
                if any(question[field]!=gold[field] for field in fields):
                    raise ValueError("gold differs from latest version")
                seen.add(question["subject"])
                global_questions.add(question["question"])
        if seen!=probe:
            raise ValueError("splits do not cover probe entities")
    if len(plan["episodes"])!=plan["requested_episodes"]:
        raise ValueError("episode count changed")
    if sum(e["partition"]=="development" for e in plan["episodes"])!=plan["requested_development"]:
        raise ValueError("development count changed")


def check_token_budget(plan, encode, max_tokens=512):
    if plan["status"]!="completed":
        return
    overflows = []
    for episode in plan["episodes"]:
        lines = episode["lines"]
        chunks = [lines[:30]]+[lines[start:start+20] for start in range(30,410,20)]
        lengths = [len(encode("\n".join(chunk))) for chunk in chunks]
        episode["token_lengths"] = dict(anchor=lengths[0],blocks=lengths[1:],max_tokens=max_tokens)
        overflows.extend(dict(episode=episode["episode"],chunk=i,tokens=n)
                         for i,n in enumerate(lengths) if n>max_tokens)
        episode.pop("episode_sha256")
        episode["episode_sha256"] = digest(episode)
    plan.pop("plan_sha256",None)
    plan["token_budget_verified"] = not overflows
    if overflows:
        plan["status"] = "token_budget_exceeded"
        plan["overflows"] = overflows
        plan["feasible_adjustments"] = {"action":"explicitly revise token budget or fact construction; no truncation performed"}
    plan["plan_sha256"] = digest(plan)


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def tokenizer_manifest(directory):
    names = ("tokenizer.json","tokenizer_config.json","special_tokens_map.json",
             "added_tokens.json","tokenizer.model","vocab.json","merges.txt","config.json")
    files = {name:file_hash(Path(directory)/name) for name in names if (Path(directory)/name).is_file()}
    if not files or not any(name in files for name in ("tokenizer.json","tokenizer.model","vocab.json")):
        raise ValueError("missing tokenizer vocabulary artifact")
    return dict(files=files,sha256=digest(files))


def token_audit(plan, plan_file_sha256, manifest):
    records = [dict(episode=e["episode"],partition=e["partition"],
                    context_sha256=e["context_sha256"],**e["token_lengths"])
               for e in plan["episodes"] if "token_lengths" in e]
    return dict(experiment="SIC-controlled-token-audit",status=plan["status"],
                plan_file_sha256=plan_file_sha256,plan_content_sha256=plan.get("plan_sha256"),
                tokenizer=manifest,max_tokens=512,add_special_tokens=False,
                checked_serial_numbered_lines=True,truncation_performed=False,
                model_loaded=False,questions_scored=False,test_scored=False,
                verified=plan.get("token_budget_verified",False),episodes=records,
                overflows=plan.get("overflows",[]))


def save_json(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--templates",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--episodes",type=int,default=24)
    parser.add_argument("--dev",type=int,default=8)
    parser.add_argument("--seed",type=int,default=SEED)
    parser.add_argument("--tokenizer",default="/share/latent/MemoryLLM-8B")
    parser.add_argument("--token-audit",help="defaults to sic_paper_token_audit.json beside output")
    args = parser.parse_args()
    import pandas as pd
    frame = pd.read_parquet(args.dataset,columns=["context"])
    plan = generate(frame["context"].tolist(),templates(args.templates),args.episodes,args.dev,args.seed,
        dict(dataset=file_hash(args.dataset),templates=file_hash(args.templates),generator=file_hash(__file__)))
    manifest = None
    if plan["status"]=="completed":
        from transformers import AutoTokenizer
        manifest = tokenizer_manifest(args.tokenizer)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
        plan["tokenizer"] = args.tokenizer
        plan["tokenizer_sha256"] = manifest["sha256"]
        check_token_budget(plan,lambda text:tokenizer(text,add_special_tokens=False).input_ids)
        if plan["status"]=="completed":
            validate(plan)
    path = Path(args.out)
    audit_path = Path(args.token_audit) if args.token_audit else path.with_name("sic_paper_token_audit.json")
    if path.resolve()==audit_path.resolve():
        raise ValueError("plan and token audit paths must differ")
    save_json(path,plan)
    save_json(audit_path,token_audit(plan,file_hash(path),manifest))
    print(json.dumps({name:plan[name] for name in ("status","capacity","actual_counts","feasible_adjustments","overflows") if name in plan}))
    if plan["status"]!="completed":
        raise SystemExit(2)


if __name__=="__main__":
    main()
