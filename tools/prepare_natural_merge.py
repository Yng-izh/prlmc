#!/usr/bin/env python3
"""Seal original natural-data training/development/formal plans before any run."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_public_retention import (DATA_REPO, REVISIONS, digest, sha,
                                      tokenizer_manifest, write_new, make_writes)

PROTOCOL = dict(budget_blocks=20, window_blocks=19, target_blocks=9, anchor_blocks=0,
    rank=32, gamma_max=.05, max_new_tokens=10, do_sample=False, stop_strings=[],
    primary_score="author case-sensitive answer containment",
    auxiliary_scores=["SQuAD normalized EM", "SQuAD normalized token F1"],
    carrier_policy="existing content-only selector; never QA-conditioned",
    compression_timing="before append; isolated pre/post queries before the next write",
    physical_memory_claim=False, input_mode="author512", seed=0)
TRAIN_PROTOCOL = dict(streams=8, contexts_per_stream=39, rounds=4,
    schedule="interleave full streams; two online compression updates per stream per round",
    optimizer="AdamW", learning_rate=1e-4, weight_decay=.01, gradient_clip=1.,
    objective="mean original gold-answer CE, balanced one QA per source context",
    teacher_weight=0., init="fresh identity residual", truncated_gradient_at_compression=True,
    planned_updates=64, checkpoint="last scheduled update only; no within-training selection")


def validate_plan(plan):
    identity = dict(plan)
    expected = identity.pop("content_sha256", None)
    if expected != digest(identity) or plan.get("schema") != "natural_merge_plan/1":
        raise ValueError("natural plan checksum/schema failure")
    if plan["protocol"] != PROTOCOL:
        raise ValueError("fixed merge protocol changed")
    if plan["partition"] not in ("train", "development", "formal", "transfer"):
        raise ValueError("unknown partition")
    experiment=plan["experiment"]
    expected_partition={"train":("train",),"E1":("development","formal"),"E2":("formal",),"E3":("formal","transfer")}
    if experiment not in expected_partition or plan["partition"] not in expected_partition[experiment]:
        raise ValueError("experiment partition mismatch")
    if not plan["episodes"] or len({e["id"] for e in plan["episodes"]})!=len(plan["episodes"]):
        raise ValueError("empty/duplicate original streams")
    contexts, qids = set(), set()
    for episode in plan["episodes"]:
        if plan["experiment"] in ("train", "E1"):
            wanted = 39 if plan["experiment"] == "train" else 19
            if len(episode["contexts"]) != wanted or len(episode["writes"]) != wanted:
                raise ValueError("wrong multi-source stream size")
        if experiment=="E2" and len(episode["writes"])!=31:
            raise ValueError("E2 requires the fixed 30-distractor protocol")
        if experiment=="E3" and len(episode["writes"])!=(31 if plan["partition"]=="transfer" else 61):
            raise ValueError("E3 requires fixed 30 transfer or 60 SQuAD distractors")
        if not episode["contexts"] or any(not c["qas"] for c in episode["contexts"]):
            raise ValueError("every evaluated original context needs its original QA")
        for context in episode["contexts"]:
            if digest(context["context"]) != context["context_sha256"]:
                raise ValueError("original natural context changed")
            key = context["context_sha256"]
            if key in contexts:
                raise ValueError("duplicate natural context across episodes")
            contexts.add(key)
            for qa in context["qas"]:
                if qa["id"] in qids or not qa["answers"] or not qa["prompt_ids"]:
                    raise ValueError("duplicate/empty original QA")
                qids.add(qa["id"])
                if qa["context_sha256"] != key or qa["source_index"] != episode["contexts"].index(context):
                    raise ValueError("QA source provenance mismatch")
                if digest(qa["prompt_ids"]) != qa["prompt_sha256"] or not qa["answer_ids"]:
                    raise ValueError("QA token identity failure")
        for write in episode["writes"]:
            ids = write["token_ids"]
            if not ids or len(ids) > 512 or digest(ids) != write["token_sha256"]:
                raise ValueError("invalid write token identity")
        if plan["experiment"] in ("train", "E1") and any(w["truncated_tokens"] for w in episode["writes"]):
            raise ValueError("multi-knowledge paragraphs must fit one complete native write")
    if plan["experiment"] == "train" and (len(plan["episodes"])!=8 or plan["partition"] != "train" or plan["training"] != TRAIN_PROTOCOL):
        raise ValueError("training scope/schedule changed")


def raw_squad(path):
    """Keep article identity, exact paragraphs, all answerable original QAs/spans."""
    result = []
    for article in json.loads(Path(path).read_text())["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            if not isinstance(context, str) or not context:
                raise ValueError("invalid original context")
            qas = []
            for qa in paragraph["qas"]:
                if qa.get("is_impossible", False) or not qa.get("answers"):
                    continue
                for answer in qa["answers"]:
                    start, text = answer["answer_start"], answer["text"]
                    if not text or context[start:start + len(text)] != text:
                        raise ValueError("original SQuAD answer span does not match context")
                qas.append(dict(id=str(qa["id"]), question=qa["question"],
                    answers=[a["text"] for a in qa["answers"]], original_answers=qa["answers"]))
            if qas:
                result.append(dict(article=article["title"], article_sha256=digest(article["title"]),
                    context=context, context_sha256=digest(context), qas=qas))
    # Never use answerability predictions/Full correctness to form this subset.
    return result


def split_rows(train_rows, dev_rows, observed, published_distractors):
    observed_hashes = {s["context_sha256"] for s in observed["samples"]}
    observed_articles = {r["article"] for r in train_rows + dev_rows if r["context_sha256"] in observed_hashes}
    if not observed_hashes.issubset({r["context_sha256"] for r in dev_rows}):
        raise ValueError("observed processed SQuAD contexts cannot all be matched to raw dev articles")
    # Exclude any article whose paragraph appears in the already observed author
    # training/distractor release. Exact substring matching handles grouped paragraphs.
    known_articles = {r["article"] for r in train_rows if any(r["context"] in text for text in published_distractors)}
    train_articles = {r["article"] for r in train_rows}
    dev_articles = sorted({r["article"] for r in dev_rows} - observed_articles - train_articles,
                          key=lambda x: digest(["natural-merge-split-v1", x]))
    development_articles = set(dev_articles[::2])
    formal_articles = set(dev_articles[1::2])
    order = lambda rows: sorted(rows, key=lambda r: digest(["natural-merge-order-v1", r["context_sha256"]]))
    return dict(train=order([r for r in train_rows if r["article"] not in known_articles | observed_articles]),
        development=order([r for r in dev_rows if r["article"] in development_articles]),
        formal=order([r for r in dev_rows if r["article"] in formal_articles])), dict(
        observed_context_hashes=sorted(observed_hashes), observed_articles=sorted(observed_articles),
        excluded_previous_distractor_articles=sorted(known_articles),
        development_articles=sorted(development_articles), formal_articles=sorted(formal_articles))


def encode_context(row, source_index, tokenizer, training=False):
    row = copy.deepcopy(row)
    qas = sorted(row["qas"], key=lambda q: q["id"])
    # Fixed first original ID for training; all public QAs for evaluation.
    if training:
        qas = qas[:1]
    for qa in qas:
        prompt = "Questions: " + qa["question"] + " Answer:"
        qa.update(prompt=prompt, prompt_ids=tokenizer(prompt, add_special_tokens=False).input_ids,
                  answer=qa["answers"][0], context_sha256=row["context_sha256"], source_index=source_index)
        qa["prompt_sha256"] = digest(qa["prompt_ids"])
        qa["answer_ids"] = tokenizer(qa["answer"] + "</s>", add_special_tokens=False).input_ids
        if not qa["prompt_ids"] or not qa["answer_ids"] or len(qa["prompt_ids"]) + len(qa["answer_ids"]) > 512:
            raise ValueError("original QA exceeds fixed reader input; refuse truncating it")
        qa["scoring_target"] = tokenizer.decode(qa["answer_ids"]).replace("</s>", "").strip()
    row["qas"] = qas
    return row


def episode(rows, tokenizer, key, training=False, distractors=()):
    contexts = [encode_context(r, i, tokenizer, training) for i, r in enumerate(rows)]
    tokenize = lambda text: tokenizer(text, add_special_tokens=False).input_ids
    writes = []
    for i, row in enumerate(contexts):
        write = make_writes("Context: " + row["context"], "related", tokenize, tokenizer.decode, "author512")[0]
        write.update(context_sha256=row["context_sha256"], source_index=i)
        writes.append(write)
    for i, row in enumerate(distractors):
        write = make_writes(row["context"], "unrelated", tokenize, tokenizer.decode, "author512")[0]
        write.update(context_sha256=row["context_sha256"], article=row["article"], distractor_index=i)
        writes.append(write)
    return dict(id=key, contexts=contexts, writes=writes)


def seal(plan):
    plan["content_sha256"] = digest(plan)
    validate_plan(plan)
    return plan


def prepare(args):
    from transformers import AutoTokenizer
    from public_retention_eval import validate_plan as validate_public
    root = Path(args.dataset_dir)
    provenance = json.loads((root / "public_dataset_manifest.json").read_text())
    if provenance["repo"] != DATA_REPO or provenance["revision"] != REVISIONS[DATA_REPO]:
        raise ValueError("pinned original SQuAD dataset revision required")
    train_path, dev_path = root / "squad/train-v2.0.json", root / "squad/dev-v2.0.json"
    for path in (train_path, dev_path):
        if provenance["files"].get(str(path.relative_to(root))) != sha(path):
            raise ValueError("raw SQuAD file hash mismatch")
    observed = json.loads(Path(args.observed_squad_plan).read_text())
    validate_public(observed)
    if observed["task"] != "squad" or len(observed["samples"]) != 100:
        raise ValueError("the already observed 100-question SQuAD plan is mandatory")
    old_distractors = json.loads(Path(args.published_distractors).read_text())
    if not isinstance(old_distractors, list) or any(not isinstance(x, str) for x in old_distractors):
        raise ValueError("author published original distractor list required")
    expected_distractors=observed["dataset"]["files"].get("squad/unrelated_contexts_from_train.json")
    if expected_distractors!=sha(args.published_distractors):
        raise ValueError("published distractors differ from the already pinned author release")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    raw_train,raw_dev=raw_squad(train_path),raw_squad(dev_path)
    splits, audit = split_rows(raw_train, raw_dev, observed, old_distractors)
    eligible = lambda rows: [r for r in rows if 16 <= len(tokenizer("Context: " + r["context"], add_special_tokens=False).input_ids) <= 512]
    training = eligible(splits["train"])
    development = eligible(splits["development"])
    formal_e1 = eligible(splits["formal"])
    if len(training) < 8 * 39 or len(development) < 19 or len(formal_e1) < 19 or len(splits["formal"]) < 19 + args.formal_contexts:
        raise ValueError("insufficient original contexts after the fixed article/length exclusions")
    used_train = training[:8 * 39]
    reserved_training_articles = {r["article"] for r in used_train}
    # Distractors are original train paragraphs from completely unused articles.
    distractors = [r for r in splits["train"] if r["article"] not in reserved_training_articles]
    if len(distractors) < 60:
        raise ValueError("not enough held-out natural distractors")
    audit.update(original_files={str(p.relative_to(root)): sha(p) for p in (train_path, dev_path)},
        published_distractors_sha256=sha(args.published_distractors),
        observed_plan_sha256=sha(args.observed_squad_plan),
        isolation="article and exact context disjoint across train/development/formal",
        distractor_articles=sorted({r["article"] for r in distractors[:60]}),
        eligibility="E1 and training: Context-prefixed original text fits 16..512 tokens, no reader-correctness filtering",
        raw_context_counts={k: len(v) for k, v in splits.items()},
        eligible_context_counts=dict(train=len(training), development=len(development), formal=len(formal_e1)))
    base = dict(schema="natural_merge_plan/1", protocol=PROTOCOL, tokenizer=tokenizer_manifest(args.model_path),
        dataset=provenance, split_audit=audit)
    plans = {}
    plans["train"] = seal(dict(base, partition="train", experiment="train", training=TRAIN_PROTOCOL,
        episodes=[episode(used_train[i*39:(i+1)*39], tokenizer, "train-"+str(i), True) for i in range(8)]))
    plans["E1-development"] = seal(dict(base, partition="development", experiment="E1",
        episodes=[episode(development[:19], tokenizer, "E1-development")]))
    plans["E1-formal"] = seal(dict(base, partition="formal", experiment="E1",
        episodes=[episode(formal_e1[:19], tokenizer, "E1-formal")]))
    excluded = {r["context_sha256"] for r in formal_e1[:19]}
    heldout = [r for r in splits["formal"] if r["context_sha256"] not in excluded][:args.formal_contexts]
    for name, length in (("E2-squad30", 30), ("E3-squad60", 60)):
        plans[name] = seal(dict(base, partition="formal", experiment="E2" if length==30 else "E3",
            episodes=[episode([r], tokenizer, name+"-"+str(i), distractors=distractors[:length]) for i, r in enumerate(heldout)]))
    nq = json.loads(Path(args.naturalqa_plan).read_text()); validate_public(nq)
    if nq["task"] != "naturalqa" or nq["mode"] != "author512" or len(nq["unrelated_writes"]) != 30:
        raise ValueError("pinned NaturalQA author512/30 public plan required for transfer")
    if nq["tokenizer"] != base["tokenizer"]:
        raise ValueError("cross-task tokenizer mismatch")
    nq_episodes = []
    transfer_writes=[]
    for w in nq["unrelated_writes"]:
        item=copy.deepcopy(w)
        text=nq["distractor_groups"][w["distractor_group"]]["text"]
        item["context_sha256"]=digest(text)
        item["source_articles"] = sorted({r["article"] for r in raw_train if r["context"] in text})
        if set(item["source_articles"]).intersection(reserved_training_articles):
            raise ValueError("NaturalQA public distractors overlap natural training articles")
        transfer_writes.append(item)
    # Group duplicate QA contexts into one stream and preserve every original QA.
    groups = {}
    for sample in nq["samples"]:
        groups.setdefault(sample["context_sha256"], []).append(sample)
    for i, samples in enumerate(groups.values()):
        row = samples[0]
        qas = []
        for s in samples:
            answer_ids = tokenizer(s["answer"]+"</s>", add_special_tokens=False).input_ids
            qas.append(dict(id=s["id"], question=s["question"], answer=s["answer"], answers=[s["answer"]],
                prompt=s["prompt"], prompt_ids=s["prompt_ids"], prompt_sha256=digest(s["prompt_ids"]),
                answer_ids=answer_ids, scoring_target=tokenizer.decode(answer_ids).replace("</s>", "").strip(),
                context_sha256=row["context_sha256"], source_index=0))
        context = dict(context=row["context"], context_sha256=row["context_sha256"], qas=qas,
            article="NaturalQA-"+row["context_sha256"], article_sha256=digest(["nq", row["context_sha256"]]))
        nq_episodes.append(dict(id="E3-naturalqa-"+str(i), contexts=[context],
            writes=row["related_writes"]+transfer_writes))
    plans["E3-naturalqa30"] = seal(dict(base, partition="transfer", experiment="E3", dataset=nq["dataset"],
        source_public_plan_sha256=sha(args.naturalqa_plan), transfer_policy="never model selection or tuning",
        episodes=nq_episodes))
    destination = Path(args.out_dir)
    if destination.exists():
        raise ValueError("suite output directory exists; refuse replacing a sealed split")
    destination.mkdir(parents=True)
    index = dict(schema="natural_merge_suite/1", protocol=PROTOCOL, training=TRAIN_PROTOCOL, split_audit=audit, plans={})
    for name, plan in plans.items():
        path = destination/(name+".json"); write_new(path, plan)
        index["plans"][name] = dict(path=path.name, sha256=sha(path), partition=plan["partition"],
            contexts=len({c["context_sha256"] for e in plan["episodes"] for c in e["contexts"]}),
            qa=sum(len(c["qas"]) for e in plan["episodes"] for c in e["contexts"]))
    index["content_sha256"] = digest(index)
    write_new(destination/"suite.json", index)
    print(json.dumps(index, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--observed-squad-plan", required=True)
    p.add_argument("--published-distractors", required=True)
    p.add_argument("--naturalqa-plan", required=True)
    p.add_argument("--model-path", default="/dev/shm/s1d_reader_fp16")
    p.add_argument("--formal-contexts", type=int, default=32)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    if args.formal_contexts < 2:
        p.error("at least two original formal contexts required")
    prepare(args)

if __name__ == "__main__":
    main()
