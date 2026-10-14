#!/usr/bin/env python3
"""Independently audit prepared original natural inputs; never load a reader."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_public_retention import digest, sha, make_writes, tokenizer_manifest, write_new
from prepare_natural_merge import raw_squad, validate_plan, encode_context
from public_retention_eval import validate_plan as validate_public
from stage_natural_merge_data import verify_file, NAMES


def audit(args):
    from transformers import AutoTokenizer
    root = Path(args.dataset_dir)
    manifest = json.loads((root / "public_dataset_manifest.json").read_text())
    for name in NAMES:
        if verify_file(root / name, manifest["upstream_file_metadata"][name]) != manifest["files"][name]:
            raise ValueError("original asset no longer matches its pinned upstream identity")
    public_root = Path(args.processed_dir)
    processed_manifest = json.loads((public_root / "public_dataset_manifest.json").read_text())
    for name, expected in processed_manifest["files"].items():
        if sha(public_root / name) != expected:
            raise ValueError("original processed public release changed")
    suite_path = Path(args.suite)
    suite = json.loads(suite_path.read_text())
    identity = dict(suite)
    expected = identity.pop("content_sha256", None)
    if expected != digest(identity):
        raise ValueError("suite content checksum mismatch")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenize = lambda s: tokenizer(s, add_special_tokens=False).input_ids
    tokenizer_hashes = tokenizer_manifest(args.model_path)
    train_raw = raw_squad(root / "squad/train-v2.0.json")
    dev_raw = raw_squad(root / "squad/dev-v2.0.json")
    train_map = {r["context_sha256"]: r for r in train_raw}
    dev_map = {r["context_sha256"]: r for r in dev_raw}
    observed = json.loads(Path(args.observed_squad_plan).read_text())
    naturalqa = json.loads(Path(args.naturalqa_plan).read_text())
    validate_public(observed)
    validate_public(naturalqa)
    observed_hashes = {s["context_sha256"] for s in observed["samples"]}
    observed_articles = {dev_map[h]["article"] for h in observed_hashes}
    plans, summaries = {}, {}
    for name, entry in suite["plans"].items():
        path = suite_path.parent / entry["path"]
        if sha(path) != entry["sha256"]:
            raise ValueError("sealed plan file changed")
        plan = json.loads(path.read_text())
        validate_plan(plan)
        if plan["tokenizer"] != tokenizer_hashes:
            raise ValueError("prepared tokenizer mismatch")
        plans[name] = plan
        if plan["partition"] != "transfer":
            source = train_map if plan["partition"] == "train" else dev_map
            for episode in plan["episodes"]:
                for index, context in enumerate(episode["contexts"]):
                    raw = source[context["context_sha256"]]
                    original = encode_context(raw, index, tokenizer, plan["partition"] == "train")
                    if context != original:
                        raise ValueError("prepared context/QA differs from original raw source")
                    expected_write = make_writes("Context: " + raw["context"], "related", tokenize,
                                                 tokenizer.decode, "author512")[0]
                    write = episode["writes"][index]
                    if any(write[k] != v for k, v in expected_write.items()):
                        raise ValueError("prepared related write does not match raw original paragraph")
                for write in episode["writes"][len(episode["contexts"]):]:
                    raw = train_map[write["context_sha256"]]
                    expected_write = make_writes(raw["context"], "unrelated", tokenize,
                                                 tokenizer.decode, "author512")[0]
                    if write["article"] != raw["article"] or any(write[k] != v for k,v in expected_write.items()):
                        raise ValueError("natural distractor tokens/source changed")
        else:
            published = {(s["context_sha256"], s["id"]): s for s in naturalqa["samples"]}
            seen = set()
            for episode in plan["episodes"]:
                context = episode["contexts"][0]
                for qa in context["qas"]:
                    key = context["context_sha256"], qa["id"]
                    sample = published[key]
                    seen.add(key)
                    if (context["context"] != sample["context"] or qa["question"] != sample["question"]
                            or qa["answer"] != sample["answer"] or qa["prompt_ids"] != sample["prompt_ids"]):
                        raise ValueError("NaturalQA no longer contains unchanged public original fields")
                    if qa["answer_ids"] != tokenize(sample["answer"] + "</s>"):
                        raise ValueError("NaturalQA answer tokens changed")
                reference = sample["related_writes"] + naturalqa["unrelated_writes"]
                if len(reference) != len(episode["writes"]):
                    raise ValueError("NaturalQA original write coverage changed")
                for prepared, original in zip(episode["writes"], reference):
                    if any(prepared[k] != v for k,v in original.items()):
                        raise ValueError("NaturalQA prepared write differs from public plan")
            if seen != set(published):
                raise ValueError("NaturalQA original QA missing or substituted")
        summaries[name] = dict(contexts=sum(len(e["contexts"]) for e in plan["episodes"]),
            qa=sum(len(c["qas"]) for e in plan["episodes"] for c in e["contexts"]),
            streams=len(plan["episodes"]), writes_per_stream=sorted({len(e["writes"]) for e in plan["episodes"]}),
            truncated_related_writes=sum(w["truncated_tokens"] > 0 for e in plan["episodes"] for w in e["writes"] if w["role"] == "related"),
            plan_sha256=sha(path), content_sha256=plan["content_sha256"])
    def sets(name, field):
        return {c[field] for e in plans[name]["episodes"] for c in e["contexts"]}
    names = ("train", "E1-development", "E1-formal", "E2-squad30")
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if sets(a, "context_sha256") & sets(b, "context_sha256"):
                raise ValueError("train/development/formal exact context overlap")
    if sets("train", "article") & (sets("E1-development", "article") | sets("E1-formal", "article") | sets("E2-squad30", "article")):
        raise ValueError("training and evaluation article overlap")
    if sets("E1-development", "article") & (sets("E1-formal", "article") | sets("E2-squad30", "article")):
        raise ValueError("development and formal article overlap")
    for name in names:
        if sets(name, "context_sha256") & observed_hashes or sets(name, "article") & observed_articles:
            raise ValueError("observed article/context accidentally reused")
    train_articles = sets("train", "article")
    for name in ("E2-squad30", "E3-squad60", "E3-naturalqa30"):
        for e in plans[name]["episodes"]:
            for w in e["writes"]:
                if w.get("article") in train_articles or train_articles.intersection(w.get("source_articles", [])):
                    raise ValueError("natural distractor and training article overlap")
    if sets("E2-squad30", "context_sha256") != sets("E3-squad60", "context_sha256"):
        raise ValueError("30/60 conditions do not share original contexts")
    for a, b in zip(plans["E2-squad30"]["episodes"], plans["E3-squad60"]["episodes"]):
        if a["contexts"] != b["contexts"] or a["writes"] != b["writes"][:31]:
            raise ValueError("30/60 conditions do not share the fixed initial stream")
    certificate = dict(schema="natural_merge_data_certificate/1", status="passed",
        server="v100-2", suite_sha256=sha(suite_path), original_dataset=manifest,
        processed_dataset_revision=processed_manifest["revision"], tokenizer=tokenizer_hashes,
        observed_articles_excluded=sorted(observed_articles), plans=summaries,
        verified=["upstream original Git/LFS identities and byte sizes", "all processed public file SHA256",
            "all SQuAD original context and QA identities/spans", "all original write tokens retokenized",
            "unchanged NaturalQA public context/question/answer/write tokens", "article and context train/dev/formal isolation",
            "observed article exclusion", "held-out distractor articles", "same SQuAD30/60 initial stream"],
        reader_loaded=False, gpu_training_started=False, gpu_evaluation_started=False,
        scope="original SQuAD complete train/dev and published processed NaturalQA; no unused full NQ JSONL download")
    write_new(args.out, certificate)
    print(json.dumps(dict(status="passed", plans=summaries, certificate=args.out), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("dataset-dir", "processed-dir", "suite", "observed-squad-plan", "naturalqa-plan", "out"):
        p.add_argument("--" + field, required=True)
    p.add_argument("--model-path", default="/dev/shm/s1d_reader_fp16")
    audit(p.parse_args())


if __name__ == "__main__":
    main()
