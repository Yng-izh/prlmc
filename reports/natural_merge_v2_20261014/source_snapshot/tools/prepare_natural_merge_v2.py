#!/usr/bin/env python3
"""Restore original train QAs and seal a common A/B/C training curriculum.

Existing v1 train/development/formal plans are never modified. Only the same
312 isolated train contexts are used; all QA text comes from the pinned raw file.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_natural_merge import validate_plan, raw_squad, encode_context
from prepare_public_retention import digest, sha, write_new, tokenizer_manifest

TRAIN_V2 = dict(seed=0, streams=8, rounds=4, planned_updates=64,
    single_merge_updates=32, recurrent_updates=32, rank=32, gamma_max=.05,
    optimizer="AdamW", learning_rate=1e-4, weight_decay=.01, gradient_clip=1.,
    qa_schedule="one original QA per written context; sorted ID index round % QA count",
    curriculum="rounds 0/1: two isolated 19-context windows; rounds 2/3: 39-context recurrent streams",
    single_window_starts=[[0,20],[1,20]], recurrent_event_before_writes=[20,30],
    recovery_weight="1 + clamp(CE_Selection - CE_Full, 0, 2); normalized to mean one per event",
    teacher_weight=.5, teacher_temperature=1., teacher_max_gold_ce=2.,
    teacher_scope="uncompressed independent Full stream at identical raw-write prefix; no student compression",
    distillation="answer-position KL(Full || Shared), averaged over eligible training QAs",
    truncated_gradient_at_compression=True,
    checkpoint="last scheduled update only; no within-training selection",
    variants={"A":dict(architecture="block_v1",objective="mean_gold_ce"),
              "B":dict(architecture="block_v1",objective="recovery_ce_kd"),
              "C":dict(architecture="discarded_slot_attention_v1",objective="recovery_ce_kd")})


def selected_qas(contexts, round_index):
    return [c["qas"][round_index % len(c["qas"])] for c in contexts]


def validate_v2_plan(plan):
    identity = dict(plan); expected = identity.pop("content_sha256", None)
    if (expected != digest(identity) or plan.get("schema") != "natural_merge_v2_training_plan/1"
            or plan.get("training") != TRAIN_V2):
        raise ValueError("v2 training plan integrity/protocol failure")
    base = plan["base_plan"]; validate_plan(base)
    if base["partition"] != "train" or base["experiment"] != "train":
        raise ValueError("only the isolated original training plan may be enriched")
    if plan["tokenizer"] != base["tokenizer"] or len(plan["episodes"]) != 8:
        raise ValueError("v2 tokenizer/stream scope mismatch")
    qa_ids = set()
    for original, enriched in zip(base["episodes"], plan["episodes"]):
        if (enriched["id"] != original["id"] or enriched["writes"] != original["writes"]
                or len(enriched["contexts"]) != 39):
            raise ValueError("v2 raw write stream changed")
        for old, context in zip(original["contexts"], enriched["contexts"]):
            for field in ("context", "context_sha256", "article", "article_sha256"):
                if context[field] != old[field]:
                    raise ValueError("v2 original train context/article changed")
            if digest(context["context"]) != context["context_sha256"] or not context["qas"]:
                raise ValueError("empty/modified train context")
            if [q["id"] for q in context["qas"]] != sorted(q["id"] for q in context["qas"]):
                raise ValueError("train QA rotation must be deterministic")
            for qa in context["qas"]:
                if qa["id"] in qa_ids or not qa["answers"] or not qa["answer_ids"]:
                    raise ValueError("duplicate/empty original train QA")
                qa_ids.add(qa["id"])
                if (qa["context_sha256"] != context["context_sha256"]
                        or qa["source_index"] != enriched["contexts"].index(context)
                        or len(qa["prompt_ids"])+len(qa["answer_ids"]) > 512
                        or digest(qa["prompt_ids"]) != qa["prompt_sha256"]
                        or digest(qa["answer_ids"]) != qa["answer_sha256"]):
                    raise ValueError("train QA token provenance mismatch")
            restored = {q["id"]:q for q in context["qas"]}
            for q in old["qas"]:
                if any(restored.get(q["id"], {}).get(k) != q[k] for k in
                       ("question", "answers", "prompt_ids", "answer_ids")):
                    raise ValueError("previously sealed original training QA changed")


def prepare(args):
    base = json.loads(Path(args.base_train_plan).read_text()); validate_plan(base)
    if base["partition"] != "train" or base["experiment"] != "train":
        raise ValueError("train partition required")
    root = Path(args.dataset_dir)
    manifest = json.loads((root/"public_dataset_manifest.json").read_text())
    raw = root/"squad/train-v2.0.json"
    if (manifest != base["dataset"] or manifest["files"].get("squad/train-v2.0.json") != sha(raw)
            or base["split_audit"]["original_files"]["squad/train-v2.0.json"] != sha(raw)):
        raise ValueError("pinned raw training corpus changed")
    if tokenizer_manifest(args.model_path) != base["tokenizer"]:
        raise ValueError("training tokenizer changed")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    rows = {(r["article"],r["context_sha256"]):r for r in raw_squad(raw)}
    episodes = copy.deepcopy(base["episodes"])
    for ep in episodes:
        for i, context in enumerate(ep["contexts"]):
            row = rows.get((context["article"],context["context_sha256"]))
            if row is None or row["context"] != context["context"]:
                raise ValueError("train paragraph absent from pinned raw corpus")
            restored = encode_context(row, i, tokenizer, training=False)
            for q in restored["qas"]:
                q["answer_sha256"] = digest(q["answer_ids"])
            ep["contexts"][i] = restored
    plan = dict(schema="natural_merge_v2_training_plan/1", base_plan=base,
        base_plan_sha256=sha(args.base_train_plan), raw_train_sha256=sha(raw),
        tokenizer=base["tokenizer"], training=TRAIN_V2, episodes=episodes)
    plan["content_sha256"] = digest(plan); validate_v2_plan(plan)
    write_new(args.out, plan)
    print(json.dumps(dict(out=args.out, content_sha256=plan["content_sha256"],
        contexts=312, original_train_qas=sum(len(c["qas"]) for e in episodes for c in e["contexts"]),
        training=TRAIN_V2), ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-train-plan", required=True)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--model-path", default="/dev/shm/s1d_reader_fp16")
    p.add_argument("--out", required=True)
    prepare(p.parse_args())


if __name__ == "__main__": main()
