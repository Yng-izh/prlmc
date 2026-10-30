#!/usr/bin/env python3
"""Evaluate the frozen MemoryLLM-8B on the local LooGLE QA test files.

Official components are copied verbatim from the LooGLE repository pinned at
fb9aee3b9a8de064e3d4675eec2e8036327c1349:
  config/task2prompt.json     -> PROMPTS / prompt assembly
  config/task2maxlen.json     -> MAX_NEW (max_gen)
  Evaluation/automatic_metrics.py -> get_rouge_score (ROUGE-1/2/L, metric="r")
  Evaluation/automatic_eval.py    -> empty hypothesis scored as the string "None"
  Prediction/pred_opensource_models.py -> greedy decoding with repetition_penalty=2.0

The capacity is the checkpoint's native 50-block memory only; the adapted input
protocol (tail2048_native_write_v1) writes the document in <=512-token chunks and
queries the final 2,048 tokens directly.
"""
import argparse
import ast
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from longbench_memoryllm_eval import (  # noqa: E402
    atomic_json, digest, native_write, reset_model, seed_everything, sha,
    truncated_prediction,
)

DEFAULT_TASKS = ("shortdep_qa", "longdep_qa")
# Pinned upstream source revisions used for the copied prompts and metrics.
LOOGLE_COMMIT = "fb9aee3b9a8de064e3d4675eec2e8036327c1349"
MEMORYLLM_COMMIT = "5acaf52d2e60087f6bc38a06063d7558ab9a2c2c"
# config/task2maxlen.json (max_gen per task).
MAX_NEW = {"shortdep_qa": 300, "longdep_qa": 500}
# config/task2prompt.json; both QA tasks share one template.
PROMPT = ("Please answer the question based on the long texts below. \n{context}\n"
          "Question: {question}\nAnswer: ")


def load_rows(data_dir, task):
    path = Path(data_dir) / (task + ".jsonl")
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if not rows:
        raise ValueError(f"empty LooGLE file: {path}")
    for row in rows:
        for key in ("context", "question", "answer"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"invalid LooGLE row in {path}: missing {key}")
    return rows


def truncate_ids(ids, total_tokens, doc_truncation):
    if len(ids) <= total_tokens:
        return ids
    if doc_truncation == "middle":
        # LooGLE Prediction/pred_opensource_models.py: first half + last half.
        half = total_tokens // 2
        return ids[:half] + ids[-half:]
    return ids[-total_tokens:]


def build_prompt(task, row):
    return PROMPT.format(context=row["context"], question=row["question"])


def split_prompt(tokenizer, task, row, total_tokens, tail_tokens,
                 doc_truncation):
    prompt = build_prompt(task, row)
    ids = tokenizer(prompt, add_special_tokens=False, truncation=False).input_ids
    ids = truncate_ids(ids, total_tokens, doc_truncation)
    query_tokens = min(tail_tokens, len(ids))
    write_ids = ids[:-query_tokens] if query_tokens else ids
    query_ids = ids[-query_tokens:] if query_tokens else []
    return prompt, ids, write_ids, query_ids


class RougeScorer:
    """ROUGE-1/2/L exactly as LooGLE's Evaluation/automatic_metrics.py computes it.

    LooGLE calls pltrdy/rouge, which wraps rouge_score.RougeScorer with
    use_stemmer=True and maps an empty side to a zero score (ValueError). The
    reported QA metric in the official table is rouge-l recall (metric="r").
    """

    def __init__(self):
        from rouge_score import rouge_scorer
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def score(self, reference, hypothesis):
        if hypothesis == "":
            hypothesis = "None"  # LooGLE Evaluation/automatic_eval.py
        try:
            scores = self.scorer.score(reference, hypothesis)
        except ValueError:  # empty reference or hypothesis
            return {"rouge1": (0.0, 0.0, 0.0), "rouge2": (0.0, 0.0, 0.0),
                    "rougeL": (0.0, 0.0, 0.0)}
        return {k: (v.fmeasure, v.precision, v.recall)
                for k, v in scores.items()}


def generate(model, tokenizer, query_ids, max_new_tokens, repetition_penalty):
    import torch
    ids = torch.tensor([query_ids], dtype=torch.long, device=model.memory.device)
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
    eos_ids = [tokenizer.eos_token_id, newline_id]
    attention = torch.ones((1, model.num_blocks * model.num_tokens + len(query_ids)),
                           dtype=torch.long, device=ids.device)
    with torch.no_grad():
        output = model.generate(input_ids=ids, attention_mask=attention,
                                max_new_tokens=max_new_tokens, num_beams=1,
                                do_sample=False, temperature=1.0, top_p=1.0,
                                top_k=0, repetition_penalty=repetition_penalty,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=eos_ids)
    out = output[0, ids.shape[1]:].detach().cpu().tolist()
    return tokenizer.decode(out, skip_special_tokens=True), out


def build_protocol(args, tasks, drop_memory_per_layer):
    return dict(
        model="MemoryLLM-8B", capacity="native_50_blocks", capacity_blocks=50,
        latent_tokens_per_block=256, max_write_tokens=512,
        total_input_tokens=args.total_tokens,
        direct_query_tokens=args.tail_tokens,
        write_prefix_tokens=args.total_tokens - args.tail_tokens,
        max_new_tokens={t: MAX_NEW[t] for t in tasks}, seed=args.seed,
        do_sample=False, num_beams=1,
        repetition_penalty=args.repetition_penalty, training=False,
        benchmark="LooGLE test", tasks=list(tasks),
        prompt="official config/task2prompt.json (Short/Long dependency QA)",
        scoring="official LooGLE automatic_sim ROUGE: rouge_score RougeScorer("
                "rouge1/rouge2/rougeL, use_stemmer=True); headline = rouge-l "
                "recall (LooGLE get_rouge_score default metric='r'); empty "
                "hypothesis scored as 'None'",
        doc_truncation=args.doc_truncation,
        input_protocol=(f"tail{args.tail_tokens}_native_write_v1; "
                        "not exact legacy or M+ reproduction"),
        seed_reset="reset to 0 before each sample; independent of worker scheduling",
        memory_reset="checkpoint values and initialized flag before every sample",
        native_eviction="unmodified model.inject_memory(update_memory=True)",
        drop_memory_per_layer=bool(drop_memory_per_layer))


def evaluate(args):
    if args.capacity_blocks != 50:
        raise ValueError("this protocol supports native checkpoint capacity only: --capacity-blocks 50")
    tasks = tuple(args.tasks) if args.tasks else DEFAULT_TASKS
    if any(t not in DEFAULT_TASKS for t in tasks):
        raise ValueError("unsupported task")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max-samples must be positive")
    if args.seed != 0:
        raise ValueError("the fixed evaluation seed is 0")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("output exists; refuse overwrite")
    task_rows = {t: load_rows(args.data_dir, t) for t in tasks}
    data_identity = {t: dict(sha256=sha(Path(args.data_dir) / (t + ".jsonl")),
                            n=len(task_rows[t])) for t in tasks}
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    seed_everything(args.seed)
    import torch
    from transformers import AutoTokenizer
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = MemoryLLM.from_pretrained(args.model_path, torch_dtype=torch.float16,
        attn_implementation="sdpa", local_files_only=True).to("cuda:0").eval()
    model.requires_grad_(False)
    if int(model.num_blocks) != 50 or int(model.num_tokens) != 256:
        raise ValueError("checkpoint is not a native 50-block MemoryLLM")
    if tuple(model.memory.shape) != (32, 12800, 4096):
        raise ValueError("expected unchanged MemoryLLM-8B native memory shape")
    initial_memory = model.memory.detach().cpu().clone()
    initial_initialized = model.initialized.detach().cpu().clone()
    rouge = RougeScorer()
    report = dict(
        schema="loogle_memoryllm_results/1", status="running",
        protocol=build_protocol(args, tasks, model.drop_memory_per_layer),
        identity=dict(model_path=args.model_path, data_dir=args.data_dir,
            doc_truncation=args.doc_truncation, tail_tokens=args.tail_tokens,
            model_config_sha256=hashlib.sha256(
                (Path(args.model_path) / "config.json").read_bytes()).hexdigest(),
            loogle_source_commit=LOOGLE_COMMIT,
            memoryllm_source_commit=MEMORYLLM_COMMIT,
            actual_reader_sha256=sha(ROOT / "methods/memoryllm_modeling/modeling_memoryllm.py"),
            actual_entrypoint_sha256=sha(__file__), data_files=data_identity,
            initial_memory_sha256=hashlib.sha256(
                initial_memory.numpy().tobytes()).hexdigest()),
        runtime=dict(torch=torch.__version__,
            transformers=__import__("transformers").__version__,
            cuda=torch.version.cuda, device=args.cuda_device,
            gpu=torch.cuda.get_device_name(0)),
        full_test_split=args.max_samples is None, max_samples=args.max_samples,
        training_performed=False, optimizer_created=False,
        tasks=list(tasks), samples=[], progress=None)
    started = time.monotonic()
    atomic_json(output, report)
    for task in tasks:
        rows = task_rows[task]
        if args.max_samples is not None:
            rows = rows[:args.max_samples]
        for sample_index, row in enumerate(rows):
            sample_started = time.monotonic()
            seed_everything(args.seed)
            reset_model(model, initial_memory, initial_initialized,
                        reset_memory=args.reset_memory)
            prompt, all_ids, write_ids, query_ids = split_prompt(
                tokenizer, task, row, args.total_tokens,
                tail_tokens=args.tail_tokens, doc_truncation=args.doc_truncation)
            original_tokens = len(tokenizer(prompt, add_special_tokens=False,
                                            truncation=False).input_ids)
            writes = [write_ids[i:i + 512] for i in range(0, len(write_ids), 512)]
            for chunk in writes:
                native_write(model, chunk)
            prediction, output_ids = generate(
                model, tokenizer, query_ids, MAX_NEW[task], args.repetition_penalty)
            truncated = truncated_prediction(prediction)
            if model.memory.shape[1] != 12800 or int(model.num_blocks) != 50:
                raise ValueError("native capacity changed during evaluation")
            raw_r = rouge.score(row["answer"], prediction)
            cut_r = rouge.score(row["answer"], truncated)
            report["samples"].append(dict(
                task=task, sample_index=sample_index, id=row.get("id"),
                doc_id=row.get("doc_id"), title=row.get("title"),
                source_row_sha256=digest(row), context_sha256=digest(row["context"]),
                input_token_sha256=digest(all_ids), original_prompt_tokens=original_tokens,
                truncated_tokens=original_tokens - len(all_ids),
                input_tokens=len(all_ids), prefix_tokens=len(write_ids),
                query_tokens=len(query_ids), write_count=len(writes),
                native_capacity_blocks=int(model.num_blocks),
                native_active_slots=int(model.memory.shape[1]),
                native_evicted_slots_per_layer=256 * len(writes),
                max_new_tokens=MAX_NEW[task], output_token_count=len(output_ids),
                seconds=time.monotonic() - sample_started,
                prediction=prediction, output_tokens=output_ids,
                answer=row["answer"],
                rouge_l_recall=raw_r["rougeL"][2], rouge_l_f=raw_r["rougeL"][0],
                rouge_1_recall=raw_r["rouge1"][2], rouge_2_recall=raw_r["rouge2"][2],
                rouge_l_recall_truncated=cut_r["rougeL"][2],
                rouge_l_f_truncated=cut_r["rougeL"][0],
                rouge_1_recall_truncated=cut_r["rouge1"][2],
                rouge_2_recall_truncated=cut_r["rouge2"][2]))
            report["progress"] = dict(task=task, sample=sample_index + 1,
                                     total=len(rows))
            report["elapsed_seconds"] = time.monotonic() - started
            atomic_json(output, report)
            print(json.dumps(dict(**report["progress"],
                rouge_l_r=report["samples"][-1]["rouge_l_recall_truncated"],
                seconds=report["samples"][-1]["seconds"])), flush=True)
    report["status"] = "complete"
    report["summary"] = {}
    for task in tasks:
        picked = [s for s in report["samples"] if s["task"] == task]
        report["summary"][task] = dict(
            n=len(picked),
            rouge_l_recall=100 * float(np.mean([s["rouge_l_recall_truncated"] for s in picked])),
            rouge_l_f=100 * float(np.mean([s["rouge_l_f_truncated"] for s in picked])),
            rouge_1_recall=100 * float(np.mean([s["rouge_1_recall_truncated"] for s in picked])),
            rouge_2_recall=100 * float(np.mean([s["rouge_2_recall_truncated"] for s in picked])),
            rouge_l_recall_raw=100 * float(np.mean([s["rouge_l_recall"] for s in picked])))
    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    atomic_json(output, report)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--total-tokens", type=int, required=True,
                   choices=(8192, 16384, 32768, 128000),
                   help="document token budget including the directly queried tail")
    p.add_argument("--capacity-blocks", type=int, default=50, choices=(50,))
    p.add_argument("--tail-tokens", dest="tail_tokens", type=int, default=2048,
                   choices=(512, 1024, 2048),
                   help="tokens queried directly at the tail (default 2048)")
    p.add_argument("--doc-truncation", dest="doc_truncation", default="tail",
                   choices=("tail", "middle"),
                   help="tail keeps the last total-tokens (frozen PRLM-C protocol); "
                        "middle reproduces LooGLE pred_opensource_models.py")
    p.add_argument("--repetition-penalty", dest="repetition_penalty",
                   type=float, default=2.0,
                   help="official LooGLE generation uses 2.0; set 1.0 to match "
                        "the frozen LongBench decoding")
    p.add_argument("--reset-memory", action="store_true",
                   help="clear the checkpoint's initial residual pool before every sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tasks", nargs="+", choices=DEFAULT_TASKS)
    p.add_argument("--max-samples", type=int, default=None,
                   help="debug smoke limit; omit for the complete official test split")
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()
