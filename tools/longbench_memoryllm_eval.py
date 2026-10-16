#!/usr/bin/env python3
"""Evaluate the frozen MemoryLLM-8B on the local LongBench test files.

This entrypoint intentionally implements one capacity only: the checkpoint's
native 50-block memory. It uses the official prompts and native writer, with
an explicitly adapted input protocol: the final 2,048
tokens are queried directly and the preceding tokens are written in chunks of
at most 512 tokens. No training, PRLM-C compactor, or alternative capacity is
available in this script.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ("hotpotqa", "2wikimqa", "musique", "qasper", "multifieldqa_en", "narrativeqa")
# Pinned upstream source revisions used for the copied prompts and metric.
LONGBENCH_COMMIT = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
MEMORYLLM_COMMIT = "5acaf52d2e60087f6bc38a06063d7558ab9a2c2c"
MAX_NEW = {
    "hotpotqa": 32, "2wikimqa": 32, "musique": 32,
    "multifieldqa_en": 64, "qasper": 128, "narrativeqa": 128,
}
PROMPTS = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def qa_f1(prediction, answer):
    import re
    import string
    from collections import Counter
    def norm(s):
        s = s.lower().translate(str.maketrans("", "", string.punctuation))
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        return " ".join(s.split())
    p, g = norm(prediction).split(), norm(answer).split()
    common = sum((Counter(p) & Counter(g)).values())
    # LongBench returns zero even when both normalized strings are empty.
    if common == 0:
        return 0.0
    return 2.0 * common / (len(p) + len(g))


def score(prediction, answers):
    return max(qa_f1(prediction, answer) for answer in answers)


def load_rows(data_dir, task):
    path = Path(data_dir) / (task + ".jsonl")
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if not rows or any(not isinstance(r.get("context"), str) or not r["answers"] for r in rows):
        raise ValueError(f"invalid LongBench rows: {path}")
    return rows


def split_prompt(tokenizer, task, row, total_tokens):
    prompt = PROMPTS[task].format(context=row["context"], input=row["input"])
    ids = tokenizer(prompt, add_special_tokens=False, truncation=False).input_ids
    if len(ids) > total_tokens:
        # Explicit MemoryLLM-style tail truncation, applied once to token IDs.
        # This differs from upstream LongBench's middle truncation.
        ids = ids[-total_tokens:]
    query_tokens = min(2048, len(ids))
    write_ids = ids[:-query_tokens] if query_tokens else ids
    query_ids = ids[-query_tokens:] if query_tokens else []
    return prompt, ids, write_ids, query_ids


def reset_model(model, initial_memory, initial_initialized):
    import torch
    with torch.no_grad():
        model.memory.copy_(initial_memory)
        model.initialized.copy_(initial_initialized)
        if hasattr(model, "memory_active_mask"):
            model.memory_active_mask.fill_(True)


def native_write(model, token_ids):
    import torch
    ids = torch.tensor([token_ids], dtype=torch.long, device=model.memory.device)
    attention = torch.ones((1, model.num_tokens + len(token_ids)), dtype=torch.long, device=ids.device)
    with torch.no_grad():
        model.inject_memory(ids, context_attention_mask=attention, update_memory=True)


def generate(model, tokenizer, query_ids, max_new_tokens):
    import torch
    ids = torch.tensor([query_ids], dtype=torch.long, device=model.memory.device)
    attention = torch.ones((1, model.num_blocks * model.num_tokens + len(query_ids)),
                           dtype=torch.long, device=ids.device)
    with torch.no_grad():
        output = model.generate(input_ids=ids, attention_mask=attention,
                                max_new_tokens=max_new_tokens, num_beams=1,
                                do_sample=False, temperature=1.0, top_p=1.0, top_k=0,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id)
    out = output[0, ids.shape[1]:].detach().cpu().tolist()
    return tokenizer.decode(out, skip_special_tokens=True), out


def evaluate(args):
    if args.capacity_blocks != 50:
        raise ValueError("this protocol supports native checkpoint capacity only: --capacity-blocks 50")
    if args.total_tokens not in (8192, 16384):
        raise ValueError("--total-tokens must be 8192 or 16384")
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
    data_identity = {t: dict(sha256=sha(Path(args.data_dir)/(t+".jsonl")),
                             n=len(task_rows[t])) for t in tasks}
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    seed_everything(args.seed)
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, str(ROOT))
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = MemoryLLM.from_pretrained(args.model_path, torch_dtype=torch.float16,
        attn_implementation="sdpa", local_files_only=True).to("cuda:0").eval()
    model.requires_grad_(False)
    if int(model.num_blocks) != 50 or int(model.num_tokens) != 256 or int(model.initialized) != 1:
        raise ValueError("checkpoint is not an initialized native 50-block MemoryLLM")
    if tuple(model.memory.shape) != (32, 12800, 4096):
        raise ValueError("expected unchanged MemoryLLM-8B native memory shape")
    initial_memory = model.memory.detach().cpu().clone()
    initial_initialized = model.initialized.detach().cpu().clone()
    initial_memory_sha256 = hashlib.sha256(initial_memory.numpy().tobytes()).hexdigest()
    report = dict(schema="longbench_memoryllm_results/1", status="running",
        protocol=dict(model="MemoryLLM-8B", capacity="native_50_blocks", capacity_blocks=50,
                      latent_tokens_per_block=256, max_write_tokens=512,
                      total_input_tokens=args.total_tokens,
                      direct_query_tokens=2048, write_prefix_tokens=args.total_tokens - 2048,
                      max_new_tokens={t: MAX_NEW[t] for t in tasks}, seed=args.seed,
                      do_sample=False, num_beams=1, training=False,
                      benchmark="LongBench test", scoring="official QA F1",
                      prompt_truncation="keep last total_input_tokens; no decode/re-encode",
                      input_protocol="tail2048_native_write_v1; not exact legacy or M+ reproduction",
                      seed_reset="reset to 0 before each sample; independent of worker scheduling",
                      memory_reset="checkpoint values and initialized flag before every sample",
                      native_eviction="unmodified model.inject_memory(update_memory=True)",
                      drop_memory_per_layer=bool(model.drop_memory_per_layer)),
        identity=dict(model_path=args.model_path, data_dir=args.data_dir,
                      model_config_sha256=hashlib.sha256((Path(args.model_path)/"config.json").read_bytes()).hexdigest(),
                      longbench_source_commit=LONGBENCH_COMMIT,
                      memoryllm_source_commit=MEMORYLLM_COMMIT,
                      actual_reader_sha256=sha(ROOT/"methods/memoryllm_modeling/modeling_memoryllm.py"),
                      actual_entrypoint_sha256=sha(__file__), data_files=data_identity,
                      tokenizer_files={n: sha(Path(args.model_path)/n) for n in
                          ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
                          if (Path(args.model_path)/n).is_file()},
                      fp16_cache_manifest_sha256=sha(Path(args.model_path)/"fp16_cache_manifest.json")
                          if (Path(args.model_path)/"fp16_cache_manifest.json").is_file() else None,
                      initial_memory_sha256=initial_memory_sha256),
        runtime=dict(torch=torch.__version__, transformers=__import__("transformers").__version__,
                     cuda=torch.version.cuda, device=args.cuda_device, gpu=torch.cuda.get_device_name(0)),
        full_test_split=args.max_samples is None, max_samples=args.max_samples,
        training_performed=False, optimizer_created=False,
        tasks=list(tasks), samples=[], progress=None)
    started = time.monotonic()
    data_zip = Path(args.data_dir).parent / "data.zip"
    if data_zip.is_file():
        report["identity"]["data_zip_sha256"] = sha(data_zip)
    atomic_json(output, report)
    for task in tasks:
        rows = task_rows[task]
        if args.max_samples is not None:
            rows = rows[:args.max_samples]
        for sample_index, row in enumerate(rows):
            sample_started = time.monotonic()
            seed_everything(args.seed)
            reset_model(model, initial_memory, initial_initialized)
            prompt, all_ids, write_ids, query_ids = split_prompt(tokenizer, task, row, args.total_tokens)
            original_tokens = len(tokenizer(prompt, add_special_tokens=False, truncation=False).input_ids)
            writes = [write_ids[i:i + 512] for i in range(0, len(write_ids), 512)]
            for chunk in writes:
                native_write(model, chunk)
            prediction, output_ids = generate(model, tokenizer, query_ids, MAX_NEW[task])
            if model.memory.shape[1] != 12800 or int(model.num_blocks) != 50:
                raise ValueError("native capacity changed during evaluation")
            report["samples"].append(dict(task=task, sample_index=sample_index, id=row.get("_id"),
                source_row_sha256=digest(row), context_sha256=digest(row["context"]),
                input_token_sha256=digest(all_ids), original_prompt_tokens=original_tokens,
                truncated_tokens=original_tokens-len(all_ids),
                input_tokens=len(all_ids), prefix_tokens=len(write_ids), query_tokens=len(query_ids),
                write_count=len(writes), write_chunk_tokens=[len(x) for x in writes],
                native_capacity_blocks=int(model.num_blocks), native_active_slots=int(model.memory.shape[1]),
                native_evicted_slots_per_layer=256*len(writes),
                max_new_tokens=MAX_NEW[task], output_token_count=len(output_ids),
                seconds=time.monotonic()-sample_started,
                prediction=prediction, output_tokens=output_ids, answers=row["answers"],
                official_qa_f1=score(prediction, row["answers"])))
            report["progress"] = dict(task=task, sample=sample_index + 1, total=len(rows))
            report["elapsed_seconds"] = time.monotonic()-started
            atomic_json(output, report)
            print(json.dumps(dict(**report["progress"], f1=report["samples"][-1]["official_qa_f1"],
                                  seconds=report["samples"][-1]["seconds"])), flush=True)
    report["status"] = "complete"
    report["summary"] = {task: dict(n=sum(s["task"] == task for s in report["samples"]),
        qa_f1=100 * np.mean([s["official_qa_f1"] for s in report["samples"] if s["task"] == task]))
        for task in tasks}
    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    atomic_json(output, report)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--total-tokens", "--max_length", dest="total_tokens", type=int,
                   required=True, choices=(8192, 16384))
    p.add_argument("--capacity-blocks", type=int, default=50, choices=(50,))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tasks", nargs="+", choices=DEFAULT_TASKS)
    p.add_argument("--max-samples", type=int, default=None,
                   help="debug smoke limit; omit for the complete official test split")
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()
