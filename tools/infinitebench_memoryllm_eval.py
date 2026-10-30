#!/usr/bin/env python3
"""Evaluate the frozen MemoryLLM-8B on the local InfiniteBench book tasks.

Official components are copied verbatim from the InfiniteBench repository pinned
at 51d9b37b0f1790ead936df2243abbf7f0420e439:
  src/prompt.py          -> gpt4_templates for longbook_qa_eng / longbook_choice_eng
  src/eval_utils.py      -> DATA_NAME_TO_MAX_NEW_TOKENS (40 for both tasks),
                            get_answer (choice labels carry both the option text
                            and its letter)
  src/compute_scores.py  -> normalize_answer / f1_score / qa_f1_score and
                            get_score_one_longbook_choice_eng  (En.QA = max-over-
                            references token F1, En.MC = option-letter accuracy)

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
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from longbench_memoryllm_eval import (  # noqa: E402
    atomic_json, digest, native_write, reset_model, seed_everything, sha,
    truncated_prediction,
)

DEFAULT_TASKS = ("longbook_qa_eng", "longbook_choice_eng")
TASK_TITLE = {"longbook_qa_eng": "En.QA", "longbook_choice_eng": "En.MC"}
# Pinned upstream source revisions used for the copied prompts and metric.
INFINITEBENCH_COMMIT = "51d9b37b0f1790ead936df2243abbf7f0420e439"
MEMORYLLM_COMMIT = "5acaf52d2e60087f6bc38a06063d7558ab9a2c2c"
# src/eval_utils.py DATA_NAME_TO_MAX_NEW_TOKENS.
MAX_NEW = {"longbook_qa_eng": 40, "longbook_choice_eng": 40}
# src/prompt.py gpt4_templates (the template used for models without a
# model-specific override).
PROMPTS = {
    "longbook_qa_eng": "Read the book below and answer a question.\n\n{context}\n\n"
                       "Question: {question}\n\nBe very concise.",
    "longbook_choice_eng": "Read the book and answer the question.\n\n{context}\n\n"
                           "Question: {question}\n\nOnly one of the following options "
                           "is correct, tell me the answer using one single letter "
                           "(A, B, C, or D). Don't say anything else.\n"
                           "A. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}",
}


# --------------------------------------------------------------------------
# Official scoring code (src/compute_scores.py), copied verbatim.
# --------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def qa_f1_score(pred: str, ground_truths) -> float:
    """Computes the F1, recall, and precision."""
    f1 = 0
    prec = 0
    recall = 0
    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(pred)
        normalized_ground_truth = normalize_answer(ground_truth)

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        scores = f1_score(prediction_tokens, ground_truth_tokens)
        this_f1, this_prec, this_recall = scores
        f1 = max(f1, this_f1)
        prec = max(prec, this_prec)
        recall = max(recall, this_recall)
    return f1


def get_score_one_longbook_choice_eng(pred, label, model_name=None) -> bool:
    # Just use the first letter as the prediction
    pred = pred.strip()
    pattern = r"\b[A-D]\b(?!.*\b[A-D]\b)"

    match = re.search(pattern, pred)
    if match:
        extracted_pred = match.group(0)
        if extracted_pred in label:
            return True
    if pred == "":
        return False
    if pred[0] in "ABCD":
        return pred[0] in label
    if pred in label:
        return True
    # Find a answer prefix
    for c in ["\n", '"', "'", ".", ",", "?", "!", "{", "}"]:
        pred = pred.replace(c, " ")
    while "  " in pred:
        pred = pred.replace("  ", " ")
    ans_prefixes = [
        "answer is:",
        "answer:",
        "answer is",
        "option is",
    ]
    for prefix in ans_prefixes:
        idx = pred.find(prefix)
        if idx == -1:
            continue
        # The prediction ends with this prefix
        if len(pred) < idx + len(prefix) + 1:
            return False
        after_prefix = pred[idx + len(prefix) + 1:]
        for s in label:
            if after_prefix.startswith(s):
                return True
        return False

    # Finally, just find the first occurrence of A, B, C, or D.
    words = pred.split()
    for word in words:
        if word in "ABCD":
            return word in label
    return False


# --------------------------------------------------------------------------
# Data handling
# --------------------------------------------------------------------------
def parse_field(value):
    """The released jsonl stores lists as strings (e.g. "['Peyton']")."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return [text]
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        return [parsed]
    raise ValueError(f"unsupported field type: {type(value)}")


def get_answer(row, task):
    """src/eval_utils.py get_answer: choice labels carry text and letter."""
    answer = parse_field(row["answer"])
    if task != "longbook_choice_eng":
        return answer
    options = parse_field(row["options"])
    letters = "ABCD"
    if len(answer) == 1 and answer[0] in options:
        return [answer[0], letters[options.index(answer[0])]]
    if len(answer) == 2 and isinstance(answer[1], str) and answer[1] in letters:
        return answer
    raise ValueError(f"unresolvable choice answer: {answer!r} / {options!r}")


def load_rows(data_dir, task):
    path = Path(data_dir) / (task + ".jsonl")
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if not rows:
        raise ValueError(f"empty InfiniteBench file: {path}")
    for row in rows:
        for key in ("context", "input", "answer"):
            if key not in row:
                raise ValueError(f"invalid InfiniteBench row in {path}: missing {key}")
        if task == "longbook_choice_eng" and len(parse_field(row["options"])) != 4:
            raise ValueError(f"choice row without 4 options in {path}")
    return rows


def build_prompt(task, row):
    if task == "longbook_choice_eng":
        options = parse_field(row["options"])
        return PROMPTS[task].format(context=row["context"], question=row["input"],
                                    option_a=options[0], option_b=options[1],
                                    option_c=options[2], option_d=options[3])
    return PROMPTS[task].format(context=row["context"], question=row["input"])


def truncate_ids(ids, total_tokens, doc_truncation):
    if len(ids) <= total_tokens:
        return ids
    if doc_truncation == "middle":
        half = total_tokens // 2
        return ids[:half] + ids[-half:]
    return ids[-total_tokens:]


def split_prompt(tokenizer, task, row, total_tokens, tail_tokens, doc_truncation):
    prompt = build_prompt(task, row)
    ids = tokenizer(prompt, add_special_tokens=False, truncation=False).input_ids
    ids = truncate_ids(ids, total_tokens, doc_truncation)
    query_tokens = min(tail_tokens, len(ids))
    write_ids = ids[:-query_tokens] if query_tokens else ids
    query_ids = ids[-query_tokens:] if query_tokens else []
    return prompt, ids, write_ids, query_ids


def generate(model, tokenizer, query_ids, max_new_tokens):
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
                                top_k=0, pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=eos_ids)
    out = output[0, ids.shape[1]:].detach().cpu().tolist()
    return tokenizer.decode(out, skip_special_tokens=True), out


def score(prediction, labels, task):
    if task == "longbook_choice_eng":
        return 1.0 if get_score_one_longbook_choice_eng(prediction, labels) else 0.0
    return qa_f1_score(prediction, labels)


def build_protocol(args, tasks, drop_memory_per_layer):
    return dict(
        model="MemoryLLM-8B", capacity="native_50_blocks", capacity_blocks=50,
        latent_tokens_per_block=256, max_write_tokens=512,
        total_input_tokens=args.total_tokens,
        direct_query_tokens=args.tail_tokens,
        write_prefix_tokens=args.total_tokens - args.tail_tokens,
        max_new_tokens={t: MAX_NEW[t] for t in tasks}, seed=args.seed,
        do_sample=False, num_beams=1, training=False,
        benchmark="InfiniteBench",
        tasks={t: TASK_TITLE[t] for t in tasks},
        prompt="official src/prompt.py gpt4_templates (no model-specific override)",
        scoring="official src/compute_scores.py: En.QA = max-over-reference token "
                "F1 (normalize_answer), En.MC = option-letter accuracy",
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
    report = dict(
        schema="infinitebench_memoryllm_results/1", status="running",
        protocol=build_protocol(args, tasks, model.drop_memory_per_layer),
        identity=dict(model_path=args.model_path, data_dir=args.data_dir,
            doc_truncation=args.doc_truncation, tail_tokens=args.tail_tokens,
            model_config_sha256=hashlib.sha256(
                (Path(args.model_path) / "config.json").read_bytes()).hexdigest(),
            infinitebench_source_commit=INFINITEBENCH_COMMIT,
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
            prediction, output_ids = generate(model, tokenizer, query_ids, MAX_NEW[task])
            truncated = truncated_prediction(prediction)
            if model.memory.shape[1] != 12800 or int(model.num_blocks) != 50:
                raise ValueError("native capacity changed during evaluation")
            labels = get_answer(row, task)
            report["samples"].append(dict(
                task=task, task_title=TASK_TITLE[task], sample_index=sample_index,
                id=row.get("id"),
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
                labels=labels,
                score=score(prediction, labels, task),
                score_truncated=score(truncated, labels, task)))
            report["progress"] = dict(task=task, sample=sample_index + 1,
                                     total=len(rows))
            report["elapsed_seconds"] = time.monotonic() - started
            atomic_json(output, report)
            print(json.dumps(dict(**report["progress"],
                score=report["samples"][-1]["score_truncated"],
                seconds=report["samples"][-1]["seconds"])), flush=True)
    report["status"] = "complete"
    report["summary"] = {}
    for task in tasks:
        picked = [s for s in report["samples"] if s["task"] == task]
        metric = "accuracy" if task == "longbook_choice_eng" else "qa_f1"
        report["summary"][task] = dict(
            task_title=TASK_TITLE[task], n=len(picked), metric=metric,
            score=100 * float(np.mean([s["score_truncated"] for s in picked])),
            score_raw=100 * float(np.mean([s["score"] for s in picked])))
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
                   help="tail keeps the last total-tokens (frozen PRLM-C protocol)")
    p.add_argument("--reset-memory", action="store_true",
                   help="clear the checkpoint's initial residual pool before every sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tasks", nargs="+", choices=DEFAULT_TASKS)
    p.add_argument("--max-samples", type=int, default=None,
                   help="debug smoke limit; omit for the complete official test split")
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()
