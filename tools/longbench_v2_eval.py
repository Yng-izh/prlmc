#!/usr/bin/env python3
"""LongBench v2 evaluation of native MemoryLLM-8B (main-table row 1).

Official code copied verbatim from THUDM/LongBench @ 2e00731f8d0bff23dc4325161044d0ed8af94c1e:
  prompts/0shot.txt -> TEMPLATE_0SHOT
  pred.py           -> build_prompt / extract_answer / max_new_tokens=128 / maxlen
  result.py         -> aggregate() (Overall, Easy, Hard, Short, Medium, Long)

Runtime side reuses this repository's native protocol
(tools/longbench_memoryllm_eval.py, tail2048_native_write_v1): the document is
written into latent memory in 512-token chunks through the unmodified
model.inject_memory(update_memory=True); only the final --tail-tokens tokens of
the prompt (question + choices + answer-format instruction) are read back.

Recorded deviations from the official submission setting:
  * greedy decoding (the official API call uses temperature 0.1);
  * newline added to the EOS set, as everywhere else in this repository.
--total-tokens 128000 reproduces the official maxlen, so no truncation happens
for inputs within the official 120,000-token limit.

Usage:
  PRLMC_READER_PATH=/dev/shm/s1d_reader_fp16 \
  python tools/longbench_v2_eval.py \
      --data-json /data/zhangjiayang/yyz/datasets/LongBench-v2/data.json \
      --model-path /dev/shm/s1d_reader_fp16 \
      --out reports/main_table/memoryllm8b_lbv2_all_128k.json \
      --length all --total-tokens 128000 --cuda-device 1
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from longbench_memoryllm_eval import (  # noqa: E402
    atomic_json, digest, generate, native_write, reset_model, seed_everything,
    sha, truncated_prediction,
)

LONGBENCH_V2_COMMIT = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
MEMORYLLM_COMMIT = "5acaf52d2e60087f6bc38a06063d7558ab9a2c2c"
MAX_NEW_TOKENS = 128      # pred.py, non-CoT branch
OFFICIAL_MAXLEN = 120000  # config/model2maxlen.json, Llama-class entries
LENGTH_GROUPS = ("all", "short", "medium", "long")
FIELDS = ("_id", "domain", "sub_domain", "difficulty", "length", "question",
          "choice_A", "choice_B", "choice_C", "choice_D", "answer", "context")

TEMPLATE_0SHOT = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)"."""


def build_prompt(row, context):
    """Official pred.py prompt construction, unchanged."""
    return (TEMPLATE_0SHOT
            .replace('$DOC$', context.strip())
            .replace('$Q$', row['question'].strip())
            .replace('$C_A$', row['choice_A'].strip())
            .replace('$C_B$', row['choice_B'].strip())
            .replace('$C_C$', row['choice_C'].strip())
            .replace('$C_D$', row['choice_D'].strip()))


def extract_answer(response):
    """Official pred.py extraction: two regexes, no fallback guessing."""
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    match = re.search(r'The correct answer is ([A-D])', response)
    return match.group(1) if match else None


def truncate_ids(ids, budget, mode):
    """Official pred.py middle truncation (ids[:half] + ids[-half:]) or tail."""
    if len(ids) <= budget:
        return ids
    if mode == 'middle':
        half = budget // 2
        return ids[:half] + ids[-half:]
    return ids[-budget:]


def load_rows(path, length_group):
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(raw, dict):
        raw = raw.get('data') or raw.get('train') or []
    rows = []
    for item in raw:
        if length_group != 'all' and item.get('length') != length_group:
            continue
        rows.append({k: item.get(k) for k in FIELDS})
    if not rows:
        raise ValueError(f"no rows for length group {length_group!r} in {path}")
    bad = [r['_id'] for r in rows
           if not isinstance(r.get('context'), str) or r.get('answer') not in ('A', 'B', 'C', 'D')]
    if bad:
        raise ValueError(f"invalid LongBench v2 rows: {bad[:5]}")
    return rows


def aggregate(samples):
    """Official result.py aggregation, verbatim in shape."""
    easy = hard = short = medium = long = 0
    easy_acc = hard_acc = short_acc = medium_acc = long_acc = 0
    for s in samples:
        acc = int(s['judge'])
        if s['difficulty'] == 'easy':
            easy += 1
            easy_acc += acc
        else:
            hard += 1
            hard_acc += acc
        if s['length'] == 'short':
            short += 1
            short_acc += acc
        elif s['length'] == 'medium':
            medium += 1
            medium_acc += acc
        else:
            long += 1
            long_acc += acc

    def pct(hit, total):
        return round(100.0 * hit / total, 1) if total else None

    n = len(samples)
    return dict(n=n,
                overall=pct(easy_acc + hard_acc, n),
                easy=pct(easy_acc, easy), easy_n=easy,
                hard=pct(hard_acc, hard), hard_n=hard,
                short=pct(short_acc, short), short_n=short,
                medium=pct(medium_acc, medium), medium_n=medium,
                long=pct(long_acc, long), long_n=long)


def evaluate(args):
    if args.tail_tokens >= args.total_tokens:
        raise ValueError("--tail-tokens must be smaller than --total-tokens")
    if args.seed != 0:
        raise ValueError("the fixed evaluation seed is 0")
    rows = load_rows(args.data_json, args.length)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("max-samples must be positive")
        rows = rows[:args.max_samples]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("output exists; refuse overwrite")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
    seed_everything(args.seed)
    from transformers import AutoTokenizer
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = MemoryLLM.from_pretrained(args.model_path, torch_dtype=torch.float16,
                                      attn_implementation="sdpa",
                                      local_files_only=True).to("cuda:0").eval()
    model.requires_grad_(False)
    if int(model.num_blocks) != 50 or int(model.num_tokens) != 256:
        raise ValueError("checkpoint is not a native 50-block MemoryLLM")
    if tuple(model.memory.shape) != (32, 12800, 4096):
        raise ValueError("expected unchanged MemoryLLM-8B native memory shape")
    initial_memory = model.memory.detach().cpu().clone()
    initial_initialized = model.initialized.detach().cpu().clone()

    doc_budget = args.total_tokens - args.tail_tokens
    report = dict(
        schema="longbench_v2_memoryllm_results/1", status="running",
        protocol=dict(
            model="MemoryLLM-8B", capacity_blocks=50,
            latent_tokens_per_block=256, max_write_tokens=512,
            memory_budget_gib=3.125, total_input_tokens=args.total_tokens,
            doc_token_budget=doc_budget, direct_query_tokens=args.tail_tokens,
            official_maxlen=OFFICIAL_MAXLEN,
            doc_truncation=args.doc_truncation,
            max_new_tokens=MAX_NEW_TOKENS, seed=args.seed, do_sample=False,
            num_beams=1, training=False, benchmark="LongBench v2",
            length_group=args.length,
            prompt="official prompts/0shot.txt (no CoT, no RAG, no no-context)",
            scoring="official extract_answer + exact match judge; "
                    "result.py Overall/Easy/Hard/Short/Medium/Long",
            memory_reset="checkpoint values and initialized flag before every sample",
            native_eviction="unmodified model.inject_memory(update_memory=True)",
            deviations=["greedy decoding vs official temperature 0.1",
                        "newline added to the EOS set"]),
        identity=dict(model_path=args.model_path, data_json=args.data_json,
                      data_json_sha256=sha(args.data_json), data_rows=len(rows),
                      longbench_source_commit=LONGBENCH_V2_COMMIT,
                      memoryllm_source_commit=MEMORYLLM_COMMIT,
                      actual_reader_sha256=sha(ROOT / "methods/memoryllm_modeling/modeling_memoryllm.py"),
                      actual_entrypoint_sha256=sha(__file__)),
        runtime=dict(torch=torch.__version__,
                     transformers=__import__("transformers").__version__,
                     cuda=torch.version.cuda, device=args.cuda_device,
                     gpu=torch.cuda.get_device_name(0)),
        full_group=args.max_samples is None, max_samples=args.max_samples,
        samples=[], progress=None)
    started = time.monotonic()
    atomic_json(output, report)

    for index, row in enumerate(rows):
        sample_started = time.monotonic()
        seed_everything(args.seed)
        reset_model(model, initial_memory, initial_initialized,
                    reset_memory=args.reset_memory)
        context_ids = tokenizer(row['context'], add_special_tokens=False,
                                truncation=False).input_ids
        kept = truncate_ids(context_ids, doc_budget, args.doc_truncation)
        prompt = build_prompt(row, tokenizer.decode(kept, skip_special_tokens=True))
        ids = truncate_ids(tokenizer(prompt, add_special_tokens=False,
                                     truncation=False).input_ids,
                           args.total_tokens, args.doc_truncation)
        query_ids = ids[-args.tail_tokens:]
        write_ids = ids[:-args.tail_tokens]
        writes = [write_ids[i:i + 512] for i in range(0, len(write_ids), 512)]
        for chunk in writes:
            native_write(model, chunk)
        prediction, output_ids = generate(model, tokenizer, query_ids, MAX_NEW_TOKENS)
        pred = extract_answer(prediction.strip())
        report["samples"].append(dict(
            id=row['_id'], length=row['length'], difficulty=row['difficulty'],
            domain=row.get('domain'), sub_domain=row.get('sub_domain'),
            context_chars=len(row['context']), context_tokens=len(context_ids),
            context_tokens_dropped=len(context_ids) - len(kept),
            input_tokens=len(ids), prefix_tokens=len(write_ids),
            query_tokens=len(query_ids), write_count=len(writes),
            write_chunk_tokens=[len(x) for x in writes],
            native_evicted_slots_per_layer=256 * len(writes),
            native_capacity_blocks=int(model.num_blocks),
            native_active_slots=int(model.memory.shape[1]),
            seconds=time.monotonic() - sample_started,
            prediction=prediction, truncated_prediction=truncated_prediction(prediction),
            output_tokens=output_ids, answer=row['answer'], pred=pred,
            judge=(pred == row['answer'])))
        report["progress"] = dict(group=args.length, done=index + 1, total=len(rows),
                                  last_judge=report["samples"][-1]["judge"])
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output, report)
        print(json.dumps(dict(done=index + 1, total=len(rows),
                              pred=pred, answer=row['answer'],
                              judge=report["samples"][-1]["judge"],
                              seconds=round(report["samples"][-1]["seconds"], 1))),
              flush=True)

    report["status"] = "complete"
    report["summary"] = aggregate(report["samples"])
    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    atomic_json(output, report)
    s = report["summary"]
    print(f"Model\tOverall\tEasy\tHard\tShort\tMedium\tLong\n"
          f"MemoryLLM-8B\t{s['overall']}\t{s['easy']}\t{s['hard']}\t"
          f"{s['short']}\t{s['medium']}\t{s['long']}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-json", required=True,
                   help="LongBench v2 data.json (official release)")
    p.add_argument("--model-path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--length", default="all", choices=LENGTH_GROUPS,
                   help="official length field; 'all' runs the complete 503-question split")
    p.add_argument("--total-tokens", type=int, default=128000,
                   choices=(8192, 16384, 32768, 128000))
    p.add_argument("--tail-tokens", dest="tail_tokens", type=int, default=2048,
                   choices=(512, 1024, 2048))
    p.add_argument("--doc-truncation", dest="doc_truncation", default="middle",
                   choices=("middle", "tail"),
                   help="middle = official pred.py truncation; tail = keep the end")
    p.add_argument("--reset-memory", action="store_true",
                   help="clear the checkpoint's initial residual pool before every sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None,
                   help="debug smoke limit; omit for the complete official split")
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()
