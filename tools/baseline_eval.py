#!/usr/bin/env python3
"""Unified entrypoints for the external baselines of Table 1.

Methods
  streamingllm  attention sinks + recent window over a plain transformers reader
                (pure DynamicCache manipulation, no extra dependency)
  kivi          official jy-yuan/KIVI 2-bit KV quantization
                (references/baselines/kivi/models/llama_kivi.py + quant/ triton kernels)
  mplus         official wangyu-ustc/MemoryLLM mplus branch: native 50-block STM +
                capacity-capped LTM with chunk retrieval before answering
  restkv        NOT runnable without flash_attn (its only implementation is a
                flash-attn2 forward monkeypatch) -> refuses explicitly

Scoring and data loading reuse tools/longbench_memoryllm_eval.py, so every method
sees the same official prompts, the same truncation rule and the same QA F1.

Byte budget: the unified historical-memory budget is 3.125 GiB. Each method
converts that into its own unit and records the conversion in the protocol block;
nothing is silently "the same number of tokens".
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from longbench_memoryllm_eval import (  # noqa: E402
    DEFAULT_TASKS, MAX_NEW, PROMPTS, atomic_json, digest, load_rows, qa_f1,
    score, seed_everything, sha, truncated_prediction,
)

REF = ROOT / "references/baselines"
BUDGET_GIB = 3.125
LATENT_SLOTS = 12800          # 50 blocks x 256, MemoryLLM native capacity
WRITE_CHUNK = 512


# --------------------------------------------------------------------------
# budget accounting
# --------------------------------------------------------------------------
def kv_bytes_per_token(config, bits):
    """Bytes of KV state per token for a decoder with this config."""
    layers = config.num_hidden_layers
    kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    return layers * 2 * kv_heads * head_dim * bits / 8.0


def budget_record(method, config, extra):
    record = dict(method=method, budget_gib=BUDGET_GIB,
                  budget_bytes=int(BUDGET_GIB * 2 ** 30),
                  latent_slots_equivalent=LATENT_SLOTS, **extra)
    if config is not None:
        record["kv_bytes_per_token_fp16"] = kv_bytes_per_token(config, 16)
        for bits in (2, 4):
            record[f"kv_bytes_per_token_{bits}bit"] = kv_bytes_per_token(config, bits)
            record[f"tokens_at_budget_{bits}bit"] = int(
                BUDGET_GIB * 2 ** 30 / kv_bytes_per_token(config, bits))
    return record


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def build_ids(tokenizer, task, row, max_tokens, mode):
    prompt = PROMPTS[task].format(context=row["context"], input=row["input"])
    ids = tokenizer(prompt, add_special_tokens=False, truncation=False).input_ids
    if len(ids) <= max_tokens:
        return prompt, ids
    if mode == "middle":                      # official LongBench truncation
        half = max_tokens // 2
        return prompt, ids[:half] + ids[-half:]
    return prompt, ids[-max_tokens:]


def write_json_report(path, schema, protocol, identity, samples, summary, extra):
    report = dict(schema=schema, status="complete", protocol=protocol,
                  identity=identity, samples=samples, summary=summary,
                  training_performed=False, optimizer_created=False, **extra)
    atomic_json(path, report)
    return report


# --------------------------------------------------------------------------
# StreamingLLM: sinks + recent window, no extra dependency
# --------------------------------------------------------------------------
def streamingllm_generate(model, tokenizer, ids, max_new_tokens, n_sink, window, device):
    import torch
    eos = [tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]]
    cache, produced = None, []
    current = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_ids=current, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        for layer in range(len(cache.key_cache)):
            keys, values = cache.key_cache[layer], cache.value_cache[layer]
            if keys.shape[-2] > window:
                cache.key_cache[layer] = torch.cat(
                    [keys[:, :, :n_sink], keys[:, :, -(window - n_sink):]], dim=-2)
                cache.value_cache[layer] = torch.cat(
                    [values[:, :, :n_sink], values[:, :, -(window - n_sink):]], dim=-2)
        nxt = int(out.logits[:, -1, :].argmax(dim=-1))
        if nxt in eos:
            break
        produced.append(nxt)
        current = torch.tensor([[nxt]], dtype=torch.long, device=device)
    return tokenizer.decode(produced, skip_special_tokens=True), produced


def run_streamingllm(args, tasks, task_rows):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, local_files_only=True,
        attn_implementation="sdpa").to("cuda:0").eval()
    model.requires_grad_(False)
    max_tokens = args.window + args.max_context_tokens
    protocol = dict(model=args.model_path, method="StreamingLLM",
                    n_sink=args.n_sink, window=args.window,
                    doc_truncation=args.doc_truncation,
                    prompt="official LongBench prompts (tools/longbench_memoryllm_eval.py)",
                    scoring="official QA F1, newline-truncated",
                    do_sample=False, num_beams=1, seed=args.seed,
                    budget=budget_record("StreamingLLM", config,
                                         dict(window_tokens=args.window,
                                              sink_tokens=args.n_sink)))
    samples = []
    for task in tasks:
        rows = task_rows[task][:args.max_samples] if args.max_samples else task_rows[task]
        for index, row in enumerate(rows):
            started = time.monotonic()
            seed_everything(args.seed)
            prompt, ids = build_ids(tokenizer, task, row, max_tokens, args.doc_truncation)
            text, tokens = streamingllm_generate(model, tokenizer, ids, MAX_NEW[task],
                                                 args.n_sink, args.window, model.device)
            samples.append(dict(task=task, sample_index=index, id=row.get("id"),
                                input_tokens=len(ids), prediction=text,
                                output_tokens=tokens,
                                qa_f1=score(truncated_prediction(text), row["answers"]),
                                seconds=time.monotonic() - started))
            print(json.dumps(dict(task=task, sample=index + 1, f1=samples[-1]["qa_f1"])), flush=True)
    return protocol, samples, model


# --------------------------------------------------------------------------
# KIVI
# --------------------------------------------------------------------------
def max_tokens_for(task, args):
    return args.max_context_tokens


def run_kivi(args, tasks, task_rows):
    import torch
    sys.path.insert(0, str(REF / "kivi"))
    from transformers import AutoTokenizer, LlamaConfig
    from models.llama_kivi import LlamaForCausalLM_KIVI
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    config = LlamaConfig.from_pretrained(args.model_path, local_files_only=True)
    config.k_bits = args.k_bits
    config.v_bits = args.v_bits
    config.group_size = args.group_size
    config.residual_length = args.residual_length
    model = LlamaForCausalLM_KIVI.from_pretrained(
        args.model_path, config=config, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, local_files_only=True).to("cuda:0").eval()
    model.requires_grad_(False)
    bits = min(args.k_bits, args.v_bits)
    equal_budget_tokens = int(BUDGET_GIB * 2 ** 30 / kv_bytes_per_token(config, bits))
    protocol = dict(model=args.model_path, method="KIVI",
                    k_bits=args.k_bits, v_bits=args.v_bits,
                    group_size=args.group_size, residual_length=args.residual_length,
                    doc_truncation=args.doc_truncation,
                    prompt="official LongBench prompts (tools/longbench_memoryllm_eval.py)",
                    scoring="official QA F1, newline-truncated",
                    do_sample=False, num_beams=1, seed=args.seed,
                    budget=budget_record("KIVI", config,
                                         dict(equal_budget_tokens=equal_budget_tokens)))
    samples = []
    for task in tasks:
        rows = task_rows[task][:args.max_samples] if args.max_samples else task_rows[task]
        for index, row in enumerate(rows):
            started = time.monotonic()
            seed_everything(args.seed)
            prompt, ids = build_ids(tokenizer, task, row, max_tokens_for(task, args),
                                    args.doc_truncation)
            input_ids = torch.tensor([ids], dtype=torch.long, device="cuda:0")
            with torch.no_grad():
                out = model.generate(input_ids=input_ids,
                                     max_new_tokens=MAX_NEW[task], num_beams=1,
                                     do_sample=False, pad_token_id=tokenizer.pad_token_id)
            text = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
            samples.append(dict(task=task, sample_index=index, id=row.get("id"),
                                input_tokens=len(ids), prediction=text,
                                qa_f1=score(truncated_prediction(text), row["answers"]),
                                seconds=time.monotonic() - started))
            print(json.dumps(dict(task=task, sample=index + 1, f1=samples[-1]["qa_f1"])), flush=True)
    return protocol, samples, model


# --------------------------------------------------------------------------
# M+ : native 50-block STM + capacity-capped LTM
# --------------------------------------------------------------------------
class ChunkStore:
    """LTM side of M+: document chunks scored by lexical overlap at query time."""

    def __init__(self, max_chunks, max_chars):
        self.max_chunks, self.max_chars = max_chunks, max_chars
        self.chunks = []

    def add(self, text):
        self.chunks.append(text)
        while len(self.chunks) > self.max_chunks or sum(map(len, self.chunks)) > self.max_chars:
            self.chunks.pop(0)

    def retrieve(self, question, top_k):
        tokens = set(question.lower().split())
        ranked = sorted(self.chunks,
                        key=lambda c: len(tokens & set(c.lower().split())), reverse=True)
        return ranked[:top_k]


def run_mplus(args, tasks, task_rows):
    import torch
    sys.path.insert(0, str(REF / "memoryllm_mplus"))
    from transformers import AutoTokenizer
    from modeling_mplus import MPlus
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = MPlus.from_pretrained(args.model_path, torch_dtype=torch.float16,
                                  local_files_only=True).to("cuda:0").eval()
    model.requires_grad_(False)
    initial_memory = model.memory.detach().cpu().clone()
    initial_initialized = model.initialized.detach().cpu().clone()
    ltm_chars = int(args.ltm_gib * 2 ** 30 / 2)      # fp16 bytes -> chars ~= bytes/2
    protocol = dict(model=args.model_path, method="M+ (capacity-capped)",
                    stm="native 50-block latent memory, 512-token writes",
                    ltm_chars=ltm_chars, retrieval=args.retrieval_top_k,
                    max_context_tokens=args.max_context_tokens,
                    doc_truncation=args.doc_truncation,
                    prompt="official LongBench prompts (tools/longbench_memoryllm_eval.py)",
                    scoring="official QA F1, newline-truncated",
                    do_sample=False, num_beams=1, seed=args.seed,
                    budget=dict(budget_gib=BUDGET_GIB, stm_slots=LATENT_SLOTS,
                                ltm_gib=args.ltm_gib,
                                note="STM slots + capped LTM text store both counted"))
    samples = []
    for task in tasks:
        rows = task_rows[task][:args.max_samples] if args.max_samples else task_rows[task]
        for index, row in enumerate(rows):
            started = time.monotonic()
            seed_everything(args.seed)
            with torch.no_grad():
                model.memory.copy_(initial_memory)
                model.initialized.copy_(initial_initialized)
            prompt, ids = build_ids(tokenizer, task, row, args.max_context_tokens,
                                    args.doc_truncation)
            query_tokens = min(args.tail_tokens, len(ids))
            write_ids, query_ids = ids[:-query_tokens], ids[-query_tokens:]
            store = ChunkStore(max_chunks=10 ** 6, max_chars=ltm_chars)
            write_count = 0
            for start in range(0, len(write_ids), WRITE_CHUNK):
                chunk = write_ids[start:start + WRITE_CHUNK]
                store.add(tokenizer.decode(chunk, skip_special_tokens=True))
                tensor = torch.tensor([chunk], dtype=torch.long, device=model.memory.device)
                attention = torch.ones((1, model.num_tokens + len(chunk)),
                                       dtype=torch.long, device=tensor.device)
                with torch.no_grad():
                    model.inject_memory(tensor, context_attention_mask=attention,
                                        update_memory=True)
                write_count += 1
            retrieved = store.retrieve(row["input"], args.retrieval_top_k)
            if retrieved:
                extra = tokenizer("\n".join(retrieved), add_special_tokens=False,
                                  truncation=False).input_ids[-query_tokens:]
                query_ids = extra + query_ids
            tensor = torch.tensor([query_ids], dtype=torch.long, device=model.memory.device)
            attention = torch.ones((1, model.num_blocks * model.num_tokens + len(query_ids)),
                                   dtype=torch.long, device=tensor.device)
            with torch.no_grad():
                out = model.generate(input_ids=tensor, attention_mask=attention,
                                     max_new_tokens=MAX_NEW[task], num_beams=1,
                                     do_sample=False, pad_token_id=tokenizer.pad_token_id)
            text = tokenizer.decode(out[0, tensor.shape[1]:], skip_special_tokens=True)
            samples.append(dict(task=task, sample_index=index, id=row.get("id"),
                                input_tokens=len(ids), write_count=write_count,
                                retrieved_chunks=len(retrieved), prediction=text,
                                qa_f1=score(truncated_prediction(text), row["answers"]),
                                seconds=time.monotonic() - started))
            print(json.dumps(dict(task=task, sample=index + 1, f1=samples[-1]["qa_f1"])), flush=True)
    return protocol, samples, model


# --------------------------------------------------------------------------
def evaluate(args):
    if args.method == "restkv":
        raise SystemExit(
            "ReST-KV cannot run here: restkv/monkeypatch.py patches "
            "LlamaFlashAttention2.forward and restkv/*_model.py ship only flash-attn2 "
            "forwards, so flash_attn==2.7.2.post1 is mandatory. Install it on the server "
            "(needs working DNS) or move ReST-KV to the non-equal-budget reference table.")
    tasks = tuple(args.tasks) if args.tasks else DEFAULT_TASKS
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ValueError("output exists; refuse overwrite")
    task_rows = {t: load_rows(args.data_dir, t) for t in tasks}
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    import torch
    seed_everything(args.seed)
    runner = {"streamingllm": run_streamingllm, "kivi": run_kivi,
              "mplus": run_mplus}[args.method]
    protocol, samples, model = runner(args, tasks, task_rows)
    summary = {}
    for task in tasks:
        picked = [s for s in samples if s["task"] == task]
        summary[task] = dict(n=len(picked),
                             qa_f1=100 * float(np.mean([s["qa_f1"] for s in picked])))
    identity = dict(model_path=args.model_path, data_dir=args.data_dir,
                    data_files={t: dict(sha256=sha(Path(args.data_dir) / (t + ".jsonl")),
                                        n=len(task_rows[t])) for t in tasks},
                    entrypoint_sha256=sha(__file__),
                    baseline_source=str(REF / {"kivi": "kivi", "mplus": "memoryllm_mplus",
                                               "streamingllm": "."}[args.method]))
    write_json_report(out, "baseline_kv_results/1", protocol, identity, samples, summary,
                      dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           full_test_split=args.max_samples is None))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True,
                   choices=("streamingllm", "kivi", "mplus", "restkv"))
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--tasks", nargs="+", choices=DEFAULT_TASKS)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--doc-truncation", default="middle", choices=("tail", "middle"))
    p.add_argument("--max-context-tokens", type=int, default=32768)
    p.add_argument("--tail-tokens", type=int, default=2048)
    p.add_argument("--window", type=int, default=4096, help="StreamingLLM window")
    p.add_argument("--n-sink", type=int, default=4, help="StreamingLLM sink tokens")
    p.add_argument("--k-bits", type=int, default=2)
    p.add_argument("--v-bits", type=int, default=2)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--residual-length", type=int, default=128)
    p.add_argument("--ltm-gib", type=float, default=1.0)
    p.add_argument("--retrieval-top-k", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()
