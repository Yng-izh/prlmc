"""Paired native QA evaluation. Import and --help never load a GPU model.

Uses original context/questions/answer aliases, with deterministic generation.
The uncompacted arm is valid only when EVERY context chunk was retained without
an eviction, automatic compaction, dropped write, or token truncation. Capacity
failure is recorded as such, never reported as a full-memory accuracy result.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import string
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def native_pairs(record):
    questions, answers = list(record["questions"]), list(record["answers"])
    if len(questions) != len(answers) or not questions:
        raise ValueError("questions/answers must be nonempty and aligned")
    metadata = record.get("metadata") or {}
    ids = metadata.get("qa_pair_ids")
    if ids is not None and len(ids) != len(questions):
        raise ValueError("qa_pair_ids must align with questions")
    pairs = []
    for index, (question, answer) in enumerate(zip(questions, answers)):
        aliases = [answer] if isinstance(answer, str) else list(answer)
        if not isinstance(question, str) or not question.strip():
            raise ValueError("invalid native question")
        if not aliases or any(not isinstance(a, str) or not a.strip() for a in aliases):
            raise ValueError("invalid native answer aliases")
        pairs.append({"index": index, "id": str(ids[index]) if ids is not None else str(index),
                      "question": question, "answers": aliases})
    return pairs


def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def exact_match(prediction, aliases):
    # Named explicitly: this is normalized alias EM, not the benchmark's
    # official metric unless its scorer is independently verified equivalent.
    return any(normalize(prediction) == normalize(alias) for alias in aliases)


def split_context(context, count_tokens, render, max_tokens):
    """Lossless character slices; budget includes the memorization template."""
    if not context or max_tokens < 1:
        raise ValueError("nonempty context and positive budget required")
    if count_tokens(render("")) >= max_tokens:
        raise ValueError("memorization template exhausts the token budget")
    chunks, offset = [], 0
    while offset < len(context):
        # A bounded search remains lossless; chunks need not be maximal.
        lo, hi = offset + 1, min(len(context), offset + max_tokens * 8)
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            if count_tokens(render(context[offset:mid])) <= max_tokens:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        if best is None:
            raise ValueError("no context character fits the token budget")
        chunk = context[offset:best]
        if count_tokens(render(chunk)) > max_tokens:
            raise AssertionError("chunk token budget violated")
        chunks.append(chunk)
        offset = best
    assert "".join(chunks) == context
    return chunks


def summarize(pairs):
    if not pairs:
        raise ValueError("empty evaluation")
    total = len(pairs)
    full = sum(p["full_correct"] for p in pairs)
    compressed = sum(p["compressed_correct"] for p in pairs)
    lost = sum(p["full_correct"] and not p["compressed_correct"] for p in pairs)
    gained = sum(not p["full_correct"] and p["compressed_correct"] for p in pairs)
    return {"n": total, "full_correct": full, "compressed_correct": compressed,
            "full_em": full / total, "compressed_em": compressed / total,
            "lost": lost, "gained": gained,
            "retained": full - lost, "net_change": gained - lost}


def atomic_json(path, payload):
    # Serialize first: a non-serializable field never truncates an old result.
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def evaluate(args):
    # Heavy imports live here: CPU tests/--help cannot trigger S0 GPU selection.
    import pandas as pd
    import torch
    import yaml
    from methods.prlmc.agent import get_backbone
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory
    from utils.templates import get_template

    if not 0 < args.ratio < 1:
        raise ValueError("--ratio must be between zero and one")
    if args.arm not in ("carrier_residual", "select_blocks", "full_rewrite"):
        raise ValueError("unknown arm")
    if args.arm == "carrier_residual" and not args.carrier_ckpt:
        raise ValueError("trained carrier comparison requires --carrier-ckpt")
    frame = pd.read_parquet(args.dataset)
    record = frame.iloc[args.row]
    context = record["context"]
    if not isinstance(context, str):
        raise ValueError("context must be original dataset text")
    pairs = native_pairs(record)
    if args.limit_questions is not None:
        if args.limit_questions <= 0:
            raise ValueError("--limit-questions must be positive")
        pairs = pairs[:args.limit_questions]
    torch.manual_seed(args.seed)
    model, tokenizer = get_backbone(args.model, "float16", args.device)
    model.eval()
    model.requires_grad_(False)
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config = dict(raw["prlmc"])
    # full_rewrite = MemoryLLM 原生行为：每写一步整池重编码，无选择无压缩。
    # compact_mode=none 仅占位；该臂只答一次题，永不到达压缩分支。
    if args.arm == "full_rewrite":
        config.update(device=args.device, dtype="float16", seed=args.seed,
                      mode="full_rewrite", write_mode="append", gate_mode="ones",
                      tau_alloc=-1.0, projection="none", anchor_source="none",
                      trigger="none", compact_mode="none", carrier_ckpt=None,
                      basis_mode="none", functional_control="none", compact_probe=False,
                      stats_dir=None, active_memory_mask=False)
    else:
        config.update(device=args.device, dtype="float16", seed=args.seed,
                      mode="ours", write_mode="append", gate_mode="ones", tau_alloc=-1.0,
                      projection="none", anchor_source="first_chunk", trigger="none",
                      compact_mode=args.arm, carrier_ckpt=args.carrier_ckpt,
                      basis_mode="none", functional_control="none", compact_probe=False,
                      stats_dir=None, active_memory_mask=True)
    manager = ProtectedResidualMemory(model, tokenizer, PRLMCConfig.from_dict(config))
    memorize = get_template(args.sub_dataset, "memorize", "prlmc")
    query = get_template(args.sub_dataset, "query", "prlmc")
    render = lambda chunk: memorize.format(context=chunk, time_stamp="")
    count = lambda text: len(tokenizer(text, add_special_tokens=False).input_ids)
    chunks = split_context(context, count, render, args.chunk_tokens)
    result = {"status": "preparing", "dataset": args.dataset, "row": args.row,
              "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
              "context_chars": len(context), "context_chunks": len(chunks),
              "native_questions_total": len(record["questions"]),
              "evaluated_questions": len(pairs), "metric": "normalized_alias_exact_match",
              "protocol": "complete_context_then_one_manual_compaction",
              "arm": args.arm, "carrier_ckpt": args.carrier_ckpt if args.arm == "carrier_residual" else None,
              "ratio": args.ratio if args.arm != "full_rewrite" else None, "seed": args.seed,
              "writes": [], "pairs": [],
              "query_template": query, "memorize_template": memorize,
              "max_new_tokens": args.max_new_tokens}
    # full_rewrite 臂：每写一步整池重编码（MemoryLLM 原生行为），不存在
    # "未压缩完整记忆"这一参照，容量检查与压缩分支都不适用。
    if args.arm == "full_rewrite":
        for index, chunk in enumerate(chunks):
            text = render(chunk)
            if count(text) > args.chunk_tokens:
                raise AssertionError("write would truncate the original context")
            write = manager.write(text, max_tokens=args.chunk_tokens)
            result["writes"].append(write)
            if write.get("dropped"):
                result.update(status="invalid_full_rewrite", failed_chunk=index)
                atomic_json(args.out, result)
                return result
        result["rewrite_memory_stats"] = manager.stats()

        def generate(question):
            prompt = query.format(question=question)
            ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
            with torch.inference_mode():
                output = model.generate(input_ids=ids, do_sample=False,
                                        max_new_tokens=args.max_new_tokens,
                                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                                        else tokenizer.eos_token_id)
            return tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)

        for pair in pairs:
            prediction = generate(pair["question"])
            result["pairs"].append({**pair, "rewrite_prediction": prediction,
                                    "rewrite_correct": exact_match(prediction, pair["answers"])})
        n_correct = sum(p["rewrite_correct"] for p in result["pairs"])
        result.update(status="completed", summary={"n": len(result["pairs"]),
                       "rewrite_correct": n_correct,
                       "rewrite_em": n_correct / len(result["pairs"])})
        atomic_json(args.out, result)
        return result
    # Conservative append-only admission: each subsequent injection can occupy
    # one full block. Do not silently evict old facts from the 'full' arm.
    if (len(chunks) - 1) * manager.block > manager.layout.working_capacity:
        result.update(status="capacity_exceeded", working_capacity=manager.layout.working_capacity,
                      required_working_slots_upper_bound=(len(chunks) - 1) * manager.block,
                      reason="complete context exceeds conservative append-only admission")
        atomic_json(args.out, result)
        return result
    for index, chunk in enumerate(chunks):
        text = render(chunk)
        if count(text) > args.chunk_tokens:
            raise AssertionError("write would truncate the original context")
        write = (manager.initialize_anchor(text, max_tokens=args.chunk_tokens) if index == 0
                 else manager.write(text, max_tokens=args.chunk_tokens))
        result["writes"].append(write)
        if write.get("dropped") or write.get("evicted_blocks") or write.get("compacted"):
            result.update(status="invalid_full_memory", failed_chunk=index)
            atomic_json(args.out, result)
            return result
    if manager.evicted_blocks or manager.compaction_count or manager.skipped_chunks:
        raise RuntimeError("full-memory integrity check failed")
    result["full_memory_stats"] = manager.stats()

    def generate(question):
        prompt = query.format(question=question)
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
        with torch.inference_mode():
            output = model.generate(input_ids=ids, do_sample=False,
                                    max_new_tokens=args.max_new_tokens,
                                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                                    else tokenizer.eos_token_id)
        return tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)

    for pair in pairs:
        prediction = generate(pair["question"])
        result["pairs"].append({**pair, "full_prediction": prediction,
                               "full_correct": exact_match(prediction, pair["answers"])})
    result["status"] = "full_scored"
    atomic_json(args.out, result)
    report = manager.compact_with_report(reason="native_qa_paired", ratio=args.ratio)
    result["compaction_report"] = report
    if report.get("freed_slots", 0) <= 0:
        result["status"] = "compaction_skipped"
        atomic_json(args.out, result)
        return result
    result["compressed_memory_stats"] = manager.stats()
    for pair in result["pairs"]:
        prediction = generate(pair["question"])
        pair.update(compressed_prediction=prediction,
                    compressed_correct=exact_match(prediction, pair["answers"]))
    result.update(status="completed", summary=summarize(result["pairs"]))
    atomic_json(args.out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="/share/latent/MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet")
    parser.add_argument("--row", type=int, default=3)
    parser.add_argument("--sub-dataset", default="factconsolidation_mh_262k")
    parser.add_argument("--config", default=str(ROOT / "configs/agent_conf/prlmc_e6.yaml"))
    parser.add_argument("--model", default="/share/latent/MemoryLLM-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--arm", choices=["select_blocks", "carrier_residual", "full_rewrite"], default="select_blocks")
    parser.add_argument("--carrier-ckpt")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit-questions", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="/tmp/native_qa_pair.json")
    result = evaluate(parser.parse_args())
    print(json.dumps({"status": result["status"], "summary": result.get("summary")}))
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
