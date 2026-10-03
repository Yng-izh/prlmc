"""Train a carrier residual on native QA, then test on disjoint native QA.

The context is written once. Training uses native questions from train_row;
evaluation uses a disjoint question set from eval_row over the same context.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.environ.get("PRLMC_TOOLS", "/share/latent/PRLM-C/tools"))

from native_qa_compare import atomic_json, exact_match, native_pairs, split_context


def evaluate(args):
    import pandas as pd
    import torch
    import yaml
    from methods.prlmc.agent import get_backbone
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory
    from methods.prlmc.probes import Probe, train_carrier_compactor
    from methods.prlmc.consolidate import CarrierResidualCompactor
    from utils.templates import get_template

    frame = pd.read_parquet(args.dataset)
    train_record = frame.iloc[args.train_row]
    eval_record = frame.iloc[args.eval_row]
    train_context = str(train_record["context"])
    eval_context = str(eval_record["context"])
    if train_context != eval_context:
        raise ValueError("train/eval rows must share the exact context for this warmup")

    train_pairs = native_pairs(train_record)
    eval_pairs = native_pairs(eval_record)
    overlap = {p["question"] for p in train_pairs} & {p["question"] for p in eval_pairs}
    if overlap:
        raise ValueError("train/eval question sets overlap: %d" % len(overlap))
    if args.train_limit <= 0 or args.train_limit > len(train_pairs):
        raise ValueError("invalid --train-limit")
    train_pairs = train_pairs[: args.train_limit]

    torch.manual_seed(args.seed)
    model, tokenizer = get_backbone(args.model, "float16", args.device)
    model.eval()
    model.requires_grad_(False)
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config = dict(raw["prlmc"])
    config.update(
        device=args.device, dtype="float16", seed=args.seed,
        mode="ours", write_mode="append", gate_mode="ones", tau_alloc=-1.0,
        projection="none", anchor_source="first_chunk", trigger="none",
        compact_mode="carrier_residual", carrier_ckpt=None,
        basis_mode="none", functional_control="none", compact_probe=False,
        stats_dir=None, active_memory_mask=True,
        carrier_gamma_max=args.gamma_max,
    )
    manager = ProtectedResidualMemory(model, tokenizer, PRLMCConfig.from_dict(config))
    memorize = get_template(args.sub_dataset, "memorize", "prlmc")
    query = get_template(args.sub_dataset, "query", "prlmc")
    render = lambda chunk: memorize.format(context=chunk, time_stamp="")
    count = lambda text: len(tokenizer(text, add_special_tokens=False).input_ids)
    chunks = split_context(train_context, count, render, args.chunk_tokens)

    for index, chunk in enumerate(chunks):
        text = render(chunk)
        write = manager.initialize_anchor(text, max_tokens=args.chunk_tokens) if index == 0 else manager.write(text, max_tokens=args.chunk_tokens)
        if write.get("dropped") or write.get("evicted_blocks") or write.get("compacted"):
            raise RuntimeError("context write was not lossless at chunk %d: %r" % (index, write))
    if manager.used < 2 * manager.block:
        raise RuntimeError("context is too short for functional warmup")

    def generate(question):
        prompt = query.format(question=question)
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
        with torch.inference_mode():
            output = model.generate(
                input_ids=ids, do_sample=False, max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )
        return tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)

    train_probes = [
        Probe(question=p["question"], answers=p["answers"], split="train",
              prompt=query.format(question=p["question"]))
        for p in train_pairs
    ]
    # The actual compaction is half of the oldest working memory, excluding newest.
    protect = max(1, manager.newest_width)
    window = min(manager.used - protect, max(manager.cfg.compact_min_slots, manager.used // 2))
    window = max(manager.block, (window // manager.block) * manager.block)
    window = min(window, manager.used - protect)
    target = max(manager.block, (int(window * args.ratio) // manager.block) * manager.block)
    if target >= window:
        raise RuntimeError("invalid native warmup shape %d -> %d" % (window, target))
    window_blocks = window // manager.block
    target_blocks = target // manager.block

    compactor = CarrierResidualCompactor(
        num_layers=manager.num_layers, hidden=manager.hidden, block=manager.block,
        rank=int(config.get("carrier_rank", 32)), gamma_max=float(args.gamma_max),
    ).to(device=args.device, dtype=torch.float32)
    optimiser = torch.optim.AdamW(compactor.parameters(), lr=args.lr)
    warmup = train_carrier_compactor(
        manager, train_probes, steps=args.steps, lr=args.lr, stage="B",
        window_blocks=window_blocks, target_blocks=target_blocks,
        max_probe_tokens=args.max_probe_tokens, use_amp=False,
        compactor=compactor, optimiser=optimiser,
        ckpt_out=args.ckpt,
        teacher_cache_key=("native", args.train_row, args.eval_row,
                          hashlib.sha256(train_context.encode()).hexdigest(), args.train_limit),
    )
    compactor.eval()
    compactor.trained = True
    manager._carrier = compactor
    manager.compactor = compactor
    manager.carrier_compactor = compactor

    pairs = []
    for pair in eval_pairs:
        prediction = generate(pair["question"])
        pairs.append({**pair, "full_prediction": prediction,
                      "full_correct": exact_match(prediction, pair["answers"])})
    full_count = sum(p["full_correct"] for p in pairs)
    report = manager.compact_with_report(reason="native_qa_functional_warmup", ratio=args.ratio)
    if report.get("freed_slots", 0) <= 0:
        raise RuntimeError("trained compactor did not compact: %r" % report)
    for pair in pairs:
        prediction = generate(pair["question"])
        pair.update(compressed_prediction=prediction,
                    compressed_correct=exact_match(prediction, pair["answers"]))
    compressed_count = sum(p["compressed_correct"] for p in pairs)
    lost = sum(p["full_correct"] and not p["compressed_correct"] for p in pairs)
    gained = sum((not p["full_correct"]) and p["compressed_correct"] for p in pairs)
    result = {
        "status": "completed", "train_row": args.train_row, "eval_row": args.eval_row,
        "train_questions": args.train_limit, "eval_questions": len(eval_pairs),
        "context_sha256": hashlib.sha256(train_context.encode()).hexdigest(),
        "context_chunks": len(chunks), "train_eval_question_overlap": len(overlap),
        "protocol": "native_questions_train_disjoint_native_questions_eval",
        "window_blocks": window_blocks, "target_blocks": target_blocks,
        "ratio": args.ratio, "seed": args.seed, "warmup": warmup,
        "compaction_report": report, "pairs": pairs,
        "summary": {"n": len(pairs), "full_correct": full_count,
                     "compressed_correct": compressed_count,
                     "full_em": full_count / len(pairs),
                     "compressed_em": compressed_count / len(pairs),
                     "lost": lost, "gained": gained,
                     "net_change": gained - lost},
    }
    atomic_json(args.out, result)
    if args.ckpt and not os.path.exists(args.ckpt):
        raise RuntimeError("checkpoint was not written: %s" % args.ckpt)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="/share/latent/MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet")
    parser.add_argument("--train-row", type=int, default=0)
    parser.add_argument("--eval-row", type=int, default=4)
    parser.add_argument("--sub-dataset", default="factconsolidation_mh_262k")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="/share/latent/MemoryLLM-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-limit", type=int, default=24)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma-max", type=float, default=0.05)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--max-probe-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = evaluate(args)
    print(json.dumps({"status": result["status"], "summary": result["summary"], "warmup": result["warmup"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
