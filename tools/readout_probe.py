"""Paired readout-probe evaluation: A (instruction), B (few-shot), C (attn gain).

Both arms score the SAME row's questions over the SAME pool:

* full:   the lossless uncompressed pool (append-only, no compaction ever ran)
* fixed:  the compressed pool produced by ONE native_qa_compare compaction,
          i.e. exactly the pool whose row-4 EM the readout analysis scored.

The probe (instruction / few-shot / attn-scale) is applied identically to both
arms, so `lost` isolates compression loss and the full-vs-baseline gap
isolates the readout probe's effect on the uncompressed arm. Officials metrics
(exact_match / substring_exact_match via two-tier max) are recomputed from raw
predictions with the benchmark's own normalize/parse rules, because the driver
metric in native_qa_compare is a placeholder.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import string
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.environ.get("PRLMC_TOOLS", "/share/latent/PRLM-C/tools"))

from native_qa_compare import atomic_json, native_pairs, split_context  # noqa: E402


# --- official MemoryAgentBench scorer (factconsolidation -> default_post_process)

def normalize_answer(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def parse_output(text, answer_prefix="Answer:"):
    patterns = [
        re.compile(f"(?:{answer_prefix})(.*)(?:\n|$)", flags=re.IGNORECASE),
        re.compile(r"(?:^)(.*)(?:\n|$)"),
    ]
    for pattern in patterns:
        m = pattern.search(text)
        if m:
            return re.sub(f"^{re.escape(answer_prefix)}", "", m[1].strip(),
                          flags=re.IGNORECASE).strip()
    return None


def official_score(prediction, answers):
    """Two-tier max over (raw, parsed) exactly like default_post_process."""
    def metrics(pred):
        return {
            "em": any(normalize_answer(pred) == normalize_answer(a) for a in answers),
            "sub": any(normalize_answer(a) in normalize_answer(pred) for a in answers),
        }
    out = metrics(prediction)
    parsed = parse_output(prediction)
    if parsed is not None:
        p = metrics(parsed)
        out = {k: out[k] or p[k] for k in out}
    return out["em"], out["sub"]


# --- prompt variants ---------------------------------------------------------

# B: instruction that fights the prior directly. The baseline query already
# says "only from the knowledge pool rather than the real facts in real world"
# once, inside its knowledge-management framing; this variant restates the
# override as a hard rule the reader must obey after reading the pool, with
# the newest-fact rule called out because the observed failures answer older
# or world-known values.
INSTRUCTION = (
    "Based on the knowledge pool you have memorized, answer the question. "
    "CRITICAL RULE: your answer must come ONLY from the memorized knowledge "
    "pool, NOT from your prior knowledge of the real world. If the two "
    "disagree, the knowledge pool always wins. Facts are serially numbered; "
    "the larger serial number is the newest and overrides older ones. Answer "
    "with the fact itself, a very short phrase, no other words. Do NOT answer "
    "with what is true in the real world unless the pool says so.\n\n"
    "Question: {question}\nAnswer:"
)

# D: few-shot anchoring + stop-token guard. The few-shot format makes the
# model continue generating new "Question: ... Answer: ..." pairs after the
# first answer; those literal "Answer:" lines hijack the official parse tier.
# Stopping at the first newline (via a stop string) keeps the answer and the
# scorer in agreement.
FEWSHOT = (
    "You are a knowledge management system. Each fact in the knowledge pool "
    "is provided with a serial number at the beginning, and the newer fact "
    "has larger serial number. You need to solve the conflicts of facts in "
    "the knowledge pool by finding the newest fact with larger serial "
    "number. Your answer must come ONLY from the knowledge pool, never from "
    "real-world knowledge; when they disagree, the knowledge pool wins. "
    "Give only the short answer, nothing else.\n\n"
    "Example 1:\n"
    "Question: What is the sport that Blair Walsh plays in which he is known "
    "for his position on the team?\n"
    "Answer: rugby\n\n"
    "Example 2:\n"
    "Question: Which country does the current CEO of Twitter hold citizenship of?\n"
    "Answer: France\n\n"
    "Example 3:\n"
    "Question: What is the country of citizenship of the author of \"Starship Troopers\"?\n"
    "Answer: United Kingdom\n\n"
    "Question: {question}\n"
    "Answer:"
)


def build_prompt(mode, base_query, question):
    if mode in ("baseline", "attn_scale"):
        return base_query.format(question=question)
    if mode == "instruction":
        return INSTRUCTION.format(question=question)
    if mode in ("fewshot", "fewshot_attn"):
        return FEWSHOT.format(question=question)
    raise ValueError("unknown readout mode %r" % mode)


def evaluate(args):
    import pandas as pd
    import torch
    import yaml
    from methods.prlmc.agent import get_backbone
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory
    from utils.templates import get_template

    if args.mode == "attn_scale" and float(args.attn_scale) == 1.0:
        raise ValueError("attn_scale mode requires a scale != 1.0")
    if args.mode not in ("attn_scale", "fewshot_attn") and args.attn_scale is not None:
        raise ValueError("--attn-scale only applies to attn_scale/fewshot_attn modes")
    scale = float(args.attn_scale) if args.attn_scale is not None else None
    if args.mode == "fewshot_attn" and scale is None:
        raise ValueError("fewshot_attn mode requires --attn-scale")

    frame = pd.read_parquet(args.dataset)
    record = frame.iloc[args.row]
    context = str(record["context"])
    pairs = native_pairs(record)
    torch.manual_seed(args.seed)
    model, tokenizer = get_backbone(args.model, "float16", args.device)
    model.eval()
    model.requires_grad_(False)
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config = dict(raw["prlmc"])
    config.update(device=args.device, dtype="float16", seed=args.seed,
                  mode="ours", write_mode="append", gate_mode="ones", tau_alloc=-1.0,
                  projection="none", anchor_source="first_chunk", trigger="none",
                  compact_mode=args.compact_mode, carrier_ckpt=args.carrier_ckpt,
                  basis_mode="none", functional_control="none", compact_probe=False,
                  stats_dir=None, active_memory_mask=True)
    manager = ProtectedResidualMemory(model, tokenizer, PRLMCConfig.from_dict(config))
    memorize = get_template(args.sub_dataset, "memorize", "prlmc")
    query = get_template(args.sub_dataset, "query", "prlmc")
    render = lambda chunk: memorize.format(context=chunk, time_stamp="")
    count = lambda text: len(tokenizer(text, add_special_tokens=False).input_ids)
    chunks = split_context(context, count, render, args.chunk_tokens)

    for index, chunk in enumerate(chunks):
        text = render(chunk)
        write = (manager.initialize_anchor(text, max_tokens=args.chunk_tokens) if index == 0
                 else manager.write(text, max_tokens=args.chunk_tokens))
        if write.get("dropped") or write.get("evicted_blocks") or write.get("compacted"):
            raise RuntimeError("context write was not lossless at chunk %d: %r" % (index, write))
    if manager.evicted_blocks or manager.compaction_count or manager.skipped_chunks:
        raise RuntimeError("full-memory integrity check failed")

    # Probe C installs the reader-side attention gain on the backbone.
    if args.mode in ("attn_scale", "fewshot_attn") and scale is not None:
        model.memory_attn_scale.fill_(scale)

    def generate(question):
        prompt = build_prompt(args.mode, query, question)
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
        stop = ["\n"] if args.mode in ("fewshot", "fewshot_attn") else None
        with torch.inference_mode():
            output = model.generate(input_ids=ids, do_sample=False,
                                    max_new_tokens=args.max_new_tokens,
                                    stop_strings=stop, tokenizer=tokenizer,
                                    pad_token_id=tokenizer.pad_token_id
                                    if tokenizer.pad_token_id is not None
                                    else tokenizer.eos_token_id)
        return tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)

    def score_arm(key):
        em_sum = sub_sum = 0
        for pair in pairs:
            pred = pair["%s_prediction" % key]
            em, sub = official_score(pred, pair["answers"])
            pair["%s_em" % key] = em
            pair["%s_sub" % key] = sub
            em_sum += em
            sub_sum += sub
        return em_sum, sub_sum

    result = {"status": "full_scored", "dataset": args.dataset, "row": args.row,
              "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
              "context_chars": len(context), "context_chunks": len(chunks),
              "evaluated_questions": len(pairs),
              "protocol": "readout_probe_full_then_one_manual_compaction",
              "mode": args.mode, "attn_scale": args.attn_scale,
              "compact_mode": args.compact_mode, "ratio": args.ratio,
              "carrier_ckpt": args.carrier_ckpt, "seed": args.seed,
              "prompt_variant": (None if args.mode == "attn_scale"
                                else ("instruction" if args.mode == "instruction"
                                      else "fewshot")),
              "fewshot_attn_scale": scale if args.mode == "fewshot_attn" else None,
              "max_new_tokens": args.max_new_tokens, "pairs": pairs,
              "full_memory_stats": manager.stats()}

    for pair in pairs:
        pair["full_prediction"] = generate(pair["question"])
    em, sub = score_arm("full")
    result["summary_full"] = {"n": len(pairs), "em": em, "sub": sub,
                              "em_rate": em / len(pairs), "sub_rate": sub / len(pairs)}
    atomic_json(args.out, result)

    report = manager.compact_with_report(reason="readout_probe", ratio=args.ratio)
    result["compaction_report"] = report
    if report.get("freed_slots", 0) <= 0:
        result["status"] = "compaction_skipped"
        atomic_json(args.out, result)
        return result
    result["compressed_memory_stats"] = manager.stats()
    for pair in pairs:
        pair["compressed_prediction"] = generate(pair["question"])
    em, sub = score_arm("compressed")
    full_em = result["summary_full"]["em"]
    full_sub = result["summary_full"]["sub"]
    compressed_em, compressed_sub = em, sub
    lost = sum(p["full_em"] and not p["compressed_em"] for p in pairs)
    gained = sum((not p["full_em"]) and p["compressed_em"] for p in pairs)
    result["summary_compressed"] = {"n": len(pairs), "em": em, "sub": sub,
                                    "em_rate": em / len(pairs), "sub_rate": sub / len(pairs)}
    result["summary"] = {"n": len(pairs),
                         "full_em": full_em, "full_sub": full_sub,
                         "compressed_em": compressed_em, "compressed_sub": compressed_sub,
                         "lost": lost, "gained": gained, "net_change": gained - lost}
    result["status"] = "completed"
    atomic_json(args.out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="/share/latent/MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet")
    parser.add_argument("--row", type=int, default=4)
    parser.add_argument("--sub-dataset", default="factconsolidation_mh_262k")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="/share/latent/MemoryLLM-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=["baseline", "instruction", "fewshot", "attn_scale", "fewshot_attn"], required=True)
    parser.add_argument("--attn-scale", type=float)
    parser.add_argument("--compact-mode", default="select_blocks")
    parser.add_argument("--carrier-ckpt")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    seed_default = 0
    parser.add_argument("--seed", type=int, default=seed_default)
    parser.add_argument("--out", default="/tmp/readout_probe.json")
    args = parser.parse_args()
    result = evaluate(args)
    print(json.dumps({"status": result["status"], "summary": result.get("summary")}, ensure_ascii=False))
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
