#!/usr/bin/env python3
"""Frozen shared E6d on complete natural EventQA contexts, with honest capacity failures.

This is a one-compaction transfer test. It never silently crops a long context
or calls a cropped window a complete natural scene. EventQA's official primary
score is all-answer-elements recall, rather than fact-consolidation alias EM.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from native_qa_compare import atomic_json, native_pairs, split_context
from s1d_shared_compactor_eval import (COMPARISON_ARMS, _candidate_pool, _generate,
                                     _make_compactor, arm_summary, baseline_indices,
                                     sha, source_manifest, state_digest)

OFFICIAL = ROOT / "reports/s1d_round_20261013/official_source/eval_other_utils.py"
_RECALL_CODE = None


def eventqa_score(prediction, answers):
    if not answers or any(not isinstance(answer, str) or not answer for answer in answers):
        raise ValueError("EventQA requires a nonempty list of required answer elements")
    # Read only the actual upstream recall expression from archived source;
    # avoid importing unrelated nltk/rouge/editdistance dependencies.
    global _RECALL_CODE
    if _RECALL_CODE is None:
        tree = ast.parse(OFFICIAL.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "_process_eventqa_dataset")
        expr = next(node.value for node in function.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "recall_score" for target in node.targets))
        _RECALL_CODE = compile(ast.Expression(expr), str(OFFICIAL), "eval")
    value = eval(_RECALL_CODE,
                 {"prediction": prediction, "answer": answers, "sum": sum, "len": len, "__builtins__": {}})
    return value == 1, value == 1


def context_capacity(chunks, block, working_capacity):
    count = len(chunks) - 1
    if count * block > working_capacity:
        return "capacity_exceeded"
    if count <= 9:
        return "insufficient_blocks_for_9_block_comparison"
    return "eligible"


def validate_natural_freeze(freeze_path, dataset, checkpoint, prompt, model_path, rows):
    frozen = json.loads(Path(freeze_path).read_text())
    manifest = frozen.get("manifest", {})
    protocol = frozen.get("natural_protocol") or {}
    if (frozen.get("status") != "frozen" or frozen.get("all_development_complete") is not True or
            manifest.get("checkpoint_sha256") != sha(checkpoint) or manifest.get("source_manifest") != source_manifest() or
            manifest.get("gamma_max") != .05 or
            manifest.get("model_manifest_sha256") != sha(Path(model_path) / "fp16_cache_manifest.json") or
            protocol.get("dataset_sha256") != sha(dataset) or protocol.get("prompt_sha256") != sha(prompt) or
            not set(rows).issubset(protocol.get("rows", [])) or protocol.get("max_new_tokens") != 32 or
            protocol.get("scorer") != "official eventqa_recall"):
        raise ValueError("natural evaluation differs from frozen method/data/prompt/row protocol")
    return protocol


def run(args):
    out = Path(args.out)
    if out.exists():
        raise ValueError("natural evaluation exists; refuse overwrite")
    validate_natural_freeze(args.freeze, args.dataset, args.checkpoint, args.prompt_file, args.model_path, [args.row])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    os.environ["S0B_GPU_SET"] = "1"
    import pandas as pd
    import torch
    import yaml
    from transformers import AutoTokenizer
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory
    from s1d_experiment1 import memory_digest
    frame = pd.read_parquet(args.dataset)
    if not 0 <= args.row < len(frame):
        raise ValueError("row outside dataset")
    record = frame.iloc[args.row]
    context = record["context"]
    if not isinstance(context, str) or not context:
        raise ValueError("original complete natural context required")
    items = native_pairs(record)
    if len({item["question"] for item in items}) != len(items):
        raise ValueError("duplicate natural question strings require explicit QA-ID adaptation")
    prompt = Path(args.prompt_file).read_text()
    if "{question}" not in prompt:
        raise ValueError("fixed natural prompt must contain {question}")
    # Shape admission uses tokenizer/config only, before loading the 8B model.
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    count = lambda text: len(tokenizer(text, add_special_tokens=False).input_ids)
    chunks = split_context(context, count, lambda text: text, 512)
    config_json = json.loads((Path(args.model_path) / "config.json").read_text())
    block = config_json["num_tokens"]
    working_capacity = config_json["num_blocks"] * block - 512
    status = context_capacity(chunks, block, working_capacity)
    report = dict(status="running" if status == "eligible" else status,
        dataset_sha256=sha(args.dataset), row=args.row, context_sha256=hashlib.sha256(context.encode()).hexdigest(),
        checkpoint_sha256=sha(args.checkpoint), source_manifest=source_manifest(),
        freeze_sha256=sha(args.freeze),
        model_manifest_sha256=sha(Path(args.model_path) / "fp16_cache_manifest.json"),
        official_scorer_sha256=sha(OFFICIAL), prompt_sha256=sha(args.prompt_file),
        protocol=dict(task="EventQA", metric="official eventqa_recall (all required elements)",
            writer="raw natural text; lossless <=512-token chunks", compression="one complete-source compaction to9 blocks",
            gamma_max=.05, seed=0, random_seed=0, arms=COMPARISON_ARMS, max_new_tokens=32, stop_strings=["\n"],
            training=False, checkpoint_selection=False, prompt=prompt),
        context_chunks=len(chunks), written_block_count=len(chunks) - 1,
        complete_context=True, truncation_performed=False, training_performed=False, predictions={})
    atomic_json(out, report)
    if status != "eligible":
        raise ValueError("natural context admission failed: " + status)
    started = time.monotonic()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_num_threads(4)
    raw = yaml.safe_load((ROOT / "configs/agent_conf/prlmc_e6.yaml").read_text())["prlmc"]
    raw.update(device="cuda:0", dtype="float16", active_memory_mask=True, trigger="none", mode="ours",
               write_mode="append", gate_mode="ones", tau_alloc=-1., projection="none", basis_mode="none",
               functional_control="none", compact_probe=False, stats_dir=None, carrier_gamma_max=.05)
    model = MemoryLLM.from_pretrained(args.model_path, torch_dtype=torch.float16,
                                     attn_implementation="sdpa").to("cuda:0").eval()
    model.requires_grad_(False)
    model.initialized.zero_()
    manager = ProtectedResidualMemory(model, tokenizer, PRLMCConfig.from_dict(raw))
    saved = torch.load(args.checkpoint, map_location="cuda:0", weights_only=True)
    if saved.get("evaluation_eligible") is False or saved.get("effective_gamma_max", .05) != .05:
        raise ValueError("ineligible or gamma-mismatched checkpoint")
    compactor = _make_compactor(manager, saved["state_dict"], .05, "cuda:0")
    frozen = state_digest(compactor)
    if saved.get("final_state_sha256", frozen) != frozen:
        raise ValueError("checkpoint state hash mismatch")
    writes = []
    for index, chunk in enumerate(chunks):
        write = manager.initialize_anchor(chunk, max_tokens=512) if index == 0 else manager.write(chunk, max_tokens=512)
        if write.get("dropped") or write.get("evicted_blocks") or write.get("compacted"):
            raise ValueError("natural write changed append-only source")
        writes.append(write)
    if manager.used != (len(chunks) - 1) * block:
        raise ValueError("natural memory block shape changed")
    source_hash = memory_digest(model.memory)
    report.update(writes=writes, source_memory_sha256=source_hash, compactor_state_sha256_before=frozen,
                  arm_metadata={})
    for name in COMPARISON_ARMS:
        with torch.no_grad():
            pool, mask = (None, model._buffers["memory_active_mask"].clone()) if name == "Full" else _candidate_pool(
                manager, compactor, name, "cuda:0", window_blocks=len(chunks) - 1)
        report["arm_metadata"][name] = dict(active_slots=int(mask.sum()))
        if name in ("Recent9", "Random9"):
            report["arm_metadata"][name]["kept_blocks"] = baseline_indices(name, window_blocks=len(chunks) - 1)
        def progress(values):
            report["predictions"][name] = values
            atomic_json(out, report)
        report["predictions"][name] = _generate(model, manager, items, pool, mask, progress,
            prompt_template=prompt, scorer=eventqa_score)
        if memory_digest(model.memory) != source_hash or state_digest(compactor) != frozen:
            raise ValueError("natural evaluation changed frozen memory/module")
        del pool, mask
        torch.cuda.empty_cache()
    # Here em names are compatibility fields; explicitly labeled EventQA recall.
    maps = {name: {question: row["em"] for question, row in values.items()} for name, values in report["predictions"].items()}
    report.update(status="complete", summary=arm_summary(maps), compactor_state_sha256_after=frozen,
                  source_memory_unchanged=True, compactor_unchanged=True,
                  costs=dict(elapsed_seconds=time.monotonic() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    atomic_json(out, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-path", default="/dev/shm/s1d_reader_fp16")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--cuda-device", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise ValueError("natural output exists; refuse overwrite")
    try:
        run(args)
    except Exception as exc:
        if Path(args.out).exists():
            report = json.loads(Path(args.out).read_text())
            if report.get("status") == "running":
                report.update(status="failed", failure=dict(type=type(exc).__name__, message=str(exc)))
                atomic_json(args.out, report)
        raise


if __name__ == "__main__":
    main()
