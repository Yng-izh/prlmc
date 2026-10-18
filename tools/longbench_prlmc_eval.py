#!/usr/bin/env python3
"""LongBench evaluation of the PRLM-C natural-merge (v5 Dsup) method.

Same LongBench protocol as tools/longbench_memoryllm_eval.py
(tail2048_native_write_v1: final 2,048 tokens queried directly, the prefix
written in chunks of at most 512 tokens), with two differences:

1. The writer is the PRLM-C managed streaming path: writes go into a
   20-block working budget inside the checkpoint's physical 50-block pool,
   and a 19->9 block slot merge (the trained Dsup
   `grouped_gated_sparse_slot_v5` module) consolidates the oldest 19 blocks
   into 9 whenever the budget would be exceeded. No native random eviction
   runs: `inject_memory` is called with `update_memory=False` and the delta
   is placed by the manager, then the active mask hides the free region.
2. Generation is unchanged, but both raw QA F1 and the newline-truncated
   QA F1 are recorded per sample: MemoryLLM's decoder keeps generating
   "\n\nQuestion: ..." after the answer, so the raw F1 is depressed by a
   decoding artifact rather than by memory quality. The truncated score
   matches the official LongBench post-processing.

--arm Selection runs the identical protocol with the pure importance-weighted
block selector (no residual merge module), the same comparison axis as the
v5 development stream.

No training, no optimizer, greedy decoding, fixed seed 0.
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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from longbench_memoryllm_eval import (  # noqa: E402
    DEFAULT_TASKS, LONGBENCH_COMMIT, MAX_NEW, MEMORYLLM_COMMIT,
    digest, score, seed_everything, sha, split_prompt, atomic_json,
)
from natural_merge_runtime import (  # noqa: E402
    MergeStream, load_runtime, reset, source_manifest,
)
from public_retention_eval import budget_geometry  # noqa: E402

# transformers>=4.50 moved prepare_inputs_for_generation onto GenerationMixin,
# but peft's LoRA wrapper still probes it on the inner decoder and raises
# AttributeError when MemoryLLM-8B wraps its LlamaModel with lora_config.
# MemoryLLM vendors its OWN copy of LlamaModel (methods/memoryllm_modeling/
# modeling_memoryllm.py), so the shim must target that vendored class, not the
# library's. Generation itself goes through MemoryLLM's own override at
# modeling_memoryllm.py:1212; this shim only satisfies peft's probe.
import importlib
_mllm = importlib.import_module("methods.memoryllm_modeling.modeling_memoryllm")  # noqa: E402
from transformers import GenerationMixin  # noqa: E402
if not hasattr(_mllm.LlamaModel, "prepare_inputs_for_generation"):
    _mllm.LlamaModel.prepare_inputs_for_generation = GenerationMixin.prepare_inputs_for_generation
# The vendored LlamaForCausalLM likewise misses GenerationMixin.generate (and
# its helpers) under transformers 4.57; restore them so MemoryLLM generation
# and peft's wrapper construction both work as they did under 4.48.
# Bind every GenerationMixin member the vendored copy lacks; MemoryLLM keeps
# its own forward/prepare_inputs_for_generation overrides, so only the 4.57
# generation helpers are restored.
import inspect as _inspect
# dunder list: getmembers() exposes object base entries (__class__, __init__
# subclass hooks, ...) whose setattr on the class raises TypeError or breaks
# the MRO; bind only real GenerationMixin API members.
_DUNDERS = {"__class__", "__delattr__", "__dict__", "__dir__", "__doc__",
    "__eq__", "__format__", "__ge__", "__getattribute__", "__gt__",
    "__hash__", "__init__", "__init_subclass__", "__le__", "__lt__",
    "__module__", "__ne__", "__new__", "__reduce__", "__reduce_ex__",
    "__repr__", "__setattr__", "__sizeof__", "__str__", "__subclasshook__",
    "__weakref__", "__annotations__", "__slotnames__", "__firstlineno__",
    "__static_attributes__", "__mro_entries__"}
for _name, _attr in _inspect.getmembers(GenerationMixin):
    if _name in ("forward", "prepare_inputs_for_generation",
                 "_supports_default_dynamic_cache") or _name in _DUNDERS:
        continue
    if not hasattr(_mllm.LlamaForCausalLM, _name):
        # staticmethods must stay static on the subclass, otherwise the
        # instance is injected as the first (expand_size) parameter.
        if isinstance(_inspect.getattr_static(GenerationMixin, _name), staticmethod):
            setattr(_mllm.LlamaForCausalLM, _name, staticmethod(_attr))
        else:
            setattr(_mllm.LlamaForCausalLM, _name, _attr)
# _supports_default_dynamic_cache does cls._is_stateful via __class_getitem__
# style class lookup that resolves to GenerationMixin itself; pin the flag on
# the vendored subclass so the class-level lookup succeeds.
# _is_stateful lives on PreTrainedModel in 4.57 (modeling_utils.py:1941);
# the vendored model predates that base, so pin it directly. Note the earlier
# getattr(GenerationMixin, ...) fallback was wrong — GenerationMixin never
# defines it, which is exactly the AttributeError seen in generate().
if not hasattr(_mllm.LlamaForCausalLM, "_is_stateful"):
    from transformers.modeling_utils import PreTrainedModel
    _mllm.LlamaForCausalLM._is_stateful = getattr(PreTrainedModel, "_is_stateful", False)
# The classmethod _supports_default_dynamic_cache is bound through
# ContinuousMixin in 4.57's GenerationMixin and disappears from the vendored
# class lookup (cls resolves to bare GenerationMixin -> _is_stateful miss);
# rebind it as a real classmethod on the vendored subclass.
if not hasattr(_mllm.LlamaForCausalLM, "_supports_default_dynamic_cache"):
    # Assign the ORIGINAL classmethod descriptor, not a re-wrapped bound
    # function: accessing GenerationMixin._supports_default_dynamic_cache on
    # the class already binds it to GenerationMixin, so classmethod(...__func__)
    # still resolves cls=GenerationMixin inside _is_stateful lookup and raises
    # AttributeError. Using the raw descriptor from __dict__ rebinds cls to
    # the vendored subclass.
    _mllm.LlamaForCausalLM._supports_default_dynamic_cache =         GenerationMixin.__dict__["_supports_default_dynamic_cache"]
# The vendored model never sets self.generation_config (its config ships no
# generation block), which the 4.57 _prepare_generation_config dereferences;
# install a default from the model config like PreTrainedModel.__init__ does.
from transformers import GenerationConfig  # noqa: E402
_orig_mllm_init = _mllm.MemoryLLM.__init__
def _mllm_init_with_gencfg(self, config, *a, **k):
    _orig_mllm_init(self, config, *a, **k)
    if getattr(self, "generation_config", None) is None:
        self.generation_config = GenerationConfig.from_model_config(config)
_mllm.MemoryLLM.__init__ = _mllm_init_with_gencfg

BUDGET_BLOCKS = 20
BUDGET_CHOICES = (8, 10, 12, 20)
WINDOW_BLOCKS = 19
TARGET_BLOCKS = 9
MODEL_PATH = "/share/latent/MemoryLLM-8B"
# v5 architecture string. A v4 checkpoint cannot be evaluated here: the v5
# module changed routing, the gate inputs and the native path, and
# `validate_merge_checkpoint` rejects the retired training record, so the
# checkpoint has to be passed explicitly instead of silently defaulting.
ARCHITECTURE = "grouped_gated_sparse_slot_v5"
METHOD = "PRLM-C natural-merge v5 Dsup"


def truncated_prediction(prediction):
    """Official LongBench post-processing: cut at the first newline."""
    cut = prediction.find("\n")
    return prediction if cut < 0 else prediction[:cut]


def generate(model, tokenizer, query_ids, max_new_tokens):
    ids = torch.tensor([query_ids], dtype=torch.long, device=model.memory.device)
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
    eos_ids = [tokenizer.eos_token_id, newline_id]
    attention = torch.ones((1, model.memory.shape[1] + len(query_ids)),
                           dtype=torch.long, device=ids.device)
    with torch.no_grad():
        output = model.generate(input_ids=ids, attention_mask=attention,
                                max_new_tokens=max_new_tokens, num_beams=1,
                                do_sample=False, temperature=1.0, top_p=1.0, top_k=0,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=eos_ids)
    out = output[0, ids.shape[1]:].detach().cpu().tolist()
    return tokenizer.decode(out, skip_special_tokens=True), out


def run_sample(stream, writes):
    """Write the whole prefix through the streaming path; return events."""
    compression_events = []
    for write_index, chunk in enumerate(writes):
        record = dict(role="write", token_ids=chunk, token_sha256=digest(chunk))
        entry = stream.write(record, write_index)
        if entry is None:
            raise ValueError("budgeted stream refused a write; protocol violation")
        while len(stream.events) > len(compression_events):
            event = stream.events[len(compression_events)]
            fusion = event.get("fusion", {})
            event_entry = dict(
                before_write=event["before_write"],
                source_slots=event["source_slots"],
                target_slots=event["target_slots"],
                remaining_slots=event["remaining_slots"],
                selected_blocks=event["selected_blocks"],
                seconds=event["seconds"],
                gamma=fusion.get("gamma"),
                fusion_source_count=len(fusion.get("fusion_source_indices", [])))
            compression_events.append(event_entry)
            # Plan A Consolidate arm carries its slot-level merge report under
            # "consolidation" (merged_slots/merge_tau/kept_slots); forward it
            # verbatim so the smoke report proves the consolidator fired.
            if event.get("consolidation") is not None:
                event_entry["consolidation"] = event["consolidation"]
    return compression_events


def build_consolidate_merge(merge, arm, manager):
    """--arm Consolidate swaps the (unused) learned module for the zero-parameter
    slot-level consolidator; every other arm keeps the loaded merge untouched."""
    if arm != "Consolidate":
        return merge
    from methods.prlmc.slot_consolidate import SlotConsolidateMerge
    return SlotConsolidateMerge(manager.num_layers, manager.hidden, manager.block)


def evaluate(args):
    if args.total_tokens not in (8192, 16384):
        raise ValueError("--total-tokens must be 8192 or 16384")
    tasks = tuple(args.tasks) if args.tasks else DEFAULT_TASKS
    if not 0 <= args.shard < args.num_shards or args.num_shards < 1:
        raise ValueError("invalid shard/num-shards")
    tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    if not tasks:
        raise ValueError("this shard has no tasks")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max-samples must be positive")
    if args.seed != 0:
        raise ValueError("the fixed evaluation seed is 0")
    if args.arm == "SharedMerge":
        if not args.checkpoint or not Path(args.checkpoint).is_file():
            raise ValueError("--arm SharedMerge needs an existing --checkpoint")
    elif args.arm == "Consolidate":
        if args.checkpoint is not None:
            raise ValueError("--arm Consolidate is training-free; drop --checkpoint")
    elif args.checkpoint is not None:
        raise ValueError("--arm Selection mutates no learned module; drop --checkpoint")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if output.exists():
        if not args.resume:
            raise ValueError("output exists; refuse overwrite (use --resume)")
        prev = json.loads(output.read_text())
        if (prev.get("protocol", {}).get("total_input_tokens") != args.total_tokens
                or prev.get("identity", {}).get("checkpoint") != args.checkpoint
                or prev.get("protocol", {}).get("merge_arm") != args.arm):
            raise ValueError("--resume output was produced by a different protocol/checkpoint")
        done_ids = {(s["task"], s["sample_index"]) for s in prev["samples"]}
        report = prev
    else:
        report = None
    from longbench_memoryllm_eval import load_rows
    task_rows = {t: load_rows(args.data_dir, t) for t in tasks}
    data_identity = {t: dict(sha256=sha(Path(args.data_dir) / (t + ".jsonl")),
                             n=len(task_rows[t])) for t in tasks}
    # NOTE: do NOT overwrite an externally set CUDA_VISIBLE_DEVICES (e.g.
    # CUDA_VISIBLE_DEVICES=9 with --cuda-device 0 selects physical GPU 9);
    # overwriting it here re-exposed all 10 GPUs and landed model weights on
    # busy physical GPU 0 -> OOM. Only set it when the caller left it unset.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.cuda_device))
    seed_everything(args.seed)

    # load_runtime pins the reader fork, the manager configuration
    # (carrier_residual, active mask, k_max = 20 blocks) and validates the
    # checkpoint architecture/training/budget against TRAIN_V4.
    model, tokenizer, manager, merge, initial, cfg, saved = load_runtime(
        MODEL_PATH, checkpoint=args.checkpoint, train=False,
        architecture=ARCHITECTURE if args.checkpoint else None,
        budget_blocks=args.budget_blocks)
    tokenizer.pad_token = tokenizer.eos_token
    if int(model.num_blocks) != 50 or int(model.num_tokens) != 256:
        raise ValueError("expected the native 50-block MemoryLLM reader")
    checkpoint_sha256 = sha(args.checkpoint) if args.checkpoint else None

    if report is None:
        if args.checkpoint and saved is None:
            raise ValueError("checkpoint file did not load")
        report = dict(schema="longbench_prlmc_results/1", status="running",
            protocol=dict(model="MemoryLLM-8B",
                method=("PRLM-C Consolidate (training-free plan A)" if args.arm == "Consolidate"
                        else METHOD if args.checkpoint else "PRLM-C Selection"),
                variant=saved["variant"] if saved else None,
                architecture=ARCHITECTURE if args.checkpoint else None,
                capacity=f"managed_{args.budget_blocks}_block_budget",
                budget_blocks=args.budget_blocks,
                window_blocks=budget_geometry(args.budget_blocks)[0],
                target_blocks=budget_geometry(args.budget_blocks)[1],
                merge_arm=args.arm,
                latent_tokens_per_block=256, max_write_tokens=512,
                total_input_tokens=args.total_tokens,
                direct_query_tokens=2048, write_prefix_tokens=args.total_tokens - 2048,
                max_new_tokens={t: MAX_NEW[t] for t in tasks}, seed=args.seed,
                do_sample=False, num_beams=1, training=False,
                benchmark="LongBench test",
                scoring="official QA F1 (raw) + newline-truncated QA F1",
                prompt_truncation="keep last total_input_tokens; no decode/re-encode",
                input_protocol="tail2048_native_write_v1; same as the native50 baseline",
                injection="inject_memory(update_memory=False); manager places the delta",
                compaction="19->9 block merge before the write that would exceed the budget",
                seed_reset="reset to 0 before each sample",
                memory_reset="checkpoint values, used=0 and full active mask before every sample"),
            identity=dict(model_path=MODEL_PATH, data_dir=args.data_dir,
                checkpoint=args.checkpoint, checkpoint_sha256=checkpoint_sha256,
                source_manifest=source_manifest(),
                longbench_source_commit=LONGBENCH_COMMIT,
                memoryllm_source_commit=MEMORYLLM_COMMIT,
                actual_entrypoint_sha256=sha(__file__),
                data_files=data_identity,
                tokenizer_files={n: sha(Path(MODEL_PATH) / n) for n in
                    ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
                    if (Path(MODEL_PATH) / n).is_file()}),
            runtime=dict(torch=__import__("torch").__version__,
                         transformers=__import__("transformers").__version__,
                         device=args.cuda_device),
            full_test_split=args.max_samples is None, max_samples=args.max_samples,
            training_performed=False, optimizer_created=False,
            tasks=list(tasks), samples=[], progress=None)
    started = time.monotonic()
    atomic_json(output, report)

    for task in tasks:
        rows = task_rows[task]
        if args.task_range is not None:
            rows = rows[args.task_range[0]:args.task_range[1]]
        if args.max_samples is not None:
            rows = rows[:args.max_samples]
        for sample_index, row in enumerate(rows):
            if (task, sample_index) in done_ids:
                continue
            sample_started = time.monotonic()
            seed_everything(args.seed)
            reset(manager, initial)
            # reset() marks the entire physical pool active; hide the unused
            # region so a zero/short prefix stays inside the 20-block budget.
            with torch.no_grad():
                manager._tile_free_region()
                manager._sync_active_mask()
            prompt, all_ids, write_ids, query_ids = split_prompt(
                tokenizer, task, row, args.total_tokens)
            original_tokens = len(tokenizer(prompt, add_special_tokens=False,
                                            truncation=False).input_ids)
            writes = [write_ids[i:i + 512] for i in range(0, len(write_ids), 512)]
            merge = build_consolidate_merge(merge, args.arm, manager)
            stream = MergeStream(manager, merge, args.arm, budget_blocks=args.budget_blocks)
            compression_events = run_sample(stream, writes)
            prediction, output_ids = generate(model, tokenizer, query_ids, MAX_NEW[task])
            truncated = truncated_prediction(prediction)
            active = int(model.memory_active_mask.sum())
            # The pre-registered geometry for budgets 8/10 reads a window one
            # block larger than the budget, so the stream may transiently hold
            # window blocks before the final compact lands back on target.
            _w_b, _t_b = budget_geometry(args.budget_blocks)
            if active != manager.used or active > max(args.budget_blocks, _w_b) * model.num_tokens:
                raise ValueError("active budget violation after writes")
            report["samples"].append(dict(
                task=task, sample_index=sample_index, id=row.get("_id"),
                source_row_sha256=digest(row), context_sha256=digest(row["context"]),
                input_token_sha256=digest(all_ids),
                original_prompt_tokens=original_tokens,
                truncated_tokens=original_tokens - len(all_ids),
                input_tokens=len(all_ids), prefix_tokens=len(write_ids),
                query_tokens=len(query_ids),
                write_count=len(writes), write_chunk_tokens=[len(x) for x in writes],
                compression_count=len(compression_events),
                compression_events=compression_events,
                peak_active_slots=stream.peak,
                final_active_slots=active,
                max_new_tokens=MAX_NEW[task], output_token_count=len(output_ids),
                seconds=time.monotonic() - sample_started,
                prediction=prediction, output_tokens=output_ids, answers=row["answers"],
                official_qa_f1=score(prediction, row["answers"]),
                official_qa_f1_truncated=score(truncated, row["answers"])))
            report["progress"] = dict(task=task, sample=sample_index + 1, total=len(rows))
            report["elapsed_seconds"] = time.monotonic() - started
            atomic_json(output, report)
            print(json.dumps(dict(**report["progress"],
                f1_truncated=report["samples"][-1]["official_qa_f1_truncated"],
                compressions=len(compression_events),
                seconds=report["samples"][-1]["seconds"])), flush=True)
    report["summary"] = {task: dict(
        n=sum(s["task"] == task for s in report["samples"]),
        qa_f1=100 * float(np.mean([s["official_qa_f1_truncated"]
                                   for s in report["samples"] if s["task"] == task])),
        qa_f1_raw=100 * float(np.mean([s["official_qa_f1"]
                                       for s in report["samples"] if s["task"] == task])))
        for task in tasks}
    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["status"] = "complete"
    atomic_json(output, report)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--total-tokens", "--max_length", dest="total_tokens", type=int,
                   default=16384, choices=(8192, 16384),
                   help="input cap; the per-sample write prefix is "
                        "min(cap-2048, sample length). Merges fire only when that "
                        "prefix reaches 20 blocks x 512 = 10240 tokens, so 8192 "
                        "(prefix <= 6144) always has compression_count=0, while "
                        "16384 admits prefixes up to 14336 and fuses only the "
                        "samples whose own context is long enough")
    p.add_argument("--budget-blocks", dest="budget_blocks", type=int,
                   default=BUDGET_BLOCKS, choices=BUDGET_CHOICES,
                   help="managed working budget in native blocks (8/10/12/20); "
                        "lowers k_max so compaction fires more often, e.g. "
                        "repeated 19->9 then 9->4 (budget 8) compressions")
    p.add_argument("--checkpoint", default=None,
                   help="trained Dsup/Dnative checkpoint; required for --arm SharedMerge, "
                        "unused for --arm Selection")
    p.add_argument("--arm", default="SharedMerge",
                   choices=("SharedMerge", "Selection", "Consolidate"),
                   help="SharedMerge = trained Dsup residual merge (our method); "
                        "Selection = same-budget pure block selector control")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tasks", nargs="+", choices=DEFAULT_TASKS)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--task-range", nargs=2, type=int, default=None, metavar=("START","END"),
                   help="evaluate only samples [START,END) of each task")
    p.add_argument("--resume", action="store_true",
                   help="skip samples already present in --out (same tasks/shard)")
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()
