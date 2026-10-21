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

--arm Full is the same-path no-compression reference. It reuses the existing
"Full" streaming arm (`natural_merge_runtime.MergeStream` ->
`public_retention_eval.StreamingMemory.write`, the arm the natural-data training
and evaluation streams already run) instead of adding a second oracle: the
compaction trigger is skipped for this arm, every written block stays resident,
and the capacity limit is the whole native 50-block pool. Nothing is compacted,
so the prefix is read back exactly as written. This is the ceiling a merge
operator is bounded by: a merge can only recover what compression destroyed, so
its effect has to sit inside `Full - Selection` (see tools/ceiling_report.py).

No training, no optimizer, greedy decoding, fixed seed 0.
"""
import argparse
import hashlib
import inspect
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
    MergeStream, critical_manifest, load_runtime, reset, resolve_architecture,
    source_manifest, validate_merge_checkpoint,
)
from public_retention_eval import budget_geometry, compaction_trigger  # noqa: E402
from methods.prlmc.consolidate import _SELECT_LAYER  # noqa: E402

# transformers>=4.50 moved prepare_inputs_for_generation onto GenerationMixin,
# but peft's LoRA wrapper still probes it on the inner decoder and raises
# AttributeError when MemoryLLM-8B wraps its LlamaModel with lora_config.
# MemoryLLM vendors its OWN copy of LlamaModel (methods/memoryllm_modeling/
# modeling_memoryllm.py), so the shim must target that vendored class, not the
# library's. Generation itself goes through MemoryLLM's own override at
# modeling_memoryllm.py:1212; this shim only satisfies peft's probe.
_SHIMS_INSTALLED = False


def install_generation_compatibility():
    """Bind the transformers 4.57 generation API onto the vendored MemoryLLM classes."""
    global _SHIMS_INSTALLED
    if _SHIMS_INSTALLED:
        return
    _SHIMS_INSTALLED = True
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
TAIL_CHOICES = (512, 1024, 2048)
TOTAL_TOKEN_CHOICES = (8192, 16384, 32768)
# The pre-registered tail2048_native_write_v1 protocol is validated for these
# prefixes only; 32768 stays selectable so the CLI reports the limitation
# explicitly instead of silently clamping.
VALIDATED_TOTAL_TOKENS = (8192, 16384)
PHYSICAL_BLOCKS = 50
# The no-compression reference arm. It is the arm name the existing streaming
# runtime already understands (`ARMS` in public_retention_eval): `write` skips
# the compaction trigger for it and its capacity limit is `m.total`, so no new
# oracle code is needed -- only the wiring into this entrypoint.
FULL_ARM = "Full"
# Fields that identify a run. --resume refuses to append to an output whose
# identity differs in any of them (a mixed restart already produced 2 bogus
# samples in the v4 16k g7 run).
IDENTITY_KEYS = ("arm", "checkpoint", "checkpoint_sha256", "architecture", "variant",
                 "total_tokens", "tail_tokens", "budget_blocks", "window_blocks",
                 "target_blocks", "trigger_blocks", "select_layer", "reference_layer",
                 "critical_source_manifest", "code_hash_verified")
# The reader is read from the GPFS FP32 checkpoint by default. On a host that
# has staged the verified FP16 cache in /dev/shm the same weights load in
# seconds instead of ~12 minutes; the staged copy is byte-verified by
# tools/stage_s1d_reader_cache.py, so pointing at it does not change results.
# Override with PRLMC_READER_PATH; the resolved path is recorded in identity.
MODEL_PATH = os.environ.get("PRLMC_READER_PATH") or "/share/latent/MemoryLLM-8B"
# v5 architecture string. A v4 checkpoint cannot be evaluated here: the v5
# module changed routing, the gate inputs and the native path, and
# `validate_merge_checkpoint` rejects the retired training record, so the
# checkpoint has to be passed explicitly instead of silently defaulting.
ARCHITECTURE = "grouped_gated_sparse_slot_v5"
ARCHITECTURE_V6 = "grouped_gated_sparse_slot_v6"
METHOD = "PRLM-C natural-merge v5 Dsup"
METHOD_V6 = "PRLM-C natural-merge v6 DsupV6"


def truncated_prediction(prediction):
    """Official LongBench post-processing: cut at the first newline."""
    cut = prediction.find("\n")
    return prediction if cut < 0 else prediction[:cut]


def parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def validate_geometry(budget_blocks, window_blocks, target_blocks, total_blocks=PHYSICAL_BLOCKS):
    """Reject impossible merge geometry before any model is loaded."""
    if budget_blocks not in BUDGET_CHOICES:
        raise ValueError(f"budget_blocks must be one of {BUDGET_CHOICES}, got {budget_blocks}")
    if not 0 < target_blocks < window_blocks:
        raise ValueError("invalid compression geometry: need 0 < target_blocks < "
                         f"window_blocks, got window={window_blocks} target={target_blocks}")
    if target_blocks > budget_blocks:
        raise ValueError("invalid compression geometry: target_blocks "
                         f"{target_blocks} exceeds budget_blocks {budget_blocks}")
    if window_blocks > total_blocks:
        raise ValueError("invalid compression geometry: window_blocks "
                         f"{window_blocks} exceeds the physical {total_blocks}-block pool")
    return dict(budget_blocks=budget_blocks, window_blocks=window_blocks,
                target_blocks=target_blocks, total_blocks=total_blocks)


def effective_geometry(args, total_blocks=PHYSICAL_BLOCKS):
    """Explicit --window-blocks/--target-blocks win; None falls back to the table."""
    window_blocks, target_blocks = budget_geometry(args.budget_blocks)
    if getattr(args, "window_blocks", None) is not None:
        window_blocks = args.window_blocks
    if getattr(args, "target_blocks", None) is not None:
        target_blocks = args.target_blocks
    validate_geometry(args.budget_blocks, window_blocks, target_blocks, total_blocks)
    return window_blocks, target_blocks


def effective_trigger(args, window_blocks):
    """Resident-block count at which the managed stream compacts.

    `None` keeps the pre-registered `min(budget, window+1)` rule (14 blocks for
    the 20-budget 13->9 geometry). An explicit `--trigger-blocks` moves the
    compaction point to the declared budget so the stream actually spends it:
    with trigger 20 the resident width cycles 16..20 instead of 10..14, and the
    uncompressed tail is 7 blocks instead of 1, which is the layout the
    independent-window training path sees (19 resident, 6-block tail).
    """
    explicit = getattr(args, "trigger_blocks", None)
    if explicit is None:
        return compaction_trigger(args.budget_blocks, window_blocks)
    if explicit < window_blocks:
        raise ValueError(
            f"--trigger-blocks {explicit} is smaller than the {window_blocks}-block "
            "compression window; the window must be fully resident when compaction fires")
    if explicit > args.budget_blocks:
        raise ValueError(
            f"--trigger-blocks {explicit} exceeds --budget-blocks {args.budget_blocks}; "
            "the stream would hold more blocks than the declared budget")
    return explicit


def stream_kwargs(args, merge_stream_cls=MergeStream):
    """MergeStream kwargs plus the geometry the stream will actually run.

    Explicit geometry needs the v7 streaming runtime (MergeStream window_blocks/
    target_blocks); this checkout may predate it, so refuse loudly instead of
    silently running the pre-registered table.
    """
    if args.arm == FULL_ARM:
        # The Full arm never compacts, so the managed budget/geometry are inert:
        # `StreamingMemory.write` skips the trigger for this arm and its capacity
        # limit is the whole physical pool. Keep the pre-registered default
        # budget so the stream constructor validates as usual, and report no
        # compaction geometry at all (`None`).
        return dict(budget_blocks=BUDGET_BLOCKS), None
    window_blocks, target_blocks = effective_geometry(args)
    kwargs = dict(budget_blocks=args.budget_blocks)
    if getattr(args, "window_blocks", None) is not None or getattr(args, "target_blocks", None) is not None:
        parameters = inspect.signature(merge_stream_cls.__init__).parameters
        if not {"window_blocks", "target_blocks"} <= set(parameters):
            raise RuntimeError(
                "--window-blocks/--target-blocks need a MergeStream that accepts "
                "explicit window_blocks/target_blocks (v7 streaming geometry, "
                "experiments/v7-spec-contract.md C8); this runtime only supports "
                "the pre-registered budget_geometry table")
        kwargs.update(window_blocks=window_blocks, target_blocks=target_blocks)
    if getattr(args, "trigger_blocks", None) is not None:
        effective_trigger(args, window_blocks)      # validate before any load
        parameters = inspect.signature(merge_stream_cls.__init__).parameters
        if "trigger_blocks" not in parameters:
            raise RuntimeError(
                "--trigger-blocks needs a MergeStream that accepts trigger_blocks "
                "(full-load trigger); this runtime only supports the "
                "min(budget, window+1) rule")
        kwargs.update(trigger_blocks=args.trigger_blocks)
    return kwargs, (window_blocks, target_blocks)


def layer_identity(saved, merge=None, num_layers=None):
    """Carrier-selector and routing reference layers recorded in the identity."""
    variant = (saved or {}).get("module_configuration") or {}
    fallback = _SELECT_LAYER if num_layers is None else min(_SELECT_LAYER, num_layers - 1)
    select_layer = variant.get("carrier_selector_layer")
    if select_layer is None:
        select_layer = fallback
    reference_layer = variant.get("reference_layer")
    if reference_layer is None:
        reference_layer = getattr(merge, "reference_layer", None)
    if reference_layer is None:
        reference_layer = fallback
    return select_layer, reference_layer


def current_identity(args, saved, geometry, checkpoint_sha256, code_hash_verified):
    """The full run identity --resume compares field by field."""
    select_layer, reference_layer = layer_identity(saved)
    if geometry is None:
        # No-compression reference: the capacity is the physical pool, not a
        # managed budget, and no merge module or carrier layer is involved.
        budget_blocks, window_blocks, target_blocks = PHYSICAL_BLOCKS, None, None
        trigger_blocks = None
        select_layer = reference_layer = None
    else:
        budget_blocks = args.budget_blocks
        window_blocks, target_blocks = geometry
        trigger_blocks = effective_trigger(args, window_blocks)
    return dict(arm=args.arm, checkpoint=args.checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                architecture=(None if args.arm == FULL_ARM
                              else resolve_architecture(saved, default=ARCHITECTURE)),
                variant=(saved or {}).get("variant"),
                total_tokens=args.total_tokens, tail_tokens=args.tail_tokens,
                budget_blocks=budget_blocks,
                window_blocks=window_blocks, target_blocks=target_blocks,
                trigger_blocks=trigger_blocks,
                select_layer=select_layer, reference_layer=reference_layer,
                critical_source_manifest=critical_manifest(),
                code_hash_verified=code_hash_verified)


def identity_mismatch(previous, current):
    """Field -> (stored, requested) for every identity field that changed."""
    previous = previous or {}
    return {key: (previous.get(key), current.get(key)) for key in IDENTITY_KEYS
            if previous.get(key) != current.get(key)}


def validate_resume_samples(samples, identity):
    """Every stored sample must come from the same arm and checkpoint file."""
    for sample in samples:
        where = f"{sample.get('task')}/{sample.get('sample_index')}"
        if sample.get("arm") != identity["arm"]:
            raise RuntimeError(f"--resume output mixes arms at {where}: stored "
                               f"{sample.get('arm')!r} != {identity['arm']!r}")
        if sample.get("checkpoint_sha256") != identity["checkpoint_sha256"]:
            raise RuntimeError(f"--resume output mixes checkpoints at {where}: stored "
                               f"{sample.get('checkpoint_sha256')!r} != "
                               f"{identity['checkpoint_sha256']!r}")


def merge_subset(samples):
    """Active-merge subset: compression_count > 0, same rule as fused_subset_report."""
    return [s for s in samples if int(s.get("compression_count", 0)) > 0]


def no_merge_subset(samples):
    return [s for s in samples if int(s.get("compression_count", 0)) == 0]


def subset_summary(samples):
    if not samples:
        return dict(n=0, qa_f1=None, qa_f1_raw=None, mean_compressions=None)
    return dict(n=len(samples),
                qa_f1=100 * float(np.mean([s["official_qa_f1_truncated"] for s in samples])),
                qa_f1_raw=100 * float(np.mean([s["official_qa_f1"] for s in samples])),
                mean_compressions=float(np.mean([int(s.get("compression_count", 0))
                                                 for s in samples])))


def summarize_task(samples, task):
    """Per-task summary; the active-merge subset is reported next to the full split."""
    task_samples = [s for s in samples if s["task"] == task]
    summary = subset_summary(task_samples)
    summary.update(active_merge_subset=subset_summary(merge_subset(task_samples)),
                   no_merge_subset=subset_summary(no_merge_subset(task_samples)))
    return summary


def evaluation_flags(checkpoint, code_hash_verified):
    """An unverified checkpoint may be evaluated for diagnosis only."""
    if checkpoint and code_hash_verified is not True:
        return dict(evaluation_eligible=False,
                    warning=("WARNING: checkpoint source hashes were not verified "
                             "(strict_code_hash=False or a historical checkpoint); "
                             "these numbers are historical diagnosis only and must "
                             "not be reported as publishable results"))
    return dict(evaluation_eligible=True, warning=None)


def protocol_record(args, tasks, saved, geometry):
    """Protocol block written to the result JSON (geometry from the CLI)."""
    if args.arm == FULL_ARM:
        # No merge module is loaded and nothing is compacted; the recorded
        # capacity is the native pool, never a managed budget.
        return dict(model="MemoryLLM-8B",
            method="PRLM-C Full reference (no compression; every block retained)",
            variant=None, architecture=None,
            capacity="native_50_blocks", capacity_blocks=PHYSICAL_BLOCKS,
            budget_blocks=PHYSICAL_BLOCKS, window_blocks=None, target_blocks=None,
            trigger_blocks=None,
            merge_arm=args.arm,
            latent_tokens_per_block=256, max_write_tokens=512,
            total_input_tokens=args.total_tokens,
            direct_query_tokens=args.tail_tokens,
            write_prefix_tokens=args.total_tokens - args.tail_tokens,
            max_new_tokens={t: MAX_NEW[t] for t in tasks}, seed=args.seed,
            do_sample=False, num_beams=1, training=False,
            benchmark="LongBench test",
            scoring="official QA F1 (raw) + newline-truncated QA F1",
            prompt_truncation="keep last total_input_tokens; no decode/re-encode",
            input_protocol=(f"tail{args.tail_tokens}_native_write_v1; "
                            "same as the native50 baseline"),
            injection="inject_memory(update_memory=False); manager places the delta",
            compaction="none: the Full arm retains every written block "
                       "(no-compression ceiling, not a managed budget)",
            seed_reset="reset to 0 before each sample",
            memory_reset="checkpoint values, used=0 and full active mask before every sample")
    window_blocks, target_blocks = geometry
    trigger_blocks = effective_trigger(args, window_blocks)
    explicit_trigger = getattr(args, "trigger_blocks", None) is not None
    return dict(model="MemoryLLM-8B",
        method=("PRLM-C Consolidate (training-free plan A)" if args.arm == "Consolidate"
                else (METHOD_V6 if saved and saved.get("architecture", "").endswith("_v6")
                      else METHOD) if args.checkpoint else "PRLM-C Selection"),
        variant=saved.get("variant") if saved else None,
        architecture=resolve_architecture(saved, default=ARCHITECTURE),
        capacity=f"managed_{args.budget_blocks}_block_budget",
        budget_blocks=args.budget_blocks,
        window_blocks=window_blocks, target_blocks=target_blocks,
        trigger_blocks=trigger_blocks,
        merge_arm=args.arm,
        latent_tokens_per_block=256, max_write_tokens=512,
        total_input_tokens=args.total_tokens,
        direct_query_tokens=args.tail_tokens,
        write_prefix_tokens=args.total_tokens - args.tail_tokens,
        max_new_tokens={t: MAX_NEW[t] for t in tasks}, seed=args.seed,
        do_sample=False, num_beams=1, training=False,
        benchmark="LongBench test",
        scoring="official QA F1 (raw) + newline-truncated QA F1",
        prompt_truncation="keep last total_input_tokens; no decode/re-encode",
        input_protocol=(f"tail{args.tail_tokens}_native_write_v1; "
                        "same as the native50 baseline"),
        injection="inject_memory(update_memory=False); manager places the delta",
        compaction=(f"{window_blocks}->{target_blocks} block merge at "
                    f"used>={trigger_blocks} "
                    + ("(trigger=budget, full-load: the declared budget is spent "
                       f"before compacting; resident width cycles {target_blocks + (trigger_blocks - window_blocks)}..{trigger_blocks} blocks, "
                       f"{trigger_blocks - window_blocks} uncompressed tail blocks)"
                       if explicit_trigger else
                       "(trigger=min(budget,window+1))")),
        seed_reset="reset to 0 before each sample",
        memory_reset="checkpoint values, used=0 and full active mask before every sample")


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


def checkpoint_metadata(path):
    """Checkpoint dict only (no model weights are loaded into modules)."""
    if not path:
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


def dry_run_plan(args, tasks, geometry):
    """Full configuration/identity/geometry report; loads no model, writes no file."""
    saved = checkpoint_metadata(args.checkpoint)
    checkpoint_sha256 = sha(args.checkpoint) if args.checkpoint else None
    warnings = []
    code_hash_verified = None
    if args.checkpoint:
        try:
            code_hash_verified = validate_merge_checkpoint(
                saved, saved.get("architecture"), strict_code_hash=args.strict_code_hash)
        except ValueError as exc:
            code_hash_verified = False
            warnings.append(str(exc))
    data_files = None
    if args.data_dir and Path(args.data_dir).is_dir():
        data_files = {t: dict(sha256=sha(Path(args.data_dir) / (t + ".jsonl")))
                      for t in tasks if (Path(args.data_dir) / (t + ".jsonl")).is_file()}
    identity = current_identity(args, saved, geometry, checkpoint_sha256, code_hash_verified)
    flags = evaluation_flags(args.checkpoint, code_hash_verified)
    resolved_geometry = (dict(budget_blocks=PHYSICAL_BLOCKS, window_blocks=None,
                              target_blocks=None, trigger_blocks=None,
                              total_blocks=PHYSICAL_BLOCKS)
                         if geometry is None
                         else dict(validate_geometry(args.budget_blocks, geometry[0], geometry[1]),
                                   trigger_blocks=effective_trigger(args, geometry[0])))
    plan = dict(dry_run=True, out=str(args.out),
                geometry=resolved_geometry,
                protocol=protocol_record(args, tasks, saved, geometry),
                identity=dict(identity, model_path=MODEL_PATH, data_dir=args.data_dir,
                              source_manifest=source_manifest(), data_files=data_files),
                evaluation_eligible=flags["evaluation_eligible"], warning=flags["warning"],
                warnings=warnings)
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if flags["warning"]:
        print(flags["warning"], file=sys.stderr)
    for warning in warnings:
        print(f"WARNING: dry-run would reject this run: {warning}", file=sys.stderr)
    return plan


def evaluate(args):
    if args.total_tokens not in TOTAL_TOKEN_CHOICES:
        raise ValueError(f"--total-tokens must be one of {TOTAL_TOKEN_CHOICES}")
    if args.total_tokens not in VALIDATED_TOTAL_TOKENS:
        raise ValueError(
            f"--total-tokens {args.total_tokens} is not enabled: the pre-registered "
            f"tail2048_native_write_v1 streaming protocol is validated for "
            f"{VALIDATED_TOTAL_TOKENS} only, and a 32768-token prefix has never been "
            "measured through this path; t9 must dry-run it before it can be reported")
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
    elif args.arm == FULL_ARM:
        if args.checkpoint is not None:
            raise ValueError("--arm Full keeps every block and loads no merge "
                             "module; drop --checkpoint")
        if args.window_blocks is not None or args.target_blocks is not None:
            raise ValueError("--arm Full performs no compaction, so "
                             "--window-blocks/--target-blocks do not apply")
        if args.trigger_blocks is not None:
            raise ValueError("--arm Full performs no compaction, so --trigger-blocks "
                             "does not apply")
        if args.budget_blocks != BUDGET_BLOCKS:
            raise ValueError("--arm Full runs at the native 50-block capacity, so "
                             f"--budget-blocks {args.budget_blocks} would mislabel it")
    elif args.checkpoint is not None:
        raise ValueError("--arm Selection mutates no learned module; drop --checkpoint")
    # `None` means "no compaction geometry"; only the Full arm runs without one.
    geometry = None if args.arm == FULL_ARM else effective_geometry(args)  # rejects impossible geometry before any load
    # Runtime handshake, validated before any model load: an explicit geometry
    # must be one the streaming runtime can actually run. Without this the
    # --dry-run path returned before stream_kwargs and printed a green plan for
    # a geometry the runtime would refuse (t8 finding).
    stream_kwargs(args)
    if args.dry_run:
        return dry_run_plan(args, tasks, geometry)
    if not args.data_dir:
        raise ValueError("--data-dir is required unless --dry-run")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    report = None
    if output.exists():
        if not args.resume:
            raise ValueError("output exists; refuse overwrite (use --resume)")
        prev = json.loads(output.read_text())
        if (prev.get("protocol", {}).get("total_input_tokens") != args.total_tokens
                or prev.get("identity", {}).get("checkpoint") != args.checkpoint
                or prev.get("protocol", {}).get("merge_arm") != args.arm):
            raise RuntimeError("--resume output was produced by a different protocol/checkpoint")
        done_ids = {(s["task"], s["sample_index"]) for s in prev["samples"]}
        report = prev
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
    install_generation_compatibility()

    # load_runtime pins the reader fork, the manager configuration
    # (carrier_residual, active mask, k_max = 20 blocks) and validates the
    # checkpoint architecture/training/budget against TRAIN_V4. strict_code_hash
    # defaults to True: a historical checkpoint without source_manifest is
    # refused unless the operator opts out explicitly.
    model, tokenizer, manager, merge, initial, cfg, saved = load_runtime(
        MODEL_PATH, checkpoint=args.checkpoint, train=False,
        architecture=None,
        budget_blocks=args.budget_blocks,
        strict_code_hash=args.strict_code_hash)
    tokenizer.pad_token = tokenizer.eos_token
    if int(model.num_blocks) != 50 or int(model.num_tokens) != 256:
        raise ValueError("expected the native 50-block MemoryLLM reader")
    checkpoint_sha256 = sha(args.checkpoint) if args.checkpoint else None
    stream_arguments, geometry = stream_kwargs(args)
    if geometry is None:
        if manager.total // manager.block != PHYSICAL_BLOCKS:
            raise ValueError("--arm Full needs the native 50-block pool")
    else:
        validate_geometry(args.budget_blocks, geometry[0], geometry[1],
                          manager.total // manager.block)
    code_hash_verified = saved.get("code_hash_verified") if saved else None
    identity = current_identity(args, saved, geometry, checkpoint_sha256, code_hash_verified)
    flags = evaluation_flags(args.checkpoint, code_hash_verified)
    if report is None:
        if args.checkpoint and saved is None:
            raise ValueError("checkpoint file did not load")
        report = dict(schema="longbench_prlmc_results/1", status="running",
            code_hash_verified=code_hash_verified,
            evaluation_eligible=flags["evaluation_eligible"],
            warning=flags["warning"],
            protocol=protocol_record(args, tasks, saved, geometry),
            identity=dict(identity, model_path=MODEL_PATH, data_dir=args.data_dir,
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
    else:
        mismatch = identity_mismatch(report.get("identity"), identity)
        if mismatch:
            raise RuntimeError("--resume identity mismatch: "
                               + json.dumps(mismatch, sort_keys=True, default=str))
        validate_resume_samples(report["samples"], identity)
        report.update(code_hash_verified=code_hash_verified,
                      evaluation_eligible=flags["evaluation_eligible"],
                      warning=flags["warning"])
    if flags["warning"]:
        print(flags["warning"], file=sys.stderr)
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
                tokenizer, task, row, args.total_tokens, tail_tokens=args.tail_tokens)
            original_tokens = len(tokenizer(prompt, add_special_tokens=False,
                                            truncation=False).input_ids)
            writes = [write_ids[i:i + 512] for i in range(0, len(write_ids), 512)]
            merge = build_consolidate_merge(merge, args.arm, manager)
            stream = MergeStream(manager, merge, args.arm, **stream_arguments)
            compression_events = run_sample(stream, writes)
            prediction, output_ids = generate(model, tokenizer, query_ids, MAX_NEW[task])
            truncated = truncated_prediction(prediction)
            active = int(model.memory_active_mask.sum())
            # The pre-registered geometry for budgets 8/10 reads a window one
            # block larger than the budget, so the stream may transiently hold
            # window blocks before the final compact lands back on target. The
            # Full arm has no budget: its limit is the whole physical pool.
            capacity_blocks = (PHYSICAL_BLOCKS if geometry is None
                               else max(args.budget_blocks, geometry[0]))
            if active != manager.used or active > capacity_blocks * model.num_tokens:
                raise ValueError("active budget violation after writes")
            report["samples"].append(dict(
                task=task, sample_index=sample_index, id=row.get("_id"),
                arm=args.arm, checkpoint_sha256=checkpoint_sha256,
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
    report["summary"] = {task: summarize_task(report["samples"], task) for task in tasks}
    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["status"] = "complete"
    atomic_json(output, report)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None,
                   help="LongBench test directory; required unless --dry-run")
    p.add_argument("--out", default=None,
                   help="result JSON path; required unless --dry-run")
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--total-tokens", "--max_length", dest="total_tokens", type=int,
                   default=16384, choices=TOTAL_TOKEN_CHOICES,
                   help="input cap; the per-sample write prefix is "
                        "min(cap-tail, sample length). Merges fire only when that "
                        "prefix reaches 20 blocks x 512 = 10240 tokens, so 8192 "
                        "(prefix <= 6144) always has compression_count=0, while "
                        "16384 admits prefixes up to 14336 and fuses only the "
                        "samples whose own context is long enough. 32768 is "
                        "accepted by the CLI but rejected explicitly: the "
                        "protocol is validated for 8192/16384 only")
    p.add_argument("--tail-tokens", dest="tail_tokens", type=int, default=2048,
                   choices=TAIL_CHOICES,
                   help="tokens queried directly at the tail (default 2048)")
    p.add_argument("--window-blocks", dest="window_blocks", type=int, default=None,
                   help="explicit merge window in native blocks (default: the "
                        "pre-registered budget_geometry table, 19 for budget 20); "
                        "requires the v7 streaming runtime")
    p.add_argument("--target-blocks", dest="target_blocks", type=int, default=None,
                   help="explicit merge target in native blocks (default 9 for "
                        "budget 20)")
    p.add_argument("--trigger-blocks", dest="trigger_blocks", type=int, default=None,
                   help="resident block count at which compaction fires (default: "
                        "min(budget, window+1) = 14 for budget 20 / window 13). "
                        "Set it to the budget (20) for the full-load layout: the "
                        "stream fills to 20 blocks, compresses the oldest 13 into "
                        "9, keeps the newest 7 uncompressed, and cycles 16..20 "
                        "resident instead of 10..14")
    p.add_argument("--strict-code-hash", dest="strict_code_hash", type=parse_bool,
                   default=True, metavar="{true,false}",
                   help="verify the checkpoint source_manifest against the live "
                        "operator code (default true); false loads a historical "
                        "checkpoint for diagnosis only and marks the run "
                        "evaluation_eligible=false")
    p.add_argument("--dry-run", action="store_true",
                   help="print the resolved protocol/identity/geometry and exit "
                        "without loading the model or writing --out")
    p.add_argument("--budget-blocks", dest="budget_blocks", type=int,
                   default=BUDGET_BLOCKS, choices=BUDGET_CHOICES,
                   help="managed working budget in native blocks (8/10/12/20); "
                        "lowers k_max so compaction fires more often, e.g. "
                        "repeated 19->9 then 9->4 (budget 8) compressions")
    p.add_argument("--checkpoint", default=None,
                   help="trained Dsup/Dnative checkpoint; required for --arm SharedMerge, "
                        "unused for --arm Selection")
    p.add_argument("--arm", default="SharedMerge",
                   choices=("SharedMerge", "Selection", "Consolidate", FULL_ARM),
                   help="SharedMerge = trained Dsup residual merge (our method); "
                        "Selection = same-budget pure block selector control; "
                        "Full = no-compression reference (every block retained in "
                        "the native 50-block pool)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tasks", nargs="+", choices=DEFAULT_TASKS)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--task-range", nargs=2, type=int, default=None, metavar=("START","END"),
                   help="evaluate only samples [START,END) of each task")
    p.add_argument("--resume", action="store_true",
                   help="skip samples already present in --out; refuses an output "
                        "whose stored identity differs from the current run")
    return p


def main():
    p = build_parser()
    args = p.parse_args()
    if not args.dry_run and (not args.data_dir or not args.out):
        p.error("--data-dir and --out are required unless --dry-run")
    evaluate(args)


if __name__ == "__main__":
    main()
