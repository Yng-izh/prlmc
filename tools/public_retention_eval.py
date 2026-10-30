#!/usr/bin/env python3
"""Frozen SharedE6d streaming retention on pinned original public inputs."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_public_retention import digest, sha, tokenizer_manifest

E6D_SHA = "a9fe2ea3f7a4e2c21590cf2219591edb11fc65297ce36ddeed96c236d8020c5d"
ARMS = ("Full", "Selection", "Recent", "SharedE6d", "NativeDefault", "Consolidate")


def shard_samples(plan, index=0, count=1):
    if not 1 <= count <= len(plan["samples"]) or not 0 <= index < count:
        raise ValueError("invalid sample shard")
    # Retain original sample IDs and full plan identity, without rewriting data.
    return [s for s in plan["samples"] if s["sample_index"] % count == index]


def author_score(prediction, target):
    target = target.replace("</s>", "").strip()
    if not target:
        raise ValueError("empty scoring target")
    return target in prediction


def validate_plan(plan):
    identity = dict(plan)
    expected = identity.pop("content_sha256", None)
    if expected != digest(identity) or plan.get("schema") != "public_retention/1" or plan.get("status") != "prepared":
        raise ValueError("prepared plan integrity failure")
    p = plan["protocol"]
    if (p["budget_blocks"], p["window_blocks"], p["target_blocks"], p["anchor_blocks"],
            p["max_new_tokens"], p["rank"], p["gamma_max"], p["training"]) != (20, 19, 9, 0, 10, 32, .05, False):
        raise ValueError("fixed retention protocol changed")
    if p["arms"] != list(ARMS) or p["do_sample"] is not False or p["stop_strings"]:
        raise ValueError("fixed arm/generation protocol changed")
    samples = plan["samples"]
    if not samples or [s["filtered_index"] for s in samples] != plan["official_indices"]:
        raise ValueError("official sample indices changed")
    if [s["sample_index"] for s in samples] != list(range(len(samples))):
        raise ValueError("sample IDs changed")
    for sample in samples:
        if digest(sample["context"]) != sample["context_sha256"] or not sample["answer"]:
            raise ValueError("original sample changed")
        for write in sample["related_writes"] + plan["unrelated_writes"]:
            ids = write["token_ids"]
            if not ids or len(ids) > 512 or any(type(i) is not int or i < 0 for i in ids) or digest(ids) != write["token_sha256"]:
                raise ValueError("prepared write token integrity failure")


@contextmanager
def isolated_query(model):
    """Queries cannot become writes, including when generation raises."""
    import torch
    memory = model.memory
    snapshot = memory.detach().cpu().clone()
    initialized = model.initialized.clone()
    mask = model._buffers.get("memory_active_mask")
    mask_value = None if mask is None else mask.clone()
    modes = [(m, m.training) for m in model.modules()]
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    device = memory.device
    try:
        model.eval()
        yield
    finally:
        # Restore parameter binding, storage, values, mask and RNG even if the
        # native generation path replaces memory.data or the parameter itself.
        if model.memory is not memory:
            model.memory = memory
        with torch.no_grad():
            if memory.shape == snapshot.shape and memory.device == device:
                memory.copy_(snapshot)
            else:
                memory.data = snapshot.to(device)
            model.initialized.copy_(initialized)
        if mask is None:
            model._buffers.pop("memory_active_mask", None)
        else:
            model._buffers["memory_active_mask"] = mask
            mask.copy_(mask_value)
        for module, mode in modes:
            module.training = mode
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def generate(model, tokenizer, sample, target):
    import torch
    ids = torch.tensor([sample["prompt_ids"]], device=model.memory.device, dtype=torch.long)
    with isolated_query(model), torch.inference_mode():
        # Exact author generation length and no newline stop. The prefix width
        # is physical and fixed; logical masking is applied by the reader fork.
        attention = torch.ones((1, model.memory.shape[1] + ids.shape[1]),
                               dtype=torch.long, device=ids.device)
        output = model.generate(input_ids=ids, attention_mask=attention, max_new_tokens=10,
                                do_sample=False, use_cache=True,
                                pad_token_id=tokenizer.pad_token_id)
        tokens = output[0, ids.shape[1]:].detach().cpu().tolist()
        prediction = tokenizer.decode(tokens, skip_special_tokens=False)
    return dict(prediction=prediction, token_ids=tokens, score=author_score(prediction, target))


def budget_geometry(budget_blocks):
    """Pre-registered (window_blocks, target_blocks) per managed budget."""
    table = {8: (9, 4), 10: (11, 5), 12: (11, 5), 20: (19, 9)}
    if budget_blocks not in table:
        if budget_blocks == 50:
            return 19, 9    # native full pool: 12800 slots, 19 -> 9
    raise ValueError("budget_blocks must be one of 8/10/12/20/50")
    return table[budget_blocks]


def compaction_trigger(budget_blocks, window_blocks):
    """Resident block count at which the managed stream compacts (v7 C8/D-07).

    The rule is `used >= min(budget, window + 1)`: the window has to be fully
    resident before it can be compressed, and a stream whose budget is smaller
    than its window has to be allowed to reach the window (transient overshoot
    of one block, then back to target). For every pre-registered geometry this
    is grid-identical to the old `used + block > budget and used >= window`
    rule. This function returns the budget/window term (8, 10, 12, 20, and 14
    for the v7 20-budget 13->9 geometry); `write` additionally requires
    `used >= window`, so the effective first compaction is 9/11/12/20/14 blocks
    respectively. Firing at 20 with a 13-block window would silently compress
    only 13 of the 20 resident blocks, which is why the rule lives on the write
    path and not in the CLI.
    """
    return min(int(budget_blocks), int(window_blocks) + 1)


class StreamingMemory:
    """Explicit preappend 19→9 compression; no answer-derived goals/anchors."""
    def __init__(self, manager, compactor, arm, budget_blocks=20, window_blocks=None,
                 target_blocks=None, strict_budget=False, trigger_blocks=None):
        if arm not in ARMS:
            raise ValueError("unknown arm")
        # budget_blocks is the managed working budget (8/10/12/20 native blocks).
        # The pre-registered compression geometry (user contract) is a fixed
        # table, not a formula: budget 8 -> window 9 -> target 4, budget 10 ->
        # 11 -> 5, budget 12 -> 11 -> 5, budget 20 -> 19 -> 9. For budgets 8 and
        # 10 the window exceeds the budget by one block: the compact input may
        # reach window blocks (transient overshoot), then lands back on target
        # blocks, so the steady state after every completed write cycle stays at
        # or below the budget.
        #
        # `None` keeps that table; an explicit pair overrides it (C8/D-08). The
        # old code overwrote the caller's values with the table, which made
        # `--window-blocks 13 --target-blocks 9` a silent no-op, and its
        # validation was dead code (`A and B` precedence made the guard never
        # fire). Both are fixed by resolving through `resolve_geometry`, which
        # raises on a conflict instead of silently correcting it.
        self.m = manager
        self.compactor = compactor
        self.arm = arm
        if budget_blocks not in (8, 10, 12, 20, 50):
            raise ValueError("budget_blocks must be one of 8/10/12/20")
        from natural_merge_runtime import resolve_geometry
        self.budget_blocks = int(budget_blocks)
        self.window_blocks, self.target_blocks = resolve_geometry(
            budget_blocks, window_blocks, target_blocks,
            total_blocks=manager.total // manager.block,
        )
        # Declared vs. actually allowed capacity. For the 8/10-block geometries
        # the pre-registered window (9/11) is one block larger than the budget,
        # so the stream may transiently hold 9/11 blocks before compacting back
        # to the target. That is a real capacity advantage against any method
        # held at 8/10, so it must be reported, never silently absorbed into a
        # "managed_8_block_budget" label. `strict_budget=True` refuses the
        # overshoot entirely (the window has to fit inside the budget).
        #
        # `trigger_blocks` (full-load trigger) moves the compaction point off the
        # `min(budget, window+1)` rule so the declared budget is actually used:
        # with budget 20, window 13, target 9 and trigger 20 the stream fills to
        # 20 resident blocks, compresses the oldest 13 into 9, and keeps the
        # newest 7 uncompressed -- resident width then cycles 16..20 instead of
        # 10..14. That both spends the declared 20-block budget and matches the
        # training-time window layout (19 resident, 6-block uncompressed tail).
        self.trigger_blocks = None if trigger_blocks is None else int(trigger_blocks)
        if self.trigger_blocks is not None:
            if self.trigger_blocks < self.window_blocks:
                raise ValueError(
                    f"trigger_blocks {self.trigger_blocks} is smaller than the "
                    f"{self.window_blocks}-block compression window; the window "
                    "would not be fully resident when compaction fires")
            if self.trigger_blocks > self.budget_blocks:
                raise ValueError(
                    f"trigger_blocks {self.trigger_blocks} exceeds the declared "
                    f"budget {self.budget_blocks}; the stream would hold more "
                    "blocks than it reports")
        self.effective_trigger_blocks = (self.trigger_blocks if self.trigger_blocks is not None
                                         else compaction_trigger(self.budget_blocks,
                                                                 self.window_blocks))
        self.peak_allowed_blocks = max(self.budget_blocks, self.window_blocks,
                                       self.effective_trigger_blocks)
        self.capacity_overshoot_blocks = self.peak_allowed_blocks - self.budget_blocks
        if strict_budget and self.capacity_overshoot_blocks:
            raise ValueError(
                f"strict budget {self.budget_blocks} violated by the compression "
                f"window {self.window_blocks}; pass explicit window_blocks <= "
                "budget_blocks (and target_blocks < window_blocks) to stay inside "
                "the declared capacity")
        self.budget = self.budget_blocks * manager.block
        self.window = self.window_blocks * manager.block
        self.target = self.target_blocks * manager.block
        self.events = []
        self.peak = 0

    def compact(self, write_index):
        import torch
        from methods.prlmc.consolidate import select_block_compact, recent_block_compact, _select_block_indices
        m = self.m
        start = m.total - m.used
        used_before = int(m.used)
        source = m._memory[:, start:start + self.window].contiguous()
        weights = m._window_importance(start, self.window)
        # Generational ages of exactly this window, in temporal order (C1/C2).
        # This is the write path every v7 evaluation number comes from, so the
        # quota has to be settled here and not only in the training runtime.
        ages = m.window_block_ages(start, self.window)
        kwargs = dict(target=self.target, block=m.block, weights=weights,
                      goal_direction=None, select_weights=m.select_weights,
                      block_ages=ages)
        begin = time.monotonic()
        if self.arm == "SharedE6d":
            output = self.compactor(source, **kwargs).to(m.dtype)
            selected = _select_block_indices(source, self.target, m.block, weights, None,
                                             m.select_weights, block_ages=ages)
        elif self.arm == "Selection":
            output = select_block_compact(source, **kwargs).to(m.dtype)
            selected = _select_block_indices(source, self.target, m.block, weights, None,
                                             m.select_weights, block_ages=ages)
        elif self.arm == "Recent":
            output = recent_block_compact(source, target=self.target, block=m.block).to(m.dtype)
            selected = torch.arange(self.window // m.block - self.target // m.block, self.window // m.block)
        else:
            raise ValueError("reference arm cannot use managed compression")
        if output.shape != (m.num_layers, self.target, m.hidden) or not torch.isfinite(output).all():
            raise ValueError("invalid compactor output; no eviction fallback")
        # Keep the newest native block unmodified so the next injection has the
        # same predecessor position. No fact is placed in a protected anchor.
        # No tail clone and no cat: the surviving tail already occupies exactly the
        # slots the compacted pool needs ([start+window, total)), so only the fused
        # head has to be written. Slot placement and values are identical to the old
        # cat((output, tail)) + full-pool copy, with ~4 GB less transient traffic.
        m.used = self.target + (used_before - self.window)
        head = m.total - m.used
        m._memory[:, head:head + self.target].copy_(output)
        # Settle the ages from the carriers the merge actually kept: a surviving
        # carrier is one round older, the uncompressed tail keeps its age, the
        # rest of the pool resets to the young generation (D-04). The old code
        # had no age bookkeeping on this path at all, so the evaluation stream
        # silently ran the young-generation selector forever.
        m.note_merge(selected, self.window // m.block,
                     target_blocks=self.target // m.block, used_before=used_before)
        m._tile_free_region()
        self.events.append(dict(before_write=write_index, source_slots=self.window, target_slots=self.target,
            remaining_slots=m.used, selected_blocks=selected.cpu().tolist(),
            selected_block_ages=ages[selected].cpu().tolist(),
            seconds=time.monotonic() - begin))

    def write(self, record, index):
        import torch
        m = self.m
        with torch.no_grad():
            # Trigger rule (C8/D-07). `min(budget, window+1)` is grid-identical
            # to the old `used + block > budget` gate for the pre-registered
            # 8/10/12/20 geometries, and fires at 14 blocks for the v7
            # 20-budget 13->9 geometry instead of silently waiting for 20.
            # An explicit `trigger_blocks` (full-load trigger) replaces that term
            # so the declared budget is spent before compacting.
            if self.arm not in ("Full", "NativeDefault") \
                    and m.used >= self.window \
                    and m.used >= self.effective_trigger_blocks * m.block:
                self.compact(index)
            if self.arm == "Full" and m.used + m.block > m.total:
                return None  # Honest Full-capacity failure; never truncate the stream.
            ids = torch.tensor([record["token_ids"]], dtype=torch.long, device=m._memory.device)
            attention = torch.ones((1, m.block + ids.shape[1]), dtype=torch.long, device=ids.device)
            delta = m.model.inject_memory(ids, context_attention_mask=attention,
                                          update_memory=self.arm == "NativeDefault")
            if self.arm == "NativeDefault":
                # Published native random-slot eviction operates on the entire
                # checkpoint pool; report separately from same-budget arms.
                m.used = m.total
                m.model.memory_active_mask.fill_(True)
            else:
                delta = delta.detach()[0]
                if delta.shape != (m.num_layers, m.block, m.hidden):
                    raise ValueError("unexpected injection shape; refuse dropping/tiling original writes")
                if m.used:
                    old_start = m.total - m.used
                    # On-device block shift: the same fp16 values end up in the same
                    # slots as the old CPU-staged move, without the two `used`-sized
                    # PCIe transfers. Ascending order is safe because every
                    # destination sits exactly one block below its source, i.e.
                    # outside the region that has not been read yet.
                    for b in range(m.used // m.block):
                        src = old_start + b * m.block
                        m._memory[:, src - m.block:src].copy_(m._memory[:, src:src + m.block])
                m._memory[:, -m.block:].copy_(delta)
                m.used += m.block
                # Age bookkeeping on the real write path (C2): exactly one
                # native block was appended and nothing was dropped. Bumping
                # `age_append_events` is also what makes `MergeStream.write`
                # skip its own reconciliation, so the append is never counted
                # twice.
                m.note_append(appended=1)
                m._tile_free_region()
            active = int(m.model.memory_active_mask.sum())
            # The allowed ceiling is the declared peak (`peak_allowed_blocks`),
            # not the declared budget: for the 8/10 geometries the window is one
            # block larger and the overshoot is reported, not hidden.
            limit = m.total if self.arm in ("Full", "NativeDefault") \
                else self.peak_allowed_blocks * m.block
            if active != m.used or active > limit:
                raise ValueError("active budget violation")
            self.peak = max(self.peak, active)
        return dict(index=index, role=record["role"], input_tokens=len(record["token_ids"]),
            below_author_16_token_training_floor=len(record["token_ids"]) < 16,
            input_sha256=record["token_sha256"], active_slots=active,
            compression_count=len(self.events), native_random_evicted_slots=m.block if self.arm == "NativeDefault" else 0)


def summarize(report):
    samples = report["samples"]
    required = ("Selection", "Recent", "SharedE6d", "NativeDefault")
    if any(s["arms"].get(a, {}).get("status") != "complete" for s in samples for a in required):
        raise ValueError("cannot summarize incomplete paired arms")
    def final(s, a):
        trace = s["arms"][a]["trace"]
        return bool(trace[-1]["generation"]["score"])
    scores = {a: [final(s, a) for s in samples if s["arms"][a]["status"] == "complete"] for a in ARMS}
    gained, harmed, lost, rescued, eligible = [], [], [], [], []
    for s in samples:
        key = s["sample_index"]
        selection, shared = final(s, "Selection"), final(s, "SharedE6d")
        if shared and not selection:
            gained.append(key)
        if selection and not shared:
            harmed.append(key)
        if s["arms"]["Full"]["status"] == "complete" and final(s, "Full"):
            eligible.append(key)
            if not selection:
                lost.append(key)
                if shared:
                    rescued.append(key)
    compressed = [s["sample_index"] for s in samples if s["arms"]["SharedE6d"]["compressions"]]
    answerable = [s["sample_index"] for s in samples
                  if any(row["generation"]["score"] for arm in ("Full", "NativeDefault")
                         for row in s["arms"][arm]["trace"] if row["role"] == "related")]
    status = ("not_applicable_no_compression" if not compressed else
              "reader_reference_failed" if not answerable else "paired_results_available")
    clusters = {}
    for s in samples:
        key = s.get("context_sha256", "sample-" + str(s["sample_index"]))
        clusters.setdefault(key, []).append(int(final(s, "SharedE6d")) - int(final(s, "Selection")))
    # Resample whole original contexts, carrying all their QA records together.
    import random
    rng = random.Random(0)
    values = list(clusters.values())
    boot = []
    for _ in range(2000):
        drawn = [values[rng.randrange(len(values))] for _ in values]
        boot.append(sum(sum(v) for v in drawn) / sum(len(v) for v in drawn))
    boot.sort()
    curves = {}
    for a in ARMS:
        max_steps = max(len(s["arms"][a]["trace"]) for s in samples)
        curves[a] = []
        for i in range(max_steps):
            rows = [s["arms"][a]["trace"][i] for s in samples if len(s["arms"][a]["trace"]) > i]
            curves[a].append(dict(write=i + 1, n=len(rows), correct=sum(r["generation"]["score"] for r in rows),
                score=sum(r["generation"]["score"] for r in rows) / len(rows),
                warning="lossless samples can have different related-document chunk counts; final paired scores remain primary"))
    return dict(validity=status, n=len(samples), retention_by_write=curves,
        arms={a: dict(n=len(v), correct=sum(v),
        answer_containment=sum(v) / len(v) if v else None) for a, v in scores.items()},
        full_capacity_failed=[s["sample_index"] for s in samples if s["arms"]["Full"]["status"] != "complete"],
        compressed_samples=compressed, related_readout_answerable=answerable,
        reader_unanswerable=[s["sample_index"] for s in samples if s["sample_index"] not in answerable],
        shared_vs_selection=dict(gained=gained, harmed=harmed, net=len(gained) - len(harmed),
                                  delta=(len(gained) - len(harmed)) / len(samples)),
        full_final_correct=eligible, strict_full_selection_lost=lost, strict_rescued=rescued,
        conclusion_allowed=status == "paired_results_available" and len(compressed) == len(samples),
        context_bootstrap=dict(n_contexts=len(values), resamples=2000, seed=0,
            ci95=[boot[49], boot[1949]],
            limitation="conditional on this fixed shared distractor stream; not cross-stream or cross-task uncertainty"),
        inference_unit="original context; QAs and common distractor stream are correlated",
        claim="paired retention evidence only; no automatic cross-task generalization claim")


def run(args):
    from native_qa_compare import atomic_json
    plan = json.loads(Path(args.plan).read_text())
    validate_plan(plan)
    shard_index = getattr(args, "shard_index", 0)
    shard_count = getattr(args, "shard_count", 1)
    selected_samples = shard_samples(plan, shard_index, shard_count)
    if tokenizer_manifest(args.model_path) != plan["tokenizer"]:
        raise ValueError("tokenizer differs from prepared original inputs")
    if sha(args.checkpoint) != E6D_SHA:
        raise ValueError("this protocol evaluates only the fixed existing E6d checkpoint")
    if Path(args.out).exists():
        raise ValueError("output exists; refuse overwrite")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    os.environ["S0B_GPU_SET"] = "1"
    import torch
    import yaml
    from transformers import AutoTokenizer
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory
    from s1d_shared_compactor_eval import _make_compactor, state_digest
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = MemoryLLM.from_pretrained(args.model_path, torch_dtype=torch.float16,
        attn_implementation="sdpa", local_files_only=True).to("cuda:0").eval()
    model.requires_grad_(False)
    if int(model.initialized) != 1:
        raise ValueError("fixed checkpoint must be initialized; cold-start study is paused")
    if not hasattr(model, "memory_active_mask"):
        raise ValueError("budget-aware reader fork required")
    if float(getattr(model, "memory_attn_scale", 1.)) != 1.:
        raise ValueError("reader attention gain must remain 1")
    checkpoint_memory = model.memory.detach().cpu().clone()
    checkpoint_initialized = model.initialized.clone()
    cfg = yaml.safe_load((ROOT / "configs/agent_conf/prlmc_e6.yaml").read_text())["prlmc"]
    cfg.update(k_anchor=0, anchor_source="none", k_max=20 * model.num_tokens, device="cuda:0",
        dtype="float16", active_memory_mask=True, trigger="none", compact_mode="carrier_residual",
        projection="none", basis_mode="none", compact_probe=False, stats_dir=None,
        carrier_gamma_max=.05, write_mode="append", gate_mode="ones", functional_control="none")
    manager = ProtectedResidualMemory(model, tokenizer, PRLMCConfig.from_dict(cfg))
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved.get("evaluation_eligible") is False or saved.get("effective_gamma_max", .05) != .05:
        raise ValueError("checkpoint ineligible or gamma changed")
    compactor = _make_compactor(manager, saved["state_dict"], .05, "cuda:0")
    frozen = state_digest(compactor)
    if saved.get("final_state_sha256", frozen) != frozen:
        raise ValueError("checkpoint state certificate mismatch")
    config_identity = dict(cfg)
    source_files = [Path(__file__), ROOT / "tools/prepare_public_retention.py",
        ROOT / "methods/prlmc/consolidate.py", ROOT / "methods/prlmc/layout.py",
        ROOT / "methods/memoryllm_modeling/modeling_memoryllm.py"]
    report = dict(schema="public_retention_results/1", status="running", task=plan["task"], mode=plan["mode"],
        identity=dict(plan_sha256=sha(args.plan), dataset=plan["dataset"], checkpoint_sha256=sha(args.checkpoint),
            model_manifest_sha256=sha(Path(args.model_path) / "fp16_cache_manifest.json"),
            sources={str(p.relative_to(ROOT)): sha(p) for p in source_files}, manager_config=config_identity),
        protocol=plan["protocol"],
        capacity=dict(
            declared_budget_blocks=plan["protocol"]["budget_blocks"],
            window_blocks=plan["protocol"]["window_blocks"],
            target_blocks=plan["protocol"]["target_blocks"],
            peak_allowed_blocks=max(plan["protocol"]["budget_blocks"],
                                    plan["protocol"]["window_blocks"]),
            capacity_overshoot_blocks=max(0, plan["protocol"]["window_blocks"]
                                          - plan["protocol"]["budget_blocks"])),
        samples=[],
        sharding=dict(index=shard_index, count=shard_count,
                      sample_indices=[s["sample_index"] for s in selected_samples]),
        training_performed=False, optimizer_created=False,
        physical_slots=manager.total, matched_budget_slots=20 * manager.block,
        shared_state_sha256_before=frozen, shared_parameter_count=sum(p.numel() for p in compactor.parameters()),
        shared_state_bytes=sum(t.numel() * t.element_size() for t in compactor.state_dict().values()),
        runtime=dict(torch=torch.__version__, transformers=__import__("transformers").__version__,
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), physical_device=args.cuda_device))
    atomic_json(args.out, report)
    started = time.monotonic()
    try:
        for sample in selected_samples:
            target = tokenizer.decode(tokenizer(sample["answer"] + "</s>", add_special_tokens=False).input_ids)
            row = dict(sample_index=sample["sample_index"], original_id=sample["id"],
                       context_sha256=sample["context_sha256"], question=sample["question"],
                       original_answer=sample["answer"], scoring_target=target, arms={})
            report["samples"].append(row)
            writes = sample["related_writes"] + plan["unrelated_writes"]
            for arm in ARMS:
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                with torch.no_grad():
                    model.memory.data = checkpoint_memory.to("cuda:0")
                    model.initialized.copy_(checkpoint_initialized)
                    model.memory_active_mask.fill_(True)
                manager.used = 0
                manager.goal_direction = None
                manager._active_mask_used = None
                stream = StreamingMemory(manager, compactor, arm,
                                         strict_budget=bool(getattr(args, "strict_budget", False)))
                arm_row = dict(status="running", trace=[], compressions=stream.events,
                    reference="whole native pipeline, full physical capacity" if arm == "NativeDefault" else "managed append, matched writes")
                row["arms"][arm] = arm_row
                for i, write in enumerate(writes):
                    entry = stream.write(write, i)
                    if entry is None:
                        arm_row.update(status="capacity_exceeded", failed_before_write=i)
                        break
                    entry["generation"] = generate(model, tokenizer, sample, target)
                    arm_row["trace"].append(entry)
                    report["progress"] = dict(sample=sample["sample_index"], arm=arm, write=i + 1, writes=len(writes))
                    atomic_json(args.out, report)
                    print(json.dumps(report["progress"]), flush=True)
                else:
                    arm_row["status"] = "complete"
                arm_row["peak_active_slots"] = stream.peak
                arm_row["capacity"] = dict(
                    declared_budget_blocks=stream.budget_blocks,
                    window_blocks=stream.window_blocks,
                    target_blocks=stream.target_blocks,
                    peak_allowed_blocks=stream.peak_allowed_blocks,
                    capacity_overshoot_blocks=stream.capacity_overshoot_blocks,
                    peak_observed_blocks=stream.peak // manager.block,
                    within_declared_budget=stream.peak // manager.block <= stream.budget_blocks)
                if state_digest(compactor) != frozen or any(p.requires_grad for p in compactor.parameters()) or any(p.requires_grad for p in model.parameters()):
                    raise ValueError("shared module changed during frozen evaluation")
                atomic_json(args.out, report)
        report.update(status="complete", summary=summarize(report), shared_state_sha256_after=state_digest(compactor),
            costs=dict(elapsed_seconds=time.monotonic() - started,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    except Exception as exc:
        report.update(status="failed", failure=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        atomic_json(args.out, report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--model-path", default="/dev/shm/s1d_reader_fp16")
    p.add_argument("--checkpoint", default="runs/e6d_interleave_ckpt.pt")
    p.add_argument("--cuda-device", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--strict-budget", action="store_true",
                   help="refuse geometries whose compression window exceeds the declared "
                        "budget (8/10-block defaults overshoot by one block); use for "
                        "capacity-matched comparisons")
    run(p.parse_args())


if __name__ == "__main__":
    main()
