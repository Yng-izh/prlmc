"""Shared natural-merge runtime; no changes to the live author512 entrypoint."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_public_retention import digest, sha, tokenizer_manifest
from prepare_natural_merge import PROTOCOL
from public_retention_eval import StreamingMemory, budget_geometry, generate, E6D_SHA

SOURCE_FILES = ["configs/agent_conf/prlmc_e6.yaml", "methods/prlmc/config.py",
    "methods/prlmc/importance.py", "methods/prlmc/capacity.py", "methods/prlmc/subspace.py",
    "tools/native_qa_compare.py", "methods/prlmc/merge.py", "methods/prlmc/consolidate.py", "methods/prlmc/layout.py",
    "methods/prlmc/probes.py", "tools/natural_merge_runtime.py", "tools/prepare_natural_merge.py",
    "tools/natural_merge_eval.py", "tools/natural_merge_train.py", "tools/verify_natural_merge.py",
    "tools/public_retention_eval.py", "tools/prepare_public_retention.py",
    "tools/freeze_natural_merge.py", "tools/run_natural_merge_parallel.py",
    "methods/prlmc/slot_merge.py", "tools/prepare_natural_merge_v2.py",
    "tools/natural_merge_v2_train.py",
    "tools/run_natural_merge_v2.py", "tools/summarize_natural_merge_v2.py",
    "tools/natural_merge_v2_training_fit.py",
    "methods/prlmc/gated_slot_merge.py", "tools/prepare_natural_merge_v3.py",
    "tools/natural_merge_v3_objective.py", "tools/natural_merge_v3_train.py",
    "tools/natural_merge_v3_training_fit.py", "tools/run_natural_merge_v3.py",
    "tools/summarize_natural_merge_v3.py",
    "methods/prlmc/native_skip_merge.py", "tools/prepare_natural_merge_v4.py",
    "tools/natural_merge_v4_objective.py", "tools/natural_merge_v4_train.py",
    "tools/natural_merge_v4_training_fit.py", "tools/run_natural_merge_v4.py",
    "tools/summarize_natural_merge_v4.py",
    "tools/prepare_natural_merge_v7.py", "tools/natural_merge_v7_train.py",
    "methods/memoryllm_modeling/modeling_memoryllm.py"]


# Operator files whose numerics a checkpoint claims to reproduce. Narrower than
# SOURCE_FILES on purpose: evaluation/reporting edits must not invalidate a
# checkpoint, but any change to these six files makes the stored residual merge
# compute different numbers.
CRITICAL_MERGE_FILES = ["methods/prlmc/consolidate.py", "methods/prlmc/layout.py",
    "methods/prlmc/importance.py", "methods/prlmc/slot_merge.py",
    "methods/prlmc/gated_slot_merge.py", "methods/prlmc/native_skip_merge.py"]
if not set(CRITICAL_MERGE_FILES).issubset(SOURCE_FILES):
    raise RuntimeError("critical merge files must be covered by SOURCE_FILES")


def source_manifest():
    return {name: sha(ROOT/name) for name in SOURCE_FILES}


def critical_manifest():
    """Current SHA256 of the operator files a checkpoint must match."""
    return {name: sha(ROOT/name) for name in CRITICAL_MERGE_FILES}


def tensor_digest(tensor):
    import hashlib
    x = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(str((tuple(x.shape), str(x.dtype))).encode())
    h.update(x.numpy().tobytes())
    return h.hexdigest()


def state_digest(module):
    return digest({name: tensor_digest(t) for name, t in sorted(module.state_dict().items())})


def scores(prediction, qa, auxiliary_prediction=None):
    """Author primary and SQuAD auxiliaries are never pooled into one score."""
    import re
    import string
    from collections import Counter
    def normalize(s):
        s = s.lower().translate(str.maketrans("", "", string.punctuation))
        return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())
    target = qa["scoring_target"]
    if not target:
        raise ValueError("empty original author target")
    p = normalize(prediction if auxiliary_prediction is None else auxiliary_prediction)
    golds = [normalize(a) for a in qa["answers"]]
    def f1(g):
        pt, gt = p.split(), g.split()
        if not pt or not gt:
            return float(pt == gt)
        common = sum((Counter(pt) & Counter(gt)).values())
        return 2 * common / (len(pt) + len(gt))
    return dict(author_containment=target in prediction, normalized_em=any(p == g for g in golds),
                normalized_f1=max(f1(g) for g in golds))


def evaluate_questions(model, tokenizer, qas):
    result = []
    for qa in qas:
        gen = generate(model, tokenizer, qa, qa["scoring_target"])
        auxiliary_prediction=tokenizer.decode(gen["token_ids"],skip_special_tokens=True)
        result.append(dict(id=qa["id"], question=qa["question"], original_answers=qa["answers"],
            scoring_target=qa["scoring_target"], context_sha256=qa["context_sha256"],
            source_index=qa["source_index"], prediction=gen["prediction"], token_ids=gen["token_ids"],
            auxiliary_prediction=auxiliary_prediction, scores=scores(gen["prediction"], qa, auxiliary_prediction)))
    return result


def resolve_geometry(budget_blocks, window_blocks=None, target_blocks=None,
                     total_blocks=None):
    """Effective (window, target) in native blocks for the streaming runtime.

    `None` means "use the pre-registered table", so the 8/10/12/20 geometries are
    unchanged; an explicit value must be internally consistent and fit the
    physical pool. Conflicts raise rather than being silently corrected, because
    a geometry the caller did not ask for is an unreportable experiment.
    """
    default_window, default_target = budget_geometry(budget_blocks)
    window = default_window if window_blocks is None else int(window_blocks)
    target = default_target if target_blocks is None else int(target_blocks)
    if not 0 < target < window:
        raise ValueError(
            "invalid compression geometry: need 0 < target_blocks < window_blocks, "
            f"got window={window} target={target}")
    if target > budget_blocks:
        raise ValueError(
            f"invalid compression geometry: target_blocks {target} exceeds "
            f"budget_blocks {budget_blocks}")
    if total_blocks is not None and window > int(total_blocks):
        raise ValueError(
            f"invalid compression geometry: window_blocks {window} exceeds the "
            f"physical {int(total_blocks)}-block pool")
    return window, target


class MergeStream(StreamingMemory):
    def __init__(self, manager, compactor, arm, on_compression=None, carrier_schedule=None,
                 budget_blocks=20, window_blocks=None, target_blocks=None,
                 trigger_blocks=None):
        aliases = {"SharedMerge": "SharedE6d", "CarrierOnlyFusion": "SharedE6d"}
        # Resolved before the base class runs: `None` keeps the pre-registered
        # table, an explicit value must be consistent, and the base class is
        # handed the resolved pair so the stream runs exactly what was validated
        # (C8/D-08). The assignment after `super()` is what makes the geometry
        # stick even on a `StreamingMemory` that still recomputes the table.
        geometry = resolve_geometry(
            budget_blocks, window_blocks, target_blocks,
            total_blocks=manager.total // manager.block,
        )
        super().__init__(manager, compactor, aliases.get(arm, arm),
                         budget_blocks=budget_blocks,
                         window_blocks=geometry[0], target_blocks=geometry[1],
                         trigger_blocks=trigger_blocks)
        self.merge_arm = arm
        self.on_compression = on_compression
        self.carrier_schedule = carrier_schedule
        self.window_blocks, self.target_blocks = geometry
        self.window = geometry[0] * manager.block
        self.target = geometry[1] * manager.block

    def write(self, record, index):
        """Streaming append with generational bookkeeping (v7 C2).

        `StreamingMemory.write` appends one native block per record but is owned
        by the evaluation layer, so the age array is reconciled here instead of
        being assumed. The reconciliation is exact, not heuristic: a merge
        removes `window - target` slots and the write then appends exactly one
        block, so the resident width before the append is known.

        `note_append` bumps `manager.age_append_events`. If the streaming path
        ever settles appends itself, that counter moves and this reconciliation
        becomes a no-op, which is what keeps the two layers from counting the
        same append twice.
        """
        m = self.m
        events_before = getattr(m, "age_append_events", None)
        if events_before is None:
            return super().write(record, index)
        used_before = int(m.used)
        merges_before = len(self.events)
        result = super().write(record, index)
        if m.age_append_events != events_before:
            return result
        if self.arm == "NativeDefault":
            # The published baseline overwrites the whole pool and reports
            # `used = total`; there is no per-block provenance to carry.
            return result
        merged = len(self.events) - merges_before
        expected = used_before - (self.window - self.target) * merged
        appended = (int(m.used) - expected) // m.block
        if appended > 0:
            m.note_append(appended=appended)
        return result

    def compact(self, write_index):
        import torch
        from methods.prlmc.consolidate import _select_block_indices
        m = self.m
        start = m.total - m.used
        source = m._memory[:, start:start+self.window].contiguous()
        weights = m._window_importance(start, self.window)
        ages = m.window_block_ages(start, self.window)
        selected = _select_block_indices(source, self.target, m.block, weights, None, m.select_weights,
                                         block_ages=ages)
        if self.carrier_schedule is not None:
            if len(self.events) >= len(self.carrier_schedule):
                raise ValueError("replayed carrier schedule ended early")
            selected = torch.as_tensor(self.carrier_schedule[len(self.events)], device=source.device, dtype=torch.long)
            if selected.shape != (self.target//m.block,) or selected.unique().numel()!=selected.numel() or (selected<0).any() or (selected>=self.window//m.block).any():
                raise ValueError("invalid replayed carrier schedule")
        if self.on_compression:
            self.on_compression("before", write_index, selected)
        if self.merge_arm in ("SharedMerge", "CarrierOnlyFusion"):
            allow = torch.ones(self.window//m.block, dtype=torch.bool, device=source.device)
            if self.merge_arm == "CarrierOnlyFusion":
                allow.zero_(); allow[selected] = True
            begun = time.monotonic()
            output = self.compactor(source, target=self.target, block=m.block,
                carrier_indices=selected, source_allow_mask=allow).to(m.dtype)
            install(m, output, self.window, selected_blocks=selected)
            self.events.append(dict(before_write=write_index, source_slots=self.window,
                target_slots=self.target, remaining_slots=m.used, selected_blocks=selected.cpu().tolist(),
                selected_block_ages=ages[selected].cpu().tolist(),
                seconds=time.monotonic()-begun, fusion=dict(self.compactor.last_fusion)))
        elif self.merge_arm == "Consolidate":
            import torch
            from methods.prlmc.slot_consolidate import consolidate_slots
            begun = time.monotonic()
            output, kept, report = consolidate_slots(
                source, target=self.target, block=m.block, weights=weights)
            output = output.to(m.dtype)
            kept_blocks = sorted({s // m.block for s in report["kept_slots"]})
            install(m, output, self.window, selected_blocks=kept_blocks)
            self.events.append(dict(before_write=write_index, source_slots=self.window,
                target_slots=self.target, remaining_slots=m.used,
                selected_blocks=kept_blocks,
                selected_block_ages=ages[torch.as_tensor(kept_blocks, dtype=torch.long)].cpu().tolist(),
                seconds=time.monotonic() - begun,
                consolidation=report))
        elif self.merge_arm == "Selection" and self.carrier_schedule is not None:
            begun = time.monotonic()
            blocks = source.reshape(m.num_layers, self.window//m.block, m.block, m.hidden)
            output = blocks.index_select(1, selected).flatten(1,2)
            install(m, output, self.window, selected_blocks=selected)
            self.events.append(dict(before_write=write_index, source_slots=self.window,
                target_slots=self.target, remaining_slots=m.used, selected_blocks=selected.cpu().tolist(),
                selected_block_ages=ages[selected].cpu().tolist(),
                seconds=time.monotonic()-begun, carrier_policy="replayed content-only OURS schedule"))
        else:
            super().compact(write_index)
            # `StreamingMemory.compact` keeps its own selection private. It is
            # settled here only when it publishes it, so the fallback arm never
            # invents a carrier set it cannot name.
            published = getattr(self, "last_selected_blocks", None)
            if published is not None:
                m.note_merge(published, self.window // m.block)
        if self.on_compression:
            self.on_compression("after", write_index, selected)


def install(manager, candidate, window, selected_blocks=None):
    """Same suffix placement for inference, training and causal ablation.

    `selected_blocks` are the window-relative native blocks whose content the
    candidate carries. When given, each surviving carrier's age advances by one
    (it survived a round), the uncompressed tail keeps its age, and everything
    else in the pool resets to the young generation (C2). Passing `None` keeps
    the pre-v7 behaviour of leaving the ages untouched (D-12).
    """
    import torch
    m = manager
    target = candidate.shape[1]
    if candidate.shape != (m.num_layers, target, m.hidden) or not torch.isfinite(candidate).all():
        raise ValueError("nonfinite or malformed merge output")
    if not 0 < target < window <= m.used or target % m.block or window % m.block:
        raise ValueError("invalid native-block merge placement")
    start = m.total-m.used
    used_before = int(m.used)
    tail = m._memory[:, start+window:].clone()
    with torch.no_grad():
        replacement = torch.cat((candidate.to(m.dtype), tail), dim=1)
        m.used = replacement.shape[1]
        m._memory[:, m.total-m.used:].copy_(replacement)
        m._tile_free_region()
    if int(m.model.memory_active_mask.sum()) != m.used:
        raise ValueError("installed merge mask/count mismatch")
    if selected_blocks is not None:
        m.note_merge(
            selected_blocks,
            window // m.block,
            target_blocks=target // m.block,
            used_before=used_before,
        )


@contextmanager
def native_capacity(model, blocks):
    """Checkpoint-initialized native capacity baseline; restored even on failure.

    Narrow physical pool and native rotary offsets differ from the managed
    masked pool. This is a system baseline, not a residual-module ablation.
    """
    import torch
    if not bool(model.initialized.item()) or blocks <= 0 or blocks > model.num_blocks:
        raise ValueError("invalid checkpoint-initialized native capacity")
    memory, mask = model.memory, model._buffers.get("memory_active_mask")
    old_blocks = model.num_blocks
    names = ("num_blocks", "num_memory_tokens")
    saved = {name: (hasattr(model.config, name), getattr(model.config, name, None)) for name in names}
    initialized = model.initialized.clone()
    try:
        model.num_blocks = blocks
        model.config.num_blocks = blocks
        model.config.num_memory_tokens = blocks*model.num_tokens
        model.memory = torch.nn.Parameter(memory[:, -blocks*model.num_tokens:].detach().clone(), requires_grad=False)
        model._buffers["memory_active_mask"] = torch.ones(blocks*model.num_tokens, device=memory.device, dtype=torch.bool)
        yield
    finally:
        model.memory = memory
        model.num_blocks = old_blocks
        model._buffers["memory_active_mask"] = mask
        model.initialized.copy_(initialized)
        for name, (present, value) in saved.items():
            if present:
                setattr(model.config, name, value)
            elif hasattr(model.config, name):
                delattr(model.config, name)


def native_write(model, record, index):
    import torch
    ids = torch.tensor([record["token_ids"]], dtype=torch.long, device=model.memory.device)
    attention = torch.ones((1, model.num_tokens+ids.shape[1]), dtype=torch.long, device=ids.device)
    started = time.monotonic()
    with torch.no_grad():
        delta = model.inject_memory(ids, context_attention_mask=attention, update_memory=True)
    if delta.shape[-2] != model.num_tokens or model.memory.shape[1] != model.num_blocks*model.num_tokens:
        raise ValueError("native write shape changed")
    if model.memory_active_mask is not None:
        model.memory_active_mask.fill_(True)
    return dict(index=index, role=record["role"], active_slots=model.memory.shape[1],
        input_tokens=len(record["token_ids"]), input_sha256=record["token_sha256"],
        native_random_evicted_slots=model.num_tokens, seconds=time.monotonic()-started)


V7_ARCHITECTURES = ("grouped_gated_sparse_slot_v7", "grouped_gated_sparse_native_v7")


def v7_merge_classes():
    """Resolve the v7 classes at call time.

    They are defined by the v7 architecture patch in `gated_slot_merge.py` /
    `native_skip_merge.py`; the registry names are declared here so a v7
    checkpoint fails loudly ("unavailable") instead of being re-interpreted as
    some other architecture when those classes are missing.
    """
    from methods.prlmc.gated_slot_merge import SupervisedDSlotMergeV7
    from methods.prlmc.native_skip_merge import NativeSkipSlotMergeV7
    return {"grouped_gated_sparse_slot_v7": SupervisedDSlotMergeV7,
            "grouped_gated_sparse_native_v7": NativeSkipSlotMergeV7}


def build_merge(architecture, num_layers, hidden, block):
    from methods.prlmc.merge import SharedMemoryMerge
    from methods.prlmc.slot_merge import SlotMemoryMerge
    from methods.prlmc.gated_slot_merge import (BudgetedSlotMemoryMerge, GraftedSlotMemoryMerge,
        GraftedSlotMemoryMergeV6, SupervisedDSlotMergeV6, NativeSkipSlotMergeV6)
    from methods.prlmc.native_skip_merge import SupervisedDSlotMerge, NativeSkipSlotMerge
    classes = {"block_v1": SharedMemoryMerge, "discarded_slot_attention_v1": SlotMemoryMerge,
               "budgeted_dense_slot_v3": BudgetedSlotMemoryMerge,
               "grouped_gated_sparse_slot_v3": GraftedSlotMemoryMerge,
               # v4 is retired: its checkpoints were trained with layer-0 carrier
               # selection, learned-only routing and a single shared native gain,
               # none of which this code reproduces. Keeping the name out of the
               # registry makes reusing an old checkpoint a loud error instead of
               # a silent re-interpretation.
               "grouped_gated_sparse_slot_v5": SupervisedDSlotMerge,
               "grouped_gated_sparse_native_v5": NativeSkipSlotMerge,
               # v6 stays registered for read-only reproduction of its
               # checkpoints; v7 is the main line and never loads a v6 state.
               "grouped_gated_sparse_slot_v6": GraftedSlotMemoryMergeV6,
               "grouped_gated_sparse_native_v6": NativeSkipSlotMergeV6}
    if architecture in V7_ARCHITECTURES:
        try:
            classes.update(v7_merge_classes())
        except ImportError as error:
            raise ValueError(
                "v7 merge classes are unavailable in this checkout; the v7 "
                "architecture patch must land before a v7 checkpoint can load"
            ) from error
    if architecture not in classes:
        raise ValueError("unknown shared merge architecture")
    kwargs = dict(rank=32, gamma_max=.05)
    training = training_for(architecture)
    if training is not None:
        spec = next(v for v in training["variants"].values() if v["architecture"] == architecture)
        kwargs.update(gamma_max=spec["local_gate_max"], initial_gate=training["initial_gate"],
                      global_rms_budget=training["global_rms_budget"])
        if architecture != "budgeted_dense_slot_v3":
            kwargs.update(num_groups=spec["groups"], reference_layer=spec["reference_layer"], topk=spec["topk"])
        if architecture.endswith(("_v5", "_v6", "_v7")):
            kwargs.update(native_prior_weight=training["native_prior_weight"],
                          route_mix_temperature=training["route_mix_temperature"],
                          balance_strength=training["balance_strength"])
    return classes[architecture](num_layers, hidden, block, **kwargs)


def training_for(architecture):
    """Sealed supervision record a checkpoint must reproduce, or None for baselines."""
    if architecture.endswith("_v3"):
        from prepare_natural_merge_v3 import TRAIN_V3 as training
        return training
    if architecture.endswith(("_v4", "_v5")):
        from prepare_natural_merge_v4 import TRAIN_V4 as training
        return training
    if architecture.endswith("_v6"):
        from prepare_natural_merge_v6 import TRAIN_V6 as training
        return training
    if architecture.endswith("_v7"):
        from prepare_natural_merge_v7 import TRAIN_V7 as training
        return training
    return None


def _unverified_warning(reason):
    print(f"WARNING: {reason}; loading anyway because strict_code_hash=False. "
          "This is historical diagnosis only, never evaluation-eligible evidence.",
          file=sys.stderr)


def _verify_source_manifest(saved, strict_code_hash):
    """Compare the checkpoint's recorded operator hashes against the live code.

    Returns True when the manifest was verified, False when the checkpoint
    carries no provenance record. Raises ValueError on any mismatch when
    strict_code_hash is set; otherwise warns and returns False.
    """
    manifest = saved.get("source_manifest")
    # Metadata-only dicts (no weights) are structural fixtures used by the CPU
    # contract tests; provenance can only be checked for executable
    # checkpoints, and every real checkpoint carries "state_dict".
    if manifest is None and "state_dict" not in saved:
        return False
    if manifest is None:
        reason = ("checkpoint has no source_manifest; its operator code cannot be "
                  "verified (v5 and earlier checkpoints predate the manifest)")
        if strict_code_hash:
            raise ValueError(reason + "; pass strict_code_hash=False for historical "
                                      "diagnosis only")
        _unverified_warning(reason)
        return False
    if not isinstance(manifest, dict):
        raise ValueError("checkpoint source_manifest is not a file->sha256 mapping")
    mismatches = []
    for name in CRITICAL_MERGE_FILES:
        code = sha(ROOT/name)
        recorded = manifest.get(name)
        if recorded is None:
            mismatches.append(f"{name}: ckpt <missing> != code {code[:8]}")
        elif recorded != code:
            mismatches.append(f"{name}: ckpt {recorded[:8]} != code {code[:8]}")
    if mismatches:
        detail = "checkpoint/source code mismatch: " + "; ".join(mismatches)
        if strict_code_hash:
            raise ValueError(detail)
        _unverified_warning(detail)
        return False
    return True


def validate_merge_checkpoint(saved, architecture, strict_code_hash=True):
    """Structural checks plus (by default) operator-source hash verification.

    strict_code_hash=False only exists for historical diagnosis: it downgrades
    manifest problems to a stderr WARNING and callers must then treat the
    checkpoint as evaluation_eligible=False.
    """
    if saved.get("effective_gamma_max", .05) != .05 or saved.get("evaluation_eligible") is False:
        raise ValueError("ineligible/gamma-mismatched checkpoint")
    verified = _verify_source_manifest(saved, strict_code_hash)
    training = training_for(architecture)
    if training is not None:
        variant = saved.get("variant")
        if (variant not in training["variants"] or saved.get("training") != training
                or saved.get("module_configuration") != training["variants"][variant]
                or saved["module_configuration"]["architecture"] != architecture
                or saved.get("global_rms_budget") != training["global_rms_budget"]
                or saved.get("rank") != training["rank"]):
            raise ValueError("C/D checkpoint architecture/training/budget mismatch")
    return verified


def resolve_architecture(saved, requested=None, default="block_v1"):
    """Architecture actually in use: checkpoint metadata wins over the default.

    Evaluation entry points must not pin a generation: a DsupV6/DnativeV6
    checkpoint has to be evaluated (and reported) as v6, otherwise the v5
    default trips the architecture check below. `requested` is only for callers
    that want to force a generation; a disagreement with the checkpoint raises.
    """
    saved_architecture = (saved or {}).get("architecture")
    if requested and saved_architecture and requested != saved_architecture:
        raise ValueError("requested/checkpoint architecture mismatch")
    return saved_architecture or requested or default


def check_similarity_margin(saved, merge):
    """Refuse a checkpoint trained under a different similarity margin.

    The margin is a sealed-plan constant (`SIM_MARGIN_V6`), not a CLI knob: a
    checkpoint records the value it was trained with, so evaluating it with a
    different built-in default would silently change the routing mask.
    """
    recorded = (saved or {}).get("sim_margin")
    live = getattr(merge, "sim_margin", None)
    if recorded is None or live is None:
        return
    if float(recorded) != float(live):
        raise ValueError(
            "checkpoint/built similarity margin mismatch: trained with "
            f"{float(recorded)}, module default is {float(live)}")


def load_merge_state(merge, saved, architecture, strict_code_hash=True):
    """Validate provenance/architecture *before* any weight reaches the module."""
    verified = validate_merge_checkpoint(saved, architecture, strict_code_hash=strict_code_hash)
    merge.load_state_dict(saved["state_dict"], strict=True)
    return verified


def load_runtime(model_path, checkpoint=None, train=False, architecture=None,
                 budget_blocks=20, strict_code_hash=True, window_blocks=None,
                 target_blocks=None):
    import torch
    import yaml
    from transformers import AutoTokenizer
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory
    from methods.prlmc.merge import SharedMemoryMerge
    from methods.prlmc.slot_merge import SlotMemoryMerge
    torch.set_num_threads(4)
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = MemoryLLM.from_pretrained(model_path, torch_dtype=torch.float16,
        attn_implementation="sdpa", local_files_only=True).to("cuda:0").eval().requires_grad_(False)
    if int(model.initialized) != 1 or not hasattr(model, "memory_active_mask"):
        raise ValueError("initialized checkpoint and active-mask reader fork required")
    if float(getattr(model, "memory_attn_scale", 1.)) != 1.:
        raise ValueError("reader attention gain must stay one")
    cfg = yaml.safe_load((ROOT/"configs/agent_conf/prlmc_e6.yaml").read_text())["prlmc"]
    if budget_blocks not in (8, 10, 12, 20):
        raise ValueError("budget_blocks must be one of 8/10/12/20")
    cfg.update(k_anchor=0, anchor_source="none", k_max=budget_blocks*model.num_tokens,
        device="cuda:0",
        dtype="float16", active_memory_mask=True, trigger="none", compact_mode="carrier_residual",
        projection="none", basis_mode="none", compact_probe=False, stats_dir=None,
        carrier_gamma_max=.05, carrier_rank=32, write_mode="append", gate_mode="ones",
        functional_control="none", carrier_ckpt=None)
    manager = ProtectedResidualMemory(model, tokenizer, PRLMCConfig.from_dict(cfg))
    # Resolve the streaming geometry here as well, so the caller can hand the
    # same numbers to `MergeStream` instead of re-deriving them (C8): `None`
    # keeps the pre-registered table, and a conflict with the budget or the
    # physical pool fails at load time rather than on the first compaction.
    geometry = resolve_geometry(
        budget_blocks, window_blocks, target_blocks, total_blocks=manager.total // manager.block
    )
    cfg.update(window_blocks=geometry[0], target_blocks=geometry[1])
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True) if checkpoint else None
    architecture = resolve_architecture(saved, architecture)
    merge = build_merge(architecture, manager.num_layers, manager.hidden, manager.block).to("cuda:0")
    if checkpoint:
        check_similarity_margin(saved, merge)
        verified = load_merge_state(merge, saved, architecture, strict_code_hash=strict_code_hash)
        # Downstream reports must treat an unverified checkpoint as ineligible.
        saved["code_hash_verified"] = verified
    merge.train(train).requires_grad_(train)
    initial = model.memory.detach().cpu().clone()
    return model, tokenizer, manager, merge, initial, cfg, saved


def reset(manager, initial):
    import torch
    model = manager.model
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    with torch.no_grad():
        model.memory.copy_(initial.to(model.memory.device))
        model.initialized.fill_(1)
        model.memory_active_mask.fill_(True)
    manager.used = 0
    manager.n_consolidated = 0
    manager.goal_direction = None
    manager._active_mask_used = None
    manager.reset_block_ages()


def costs(manager, merge):
    ages = manager.block_age_summary()
    return dict(physical_slots=manager.total, effective_budget_slots=manager.layout.k_max,
        physical_pool_bytes=manager._memory.numel()*manager._memory.element_size(),
        shared_parameters=sum(p.numel() for p in merge.parameters()),
        shared_state_bytes=sum(t.numel()*t.element_size() for t in merge.state_dict().values()),
        # Generational diagnostics (v7 C1): how old the resident content is.
        block_age_max=ages["block_age_max"], old_generation_slots=ages["old_generation_slots"],
        claim="active-slot budget reduction; physical memory allocation is unchanged")


def model_identity(model_path):
    path = Path(model_path)
    manifest = path/"fp16_cache_manifest.json"
    if not manifest.is_file():
        raise ValueError("certified FP16 cache manifest required")
    return dict(cache_manifest_sha256=sha(manifest), config_sha256=sha(path/"config.json"),
                tokenizer=tokenizer_manifest(path))
