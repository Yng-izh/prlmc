"""Shared natural-merge runtime; no changes to the live author512 entrypoint."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_utils import digest, sha, tokenizer_manifest
from stream_eval import StreamingMemory, budget_geometry, generate, E6D_SHA

# Files whose content a new checkpoint records as its provenance. The operator
# files below are the ones the V10 numerics actually come from; the three
# historical names (`slot_merge`, `gated_slot_merge`, `native_skip_merge`) are
# hash-pinned by `full_v10.pt`, which is why they are still listed.
SOURCE_FILES = ["configs/agent_conf/prlmc_e6.yaml", "methods/prlmc/config.py",
    "methods/prlmc/importance.py", "methods/prlmc/capacity.py", "methods/prlmc/subspace.py",
    "methods/prlmc/consolidate.py", "methods/prlmc/layout.py", "methods/prlmc/probes.py",
    "methods/prlmc/manifold_graft.py", "methods/prlmc/slot_merge.py",
    "methods/prlmc/gated_slot_merge.py", "methods/prlmc/native_skip_merge.py",
    "methods/memoryllm_modeling/modeling_memoryllm.py",
    "tools/io_utils.py", "tools/stream_eval.py", "tools/v9_geometry_50block.py",
    "tools/plan.py", "tools/objective.py", "tools/train.py", "tools/runtime.py",
    "tools/eval_core.py", "tools/eval_lbv2.py", "tools/lbv2_data.py", "tools/native_eval.py"]


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


def planned_compactions(total_writes, block, window_blocks, target_blocks, trigger_blocks):
    """Number of compactions a stream of `total_writes` blocks performs.

    Mirrors `StreamingMemory.write`: compaction fires when the resident width is
    at least the window and at least the trigger, and returns the stream to
    `target + (used - window)` blocks. `--fusion-policy last` needs this count
    before the first write, because "the last compaction" is only identifiable
    once the whole write stream is known.
    """
    used, count = 0, 0
    window, target = window_blocks * block, target_blocks * block
    trigger = trigger_blocks * block
    for _ in range(int(total_writes)):
        if used >= window and used >= trigger:
            count += 1
            used = target + (used - window)
        used += block
    return count


class MergeStream(StreamingMemory):
    """Streaming compaction with an explicit fusion policy.

    `fusion_policy="all"` fuses at every compaction (the full V10 arm).
    `fusion_policy="last"` fuses only at the final compaction and runs the
    Selection path everywhere else: it isolates the effect of one merge from the
    cumulative effect of ~20 repeated merges.

    `write_provenance` records, for every resident block, the write index it
    descends from, so a merge can name exactly which writes it dropped.
    """

    def __init__(self, manager, compactor, arm, on_compression=None, carrier_schedule=None,
                 budget_blocks=20, window_blocks=None, target_blocks=None,
                 trigger_blocks=None, fusion_policy="all", total_writes=None):
        aliases = {"SharedMerge": "SharedE6d"}
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
        if fusion_policy not in ("all", "last"):
            raise ValueError(f"unknown fusion_policy {fusion_policy!r}")
        if fusion_policy == "last" and total_writes is None:
            raise ValueError("fusion_policy 'last' needs the number of writes up "
                             "front; otherwise the final compaction is unknown")
        self.fusion_policy = fusion_policy
        self.planned_compaction_count = (None if total_writes is None else
            planned_compactions(total_writes, manager.block, geometry[0], geometry[1],
                                self.effective_trigger_blocks))
        self.write_provenance = []

    def _fuses_now(self):
        if self.fusion_policy == "all":
            return True
        return len(self.events) == self.planned_compaction_count - 1

    def write(self, record, index):
        """Streaming append with generational and provenance bookkeeping.

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
            result = super().write(record, index)
            self._note_written(index, result)
            return result
        used_before = int(m.used)
        merges_before = len(self.events)
        result = super().write(record, index)
        if m.age_append_events == events_before and self.arm != "NativeDefault":
            merged = len(self.events) - merges_before
            expected = used_before - (self.window - self.target) * merged
            appended = (int(m.used) - expected) // m.block
            if appended > 0:
                m.note_append(appended=appended)
        self._note_written(index, result)
        return result

    def _note_written(self, index, result):
        # The published baseline overwrites the whole pool and reports
        # `used = total`; there is no per-block provenance to carry there.
        if result is not None and self.arm != "NativeDefault":
            self.write_provenance.append(int(index))

    def _settle_provenance(self, selected, fused, write_index):
        """Name the writes this merge kept and dropped, oldest window first."""
        window_blocks = self.window // self.m.block
        provenance = self.write_provenance
        if len(provenance) < window_blocks:
            return
        keep = set(int(i) for i in selected)
        window_writes = list(provenance[:window_blocks])
        kept = [window_writes[i] for i in range(window_blocks) if i in keep]
        dropped = [window_writes[i] for i in range(window_blocks) if i not in keep]
        self.write_provenance = kept + list(provenance[window_blocks:])
        self.events[-1].update(fused=bool(fused), window_writes=window_writes,
                               kept_writes=kept, dropped_writes=dropped)

    def compact(self, write_index):
        import torch
        m = self.m
        start = m.total - m.used
        source = m._memory[:, start:start+self.window].contiguous()
        weights = m._window_importance(start, self.window)
        ages = m.window_block_ages(start, self.window)
        # v9: freeze mask + archive pinning for exactly this window, then the
        # two-stage anchor-bridge selection when the operator asks for it.
        if hasattr(m, "fused_window"):
            frozen, pinned = m.fused_window(start, self.window)
            m.pinned_blocks = pinned
        else:  # lighter managers used by the CPU contracts carry no lock
            frozen, pinned = None, None
        from methods.prlmc.consolidate import select_window_carrier_blocks
        picked = select_window_carrier_blocks(
            window_memory=source,
            target_blocks=self.target // m.block,
            block_size=m.block,
            block_ages=ages,
            condensed_slot_mask=None,
        )
        selected = torch.as_tensor(picked, device=source.device, dtype=torch.long)
        if self.carrier_schedule is not None:
            if len(self.events) >= len(self.carrier_schedule):
                raise ValueError("replayed carrier schedule ended early")
            selected = torch.as_tensor(self.carrier_schedule[len(self.events)], device=source.device, dtype=torch.long)
            if selected.shape != (self.target//m.block,) or selected.unique().numel()!=selected.numel() or (selected<0).any() or (selected>=self.window//m.block).any():
                raise ValueError("invalid replayed carrier schedule")
        if self.on_compression:
            self.on_compression("before", write_index, selected)
        fuse = self._fuses_now()
        if self.merge_arm == "SharedMerge" and fuse:
            allow = torch.ones(self.window//m.block, dtype=torch.bool, device=source.device)
            begun = time.monotonic()
            # Only the V10 operator declares `bridge_blocks`; a sealed reference
            # operator keeps its exact call signature here.
            v9_kwargs = (dict(frozen_slots=frozen, pinned_blocks=pinned)
                         if getattr(self.compactor, "bridge_blocks", None) is not None else {})
            output = self.compactor(source, target=self.target, block=m.block,
                carrier_indices=selected, source_allow_mask=allow, **v9_kwargs).to(m.dtype)
            install(m, output, self.window, selected_blocks=selected,
                    fused_offsets=None)  # V10 re-derives graft slots each merge; no write-once lock
            self.events.append(dict(before_write=write_index, source_slots=self.window,
                target_slots=self.target, remaining_slots=m.used, selected_blocks=selected.cpu().tolist(),
                selected_block_ages=ages[selected].cpu().tolist(),
                seconds=time.monotonic()-begun, fusion=dict(self.compactor.last_fusion)))
            settled = selected.cpu().tolist()
        elif self.carrier_schedule is not None:
            # Selection with the recorded carrier set: no fusion, same blocks.
            begun = time.monotonic()
            blocks = source.reshape(m.num_layers, self.window//m.block, m.block, m.hidden)
            output = blocks.index_select(1, selected).flatten(1,2)
            install(m, output, self.window, selected_blocks=selected)
            self.events.append(dict(before_write=write_index, source_slots=self.window,
                target_slots=self.target, remaining_slots=m.used, selected_blocks=selected.cpu().tolist(),
                selected_block_ages=ages[selected].cpu().tolist(),
                seconds=time.monotonic()-begun, carrier_policy="replayed content-only OURS schedule"))
            settled = selected.cpu().tolist()
        else:
            super().compact(write_index)
            # `StreamingMemory.compact` keeps its own selection private; it now
            # publishes it, so the fallback arm never invents a carrier set.
            settled = getattr(self, "last_selected_blocks", None) or []
        self._settle_provenance(settled, fuse, write_index)
        if self.on_compression:
            self.on_compression("after", write_index, selected)


def install(manager, candidate, window, selected_blocks=None, fused_offsets=None):
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
    # v9 M4: every install moves the pool, so the write-once flags move here --
    # the one place inference, training and ablation all pass through. Then the
    # bridge slots that just took a high-amplitude write are frozen for good.
    if hasattr(m, "remap_fused"):
        new_start = m.total - (used_before - window + target)
        m.remap_fused(start, window, target, new_start, selected_blocks, used_before)
        m.mark_fused_slots(fused_offsets or [], base=new_start)


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


ARCHITECTURE = "grouped_gated_sparse_native_v9"


def build_merge(architecture, num_layers, hidden, block, fusion_mode="sealed", training=None):
    """Build the V10 manifold-grafting operator.

    This checkout implements exactly one architecture. A checkpoint trained by an
    earlier generation fails here with a named error instead of being silently
    re-interpreted by a similar-looking operator.
    """
    from methods.prlmc.native_skip_merge import NativeSkipSlotMergeV10
    if architecture != ARCHITECTURE:
        raise ValueError(f"unknown shared merge architecture {architecture!r}; this "
                         f"checkout implements only {ARCHITECTURE!r} (V10 manifold grafting)")
    if fusion_mode != "sealed":
        raise ValueError(f"unknown fusion_mode {fusion_mode!r}")
    kwargs = dict(rank=32, gamma_max=.05)
    # Callers that own a training record (the paper geometry) hand it in directly;
    # otherwise fall back to the sealed record registered for the architecture.
    training = training if training is not None else training_for(architecture)
    if training is not None:
        spec = next(v for v in training["variants"].values() if v["architecture"] == architecture)
        kwargs.update(gamma_max=spec["local_gate_max"], initial_gate=training["initial_gate"],
                      global_rms_budget=training["global_rms_budget"],
                      num_groups=spec["groups"], reference_layer=spec["reference_layer"],
                      topk=spec["topk"],
                      top_sharp=spec.get("top_sharp", 1),
                      route_temperature_sharp=spec.get("route_temperature_sharp", .1),
                      bridge_fraction=spec.get("bridge_fraction", .156),
                      bridge_gamma_max=spec.get("bridge_gamma_max", .6),
                      significance_affinity_weight=spec.get("significance_affinity_weight", .5),
                      ngram_kernel=tuple(training.get("ngram_kernel", (.25, .5, .25))),
                      bridge_blocks=int(training.get("bridge_blocks", 3)),
                      native_init=float(training.get("native_init", 0.)),
                      native_prior_weight=training["native_prior_weight"],
                      route_mix_temperature=training["route_mix_temperature"],
                      balance_strength=training["balance_strength"])
    return NativeSkipSlotMergeV10(num_layers, hidden, block, **kwargs)


def training_for(architecture, content_sha256=None):
    """The sealed supervision record this architecture must reproduce.

    Resolved from the on-disk plan file the checkpoint names. Returns None when
    that file is not available, in which case the record embedded in the
    checkpoint is used for construction and the deep comparison is skipped.
    """
    if architecture != ARCHITECTURE:
        return None
    from plan import sealed_training
    return sealed_training(content_sha256)


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
    training = training_for(architecture, saved.get("train_content_sha256"))
    if training is not None:
        variant = saved.get("variant")
        recorded = saved.get("training") or {}
        # Every sealed field must match. Run-time knobs the trainer adds after the
        # plan is sealed (e.g. `qa_batch`, a batching choice with no effect on the
        # objective) are allowed to differ.
        sealed_mismatch = {k: (recorded.get(k), training[k])
                           for k in training if recorded.get(k) != training[k]}
        if (variant not in training["variants"] or sealed_mismatch
                or saved.get("module_configuration") != training["variants"][variant]
                or saved["module_configuration"]["architecture"] != architecture
                or saved.get("global_rms_budget") != training["global_rms_budget"]
                or saved.get("rank") != training["rank"]):
            raise ValueError("C/D checkpoint architecture/training/budget mismatch: "
                             f"{sorted(sealed_mismatch)}")
    return verified


def resolve_architecture(saved, requested=None, default=ARCHITECTURE):
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
                 target_blocks=None, fusion_mode="sealed", training=None):
    import torch
    import yaml
    from transformers import AutoTokenizer
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory
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
    if budget_blocks not in (8, 10, 12, 20, 50):
        raise ValueError("budget_blocks must be one of 8/10/12/20/50")
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
    merge = build_merge(architecture, manager.num_layers, manager.hidden, manager.block,
                        fusion_mode=fusion_mode, training=training).to("cuda:0")
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
