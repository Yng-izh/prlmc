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
from public_retention_eval import StreamingMemory, generate, E6D_SHA

SOURCE_FILES = ["configs/agent_conf/prlmc_e6.yaml", "methods/prlmc/config.py",
    "methods/prlmc/importance.py", "methods/prlmc/capacity.py", "methods/prlmc/subspace.py",
    "tools/native_qa_compare.py", "methods/prlmc/merge.py", "methods/prlmc/consolidate.py", "methods/prlmc/layout.py",
    "methods/prlmc/probes.py", "tools/natural_merge_runtime.py", "tools/prepare_natural_merge.py",
    "tools/natural_merge_eval.py", "tools/natural_merge_train.py", "tools/verify_natural_merge.py",
    "tools/public_retention_eval.py", "tools/prepare_public_retention.py",
    "tools/freeze_natural_merge.py", "tools/run_natural_merge_parallel.py",
    "methods/memoryllm_modeling/modeling_memoryllm.py"]


def source_manifest():
    return {name: sha(ROOT/name) for name in SOURCE_FILES}


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


class MergeStream(StreamingMemory):
    def __init__(self, manager, compactor, arm, on_compression=None, carrier_schedule=None):
        aliases = {"SharedMerge": "SharedE6d", "CarrierOnlyFusion": "SharedE6d"}
        super().__init__(manager, compactor, aliases.get(arm, arm))
        self.merge_arm = arm
        self.on_compression = on_compression
        self.carrier_schedule = carrier_schedule

    def compact(self, write_index):
        import torch
        from methods.prlmc.consolidate import _select_block_indices
        m = self.m
        start = m.total - m.used
        source = m._memory[:, start:start+self.window].contiguous()
        weights = m._window_importance(start, self.window)
        selected = _select_block_indices(source, self.target, m.block, weights, None, m.select_weights)
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
            install(m, output, self.window)
            self.events.append(dict(before_write=write_index, source_slots=self.window,
                target_slots=self.target, remaining_slots=m.used, selected_blocks=selected.cpu().tolist(),
                seconds=time.monotonic()-begun, fusion=dict(self.compactor.last_fusion)))
        elif self.merge_arm == "Selection" and self.carrier_schedule is not None:
            begun = time.monotonic()
            blocks = source.reshape(m.num_layers, self.window//m.block, m.block, m.hidden)
            output = blocks.index_select(1, selected).flatten(1,2)
            install(m, output, self.window)
            self.events.append(dict(before_write=write_index, source_slots=self.window,
                target_slots=self.target, remaining_slots=m.used, selected_blocks=selected.cpu().tolist(),
                seconds=time.monotonic()-begun, carrier_policy="replayed content-only OURS schedule"))
        else:
            super().compact(write_index)
        if self.on_compression:
            self.on_compression("after", write_index, selected)


def install(manager, candidate, window):
    """Same suffix placement for inference, training and causal ablation."""
    import torch
    m = manager
    target = candidate.shape[1]
    if candidate.shape != (m.num_layers, target, m.hidden) or not torch.isfinite(candidate).all():
        raise ValueError("nonfinite or malformed merge output")
    if not 0 < target < window <= m.used or target % m.block or window % m.block:
        raise ValueError("invalid native-block merge placement")
    start = m.total-m.used
    tail = m._memory[:, start+window:].clone()
    with torch.no_grad():
        replacement = torch.cat((candidate.to(m.dtype), tail), dim=1)
        m.used = replacement.shape[1]
        m._memory[:, m.total-m.used:].copy_(replacement)
        m._tile_free_region()
    if int(m.model.memory_active_mask.sum()) != m.used:
        raise ValueError("installed merge mask/count mismatch")


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


def load_runtime(model_path, checkpoint=None, train=False):
    import torch
    import yaml
    from transformers import AutoTokenizer
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory
    from methods.prlmc.merge import SharedMemoryMerge
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
    cfg.update(k_anchor=0, anchor_source="none", k_max=20*model.num_tokens, device="cuda:0",
        dtype="float16", active_memory_mask=True, trigger="none", compact_mode="carrier_residual",
        projection="none", basis_mode="none", compact_probe=False, stats_dir=None,
        carrier_gamma_max=.05, carrier_rank=32, write_mode="append", gate_mode="ones",
        functional_control="none", carrier_ckpt=None)
    manager = ProtectedResidualMemory(model, tokenizer, PRLMCConfig.from_dict(cfg))
    merge = SharedMemoryMerge(manager.num_layers, manager.hidden, manager.block, rank=32, gamma_max=.05).to("cuda:0")
    saved = None
    if checkpoint:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        merge.load_state_dict(saved["state_dict"], strict=True)
        if saved.get("effective_gamma_max", .05) != .05 or saved.get("evaluation_eligible") is False:
            raise ValueError("ineligible/gamma-mismatched checkpoint")
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


def costs(manager, merge):
    return dict(physical_slots=manager.total, effective_budget_slots=20*manager.block,
        physical_pool_bytes=manager._memory.numel()*manager._memory.element_size(),
        shared_parameters=sum(p.numel() for p in merge.parameters()),
        shared_state_bytes=sum(t.numel()*t.element_size() for t in merge.state_dict().values()),
        claim="active-slot budget reduction; physical memory allocation is unchanged")


def model_identity(model_path):
    path = Path(model_path)
    manifest = path/"fp16_cache_manifest.json"
    if not manifest.is_file():
        raise ValueError("certified FP16 cache manifest required")
    return dict(cache_manifest_sha256=sha(manifest), config_sha256=sha(path/"config.json"),
                tokenizer=tokenizer_manifest(path))
