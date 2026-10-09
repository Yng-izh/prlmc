"""Diagnose the identity-at-init failure on the held-out window (2,3450).

Reproduces the exact sequence: build window -> evaluate d0 -> run N Jacobian
cycles (_pool_in_place + restore) -> evaluate again. Prints where the
correctness map first diverges, and whether the pool bytes are bit-identical
across a cycle.
"""
import argparse
import os
import sys

from s1_local_protected import digest

p = argparse.ArgumentParser()
p.add_argument("--carrier-ckpt", required=True)
p.add_argument("--out", required=True)
args = p.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
os.environ["S0B_GPU_SET"] = "1"

import torch  # noqa: E402
import yaml  # noqa: E402
import e6a_overfit as E  # noqa: E402
from s0_functional_sensitivity import (build_window_and_probes,  # noqa: E402
                                       margin_scalar, CONFIG, MODEL)
from methods.prlmc.config import PRLMCConfig  # noqa: E402
from methods.prlmc.consolidate import CarrierResidualCompactor  # noqa: E402
from methods.prlmc.layout import ProtectedResidualMemory  # noqa: E402
from methods.prlmc.probes import (_pool_in_place, _compacted_mask,  # noqa: E402
                                  active_mask_override, differentiable_memory)
from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from contextlib import nullcontext  # noqa: E402

torch.manual_seed(0)
with open(CONFIG) as f:
    config = dict(yaml.safe_load(f)["prlmc"])
config.update(device="cuda:0", dtype="float16", stats_dir="./outputs/prlmc-s1c-diag")
cfg = PRLMCConfig.from_dict(config)
model = MemoryLLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                  attn_implementation="sdpa").to("cuda:0")
model.initialized.zero_()
for x in model.parameters():
    x.requires_grad_(False)
model.train()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
manager = ProtectedResidualMemory(model, AutoTokenizer.from_pretrained(MODEL), cfg)
ns = argparse.Namespace(probes=60, max_probe_tokens=256)
state = torch.load(args.carrier_ckpt, map_location="cuda:0", weights_only=True)
window_used = E.WINDOW_BLOCKS * manager.block
target_used = E.TARGET_BLOCKS * manager.block

lines = E.read_fact_lines(2)
built = build_window_and_probes(manager, {2: lines}, 2, 3450, ns)
assert built is not None
covered, indices, dropped, original = built
by_q = {q.question: q for q in original}
by_q.update({q.question: q for q in
             E.probes_from_lines(lines, indices, len(indices) + 1)})
probes = list(by_q.values())
mask = _compacted_mask(manager, manager.used - window_used + target_used, "cuda:0")

encoded = {}
for q in probes:
    stem = q.prompt if q.prompt else f"{q.question}\nAnswer:"
    tokens = manager.tokenizer(f"{stem} {q.answers[0]}", return_tensors="pt",
                               add_special_tokens=False).input_ids
    stem_len = len(manager.tokenizer(stem, add_special_tokens=False).input_ids)
    encoded[q.question] = (tokens[:, :256].to("cuda:0"), stem_len)


def base():
    arm = CarrierResidualCompactor(
        num_layers=manager.num_layers, hidden=manager.hidden,
        block=manager.block, rank=int(cfg.carrier_rank),
        gamma_max=float(cfg.carrier_gamma_max)).to(device="cuda:0", dtype=torch.float32)
    arm.load_state_dict(state["state_dict"])
    for x in arm.parameters():
        x.requires_grad_(False)
    return arm


def candidate(arm):
    source = manager._memory[:, manager.total - manager.used:manager.total, :].to("cuda:0").clone()
    weights = manager._window_importance(manager.total - manager.used, window_used)
    return arm(source, weights, target=target_used, block=manager.block,
               goal_direction=manager.goal_direction,
               select_weights=manager.select_weights)


@torch.no_grad()
def evaluate(arm=None):
    correct = {}
    pool, mask_m = (None, None) if arm is None else E.candidate_pool(manager, arm, "cuda:0")
    context = differentiable_memory(model, pool) if pool is not None else nullcontext()
    guard = active_mask_override(model, mask_m) if mask_m is not None else nullcontext()
    with context, guard:
        for q in probes:
            input_ids, stem_len = encoded[q.question]
            logits = model(input_ids=input_ids, return_dict=True).logits
            span = logits[:, stem_len - 1:-1, :]
            labels = input_ids[:, stem_len:]
            correct[q.question] = bool((span.argmax(dim=-1) == labels).all(dim=-1).item())
    del pool, mask_m
    torch.cuda.empty_cache()
    return correct


d0 = evaluate(base())
print("[diag] d0 correct: %d/%d" % (sum(d0.values()), len(d0)), flush=True)

# bit-identity of the pool across one Jacobian cycle
arm = base()
arm.down.weight.requires_grad_(True)
first_q = next(iter(by_q))
before = manager.model.memory.detach().cpu().clone()
cand = candidate(arm)
with _pool_in_place(manager, "cuda:0", cand, window_used, target_used) as injection:
    pool = injection.reinject()
    scalar = margin_scalar(manager, by_q[first_q], pool, mask, 256)
scalar.backward()
after = manager.model.memory.detach().cpu()
print("[diag] pool bit-identical after 1 cycle:",
      bool(torch.equal(before, after)), flush=True)
del before, after
arm.down.weight.grad = None
arm.down.weight.requires_grad_(False)
del scalar, cand, pool
torch.cuda.empty_cache()

# also: does a plain re-evaluate without any cycle still match?
recheck = evaluate(base())
print("[diag] re-eval without cycle: diffs =",
      [q for q in d0 if recheck[q] != d0[q]][:5], flush=True)
print("[diag] done", flush=True)
