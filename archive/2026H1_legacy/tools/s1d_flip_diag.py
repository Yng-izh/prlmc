"""Diagnose identity-at-init flip on dev window (3,16370).

Replicates the exact driver sequence in s1d_stage2_dev.py and instruments it:
  A. same arm (d0) evaluated twice back-to-back  -> forward determinism
  B. d0 vs freshly parametrized arm (value=0)     -> the failing comparison
  C. evaluate_generation(d0) sandwiched between
     two evaluate(d0) calls                        -> generation side effects
  D. bitwise pool comparison d0 vs arm             -> compactor identity
  E. RNG state / manager.used / memory checksum at every step
Writes a JSON report and prints progress.
"""
import argparse
import gc
import json
import os
import sys

from s1_local_protected import digest

p = argparse.ArgumentParser()
p.add_argument("--carrier-ckpt", default="runs/e6d_interleave_ckpt.pt")
p.add_argument("--selfsup-rows", default="runs/s1d_selfsup_rows_v2.pt")
p.add_argument("--rows", default="runs/s1c_global_rows.pt")
p.add_argument("--row", type=int, default=3)
p.add_argument("--start", type=int, default=16370)
p.add_argument("--cuda-device", type=int, default=1)
p.add_argument("--max-new-tokens", type=int, default=32)
p.add_argument("--out", required=True)
args = p.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
os.environ["S0B_GPU_SET"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch  # noqa: E402
import yaml  # noqa: E402
import e6a_overfit as E  # noqa: E402
from s0_functional_sensitivity import build_window_and_probes, CONFIG, MODEL  # noqa: E402
from methods.prlmc.config import PRLMCConfig  # noqa: E402
from methods.prlmc.consolidate import CarrierResidualCompactor  # noqa: E402
from methods.prlmc.layout import ProtectedResidualMemory  # noqa: E402
from methods.prlmc.probes import (_compacted_mask, differentiable_memory,  # noqa: E402
                                  active_mask_override)
from s1c_wash import make_arm_displacement  # noqa: E402
from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from torch.nn.utils import parametrize  # noqa: E402
from contextlib import nullcontext  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(4)
with open(CONFIG) as f:
    config = dict(yaml.safe_load(f)["prlmc"])
config.update(device="cuda:0", dtype="float16",
              stats_dir="./outputs/prlmc-s1d-flip-diag")
cfg = PRLMCConfig.from_dict(config)
model = MemoryLLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                  attn_implementation="sdpa").to("cuda:0")
model.initialized.zero_()
for x in model.parameters():
    x.requires_grad_(False)
model.train()
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False})
manager = ProtectedResidualMemory(model, AutoTokenizer.from_pretrained(MODEL), cfg)
from s1_local_protected import disable_reader_dropout  # noqa: E402
disable_reader_dropout(model)
probe_args = argparse.Namespace(probes=60, max_probe_tokens=256)
state = torch.load(args.carrier_ckpt, map_location="cuda:0", weights_only=True)
ss_cache = torch.load(args.selfsup_rows, map_location="cpu", weights_only=False)
gold = torch.load(args.rows, map_location="cpu", weights_only=False)
window_used = E.WINDOW_BLOCKS * manager.block
target_used = E.TARGET_BLOCKS * manager.block


def orth(rows):
    """SVD-ordered orthonormal basis, identical to s1d_stage2_dev.orth."""
    _u, s, vh = torch.linalg.svd(rows.double(), full_matrices=False)
    keep = s > max(1e-10, float(s.max()) * 1e-7)
    return vh[keep].T.contiguous()


def randperm_would_fire():
    """Whether any reader-path randperm can fire: guard disabled?"""
    return {"model.training": bool(model.training),
            "initialized": int(model.initialized.item())}


def base():
    arm = CarrierResidualCompactor(
        num_layers=manager.num_layers, hidden=manager.hidden,
        block=manager.block, rank=int(cfg.carrier_rank),
        gamma_max=float(cfg.carrier_gamma_max)).to(
            device="cuda:0", dtype=torch.float32)
    arm.load_state_dict(state["state_dict"])
    for x in arm.parameters():
        x.requires_grad_(False)
    return arm


def encode(probes):
    encoded = {}
    for q in probes:
        stem = q.prompt if q.prompt else f"{q.question}\nAnswer:"
        tokens = manager.tokenizer(f"{stem} {q.answers[0]}", return_tensors="pt",
                                   add_special_tokens=False).input_ids
        stem_len = len(manager.tokenizer(stem, add_special_tokens=False).input_ids)
        encoded[q.question] = (tokens[:, :256].to("cuda:0"), stem_len)
    return encoded


@torch.no_grad()
def evaluate(encoded, probes, arm=None):
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
            correct[q.question] = bool(
                (span.argmax(dim=-1) == labels).all(dim=-1).item())
            del logits, span, labels
    del pool, mask_m
    torch.cuda.empty_cache()
    return correct


def exact_match(pred, answers):
    p = " ".join(str(pred).strip().lower().split())
    for a in answers:
        if p == " ".join(str(a).strip().lower().split()):
            return True
    return False


@torch.no_grad()
def evaluate_generation(probes, arm=None):
    was_training = model.training
    model.eval()
    correct, predictions = {}, {}
    try:
        pool, mask_m = (None, None) if arm is None else E.candidate_pool(manager, arm, "cuda:0")
        context = differentiable_memory(model, pool) if pool is not None else nullcontext()
        guard = active_mask_override(model, mask_m) if mask_m is not None else nullcontext()
        pad_id = (manager.tokenizer.pad_token_id
                  if manager.tokenizer.pad_token_id is not None
                  else manager.tokenizer.eos_token_id)
        with context, guard:
            for q in probes:
                stem = q.prompt if q.prompt else f"{q.question}\nAnswer:"
                ids = manager.tokenizer(stem, return_tensors="pt",
                                        add_special_tokens=False).input_ids.to("cuda:0")
                outp = model.generate(input_ids=ids, do_sample=False,
                                      max_new_tokens=args.max_new_tokens,
                                      pad_token_id=pad_id)
                pred = manager.tokenizer.decode(outp[0, ids.shape[1]:],
                                                skip_special_tokens=True)
                predictions[q.question] = pred
                correct[q.question] = exact_match(pred, q.answers)
                del ids, outp
        del pool, mask_m
    finally:
        if was_training:
            model.train()
    torch.cuda.empty_cache()
    return correct, predictions


report = {"window": [args.row, args.start], "steps": []}


def step(tag, **kv):
    entry = {"tag": tag}
    entry.update(kv)
    report["steps"].append(entry)
    print("[diag] %s %s" % (tag, json.dumps({k: v for k, v in kv.items()
                                             if k != "flips"})), flush=True)


def flips(a, b):
    return [q for q in a if a[q] != b[q]]


def mem_state():
    with torch.no_grad():
        return {"manager.used": int(manager.used),
                "memory_sha": hashlib_sample(model.memory),
                "training": bool(model.training)}


def hashlib_sample(t):
    """Cheap content fingerprint of a fp16 tensor (hash of first/last slices)."""
    with torch.no_grad():
        x = t.detach().to(torch.float32)
        h = hash((tuple(x[0, :512, :4].flatten().tolist()),
                  tuple(x[-1, -512:, :4].flatten().tolist()),
                  float(x.float().abs().sum().item())))
    return h


import hashlib  # noqa: E402


def fingerprint(t):
    with torch.no_grad():
        x = t.detach().to(torch.float16).cpu().numpy().tobytes()
    return hashlib.sha256(x).hexdigest()[:16]


out = open(args.out, "w") if False else None

print("[diag] building window (%d,%d) ..." % (args.row, args.start), flush=True)
line_cache = E.read_fact_lines(args.row)
built = build_window_and_probes(manager, {args.row: line_cache}, args.row,
                                args.start, probe_args)
assert built is not None
covered, indices, dropped, original = built
by_q = {q.question: q for q in original}
by_q.update({q.question: q for q in
             E.probes_from_lines(line_cache, indices, len(indices) + 1)})
probes = list(by_q.values())
mask = _compacted_mask(manager, manager.used - window_used + target_used, "cuda:0")
encoded = encode(probes)
print("[diag] probes=%d encoded" % len(probes), flush=True)
step("window_built", n_probes=len(probes), **mem_state())

# -- baseline arms --------------------------------------------------------
d0_arm = base()

select_arm = base()
with torch.no_grad():
    select_arm.up.weight.zero_()

wk = "%d,%d" % (args.row, args.start)
ss_rows = torch.stack([ss_cache["rows_by_window"][wk][q]
                       for q in sorted(ss_cache["rows_by_window"][wk],
                                       key=int)]).double()
U_ss = orth(ss_rows)[:, :13].contiguous()

# A. same d0 arm, twice back-to-back ---------------------------------------
s0 = mem_state()
rA1 = evaluate(encoded, probes, d0_arm)
s1_ = mem_state()
rA2 = evaluate(encoded, probes, d0_arm)
step("A_same_arm_twice", n_flip=len(flips(rA1, rA2)),
     flips=flips(rA1, rA2)[:5], state_before=s0, state_after=s1_)

# B. d0 vs parametrized arm (the exact failing comparison) ----------------
torch.manual_seed(0)
arm = base()
displacement = make_arm_displacement(arm.down.weight, U_ss, torch)
parametrize.register_parametrization(arm.down, "weight", displacement)
with torch.no_grad():
    w_equal = bool(torch.equal(arm.down.weight, d0_arm.down.weight))
rB_d0 = evaluate(encoded, probes, d0_arm)
rB_arm = evaluate(encoded, probes, arm)
step("B_d0_vs_arm", weight_bit_equal=w_equal, n_flip=len(flips(rB_d0, rB_arm)),
     flips=flips(rB_d0, rB_arm)[:5])
del arm, displacement
gc.collect(); torch.cuda.empty_cache()

# D. bitwise pool comparison ----------------------------------------------
d0_arm2 = base()
torch.manual_seed(0)
arm2 = base()
disp2 = make_arm_displacement(arm2.down.weight, U_ss, torch)
parametrize.register_parametrization(arm2.down, "weight", disp2)
with torch.no_grad():
    p_d0 = E.candidate_pool(manager, d0_arm2, "cuda:0")[0]
    p_arm = E.candidate_pool(manager, arm2, "cuda:0")[0]
    pool_equal = bool(torch.equal(p_d0, p_arm))
    d0_hand = None
    del p_d0, p_arm
step("D_pool_bit_equal", pool_bit_equal=pool_equal)
del arm2, disp2, d0_arm2
gc.collect(); torch.cuda.empty_cache()

# C. evaluate_generation sandwich -----------------------------------------
rC1 = evaluate(encoded, probes, d0_arm)
gen_d0, _gen_pred = evaluate_generation(probes, d0_arm)
rC2 = evaluate(encoded, probes, d0_arm)
step("C_generation_sandwich", n_flip_before=len(flips(rC1, rA1)),
     n_flip_after=len(flips(rC2, rC1)),
     flip_after=flips(rC2, rC1)[:5], gen_em=sum(gen_d0.values()),
     state_after=mem_state())

# E. once more with the arm, after generation happened ---------------------
torch.manual_seed(0)
arm3 = base()
disp3 = make_arm_displacement(arm3.down.weight, U_ss, torch)
parametrize.register_parametrization(arm3.down, "weight", disp3)
rE_d0 = evaluate(encoded, probes, d0_arm)
rE_arm = evaluate(encoded, probes, arm3)
step("E_after_generation_d0_vs_arm",
     n_flip=len(flips(rE_d0, rE_arm)), flips=flips(rE_d0, rE_arm)[:5])
del arm3, disp3
gc.collect(); torch.cuda.empty_cache()

# F. determinism of model.memory across evaluates -------------------------
step("F_final_state", **mem_state())

with open(args.out, "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("[diag] complete -> %s" % args.out, flush=True)
