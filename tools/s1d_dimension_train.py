#!/usr/bin/env python
"""C13/C20/C24/C28/C32: isolated dimension-only derivative of the C driver.

Only write-time facts define the bases. Original C reader/injection/training
functions are preserved and checked by CPU regression. No batch auto-launch.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import string
import re
import sys
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch
from s1d_dimension_basis import (VARIANTS, projection_protocol, select_protection,
                            build_basis, tensor_sha256, validate_protection_cache, save_new_cache)

DEV_WINDOWS = [(3, 16370), (3, 17510)]
ALL_TRAIN_WINDOWS = [(3, 17890), (3, 17130), (3, 16750), (3, 14470),
                     (3, 17510), (3, 14850), (3, 16370), (3, 15230)]
ARMS = ("selfsup_13d", "random_13d", "unconstrained")
ORACLE_ARM = "oracle_13d"
PROTECTED_ORACLE_ARM = "oracle_13d_protected"
ALL_ARMS = ARMS + (ORACLE_ARM, PROTECTED_ORACLE_ARM)
QA_ORACLE_ARMS = (ORACLE_ARM, PROTECTED_ORACLE_ARM)
MAX_SELFSUP_LINES = 40
MASK_SPAN = 4
EXP1_ARMS = ("A", "B", "C", "D")

PROTOCOL = {
    "experiment": "S1d-B stage 2 dev: answer-free 13D residual training",
    "arms": list(ARMS),
    "supervision": ("tail-4-token autoregressive CE of the window's own "
                    "dropped-block fact lines under the candidate pool; "
                    "no QA answer ever enters a gradient step"),
    "basis": "frozen per-window selfsup SVD-13 basis (runs/s1d_selfsup_rows_v2.pt)",
    "budget": "displacement norm clamp 1.0 every step, identical lr/steps/optimizer",
    "dev_windows": [list(w) for w in DEV_WINDOWS],
    "selection_rule": "windows picked on stage-1 coverage (best/worst gRho), not QA",
    "metrics": ("new_recovery = full∧¬d0∧trained; d0_harmed = d0∧full harmed; "
                "net = new_recovery - d0_harmed; both teacher-forced exact "
                "and greedy-generation EM recorded"),
    "seed": 0,
}


def select_fact_cohorts(dropped_records, kept_records, encoded, full_ce, d0_ce,
                        fact_max=40, full_ce_max=2.0, delta_threshold=0.10):
    damaged = []
    for record in dropped_records:
        key = record["key"]
        if key not in encoded:
            continue
        delta = d0_ce[key] - full_ce[key]
        if (math.isfinite(delta) and full_ce[key] <= full_ce_max
                and delta >= delta_threshold):
            damaged.append((delta, key))
    damaged.sort(key=lambda item: (-item[0], item[1]))
    thresholded = [key for _delta, key in damaged]
    basis_keys = thresholded[:fact_max]
    rescue_keys = basis_keys[:fact_max // 2]
    protect = [(d0_ce[r["key"]], r["key"]) for r in kept_records
               if r["key"] in encoded and r["key"] not in rescue_keys
               and d0_ce[r["key"]] <= full_ce_max]
    protect.sort(key=lambda item: (item[0], item[1]))
    protect_keys = [key for _ce, key in protect[:fact_max - len(rescue_keys)]]
    return thresholded, basis_keys, rescue_keys, protect_keys


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def orth(rows: torch.Tensor) -> torch.Tensor:
    """SVD-ordered orthonormal basis (s1_local_protected.orth semantics)."""
    _, s, vh = torch.linalg.svd(rows.double(), full_matrices=False)
    keep = s > max(1e-10, float(s.max()) * 1e-7)
    return vh[keep].T.contiguous()


def normalize(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def exact_match(prediction: str, aliases) -> bool:
    return any(normalize(prediction) == normalize(a) for a in aliases)


def memory_digest(memory):
    """Hash every memory byte for evaluation-integrity checks."""
    h = hashlib.sha256()
    for layer in memory.detach():
        h.update(layer.contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


@contextmanager
def reader_evaluation(model, verify_memory=False):
    """Run reader evaluation and restore every mutable runtime state."""
    modes = [(module, module.training) for module in model.modules()]
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    memory = getattr(model, "memory", None)
    memory_snapshot = (memory.detach().cpu().clone()
                       if verify_memory and memory is not None else None)
    mask = model._buffers.get("memory_active_mask")
    had_mask = "memory_active_mask" in model._buffers
    try:
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        if getattr(model, "memory", None) is not memory:
            raise RuntimeError("reader evaluation changed memory object")
        if (("memory_active_mask" in model._buffers) != had_mask or
                model._buffers.get("memory_active_mask") is not mask):
            raise RuntimeError("reader evaluation did not restore active mask")
        if memory_snapshot is not None:
            changed = not torch.equal(memory.detach().cpu(), memory_snapshot)
            if changed:
                with torch.no_grad():
                    memory.copy_(memory_snapshot.to(memory.device))
                print("[s1d2] generation memory mutation restored", flush=True)
            del memory_snapshot


def window_sets(full, select_correct, d0):
    """Cohort derivation, s1c_global_oracle.window_sets verbatim."""
    questions = list(full)
    if set(questions) != set(select_correct) or set(questions) != set(d0):
        raise ValueError("baseline question universes differ")
    retained = [q for q in questions if full[q] and select_correct[q]]
    rescue = [q for q in questions if full[q] and not select_correct[q]]
    return {"original_lost": [], "retained": retained, "rescue": rescue,
            "protect": [q for q in retained if d0[q]],
            "repair": [q for q in retained if not d0[q]],
            "rescue_d0_correct": [q for q in rescue if d0[q]],
            "rescue_d0_wrong": [q for q in rescue if not d0[q]]}


def stage2_metric(correct, sets, d0_correct):
    """S1c counting flaw fixed: recovery counted only where D0 was WRONG.

    S1c's rescue_recovered counted every full∧select-wrong question that the
    arm got right, including ones D0 already answered correctly (no new
    recovery).  Here:
      new_recovery  = rescue_d0_wrong questions the arm now gets right
      d0_harmed     = questions D0 got right (any cohort) the arm gets wrong
      net           = new_recovery - d0_harmed
    """
    entry = {"correct": dict(correct)}
    entry["new_recovery"] = sum(1 for q in sets["rescue_d0_wrong"]
                                if correct[q])
    entry["new_recovery_questions"] = [q for q in sets["rescue_d0_wrong"]
                                       if correct[q]]
    entry["rescue_recovered_s1c_style"] = sum(1 for q in sets["rescue"]
                                               if correct[q])
    d0_right = [q for q, v in d0_correct.items() if v]
    entry["d0_harmed"] = sum(1 for q in d0_right if not correct[q])
    entry["d0_harmed_questions"] = [q for q in d0_right if not correct[q]]
    entry["d0_new_recovery"] = entry["new_recovery"]
    entry["net_vs_d0"] = entry["new_recovery"] - entry["d0_harmed"]
    entry["accuracy"] = sum(correct.values()) / len(correct)
    return entry


def select_lines_v2(lines, line_to_block, dropped, e6a):
    """Stage-1 v2 pool rule verbatim (primary dropped-block latest-owner)."""
    latest = {}
    for index, text in enumerate(lines):
        found = e6a.first_match(text)
        if found is None:
            latest[("raw", index)] = (index, None)
            continue
        slot, subject, answer, _question = found
        latest[(slot, subject)] = (index, answer)

    def pick(pool_blocks):
        out = []
        for index, _owner in sorted(latest.values()):
            idx = index
            if idx not in line_to_block or line_to_block[idx] not in pool_blocks:
                continue
            if idx not in out:
                out.append(idx)
        return out

    primary = pick(set(dropped))
    aux = pick(set(range(19)) - set(dropped))
    return primary, aux


class QAParallel:
    """Replicated frozen readers with sharded questions and summed alpha gradients."""

    def __init__(self, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size

    def shard(self, values):
        return values[self.rank::self.world_size]

    def gather(self, value):
        if self.world_size == 1:
            return [value]
        gathered = [None] * self.world_size
        torch.distributed.all_gather_object(gathered, value)
        return gathered

    def merge(self, local, keys):
        merged = {}
        for part in self.gather(local):
            if set(merged) & set(part):
                raise RuntimeError("duplicate QA shard keys")
            merged.update(part)
        if set(merged) != set(keys):
            raise RuntimeError("incomplete QA shard coverage")
        return {key: merged[key] for key in keys}

    def assert_same(self, value, label):
        if any(other != value for other in self.gather(value)):
            raise RuntimeError("distributed mismatch: " + label)

    def reduce_step(self, parameters, totals):
        if self.world_size == 1:
            return totals
        for parameter in parameters:
            gradient = (parameter.grad.detach().cpu().clone() if parameter.grad is not None
                        else torch.zeros_like(parameter, device="cpu"))
            torch.distributed.all_reduce(gradient, op=torch.distributed.ReduceOp.SUM)
            parameter.grad = gradient.to(parameter.device)
        sums = torch.tensor(totals, dtype=torch.float64)
        torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
        return sums.tolist()



def validate_projection_arguments(args):
    if (args.exp1_arm != "C" or args.risk_protection or not args.fact_cache
            or not args.c_reference or not args.skip_generation
            or args.lr != 0.003 or args.budget != 1.0 or args.qa_max_tokens != 256
            or args.fact_delta_threshold != 0.10 or args.fact_full_ce_max != 2.0
            or args.fact_max != 40 or args.steps not in (1, 60)
            or args.measure_every != (1 if args.steps == 1 else 10)
            or args.window not in ("3,16370", "3,17510")):
        raise ValueError("fixed C projection protocol violated")
    if (args.prepare_protection_cache or VARIANTS[args.basis_variant][1]) and not args.protection_rows:
        raise ValueError("protection row cache path required")
    if Path(args.out).suffix != ".json":
        raise ValueError("output report must have .json suffix")
    if not args.prepare_protection_cache and Path(args.out).with_suffix(".alpha.pt").exists():
        raise ValueError("alpha checkpoint exists; refusing overwrite")
    if args.protection_rows and Path(args.protection_rows).suffix != ".pt":
        raise ValueError("protection cache must have .pt suffix")
    if args.prepare_protection_cache and Path(args.protection_rows).exists():
        raise ValueError("protection cache exists; refusing overwrite")


def validate_c_reference(reference, protocol, window, sets, maps, n_supervision):
    if (reference.get("status") != "complete" or len(reference.get("windows", [])) != 1
            or reference["protocol"].get("exp1_arm") != "C"
            or reference["schema_version"] != "s1d_experiment1/1"):
        raise ValueError("reference must be published original C")
    w = reference["windows"][0]
    if w["window"] != window or w["sets"] != sets or w["n_supervision_lines"] != n_supervision:
        raise ValueError("original C window/cohort/supervision mismatch")
    for k, v in maps.items():
        if w["baselines"][k]["correct"] != v:
            raise ValueError("original C TF baseline mismatch: " + k)
    for k in ("checkpoint_sha256", "selfsup_rows_sha256", "lr", "budget", "qa_max_tokens",
              "skip_generation", "seed", "optimizer", "weight_decay", "fact_delta_threshold",
              "fact_full_ce_max", "fact_max"):
        if protocol[k] != reference["protocol"][k]:
            raise ValueError("original C protocol mismatch: " + k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="runs/s1c_global_rows.pt")
    ap.add_argument("--selfsup-rows", default="runs/s1d_selfsup_rows_v2.pt")
    ap.add_argument("--carrier-ckpt", default="runs/e6d_interleave_ckpt.pt")
    ap.add_argument("--out", default="runs/s1d_stage2_dev.json")
    ap.add_argument("--cuda-device", type=int, default=9)
    ap.add_argument("--cuda-devices", default=None,
                    help="physical GPU indices in rank order; launch with torchrun")
    ap.add_argument("--model-path", default=None,
                    help="equivalent local reader checkpoint cache; default original MODEL")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--measure-every", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--qa-max-tokens", type=int, default=256)
    ap.add_argument("--skip-generation", action="store_true",
                    help="skip expensive greedy generation baselines and final GEN scoring")
    ap.add_argument("--windows", default="dev",
                    choices=["dev"], help="formal run is a separate frozen driver")
    ap.add_argument("--window", default=None,
                    help="comma pair row,start -- run a single window (sharding); "
                         "output must be per-window when used")
    ap.add_argument("--arms", nargs="+", choices=ALL_ARMS,
                    default=list(ARMS))
    ap.add_argument("--exp1-arm", choices=EXP1_ARMS, default="C",
                    help="experiment 1 arm: A=old basis/old tail loss, "
                         "B=old basis/new fact loss, C=new basis/old loss, "
                         "D=new basis/new fact loss")
    ap.add_argument("--fact-delta-threshold", type=float, default=0.10,
                    help="minimum D0 minus Full fact answer CE for new-basis facts")
    ap.add_argument("--fact-full-ce-max", type=float, default=2.0)
    ap.add_argument("--fact-cache", default=None,
                    help="shared fact gradient/selection cache for B/C/D")
    ap.add_argument("--fact-max", type=int, default=40,
                    help="maximum fact targets and retained fact protection targets")
    ap.add_argument("--risk-protection", action="store_true",
                    help="fixed C-risk-v1: top20 kept D0 CE<=2, descending CE; lambda=1")
    ap.add_argument("--basis-variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--protection-rows", help="shared D0 kept-fact gradient cache")
    ap.add_argument("--prepare-protection-cache", action="store_true",
                    help="write gradient cache and exit before optimizer construction")
    ap.add_argument("--c-reference", required=True, help="published original C report for this window")
    args = ap.parse_args()
    validate_projection_arguments(args)

    exp1_mode = args.exp1_arm
    if args.risk_protection:
        from s1d_risk_protection import select_risk_facts, risk_protocol
        if (exp1_mode != "C" or not args.fact_cache or not args.skip_generation
                or args.lr != 0.003 or args.budget != 1.0 or args.qa_max_tokens != 256
                or args.fact_delta_threshold != 0.10 or args.fact_full_ce_max != 2.0
                or args.steps not in (1, 60) or args.measure_every != (1 if args.steps == 1 else 10)):
            raise SystemExit("C-risk-v1 requires frozen C settings, cached basis, TF-only, 1-step smoke or 60-step formal")
    if exp1_mode is None:
        raise SystemExit("--exp1-arm A/B/C/D required; use original driver for other experiments")
    if args.window not in ("3,16370", "3,17510"):
        raise SystemExit("experiment 1 is restricted to the two development windows")
    if args.fact_max != 40 or args.budget <= 0 or args.steps <= 0 or args.measure_every <= 0:
        raise SystemExit("invalid protocol: fact cap must be 40; budget/steps/cadence positive")
    if exp1_mode is not None:
        # The experiment-1 launcher runs one arm per process. Keep the legacy
        # arm list out of the report and avoid QA-oracle code paths.
        args.arms = ["selfsup_13d"]

    SCRIPT_SHA = sha256_file(os.path.abspath(__file__))
    out = Path(args.out)
    if out.exists():
        raise SystemExit("output exists: %s" % out)
    run_windows = DEV_WINDOWS
    if args.window is not None:
        row_s, start_s = args.window.split(",")
        run_windows = [(int(row_s), int(start_s))]

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    devices = ([int(x) for x in args.cuda_devices.split(",")]
               if args.cuda_devices else [args.cuda_device])
    if len(set(devices)) != len(devices) or len(devices) != world_size:
        raise SystemExit("GPU list must contain one unique device per torchrun rank")
    if world_size > 1 and exp1_mode is not None:
        raise SystemExit("experiment 1 uses one process per physical GPU")
    if world_size > 1 and (args.arms != [PROTECTED_ORACLE_ARM] or not args.skip_generation):
        raise SystemExit("distributed execution requires protected oracle and --skip-generation")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank])
    os.environ["S0B_GPU_SET"] = "1"
    if world_size > 1:
        from datetime import timedelta
        torch.distributed.init_process_group("gloo", timeout=timedelta(minutes=30))
    parallel = QAParallel(rank, world_size)
    print("[s1d2] rank=%d/%d physical_gpu=%d" %
          (rank, world_size, devices[rank]), flush=True)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch.nn.functional as F
    import e6a_overfit as E
    from s0_functional_sensitivity import build_window_and_probes, margin_scalar, CONFIG, MODEL
    from s1_local_protected import disable_reader_dropout, digest
    from s1c_wash import make_arm_displacement, random_basis
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory
    from methods.prlmc.probes import (_pool_in_place, _compacted_mask,
                                      differentiable_memory, active_mask_override)
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from transformers import AutoTokenizer
    from torch.nn.utils import parametrize

    torch.manual_seed(0)
    torch.set_num_threads(4)
    run_started = time.monotonic()
    with open(CONFIG) as f:
        import yaml
        config = dict(yaml.safe_load(f)["prlmc"])
    config.update(device="cuda:0", dtype="float16",
                  stats_dir="./outputs/prlmc-s1d-stage2-dev")
    cfg = PRLMCConfig.from_dict(config)
    reader_path = args.model_path or MODEL
    model = MemoryLLM.from_pretrained(reader_path, torch_dtype=torch.float16,
                                      attn_implementation="sdpa").to("cuda:0")
    model.initialized.zero_()
    for x in model.parameters():
        x.requires_grad_(False)
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    manager = ProtectedResidualMemory(model, AutoTokenizer.from_pretrained(reader_path), cfg)
    disable_reader_dropout(model)
    probe_args = argparse.Namespace(probes=60, max_probe_tokens=256)
    from methods.prlmc.consolidate import CarrierResidualCompactor
    state = torch.load(args.carrier_ckpt, map_location="cuda:0", weights_only=True)

    ss_cache = torch.load(args.selfsup_rows, map_location="cpu", weights_only=False)
    if ss_cache["checkpoint_sha256"] != digest(args.carrier_ckpt):
        raise SystemExit("selfsup rows checkpoint mismatch")

    window_used = E.WINDOW_BLOCKS * manager.block
    target_used = E.TARGET_BLOCKS * manager.block

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

    def candidate(arm):
        source = manager._memory[:, manager.total - manager.used:manager.total, :].to("cuda:0").clone()
        weights = manager._window_importance(manager.total - manager.used, window_used)
        return arm(source, weights, target=target_used, block=manager.block,
                   goal_direction=manager.goal_direction, select_weights=manager.select_weights)

    line_cache = {}

    def fresh_window(row, start):
        line_cache.setdefault(row, E.read_fact_lines(row))
        lines = line_cache[row]
        built = build_window_and_probes(manager, line_cache, row, start, probe_args)
        if built is None:
            raise RuntimeError("window build failed (%d,%d)" % (row, start))
        covered, indices, dropped, original = built
        return covered, indices, dropped, original

    def encode(probes):
        encoded = {}
        for q in probes:
            stem = q.prompt if q.prompt else f"{q.question}\nAnswer:"
            tokens = manager.tokenizer(f"{stem} {q.answers[0]}", return_tensors="pt",
                                       add_special_tokens=False).input_ids
            stem_len = len(manager.tokenizer(stem, add_special_tokens=False).input_ids)
            width = min(args.qa_max_tokens, tokens.shape[1])
            if stem_len <= 0 or stem_len >= tokens.shape[1]:
                raise RuntimeError("missing answer span: " + q.question)
            start = max(0, tokens.shape[1] - width)
            stem_len -= start
            if stem_len <= 0 or stem_len >= width:
                raise RuntimeError("QA token budget cuts answer span: " + q.question)
            encoded[q.question] = (tokens[:, start:start + width].to("cuda:0"), stem_len)
        return encoded

    @torch.no_grad()
    def evaluate(encoded, probes, arm=None):
        correct = {}
        pool, mask_m = (None, None) if arm is None else E.candidate_pool(manager, arm, "cuda:0")
        context = differentiable_memory(model, pool) if pool is not None else nullcontext()
        guard = active_mask_override(model, mask_m) if mask_m is not None else nullcontext()
        with reader_evaluation(model), context, guard:
            for q in parallel.shard(probes):
                input_ids, stem_len = encoded[q.question]
                logits = model(input_ids=input_ids, use_cache=False, return_dict=True).logits
                span = logits[:, stem_len - 1:-1, :]
                labels = input_ids[:, stem_len:]
                correct[q.question] = bool((span.argmax(dim=-1) == labels).all(dim=-1).item())
                del logits, span, labels
        del pool, mask_m
        torch.cuda.empty_cache()
        return parallel.merge(correct, [q.question for q in probes])

    @torch.no_grad()
    def evaluate_generation(probes, arm=None):
        """Real greedy generation EM (native_qa_compare semantics).

        The model lives in train() + gradient checkpointing for the training
        loop; in that state HF forces use_cache=False and every decode step
        recomputes the full (pool+prompt) forward (~60 s/probe, hours per
        evaluation -- the failure of the first dev launch). Switching to
        eval() for the duration of the generation restores the KV cache
        (~30x faster). Memory state is untouched; switched back after.
        """
        correct, predictions = {}, {}
        with reader_evaluation(model, verify_memory=True):
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
                    outp = model.generate(input_ids=ids,
                                          attention_mask=torch.ones_like(ids),
                                          do_sample=False,
                                          max_new_tokens=args.max_new_tokens,
                                          pad_token_id=pad_id)
                    pred = manager.tokenizer.decode(outp[0, ids.shape[1]:],
                                                    skip_special_tokens=True)
                    predictions[q.question] = pred
                    correct[q.question] = exact_match(pred, q.answers)
                    del ids, outp
            del pool, mask_m
        torch.cuda.empty_cache()
        return correct, predictions

    @torch.no_grad()
    def answer_ce(encoded, questions, arm=None):
        """Teacher-forced answer-token CE used by the protected oracle."""
        losses = {}
        pool, mask_m = (None, None) if arm is None else E.candidate_pool(manager, arm, "cuda:0")
        context = differentiable_memory(model, pool) if pool is not None else nullcontext()
        guard = active_mask_override(model, mask_m) if mask_m is not None else nullcontext()
        with reader_evaluation(model), context, guard:
            for question in parallel.shard(questions):
                input_ids, stem_len = encoded[question]
                labels = input_ids.clone()
                labels[:, :stem_len] = -100
                logits = model(input_ids=input_ids, use_cache=False, return_dict=True).logits
                losses[question] = float(F.cross_entropy(
                    logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                    labels[:, 1:].reshape(-1), ignore_index=-100))
                del logits, labels
        del pool, mask_m
        torch.cuda.empty_cache()
        return parallel.merge(losses, questions)

    def supervision_lines(row, start, line_to_block, dropped):
        """Stage-1 v2 supervision pool: up to 40 valid dropped-block lines."""
        lines = line_cache[row]
        primary, aux = select_lines_v2(lines, line_to_block, dropped, E)
        chosen, skipped = [], 0
        for line_idx in primary + aux:
            if len(chosen) >= MAX_SELFSUP_LINES:
                break
            text = lines[line_idx].rstrip(".")
            tokens = manager.tokenizer(text, return_tensors="pt",
                                       add_special_tokens=False).input_ids
            if tokens.shape[1] < 2 * MASK_SPAN + 2:
                skipped += 1
                continue
            chosen.append((line_idx, tokens))
        return chosen, skipped

    def fact_records(row, indices, line_to_block, dropped_blocks=None):
        """Build answer-masked targets from latest raw fact statements.

        These targets are derived only from the write-time fact lines.  They do
        not call the benchmark QA renderer, so evaluation questions cannot leak
        into either the new basis or the new training loss.
        """
        lines = line_cache[row]
        latest = {}
        for line_idx, text in enumerate(lines):
            found = E.first_match(text)
            if found is not None:
                slot, subject, answer, question = found
                latest[(slot, subject)] = (line_idx, answer, question)
        records = []
        for line_idx in sorted(indices):
            block = line_to_block.get(line_idx)
            if dropped_blocks is not None and block not in dropped_blocks:
                continue
            found = E.first_match(lines[line_idx])
            if found is None:
                continue
            slot, subject, answer, question = found
            owner = latest.get((slot, subject))
            if owner is None or owner[0] != line_idx:
                continue
            raw = lines[line_idx].rstrip(".").strip()
            answer = answer.strip()
            if not answer or not raw.endswith(answer):
                continue
            prefix = raw[:-len(answer)].rstrip()
            key = "fact:%d:%d:%s" % (line_idx, slot, "|".join(subject))
            records.append({"key": key, "line_idx": line_idx,
                            "block": block, "prefix": prefix,
                            "answer": answer, "question_format": question})
        return records

    def encode_fact_records(records):
        encoded = {}
        for record in records:
            prefix, answer = record["prefix"], record["answer"]
            text = "%s %s" % (prefix, answer)
            result = manager.tokenizer(text, return_tensors="pt", add_special_tokens=False,
                                       return_offsets_mapping=True)
            tokens = result.input_ids
            offsets = result.offset_mapping[0].tolist()
            answer_char = len(prefix) + 1
            stem_len = next((i for i, (_lo, hi) in enumerate(offsets)
                             if hi > answer_char), len(offsets))
            if any(lo < len(prefix) for lo, hi in offsets[stem_len:] if hi > answer_char):
                raise RuntimeError("fact answer token crosses prefix boundary")
            width = min(args.qa_max_tokens, tokens.shape[1])
            offset = max(0, tokens.shape[1] - width)
            stem_len -= offset
            if stem_len <= 0 or stem_len >= width:
                raise RuntimeError("fact token budget removes answer or prefix: " + record["key"])
            encoded[record["key"]] = (tokens[:, offset:offset + width].to("cuda:0"), stem_len)
        return encoded

    def fact_basis(arm, encoded, keys, mask):
        """SVD basis of D0 answer-span gradients for damaged facts."""
        if not keys:
            raise RuntimeError("no damaged fact targets for new basis")
        arm.down.weight.requires_grad_(True)
        rows = []
        try:
            for key in keys:
                input_ids, stem_len = encoded[key]
                local_cand = candidate(arm)
                with torch.autograd.graph.save_on_cpu(pin_memory=True), _pool_in_place(manager, "cuda:0", local_cand,
                                    window_used, target_used) as injection:
                    pool = injection.reinject()
                    with differentiable_memory(model, pool), active_mask_override(model, mask):
                        logits = model(input_ids=input_ids, use_cache=False,
                                       return_dict=True).logits
                        loss = F.cross_entropy(
                            logits[:, stem_len - 1:-1, :].float().reshape(-1, logits.shape[-1]),
                            input_ids[:, stem_len:].reshape(-1))
                        grad = torch.autograd.grad(loss, arm.down.weight,
                                                   retain_graph=False)[0]
                    rows.append(grad.detach().float().reshape(-1).cpu())
                del local_cand, pool, logits, loss, grad
                torch.cuda.empty_cache()
        finally:
            arm.down.weight.requires_grad_(False)
        fact_rows = torch.stack(rows).double()
        basis = orth(fact_rows)
        provenance = "fact_answer_gradients"
        if basis.shape[1] < 13:
            raise RuntimeError("fact basis rank %d < 13" % basis.shape[1])
        return basis[:, :13].contiguous(), fact_rows, provenance

    def fact_gradient_rows(arm, encoded, keys, mask):
        """D0 answer-span gradients for retained write-time facts, no QA inputs."""
        if not keys:
            raise RuntimeError("no damaged fact targets for new basis")
        arm.down.weight.requires_grad_(True)
        rows = []
        try:
            for key in keys:
                input_ids, stem_len = encoded[key]
                local_cand = candidate(arm)
                with torch.autograd.graph.save_on_cpu(pin_memory=True), _pool_in_place(manager, "cuda:0", local_cand,
                                    window_used, target_used) as injection:
                    pool = injection.reinject()
                    with differentiable_memory(model, pool), active_mask_override(model, mask):
                        logits = model(input_ids=input_ids, use_cache=False,
                                       return_dict=True).logits
                        loss = F.cross_entropy(
                            logits[:, stem_len - 1:-1, :].float().reshape(-1, logits.shape[-1]),
                            input_ids[:, stem_len:].reshape(-1))
                        grad = torch.autograd.grad(loss, arm.down.weight,
                                                   retain_graph=False)[0]
                    rows.append(grad.detach().float().reshape(-1).cpu())
                del local_cand, pool, logits, loss, grad
                torch.cuda.empty_cache()
        finally:
            arm.down.weight.requires_grad_(False)
        fact_rows = torch.stack(rows).double()
        return fact_rows

    def train_answer_arm(arm, displacement, optimizer, encoded, questions, mask,
                         protect_questions=None, protect_baseline=None,
                         protect_weight=0.0, mean_grad=False):
        """QA step with fixed-weight protection and bounded microbatches."""
        optimizer.zero_grad(set_to_none=True)
        protect_questions = protect_questions or []
        protect_baseline = protect_baseline or {}
        rescue_total = 0.0
        protect_total = 0.0
        microbatch = 1
        pad_id = manager.tokenizer.eos_token_id

        def run_batch(pool, batch_questions, baselines=None):
            ids, labels = [], []
            for question in batch_questions:
                input_ids, stem_len = encoded[question]
                label = input_ids.clone()
                label[:, :stem_len] = -100
                ids.append(input_ids.squeeze(0))
                labels.append(label.squeeze(0))
            input_ids = torch.nn.utils.rnn.pad_sequence(
                ids, batch_first=True, padding_value=pad_id).to("cuda:0")
            label_ids = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=-100).to("cuda:0")
            logits = model(input_ids=input_ids, use_cache=False,
                           return_dict=True).logits.float()
            token_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                label_ids[:, 1:].reshape(-1), reduction="none").reshape(label_ids.shape[0], -1)
            valid = label_ids[:, 1:] != -100
            per_question = (token_loss * valid).sum(1) / valid.sum(1).clamp_min(1)
            if baselines is None:
                batch_loss = per_question.sum() / max(1, len(questions))
            else:
                ref = torch.tensor(baselines, device=per_question.device,
                                   dtype=per_question.dtype)
                batch_loss = torch.relu(per_question - ref).sum() / max(1, len(protect_questions))
            return per_question, batch_loss

        def run_one(batch_questions, baselines=None):
            # Rebuild the alpha-dependent graph per batch so backward releases
            # the large reader graph immediately.
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                local_cand = candidate(arm)
                with _pool_in_place(manager, "cuda:0", local_cand, window_used,
                                    target_used) as injection:
                    local_pool = injection.reinject()
                    # Checkpoint recomputation must read the same candidate pool.
                    with differentiable_memory(model, local_pool), active_mask_override(model, mask):
                        per_question, batch_loss = run_batch(
                            local_pool, batch_questions, baselines)
                        scaled_loss = (protect_weight * batch_loss
                                       if baselines is not None else batch_loss)
                        scaled_loss.backward()
                    values = per_question.detach()
                    loss_value = batch_loss.detach()
            del local_pool, local_cand
            return values, loss_value

        local_questions = parallel.shard(questions)
        local_protect = parallel.shard(protect_questions)
        for lo in range(0, len(local_questions), microbatch):
            batch = local_questions[lo:lo + microbatch]
            per_question, loss = run_one(batch)
            rescue_total += float(per_question.sum())
            del per_question, loss
            if lo == 0 or (lo + 1) % 10 == 0 or lo + 1 == len(local_questions):
                print("[s1d2] rank=%d rescue backward %d/%d peak_allocated=%.2f GiB" %
                      (parallel.rank, lo + 1, len(local_questions), torch.cuda.max_memory_allocated() / 2**30), flush=True)
        for lo in range(0, len(local_protect), microbatch):
            batch = local_protect[lo:lo + microbatch]
            refs = [protect_baseline[q] for q in batch]
            per_question, penalty = run_one(batch, refs)
            protect_total += float(torch.relu(
                per_question - torch.tensor(refs, device=per_question.device,
                                            dtype=per_question.dtype)).sum())
            del per_question, penalty
            if lo == 0 or (lo + 1) % 10 == 0 or lo + 1 == len(local_protect):
                print("[s1d2] rank=%d protect backward %d/%d peak_allocated=%.2f GiB" %
                      (parallel.rank, lo + 1, len(local_protect), torch.cuda.max_memory_allocated() / 2**30), flush=True)
        rescue_total, protect_total = parallel.reduce_step(
            [displacement.value], [rescue_total, protect_total])
        optimizer.step()
        displacement.clamp(args.budget)
        torch.cuda.empty_cache()
        if mean_grad:
            print("[s1d2] microbatch objective complete", flush=True)
        return {
            "loss": rescue_total / max(1, len(questions)) +
                    protect_weight * protect_total / max(1, len(protect_questions)),
            "rescue_loss": rescue_total / max(1, len(questions)),
            "protect_penalty": protect_total / max(1, len(protect_questions)),
        }

    def backward_fact_protection(arm, encoded_facts, keys, refs, mask):
        """Accumulate mean hinge into the existing alpha gradient before step."""
        total = 0.0
        for key in keys:
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                cand = candidate(arm)
                with _pool_in_place(manager, "cuda:0", cand, window_used,
                                    target_used) as injection:
                    pool = injection.reinject()
                    ids, boundary = encoded_facts[key]
                    ids = ids.to("cuda:0")
                    labels = ids.clone()
                    labels[:, :boundary] = -100
                    with differentiable_memory(model, pool), active_mask_override(model, mask):
                        logits = model(input_ids=ids, use_cache=False, return_dict=True).logits
                        ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                             labels[:, 1:].reshape(-1))
                        penalty = torch.relu(ce - refs[key])
                        (penalty / len(keys)).backward()
                        total += float(penalty.detach())
                del cand, pool, logits, ce, penalty, labels, ids
        return total / max(1, len(keys))

    def train_arm(arm, displacement, optimizer, sup, mask,
                  protection=None):
        """Old summed tail CE, with candidate binding alive during recomputation."""
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for line_idx, tokens in sup:
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                cand = candidate(arm)
                with _pool_in_place(manager, "cuda:0", cand, window_used,
                                    target_used) as injection:
                    pool = injection.reinject()
                    input_ids = tokens.to("cuda:0")
                    labels = input_ids.clone()
                    labels[:, :-MASK_SPAN] = -100
                    with differentiable_memory(model, pool), active_mask_override(model, mask):
                        logits = model(input_ids=input_ids, use_cache=False,
                                       return_dict=True).logits
                        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                               labels[:, 1:].reshape(-1))
                        loss.backward()
                        total += float(loss.detach())
                del cand, pool, loss, logits
        protect_penalty = 0.0
        if protection is not None:
            protect_penalty = backward_fact_protection(arm, *protection, mask)
        if not torch.isfinite(displacement.value.grad).all():
            raise RuntimeError("nonfinite tail gradient")
        optimizer.step()
        displacement.clamp(args.budget)
        torch.cuda.empty_cache()
        if protection is not None:
            return {"loss": total + protect_penalty,
                    "tail_loss_mean": total / max(1, len(sup)),
                    "tail_loss_sum": total, "protect_penalty": protect_penalty}
        return total / max(1, len(sup))

    if any(name in args.arms for name in QA_ORACLE_ARMS) and (
            args.window != "3,17510" or len(args.arms) != 1):
        raise SystemExit("QA oracle arms require an isolated run on window 3,17510")

    report = {
        "schema_version": ("s1d_experiment1/1" if exp1_mode is not None
                            else "s1d_stage2_dev/1"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "driver_sha256": SCRIPT_SHA,
        "protocol": {**PROTOCOL, "steps": args.steps,
                     "reader_evaluation": "eval; exact mixed-mode and RNG restoration; full-memory integrity check",
                     "selected_arms": [exp1_mode] if exp1_mode else args.arms,
                     "measure_every": args.measure_every,
                     "lr": args.lr, "budget": args.budget,
                     "max_new_tokens": args.max_new_tokens,
                     "qa_max_tokens": args.qa_max_tokens,
                     "skip_generation": args.skip_generation,
                     "checkpoint_sha256": digest(args.carrier_ckpt),
                     "selfsup_rows_sha256": sha256_file(args.selfsup_rows),
                     "evaluation_qa_used_for_training": False},
        "diagnostic_only": False if exp1_mode is not None else any(
            name in args.arms for name in QA_ORACLE_ARMS),
        "excluded_from_method_results": any(name in args.arms for name in QA_ORACLE_ARMS),
        "oracle_diagnostic": ({
            "purpose": "finite-step reachability diagnostic, not a mathematical upper bound",
            "supervision": "real QA answer-token CE on rescue_d0_wrong only",
            "basis": "unchanged selfsup SVD-13 basis; no QA-based basis rebuild",
            "evaluation": "same complete QA universe and D0-relative metrics",
        } if any(name in args.arms for name in QA_ORACLE_ARMS) else None),
        "windows": [],
    }

    if exp1_mode is not None:
        report["protocol"].update(
            experiment="PRLM-C experiment 1: fact target/basis cross-check",
            arms=[exp1_mode],
            exp1_arm=exp1_mode,
            arm_definition={
                "A": "old selfsup 13D basis + old tail-4-token CE",
                "B": "old selfsup 13D basis + new fact answer-span CE",
                "C": "new damaged-fact gradient 13D basis + old tail-4-token CE",
                "D": "new damaged-fact gradient 13D basis + new fact answer-span CE",
            }[exp1_mode],
            supervision_leakage="raw write-time fact statements only; benchmark QA is evaluation-only",
            protection="retained latest facts, fixed mean ReLU(current CE - D0 CE), weight 1.0",
            fact_delta_threshold=args.fact_delta_threshold,
            fact_max=args.fact_max, fact_full_ce_max=args.fact_full_ce_max,
            training_fact_cap="40 total; new loss up to 20 rescue + remaining retained",
            optimizer="AdamW", weight_decay=0.01,
            mathematical_upper_bound=False)
        report["diagnostic_only"] = False
        report["excluded_from_method_results"] = False
        if args.risk_protection:
            report["schema_version"] = "s1d_c_risk/1"
            report["protocol"].update(
                experiment="Fixed C basis and tail objective with write-time risk protection",
                risk_protection=risk_protocol(),
                basis="frozen C damaged-fact gradient SVD-13 from identical cached rows",
                protection="top20 kept facts by descending eligible D0 CE; lambda=1 mean hinge",
                training_fact_cap="unchanged C tail pool <=40 plus <=20 kept protection facts")

    if ORACLE_ARM in args.arms:
        report["schema_version"] = "s1d_stage2_oracle/1"
        report["protocol"].update(
            experiment="S1d dev oracle: frozen selfsup 13D, real QA answer CE",
            arms=[ORACLE_ARM], dev_windows=[[3, 17510]],
            supervision="real QA answer-token CE on rescue_d0_wrong only; first gold alias",
            selection_rule="worst stage2 selfsup dev window; post-hoc diagnostic",
            optimizer="AdamW", weight_decay=0.01,
            gradient_reduction="sum of per-question mean answer-token CE",
            mathematical_upper_bound=False)

    elif PROTECTED_ORACLE_ARM in args.arms:
        report["schema_version"] = "s1d_stage2_oracle_protected/1"
        report["protocol"]["qa_backward"] = "single-question accumulation; saved tensors CPU offload; candidate context through backward"
        report["protocol"]["parallel_qa"] = {
            "world_size": world_size, "physical_devices": devices,
            "reader_path": reader_path,
            "reader_cache_manifest_sha256": (sha256_file(Path(reader_path) / "fp16_cache_manifest.json")
                                            if args.model_path else None),
            "sharding": "round-robin QA questions; replicated frozen readers",
            "gradient_reduction": "SUM local alpha gradients with global cohort denominators",
            "evaluation": "disjoint question shards gathered to complete QA universe"}
        report["protocol"].update(
            experiment="S1d dev protected oracle: frozen selfsup 13D, QA rescue CE plus D0 protection",
            arms=[PROTECTED_ORACLE_ARM], dev_windows=[[3, 17510]],
            supervision="rescue answer-token CE plus D0-correct answer-token CE degradation penalty; first gold alias",
            protection_set="all questions D0 answers correctly",
            protect_loss="mean over questions ReLU(current answer-token CE - D0 answer-token CE)",
            protect_weight=1.0,
            selection_rule="worst stage2 selfsup dev window; post-hoc diagnostic",
            optimizer="AdamW", weight_decay=0.01,
            gradient_reduction="mean rescue answer-token CE plus mean protected hinge CE",
            mathematical_upper_bound=False)
        report["oracle_diagnostic"].update(
            supervision="rescue answer-token CE plus fixed-weight D0-correct CE degradation penalty",
            protect_weight=1.0)
    report["schema_version"] = "s1d_c_projected/1"
    report["preparation_only"] = args.prepare_protection_cache
    report["excluded_from_method_results"] = args.prepare_protection_cache
    report["protocol"].update(
        experiment="C dimension and write-time protected-basis comparison",
        projection=projection_protocol(args.basis_variant),
        basis="full-space damaged gradients; optional kept-gradient projection; SVD then truncate",
        arm_definition=args.basis_variant,
        protection="basis construction only; no hinge or additional training loss",
        training_fact_cap="unchanged C tail supervision <=40 lines",
        gradient_reduction="sum of per-line mean tail4 CE gradients",
        reported_loss_reduction="mean over supervision lines; identical to original C",
        c_reference_sha256=sha256_file(args.c_reference),
        fact_cache_sha256=sha256_file(args.fact_cache),
        projection_module_sha256=sha256_file(Path(__file__).with_name("s1d_dimension_basis.py")))
    reference = json.loads(Path(args.c_reference).read_text())
    def flush():
        if rank != 0:
            return
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        tmp.replace(out)

    flush()
    try:
        for row, start in run_windows:
            covered, indices, dropped, original = fresh_window(row, start)
            if world_size > 1:
                pool_hash = hashlib.sha256()
                for layer in model.memory.detach():
                    pool_hash.update(layer.cpu().contiguous().numpy().tobytes())
                parallel.assert_same((pool_hash.hexdigest(), manager.used), "window memory")
            mask = _compacted_mask(manager, manager.used - window_used + target_used,
                                   "cuda:0")
            line_to_block = {}
            for block_index, (lo, hi, _b, _w) in enumerate(covered):
                for line in range(lo, hi):
                    line_to_block[line] = block_index
            by_q = {q.question: q for q in original}
            by_q.update({q.question: q for q in E.probes_from_lines(
                line_cache[row], indices, len(indices) + 1)})
            probes = list(by_q.values())
            encoded = encode(probes)
            if world_size > 1:
                qa_hash = hashlib.sha256(mask.detach().cpu().numpy().tobytes())
                for question, (tokens, stem_len) in encoded.items():
                    qa_hash.update(question.encode("utf-8"))
                    qa_hash.update(str(stem_len).encode("ascii"))
                    qa_hash.update(tokens.cpu().numpy().tobytes())
                parallel.assert_same(qa_hash.hexdigest(), "QA encodings and active mask")
            full = evaluate(encoded, probes, None)
            select_arm = base()
            with torch.no_grad():
                select_arm.up.weight.zero_()
            select_correct = evaluate(encoded, probes, select_arm)
            del select_arm
            d0_arm = base()
            d0_correct = evaluate(encoded, probes, d0_arm)
            sets = window_sets(full, select_correct, d0_correct)
            protect_questions = [q for q, ok in d0_correct.items() if ok]
            protect_baseline = (answer_ce(encoded, protect_questions, d0_arm)
                                if PROTECTED_ORACLE_ARM in args.arms else {})
            print("[s1d2] (%d,%d) rescue=%d rescue_d0_wrong=%d protect=%d"
                  % (row, start, len(sets["rescue"]),
                     len(sets["rescue_d0_wrong"]), len(sets["protect"])), flush=True)

            sup, n_skipped = supervision_lines(row, start, line_to_block, dropped)
            print("[s1d2] (%d,%d) supervision lines=%d skipped_short=%d"
                  % (row, start, len(sup), n_skipped), flush=True)
            if len(sup) < 2 * MASK_SPAN + 2:
                raise RuntimeError("supervision pool too small")

            reference_sets = {k: sets[k] for k in
                ("rescue", "protect", "repair", "retained", "rescue_d0_correct", "rescue_d0_wrong")}
            validate_c_reference(reference, report["protocol"], [row, start], reference_sets,
                                 {"full": full, "select": select_correct, "d0": d0_correct}, len(sup))
            wk = "%d,%d" % (row, start)
            ss_rows = torch.stack([ss_cache["rows_by_window"][wk][q]
                                   for q in sorted(ss_cache["rows_by_window"][wk],
                                                   key=int)]).double()
            U_ss = orth(ss_rows)[:, :13].contiguous()
            U_fact = None
            fact_started = time.monotonic()
            basis_seconds = 0.0
            fact_rows_artifact = None
            if exp1_mode == "A":
                # A is the exact old-method control; it does not need fact
                # target discovery or additional Full/D0 forward passes.
                fact_records_all = fact_by_key = []
                fact_encoded = {}
                fact_full_ce = fact_d0_ce = {}
                fact_keys = fact_protect_keys = []
                fact_protect_baseline = {}
                thresholded = []
                basis_fact_keys = []
            else:
                fact_records_all = fact_records(row, indices, line_to_block)
                fact_records_drop = fact_records(row, indices, line_to_block, set(dropped))
                fact_records_keep = fact_records(
                    row, indices, line_to_block,
                    set(range(E.WINDOW_BLOCKS)) - set(dropped))
                fact_encoded = encode_fact_records(fact_records_all)
                fact_by_key = {r["key"]: r for r in fact_records_all
                               if r["key"] in fact_encoded}
                fact_keys_all = list(fact_by_key)
                cached = None
                if args.fact_cache:
                    cached = torch.load(args.fact_cache, map_location="cpu", weights_only=False)
                    if cached.get("checkpoint_sha256") != digest(args.carrier_ckpt):
                        raise RuntimeError("fact cache checkpoint mismatch")
                    if (set(cached["full_ce"]) != set(fact_keys_all)
                            or set(cached["d0_ce"]) != set(fact_keys_all)):
                        raise RuntimeError("fact cache window/complete CE map mismatch")
                    fact_full_ce = {key: float(cached["full_ce"][key]) for key in fact_keys_all}
                    fact_d0_ce = {key: float(cached["d0_ce"][key]) for key in fact_keys_all}
                else:
                    fact_full_ce = answer_ce(fact_encoded, fact_keys_all, None)
                    fact_d0_ce = answer_ce(fact_encoded, fact_keys_all, d0_arm)
                thresholded, basis_fact_keys, fact_keys, fact_protect_keys = select_fact_cohorts(
                    fact_records_drop, fact_records_keep, fact_encoded, fact_full_ce, fact_d0_ce,
                    args.fact_max, args.fact_full_ce_max, args.fact_delta_threshold)
                if not fact_keys:
                    raise RuntimeError("no valid dropped latest facts for new loss")
                if args.risk_protection:
                    fact_protect_keys = select_risk_facts(
                        fact_records_keep, fact_encoded, fact_d0_ce)
                    if not fact_protect_keys:
                        raise RuntimeError("no eligible retained risk facts; no fallback")
                fact_protect_baseline = {key: fact_d0_ce[key] for key in fact_protect_keys}
                cached_fact_rows = None
                if cached is not None:
                    if list(cached["basis_fact_keys"]) != basis_fact_keys:
                        raise RuntimeError("fact cache selected basis cohort mismatch")
                    cached_fact_rows = cached["rows"].double()
                    if cached_fact_rows.shape != (len(basis_fact_keys), 131072):
                        raise RuntimeError("fact cache gradient shape mismatch")
                    if not torch.isfinite(cached_fact_rows).all():
                        raise RuntimeError("fact cache gradient is not finite")
            # Validate original C before any new gradient collection.
            ba = reference["windows"][0]["arms"]["C"]
            if (basis_fact_keys != ba["basis_fact_keys"] or fact_full_ce != ba["fact_full_ce"]
                    or fact_d0_ce != ba["fact_d0_ce"] or list(dropped) != ba["dropped_blocks"]):
                raise ValueError("original C fact selection/input drift")
            fact_rows = cached_fact_rows
            if fact_rows is None:
                raise ValueError("original C raw gradient cache required")
            original_c13_basis_sha256 = hashlib.sha256(
                orth(fact_rows)[:, :13].contiguous().numpy().tobytes()).hexdigest()
            if original_c13_basis_sha256 != ba["basis_sha256"]:
                raise ValueError("original C raw gradients do not reproduce published basis")
            protection_keys = select_protection(fact_records_keep, fact_encoded, fact_d0_ce)
            source_sha256 = memory_digest(model.memory)
            metadata = dict(window=[row, start], carrier_sha256=digest(args.carrier_ckpt),
                fact_cache_sha256=sha256_file(args.fact_cache),
                c_reference_sha256=sha256_file(args.c_reference),
                source_memory_sha256=source_sha256,
                active_mask_sha256=tensor_sha256(mask),
                context_lines_sha256=hashlib.sha256("\n".join(line_cache[row]).encode()).hexdigest(),
                qa_max_tokens=args.qa_max_tokens,
                gradient_driver_sha256=SCRIPT_SHA,
                projection_module_sha256=report["protocol"]["projection_module_sha256"],
                reader_manifest_sha256=(sha256_file(Path(reader_path)/"fp16_cache_manifest.json")
                    if (Path(reader_path)/"fp16_cache_manifest.json").exists() else None))
            if args.prepare_protection_cache:
                if not protection_keys:
                    raise ValueError("no eligible retained facts; no fallback")
                started = time.monotonic()
                protection_rows = fact_gradient_rows(d0_arm, fact_encoded, protection_keys, mask)
                if memory_digest(model.memory) != source_sha256:
                    raise ValueError("gradient collection changed source memory")
                saved = dict(schema="s1d_protection_rows/1", metadata=metadata,
                    keys=protection_keys, rows=protection_rows,
                    rows_sha256=tensor_sha256(protection_rows))
                validate_protection_cache(saved, metadata, protection_keys, fact_rows.shape[1])
                # Exclusive creation protects shared artifacts from concurrent preparation.
                cache_path = Path(args.protection_rows)
                save_new_cache(saved, cache_path)
                report.update(status="complete", elapsed_seconds=time.monotonic()-run_started,
                    preparation=dict(metadata=metadata, keys=protection_keys,
                        rows_sha256=saved["rows_sha256"], path=str(cache_path),
                        cache_sha256=sha256_file(cache_path),
                        elapsed_seconds=time.monotonic()-started))
                flush()
                return
            dimension, projected = VARIANTS[args.basis_variant]
            protection_rows = None
            if projected:
                saved = torch.load(args.protection_rows, map_location="cpu", weights_only=False)
                protection_rows = validate_protection_cache(saved, metadata, protection_keys, fact_rows.shape[1])
            U_fact, projection_audit = build_basis(fact_rows, dimension, protection_rows)
            protection_cache_sha256 = sha256_file(args.protection_rows) if projected else None
            fact_protect_keys = protection_keys if projected else []
            fact_protect_baseline = {k: fact_d0_ce[k] for k in fact_protect_keys}
            fact_basis_provenance = "cached damaged rows" + (" projected against cached kept rows" if projected else "")
            fact_rows_artifact = args.fact_cache
            report["projection_audit"] = projection_audit
            report["protection_cache_sha256"] = protection_cache_sha256
            report["source_metadata"] = metadata
            basis_sha256 = hashlib.sha256(
                U_ss.contiguous().numpy().tobytes()).hexdigest()
            fact_basis_sha256 = (hashlib.sha256(
                U_fact.contiguous().numpy().tobytes()).hexdigest()
                if U_fact is not None else None)
            parallel.assert_same(basis_sha256, "frozen basis")
            random_ul = random_basis(131072, 13, 0, torch)

            if exp1_mode is not None:
                print("[s1d2] fact targets=%d thresholded=%d protect=%d basis=%s"
                      % (len(fact_keys), len(thresholded), len(fact_protect_keys),
                         fact_basis_provenance or "old"), flush=True)

            if args.skip_generation:
                gen_full = gen_d0 = None
                gen_full_pred = gen_d0_pred = {}
                print("[s1d2] generation baselines skipped by protocol", flush=True)
            else:
                gen_full, gen_full_pred = evaluate_generation(probes, None)
                gen_d0, gen_d0_pred = evaluate_generation(probes, d0_arm)
                d0_after_generation = evaluate(encoded, probes, d0_arm)
                if d0_after_generation != d0_correct:
                    raise RuntimeError("D0 changed after generation evaluation")
                print("[s1d2] D0 generation-sandwich verified", flush=True)
            d0_arm_alive = True

            window_report = {
                "window": [row, start],
                "sets": {k: sets[k] for k in
                         ("rescue", "protect", "repair", "retained",
                          "rescue_d0_correct", "rescue_d0_wrong")},
                "baselines": {
                    "full": stage2_metric(full, sets, d0_correct),
                    "select": stage2_metric(select_correct, sets, d0_correct),
                    "d0": stage2_metric(d0_correct, sets, d0_correct),
                    "gen_full": (stage2_metric(gen_full, sets, gen_d0)
                                 if gen_full is not None else None),
                    "gen_d0": (stage2_metric(gen_d0, sets, gen_d0)
                               if gen_d0 is not None else None),
                    "gen_predictions_d0": gen_d0_pred,
                    "gen_predictions_full": gen_full_pred,
                },
                "n_supervision_lines": len(sup),
                "arms": {},
            }
            # Keep d0_arm alive through the arm loop: each arm's identity-at-init
            # check re-evaluates D0 in the same kernel/memory-layout state.
            # Released after the window's arms finish.
            gc.collect()
            torch.cuda.empty_cache()

            if exp1_mode is not None:
                exp1_basis = U_ss if exp1_mode in ("A", "B") else U_fact
                arms_plan = ((exp1_mode, exp1_basis),)
            else:
                arms_plan = (("selfsup_13d", U_ss),
                             ("random_13d", random_ul),
                             ("unconstrained", None),
                             (ORACLE_ARM, U_ss),
                             (PROTECTED_ORACLE_ARM, U_ss))
            for name, basis in arms_plan:
                if exp1_mode is None and name not in args.arms:
                    continue
                torch.manual_seed(0)
                arm = base()
                displacement = make_arm_displacement(arm.down.weight, basis, torch)
                if basis is None:
                    displacement = make_arm_displacement(arm.down.weight, None, torch)
                parametrize.register_parametrization(arm.down, "weight", displacement)
                optimizer = torch.optim.AdamW([displacement.value], lr=args.lr,
                                              weight_decay=0.01)
                if not torch.equal(arm.down.weight, d0_arm.down.weight):
                    raise RuntimeError("identity-at-init weight mismatch: %s" % name)
                # Functional equality remains strict; it guards the full reader path.
                d0_ref = evaluate(encoded, probes, d0_arm) if d0_arm_alive else None
                initial = evaluate(encoded, probes, arm)
                mismatches = [q for q in initial if initial[q] != d0_ref[q]]
                if mismatches:
                    raise RuntimeError("identity-at-init failed: %s/%d,%d "
                                       "flipped=%s" % (name, row, start,
                                                       mismatches[:5]))
                print("[s1d2] %s identity-at-init verified" % name, flush=True)
                history = []
                arm_entry = {"history": history,
                             "trainable_dof": int(displacement.value.numel()),
                             "final": stage2_metric(initial, sets, d0_correct),
                             "initial_identity_verified": True}
                if exp1_mode is not None:
                    arm_entry.update(
                        basis_type=("old_selfsup_13d" if exp1_mode in ("A", "B")
                                    else "new_damaged_fact_answer_13d"),
                        loss_type=("tail_4_token" if exp1_mode in ("A", "C")
                                   else "fact_answer_span"),
                        basis_sha256=(basis_sha256 if exp1_mode in ("A", "B")
                                      else fact_basis_sha256),
                        old_basis_sha256=basis_sha256,
                        fact_basis_provenance=fact_basis_provenance,
                        basis_fact_keys=basis_fact_keys,
                        fact_discovery_and_basis_seconds=time.monotonic() - fact_started,
                        basis_seconds=basis_seconds, fact_rows_artifact=fact_rows_artifact,
                        explicit_basis_fp32_bytes=131072 * 13 * 4,
                        explicit_basis_fp16_bytes=131072 * 13 * 2,
                        fact_full_ce=fact_full_ce, fact_d0_ce=fact_d0_ce,
                        fact_target_records=[fact_by_key[k] for k in fact_keys],
                        fact_target_keys=fact_keys,
                        fact_target_lines=[fact_by_key[key]["line_idx"] for key in fact_keys],
                        fact_target_deltas={key: fact_d0_ce[key] - fact_full_ce[key]
                                            for key in fact_keys},
                        protection_fact_keys=fact_protect_keys,
                        protection_fact_records=[fact_by_key[k] for k in fact_protect_keys],
                        basis_fact_records=[fact_by_key[k] for k in basis_fact_keys],
                        dropped_blocks=list(dropped),
                        protection_fact_baseline_ce=fact_protect_baseline,
                        fact_delta_threshold=args.fact_delta_threshold)
                arm_entry.update(
                    basis_type="projected_damaged_fact_gradients" if projected else "damaged_fact_gradients_%dd" % dimension,
                    projection_variant=args.basis_variant,
                    projection_audit=projection_audit,
                    original_c13_basis_sha256=original_c13_basis_sha256,
                    explicit_basis_fp32_bytes=displacement.basis.numel()*4,
                    explicit_basis_fp16_bytes=displacement.basis.numel()*2,
                    projection_basis_sha256=tensor_sha256(displacement.basis),
                    protection_cache_sha256=protection_cache_sha256,
                    projection_all_fact_records=[r for r in fact_records_all if r["key"] in fact_encoded],
                    projection_kept_fact_records=[r for r in fact_records_keep if r["key"] in fact_encoded],
                    tail_supervision_token_sha256=hashlib.sha256(
                        b"".join(tokens.contiguous().numpy().tobytes() for _, tokens in sup)).hexdigest(),
                    tail_supervision_line_indices=[i for i, _ in sup],
                    initial_correct=initial)
                if exp1_mode:
                    frozen_basis = displacement.basis.detach().clone()
                    if args.risk_protection:
                        arm_entry["risk_protection"] = risk_protocol()
                        arm_entry["risk_candidate_records"] = [
                            r for r in fact_records_keep if r["key"] in fact_encoded]
                        arm_entry["risk_all_fact_records"] = [
                            r for r in fact_records_all if r["key"] in fact_encoded]
                        arm_entry["tail_supervision_token_sha256"] = hashlib.sha256(
                            b"".join(tokens.contiguous().numpy().tobytes() for _, tokens in sup)).hexdigest()
                        arm_entry["tail_supervision_line_indices"] = [i for i, _ in sup]
                        arm_entry["risk_selection_sha256"] = sha256_file(
                            str(Path(__file__).with_name("s1d_risk_protection.py")))
                    if any(p.requires_grad for p in model.parameters()):
                        raise RuntimeError("reader is not frozen")
                    trainable = [p for p in arm.parameters() if p.requires_grad]
                    if trainable != [displacement.value]:
                        raise RuntimeError("only alpha must be trainable")
                window_report["arms"][name] = arm_entry
                if name in QA_ORACLE_ARMS:
                    if U_ss.shape != (131072, 13) or not sets["rescue_d0_wrong"]:
                        raise RuntimeError("QA oracle requires rank 13 and nonempty QA supervision")
                    frozen_basis = displacement.basis.detach().clone()
                    arm_entry.update(
                        basis_sha256=basis_sha256,
                        supervision_questions=sets["rescue_d0_wrong"],
                        n_supervision_questions=len(sets["rescue_d0_wrong"]),
                        supervision_answers={q: by_q[q].answers[0]
                                             for q in sets["rescue_d0_wrong"]})
                    if name == PROTECTED_ORACLE_ARM:
                        arm_entry.update(
                            protection_questions=protect_questions,
                            n_protection_questions=len(protect_questions),
                            protection_weight=1.0,
                            protection_baseline_ce=protect_baseline,
                            protection_loss="mean ReLU(current CE - D0 CE)")
                    if any(p.requires_grad for p in model.parameters()):
                        raise RuntimeError("QA oracle reader must remain frozen")
                    trainable = [p for p in arm.parameters() if p.requires_grad]
                    if len(trainable) != 1 or trainable[0] is not displacement.value:
                        raise RuntimeError("QA oracle must train only alpha")
                report["windows"] = [w for w in report["windows"]
                                     if w["window"] != [row, start]]
                report["windows"].append(window_report)
                flush()
                train_started = time.monotonic()
                for step in range(1, args.steps + 1):
                    print("[s1d2] %s training step %d/%d" %
                          (name, step, args.steps), flush=True)
                    if args.risk_protection:
                        loss_info = train_arm(arm, displacement, optimizer, sup, mask,
                            protection=(fact_encoded, fact_protect_keys, fact_protect_baseline))
                    elif exp1_mode in ("B", "D"):
                        loss_info = train_answer_arm(
                            arm, displacement, optimizer, fact_encoded,
                            fact_keys, mask,
                            protect_questions=fact_protect_keys,
                            protect_baseline=fact_protect_baseline,
                            protect_weight=1.0,
                            mean_grad=True)
                    elif name == ORACLE_ARM:
                        loss_info = train_answer_arm(
                            arm, displacement, optimizer, encoded,
                            sets["rescue_d0_wrong"], mask)
                    elif name == PROTECTED_ORACLE_ARM:
                        loss_info = train_answer_arm(
                            arm, displacement, optimizer, encoded,
                            sets["rescue_d0_wrong"], mask,
                            protect_questions=protect_questions,
                            protect_baseline=protect_baseline,
                            protect_weight=1.0,
                            mean_grad=True)
                    else:
                        loss_info = {"loss": train_arm(
                            arm, displacement, optimizer, sup, mask)}
                    loss = loss_info["loss"]
                    if world_size > 1:
                        parallel.assert_same(displacement.value.detach().cpu().tolist(), "alpha update")
                    if step % args.measure_every == 0 or step == args.steps:
                        tf = evaluate(encoded, probes, arm)
                        tf_m = stage2_metric(tf, sets, d0_correct)
                        tf_m["step"] = step
                        tf_m["tf_correct_count"] = sum(tf.values())
                        tf_m["training_elapsed_seconds"] = time.monotonic() - train_started
                        tf_m["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
                        tf_m["sup_loss"] = loss
                        if args.risk_protection:
                            for key in ("tail_loss_mean", "tail_loss_sum", "protect_penalty"):
                                tf_m[key] = loss_info[key]
                        if "rescue_loss" in loss_info:
                            tf_m["rescue_loss"] = loss_info["rescue_loss"]
                            tf_m["protect_penalty"] = loss_info.get("protect_penalty", 0.0)
                        tf_m["deviation"] = float(displacement.value.detach().norm())
                        tf_m["down_delta_norm"] = float((arm.down.weight.detach()-d0_arm.down.weight.detach()).norm())
                        if exp1_mode or name in QA_ORACLE_ARMS:
                            if not torch.equal(displacement.basis, frozen_basis):
                                raise RuntimeError("oracle basis changed during training")
                            tf_m["basis_unchanged"] = True
                            tf_m["alpha"] = displacement.value.detach().cpu().tolist()
                        if step == args.steps and not args.skip_generation:
                            # Generation EM only at the final step.
                            gen, gen_pred = evaluate_generation(probes, arm)
                            gen_m = stage2_metric(gen, sets, gen_d0)
                            gen_m["predictions"] = gen_pred
                            tf_m["gen"] = gen_m
                            print_extra = (" | GEN newR=%d harm=%d net=%d"
                                           % (gen_m["new_recovery"],
                                              gen_m["d0_harmed"],
                                              gen_m["net_vs_d0"]))
                        else:
                            print_extra = ""
                        history.append(tf_m)
                        arm_entry["final"] = tf_m
                        report["windows"] = [w for w in report["windows"]
                                             if w["window"] != [row, start]]
                        report["windows"].append(window_report)
                        flush()
                        print("[s1d2] (%d,%d) %s step=%d loss=%.4f dev=%.3f "
                              "TF newR=%d harm=%d net=%d%s"
                              % (row, start, name, step, loss, tf_m["deviation"],
                                 tf_m["new_recovery"], tf_m["d0_harmed"],
                                 tf_m["net_vs_d0"], print_extra), flush=True)
                if exp1_mode:
                    checkpoint = out.with_suffix(".alpha.pt")
                    torch.save({"window": [row, start], "arm": name,
                                "alpha": displacement.value.detach().cpu(),
                                "basis": displacement.basis.detach().cpu(),
                                "checkpoint_sha256": digest(args.carrier_ckpt),
                                "driver_sha256": SCRIPT_SHA,
                                "projection_variant": args.basis_variant,
                                "projection_protocol": projection_protocol(args.basis_variant),
                                "projection_basis_sha256": tensor_sha256(displacement.basis),
                                "fact_cache_sha256": sha256_file(args.fact_cache),
                                "protection_cache_sha256": protection_cache_sha256}, checkpoint)
                    arm_entry["residual_checkpoint"] = str(checkpoint)
                    arm_entry["residual_checkpoint_sha256"] = sha256_file(checkpoint)
                del arm, displacement, optimizer
                gc.collect()
                torch.cuda.empty_cache()
            del d0_arm
            d0_arm_alive = False
            report["windows"] = [w for w in report["windows"]
                                 if w["window"] != [row, start]]
            report["windows"].append(window_report)
            gc.collect()
            torch.cuda.empty_cache()
            flush()
        if world_size > 1:
            torch.distributed.barrier()
        report.update(status="complete", elapsed_seconds=time.monotonic() - run_started)
        flush()
        print("[s1d2] dev complete", flush=True)
    except Exception as exc:
        report.update(status="failed",
                      failure={"type": type(exc).__name__, "message": str(exc)})
        flush()
        raise


if __name__ == "__main__":
    main()
