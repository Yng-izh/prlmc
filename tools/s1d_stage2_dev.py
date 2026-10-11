#!/usr/bin/env python
"""S1d-B stage 2 dev: answer-free residual training on the self-sup 13D basis.

The stage-1 question (can an answer-free basis be built?) closed with gate
PASS (runs/s1d_selfsup_basis_v2.json).  This driver answers the stage-2
question the user posed: does *training* the 13 residual coefficients on the
answer-free completion loss actually rescue knowledge the compaction lost?

    D_new = D0 + reshape(U_selfsup @ alpha)     (13 trainable DOF per window)

Arms per window (identical supervision, budget and steps):
  * selfsup_13d     -- the method under test (frozen window self-sup basis)
  * random_13d      -- same DOF, seed-0 random orthonormal basis
  * unconstrained   -- full down.weight displacement (does the constraint
                       cost recovery?)
  * d0 / select     -- baselines, no training

Training loss: tail-4-token autoregressive CE of the window's OWN dropped-block
fact lines under the candidate pool -- exactly the stage-1 supervision.  NO
held-out QA answer ever enters a gradient step.  QA answers are used only to
SCORE the arms after training (diagnostic readout), which is the same
train/score split every S1c arm used.

Dev windows: (3,16370) (best gRho 0.696) and (3,17510) (worst gRho 0.370) --
chosen on stage-1 geometry, NOT on QA results, per the user's directive.
Held-out formal windows (2,3450)(2,3830) remain untouched until the protocol
is frozen.

Metrics fix the S1c counting flaw the user called out: recovery is counted
ONLY over questions D0 got WRONG and full-pool got right (true new recovery),
harm is counted over questions D0 got RIGHT (additional damage), and net =
new_recovery - d0_harmed.  Both teacher-forced answer-token exact match AND
real greedy generation EM (normalized alias EM, native_qa_compare semantics,
max_new_tokens=32) are recorded per question.

Budget fairness: every optimized arm clamps its displacement to the same norm
ceiling (default 1.0) every step and the realized norm is logged; identical
steps/lr/optimizer; identity-at-init verified against D0.
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
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="runs/s1c_global_rows.pt")
    ap.add_argument("--selfsup-rows", default="runs/s1d_selfsup_rows_v2.pt")
    ap.add_argument("--carrier-ckpt", default="runs/e6d_interleave_ckpt.pt")
    ap.add_argument("--out", default="runs/s1d_stage2_dev.json")
    ap.add_argument("--cuda-device", type=int, default=9)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--measure-every", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--windows", default="dev",
                    choices=["dev"], help="formal run is a separate frozen driver")
    ap.add_argument("--window", default=None,
                    help="comma pair row,start -- run a single window (sharding); "
                         "output must be per-window when used")
    ap.add_argument("--arms", nargs="+", choices=ALL_ARMS,
                    default=list(ARMS))
    args = ap.parse_args()

    SCRIPT_SHA = sha256_file(os.path.abspath(__file__))
    out = Path(args.out)
    if out.exists():
        raise SystemExit("output exists: %s" % out)
    run_windows = DEV_WINDOWS
    if args.window is not None:
        row_s, start_s = args.window.split(",")
        run_windows = [(int(row_s), int(start_s))]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    os.environ["S0B_GPU_SET"] = "1"
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
    with open(CONFIG) as f:
        import yaml
        config = dict(yaml.safe_load(f)["prlmc"])
    config.update(device="cuda:0", dtype="float16",
                  stats_dir="./outputs/prlmc-s1d-stage2-dev")
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
    disable_reader_dropout(model)
    probe_args = argparse.Namespace(probes=60, max_probe_tokens=256)
    from methods.prlmc.consolidate import CarrierResidualCompactor
    state = torch.load(args.carrier_ckpt, map_location="cuda:0", weights_only=True)

    ss_cache = torch.load(args.selfsup_rows, map_location="cpu", weights_only=False)
    if ss_cache["checkpoint_sha256"] != digest(args.carrier_ckpt):
        raise SystemExit("selfsup rows checkpoint mismatch")
    gold = torch.load(args.rows, map_location="cpu", weights_only=False)
    if gold["checkpoint_sha256"] != digest(args.carrier_ckpt):
        raise SystemExit("gold rows checkpoint mismatch")

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
            width = min(256, tokens.shape[1])
            if stem_len <= 0 or stem_len >= width:
                raise RuntimeError("missing answer span: " + q.question)
            encoded[q.question] = (tokens[:, :256].to("cuda:0"), stem_len)
        return encoded

    @torch.no_grad()
    def evaluate(encoded, probes, arm=None):
        correct = {}
        pool, mask_m = (None, None) if arm is None else E.candidate_pool(manager, arm, "cuda:0")
        context = differentiable_memory(model, pool) if pool is not None else nullcontext()
        guard = active_mask_override(model, mask_m) if mask_m is not None else nullcontext()
        with reader_evaluation(model), context, guard:
            for q in probes:
                input_ids, stem_len = encoded[q.question]
                logits = model(input_ids=input_ids, use_cache=False, return_dict=True).logits
                span = logits[:, stem_len - 1:-1, :]
                labels = input_ids[:, stem_len:]
                correct[q.question] = bool((span.argmax(dim=-1) == labels).all(dim=-1).item())
                del logits, span, labels
        del pool, mask_m
        torch.cuda.empty_cache()
        return correct

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
            for question in questions:
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
        return losses

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

    def train_answer_arm(arm, displacement, optimizer, encoded, questions, mask,
                         protect_questions=None, protect_baseline=None,
                         protect_weight=0.0, mean_grad=False):
        """QA step with optional fixed-weight no-degradation penalty."""
        optimizer.zero_grad(set_to_none=True)
        cand = candidate(arm)
        rescue_total = 0.0
        protect_total = 0.0
        protect_questions = protect_questions or []
        protect_baseline = protect_baseline or {}
        with _pool_in_place(manager, "cuda:0", cand, window_used,
                            target_used) as injection:
            pool = injection.reinject()
            for question in questions:
                input_ids, stem_len = encoded[question]
                labels = input_ids.clone()
                labels[:, :stem_len] = -100
                with differentiable_memory(model, pool), active_mask_override(model, mask):
                    logits = model(input_ids=input_ids, use_cache=False, return_dict=True).logits
                shift_logits = logits[:, :-1, :].float()
                shift_labels = labels[:, 1:]
                loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1), ignore_index=-100)
                (loss / len(questions) if mean_grad else loss).backward(retain_graph=True)
                rescue_total += float(loss)
                del loss, logits
            for question in protect_questions:
                input_ids, stem_len = encoded[question]
                labels = input_ids.clone()
                labels[:, :stem_len] = -100
                with differentiable_memory(model, pool), active_mask_override(model, mask):
                    logits = model(input_ids=input_ids, use_cache=False, return_dict=True).logits
                shift_logits = logits[:, :-1, :].float()
                shift_labels = labels[:, 1:]
                loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1), ignore_index=-100)
                penalty = torch.relu(loss - protect_baseline[question])
                (protect_weight * penalty / len(protect_questions)
                 if mean_grad else protect_weight * penalty).backward(retain_graph=True)
                protect_total += float(penalty)
                del penalty, loss, logits
        optimizer.step()
        displacement.clamp(args.budget)
        del cand, pool
        torch.cuda.empty_cache()
        return {
            "loss": rescue_total / max(1, len(questions)) +
                    protect_weight * protect_total / max(1, len(protect_questions)),
            "rescue_loss": rescue_total / max(1, len(questions)),
            "protect_penalty": protect_total / max(1, len(protect_questions)),
        }

    def train_arm(arm, displacement, optimizer, sup, mask):
        """One training step over the answer-free completion CE."""
        optimizer.zero_grad(set_to_none=True)
        cand = candidate(arm)
        total = 0.0
        with _pool_in_place(manager, "cuda:0", cand, window_used,
                            target_used) as injection:
            pool = injection.reinject()
            for _line_idx, tokens in sup:
                input_ids = tokens.to("cuda:0")
                labels = input_ids.clone()
                labels[:, :-MASK_SPAN] = -100
                with differentiable_memory(model, pool), active_mask_override(model, mask):
                    logits = model(input_ids=input_ids, return_dict=True).logits
                shift_logits = logits[:, :-1, :].float()
                shift_labels = labels[:, 1:]
                loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1))
                loss.backward(retain_graph=True)
                total += float(loss)
                del loss, logits
        optimizer.step()
        displacement.clamp(args.budget)
        del cand, pool
        torch.cuda.empty_cache()
        return total / max(1, len(sup))

    if any(name in args.arms for name in QA_ORACLE_ARMS) and (
            args.window != "3,17510" or len(args.arms) != 1):
        raise SystemExit("QA oracle arms require an isolated run on window 3,17510")

    report = {
        "schema_version": "s1d_stage2_dev/1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "driver_sha256": SCRIPT_SHA,
        "protocol": {**PROTOCOL, "steps": args.steps,
                     "reader_evaluation": "eval; exact mixed-mode and RNG restoration; full-memory integrity check",
                     "selected_arms": args.arms,
                     "measure_every": args.measure_every,
                     "lr": args.lr, "budget": args.budget,
                     "max_new_tokens": args.max_new_tokens,
                     "checkpoint_sha256": digest(args.carrier_ckpt),
                     "selfsup_rows_sha256": sha256_file(args.selfsup_rows),
                     "gold_rows_sha256": sha256_file(args.rows)},
        "diagnostic_only": any(name in args.arms for name in QA_ORACLE_ARMS),
        "excluded_from_method_results": any(name in args.arms for name in QA_ORACLE_ARMS),
        "oracle_diagnostic": ({
            "purpose": "finite-step reachability diagnostic, not a mathematical upper bound",
            "supervision": "real QA answer-token CE on rescue_d0_wrong only",
            "basis": "unchanged selfsup SVD-13 basis; no QA-based basis rebuild",
            "evaluation": "same complete QA universe and D0-relative metrics",
        } if any(name in args.arms for name in QA_ORACLE_ARMS) else None),
        "windows": [],
    }

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
    def flush():
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        tmp.replace(out)

    flush()
    try:
        for row, start in run_windows:
            covered, indices, dropped, original = fresh_window(row, start)
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

            wk = "%d,%d" % (row, start)
            ss_rows = torch.stack([ss_cache["rows_by_window"][wk][q]
                                   for q in sorted(ss_cache["rows_by_window"][wk],
                                                   key=int)]).double()
            U_ss = orth(ss_rows)[:, :13].contiguous()
            basis_sha256 = hashlib.sha256(
                U_ss.contiguous().numpy().tobytes()).hexdigest()
            random_ul = random_basis(131072, 13, 0, torch)

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
                    "gen_full": stage2_metric(gen_full, sets, gen_d0),
                    "gen_d0": stage2_metric(gen_d0, sets, gen_d0),
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

            arms_plan = (("selfsup_13d", U_ss),
                         ("random_13d", random_ul),
                         ("unconstrained", None),
                         (ORACLE_ARM, U_ss),
                         (PROTECTED_ORACLE_ARM, U_ss))
            for name, basis in arms_plan:
                if name not in args.arms:
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
                for step in range(1, args.steps + 1):
                    if name == ORACLE_ARM:
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
                    if step % args.measure_every == 0 or step == args.steps:
                        tf = evaluate(encoded, probes, arm)
                        tf_m = stage2_metric(tf, sets, d0_correct)
                        tf_m["step"] = step
                        tf_m["sup_loss"] = loss
                        if "rescue_loss" in loss_info:
                            tf_m["rescue_loss"] = loss_info["rescue_loss"]
                            tf_m["protect_penalty"] = loss_info.get("protect_penalty", 0.0)
                        tf_m["deviation"] = float(displacement.value.detach().norm())
                        if name in QA_ORACLE_ARMS:
                            if not torch.equal(displacement.basis, frozen_basis):
                                raise RuntimeError("oracle basis changed during training")
                            tf_m["basis_unchanged"] = True
                            tf_m["alpha"] = displacement.value.detach().cpu().tolist()
                        if step == args.steps:
                            # Generation EM only at the final step: the KV-cache
                            # fix makes it ~10 s/probe instead of ~60 s, but 60
                            # probes x 6 measurement points x 3 arms is still
                            # 2+ hours of pure eval per window. TF every 10
                            # steps, generation once at the end.
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
        report.update(status="complete")
        flush()
        print("[s1d2] dev complete", flush=True)
    except Exception as exc:
        report.update(status="failed",
                      failure={"type": type(exc).__name__, "message": str(exc)})
        flush()
        raise


if __name__ == "__main__":
    main()
