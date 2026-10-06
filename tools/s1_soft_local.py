"""S1-soft-local: fixed historical 13D UL and one-sided finite-CE protection."""
import argparse
import json
import math
import os
from pathlib import Path

from s1_local_protected import digest, disable_reader_dropout


ARMS = ("ul", "soft_p", "soft_k")
FORMAL_BASIS_SHA256 = "3af25e44b94464432a0221e6dabe8685ef2bd91c522addcbe279387425bb97b2"


def derive_sets(original, full, selected, d0):
    questions = list(full)
    if set(questions) != set(selected) or set(questions) != set(d0):
        raise ValueError("baseline question universes differ")
    if len(original) != 13 or len(set(original)) != 13:
        raise ValueError("original supervision must be 13 distinct questions")
    if any(q not in full or not full[q] or d0[q] for q in original):
        raise ValueError("historical original target no longer full-correct/D0-wrong")
    retained = [q for q in questions if full[q] and selected[q]]
    rescue = [q for q in questions if full[q] and not selected[q]]
    return {"original_lost": list(original), "retained": retained,
            "rescue": rescue, "protect": [q for q in retained if d0[q]],
            "repair": [q for q in retained if not d0[q]],
            "rescue_d0_correct": [q for q in rescue if d0[q]],
            "rescue_d0_wrong": [q for q in rescue if not d0[q]]}


def soft_regret(current, reference):
    import torch
    reference = torch.as_tensor(reference, device=current.device, dtype=current.dtype).detach()
    return (current - reference).relu().square()


def make_displacement(weight, basis, budget=1.0):
    import torch

    class Displacement(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("basis", basis.to(weight).contiguous())
            self.value = torch.nn.Parameter(weight.new_zeros(basis.shape[1]))

        def forward(self, original):
            return original + (self.basis @ self.value).reshape_as(original)

        def clamp(self):
            with torch.no_grad():
                self.value.mul_((budget / self.value.norm().clamp_min(1e-12)).clamp_max(1))

    return Displacement()


def metric_entry(scored, sets, baselines, protection=(), lam=0.0, step=0, deviation=0.0):
    correct, ce = scored["correct"], scored["ce"]
    entry = {"step": step, "correct": correct, "ce": ce, "deviation": deviation}
    groups = [("original_rescue", "original_rescued_questions", sets["original_lost"], True),
              ("rescue_recovered", "rescue_recovered_questions", sets["rescue"], True),
              ("rescue_d0_harmed", "rescue_d0_harmed_questions", sets["rescue_d0_correct"], False),
              ("repair_recovered", "repair_recovered_questions", sets["repair"], True),
              ("protect_harmed", "protect_harmed_questions", sets["protect"], False),
              ("retained_harmed", "retained_harmed_questions", sets["retained"], False),
              ("d0_new_harmed_all", "d0_new_harmed_questions_all",
               [q for q, v in baselines["d0"]["correct"].items() if v], False)]
    for count, names, group, wanted in groups:
        found = [q for q in group if correct[q] == wanted]
        entry[count], entry[names] = len(found), found
    entry["accuracy"] = sum(correct.values()) / len(correct)
    entry["net_vs_select"] = entry["rescue_recovered"] - entry["retained_harmed"]
    entry["loss_original"] = sum(ce[q] for q in sets["original_lost"])
    entry["penalty"] = (13 * lam * sum(max(0.0, ce[q] - baselines["d0"]["ce"][q]) ** 2
                                     for q in protection) / len(protection)) if protection else 0.0
    return entry


def candidate_gate(candidate, control):
    return {"original_rescue_at_least_7": candidate["original_rescue"] >= 7,
            "no_new_protect_harm": candidate["protect_harmed"] == 0,
            "rescue_not_below_control": candidate["rescue_recovered"] >= control["rescue_recovered"],
            "repair_not_below_control": candidate["repair_recovered"] >= control["repair_recovered"],
            "harm_not_above_control": candidate["retained_harmed"] <= control["retained_harmed"],
            "net_at_least_8": candidate["net_vs_select"] >= 8,
            "strict_improvement": (candidate["rescue_recovered"] > control["rescue_recovered"] or
                                   candidate["repair_recovered"] > control["repair_recovered"] or
                                   candidate["retained_harmed"] < control["retained_harmed"])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("carrier-ckpt", "basis", "historical-result", "gradient-audit", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--cuda-device", type=int, default=9)
    p.add_argument("--steps", type=int)
    p.add_argument("--measure-every", type=int)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    expected_steps, expected_cadence = (2, 1) if args.smoke else (60, 5)
    args.steps = expected_steps if args.steps is None else args.steps
    args.measure_every = expected_cadence if args.measure_every is None else args.measure_every
    out = Path(args.out)
    if out.exists() or any(out.with_suffix("." + name + ".pt").exists() for name in ARMS):
        raise SystemExit("output exists")
    if args.steps != expected_steps or args.measure_every != expected_cadence:
        raise SystemExit("protocol requires smoke 2/1 or formal 60/5")
    historical = json.loads(Path(args.historical_result).read_text())
    if historical.get("status") != "complete" or historical["protocol"]["smoke"]:
        raise SystemExit("historical result must be formal and complete")
    # The repaired run must cite the CPU audit that proved the aliased-source
    # gradient wrong and the cloned-source gradient correct; without it the
    # protocol change from `ae254bf` would be untraceable in the report.
    gradient_audit = json.loads(Path(args.gradient_audit).read_text())
    audit_verdict = {key: bool(value) for key, value in gradient_audit["verdict"].items()}
    if not all(audit_verdict.values()):
        raise SystemExit("gradient repair audit verdict failed: " + json.dumps(audit_verdict))
    if digest(args.basis) != FORMAL_BASIS_SHA256:
        raise SystemExit("formal basis hash mismatch")
    if digest(args.carrier_ckpt) != historical["protocol"]["checkpoint_sha256"]:
        raise SystemExit("source checkpoint hash mismatch")
    if digest(Path(__file__).with_name("s1_local_protected.py")) != historical["protocol"]["driver_sha256"]:
        raise SystemExit("historical driver changed")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    os.environ["S0B_GPU_SET"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    import torch.nn.functional as F
    import yaml
    import e6a_overfit as E
    from s0_functional_sensitivity import build_window_and_probes, margin_scalar, CONFIG, MODEL
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.consolidate import CarrierResidualCompactor
    from methods.prlmc.layout import ProtectedResidualMemory
    from methods.prlmc.probes import _pool_in_place, _compacted_mask, active_mask_override, differentiable_memory
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from transformers import AutoTokenizer
    from torch.nn.utils import parametrize
    from contextlib import nullcontext

    torch.manual_seed(0)
    torch.set_num_threads(4)
    with open(CONFIG) as f:
        config = dict(yaml.safe_load(f)["prlmc"])
    config.update(device="cuda:0", dtype="float16", stats_dir="./outputs/prlmc-s1-soft")
    cfg = PRLMCConfig.from_dict(config)
    model = MemoryLLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                     attn_implementation="sdpa").to("cuda:0")
    model.initialized.zero_()
    for x in model.parameters():
        x.requires_grad_(False)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    manager = ProtectedResidualMemory(model, AutoTokenizer.from_pretrained(MODEL), cfg)
    args.probes, args.max_probe_tokens = 60, 256
    lines = E.read_fact_lines(3)
    built = build_window_and_probes(manager, {3: lines}, 3, 17890, args)
    if built is None:
        raise RuntimeError("window build failed")
    covered, indices, dropped, original = built
    if float(getattr(model.config, "attention_dropout", 0.0)) != 0.0:
        raise RuntimeError("nonzero attention dropout")
    dropout = disable_reader_dropout(model)
    by_q = {q.question: q for q in original}
    by_q.update({q.question: q for q in E.probes_from_lines(lines, indices, len(indices) + 1)})
    probes = list(by_q.values())
    state = torch.load(args.carrier_ckpt, map_location="cuda:0", weights_only=True)
    saved = torch.load(args.basis, map_location="cpu", weights_only=True)
    ul = saved["ul"]
    lost = historical["sets"]["lost"]
    if saved["lost"] != lost or ul.shape != (131072, 13) or not torch.isfinite(ul).all():
        raise RuntimeError("historical 13D basis mismatch")
    torch.testing.assert_close(ul.T @ ul, torch.eye(13, dtype=ul.dtype), atol=1e-10, rtol=1e-10)
    window_used, target_used = E.WINDOW_BLOCKS * manager.block, E.TARGET_BLOCKS * manager.block
    mask = _compacted_mask(manager, manager.used - window_used + target_used, "cuda:0")

    def base(selection=False):
        arm = CarrierResidualCompactor(num_layers=manager.num_layers, hidden=manager.hidden,
            block=manager.block, rank=int(cfg.carrier_rank), gamma_max=float(cfg.carrier_gamma_max)
        ).to(device="cuda:0", dtype=torch.float32)
        arm.load_state_dict(state["state_dict"])
        if selection:
            with torch.no_grad():
                arm.up.weight.zero_()
        for x in arm.parameters():
            x.requires_grad_(False)
        return arm

    def candidate(arm):
        # Independent storage, deliberately. The historical path passed a
        # `.data` view of `model.memory` straight into the compactor: the
        # non-reentrant layer checkpoint saved that storage-backed view, then
        # `_pool_in_place` overwrote the window tail the view covered, and the
        # backward recompute read the mutated source. Forward and loss stayed
        # bit-identical either way -- only the gradient silently changed
        # (fp32/13D difference norm 0.2081410736; see
        # runs/s1_checkpoint_recompute_production_cpu.json). The clone breaks the
        # alias; the production CPU audit records the cloned path as the
        # gradient-correct one.
        source = manager._memory[:, manager.total - manager.used:manager.total, :].to("cuda:0").clone()
        weights = manager._window_importance(manager.total - manager.used, window_used)
        return arm(source, weights, target=target_used, block=manager.block,
                   goal_direction=manager.goal_direction, select_weights=manager.select_weights)

    metadata = []
    encoded_probes = {}
    for q in probes:
        stem = q.prompt if q.prompt else f"{q.question}\nAnswer:"
        tokens = manager.tokenizer(f"{stem} {q.answers[0]}", return_tensors="pt", add_special_tokens=False).input_ids
        stem_len = len(manager.tokenizer(stem, add_special_tokens=False).input_ids)
        width = min(256, tokens.shape[1])
        if stem_len <= 0 or stem_len >= width:
            raise RuntimeError("missing answer span: " + q.question)
        encoded_probes[q.question] = (tokens[:, :256].to("cuda:0"), stem_len)
        metadata.append({"question": q.question, "answers": q.answers, "prompt": stem,
                         "input_ids": tokens[0, :256].tolist(), "stem_tokens": stem_len,
                         "untruncated_tokens": tokens.shape[1], "truncated": tokens.shape[1] > 256})

    @torch.no_grad()
    def evaluate(arm=None, which=None):
        target = probes if which is None else which
        correct, ce = {}, {}
        pool, mask_m = (None, None) if arm is None else E.candidate_pool(manager, arm, "cuda:0")
        context = differentiable_memory(model, pool) if pool is not None else nullcontext()
        guard = active_mask_override(model, mask_m) if mask_m is not None else nullcontext()
        with context, guard:
            for q in target:
                input_ids, stem_len = encoded_probes[q.question]
                logits = model(input_ids=input_ids, return_dict=True).logits
                span, labels = logits[:, stem_len - 1:-1, :], input_ids[:, stem_len:]
                loss = F.cross_entropy(span.reshape(-1, span.shape[-1]).float(), labels.reshape(-1))
                ce[q.question] = float(loss)
                if not math.isfinite(ce[q.question]):
                    raise RuntimeError("nonfinite evaluation CE")
                correct[q.question] = bool((span.argmax(dim=-1) == labels).all(dim=-1).item())
                del logits, span, labels, loss
        del pool, mask_m
        torch.cuda.empty_cache()
        return {"correct": correct, "ce": ce}

    report = {"schema_version": 1, "status": "running", "diagnostic_only": True,
              "protocol": {"window": [3, 17890], "steps": args.steps,
                  "measure_every": args.measure_every, "lr": 0.003, "weight_decay": 0.01,
                  "parameter_budget": 1.0, "lambda": 1.0, "seed": 0, "smoke": args.smoke,
                  "original_probes": 60, "expanded_probes": len(probes),
                  "checkpoint_sha256": digest(args.carrier_ckpt), "basis_sha256": digest(args.basis),
                  "historical_result_sha256": digest(args.historical_result),
                  "historical_driver_sha256": historical["protocol"]["driver_sha256"],
                  "gradient_repair_audit_sha256": digest(args.gradient_audit),
                  "driver_sha256": digest(__file__), "torch_version": torch.__version__,
                  "gpu": torch.cuda.get_device_name(0), "score": "teacher-forced answer-token exact"},
              "probes": metadata, "reader_dropout_audit": dropout,
              "reader_config": model.config.to_dict(), "window_audit": {"covered": covered, "dropped": dropped},
              "baselines": {}, "sets": {}, "arms": {}}

    def flush():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False))
        tmp.replace(out)

    flush()
    try:
        for name, arm in (("full", None), ("select", base(True)), ("d0", base())):
            print("[soft] scoring baseline " + name, flush=True)
            report["baselines"][name] = evaluate(arm)
            flush()
        baselines = report["baselines"]
        checks = {"probe_count": len(probes) == 296,
                  "original_provenance": [{"question": q.question, "answers": q.answers} for q in original]
                      == historical["historical_s0_audit"]["original_probes"],
                  "select_correct": baselines["select"]["correct"] == historical["arms"]["select"]["final"]["correct"],
                  "d0_correct": baselines["d0"]["correct"] == historical["checkpoint_diagnostic"]["correct"],
                  "full_original_correct": all(baselines["full"]["correct"][q.question] ==
                      historical["historical_s0_audit"]["full"]["correct"][q.question] for q in original)}
        repeated = evaluate(base(), original)
        checks["d0_repeat"] = repeated["correct"] == {q.question: baselines["d0"]["correct"][q.question] for q in original}
        checks["d0_repeat_ce"] = repeated["ce"] == {q.question: baselines["d0"]["ce"][q.question] for q in original}
        report["baseline_checks"] = checks
        sets = derive_sets(lost, baselines["full"]["correct"], baselines["select"]["correct"], baselines["d0"]["correct"])
        report["sets"] = sets
        checks["cohort_sizes"] = len(sets["protect"]) == 31 and len(sets["repair"]) == 44
        checks["retained_matches"] = sets["retained"] == historical["sets"]["retained_full_and_select"]
        if not all(checks.values()):
            report.update(status="invalid-baseline", verdict="INVALID-BASELINE")
            flush()
            return
        print("[soft] sets " + json.dumps({k: len(v) for k, v in sets.items()}), flush=True)
        report["original_intersections"] = {k: [q for q in lost if q in sets[k]] for k in ("rescue", "protect", "repair")}
        report["baseline_metrics"] = {name: metric_entry(scored, sets, baselines) for name, scored in baselines.items()}
        torch.save({"ul": ul, "original_lost": lost, "source_basis_sha256": digest(args.basis)}, out.with_suffix(".basis.pt"))
        flush()
        for name, protection, lam in (("ul", [], 0.0), ("soft_p", sets["protect"], 1.0),
                                      ("soft_k", sets["retained"], 1.0)):
            torch.manual_seed(0)
            arm = base()
            displacement = make_displacement(arm.down.weight, ul)
            parametrize.register_parametrization(arm.down, "weight", displacement)
            optimizer = torch.optim.AdamW([displacement.value], lr=0.003, weight_decay=0.01)
            initial = evaluate(arm)
            if initial != baselines["d0"]:
                raise RuntimeError("arm identity-at-init failed: " + name)
            history = []
            report["arms"][name] = {"history": history, "trainable_dof": 13,
                                    "protection_set": protection, "lambda": lam,
                                    "final": metric_entry(initial, sets, baselines, protection, lam),
                                    "initial_identity_verified": True}
            flush()
            for step in range(1, args.steps + 1):
                optimizer.zero_grad(set_to_none=True)
                cand = candidate(arm)
                with _pool_in_place(manager, "cuda:0", cand, window_used, target_used) as injection:
                    pool = injection.reinject()
                    # Keep the original order and gradient sum exactly for the control.
                    for q in lost:
                        scalar = margin_scalar(manager, by_q[q], pool, mask, 256)
                        if scalar is None:
                            raise RuntimeError("missing supervised answer span")
                        scalar.backward(retain_graph=True)
                        del scalar
                    for q in protection:
                        scalar = margin_scalar(manager, by_q[q], pool, mask, 256)
                        if scalar is None:
                            raise RuntimeError("missing protected answer span")
                        penalty = soft_regret(scalar, baselines["d0"]["ce"][q]) * (13 * lam / len(protection))
                        penalty.backward(retain_graph=True)
                        del scalar, penalty
                if displacement.value.grad is None or not torch.isfinite(displacement.value.grad).all():
                    raise RuntimeError("missing/nonfinite displacement gradient")
                optimizer.step()
                displacement.clamp()
                del cand, pool
                torch.cuda.empty_cache()
                if step % args.measure_every == 0 or step == args.steps:
                    entry = metric_entry(evaluate(arm), sets, baselines, protection, lam, step,
                                         float(displacement.value.detach().norm()))
                    history.append(entry)
                    report["arms"][name]["final"] = entry
                    flush()
                    print("[soft] %s step=%d original=%d R=%d T=%d P_harm=%d K_harm=%d net=%d penalty=%.6f" %
                          (name, step, entry["original_rescue"], entry["rescue_recovered"],
                           entry["repair_recovered"], entry["protect_harmed"], entry["retained_harmed"],
                           entry["net_vs_select"], entry["penalty"]), flush=True)
            torch.save(arm.state_dict(), out.with_suffix("." + name + ".pt"))
            # The repaired run replaces per-question replication of the historical
            # UL final -- that trajectory was itself produced through the aliased
            # source, so its exact per-question final is a bug artefact, not a
            # control target. The repaired control keeps the historical count
            # invariants (7/6/+8) and is reported, not enforced, before the
            # candidate gate; see experiments/S1-soft-local.md.
            if name == "ul" and not args.smoke:
                final = report["arms"][name]["final"]
                expected = historical["arms"]["subspace_ul"]["final"]
                control_checks = {"original_7": final["original_rescue"] == 7,
                                  "harm_6": final["retained_harmed"] == 6,
                                  "net_8": final["net_vs_select"] == 8,
                                  "per_question_match": final["correct"] == expected["correct"]}
                report["control_checks"] = control_checks
            del arm, displacement, optimizer, initial
            torch.cuda.empty_cache()
        control, candidate_final = (report["arms"][name]["final"] for name in ("ul", "soft_p"))
        report["gate"] = candidate_gate(candidate_final, control)
        report.update(status="complete", verdict="SMOKE-ONLY" if args.smoke else
                      ("LOCAL-PASS" if all(report["gate"].values()) else "NO-IMPROVEMENT"))
        flush()
        print("[soft] verdict=" + report["verdict"], flush=True)
    except Exception as exc:
        report.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
        flush()
        raise


if __name__ == "__main__":
    main()
