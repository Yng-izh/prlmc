"""S1c-basis-wash: rebuilt-clean U_L four-arm comparison on the repaired graph.

Phase 1 of the wash program. One question: does the low-dimensional functional
sensitivity subspace keep its advantage when the 13D basis itself is rebuilt
through the gradient-correct (cloned-source) path, instead of the aliased
`.data`-view path that produced the historical basis?

Arms (all trained on the identical repaired graph, same window, same protocol;
only the basis differs):
  * corrected_ul -- U_L' rebuilt here from cloned-source Jacobian rows (13 DOF)
  * old_ul       -- the historical basis `runs/s1_local_protected.basis.pt` (13 DOF)
  * random_13d   -- a seed-0 random orthonormal 13D basis (13 DOF)
  * unconstrained-- no basis, full down.weight displacement (131072 DOF)

The historical 13-question supervision set, cohort definitions, metrics and
count invariants are imported verbatim from the s1_soft_local driver so the
numbers stay comparable. The provenance chain cites the gradient repair audit.
"""
import argparse
import json
import math
import os
from pathlib import Path

from s1_local_protected import digest, disable_reader_dropout, orth
from s1_soft_local import derive_sets, metric_entry


WASH_ARMS = ("corrected_ul", "old_ul", "random_13d", "unconstrained")
FORMAL_BASIS_SHA256 = "3af25e44b94464432a0221e6dabe8685ef2bd91c522addcbe279387425bb97b2"


def random_basis(flat_dim, rank, seed, torch):
    """A reproducible random orthonormal basis, flat_dim x rank."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = torch.randn(rank, flat_dim, generator=generator, dtype=torch.float64)
    return orth(rows)


def principal_angles(a, b, torch):
    """Principal angles (radians) between two flat_dim x rank orthonormal sets."""
    q = a.T.double() @ b.double()
    s = torch.linalg.svdvals(q)
    return torch.clamp(s, -1.0, 1.0).acos()


def wash_gate(finals):
    """Pre-registered continue/stop gate over the four finals.

    PASS: corrected keeps a clear positive net with near-zero protect harm and
    clearly beats random_13d. FAIL: the subspace advantage was alias artefact.
    MARGIN: the random gap lands in the 3..5 band -> random re-draws decide.
    """
    corrected, random = finals["corrected_ul"], finals["random_13d"]
    positive = corrected["protect_harmed"] <= 1 and corrected["net_vs_select"] >= 8
    rescue_gap = corrected["rescue_recovered"] - random["rescue_recovered"]
    net_gap = corrected["net_vs_select"] - random["net_vs_select"]
    if positive and rescue_gap >= 5 and net_gap >= 5:
        return "PASS"
    if corrected["net_vs_select"] < 8 or (rescue_gap < 3 and net_gap < 3):
        return "FAIL"
    return "MARGIN"


def margin_gate(corrected, random_draws):
    """MARGIN resolution: corrected must beat the best random draw by >= 3."""
    best_rescue = max(d["rescue_recovered"] for d in random_draws)
    best_net = max(d["net_vs_select"] for d in random_draws)
    rescue_gap = corrected["rescue_recovered"] - best_rescue
    net_gap = corrected["net_vs_select"] - best_net
    return {"rescue_gap": rescue_gap, "net_gap": net_gap,
            "resolved_pass": rescue_gap >= 3 and net_gap >= 3}


def make_arm_displacement(weight, basis, torch):
    """Bounded displacement on down.weight: `basis @ value` or full-rank.

    Same contract as s1_soft_local.make_displacement for the 13D arms; for the
    unconstrained arm the value has down.weight's own shape and the identity
    at init still holds because the value starts at zero.
    """

    class Displacement(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if basis is None:
                self.value = torch.nn.Parameter(weight.new_zeros(weight.shape))
            else:
                self.register_buffer("basis", basis.to(weight).contiguous())
                self.value = torch.nn.Parameter(weight.new_zeros(basis.shape[1]))

        def forward(self, original):
            delta = self.value if basis is None else (self.basis @ self.value).reshape_as(original)
            return original + delta

        def clamp(self, budget=1.0):
            with torch.no_grad():
                self.value.mul_((budget / self.value.norm().clamp_min(1e-12)).clamp_max(1))

    return Displacement()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("carrier-ckpt", "old-basis", "historical-result",
                 "gradient-audit", "out"):
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
    if out.exists() or any(out.with_suffix("." + name + ".pt").exists() for name in WASH_ARMS):
        raise SystemExit("output exists")
    if args.steps != expected_steps or args.measure_every != expected_cadence:
        raise SystemExit("protocol requires smoke 2/1 or formal 60/5")
    historical = json.loads(Path(args.historical_result).read_text())
    if historical.get("status") != "complete" or historical["protocol"]["smoke"]:
        raise SystemExit("historical result must be formal and complete")
    gradient_audit = json.loads(Path(args.gradient_audit).read_text())
    audit_verdict = {key: bool(value) for key, value in gradient_audit["verdict"].items()}
    if not all(audit_verdict.values()):
        raise SystemExit("gradient repair audit verdict failed: " + json.dumps(audit_verdict))
    if digest(args.old_basis) != FORMAL_BASIS_SHA256:
        raise SystemExit("historical basis hash mismatch")
    if digest(args.carrier_ckpt) != historical["protocol"]["checkpoint_sha256"]:
        raise SystemExit("source checkpoint hash mismatch")
    # `historical_result` is the s1_soft_local report: its driver_sha256 pins
    # the pre-patch s1_soft_local driver, not s1_local_protected.py. The
    # aliased-source basis driver stays pinned through
    # historical_driver_sha256 == digest(s1_local_protected.py), checked below.
    if historical["protocol"]["historical_driver_sha256"] != digest(
            Path(__file__).with_name("s1_local_protected.py")):
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
    config.update(device="cuda:0", dtype="float16", stats_dir="./outputs/prlmc-s1c")
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
    saved = torch.load(args.old_basis, map_location="cpu", weights_only=True)
    old_ul = saved["ul"]
    # s1_soft_local.json's sets use the expanded cohort vocabulary; the
    # 13-question supervision set is `original_lost`, identical to the
    # s1_local_protected basis file's `lost`.
    lost = historical["sets"]["original_lost"]
    if saved["lost"] != lost or old_ul.shape != (131072, 13) or not torch.isfinite(old_ul).all():
        raise RuntimeError("historical 13D basis mismatch")
    torch.testing.assert_close(old_ul.T @ old_ul, torch.eye(13, dtype=old_ul.dtype), atol=1e-10, rtol=1e-10)
    window_used, target_used = E.WINDOW_BLOCKS * manager.block, E.TARGET_BLOCKS * manager.block
    mask = _compacted_mask(manager, manager.used - window_used + target_used, "cuda:0")

    def base(selection=False):
        arm = CarrierResidualCompactor(num_layers=manager.num_layers, hidden=manager.hidden,
            block=manager.block, rank=int(cfg.carrier_rank), gamma_max=float(cfg.carrier_gamma_max)
        ).to(device="cuda:0", dtype=torch.float32)
        arm.load_state_dict(state["state_dict"])
        if selection:
            # `select` baseline: zero the up projection so the compactor only
            # re-selects carrier slots and never rewrites hidden content.
            with torch.no_grad():
                arm.up.weight.zero_()
        for x in arm.parameters():
            x.requires_grad_(False)
        return arm

    def candidate(arm):
        # The repaired gradient path: the source slice is cloned so the
        # non-reentrant layer checkpoint never saves an aliased view over the
        # pool tail that `_pool_in_place` then overwrites. Proven equivalent to
        # the direct gradient in runs/s1_checkpoint_recompute_production_cpu.json.
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
    def evaluate(arm=None):
        correct, ce = {}, {}
        pool, mask_m = (None, None) if arm is None else E.candidate_pool(manager, arm, "cuda:0")
        context = differentiable_memory(model, pool) if pool is not None else nullcontext()
        guard = active_mask_override(model, mask_m) if mask_m is not None else nullcontext()
        with context, guard:
            for q in probes:
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
                  "parameter_budget": 1.0, "seed": 0, "smoke": args.smoke,
                  "original_probes": 60, "expanded_probes": len(probes),
                  "checkpoint_sha256": digest(args.carrier_ckpt),
                  "old_basis_sha256": digest(args.old_basis),
                  "historical_result_sha256": digest(args.historical_result),
                  "historical_driver_sha256": historical["protocol"]["driver_sha256"],
                  "gradient_repair_audit_sha256": digest(args.gradient_audit),
                  "driver_sha256": digest(__file__), "torch_version": torch.__version__,
                  "gpu": torch.cuda.get_device_name(0), "score": "teacher-forced answer-token exact",
                  "arms": list(WASH_ARMS), "random_seed": 0},
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
            print("[wash] scoring baseline " + name, flush=True)
            report["baselines"][name] = evaluate(arm)
            flush()
        baselines = report["baselines"]
        # All baseline cross-checks compare directly against the s1_soft_local
        # report's own baselines, which already reproduce the S0 provenance
        # audit upstream; that report's baseline_checks were all true.
        soft_baselines = historical["baselines"]
        checks = {"probe_count": len(probes) == 296,
                  "select_correct": baselines["select"]["correct"] == soft_baselines["select"]["correct"],
                  "d0_correct": baselines["d0"]["correct"] == soft_baselines["d0"]["correct"],
                  "full_correct": baselines["full"]["correct"] == soft_baselines["full"]["correct"]}
        repeated = evaluate(base())
        checks["d0_repeat"] = repeated["correct"] == baselines["d0"]["correct"]
        checks["d0_repeat_ce"] = repeated["ce"] == baselines["d0"]["ce"]
        report["baseline_checks"] = checks
        sets = derive_sets(lost, baselines["full"]["correct"], baselines["select"]["correct"], baselines["d0"]["correct"])
        report["sets"] = sets
        checks["cohort_sizes"] = len(sets["protect"]) == 31 and len(sets["repair"]) == 44
        checks["retained_matches"] = sets["retained"] == historical["sets"]["retained"]
        if not all(checks.values()):
            report.update(status="invalid-baseline", verdict="INVALID-BASELINE")
            flush()
            return
        print("[wash] sets " + json.dumps({k: len(v) for k, v in sets.items()}), flush=True)
        report["original_intersections"] = {k: [q for q in lost if q in sets[k]] for k in ("rescue", "protect", "repair")}
        report["baseline_metrics"] = {name: metric_entry(scored, sets, baselines) for name, scored in baselines.items()}
        flush()

        # --- corrected basis: cloned-source Jacobian rows over the lost set ---
        print("[wash] rebuilding corrected 13D basis (cloned source)", flush=True)
        jacobian_arm = base()
        jacobian_arm.down.weight.requires_grad_(True)
        rows = {}
        for q in lost:
            jacobian_arm.down.weight.grad = None
            cand = candidate(jacobian_arm)
            with _pool_in_place(manager, "cuda:0", cand, window_used, target_used) as injection:
                pool = injection.reinject()
                scalar = margin_scalar(manager, by_q[q], pool, mask, 256)
            if scalar is None:
                raise RuntimeError("missing margin: " + q)
            scalar.backward()
            rows[q] = jacobian_arm.down.weight.grad.detach().flatten().cpu().clone()
            del scalar, cand, pool
            jacobian_arm.down.weight.grad = None
            torch.cuda.empty_cache()
            print("[wash] Jacobian %d/%d" % (len(rows), len(lost)), flush=True)
        jacobian_arm.down.weight.requires_grad_(False)
        del jacobian_arm
        corrected_ul = orth(torch.stack([rows[q] for q in lost]))
        if corrected_ul.shape[0] != 131072:
            raise RuntimeError("corrected basis dimension mismatch")
        random_ul = random_basis(131072, corrected_ul.shape[1], 0, torch)
        angles = principal_angles(corrected_ul, old_ul, torch)
        report["basis_audit"] = {
            "corrected_rank": int(corrected_ul.shape[1]), "old_rank": int(old_ul.shape[1]),
            "random_seed": 0,
            "principal_angles_to_old_rad": [float(a) for a in angles],
            "max_abs_cosine_to_old": float((corrected_ul.T @ old_ul).abs().max()),
            "overlap_norm": float((corrected_ul.T @ old_ul).norm()),
            "old_projected_energy_fraction": float(
                (old_ul.T @ corrected_ul).pow(2).sum() / old_ul.pow(2).sum())}
        torch.save({"lost": lost, "rows": rows, "corrected_ul": corrected_ul,
                    "old_ul": old_ul, "random_ul": random_ul,
                    "source_old_basis_sha256": digest(args.old_basis)},
                   out.with_suffix(".basis.pt"))
        flush()

        arms_plan = (("corrected_ul", corrected_ul), ("old_ul", old_ul),
                     ("random_13d", random_ul), ("unconstrained", None))
        for name, basis in arms_plan:
            torch.manual_seed(0)
            arm = base()
            displacement = make_arm_displacement(arm.down.weight, basis, torch)
            parametrize.register_parametrization(arm.down, "weight", displacement)
            optimizer = torch.optim.AdamW([displacement.value], lr=0.003, weight_decay=0.01)
            initial = evaluate(arm)
            if initial != baselines["d0"]:
                raise RuntimeError("arm identity-at-init failed: " + name)
            history = []
            report["arms"][name] = {"history": history,
                                    "trainable_dof": int(displacement.value.numel()),
                                    "basis_kind": name, "final": metric_entry(initial, sets, baselines),
                                    "initial_identity_verified": True}
            flush()
            for step in range(1, args.steps + 1):
                optimizer.zero_grad(set_to_none=True)
                cand = candidate(arm)
                with _pool_in_place(manager, "cuda:0", cand, window_used, target_used) as injection:
                    pool = injection.reinject()
                    for q in lost:
                        scalar = margin_scalar(manager, by_q[q], pool, mask, 256)
                        if scalar is None:
                            raise RuntimeError("missing supervised answer span")
                        scalar.backward(retain_graph=True)
                        del scalar
                if displacement.value.grad is None or not torch.isfinite(displacement.value.grad).all():
                    raise RuntimeError("missing/nonfinite displacement gradient")
                optimizer.step()
                displacement.clamp()
                del cand, pool
                torch.cuda.empty_cache()
                if step % args.measure_every == 0 or step == args.steps:
                    entry = metric_entry(evaluate(arm), sets, baselines, step=step,
                                         deviation=float(displacement.value.detach().norm()))
                    history.append(entry)
                    report["arms"][name]["final"] = entry
                    flush()
                    print("[wash] %s step=%d original=%d R=%d T=%d P_harm=%d K_harm=%d net=%d" %
                          (name, step, entry["original_rescue"], entry["rescue_recovered"],
                           entry["repair_recovered"], entry["protect_harmed"], entry["retained_harmed"],
                           entry["net_vs_select"]), flush=True)
            torch.save(arm.state_dict(), out.with_suffix("." + name + ".pt"))
            del arm, displacement, optimizer, initial
            torch.cuda.empty_cache()
        finals = {name: report["arms"][name]["final"] for name in WASH_ARMS}
        report["gate"] = wash_gate(finals)
        verdict = "SMOKE-ONLY" if args.smoke else report["gate"]
        report.update(status="complete", verdict=verdict)
        flush()
        print("[wash] verdict=" + verdict, flush=True)
    except Exception as exc:
        report.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
        flush()
        raise


if __name__ == "__main__":
    main()
