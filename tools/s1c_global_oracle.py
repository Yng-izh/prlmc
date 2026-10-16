"""S1c-global-oracle: cross-window U_global subspace, held-out four-arm test.

Phase 3, layer 1. The S1c wash proved a *single-window* 13D basis is
functional; the open question is whether that subspace is a property of the
compactor's parameter space rather than one window's data. Layer 1 answers it
with an oracle diagnostic: concatenate rescue-cohort Jacobian rows from all 8
training windows, take the leading 13 left singular vectors -> U_global, then
train four arms on the 2 held-out windows (row 2, different fact list, never
seen in U_global construction).

Arms per held-out window (identical protocol to s1c_wash):
  * global_13d     -- the frozen cross-window U_global (13 DOF)
  * random_13d     -- seed-0 random orthonormal basis (13 DOF)
  * unconstrained  -- full down.weight displacement (131072 DOF)
  * local_13d      -- the window's own rebuilt basis (13 DOF), the per-window
                      ceiling reference

Per-window supervision: that window's rescue cohort (full-correct ∧
select-wrong questions), the same set definition as s1c_wash. The oracle
diagnostic allows gold answers for arm selection via the alpha budget only in
the sense that training is supervised; no test-time adaptation happens.

Pre-registered gate (over the two held-out windows, summed):
  PASS: net >= +3 AND beats random_13d by >= 3 AND >= unconstrained
  FAIL: stop the line (functional subspace exits the main method)
  MARGIN (1..2): extend to secondary held-out windows (1,1930) (1,1550)

The U_global construction itself is train-only: held-out rows never enter the
SVD. Rows are cached so the GPU stage can resume.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path

from s1_local_protected import digest, disable_reader_dropout, orth
from s1c_wash import make_arm_displacement, random_basis


def window_sets(full, select_correct, d0):
    """Cohort derivation for an arbitrary window, no historical original.

    Same definitions as s1_soft_local.derive_sets minus the 13-question
    historical supervision: rescue = full-correct and select-wrong,
    retained = full-correct and select-correct, protect/repair split by d0.
    """
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


def window_metric(correct, sets, step=0, deviation=0.0):
    """metric_entry minus the CE and penalty fields (evaluation is EM-only)."""
    entry = {"step": step, "correct": correct, "deviation": deviation}
    groups = [("original_rescue", sets["original_lost"], True),
              ("rescue_recovered", sets["rescue"], True),
              ("rescue_d0_harmed", sets["rescue_d0_correct"], False),
              ("repair_recovered", sets["repair"], True),
              ("protect_harmed", sets["protect"], False),
              ("retained_harmed", sets["retained"], False)]
    for count, group, wanted in groups:
        found = [q for q in group if correct[q] == wanted]
        entry[count] = len(found)
        entry[count + "_questions"] = found
    entry["accuracy"] = sum(correct.values()) / len(correct)
    entry["net_vs_select"] = entry["rescue_recovered"] - entry["retained_harmed"]
    return entry
from s1c_wash import make_arm_displacement, random_basis


TRAIN_WINDOWS = [(3, 17890), (3, 17130), (3, 16750), (3, 14470),
                 (3, 17510), (3, 14850), (3, 16370), (3, 15230)]
HELD_WINDOWS = [(2, 3450), (2, 3830)]
SECONDARY_WINDOWS = [(1, 1930), (1, 1550)]
GLOBAL_RANK = 13
ORACLE_ARMS = ("global_13d", "random_13d", "unconstrained", "local_13d")


def global_gate(window_results):
    """Pre-registered gate over summed held-out finals."""
    totals = {arm: {"rescue_recovered": 0, "retained_harmed": 0, "net_vs_select": 0}
              for arm in ORACLE_ARMS}
    for result in window_results:
        for arm in ORACLE_ARMS:
            final = result["arms"][arm]["final"]
            totals[arm]["rescue_recovered"] += final["rescue_recovered"]
            totals[arm]["retained_harmed"] += final["retained_harmed"]
            totals[arm]["net_vs_select"] += final["net_vs_select"]
    net = totals["global_13d"]["net_vs_select"]
    net_random = totals["random_13d"]["net_vs_select"]
    net_unc = totals["unconstrained"]["net_vs_select"]
    rescue_gap = totals["global_13d"]["rescue_recovered"] - totals["random_13d"]["rescue_recovered"]
    net_gap = net - net_random
    if net >= 3 and rescue_gap >= 3 and net_gap >= 3 and net >= net_unc:
        verdict = "PASS"
    elif net < 0 or (rescue_gap <= 0 and net_gap <= 0) or net < net_unc - 2:
        verdict = "FAIL"
    else:
        # Boundary on one margin (gap 1..2, or unconstrained up to 2 above):
        # extend to the secondary held-out windows before ruling.
        verdict = "MARGIN"
    return {"totals": totals, "net": net, "net_random": net_random,
            "net_unconstrained": net_unc, "rescue_gap": rescue_gap,
            "net_gap": net_gap, "verdict": verdict}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("carrier-ckpt", "wash-basis", "wash-report",
                 "gradient-audit", "out", "rows-out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--cuda-device", type=int, default=9)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--measure-every", type=int, default=5)
    p.add_argument("--include-secondary", action="store_true",
                   help="also run the secondary held-out windows (boundary case)")
    args = p.parse_args()
    out = Path(args.out)
    rows_out = Path(args.rows_out)
    resume = None
    if out.exists():
        # Crash resume: keep already-finished windows, re-run only the rest.
        # A window counts as finished when all four arms reached step==steps.
        prior = json.loads(out.read_text())
        finished = {tuple(w["window"]) for w in prior.get("windows", [])
                    if all(a.get("history") and a["history"][-1]["step"] >= prior["protocol"]["steps"]
                           for a in w["arms"].values())}
        if prior.get("status") == "complete":
            raise SystemExit("output already complete")
        if finished:
            resume = (prior, finished)
            print("[global] resuming, finished windows: %s"
                  % sorted(finished), flush=True)
    wash = json.loads(Path(args.wash_report).read_text())
    if wash.get("status") != "complete" or wash.get("gate") != "PASS":
        raise SystemExit("wash gate must be PASS before the global stage")
    gradient_audit = json.loads(Path(args.gradient_audit).read_text())
    if not all(bool(v) for v in gradient_audit["verdict"].values()):
        raise SystemExit("gradient repair audit verdict failed")
    windows = list(HELD_WINDOWS) + (list(SECONDARY_WINDOWS) if args.include_secondary else [])
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
    config.update(device="cuda:0", dtype="float16", stats_dir="./outputs/prlmc-s1c-global")
    cfg = PRLMCConfig.from_dict(config)
    model = MemoryLLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                      attn_implementation="sdpa").to("cuda:0")
    model.initialized.zero_()
    for x in model.parameters():
        x.requires_grad_(False)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    manager = ProtectedResidualMemory(model, AutoTokenizer.from_pretrained(MODEL), cfg)
    # Reader stays in train() mode (checkpointing needs it) but dropout must be
    # off: s1c_wash disables it and without that call re-evaluating the same
    # pool flips answers, which broke identity-at-init on held-out windows.
    disable_reader_dropout(model)
    args.probes, args.max_probe_tokens = 60, 256
    state = torch.load(args.carrier_ckpt, map_location="cuda:0", weights_only=True)
    if digest(args.carrier_ckpt) != wash["protocol"]["checkpoint_sha256"]:
        raise SystemExit("source checkpoint hash mismatch")
    wash_basis = torch.load(args.wash_basis, map_location="cpu", weights_only=True)
    if wash_basis["source_old_basis_sha256"] != wash["protocol"]["old_basis_sha256"]:
        raise SystemExit("wash basis file mismatch")
    window_used, target_used = E.WINDOW_BLOCKS * manager.block, E.TARGET_BLOCKS * manager.block
    args_probe = argparse.Namespace(probes=60, max_probe_tokens=256)

    def fresh_window(row, start):
        fresh_window.cache.setdefault(row, E.read_fact_lines(row))
        lines = fresh_window.cache[row]
        built = build_window_and_probes(manager, fresh_window.cache, row, start, args_probe)
        if built is None:
            raise RuntimeError("window build failed (%d,%d)" % (row, start))
        covered, indices, dropped, original = built
        by_q = {q.question: q for q in original}
        by_q.update({q.question: q for q in E.probes_from_lines(
            lines, indices, len(indices) + 1)})
        return covered, indices, dropped, original, by_q, list(by_q.values())

    fresh_window.cache = {}

    def base():
        arm = CarrierResidualCompactor(num_layers=manager.num_layers, hidden=manager.hidden,
            block=manager.block, rank=int(cfg.carrier_rank), gamma_max=float(cfg.carrier_gamma_max)
        ).to(device="cuda:0", dtype=torch.float32)
        arm.load_state_dict(state["state_dict"])
        for x in arm.parameters():
            x.requires_grad_(False)
        return arm

    def candidate(arm):
        source = manager._memory[:, manager.total - manager.used:manager.total, :].to("cuda:0").clone()
        weights = manager._window_importance(manager.total - manager.used, window_used)
        return arm(source, weights, target=target_used, block=manager.block,
                   goal_direction=manager.goal_direction, select_weights=manager.select_weights)

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
        with context, guard:
            for q in probes:
                input_ids, stem_len = encoded[q.question]
                logits = model(input_ids=input_ids, return_dict=True).logits
                span = logits[:, stem_len - 1:-1, :]
                labels = input_ids[:, stem_len:]
                correct[q.question] = bool((span.argmax(dim=-1) == labels).all(dim=-1).item())
                del logits, span, labels
        del pool, mask_m
        torch.cuda.empty_cache()
        return correct

    def jacobian_row(by_q, question, arm):
        arm.down.weight.requires_grad_(True)
        arm.down.weight.grad = None
        cand = candidate(arm)
        with _pool_in_place(manager, "cuda:0", cand, window_used, target_used) as injection:
            pool = injection.reinject()
            scalar = margin_scalar(manager, by_q[question], pool, mask, 256)
        if scalar is None:
            raise RuntimeError("missing margin: " + question)
        scalar.backward()
        row = arm.down.weight.grad.detach().flatten().cpu().clone()
        arm.down.weight.grad = None
        arm.down.weight.requires_grad_(False)
        del scalar, cand, pool
        torch.cuda.empty_cache()
        return row

    # --- stage 1: rescue rows for every training window -> U_global ---
    rows_by_window = {}
    if rows_out.exists():
        cached = torch.load(rows_out, map_location="cpu", weights_only=True)
        rows_by_window = cached["rows_by_window"]
        print("[global] loaded %d cached train windows" % len(rows_by_window), flush=True)
    mask = None
    if not rows_by_window:
        for row, start in TRAIN_WINDOWS:
            covered, indices, dropped, original, by_q, probes = fresh_window(row, start)
            mask = _compacted_mask(manager, manager.used - window_used + target_used, "cuda:0")
            encoded = encode(probes)
            full = evaluate(encoded, probes, None)
            select_arm = base()
            with torch.no_grad():
                select_arm.up.weight.zero_()
            select_correct = evaluate(encoded, probes, select_arm)
            d0_correct = evaluate(encoded, probes, base())
            # train-window supervision: rescue = full ∧ ¬select ∧ ¬d0 (same
            # definition as the wash's `rescue_d0_wrong`, i.e. the questions the
            # trained operator could actually recover)
            sets = window_sets(full, select_correct, d0_correct)
            rescue = sets["rescue_d0_wrong"]
            print("[global] train window (%d,%d) rescue=%d" % (row, start, len(rescue)), flush=True)
            arm = base()
            window_rows = {}
            for i, q in enumerate(rescue, 1):
                window_rows[q] = jacobian_row(by_q, q, arm)
                print("[global] row %d/%d" % (i, len(rescue)), flush=True)
            rows_by_window["%d,%d" % (row, start)] = window_rows
            torch.save({"rows_by_window": rows_by_window,
                        "checkpoint_sha256": digest(args.carrier_ckpt)},
                       rows_out)
        del arm
        torch.cuda.empty_cache()
    all_rows = []
    for key in rows_by_window:
        all_rows.extend(rows_by_window[key].values())
    stacked = torch.stack(all_rows).double()
    u, s, _ = torch.linalg.svd(stacked.T, full_matrices=False)
    u_global = u[:, :GLOBAL_RANK].contiguous()
    energy = float((s[:GLOBAL_RANK] ** 2).sum() / (s ** 2).sum())
    print("[global] U_global rank=%d from %d rows, energy=%.4f"
          % (GLOBAL_RANK, len(all_rows), energy), flush=True)
    torch.save({"u_global": u_global, "singular_values": s.tolist(),
                "energy_fraction": energy, "n_rows": len(all_rows),
                "source_checkpoint_sha256": digest(args.carrier_ckpt)},
               out.with_suffix(".uglobal.pt"))

    # --- stage 2: held-out windows, four arms each ---
    report = {"schema_version": 1, "status": "running",
              "diagnostic_only": True, "oracle_alpha": True,
              "protocol": {"train_windows": [list(w) for w in TRAIN_WINDOWS],
                           "held_windows": [list(w) for w in windows],
                           "global_rank": GLOBAL_RANK,
                           "steps": args.steps, "measure_every": args.measure_every,
                           "lr": 0.003, "weight_decay": 0.01,
                           "parameter_budget": 1.0, "seed": 0,
                           "checkpoint_sha256": digest(args.carrier_ckpt),
                           "wash_basis_sha256": digest(args.wash_basis),
                           "wash_report_sha256": digest(args.wash_report),
                           "gradient_repair_audit_sha256": digest(args.gradient_audit),
                           "driver_sha256": digest(__file__),
                            "resumed_from_driver_sha256": (
                                resume[0]["protocol"]["driver_sha256"]
                                if resume is not None else None),
                           "torch_version": torch.__version__,
                           "gpu": torch.cuda.get_device_name(0),
                           "arms": list(ORACLE_ARMS)},
              "u_global_audit": {"n_rows": len(all_rows),
                                 "energy_fraction": energy,
                                 "top_singular_values": s[:13].tolist()},
              "windows": []}

    def flush():
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False))
        tmp.replace(out)

    flush()
    try:
        for row, start in windows:
            if resume is not None and (row, start) in resume[1]:
                report["windows"].append(next(
                    w for w in resume[0]["windows"] if w["window"] == [row, start]))
                flush()
                print("[global] (%d,%d) restored from checkpoint" % (row, start),
                      flush=True)
                continue
            covered, indices, dropped, original, by_q, probes = fresh_window(row, start)
            mask = _compacted_mask(manager, manager.used - window_used + target_used, "cuda:0")
            encoded = encode(probes)
            full = evaluate(encoded, probes, None)
            select_arm = base()
            with torch.no_grad():
                select_arm.up.weight.zero_()
            select_correct = evaluate(encoded, probes, select_arm)
            d0 = evaluate(encoded, probes, base())
            sets = window_sets(full, select_correct, d0)
            rescue = sets["rescue_d0_wrong"]
            print("[global] held window (%d,%d) rescue=%d retained=%d"
                  % (row, start, len(rescue), len(sets["retained"])), flush=True)
            baselines = {"full": {"correct": full},
                         "select": {"correct": select_correct},
                         "d0": {"correct": d0}}
            # local basis: this window's own rescue rows
            local_arm = base()
            local_rows = [jacobian_row(by_q, q, local_arm) for q in rescue]
            local_ul = orth(torch.stack(local_rows)) if local_rows else None
            del local_arm
            gc.collect()
            torch.cuda.empty_cache()
            random_ul = random_basis(131072, GLOBAL_RANK, 0, torch)
            window_report = {"window": [row, start],
                             "sets": {k: sets[k] for k in
                                      ("rescue", "protect", "repair", "retained")},
                             "arms": {}}
            arms_plan = (("global_13d", u_global), ("random_13d", random_ul),
                         ("unconstrained", None), ("local_13d", local_ul))
            for name, basis in arms_plan:
                torch.manual_seed(0)
                arm = base()
                displacement = make_arm_displacement(arm.down.weight, basis, torch)
                if basis is None:
                    displacement = make_arm_displacement(arm.down.weight, None, torch)
                parametrize.register_parametrization(arm.down, "weight", displacement)
                optimizer = torch.optim.AdamW([displacement.value], lr=0.003,
                                              weight_decay=0.01)
                initial = evaluate(encoded, probes, arm)
                if initial != d0:
                    raise RuntimeError("identity-at-init failed: %s/%d,%d"
                                       % (name, row, start))
                history = []
                window_report["arms"][name] = {
                    "history": history,
                    "trainable_dof": int(displacement.value.numel()),
                    "final": window_metric(initial, sets),
                    "initial_identity_verified": True}
                for step in range(1, args.steps + 1):
                    optimizer.zero_grad(set_to_none=True)
                    cand = candidate(arm)
                    with _pool_in_place(manager, "cuda:0", cand, window_used,
                                        target_used) as injection:
                        pool = injection.reinject()
                        for q in rescue:
                            scalar = margin_scalar(manager, by_q[q], pool, mask, 256)
                            if scalar is None:
                                raise RuntimeError("missing supervised answer span")
                            scalar.backward(retain_graph=True)
                            del scalar
                    optimizer.step()
                    displacement.clamp()
                    del cand, pool
                    torch.cuda.empty_cache()
                    if step % args.measure_every == 0 or step == args.steps:
                        entry = window_metric(evaluate(encoded, probes, arm), sets,
                                               step=step,
                                               deviation=float(displacement.value.detach().norm()))
                        history.append(entry)
                        window_report["arms"][name]["final"] = entry
                        report["windows"] = [w for w in report["windows"]
                                             if w["window"] != [row, start]]
                        report["windows"].append(window_report)
                        flush()
                        print("[global] (%d,%d) %s step=%d R=%d K_harm=%d net=%d"
                              % (row, start, name, step, entry["rescue_recovered"],
                                 entry["retained_harmed"], entry["net_vs_select"]),
                              flush=True)
                del arm, displacement, optimizer
                # Parametrize keeps a reference cycle (arm -> parametrization ->
                # weight -> ...); plain del leaves ~3 GiB on the card, which OOM'd
                # the second window's Jacobian stage. Force the cycle's collection.
                gc.collect()
                torch.cuda.empty_cache()
            report["windows"] = [w for w in report["windows"]
                                 if w["window"] != [row, start]]
            report["windows"].append(window_report)
            gc.collect()
            torch.cuda.empty_cache()
            flush()
        gate = global_gate(report["windows"][:len(windows)])
        report["gate"] = gate
        report.update(status="complete", verdict=gate["verdict"])
        flush()
        print("[global] verdict=" + gate["verdict"], flush=True)
    except Exception as exc:
        report.update(status="failed", failure={"type": type(exc).__name__,
                                                "message": str(exc)})
        flush()
        raise


if __name__ == "__main__":
    main()
