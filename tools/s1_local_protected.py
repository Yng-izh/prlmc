"""S1-local: five arms on S0 window; expanded full-and-select retained probes.
Supervised mechanism diagnostic only, not a held-out claim.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def orth(rows):
    import torch
    rows = rows.detach().cpu().double()
    if not torch.isfinite(rows).all():
        raise ValueError("non-finite rows")
    if not rows.shape[0]:
        return rows.T
    _, s, vh = torch.linalg.svd(rows, full_matrices=False)
    return vh[s > max(1e-10, float(s.max()) * 1e-7)].T.contiguous()


def protect(ul, uk):
    projected = ul - uk @ (uk.T @ ul)
    return orth(projected.T), projected.norm(dim=0)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def s0_primary_lost(historical, deterministic):
    if len(historical) != 13 or len(set(historical)) != 13:
        raise ValueError("historical S0 fixture must contain exactly 13 distinct lost probes")
    missing = sorted(set(historical) - set(deterministic))
    if missing:
        raise ValueError("historical S0 targets are no longer lost: " + json.dumps(missing))
    return list(historical)


def disable_reader_dropout(model):
    import torch
    changed = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            changed.append({"name": name, "p": module.p, "was_training": module.training})
            module.eval()
    return changed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("carrier-ckpt", "s0-rows", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--measure-every", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--budget", type=float, default=1.0)
    p.add_argument("--cuda-device", type=int, default=9)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if Path(args.out).exists():
        raise SystemExit("output exists")
    if args.steps < 1 or args.measure_every < 1 or args.budget <= 0:
        raise SystemExit("invalid training settings")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    os.environ["S0B_GPU_SET"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    import yaml
    import e6a_overfit as E
    from s0_functional_sensitivity import build_window_and_probes, margin_scalar, CONFIG, MODEL
    from s0b_coefficient_oracle import OracleCompactor
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.consolidate import CarrierResidualCompactor, FixedBasisCarrierResidualCompactor
    from methods.prlmc.functional_subspace import CoefficientOracle
    from methods.prlmc.layout import ProtectedResidualMemory
    from methods.prlmc.probes import _pool_in_place, _compacted_mask
    from torch.nn.utils import parametrize
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from transformers import AutoTokenizer

    torch.manual_seed(0)
    torch.set_num_threads(4)
    with open(CONFIG) as f:
        config = dict(yaml.safe_load(f)["prlmc"])
    config.update(device="cuda:0", dtype="float16", stats_dir="./outputs/prlmc-s1")
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
        raise SystemExit("window build failed")
    covered, indices, dropped, original = built
    # Preserve S0's seeded window writes; deterministic reader evaluation starts
    # here. Parent train mode remains on solely for gradient checkpointing.
    if float(getattr(model.config, "attention_dropout", 0.0)) != 0.0:
        raise RuntimeError("nonzero attention dropout needs explicit deterministic dispatch")
    dropout_audit = disable_reader_dropout(model)
    print("[s1] disabled %d reader dropout modules; active p>0 before disable=%d" %
          (len(dropout_audit), sum(x["was_training"] and x["p"] > 0 for x in dropout_audit)), flush=True)
    expanded = E.probes_from_lines(lines, indices, len(indices) + 1)
    by_q = {q.question: q for q in original}
    by_q.update({q.question: q for q in expanded})
    probes = list(by_q.values())
    state = torch.load(args.carrier_ckpt, map_location="cuda:0", weights_only=True)

    def base(selection=False, fixed=False):
        arm = CarrierResidualCompactor(num_layers=manager.num_layers, hidden=manager.hidden,
            block=manager.block, rank=int(cfg.carrier_rank), gamma_max=float(cfg.carrier_gamma_max)
        ).to(device="cuda:0", dtype=torch.float32)
        arm.load_state_dict(state["state_dict"])
        if selection:
            with torch.no_grad():
                arm.up.weight.zero_()
        if fixed:
            arm = FixedBasisCarrierResidualCompactor.from_warmup(arm)
        for x in arm.parameters():
            x.requires_grad_(False)
        return arm

    def score(arm, which=None):
        with torch.no_grad():
            pool, mask_m = E.candidate_pool(manager, arm, "cuda:0")
            result = E.score_probes(manager, probes if which is None else which, pool, 256, mask=mask_m)
        del pool, mask_m
        torch.cuda.empty_cache()
        return result

    print("[s1] reproducing S0 no-memory on 60 original probes", flush=True)
    no_mem = E.score_no_memory(manager, original, 256)
    print("[s1] reproducing S0 full on 60 original probes", flush=True)
    full = E.score_uncompacted(manager, original, 256)
    print("[s1] reproducing S0 checkpoint on 60 original probes", flush=True)
    before = score(base(), original)
    selected = {"status": "not-yet-scored"}
    lost = [q.question for q in original if full["correct"].get(q.question, False)
            and not no_mem["correct"].get(q.question, False)
            and not before["correct"].get(q.question, False)]
    saved = torch.load(args.s0_rows, map_location="cpu", weights_only=True)
    historical_audit = {
        "historical_lost": saved["lost_questions"],
        "deterministic_lost": lost,
        "extra_lost": sorted(set(lost) - set(saved["lost_questions"])),
        "missing_lost": sorted(set(saved["lost_questions"]) - set(lost)),
        "original_probes": [{"question": q.question, "answers": q.answers} for q in original],
        "full": full, "no_memory": no_mem, "before": before,
        "dropped": dropped, "covered": covered,
    }
    print("[s1] deterministic baseline lost=%d; historical differences: %s" %
          (len(lost), json.dumps({k: historical_audit[k] for k in
                                ("extra_lost", "missing_lost")})), flush=True)
    # Extra deterministic losses stay in the audit, never in the primary target.
    try:
        lost = s0_primary_lost(saved["lost_questions"], lost)
    except ValueError:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump({"status": "invalid-baseline", "historical_s0_audit": historical_audit,
                       "reader_dropout_audit": dropout_audit, "driver_sha256": digest(__file__)}, f, indent=2)
        raise
    print("[s1] fixed historical primary lost=%d" % len(lost), flush=True)
    print("[s1] scoring full on %d expanded probes" % len(probes), flush=True)
    full = E.score_uncompacted(manager, probes, 256)
    print("[s1] scoring expanded native selection", flush=True)
    selected = score(base(selection=True))
    print("[s1] scoring expanded checkpoint", flush=True)
    before = score(base())
    retained = [q.question for q in probes if full["correct"].get(q.question, False)
                and selected["correct"].get(q.question, False)]
    before_retained = [q for q in retained if before["correct"].get(q, False)]
    report = {"status": "running", "diagnostic_only": True, "protocol": {
        "window": [3, 17890], "original_probes": len(original), "expanded_probes": len(probes),
        "steps": args.steps, "lr": args.lr, "parameter_budget": args.budget,
        "coefficient_budget": 0.5, "seed": 0, "smoke": args.smoke,
        "gate": "final rescue >= 7; select-relative harm <= floor(oracle harm/2); "
                "harm strictly below oracle and no greater than unconstrained or UL; "
                "net vs select >= 0; retained >= 10",
        "checkpoint_sha256": digest(args.carrier_ckpt), "s0_rows_sha256": digest(args.s0_rows),
        "driver_sha256": digest(__file__), "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0)},
        "sets": {"lost": lost, "retained_full_and_select": retained,
                 "retained_also_correct_at_D0": before_retained},
        "historical_s0_audit": historical_audit, "reader_dropout_audit": dropout_audit,
        "reader_config": model.config.to_dict(), "arms": {}}

    def flush():
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(report, f, indent=2)
        tmp.replace(path)

    def metrics(scored, step=0):
        correct = scored["correct"]
        rescue = [q for q in lost if correct.get(q, False)]
        harm = [q for q in retained if not correct.get(q, False)]
        added = [q for q in before_retained if not correct.get(q, False)]
        added_all = [q for q, was_correct in before["correct"].items()
                     if was_correct and not correct.get(q, False)]
        gains = [q.question for q in probes if full["correct"].get(q.question, False)
                 and not selected["correct"].get(q.question, False) and correct.get(q.question, False)]
        return {"step": step, "rescued": len(rescue), "harmed_vs_select": len(harm),
                "newly_harmed_vs_D0": len(added), "newly_harmed_vs_D0_all": len(added_all),
                "newly_harmed_vs_D0_scope": "full-and-native-select retained only",
                "harm_rate": len(harm) / max(1, len(retained)),
                "net_vs_select": len(gains) - len(harm), "accuracy": scored["accuracy"],
                "rescued_questions": rescue, "harmed_questions": harm,
                "newly_harmed_questions": added, "newly_harmed_questions_all": added_all, "correct": correct}

    report["arms"]["select"] = {"final": metrics(selected)}
    report["checkpoint_diagnostic"] = metrics(before)
    print("[s1] original=%d expanded=%d lost=%d retained=%d D0_retained=%d" %
          (len(original), len(probes), len(lost), len(retained), len(before_retained)), flush=True)
    flush()
    if any(full["correct"].get(q.question) != historical_audit["full"]["correct"].get(q.question)
           or before["correct"].get(q.question) != historical_audit["before"]["correct"].get(q.question)
           for q in original):
        report.update(status="stopped", verdict="INVALID-BASELINE", reason="repeated original scores differ")
        flush()
        return
    if not lost:
        report.update(status="stopped", verdict="NO-GO", reason="no deterministic lost probes")
        flush()
        return
    if len(retained) < 10:
        report.update(status="stopped", verdict="NO-GO", reason="expanded retained set < 10")
        flush()
        return
    window_used, target_used = E.WINDOW_BLOCKS * manager.block, E.TARGET_BLOCKS * manager.block
    mask = _compacted_mask(manager, manager.used - window_used + target_used, "cuda:0")

    def candidate(arm):
        source = manager._memory[:, manager.total - manager.used:manager.total, :].to("cuda:0")
        weights = manager._window_importance(manager.total - manager.used, window_used)
        return arm(source, weights, target=target_used, block=manager.block,
                   goal_direction=manager.goal_direction, select_weights=manager.select_weights)

    # Fresh lost rows ensure the basis describes this exact runtime pool.
    jacobian_arm = base()
    jacobian_arm.down.weight.requires_grad_(True)
    rows = {}
    for q in dict.fromkeys(lost + retained):
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
        print("[s1] Jacobian %d/%d" % (len(rows), len(set(lost + retained))), flush=True)
    jacobian_arm.down.weight.requires_grad_(False)
    ul, uk = orth(torch.stack([rows[q] for q in lost])), orth(torch.stack([rows[q] for q in retained]))
    safe, energy = protect(ul, uk)
    overlap = float((uk.T @ safe).norm())
    report["basis"] = {"lost_rank": ul.shape[1], "retained_rank": uk.shape[1],
        "safe_rank": safe.shape[1], "surviving_column_norms": energy.tolist(), "protection_overlap": overlap}
    torch.save({"lost": lost, "retained": retained, "rows": rows, "ul": ul, "uk": uk, "safe": safe},
               str(Path(args.out).with_suffix(".basis.pt")))
    flush()
    if safe.shape[1] == 0 or overlap > 1e-5:
        report.update(status="stopped", verdict="NO-GO", reason="empty or invalid safe basis")
        flush()
        return

    class Displacement(torch.nn.Module):
        def __init__(self, weight, basis):
            super().__init__()
            self.register_buffer("basis", None if basis is None else basis.to(weight).contiguous())
            self.value = torch.nn.Parameter(torch.zeros_like(weight) if basis is None else
                                            weight.new_zeros(basis.shape[1]))
        def forward(self, weight):
            delta = self.value if self.basis is None else (self.basis @ self.value).reshape_as(weight)
            return weight + delta
        def clamp(self):
            with torch.no_grad():
                self.value.mul_((args.budget / self.value.norm().clamp_min(1e-12)).clamp_max(1))

    for name, basis in [("unconstrained", None), ("coefficient_oracle", None),
                        ("subspace_ul", ul), ("protected_usafe", safe)]:
        torch.manual_seed(0)
        arm = base(fixed=name == "coefficient_oracle")
        if name == "coefficient_oracle":
            oracle = CoefficientOracle(int(cfg.carrier_rank), max_deviation=0.5).to("cuda:0")
            # The inner arm owns checkpoint dispatch; expose E to its trainable
            # parameter scan without unfreezing any checkpoint parameter.
            arm.add_module("_s1_oracle", oracle)
            wrapper = OracleCompactor(arm, oracle)
            arm._layer_delta = wrapper._layer_delta
            trainable, displacement = [oracle.residual], None
        else:
            displacement = Displacement(arm.down.weight, basis)
            parametrize.register_parametrization(arm.down, "weight", displacement)
            wrapper, trainable = arm, [displacement.value]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr)
        history = []
        report["arms"][name] = {"history": history, "trainable_dof": sum(x.numel() for x in trainable)}
        if score(wrapper)["correct"] != before["correct"]:
            raise RuntimeError("arm identity-at-init failed: " + name)
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            cand = candidate(wrapper)
            with _pool_in_place(manager, "cuda:0", cand, window_used, target_used) as injection:
                pool = injection.reinject()
                for q in lost:
                    scalar = margin_scalar(manager, by_q[q], pool, mask, 256)
                    if scalar is None:
                        raise RuntimeError("missing training margin")
                    scalar.backward(retain_graph=True)
                    del scalar
            if any(x.grad is None or not torch.isfinite(x.grad).all() for x in trainable):
                raise RuntimeError("missing/non-finite gradient: " + name)
            optimizer.step()
            if displacement is not None:
                displacement.clamp()
            del cand, pool
            torch.cuda.empty_cache()
            if step % args.measure_every == 0 or step == args.steps:
                entry = metrics(score(wrapper), step)
                entry["deviation"] = float(displacement.value.detach().norm()) if displacement is not None else float(
                    (oracle.matrix() - torch.eye(oracle.rank, device="cuda:0")).detach().norm())
                history.append(entry)
                report["arms"][name]["final"] = entry
                flush()
                print("[s1] %s step=%d rescue=%d harm=%d/%d added_harm_retained=%d added_harm_all=%d net=%d" %
                    (name, step, entry["rescued"], entry["harmed_vs_select"], len(retained),
                     entry["newly_harmed_vs_D0"], entry["newly_harmed_vs_D0_all"], entry["net_vs_select"]), flush=True)
        torch.save(wrapper.state_dict(), str(Path(args.out).with_suffix("." + name + ".pt")))
        del wrapper, arm, optimizer, trainable, displacement
        if name == "coefficient_oracle":
            del oracle
        torch.cuda.empty_cache()
    finals = {name: arm["final"] for name, arm in report["arms"].items()}
    protected, reference = finals["protected_usafe"], finals["coefficient_oracle"]
    passed = (protected["rescued"] >= 7 and protected["harmed_vs_select"] < reference["harmed_vs_select"]
              and protected["harmed_vs_select"] <= reference["harmed_vs_select"] // 2
              and all(protected["harmed_vs_select"] <= finals[n]["harmed_vs_select"]
                      for n in ("unconstrained", "subspace_ul")) and protected["net_vs_select"] >= 0)
    report.update(status="complete", verdict="SMOKE-ONLY" if args.smoke else ("GO" if passed else "NO-GO"))
    flush()
    print("[s1] verdict=" + report["verdict"], flush=True)


if __name__ == "__main__":
    main()
