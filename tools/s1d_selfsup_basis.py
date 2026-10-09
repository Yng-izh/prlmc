#!/usr/bin/env python
"""S1d-B stage-1: window self-supervised 13D basis feasibility (no answer labels).

User directive: "保留现有 carrier_residual 结构和 E6d interleave checkpoint，
只改变低维方向的获取方式" -- the next experiment tests whether a 13D basis
built from observable, answer-free window information can cover the gold
rescue-gradient subspace better than a fixed global SVD basis.

Supervision signal (answer-free): fact-completion cross-entropy on the
window's own fact lines, scored under the D0 compaction (the trained E6d
interleave operator).  For every fact line whose supporting block selection
drops, the reader is shown the line with its tail masked; the loss forces the
pool to re-encode the masked tail.  The gradient of that loss w.r.t.
down.weight is one row of the self-supervised Jacobian, exactly parallel to
the gold rescue rows in s1c_global_oracle.jacobian_row.

Protocol (pre-registered):
  * 8 basis windows = the s1c_global_oracle TRAIN_WINDOWS (row 3, starts
    14470..17890); held-out windows (2,3450) (2,3830) are never touched.
  * Self-sup row = d/d(down.weight) CE(masked-fact-line | D0 pool), one row
    per dropped fact line, up to 40 lines per window (seed-0 fixed order).
  * U_self(w) = SVD_13 of w's self-sup rows (uncentered, matching the gold
    rows' construction in s1c_global_oracle stage 1).
  * Gold target = w's rescue_d0_wrong gold-CE Jacobian rows (same cache as
    LOWO, runs/s1c_global_rows.pt, sha 73754926...).
  * Feasibility gate (per user thresholds): mean_gRho >= 0.20 AND
    mean_gRho >= 1.5 * lowo_Rho13.  FAIL -> stop before any arm training.

All basis windows are development windows; nothing here trains anything.  The
gradient computation stage caches rows to runs/s1d_selfsup_rows.pt so the
coverage stage can resume without GPU time.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

RANKS = (13,)
TRAIN_WINDOWS = [(3, 17890), (3, 17130), (3, 16750), (3, 14470),
                 (3, 17510), (3, 14850), (3, 16370), (3, 15230)]
MAX_SELFSUP_LINES = 40
MASK_SPAN = 4           # tail tokens of each fact line to mask (lines are 7-19 tokens)
RANK = 13

PROTOCOL = {
    "experiment": "S1d-B stage 1: answer-free window basis feasibility",
    "supervision": "fact-completion CE on dropped-block fact lines under D0 pool",
    "basis": "uncentered SVD of self-sup Jacobian rows, rank 13 per window",
    "target": "gold rescue_d0_wrong CE Jacobian rows per window (s1c_global_rows.pt)",
    "gate": "mean_gRho >= 0.20 and mean_gRho >= 1.5 * lowo_Rho13 -> continue to stage 2",
    "lowo_reference": "runs/s1c_lowo_coverage.json rank13 per-window lowo_energy_coverage",
    "max_lines_per_window": MAX_SELFSUP_LINES,
    "mask_span": MASK_SPAN,
    "seed": 0,
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def orth(rows: torch.Tensor) -> torch.Tensor:
    """Orthonormal columns from row stack (n, d) -> (d, min(n,d)); reuse s1_local_protected.orth semantics."""
    q, _ = torch.linalg.qr(rows.t().double())
    return q


def energy_coverage(U: torch.Tensor, G: torch.Tensor) -> float:
    """||U^T G^T||_F^2 / ||G||_F^2, U (d,r) orthonormal, G (n,d)."""
    P = U.t() @ G.t().double()
    return float((P * P).sum() / (G.double() * G.double()).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="runs/s1c_global_rows.pt",
                    help="gold rescue rows cache (target subspace)")
    ap.add_argument("--selfsup-out", default="runs/s1d_selfsup_rows.pt")
    ap.add_argument("--out", default="runs/s1d_selfsup_basis.json")
    ap.add_argument("--lowo", default="runs/s1c_lowo_coverage.json")
    ap.add_argument("--carrier-ckpt", default="runs/e6d_interleave_ckpt.pt")
    ap.add_argument("--cuda-device", type=int, default=9)
    ap.add_argument("--skip-gradient-stage", action="store_true",
                    help="only recompute coverage from cached self-sup rows")
    args = ap.parse_args()

    SCRIPT_SHA = sha256_file(os.path.abspath(__file__))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    gold = torch.load(args.rows, map_location="cpu", weights_only=False)
    gold_wins = {wk: torch.stack([gold["rows_by_window"][wk][q]
                                  for q in sorted(gold["rows_by_window"][wk])])
                 for wk in gold["rows_by_window"]}
    lowo = json.loads(Path(args.lowo).read_text())
    lowo_r13 = {w["held_out_window"]: w["ranks"]["13"]["lowo_energy_coverage"]
                for w in lowo["windows"]}

    # ---------------- stage A: self-sup gradient rows on GPU ----------------
    if not args.skip_gradient_stage and not Path(args.selfsup_out).exists():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
        os.environ["S0B_GPU_SET"] = "1"
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch.nn.functional as F
        import e6a_overfit as E
        from s0_functional_sensitivity import build_window_and_probes, CONFIG, MODEL
        from methods.prlmc.config import PRLMCConfig
        from methods.prlmc.layout import ProtectedResidualMemory
        from methods.prlmc.probes import _pool_in_place, _compacted_mask
        from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
        from transformers import AutoTokenizer

        torch.manual_seed(0)
        torch.set_num_threads(4)
        with open(CONFIG) as f:
            import yaml
            config = dict(yaml.safe_load(f)["prlmc"])
        config.update(device="cuda:0", dtype="float16",
                      stats_dir="./outputs/prlmc-s1d-selfsup")
        cfg = PRLMCConfig.from_dict(config)
        model = MemoryLLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                          attn_implementation="sdpa").to("cuda:0")
        model.initialized.zero_()
        for x in model.parameters():
            x.requires_grad_(False)
        model.train()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        from s1_local_protected import disable_reader_dropout, digest
        manager = ProtectedResidualMemory(model, AutoTokenizer.from_pretrained(MODEL), cfg)
        disable_reader_dropout(model)
        import argparse as ap2
        args_probe = ap2.Namespace(probes=60, max_probe_tokens=256)
        from methods.prlmc.consolidate import CarrierResidualCompactor
        state = torch.load(args.carrier_ckpt, map_location="cuda:0", weights_only=True)
        if digest(args.carrier_ckpt) != gold["checkpoint_sha256"]:
            raise SystemExit("carrier checkpoint hash mismatch vs gold rows cache")

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
            built = build_window_and_probes(manager, line_cache, row, start, args_probe)
            if built is None:
                raise RuntimeError("window build failed (%d,%d)" % (row, start))
            covered, indices, dropped, original = built
            return covered, indices, dropped, original

        def selfsup_row(fact_line: str, arm, mask):
            """d/d(down.weight) of the fact-completion CE under the D0 pool."""
            arm.down.weight.requires_grad_(True)
            arm.down.weight.grad = None
            cand = candidate(arm)
            with _pool_in_place(manager, "cuda:0", cand, window_used,
                                target_used) as injection:
                pool = injection.reinject()
                text = fact_line.rstrip(".")
                tokens = manager.tokenizer(text, return_tensors="pt",
                                           add_special_tokens=False).input_ids
                n = tokens.shape[1]
                if n < 2 * MASK_SPAN + 2:
                    return None
                span = MASK_SPAN
                input_ids = tokens.to("cuda:0")
                labels = input_ids.clone()
                labels[:, :-span] = -100
                from methods.prlmc.probes import differentiable_memory, active_mask_override
                with differentiable_memory(model, pool), active_mask_override(model, mask):
                    logits = model(input_ids=input_ids, return_dict=True).logits
                shift_logits = logits[:, :-1, :].float()
                shift_labels = labels[:, 1:]
                loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1))
            loss.backward()
            row = arm.down.weight.grad.detach().flatten().cpu().clone()
            arm.down.weight.grad = None
            arm.down.weight.requires_grad_(False)
            del loss, cand, pool, logits
            torch.cuda.empty_cache()
            return row

        rows_out = {"rows_by_window": {},
                    "checkpoint_sha256": digest(args.carrier_ckpt),
                    "protocol": PROTOCOL}
        mask = None
        for row, start in TRAIN_WINDOWS:
            covered, indices, dropped, original = fresh_window(row, start)
            mask = _compacted_mask(manager, manager.used - window_used + target_used,
                                   "cuda:0")
            line_to_block = {}
            for block_index, (lo, hi, _b, _w) in enumerate(covered):
                for line in range(lo, hi):
                    line_to_block[line] = block_index
            lines = line_cache[row]
            # dropped-block fact lines, serial order, fixed seed
            dropped_lines = [i for i in sorted(line_to_block)
                             if line_to_block[i] in set(dropped)
                             and E.first_match(lines[i]) is not None]
            dropped_lines = dropped_lines[:MAX_SELFSUP_LINES]
            arm = base()
            window_rows = {}
            for i, line_idx in enumerate(dropped_lines, 1):
                rowv = selfsup_row(lines[line_idx], arm, mask)
                if rowv is not None:
                    window_rows[str(line_idx)] = rowv
                print("[s1d] (%d,%d) selfsup row %d/%d" % (row, start, i, len(dropped_lines)),
                      flush=True)
            rows_out["rows_by_window"]["%d,%d" % (row, start)] = window_rows
            torch.save(rows_out, args.selfsup_out)
            del arm
            gc.collect()
            torch.cuda.empty_cache()
            print("[s1d] (%d,%d) cached %d selfsup rows" % (row, start, len(window_rows)),
                  flush=True)

    # ---------------- stage B: coverage, fully CPU/fp64 ----------------
    ss = torch.load(args.selfsup_out, map_location="cpu", weights_only=False)
    ss_wins = {wk: torch.stack([ss["rows_by_window"][wk][q]
                                for q in sorted(ss["rows_by_window"][wk])])
               for wk in ss["rows_by_window"]}
    for wk in ss_wins:
        if ss_wins[wk].shape[0] < RANK:
            raise SystemExit("window %s has %d selfsup rows < rank %d"
                             % (wk, ss_wins[wk].shape[0], RANK))

    report = {
        "schema_version": "s1d_selfsup_basis/1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rows_sha256": sha256_file(args.rows),
        "selfsup_rows_sha256": sha256_file(args.selfsup_out),
        "checkpoint_sha256": gold["checkpoint_sha256"],
        "driver_sha256": SCRIPT_SHA,
        "protocol": PROTOCOL,
        "windows": [],
    }

    g_rhos, l_rhos, u_golds = [], [], []
    for wk in sorted(ss_wins):
        G = gold_wins[wk].double()
        U_ss = orth(ss_wins[wk].double())[:, :RANK].contiguous()
        g_rho = energy_coverage(U_ss, G)
        # gold self basis (oracle upper bound, same construction as LOWO self)
        U_gold = orth(gold_wins[wk].double())[:, :RANK].contiguous()
        u_rho = energy_coverage(U_gold, G)
        l_rho = lowo_r13[wk]
        # alignment between the self-sup basis and the gold basis
        M = U_ss.t() @ U_gold
        align = float(torch.linalg.svdvals(M).mean())
        report["windows"].append({
            "window": wk,
            "n_selfsup_rows": int(ss_wins[wk].shape[0]),
            "n_gold_rows": int(gold_wins[wk].shape[0]),
            "selfsup_r13_gold_coverage": g_rho,
            "gold_self_r13_gold_coverage": u_rho,
            "lowo_r13_gold_coverage": l_rho,
            "ss_vs_gold_align": align,
        })
        g_rhos.append(g_rho)
        l_rhos.append(l_rho)
        u_rhos = None
        print("[s1d] %s gRho=%.4f uRho=%.4f lowoRho=%.4f align=%.4f"
              % (wk, g_rho, u_rho, l_rho, align), flush=True)

    mean_g = sum(g_rhos) / len(g_rhos)
    mean_l = sum(l_rhos) / len(l_rhos)
    gate_continue = mean_g >= 0.20 and mean_g >= 1.5 * mean_l
    report["summary"] = {
        "mean_selfsup_r13_gold_coverage": mean_g,
        "mean_lowo_r13_gold_coverage": mean_l,
        "ratio_vs_lowo": mean_g / mean_l,
        "gate_continue_stage2": bool(gate_continue),
        "gate_rule": "mean gRho >= 0.20 AND mean gRho >= 1.5 * mean lowo r13",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
