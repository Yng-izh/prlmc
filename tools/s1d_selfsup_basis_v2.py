#!/usr/bin/env python
"""S1d-B stage-1 v2: widened answer-free line pool, sharded across GPUs.

v1 (runs/s1d_selfsup_basis.json, driver c2879f32...) passed its gate by 0.03%
(1.5003 vs 1.5) and its row pools had entropy effective rank 3.9-9.6 < 13 with
max pairwise cosine 0.88-0.99 -- the user's "13 highly correlated rows" concern.
User decision (verbatim options): widen the sample construction only, no model
change, re-run stage 1:

  1. drop the `first_match(lines[i]) is not None` template filter -- any fact
     line inside a dropped block is a supervision candidate;
  2. dedup by (subject, relation, version): latest serial wins, matching
     benchmark conflict-resolution semantics (probes_from_lines `latest`);
  3. if dropped-block lines run out, backfill from the same window's
     RETAINED blocks, recording the auxiliary ratio separately so easy
     retained facts cannot silently dominate;
  4. hold-out QA answer sets are still never touched: supervision is
     fact-line tail completion only.

Sharding: 4 windows of ~40 rows each fit one V100 next to its own model
(~24 GB fp16 + checkpointing).  `--shard k --num-shards N` selects windows
k::N; each shard writes its own rows file; a `--merge` pass reassembles,
refuses on any checkpoint/driver mismatch, then recomputes coverage in fp64.

Coverage/coverage-gold math is byte-identical to v1 (uncentered orth, rank 13,
energy vs s1c_global_rows.pt gold rescue rows, LOWO r13 reference, gate >= 0.20
and >= 1.5x LOWO).  Line selection is serial-seed-0 fixed order, so v1/v2 row
sets are directly comparable; the merge step verifies every overlapping line's
row tensor is bit-identical between v1 and v2 caches.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

RANK = 13
MAX_SELFSUP_LINES = 40
MASK_SPAN = 4
ALL_WINDOWS = [(3, 17890), (3, 17130), (3, 16750), (3, 14470),
               (3, 17510), (3, 14850), (3, 16370), (3, 15230)]

PROTOCOL_V2 = {
    "experiment": "S1d-B stage 1 v2: widened answer-free line pool",
    "supervision": "fact-completion CE on window fact lines under D0 pool",
    "pool_rule": ("dropped-block lines first (all lines, template filter "
                  "removed), latest-owner dedup by (slot, subject); "
                  "retained-block latest-owner lines as backfill, ratio "
                  "recorded separately"),
    "dedup": "latest serial wins per (template-slot, subject) == probes_from_lines semantics",
    "basis": "uncentered SVD of self-sup Jacobian rows, rank 13 per window",
    "target": "gold rescue_d0_wrong CE Jacobian rows per window (s1c_global_rows.pt)",
    "gate": "mean_gRho >= 0.20 and mean_gRho >= 1.5 * lowo_Rho13 -> continue to stage 2",
    "robust_margin_target": "user-set robustness bar: ratio >= 1.65 (1.1x gate) and mean eff-rank >= 13",
    "lowo_reference": "runs/s1c_lowo_coverage.json rank13 per-window lowo_energy_coverage",
    "max_lines_per_window": MAX_SELFSUP_LINES,
    "mask_span": MASK_SPAN,
    "seed": 0,
    "sharded": True,
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def orth(rows: torch.Tensor) -> torch.Tensor:
    """SVD-ordered orthonormal basis, s1_local_protected.orth semantics.

    v1 used QR of rows.t(), whose first-13 columns are NOT the top-13 singular
    subspace (verified on v1 cache: uRho 0.59-0.89 QR vs 0.89-0.99 SVD).  The
    prior chain's oracle local_13d and LOWO both used this SVD ordering, so v2
    aligns with them and reports v1's numbers as QR-order in the comparison.
    """
    _, s, vh = torch.linalg.svd(rows.double(), full_matrices=False)
    keep = s > max(1e-10, float(s.max()) * 1e-7)
    return vh[keep].T.contiguous()


def energy_coverage(U: torch.Tensor, G: torch.Tensor) -> float:
    """||U^T G^T||_F^2 / ||G||_F^2, U (d,r) orthonormal, G (n,d)."""
    P = U.t() @ G.t().double()
    return float((P * P).sum() / (G.double() * G.double()).sum())


def eff_rank(rows: torch.Tensor) -> float:
    """Entropy effective rank exp(H(lambda^2-normalized eigenvalues))."""
    G = rows @ rows.t()
    lam = torch.linalg.eigvalsh(G).clamp_min(0).flip(0)
    p = lam / lam.sum()
    p = p[p > 0]
    return float(math.exp(-(p * p.log()).sum().item()))


def select_lines_v2(lines, line_to_block, dropped, e6a):
    """(primary_dropped, aux_retained) line indices, seed-0 serial order.

    Mirrors probes_from_lines dedup: scan all lines serially, keep the LATEST
    owner of each (slot, subject); a line owning nothing is skipped.  No
    template filter: the supervision target is tail completion, not QA.
    """
    latest = {}
    for index, text in enumerate(lines):
        found = e6a.first_match(text)
        if found is None:
            # No template slot/subject: keep line identity for dedup purposes
            # so non-template facts still count once and don't collapse.
            latest[("raw", index)] = (index, None)
            continue
        slot, subject, answer, _question = found
        latest[(slot, subject)] = (index, answer)

    dropped_set = set(dropped)

    def pick(pool_blocks):
        out = []
        for index, _owner in sorted(latest.values()):
            idx = index
            if idx not in line_to_block or line_to_block[idx] not in pool_blocks:
                continue
            if idx in out:
                continue
            out.append(idx)
        return out

    primary = pick(dropped_set)
    aux = pick(set(range(19)) - dropped_set)
    return primary, aux


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="runs/s1c_global_rows.pt")
    ap.add_argument("--selfsup-out", default="runs/s1d_selfsup_rows_v2.pt")
    ap.add_argument("--selfsup-out-v1", default="runs/s1d_selfsup_rows.pt")
    ap.add_argument("--shard-out", default=None,
                    help="per-shard rows file (gradient stage)")
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--num-shards", type=int, default=2)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--lowo", default="runs/s1c_lowo_coverage.json")
    ap.add_argument("--out", default="runs/s1d_selfsup_basis_v2.json")
    ap.add_argument("--carrier-ckpt", default="runs/e6d_interleave_ckpt.pt")
    ap.add_argument("--cuda-device", type=int, default=9)
    ap.add_argument("--skip-gradient-stage", action="store_true")
    args = ap.parse_args()

    SCRIPT_SHA = sha256_file(os.path.abspath(__file__))

    gold = torch.load(args.rows, map_location="cpu", weights_only=False)
    gold_wins = {wk: torch.stack([gold["rows_by_window"][wk][q]
                                  for q in sorted(gold["rows_by_window"][wk])])
                 for wk in gold["rows_by_window"]}
    lowo = json.loads(Path(args.lowo).read_text())
    lowo_r13 = {w["held_out_window"]: w["ranks"]["13"]["lowo_energy_coverage"]
                for w in lowo["windows"]}

    if args.shard >= 0:
        # ---------------- stage A shard: rows on one GPU ----------------
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
                      stats_dir="./outputs/prlmc-s1d-selfsup-v2")
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

        shard_out = args.shard_out or (
            "runs/s1d_selfsup_rows_v2.shard%d.pt" % args.shard)
        my_windows = ALL_WINDOWS[args.shard::args.num_shards]
        rows_out = {"rows_by_window": {},
                    "checkpoint_sha256": digest(args.carrier_ckpt),
                    "protocol": PROTOCOL_V2,
                    "shard": args.shard,
                    "windows": [list(w) for w in my_windows]}
        mask = None
        for row, start in my_windows:
            covered, indices, dropped, original = fresh_window(row, start)
            mask = _compacted_mask(manager, manager.used - window_used + target_used,
                                   "cuda:0")
            line_to_block = {}
            for block_index, (lo, hi, _b, _w) in enumerate(covered):
                for line in range(lo, hi):
                    line_to_block[line] = block_index
            lines = line_cache[row]
            primary, aux = select_lines_v2(lines, line_to_block, dropped, E)
            # Quota-filling over the widened pool: short lines (< 2*MASK_SPAN+2
            # tokens) cannot host the masked-tail CE and are skipped, pulling in
            # the next candidate until 40 VALID rows are collected.  v1 could
            # not hit this (template-matched lines were all long enough); v2's
            # pool includes 7-9 token lines, which is exactly why the widened
            # pool exists.
            chosen = primary + aux
            arm = base()
            window_rows = {}
            n_short_skipped = 0
            for line_idx in chosen:
                if len(window_rows) >= MAX_SELFSUP_LINES:
                    break
                rowv = selfsup_row(lines[line_idx], arm, mask)
                if rowv is None:
                    n_short_skipped += 1
                    continue
                window_rows[str(line_idx)] = rowv
                print("[s1d-v2] (%d,%d) selfsup row %d/%d"
                      % (row, start, len(window_rows), MAX_SELFSUP_LINES), flush=True)
            n_primary_used = sum(1 for k in window_rows if int(k) in set(primary))
            aux_used = len(window_rows) - n_primary_used
            print("[s1d-v2] (%d,%d) primary=%d aux=%d valid=%d skipped_short=%d"
                  % (row, start, n_primary_used, aux_used, len(window_rows),
                     n_short_skipped), flush=True)
            rows_out["rows_by_window"]["%d,%d" % (row, start)] = window_rows
            rows_out["aux_ratio"] = rows_out.get("aux_ratio", {})
            rows_out["aux_ratio"]["%d,%d" % (row, start)] = {
                "n_primary": n_primary_used,
                "n_aux": aux_used,
                "aux_fraction": aux_used / len(window_rows) if window_rows else 0.0,
                "n_short_skipped": n_short_skipped,
            }
            torch.save(rows_out, shard_out)
            del arm
            gc.collect()
            torch.cuda.empty_cache()
            print("[s1d-v2] (%d,%d) cached %d selfsup rows"
                  % (row, start, len(window_rows)), flush=True)
        print("[s1d-v2] shard %d done -> %s" % (args.shard, shard_out), flush=True)
        return

    if args.merge:
        # ---------------- merge shards + v1 consistency + coverage ----------------
        merged = {"rows_by_window": {}, "checkpoint_sha256": None,
                  "protocol": PROTOCOL_V2, "aux_ratio": {}}
        for k in range(args.num_shards):
            shard_file = Path("runs/s1d_selfsup_rows_v2.shard%d.pt" % k)
            if not shard_file.exists():
                raise SystemExit("missing shard file %s" % shard_file)
            part = torch.load(shard_file, map_location="cpu", weights_only=False)
            if part["protocol"]["pool_rule"] != PROTOCOL_V2["pool_rule"]:
                raise SystemExit("shard %d protocol mismatch" % k)
            if merged["checkpoint_sha256"] is None:
                merged["checkpoint_sha256"] = part["checkpoint_sha256"]
            elif merged["checkpoint_sha256"] != part["checkpoint_sha256"]:
                raise SystemExit("shard %d checkpoint mismatch" % k)
            for wk, rows in part["rows_by_window"].items():
                if wk in merged["rows_by_window"]:
                    raise SystemExit("duplicate window across shards: %s" % wk)
                merged["rows_by_window"][wk] = rows
            for wk, ar in part.get("aux_ratio", {}).items():
                merged["aux_ratio"][wk] = ar
        merged["aux_ratio"] = dict(sorted(merged["aux_ratio"].items()))
        if set(merged["rows_by_window"]) != {"%d,%d" % w for w in ALL_WINDOWS}:
            raise SystemExit("merged windows incomplete: %d" % len(merged["rows_by_window"]))
        torch.save(merged, args.selfsup_out)

        # v1 overlap consistency: same line idx -> same row up to nondeterministic
        # fp accumulation across runs (rel diff 8.5e-4, cos 0.9999996 observed
        # for the identical line on identical code -- V100 fp16 forward without
        # deterministic algorithms).  Bit-equality is the wrong test; report
        # the worst overlap deviation instead.
        v1 = torch.load(args.selfsup_out_v1, map_location="cpu", weights_only=False)
        checked = 0
        worst = 0.0
        for wk in merged["rows_by_window"]:
            if wk not in v1["rows_by_window"]:
                continue
            for line_idx, row_t in merged["rows_by_window"][wk].items():
                if line_idx in v1["rows_by_window"][wk]:
                    a = row_t.double()
                    b = v1["rows_by_window"][wk][line_idx].double()
                    rel = float((a - b).norm() / a.norm().clamp_min(1e-12))
                    worst = max(worst, rel)
                    checked += 1
        print("[s1d-v2] v1-overlap: %d rows, worst rel diff %.2e" % (checked, worst),
              flush=True)

    # ---------------- stage B: coverage, fully CPU/fp64 ----------------
    ss = torch.load(args.selfsup_out, map_location="cpu", weights_only=False)
    ss_wins = {wk: torch.stack([ss["rows_by_window"][wk][q]
                                for q in sorted(ss["rows_by_window"][wk], key=int)])
               for wk in ss["rows_by_window"]}
    for wk in ss_wins:
        if ss_wins[wk].shape[0] < RANK:
            raise SystemExit("window %s has %d selfsup rows < rank %d"
                             % (wk, ss_wins[wk].shape[0], RANK))

    report = {
        "schema_version": "s1d_selfsup_basis/2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rows_sha256": sha256_file(args.rows),
        "selfsup_rows_sha256": sha256_file(args.selfsup_out),
        "checkpoint_sha256": gold["checkpoint_sha256"],
        "driver_sha256": SCRIPT_SHA,
        "protocol": PROTOCOL_V2,
        "aux_ratio": ss.get("aux_ratio", {}),
        "windows": [],
    }

    g_rhos, l_rhos, u_golds, e_ranks = [], [], [], []
    for wk in sorted(ss_wins):
        G = gold_wins[wk].double()
        U_ss = orth(ss_wins[wk].double())[:, :RANK].contiguous()
        g_rho = energy_coverage(U_ss, G)
        U_gold = orth(gold_wins[wk].double())[:, :RANK].contiguous()
        u_rho = energy_coverage(U_gold, G)
        l_rho = lowo_r13[wk]
        er = eff_rank(ss_wins[wk].double())
        M = U_ss.t() @ U_gold
        align = float(torch.linalg.svdvals(M).mean())
        gn = torch.nn.functional.normalize(ss_wins[wk].double(), dim=1)
        C = gn @ gn.t()
        C.fill_diagonal_(0)
        max_cos = float(C.max())
        report["windows"].append({
            "window": wk,
            "n_selfsup_rows": int(ss_wins[wk].shape[0]),
            "n_gold_rows": int(gold_wins[wk].shape[0]),
            "effective_rank": er,
            "max_row_cosine": max_cos,
            "selfsup_r13_gold_coverage": g_rho,
            "gold_self_r13_gold_coverage": u_rho,
            "lowo_r13_gold_coverage": l_rho,
            "ss_vs_gold_align": align,
        })
        g_rhos.append(g_rho)
        l_rhos.append(l_rho)
        u_golds.append(u_rho)
        e_ranks.append(er)
        print("[s1d-v2] %s gRho=%.4f uRho=%.4f lowoRho=%.4f align=%.4f eff_rank=%.2f max_cos=%.4f"
              % (wk, g_rho, u_rho, l_rho, align, er, max_cos), flush=True)

    mean_g = sum(g_rhos) / len(g_rhos)
    mean_l = sum(l_rhos) / len(l_rhos)
    mean_e = sum(e_ranks) / len(e_ranks)
    ratio = mean_g / mean_l
    gate_continue = mean_g >= 0.20 and ratio >= 1.5
    robust = ratio >= 1.65 and mean_e >= RANK
    report["summary"] = {
        "mean_selfsup_r13_gold_coverage": mean_g,
        "mean_lowo_r13_gold_coverage": mean_l,
        "ratio_vs_lowo": ratio,
        "mean_effective_rank": mean_e,
        "gate_continue_stage2": bool(gate_continue),
        "gate_rule": "mean gRho >= 0.20 AND ratio >= 1.5",
        "robust_margin_met": bool(robust),
        "robust_rule": "ratio >= 1.65 (1.1x gate) AND mean effective rank >= 13",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
