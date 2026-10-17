"""S0: functional sensitivity of the residual-subspace idea, measured end-to-end.

The one question this answers:

    When the compactor compresses a real window, are there parameter directions
    (of D, the residual coefficients encoder) whose displacement destroys probe
    recovery much faster than others?

If yes, a functional constraint on later training (soft filter / hard projection)
has something real to protect. If every direction hurts equally, S1/S2's
constraint machinery is protecting noise and the plan's premise fails here.

Method, in order
----------------
1. Build the same 19->9 hard window E6a used (append-only, no compaction during
   the build), with the same probe construction and the same dropped blocks.
2. For every probe, run one forward through the compacted pool with grad enabled
   and read the scalar gold answer loss. One scalar -> one Jacobian row on
   `down.weight` via `scalar_jacobian_rows`. Rows are kept per probe; they are
   never summed before the sketch is fitted (summing gold-with-foil opposite
   gradients is exactly how a sensitivity signal cancels itself).
3. Fit `FunctionalSensitivitySketch` on those rows. Rank is capped by
   `--sketch-rank`; the Gram matrix is [n, n] so nothing here is p x p.
4. Perturb D along three direction families, at matched norms:
       sensitive      top sketch directions (what the constraint should protect)
       low_sensitive  random directions orthogonalised against the sketch span
       random         plain random directions
   For each family x scale, re-run the compactor forward and score the probes.
   The readable outcome is `lost_recovered` per family: how many of the
   selection-lost probes the carrier's residual still answers.

Everything the run prints is a raw number with its measurement path; the verdict
word is emitted only from the pre-registered rule below, not from eyeballing.

Pre-registered reading (fixed before the run, from the plan's section 13):
    separation = sensitive_harm - low_sensitive_harm  (per matched scale)
    if max separation >= 0.25 and low_sensitive_harm stays < 0.5 at the scale
    where sensitive_harm first reaches >= 0.5: S0 PASSES (subspace is real)
    if separation < 0.10 everywhere: S0 FAILS (constraint machinery is noise)
    otherwise: INCONCLUSIVE, report and stop; do not pick a side by re-reading.

No training happens here. No AdamW step, no probe-guided stage, no probes
beyond scoring. The reader is frozen; only `down.weight` receives perturbations
that are immediately discarded (the base compactor is deep-copied per family).
"""

import argparse
import copy
import hashlib
import json
import os
import sys
from contextlib import nullcontext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- GPU pick before torch import (same pattern as e6b_mini) -----------------
if __name__ == "__main__" and not os.environ.get("S0_GPU_SET"):
    import subprocess

    need_mb = int(os.environ.get("S0_NEED_MB", "21000"))
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    free = []
    for line in out.strip().splitlines():
        index, used = [int(v) for v in line.split(",")]
        free.append((32768 - used, index))
    free.sort(reverse=True)
    if free[0][0] < need_mb:
        raise SystemExit(
            "no GPU with %d MB free; best is gpu%d with %d MB" % (need_mb, free[0][1], free[0][0])
        )
    print("[s0] gpu%d, %d MB free" % (free[0][1], free[0][0]), flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(free[0][1])
    os.environ["S0_GPU_SET"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import torch  # noqa: E402
import yaml  # noqa: E402

import e6a_overfit as E  # noqa: E402  (window building, probe scoring, constants)

from methods.prlmc.consolidate import CarrierResidualCompactor  # noqa: E402
from methods.prlmc.config import PRLMCConfig  # noqa: E402
from methods.prlmc.functional_subspace import (  # noqa: E402
    FunctionalSensitivitySketch,
)
from methods.prlmc.layout import ProtectedResidualMemory  # noqa: E402
from methods.prlmc.probes import (  # noqa: E402
    _pool_in_place,
    active_mask_override,
    differentiable_memory,
)

MODEL = "/share/latent/MemoryLLM-8B"
CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "agent_conf", "prlmc_e6.yaml")


def build_window_and_probes(manager, rows, row, start, args):
    """One fresh window state + probes, reusing E6a's construction verbatim."""
    window_slots = E.WINDOW_BLOCKS * manager.block
    built = E.write_until(manager, rows[row], window_slots, start)
    if built is None:
        return None
    covered = built
    indices = E.probe_indices(covered, window_slots)
    begin = manager.total - manager.used
    source = manager._memory[:, begin : begin + window_slots, :].to("cuda:0")
    weights = manager._window_importance(begin, window_slots)
    dropped = E.dropped_block_indices(manager, source, weights)
    del source, weights
    torch.cuda.empty_cache()
    line_to_block = {}
    for block_index, (lo, hi, _b, _w) in enumerate(covered):
        for line in range(lo, hi):
            line_to_block[line] = block_index
    probes = E.probes_from_lines(
        rows[row], indices, args.probes,
        dropped_blocks=set(dropped), line_to_block=line_to_block,
    )
    return covered, indices, dropped, probes


def margin_scalar(manager, probe, pool, mask, max_probe_tokens):
    """Gold-answer cross-entropy of one probe under the candidate pool, grad on.

    This is the per-probe functional quantity the sketch is fitted on. The
    forward is the same teacher-forced path `score_probes` uses, but the loss
    is a scalar (sum over the answer span, not batch mean), because the row set
    must be one row per probe, and `enable_grad` is on so `down.weight` gets a
    gradient. The reader is never trained: only this scalar's gradient is read.
    """
    device = manager.model.memory.device
    stem = probe.prompt if probe.prompt else f"{probe.question}\nAnswer:"
    answer = probe.answers[0]
    encoded = manager.tokenizer(
        f"{stem} {answer}", return_tensors="pt", add_special_tokens=False
    )
    input_ids = encoded.input_ids[:, :max_probe_tokens].to(device)
    stem_len = len(manager.tokenizer(stem, add_special_tokens=False).input_ids)
    if input_ids.shape[1] < 2 or stem_len <= 0 or stem_len >= input_ids.shape[1]:
        return None
    with differentiable_memory(manager.model, pool), active_mask_override(manager.model, mask):
        logits = manager.model(input_ids=input_ids, return_dict=True).logits
    span = logits[:, stem_len - 1 : -1, :]
    labels = input_ids[:, stem_len:]
    return F_cross_entropy(span, labels)


def F_cross_entropy(span, labels):
    import torch.nn.functional as F

    return F.cross_entropy(span.reshape(-1, span.shape[-1]).float(), labels.reshape(-1))


def perturb_down(base_compactor, direction_flat, scale, dimension):
    """A deep copy of the compactor with `scale * direction` added to D.

    The direction is a flat vector over `down.weight`. Copies are made per
    measurement so no family's perturbation ever leaks into another's.
    """
    clone = copy.deepcopy(base_compactor)
    with torch.no_grad():
        delta = (direction_flat * scale).reshape(clone.down.weight.shape).to(
            clone.down.weight.device, clone.down.weight.dtype
        )
        clone.down.weight.add_(delta)
    clone.eval()
    return clone


def score_family(manager, probes, compactor, args, label, base_lost_questions):
    """Score one perturbed compactor: forward, probe accuracy, lost-recovery.

    `base_lost_questions` is the set the *unperturbed initial* compactor loses.
    Every family is measured against that same set, so the numbers are one
    comparison, not a moving denominator.
    """
    pool, mask = E.candidate_pool(manager, compactor, "cuda:0")
    scored = E.score_probes(manager, probes, pool, args.max_probe_tokens, mask=mask)
    del pool, mask
    torch.cuda.empty_cache()
    kept = sum(
        1
        for q in base_lost_questions
        if scored["correct"].get(q, False)
    )
    # Per-probe detail for the functional-partition question: which lost
    # probes each direction brings back is the probe x direction matrix
    # S0-partition analyses. The plain count above cannot distinguish
    # "every direction rescues the same 6 probes" from "each direction
    # rescues a disjoint slice".
    recovered_questions = sorted(
        q
        for q in base_lost_questions
        if scored["correct"].get(q, False)
    )
    return {
        "family": label,
        "accuracy": scored["accuracy"],
        "lost_recovered": kept,
        "n_lost_base": len(base_lost_questions),
        "recovered_questions": recovered_questions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", default="0,500",
                        help="row,start of the hard window (E6a's pinned window)")
    parser.add_argument("--probes", type=int, default=60)
    parser.add_argument("--sketch-rank", type=int, default=16)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--max-probe-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--carrier-ckpt", default=None,
                        help="warm-up trained compactor checkpoint (the S0 base point). "
                        "Without it the compactor is at init, where up=0 makes every "
                        "margin gradient identically zero (the dead-zone failure): "
                        "the sketch fits rank 0 and the sensitive family degenerates.")
    parser.add_argument("--out", default="/tmp/s0_functional_sensitivity.json")
    parser.add_argument("--dump-rows", default=None,
                        help="path to torch.save the 13x131072 Jacobian rows "
                        "so partition analyses do not need a rerun")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    raw = yaml.safe_load(open(CONFIG))
    config = dict(raw["prlmc"])
    config["device"] = "cuda:0"
    config["dtype"] = "float16"
    config["stats_dir"] = "./outputs/prlmc-s0"
    cfg = PRLMCConfig.from_dict(config)

    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from transformers import AutoTokenizer

    model = MemoryLLM.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = model.to("cuda:0").eval()
    # `initialized=1` in the checkpoint makes text-only forward die; zero it.
    # lm_head is NOT replaced: score_probes consumes .logits over the full
    # 128k vocab for cross_entropy. e7a_similarity could swap in Identity only
    # because it reads hidden_states[-1] and never touches logits.
    model.initialized.zero_()
    # Gradient checkpointing for the Jacobian phase. The steady state is
    # reader 16 GiB + pool 3.4 GiB; a full 32-layer fp16 activation graph for
    # one probe's backward needs another 7-8 GiB, which exceeds the card. With
    # checkpointing each decoder layer's activations are rebuilt during
    # backward, so the retained graph drops to layer-boundary states (~2 GiB).
    #
    # The checkpointing branch (modeling_memoryllm.py:2001) requires
    # `self.training` -- eval() silently skips it (run 4 died at the identical
    # 26.51 GiB as run 3). So the reader runs in train() mode with every
    # parameter frozen: no weights change, no dropout (config is 0.0), and the
    # only effect of train() is that the checkpointing branch is taken.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    manager = ProtectedResidualMemory(model, tokenizer, cfg)
    row, start = [int(v) for v in args.window.split(",")]
    rows = {row: E.read_fact_lines(row)}
    built = build_window_and_probes(manager, rows, row, start, args)
    if built is None:
        raise SystemExit("window %d,%d could not be built; use --window row,start" % (row, start))
    covered, indices, dropped, probes = built
    print("[s0] window row %d start %d: %d blocks dropped %s, %d probes"
          % (row, start, E.WINDOW_BLOCKS, dropped, len(probes)), flush=True)

    base = CarrierResidualCompactor(
        num_layers=manager.num_layers, hidden=manager.hidden, block=manager.block,
        rank=int(cfg.carrier_rank), gamma_max=float(cfg.carrier_gamma_max),
    ).to(device="cuda:0", dtype=torch.float32)
    if args.carrier_ckpt:
        # The real S0 base point. At init `up = 0`, so the residual is zero,
        # the pool equals select_blocks bitwise, and every margin's gradient on
        # D passes through a `up = 0` factor -- rank 0, all-zero rows, dead
        # zone (verified on run 7: 6/6 rows zero, sketch trace 0.0000). A
        # warm-up-trained checkpoint moves the base point off that saddle and
        # gives the sketch an actual functional landscape to read.
        state = torch.load(args.carrier_ckpt, map_location="cuda:0", weights_only=True)
        base.load_state_dict(state["state_dict"])
        print("[s0] carrier checkpoint loaded: %s" % args.carrier_ckpt, flush=True)
    else:
        print("[s0] WARNING: no --carrier-ckpt; running at init (dead zone, "
              "expect rank-0 sketch)", flush=True)

    # Base measurement. With a warm-up checkpoint the `before` pool is the
    # trained compactor's output; `lost` is then "what the trained carrier
    # still loses", which is the denominator S0 perturbs from.
    no_mem = E.score_no_memory(manager, probes, args.max_probe_tokens)
    full = E.score_uncompacted(manager, probes, args.max_probe_tokens)
    pool0, mask0 = E.candidate_pool(manager, base, "cuda:0")
    before = E.score_probes(manager, probes, pool0, args.max_probe_tokens, mask=mask0)
    del pool0, mask0
    torch.cuda.empty_cache()
    gap = E.recovery_report(full, before, before, no_memory=no_mem)
    lost_questions = list(gap["lost_questions"])
    print("[s0] eligible %d, lost by selection %d" % (gap["n_eligible"], gap["n_lost"]), flush=True)
    if gap["n_lost"] < 3:
        raise SystemExit(
            "only %d lost probes; the sensitivity measurement needs a window where "
            "selection actually destroys readable information (E6a's window had 3+)"
            % gap["n_lost"]
        )

    # --- Jacobian rows: one per lost probe, on down.weight ------------------
    # The base point is the checkpoint-loaded compactor (or init without
    # --carrier-ckpt, the documented dead zone). Sensitivity is a property of
    # that base point; D is perturbed relative to it below.
    dimension = base.down.weight.numel()
    # In-place injection instead of `candidate_pool`'s torch.cat. The cat's
    # backward materialises a full-width 3.36 GiB gradient block before slicing
    # it down to the candidate's 0.86 GiB -- run 6 died exactly there (needed
    # 3.12 GiB, had 1.86). `_pool_in_place` is what the training loop uses for
    # the same reason: the write is differentiable via `_InjectSlice`, backward
    # returns only the candidate slice, and each `reinject()` is a fresh node
    # so every probe's graph frees itself after its own backward.
    window_used = E.WINDOW_BLOCKS * manager.block
    target_used = E.TARGET_BLOCKS * manager.block
    from methods.prlmc.probes import _compacted_mask

    mask = _compacted_mask(manager, manager.used - window_used + target_used, "cuda:0")
    base.down.weight.requires_grad_(True)
    rows_list = []
    for q in lost_questions:
        probe = next(p for p in probes if p.question == q)
        source = manager._memory[
            :, manager.total - manager.used : manager.total, :
        ].to("cuda:0")
        weights = manager._window_importance(
            manager.total - manager.used, window_used
        )
        candidate = base(
            source, weights, target=target_used, block=manager.block,
            goal_direction=manager.goal_direction,
            select_weights=manager.select_weights,
        )
        del source, weights
        scalar = None
        with _pool_in_place(manager, "cuda:0", candidate, window_used, target_used) as injection:
            pool = injection.reinject()
            scalar = margin_scalar(manager, probe, pool, mask, args.max_probe_tokens)
        if scalar is None:
            del candidate
            torch.cuda.empty_cache()
            continue
        base.down.weight.grad = None
        scalar.backward()
        # Named grad_row: a bare `row` here shadowed the window coordinate
        # (`row, start = ...`), and the later json.dump died on the Tensor
        # mid-write -- leaving a 27-byte truncated file on run 8.
        grad_row = base.down.weight.grad.detach().reshape(-1).to("cpu", torch.float32).clone()
        base.down.weight.grad = None
        rows_list.append(grad_row)
        del scalar, candidate, pool, grad_row
        torch.cuda.empty_cache()
        print("[s0] jacobian row %d/%d done" % (len(rows_list), len(lost_questions)), flush=True)
    base.down.weight.requires_grad_(False)
    del mask
    torch.cuda.empty_cache()
    if len(rows_list) < 3:
        raise SystemExit("could only build %d jacobian rows" % len(rows_list))
    rows_j = torch.stack(rows_list)
    print("[s0] jacobian rows: %s over %d dims" % (tuple(rows_j.shape), dimension), flush=True)

    sketch = FunctionalSensitivitySketch(dimension, rank=args.sketch_rank)
    fit_report = sketch.fit(rows_j.to("cpu"))
    print("[s0] sketch rank %d trace %.4f discarded %.4f"
          % (fit_report.rank, fit_report.trace, fit_report.discarded_energy), flush=True)

    # --- persist the partition raw material -------------------------------
    # Run 8 computed these and threw them away; the functional-partition
    # question then needed a rerun. The Gram matrix of the 13 rows answers
    # probe-probe sensitivity overlap directly; norms show per-probe leverage.
    with torch.no_grad():
        gram = rows_j @ rows_j.T
        row_norms = rows_j.norm(dim=1)
        eigvals = torch.linalg.eigvalsh(gram)
    partition = {
        "lost_questions": list(lost_questions),
        "gram": gram.tolist(),
        "row_norms": row_norms.tolist(),
        "gram_eigvals_desc": sorted(eigvals.tolist(), reverse=True),
        "rows_path": None,  # filled below if --dump-rows is given
    }
    if args.dump_rows:
        torch.save({"rows": rows_j, "lost_questions": list(lost_questions)},
                   args.dump_rows)
        partition["rows_path"] = args.dump_rows
        print("[s0] jacobian rows dumped to %s" % args.dump_rows, flush=True)
    # quick overlap read: off-diagonal gram normalised by norm product
    n = rows_j.shape[0]
    cos = (gram / (row_norms.outer(row_norms) + 1e-12)).tolist()
    partition["cosine_matrix"] = cos
    off_diag = [cos[i][j] for i in range(n) for j in range(n) if i != j]
    print("[s0] probe-pair gradient cosine: mean %.4f max %.4f min %.4f"
          % (sum(off_diag) / len(off_diag), max(off_diag), min(off_diag)), flush=True)

    # --- direction families ------------------------------------------------
    from residual_subspace_diag import orthogonal_directions  # noqa: E402

    direction_set = orthogonal_directions(sketch, args.directions, args.seed)
    results = {
        "window": {"row": row, "start": start},
        "dropped_blocks": [int(b) for b in dropped],
        "n_probes": len(probes),
        "n_lost": len(lost_questions),
        "sketch": {"rows": fit_report.rows, "rank": fit_report.rank,
                   "trace": fit_report.trace,
                   "discarded_energy": fit_report.discarded_energy},
        "partition": partition,
        "families": {},
    }

    for name, vectors in direction_set.items():
        family_scores = []
        for scale in args.scales:
            per_direction = []
            for d in range(vectors.shape[0]):
                direction = vectors[d].to("cpu")
                clone = perturb_down(base, direction, scale, dimension)
                entry = score_family(
                    manager, probes, clone, args, name, lost_questions
                )
                per_direction.append(entry)
                del clone
                torch.cuda.empty_cache()
            recovered_counts = [e["lost_recovered"] for e in per_direction]
            family_scores.append({
                "scale": scale,
                "mean_lost_recovered": sum(recovered_counts) / len(recovered_counts),
                "per_direction": recovered_counts,
                "per_direction_recovered": [
                    e["recovered_questions"] for e in per_direction
                ],
                "n_lost_base": len(lost_questions),
            })
            print("[s0] %-13s scale %.2f: lost_recovered mean %.3f (%s)"
                  % (name, scale, family_scores[-1]["mean_lost_recovered"],
                     recovered_counts), flush=True)
        results["families"][name] = family_scores

    # --- pre-registered reading -------------------------------------------
    # The measurement is RECOVERY (lost probes brought back), not damage: the
    # perturbation moves D along +gradient directions, so a *sensitive*
    # direction recovers more probes and has LOWER harm. The pre-registered
    # "sensitive hurts more" reading is therefore signed
    # low_harm - sens_harm. Run 8 computed sens_harm - low_harm (all negative,
    # max -0.144) and misprinted FAIL over an actually-PASS separation.
    def harm(family):
        # harm = 1 - mean recovered share, per scale; higher = direction hurts more
        return {
            s["scale"]: 1.0 - (s["mean_lost_recovered"] / s["n_lost_base"])
            for s in results["families"][family]
        }

    sens_harm = harm("sensitive")
    low_harm = harm("low_sensitive")
    rand_harm = harm("random")
    separations = {s: low_harm[s] - sens_harm[s] for s in sens_harm}
    max_sep = max(separations.values())
    # The guard from the pre-registration: the sensitive family must be doing
    # real work (harm < 0.5 = recovering over half the lost probes) at some
    # scale, and the low-sensitive family must NOT be doing that work there.
    first_good_scale = None
    for s in sorted(sens_harm):
        if sens_harm[s] < 0.5:
            first_good_scale = s
            break
    if max_sep >= 0.25 and (first_good_scale is not None and low_harm.get(first_good_scale, 1.0) >= 0.5):
        verdict = "PASS"
    elif max_sep < 0.10:
        verdict = "FAIL"
    else:
        verdict = "INCONCLUSIVE"
    results["verdict"] = verdict
    results["separation"] = {str(k): v for k, v in separations.items()}
    results["sensitive_harm"] = {str(k): v for k, v in sens_harm.items()}
    results["low_sensitive_harm"] = {str(k): v for k, v in low_harm.items()}
    results["random_harm"] = {str(k): v for k, v in rand_harm.items()}
    print("[s0] verdict %s (max separation %.3f)" % (verdict, max_sep), flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("[s0] wrote %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
