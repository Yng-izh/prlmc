"""SIC probe driver: G1 minimal closed loop for SDC (S1-SDC section 7 step 2).

G1 asks one question: does functional-saliency selection lose fewer probes than
native similarity selection, everything else held fixed?

Protocol (pre-registered, S1-SDC.md section 5):

  window   one live 19-block append-only window per row (E6a/E6b path)
  probes   the E6a probe set (gold answer inside the window, supporting fact
           in a *selection-dropped* block, so the gap is measurable)
  arms     select   native ``_select_block_indices`` (the baseline to beat)
           sic      top-9 by accumulated ``block_salience`` (G1's question)
           sicm     sic + ``merge_into_survivors`` (the full operator; its
                    first-order guarantee is audited by the probe-flip count)
           drop-oldest / drop-random   controls
  metric   lost = eligible probes the arm gets wrong; recovered = lost the arm
           gets right. Eligible = full-pool correct AND no-memory wrong (the
           E6a ``recovery_report`` definition, reused verbatim).
  verdict  G1 passes when sic's pooled lost is <= 0.7x select's pooled lost
           over >= 3 contexts.

How the saliency gradient is taken (the S0 path, adapted):

  The pool is written once per window. For each probe, the gold-answer
  cross-entropy is backpropagated to the *window slice of model.memory* -- not
  to compactor parameters. The reader is frozen; ``model.memory`` is the only
  tensor with requires_grad. Each probe's [L, K, d] window gradient feeds
  ``block_salience``, accumulated in fp32.

  For the merge arm, the same backward pass also deposits the gradient of the
  *survivor* blocks into per-(block, layer) ``IncrementalBasis`` objects: U in
  B' = B + (I - U U^T) f(A) is the span of exactly these gradients, which is
  what makes <dL/dB, dB> = 0 constructive rather than checked after the fact.

Memory discipline (V100-32, reader 16 GB + pool 3.4 GB):
  * gradient checkpointing on the reader (train() with all params frozen --
    the S0 finding that eval() silently skips the checkpointing branch);
  * one probe per backward, graph freed immediately;
  * the per-probe window gradient [32, 4864, 4096] fp32 is ~2.4 GB: it is
    reduced to ``block_salience`` (and, for merge arms, the survivor slices)
    inside the same ``with`` scope and deleted before the next probe.

Run on v100-2 (one window at a time, sequential over rows):
  /share/latent/mllm_env/bin/python /share/latent/PRLM-C/tools/sic_probe.py \
      --rows 0,1,2,3 --out /share/latent/PRLM-C/runs/g1_report.json
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- GPU pick before torch import (same guard as every tool here) -----------

if __name__ == "__main__" and not os.environ.get("SIC_GPU_SET"):
    need_mb = int(os.environ.get("SIC_NEED_MB", "21000"))
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
            "no GPU with %d MB free; best is gpu%d with %d MB"
            % (need_mb, free[0][1], free[0][0])
        )
    print("[sic] gpu%d, %d MB free" % (free[0][1], free[0][0]), flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(free[0][1])
    os.environ["SIC_GPU_SET"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import e6a_overfit as E  # noqa: E402 (window build, probe set, scoring, recovery)

from methods.prlmc.sic import (  # noqa: E402
    IncrementalBasis,
    block_salience,
    merge_into_survivors,
    salience_select,
)
from methods.prlmc.consolidate import _select_block_indices  # noqa: E402

DEVICE = "cuda:0"


def build_state(manager, rows, row, start, args):
    """One fresh window + probe set + the selection baseline's dropped set."""
    window_slots = E.WINDOW_BLOCKS * manager.block
    covered = E.write_until(manager, rows[row], window_slots, start)
    if covered is None:
        return None
    line_to_block = {}
    for block_index, (lo, hi, _before, _width) in enumerate(covered):
        for line in range(lo, hi):
            line_to_block[line] = block_index
    indices = E.probe_indices(covered, window_slots)
    begin = manager.total - manager.used
    source = manager._memory[:, begin: begin + window_slots, :].to(DEVICE)
    weights = manager._window_importance(begin, window_slots)
    kept_native = _select_block_indices(
        source, E.TARGET_BLOCKS * manager.block, manager.block,
        weights, manager.goal_direction, manager.select_weights,
    ).tolist()
    dropped_native = sorted(set(range(E.WINDOW_BLOCKS)) - set(kept_native))
    del source, weights
    torch.cuda.empty_cache()
    probes = E.probes_from_lines(
        rows[row], indices, args.probes,
        dropped_blocks=set(dropped_native), line_to_block=line_to_block,
    )
    return {
        "covered": covered, "indices": indices, "line_to_block": line_to_block,
        "kept_native": kept_native, "dropped_native": dropped_native,
        "probes": probes,
    }


def pool_from_blocks(manager, blocks, device=DEVICE):
    """Assemble the reader's pool from a temporal-order list of window blocks.

    Same placement as ``compressed_pool``: candidate flush against the trailing
    end, freed region masked out. Survivors are the exact bytes the reader
    produced; only ``merge_into_survivors`` output ever differs.
    """
    window_slots = E.WINDOW_BLOCKS * manager.block
    target = len(blocks) * manager.block
    source = manager._memory[:, manager.total - manager.used:
                             manager.total - manager.used + window_slots, :]
    candidate = torch.stack(
        [source[:, j * manager.block:(j + 1) * manager.block, :] for j in blocks],
        dim=1,
    ).reshape(manager.num_layers, target, manager.hidden)
    from methods.prlmc.probes import _compacted_mask, _compacted_pool

    pool = _compacted_pool(manager, device, candidate, window_slots, target)
    mask = _compacted_mask(manager, manager.used - window_slots + target, device)
    return pool, mask


@torch.no_grad()
def reference_scores(manager, state, args):
    """The three measurements every arm is judged against, on the live pool."""
    probes = state["probes"]
    no_mem = E.score_no_memory(manager, probes, args.max_probe_tokens)
    full = E.score_uncompacted(manager, probes, args.max_probe_tokens)
    return no_mem, full


def probe_loss_scalar(manager, probe, pool, mask, max_probe_tokens):
    """Gold-answer CE of one probe under `pool`, grads flowing into the pool.

    The S0 path with one substitution: the differentiated tensor is the
    substituted pool itself (``differentiable_memory`` keeps the graph), so
    ``pool.grad`` accumulates the window gradient. The pool tensor is created
    by the caller as a leaf with requires_grad.
    """
    device = pool.device if pool is not None else manager.model.memory.device
    stem = probe.prompt if probe.prompt else f"{probe.question}\nAnswer:"
    answer = probe.answers[0]
    encoded = manager.tokenizer(
        f"{stem} {answer}", return_tensors="pt", add_special_tokens=False
    )
    input_ids = encoded.input_ids[:, :max_probe_tokens].to(device)
    stem_len = len(manager.tokenizer(stem, add_special_tokens=False).input_ids)
    if stem_len >= max_probe_tokens:
        raise ValueError(
            "max_probe_tokens=%d truncates the %d-token prompt"
            % (max_probe_tokens, stem_len)
        )
    if input_ids.shape[1] < 2 or stem_len <= 0 or stem_len >= input_ids.shape[1]:
        return None
    from methods.prlmc.probes import active_mask_override, differentiable_memory

    with differentiable_memory(manager.model, pool), active_mask_override(manager.model, mask):
        logits = manager.model(input_ids=input_ids, return_dict=True).logits
    span = logits[:, stem_len - 1: -1, :]
    labels = input_ids[:, stem_len:]
    return F.cross_entropy(span.reshape(-1, span.shape[-1]).float(), labels.reshape(-1))


def harvest_gradients(manager, state, args):
    """Backprop every probe into a window-grad accumulator + survivor bases.

    Two passes are not needed: one backward per probe yields the full [L, K, d]
    window gradient, from which both (a) ``block_salience`` per block and
    (b) the survivor slices for the ``IncrementalBasis`` records are read.

    The pool used here is the FULL uncompacted window (the merge's U must be
    the sensitivity at the *survivor's* original bytes, and selection has not
    happened yet at gradient time).
    """
    probes = state["probes"]
    window_slots = E.WINDOW_BLOCKS * manager.block
    begin = manager.total - manager.used
    salience = torch.zeros(E.WINDOW_BLOCKS, dtype=torch.float32, device=DEVICE)
    bases = {}
    if args.arm in ("sicm",):
        for j in state["kept_native"]:
            for layer in range(manager.num_layers):
                bases[(j, layer)] = IncrementalBasis(cap=args.basis_cap)

    # The pool leaf: a clone of the live pool's window region so the write to
    # .grad cannot touch manager storage. `differentiable_memory` swaps this
    # tensor in for the model's parameter during the forward.
    pool = manager._memory.to(DEVICE).clone().requires_grad_(True)
    from methods.prlmc.probes import active_mask_override, differentiable_memory

    mask = None  # the uncompacted pool needs no mask override; the live one stands
    for probe in probes:
        scalar = probe_loss_scalar(manager, probe, pool, mask, args.max_probe_tokens)
        if scalar is None:
            continue
        grad = torch.autograd.grad(scalar, pool)[0]  # [L, total, d]
        window_grad = grad[:, begin: begin + window_slots, :]
        salience += block_salience(window_grad, manager.block).to(DEVICE)
        if args.arm == "sicm":
            for j in state["kept_native"]:
                lo = begin + j * manager.block
                for layer in range(manager.num_layers):
                    bases[(j, layer)].add(window_grad[layer, lo: lo + manager.block, :].reshape(-1))
        del grad, window_grad, scalar
        torch.cuda.empty_cache()
    pool = pool.detach()
    del pool
    torch.cuda.empty_cache()
    return salience, bases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="0,1,2,3")
    parser.add_argument("--start", type=int, default=30)
    parser.add_argument("--probes", type=int, default=30)
    parser.add_argument("--max-probe-tokens", type=int, default=256)
    parser.add_argument("--arm", choices=["select", "sic", "sicm", "oldest", "random"],
                        default="sic")
    parser.add_argument("--basis-cap", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="/tmp/g1_report.json")
    args = parser.parse_args()

    import yaml
    from transformers import AutoTokenizer
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory

    torch.manual_seed(args.seed)

    raw = yaml.safe_load(open(E.CONFIG))
    config = dict(raw["prlmc"])
    config.update(device=DEVICE, dtype="float16", stats_dir="./outputs/prlmc-g1",
                  active_memory_mask=True)
    cfg = PRLMCConfig.from_dict(config)

    model = MemoryLLM.from_pretrained(E.MODEL, torch_dtype=torch.float16,
                                      attn_implementation="sdpa")
    model = model.to(DEVICE)
    # The S0 lesson, applied: gradient checkpointing requires train() mode,
    # every parameter frozen so nothing but the pool leaf moves.
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    tokenizer = AutoTokenizer.from_pretrained(E.MODEL)
    manager = ProtectedResidualMemory(model, tokenizer, cfg)

    rows = {row: E.read_fact_lines(row) for row in
            [int(r) for r in args.rows.split(",")]}
    per_row = []
    pooled = {"select": [0, 0], "sic": [0, 0], "sicm": [0, 0],
              "oldest": [0, 0], "random": [0, 0]}  # [lost, recovered]

    for row in sorted(rows):
        state = build_state(manager, rows, row, args.start, args)
        if state is None:
            print("[sic] row %d: window would not build; skipped" % row, flush=True)
            continue
        probes = state["probes"]
        no_mem, full = reference_scores(manager, state, args)
        eligible = [p.question for p in probes
                    if full["correct"][p.question] and not no_mem["correct"].get(p.question, False)]

        arm_blocks = {}
        arm_blocks["select"] = state["kept_native"]
        arm_blocks["oldest"] = list(range(E.WINDOW_BLOCKS - E.TARGET_BLOCKS,
                                          E.WINDOW_BLOCKS))
        generator = torch.Generator(device="cpu").manual_seed(args.seed + row)
        arm_blocks["random"] = sorted(torch.randperm(E.WINDOW_BLOCKS, generator=generator)[
            : E.TARGET_BLOCKS].tolist())

        salience = None
        if args.arm in ("sic", "sicm"):
            salience, bases = harvest_gradients(manager, state, args)
            arm_blocks["sic"] = salience_select(salience, E.TARGET_BLOCKS)
            print("[sic] row %d salience top9: %s" % (row, arm_blocks["sic"]), flush=True)
            if args.arm == "sicm":
                window_slots = E.WINDOW_BLOCKS * manager.block
                begin = manager.total - manager.used
                source = manager._memory[:, begin: begin + window_slots, :].to(DEVICE)
                candidate, merge_report = merge_into_survivors(
                    source, arm_blocks["sic"], bases,
                    block=manager.block, reference_layer=0,
                )
                del source
                torch.cuda.empty_cache()
        elif args.arm == "sicm":
            raise SystemExit("arm sicm requires gradients; arm must be sic or sicm")

        row_entry = {"row": row, "n_probes": len(probes),
                     "n_eligible": len(eligible), "arms": {}}
        for arm in ("select", "sic", "oldest", "random"):
            blocks = arm_blocks[arm]
            pool, mask = pool_from_blocks(manager, blocks)
            scored = E.score_probes(manager, probes, pool, args.max_probe_tokens, mask=mask)
            del pool, mask
            torch.cuda.empty_cache()
            lost = [q for q in eligible if not scored["correct"].get(q, False)]
            recovered = [q for q in lost if scored["correct"].get(q, False)]
            pooled[arm][0] += len(lost)
            pooled[arm][1] += len(recovered)
            row_entry["arms"][arm] = {"kept": blocks, "lost": lost, "n_lost": len(lost),
                                      "recovered": recovered, "n_recovered": len(recovered)}
            print("[sic] row %d arm %-7s lost %d recovered %d"
                  % (row, arm, len(lost), len(recovered)), flush=True)

        if args.arm == "sicm":
            from methods.prlmc.probes import _compacted_mask
            target = E.TARGET_BLOCKS * manager.block
            window_slots = E.WINDOW_BLOCKS * manager.block
            pool = _compacted_pool(manager, DEVICE, candidate, window_slots, target)
            mask = _compacted_mask(manager, manager.used - window_slots + target, DEVICE)
            scored = E.score_probes(manager, probes, pool, args.max_probe_tokens, mask=mask)
            pre = {p.question: bool(full["correct"][p.question]) for p in probes}
            gate_flips = sorted(q for q, v in pre.items() if v and not scored["correct"].get(q, False))
            row_entry["arms"]["sicm"] = {
                "kept": arm_blocks["sic"], "lost": [q for q in eligible if not scored["correct"].get(q, False)],
                "n_lost": len([q for q in eligible if not scored["correct"].get(q, False)]),
                "flips_vs_full": gate_flips, "merge_report": merge_report,
            }
            del pool, mask, candidate
            torch.cuda.empty_cache()
            print("[sic] row %d arm sicm    lost %d flips %d"
                  % (row, row_entry["arms"]["sicm"]["n_lost"], len(gate_flips)), flush=True)

        row_entry["dropped_native"] = state["dropped_native"]
        row_entry["salience"] = (None if salience is None
                                 else [round(float(v), 6) for v in salience.tolist()])
        per_row.append(row_entry)
        del no_mem, full
        torch.cuda.empty_cache()

    # --- pre-registered G1 verdict ------------------------------------------
    select_lost = pooled["select"][0]
    sic_lost = pooled["sic"][0]
    ratio = (sic_lost / select_lost) if select_lost else float("nan")
    n_rows = len(per_row)
    verdict = ("PASS" if (n_rows >= 3 and select_lost > 0 and ratio <= 0.7)
               else "FAIL" if not (n_rows >= 3 and select_lost > 0)
               else "INCONCLUSIVE")

    report = {
        "design": {"rows": args.rows, "start": args.start, "arm": args.arm,
                   "probes": args.probes, "seed": args.seed,
                   "window_blocks": E.WINDOW_BLOCKS,
                   "target_blocks": E.TARGET_BLOCKS},
        "pooled": {k: {"lost": v[0], "recovered": v[1]} for k, v in pooled.items()},
        "sic_over_select_lost_ratio": round(ratio, 4) if ratio == ratio else None,
        "n_contexts": n_rows,
        "per_row": per_row,
        "verdict": verdict,
    }
    with open(args.out, "w") as f:
        json.dump(report, f)
    print("[sic] pooled lost: select %d, sic %d, ratio %.3f -> %s"
          % (select_lost, sic_lost, ratio if ratio == ratio else -1, verdict), flush=True)
    print("[sic] report %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
