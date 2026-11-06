#!/usr/bin/env python3
"""Experiment 2, training half: teach the 360 graft slots to carry more than they copy.

What is trained
---------------
Exactly one module (`methods.prlmc.source_compaction.SourceCompaction`, 0.52 M
parameters at rank 32). Reader, writer and the V10 graft operator are frozen;
nothing else in the system has a gradient.

What the objective is
--------------------
Real answer cross-entropy, plus a knowledge-protection term. Experiment 1B
supervised the reader-readout distance and watched it fall 20% while accuracy fell
2.5 pp, so the readout distance is not a training target here. Each step:

1. write the 19-block window and compress it with the frozen V10 operator (native
   mode): the 360 graft slots are verbatim copies of the dropped blocks;
2. measure the frozen reader's CE on the gold answer from THAT window (no grad);
3. add the module's residual to the 360 graft slots and measure the CE again;
4. minimise `CE(compacted) + protect_weight * KL(compacted || graft)`.

The KL term is the knowledge protection: it is what stops the module from trading
a fact the Graft window already answers for one it might add. It is a regulariser,
not a target -- the CE is what moves.

Why the gradient reaches the module
-----------------------------------
The reader is frozen, so the only route to d(CE)/d(graft slots) is backprop
THROUGH it. `methods.prlmc.probes` publishes the three helpers V10's own training
uses: `_pool_in_place` writes the candidate window into the reader's pool behind a
custom autograd Function, `active_mask_override` makes the reader attend to exactly
the 2,304 kept slots, and `differentiable_memory` routes the forward pass to that
pool. The first backwards are checked, not assumed: update 1 must deliver gradient
to `out_proj` (the only parameter that can move while the residual is still zero)
and update 2 must deliver gradient to every projection. If it does not, the run
stops instead of training a silently detached module.

Why the dropout is zeroed, and why the reader is checkpointed
-----------------------------------------------------------
The backward needs the full-width memory gradient (3.12 GiB for a 50-block pool at
32 layers, fp16), which does not fit next to the activations on a 40 GB card
unless the reader is gradient-checkpointed -- and this fork gates checkpointing on
`self.training`. Train mode, however, is not deterministic here: the reader
carries 384 LoRA dropout modules at p = 0.1 and `tools/train.py`'s zero-dropout
guard only inspects `config.attention_dropout`, so it misses them. Measured on
2026-11-05, two identical forwards over the same pool:

    eval mode             CE 0.877455  0.877455   deterministic
    train mode            CE 0.880165  0.840025   CUDA RNG advances
    train mode, p = 0     CE 0.877455  0.877455   deterministic, == eval
    train mode, p = 0 + checkpointing
                          CE 0.877455  0.877455   unchanged

So the script zeroes every Dropout, THEN switches to train mode and enables
checkpointing: determinism and the memory saving come together, and the training
forward stays bit-identical to the eval-mode forward the module is scored with.

Training data
-------------
A DISJOINT task, by default LongBench musique. HotpotQA, the evaluation task, is
never written during training, and the split is recorded in the report.

Usage
-----
    python tools/train_source_compaction.py --dry-run \
        --data-dir /data/.../LongBench/extracted/data --train-task musique

    python tools/train_source_compaction.py --smoke \
        --data-dir /data/.../LongBench/extracted/data --train-task musique \
        --model-path /dev/shm/s1d_reader_fp16 \
        --checkpoint runs/natural_merge_v9/full_v10.pt \
        --out runs/source_compaction/smoke.pt \
        --report runs/source_compaction/smoke.json

    python tools/train_source_compaction.py --updates 384 \
        --data-dir /data/.../LongBench/extracted/data --train-task musique \
        --model-path /dev/shm/s1d_reader_fp16 \
        --checkpoint runs/natural_merge_v9/full_v10.pt \
        --out runs/source_compaction/compaction_rank32.pt \
        --report runs/source_compaction/compaction_rank32.json
"""
import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_graft_recovery as gr                          # noqa: E402
import native_eval as nv                                  # noqa: E402
from io_utils import atomic_json, sha                     # noqa: E402
from runtime import critical_manifest, load_runtime, reset  # noqa: E402
from methods.prlmc.probes import (                        # noqa: E402
    _checkpoint_func, _compacted_mask, _pool_in_place, active_mask_override,
    differentiable_memory)
from methods.prlmc.source_compaction import (             # noqa: E402
    SCHEMA, SourceCompaction, compact_anchors, split_source_anchors)
from objective import answer_logits                       # noqa: E402
from train_core_fusion import (                           # noqa: E402
    BUDGET_BLOCKS, TARGET_BLOCKS, WINDOW_BLOCKS, chunks_of, load_documents,
    write_blocks)

SLOTS_PER_BLOCK = gr.SLOTS_PER_BLOCK
WINDOW_SLOTS = WINDOW_BLOCKS * SLOTS_PER_BLOCK
TARGET_SLOTS = TARGET_BLOCKS * SLOTS_PER_BLOCK
# `out_proj` is the only parameter with a non-zero gradient while the residual is
# still exactly zero; q/k/v only start moving after it leaves zero.
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "out_proj")


# ---------------------------------------------------------------------------
# one document
# ---------------------------------------------------------------------------
class _ScatterSlots(torch.autograd.Function):
    """Write `slots` into `pool` at scattered `positions`, keeping the graph.

    `_pool_in_place` injects a whole contiguous window, so it needs a 2,304-slot
    candidate AND its gradient even when only the 360 graft slots carry one: 604 MB
    each way at 32 layers fp16. This writes just those 360 slots (94 MB each way)
    and routes exactly their gradient, which is the difference between fitting on a
    40 GB card and not.
    """

    @staticmethod
    def forward(ctx, pool, slots, positions):
        ctx.positions = positions
        pool[:, positions, :] = slots.to(pool.dtype)
        return pool

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output[:, ctx.positions, :].clone(), None


@contextmanager
def installed(model, manager, output, anchors=None, anchor_offsets=None):
    """Install the compressed window in the reader's pool for one forward.

    The frozen window is written in place under no_grad; the 360 compacted anchors
    go through `_ScatterSlots` so their gradient survives. The displaced slots are
    saved to host memory and restored on exit, including on exception, the same way
    `_PoolInjection` does for V10's objective.
    """
    base = manager.model.memory
    start = int(manager.total) - (int(manager.used) - WINDOW_SLOTS + TARGET_SLOTS)
    saved = base[:, start:start + TARGET_SLOTS, :].clone().cpu()
    mask = _compacted_mask(manager, int(manager.used) - WINDOW_SLOTS + TARGET_SLOTS,
                           base.device)
    try:
        with torch.no_grad():
            base[:, start:start + TARGET_SLOTS, :] = output
        pool = base
        if anchors is not None:
            pool = _ScatterSlots.apply(base, anchors, start + anchor_offsets)
        with active_mask_override(model, mask):
            with differentiable_memory(model, pool):
                yield
    finally:
        with torch.no_grad():
            base[:, start:start + TARGET_SLOTS, :] = saved.to(base.device)


def reader_logits(model, manager, output, anchors, anchor_offsets, qa):
    """One reader forward on the QA pair with the compressed window installed."""
    with installed(model, manager, output, anchors, anchor_offsets):
        return answer_logits(model, qa, output.device)


def document_step(model, manager, merge, module, initial, document, args, train):
    """Write, compress, compact, and score the gold answer on one document."""
    reset(manager, initial)
    with torch.no_grad():
        write_blocks(manager, merge, chunks_of(document))
        source, _, picked = gr.select_carriers(manager, WINDOW_BLOCKS, TARGET_BLOCKS)
        frozen, pinned = gr.window_locks(manager, WINDOW_BLOCKS)
        output, telemetry = gr.compress_graft(manager, merge, source, picked,
                                              WINDOW_BLOCKS, TARGET_BLOCKS,
                                              frozen, pinned)
    parts = split_source_anchors(telemetry, SLOTS_PER_BLOCK, WINDOW_BLOCKS,
                                 device=output.device)
    qa = dict(prompt_ids=document["query_ids"], answer_ids=document["answer_ids"])
    # Built BEFORE any injection: `source` is a view into the reader's pool and the
    # injection writes into that same pool, so the candidates have to be gathered
    # first.
    anchors, info = compact_anchors(output, source, parts, module)
    offsets = parts["anchor_offsets"]

    if not train:
        with torch.no_grad():
            graft_logits, gold = reader_logits(model, manager, output, None, None, qa)
            graft_ce = float(F.cross_entropy(graft_logits, gold))
            del graft_logits
            logits, _ = reader_logits(model, manager, output, anchors, offsets, qa)
            ce = float(F.cross_entropy(logits, gold))
        return dict(ce=ce, graft_ce=graft_ce, removed=1.0 - ce / max(graft_ce, 1e-9),
                    **info)

    graft_log_prob = None
    graft_ce = None
    if args.protect_weight > 0:
        with torch.no_grad():
            graft_logits, graft_gold = reader_logits(model, manager, output, None,
                                                     None, qa)
            graft_ce = F.cross_entropy(graft_logits, graft_gold)
            graft_log_prob = F.log_softmax(graft_logits.float(), dim=-1)
            del graft_logits, graft_gold
    logits, gold = reader_logits(model, manager, output, anchors, offsets, qa)
    ce = F.cross_entropy(logits, gold)
    protection = torch.zeros((), device=ce.device)
    if graft_log_prob is not None:
        protection = F.kl_div(F.log_softmax(logits.float(), dim=-1),
                              graft_log_prob.exp(), reduction="batchmean")
        del graft_log_prob
    return dict(loss=ce + args.protect_weight * protection, ce=ce,
                graft_ce=(graft_ce if graft_ce is not None
                          else torch.zeros((), device=ce.device)),
                protection=protection, **info)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def dry_run(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    documents, counts = load_documents(tokenizer, args.data_dir, args.train_task,
                                       args.max_docs)
    if not documents:
        raise SystemExit(f"no {args.train_task} document fills a {WINDOW_BLOCKS}-block "
                         f"window")
    print(json.dumps(dict(
        dry_run=True, train_task=args.train_task, counts=counts,
        updates=args.updates, steps_per_doc=args.steps_per_doc,
        epochs=round(args.updates * args.steps_per_doc / len(documents), 2),
        val_docs=min(args.val_docs, max(0, len(documents) - 1)),
        geometry=dict(window_blocks=WINDOW_BLOCKS, target_blocks=TARGET_BLOCKS,
                      core_slots=TARGET_BLOCKS * gr.CORE_SLOTS,
                      graft_slots=TARGET_BLOCKS * gr.GRAFT_SLOTS,
                      window_slots=WINDOW_SLOTS, target_slots=TARGET_SLOTS),
        module=dict(rank=args.rank, max_ratio=args.max_ratio, gate_init=args.gate_init,
                    hidden=4096, parameters=4 * args.rank * 4096 + 1,
                    initial_residual="exactly zero (out_proj starts at zero)"),
        objective=("answer cross-entropy of the compacted window + "
                   "protect_weight * KL(compacted || graft)"),
        answers_seen=True,
        data_note=("training documents come from a different LongBench task than the "
                   "HotpotQA evaluation rows")), indent=2, ensure_ascii=False))
    return 0


def train(args):
    # Set before the first CUDA allocation: the compaction step holds the 3.4 GB
    # pool plus its 3.4 GB gradient plus the 604 MB window, and the fragmentation
    # between those three is what turns a near-fit into an OOM.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if not args.checkpoint or not Path(args.checkpoint).is_file():
        raise SystemExit("--checkpoint (the frozen V10 operator) is required")
    if Path(args.out).exists():
        raise SystemExit(f"{args.out} exists; refuse overwrite")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    documents, counts = load_documents(tokenizer, args.data_dir, args.train_task,
                                       args.max_docs)
    if len(documents) < 4:
        raise SystemExit(f"only {len(documents)} usable {args.train_task} documents")
    generator = np.random.RandomState(args.seed)
    order = generator.permutation(len(documents))
    val_count = min(args.val_docs, len(documents) // 4)
    validation = [documents[index] for index in order[:val_count]]
    training = [documents[index] for index in order[val_count:]]
    if args.smoke:
        training, validation = training[:2], validation[:2]

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.cuda_device))
    nv.seed_everything(args.seed)
    checkpoint_training = torch.load(args.checkpoint, map_location="cpu").get("training")
    model, tokenizer, manager, merge, initial, cfg, saved = load_runtime(
        args.model_path, checkpoint=args.checkpoint, train=False, architecture=None,
        budget_blocks=BUDGET_BLOCKS, strict_code_hash=args.strict_code_hash,
        training=checkpoint_training)
    merge.operator_mode = "native"
    # Zero the LoRA dropout BEFORE switching to train mode, otherwise the forward
    # that produces the gradient is not the forward the module is scored with (see
    # the docstring for the measurement). Train mode is only needed for gradient
    # checkpointing, which is what makes the 3.12 GiB memory gradient fit.
    for dropout in model.modules():
        if isinstance(dropout, nn.Dropout):
            dropout.p = 0.0
    model.train()
    if not args.no_reader_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.model.gradient_checkpointing = True
        model.model._gradient_checkpointing_func = _checkpoint_func()
    module = SourceCompaction(hidden_size=manager.hidden, rank=args.rank,
                              max_ratio=args.max_ratio,
                              gate_init=args.gate_init).to(model.memory.device).float()
    optimizer = torch.optim.Adam(module.parameters(), lr=args.lr)

    report = dict(
        schema=SCHEMA + "/training", status="running",
        data=dict(dir=args.data_dir, task=args.train_task,
                  sha256=sha(Path(args.data_dir) / (args.train_task + ".jsonl")),
                  counts=counts, train_documents=len(training),
                  validation_documents=len(validation),
                  validation_ids=[document["id"] for document in validation],
                  evaluation_task_never_written="hotpotqa"),
        reader=dict(model_path=args.model_path, frozen=True,
                    checkpoint=args.checkpoint, checkpoint_sha256=sha(args.checkpoint),
                    checkpoint_code_hash_verified=(saved or {}).get("code_hash_verified"),
                    operator_mode="native"),
        module=dict(schema=SCHEMA, rank=args.rank, max_ratio=args.max_ratio,
                    gate_init=args.gate_init, hidden=manager.hidden,
                    parameters=int(sum(p.numel() for p in module.parameters())),
                    anchors="the 360 graft slots, copied verbatim from the dropped "
                            "blocks by the frozen operator",
                    candidates="every unselected slot of each anchor's own source block",
                    initialization="q/k/v orthogonal, out_proj zero: the compacted "
                                   "window starts bit-identical to the grafted one"),
        objective=dict(loss="answer cross-entropy of the compacted window",
                       protection=f"protect_weight * KL(compacted || graft)",
                       protect_weight=args.protect_weight, answers_seen=True),
        optimizer=dict(name="Adam", lr=args.lr, grad_clip=args.grad_clip),
        updates=args.updates, steps_per_doc=args.steps_per_doc,
        critical_manifest=critical_manifest(), history=[], validation=[],
        peak_allocated_bytes=0)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.report, report)

    # A zero output projection must reproduce the grafted window EXACTLY. This is
    # the whole "the two arms differ only by the compaction" claim, so it is
    # measured once before the first update instead of being assumed.
    with torch.no_grad():
        reset(manager, initial)
        write_blocks(manager, merge, chunks_of(training[0]))
        source, _, picked = gr.select_carriers(manager, WINDOW_BLOCKS, TARGET_BLOCKS)
        frozen, pinned = gr.window_locks(manager, WINDOW_BLOCKS)
        output, telemetry = gr.compress_graft(manager, merge, source, picked,
                                              WINDOW_BLOCKS, TARGET_BLOCKS,
                                              frozen, pinned)
        parts = split_source_anchors(telemetry, SLOTS_PER_BLOCK, WINDOW_BLOCKS,
                                     device=output.device)
        anchors, identity_info = compact_anchors(output, source, parts, module)
        identity_same = bool(torch.equal(
            anchors, output.index_select(1, parts["anchor_offsets"])))
    if not identity_same:
        raise ValueError("the fresh module is not the identity: the compacted window "
                         "differs from the grafted one before any update")
    identity = document_step(model, manager, merge, module, initial, training[0],
                             args, train=False)
    if abs(identity["ce"] - identity["graft_ce"]) > 1e-6:
        raise ValueError(f"the fresh module changes the answer CE: compacted "
                         f"{identity['ce']} != grafted {identity['graft_ce']}")
    report["identity_check"] = dict(
        document=training[0]["id"], window_bit_identical=identity_same,
        ce=identity["ce"], graft_ce=identity["graft_ce"],
        anchor_slots=identity["anchor_slots"],
        candidate_slots=identity["candidate_slots"], passed=True)
    print(json.dumps({"identity_check": report["identity_check"]}), flush=True)

    started = time.monotonic()
    update = 0
    while update < args.updates:
        document = training[update % len(training)]
        for _ in range(args.steps_per_doc):
            if update >= args.updates:
                break
            step_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            measured = document_step(model, manager, merge, module, initial, document,
                                     args, train=True)
            measured["loss"].backward()
            gradients = {name: parameter.grad for name, parameter in
                         module.named_parameters() if parameter.grad is not None}
            update += 1
            # Measured, not assumed: the first update can only reach `out_proj`
            # (the residual is still exactly zero, so q/k/v sit downstream of a
            # zero matrix), and update 2 is the first one that must reach all four.
            required = ("out_proj",) if update == 1 else PROJECTIONS
            missing = [name for name in required
                       if name not in gradients
                       or not torch.isfinite(gradients[name]).all()
                       or float(gradients[name].abs().max()) == 0.0]
            if missing:
                raise ValueError(f"no gradient on {missing} at update {update}: the "
                                 f"frozen reader's backward did not reach the window")
            norm = float(torch.nn.utils.clip_grad_norm_(module.parameters(),
                                                        args.grad_clip))
            optimizer.step()
            row = dict(update=update, document=document["id"],
                       loss=float(measured["loss"].detach()),
                       ce=float(measured["ce"].detach()),
                       graft_ce=float(measured["graft_ce"].detach()),
                       removed=float(1.0 - measured["ce"].detach()
                                     / measured["graft_ce"].detach().clamp_min(1e-9)),
                       protection=float(measured["protection"].detach()),
                       grad_norm=norm, seconds=time.monotonic() - step_started,
                       gate=measured["gate"], delta=measured["relative_delta"],
                       grad_out_proj=float(gradients["out_proj"].norm())
                       if "out_proj" in gradients else None,
                       grad_gate=float(gradients["gate_logit"].norm())
                       if "gate_logit" in gradients else None)
            report["history"].append(row)
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                              for k, v in row.items()}), flush=True)
            if args.val_every and validation and (update % args.val_every == 0
                                                  or update == args.updates):
                scores = [document_step(model, manager, merge, module, initial,
                                        held, args, train=False)
                          for held in validation]
                summary = dict(
                    update=update,
                    ce=float(np.mean([score["ce"] for score in scores])),
                    graft_ce=float(np.mean([score["graft_ce"] for score in scores])))
                summary["removed"] = (1.0 - summary["ce"] / summary["graft_ce"]
                                      if summary["graft_ce"] else None)
                report["validation"].append(summary)
                print(json.dumps({key: (round(value, 5) if isinstance(value, float)
                                        else value)
                                  for key, value in summary.items()}), flush=True)
        if args.report:
            report["elapsed_seconds"] = time.monotonic() - started
            atomic_json(args.report, report)

    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    report["final_gate"] = float(torch.sigmoid(module.gate_logit.detach()))
    report["status"] = "complete"
    payload = dict(
        schema=SCHEMA + "/checkpoint", state_dict=module.state_dict(),
        config=dict(hidden_size=manager.hidden, rank=args.rank,
                    max_ratio=args.max_ratio, gate_init=args.gate_init),
        training=dict(task=args.train_task, updates=args.updates, lr=args.lr,
                      seed=args.seed, protect_weight=args.protect_weight,
                      data_sha256=report["data"]["sha256"],
                      checkpoint=args.checkpoint,
                      checkpoint_sha256=report["reader"]["checkpoint_sha256"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    report["checkpoint"] = args.out
    report["checkpoint_sha256"] = sha(args.out)
    if args.report:
        atomic_json(args.report, report)
    print(json.dumps(dict(status="complete", updates=update, out=args.out,
                          final_gate=report["final_gate"],
                          validation=report["validation"][-1:]
                          if report["validation"] else [],
                          elapsed_seconds=round(report["elapsed_seconds"], 1)),
                     indent=1, ensure_ascii=False))
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--train-task", dest="train_task", default="musique")
    p.add_argument("--model-path", default=None)
    p.add_argument("--checkpoint", default=None,
                   help="the frozen V10 operator (runs/natural_merge_v9/full_v10.pt)")
    p.add_argument("--out", default=None, help="compaction checkpoint to write")
    p.add_argument("--report", default=None, help="training report to write")
    p.add_argument("--updates", type=int, default=384)
    p.add_argument("--steps-per-doc", dest="steps_per_doc", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--protect-weight", dest="protect_weight", type=float, default=0.1,
                   help="weight of KL(compacted || graft); 0 disables the guard")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--max-ratio", dest="max_ratio", type=float, default=0.5)
    p.add_argument("--gate-init", dest="gate_init", type=float, default=0.0)
    p.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0)
    p.add_argument("--no-reader-checkpointing", dest="no_reader_checkpointing",
                   action="store_true",
                   help="only useful on a card with the headroom: without it the "
                        "3.12 GiB memory gradient does not fit")
    p.add_argument("--val-docs", dest="val_docs", type=int, default=8)
    p.add_argument("--val-every", dest="val_every", type=int, default=48)
    p.add_argument("--max-docs", dest="max_docs", type=int, default=None)
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strict-code-hash", dest="strict_code_hash", default="true",
                   choices=("true", "false"))
    p.add_argument("--smoke", action="store_true",
                   help="2 training documents, 2 validation documents")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return dry_run(args) if args.dry_run else train(args)


if __name__ == "__main__":
    raise SystemExit(main())
