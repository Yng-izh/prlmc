#!/usr/bin/env python3
"""One trainer for every module this project has put on the 360 graft slots.

Three variants, one trainer, so the runs differ only in the module
-------------------------------------------------------------------
    --variant compaction     the experiment-2 module: each anchor attends to the
                             unselected slots of its own source block
                             (`methods.prlmc.source_compaction.SourceCompaction`)
    --variant anchor-only    the same trainer with the candidate path removed: a
                             low-rank map from the anchor to its own residual
                             (`methods.prlmc.anchor_residual.AnchorResidual`)
    --variant anchor-fusion  the trained AnchorResidual, frozen, plus a trainable
                             sparse top-k candidate branch on top
                             (`methods.prlmc.anchor_fusion.FrozenAnchorFusion`)

The rank defaults are chosen so every module has the SAME parameter count
(4 x 32 x 4096 + 1 = 2 x 64 x 4096 + 1 = 4 x 32 x 4096 + 1 = 524,289). Everything
else -- data, updates, learning rate, loss, bounds, seeds -- is shared code, which
is what makes "anchor-only vs compaction vs anchor-fusion" a comparison of modules
and not of scripts.

The third variant is the only one that is not trained from the graft slots: it
starts from the finished anchor-only checkpoint, which is frozen and byte-checked
against the file it was loaded from. Its identity check is therefore different in
one respect -- the fused window must reproduce the ANCHOR-ONLY window before the
first update, not the grafted one.

What is trained
---------------
Exactly one module, 0.52 M trainable parameters. Reader, writer, the V10 graft
operator, and (in the third variant) the anchor module are frozen; nothing else in
the system has a gradient.

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

Step 3 is the only line the variants disagree on, and the disagreement is one
call: `compact_anchors(output, source, parts, module)` against
`compact_anchors(output, parts, module)` -- with or without a window to read. The
third variant calls the first with a module that has the frozen anchor stage
inside it, and additionally scores the frozen stage alone so that the report
carries the anchor-only CE next to the fused CE at every validation point.

The KL term is the knowledge protection: it is what stops the module from trading
a fact the reference window already answers for one it might add. It is a
regulariser, not a target -- the CE is what moves. The reference is the grafted
window for the two variants that start there (where KL is exactly zero at
initialisation), and the frozen AnchorOnly window for the third (where KL against
the graft would be strictly positive and its only easy gradient would be to undo
the calibration the branch is supposed to build on).

Why the gradient reaches the module
-----------------------------------
The reader is frozen, so the only route to d(CE)/d(graft slots) is backprop
THROUGH it. `methods.prlmc.probes` publishes the three helpers V10's own training
uses: `_pool_in_place` writes the candidate window into the reader's pool behind a
custom autograd Function, `active_mask_override` makes the reader attend to exactly
the 2,304 kept slots, and `differentiable_memory` routes the forward pass to that
pool. The first backwards are checked, not assumed: update 1 must deliver gradient
to the output projection (`out_proj`, or `up` in the anchor-only variant, or
`fusion.fuse_up` in the anchor-fusion variant -- the only parameter that can move
while the residual is still zero) and update 2 must deliver gradient to every
projection. The anchor-fusion variant additionally checks that the FROZEN anchor
module received no gradient at all: it is the baseline, and a baseline that trains
is not a baseline. If any of that does not hold, the run stops instead of training
a silently detached module.

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

    python tools/train_source_compaction.py --variant anchor-fusion --updates 384 \
        --data-dir /data/.../LongBench/extracted/data --train-task musique \
        --model-path /dev/shm/s1d_reader_fp16 \
        --checkpoint runs/natural_merge_v9/full_v10.pt \
        --anchor-checkpoint runs/anchor_residual/anchor_only_rank64.pt \
        --out runs/anchor_fusion/fusion_top4_rank32.pt \
        --report runs/anchor_fusion/fusion_top4_rank32.json
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
from runtime import (critical_manifest, load_runtime, reset,  # noqa: E402
                     tensor_digest)
from methods.prlmc.probes import (                        # noqa: E402
    _checkpoint_func, _compacted_mask, _pool_in_place, active_mask_override,
    differentiable_memory)
from methods.prlmc.anchor_residual import (               # noqa: E402
    SCHEMA as ANCHOR_SCHEMA, AnchorResidual,
    compact_anchors as compact_anchors_only)
from methods.prlmc.anchor_fusion import (                 # noqa: E402
    SCHEMA as FUSION_SCHEMA, FrozenAnchorFusion, SparseCandidateFusion,
    compact_anchors as compact_anchors_fusion,
    frozen_stage as anchor_stage)
from methods.prlmc.source_compaction import (             # noqa: E402
    SCHEMA, SourceCompaction, compact_anchors, split_source_anchors)
from objective import answer_logits                       # noqa: E402
from train_core_fusion import (                           # noqa: E402
    BUDGET_BLOCKS, TARGET_BLOCKS, WINDOW_BLOCKS, chunks_of, load_documents,
    write_blocks)

SLOTS_PER_BLOCK = gr.SLOTS_PER_BLOCK
WINDOW_SLOTS = WINDOW_BLOCKS * SLOTS_PER_BLOCK
TARGET_SLOTS = TARGET_BLOCKS * SLOTS_PER_BLOCK
# The output projection is the only parameter with a non-zero gradient while the
# residual is still exactly zero; the upstream half only starts moving after it
# leaves zero. Named per variant so the check is the same check.
PROJECTIONS = dict(compaction=("q_proj", "k_proj", "v_proj", "out_proj"),
                   **{"anchor-only": ("down", "up"),
                      "anchor-fusion": ("fusion.score_q", "fusion.score_k",
                                        "fusion.fuse_down", "fusion.fuse_up")})
OUTPUT_PROJECTION = dict(compaction="out_proj", **{"anchor-only": "up",
                                                  "anchor-fusion": "fusion.fuse_up"})
# The gate, by its name inside the module the variant trains.
GATE = dict(compaction="gate_logit", **{"anchor-only": "gate_logit",
                                        "anchor-fusion": "fusion.gate_logit"})
# Equal parameter count: 4 x rank x hidden + 1 for all three.
DEFAULT_RANK = dict(compaction=32, **{"anchor-only": 64, "anchor-fusion": 32})


def load_anchor(path, hidden):
    """The frozen AnchorOnly module, rebuilt from the checkpoint it was saved as."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != ANCHOR_SCHEMA + "/checkpoint":
        raise SystemExit(f"{path} is not a {ANCHOR_SCHEMA} checkpoint")
    config = dict(payload["config"])
    if int(config.get("hidden_size", -1)) != int(hidden):
        raise SystemExit(f"{path} was built for hidden {config.get('hidden_size')}, "
                         f"the reader is {hidden}")
    anchor = AnchorResidual(**config)
    anchor.load_state_dict(payload["state_dict"], strict=True)
    return anchor, config


def build_module(args, hidden):
    """The one module this variant trains, its rank, and its checkpoint schema."""
    rank = int(args.rank) if args.rank else DEFAULT_RANK[args.variant]
    if args.variant == "anchor-only":
        module = AnchorResidual(hidden_size=hidden, rank=rank,
                                max_ratio=args.max_ratio, gate_init=args.gate_init)
        return module, rank, ANCHOR_SCHEMA
    if args.variant == "anchor-fusion":
        if not args.anchor_checkpoint or not Path(args.anchor_checkpoint).is_file():
            raise SystemExit("--anchor-checkpoint (the trained AnchorOnly module) is "
                             "required by --variant anchor-fusion")
        anchor, _ = load_anchor(args.anchor_checkpoint, hidden)
        fusion = SparseCandidateFusion(hidden_size=hidden, rank=rank,
                                       top_k=args.top_k, max_ratio=args.max_ratio,
                                       gate_init=args.gate_init)
        return FrozenAnchorFusion(anchor, fusion), rank, FUSION_SCHEMA
    module = SourceCompaction(hidden_size=hidden, rank=rank,
                              max_ratio=args.max_ratio, gate_init=args.gate_init)
    return module, rank, SCHEMA


def parameter_counts(module):
    """Trainable and frozen parameter counts; the first is what the runs compare."""
    trainable = int(sum(p.numel() for p in module.parameters() if p.requires_grad))
    frozen = int(sum(p.numel() for p in module.parameters() if not p.requires_grad))
    return trainable, frozen


def compact_for_variant(args, output, source, parts, module):
    """The one call the variants disagree on."""
    if args.variant == "anchor-only":
        return compact_anchors_only(output, parts, module)
    if args.variant == "anchor-fusion":
        return compact_anchors_fusion(output, source, parts, module)
    return compact_anchors(output, source, parts, module)


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
    # first. The anchor-only variant has no candidates and never reads the window.
    anchors, info = compact_for_variant(args, output, source, parts, module)
    offsets = parts["anchor_offsets"]
    # The frozen stage of the anchor-fusion variant. It is BOTH the baseline the
    # report scores the fused window against and the reference the protection term
    # holds the branch to (see below), so it is built on every step of that variant.
    anchor_anchors = (compact_anchors_only(output, parts, module.anchor)[0]
                      if args.variant == "anchor-fusion" else None)

    if not train:
        with torch.no_grad():
            graft_logits, gold = reader_logits(model, manager, output, None, None, qa)
            graft_ce = float(F.cross_entropy(graft_logits, gold))
            del graft_logits
            anchor_ce = None
            if anchor_anchors is not None:
                logits, _ = reader_logits(model, manager, output, anchor_anchors,
                                          offsets, qa)
                anchor_ce = float(F.cross_entropy(logits, gold))
                del logits
            logits, _ = reader_logits(model, manager, output, anchors, offsets, qa)
            ce = float(F.cross_entropy(logits, gold))
        record = dict(ce=ce, graft_ce=graft_ce,
                      removed=1.0 - ce / max(graft_ce, 1e-9), **info)
        if anchor_ce is not None:
            record["anchor_ce"] = anchor_ce
            record["removed_vs_anchor"] = 1.0 - ce / max(anchor_ce, 1e-9)
        return record

    # What the protection term protects. For the two variants that START at the
    # grafted window it is the grafted window, and KL is exactly zero there, so the
    # term only stops the module drifting away from what the graft already answers.
    # For anchor-fusion the branch starts at the ANCHOR-ONLY window, where
    # KL(anchor || graft) is strictly positive: protecting the graft would put a
    # gradient on the branch whose only way to satisfy it is to undo the calibration
    # it was built on. It protects the frozen baseline instead, and KL is zero there.
    reference_anchors = anchor_anchors
    reference_name = ("anchor_ce" if args.variant == "anchor-fusion" else "graft_ce")
    reference_log_prob = None
    reference_ce = None
    if args.protect_weight > 0:
        with torch.no_grad():
            reference_logits, reference_gold = reader_logits(
                model, manager, output, reference_anchors, offsets, qa)
            reference_ce = F.cross_entropy(reference_logits, reference_gold)
            reference_log_prob = F.log_softmax(reference_logits.float(), dim=-1)
            del reference_logits, reference_gold
    logits, gold = reader_logits(model, manager, output, anchors, offsets, qa)
    ce = F.cross_entropy(logits, gold)
    protection = torch.zeros((), device=ce.device)
    if reference_log_prob is not None:
        protection = F.kl_div(F.log_softmax(logits.float(), dim=-1),
                              reference_log_prob.exp(), reduction="batchmean")
        del reference_log_prob
    record = dict(loss=ce + args.protect_weight * protection, ce=ce,
                  protection=protection, reference=reference_name, **info)
    record[reference_name] = (reference_ce if reference_ce is not None
                              else torch.zeros((), device=ce.device))
    return record


def build_module_report(args, schema, rank, hidden, trainable, frozen_parameters):
    """What the module is, in the report, per variant."""
    report = dict(schema=schema, variant=args.variant, rank=rank,
                  max_ratio=args.max_ratio, gate_init=args.gate_init, hidden=hidden,
                  trainable_parameters=trainable, frozen_parameters=frozen_parameters,
                  anchors="the 360 graft slots, copied verbatim from the dropped "
                          "blocks by the frozen operator")
    if args.variant == "anchor-fusion":
        report.update(
            top_k=args.top_k,
            frozen_anchor=dict(
                path=args.anchor_checkpoint, sha256=sha(args.anchor_checkpoint),
                schema=ANCHOR_SCHEMA + "/checkpoint", parameters=frozen_parameters,
                trainable=False,
                note="loaded from the finished anchor-only run and never updated; its "
                     "digest is compared before and after training"),
            candidates="the top-%d leftovers of each anchor's own source block, "
                       "selected by a bilinear score and mixed at their native %d "
                       "dimensions" % (args.top_k, hidden),
            bound="vector norm, not element-wise: delta = raw * bound / (bound + "
                  "rms(raw)) with bound = gate * max_ratio * rms(anchor). The "
                  "direction of the residual is untouched, and no coordinate "
                  "saturates against the bound on its own",
            initialization=(
                "the score and down projections are orthogonal and the output "
                "projection is zero: the fused window starts bit-identical to the "
                "AnchorOnly window, not to the grafted one"))
        return report
    report.update(
        candidates=("every unselected slot of each anchor's own source block"
                    if args.variant == "compaction" else
                    "none: the module takes the anchor alone and has no candidate "
                    "input at all"),
        initialization=(
            "the input projections are orthogonal and the output projection is zero: "
            "the compacted window starts bit-identical to the grafted one"))
    return report


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
    module, rank, _ = build_module(args, 4096)
    trainable, frozen = parameter_counts(module)
    print(json.dumps(dict(
        dry_run=True, train_task=args.train_task, counts=counts,
        updates=args.updates, steps_per_doc=args.steps_per_doc,
        epochs=round(args.updates * args.steps_per_doc / len(documents), 2),
        val_docs=min(args.val_docs, max(0, len(documents) - 1)),
        geometry=dict(window_blocks=WINDOW_BLOCKS, target_blocks=TARGET_BLOCKS,
                      core_slots=TARGET_BLOCKS * gr.CORE_SLOTS,
                      graft_slots=TARGET_BLOCKS * gr.GRAFT_SLOTS,
                      window_slots=WINDOW_SLOTS, target_slots=TARGET_SLOTS),
        module=dict(variant=args.variant, rank=rank, max_ratio=args.max_ratio,
                    gate_init=args.gate_init, hidden=4096,
                    top_k=(args.top_k if args.variant == "anchor-fusion" else None),
                    trainable_parameters=trainable, frozen_parameters=frozen,
                    anchor_checkpoint=(args.anchor_checkpoint
                                       if args.variant == "anchor-fusion" else None),
                    initial_residual="exactly zero (the output projection starts at "
                                     "zero)"),
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
    module, rank, schema = build_module(args, manager.hidden)
    module = module.to(model.memory.device).float()
    trainable, frozen_parameters = parameter_counts(module)
    optimizer = torch.optim.Adam(
        [p for p in module.parameters() if p.requires_grad], lr=args.lr)
    # Digest of the frozen anchor before the first update; compared at the end. An
    # Adam step cannot move a parameter that never received a gradient, but "cannot"
    # is the kind of claim this project measures.
    frozen_before = ({name: tensor_digest(p) for name, p in module.named_parameters()
                      if not p.requires_grad} if args.variant == "anchor-fusion"
                     else None)

    report = dict(
        schema=schema + "/training", status="running",
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
        module=build_module_report(args, schema, rank, manager.hidden, trainable,
                                   frozen_parameters),
        objective=dict(
            loss="answer cross-entropy of the compacted window",
            protection=("protect_weight * KL(compacted || anchor-only)"
                        if args.variant == "anchor-fusion" else
                        "protect_weight * KL(compacted || graft)"),
            protection_reference=("the frozen AnchorOnly window, because the branch "
                                  "starts there and KL against the graft would push it "
                                  "to undo the calibration"
                                  if args.variant == "anchor-fusion" else
                                  "the grafted window, which the module starts at"),
            protect_weight=args.protect_weight, answers_seen=True),
        optimizer=dict(name="Adam", lr=args.lr, grad_clip=args.grad_clip),
        updates=args.updates, steps_per_doc=args.steps_per_doc,
        critical_manifest=critical_manifest(), history=[], validation=[],
        peak_allocated_bytes=0)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.report, report)

    # A zero output projection must reproduce the baseline window EXACTLY. This is
    # the whole "the arms differ only by the new branch" claim, so it is measured
    # once before the first update instead of being assumed. The baseline is the
    # grafted window, except for anchor-fusion, whose baseline is the AnchorOnly
    # window the frozen stage produces.
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
        anchors, identity_info = compact_for_variant(args, output, source, parts,
                                                     module)
        if args.variant == "anchor-fusion":
            # The frozen stage inside the fused path, against that arm's own code.
            baseline = anchor_stage(output, parts, module)
            reference_anchors = compact_anchors_only(output, parts, module.anchor)[0]
            if not torch.equal(baseline, reference_anchors):
                raise ValueError("the frozen stage inside the fused path is not the "
                                 "AnchorOnly window: the baseline under the new "
                                 "branch is not the baseline being claimed")
            identity_same = bool(torch.equal(anchors, baseline))
            identity_reference = "anchor-only"
        else:
            identity_same = bool(torch.equal(
                anchors, output.index_select(1, parts["anchor_offsets"])))
            identity_reference = "graft"
    if not identity_same:
        raise ValueError("the fresh module is not the identity: the compacted window "
                         "differs from the baseline one before any update")
    identity = document_step(model, manager, merge, module, initial, training[0],
                             args, train=False)
    expected_ce = (identity["anchor_ce"] if args.variant == "anchor-fusion"
                   else identity["graft_ce"])
    if abs(identity["ce"] - expected_ce) > 1e-6:
        raise ValueError(f"the fresh module changes the answer CE: compacted "
                         f"{identity['ce']} != {identity_reference} {expected_ce}")
    report["identity_check"] = dict(
        document=training[0]["id"], reference=identity_reference,
        window_bit_identical=identity_same, ce=identity["ce"],
        graft_ce=identity["graft_ce"],
        anchor_ce=identity.get("anchor_ce"),
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
            # Measured, not assumed: the first update can only reach the output
            # projection (the residual is still exactly zero, so the upstream half
            # sits downstream of a zero matrix), and update 2 is the first one that
            # must reach every projection.
            required = ((OUTPUT_PROJECTION[args.variant],) if update == 1
                        else PROJECTIONS[args.variant])
            missing = [name for name in required
                       if name not in gradients
                       or not torch.isfinite(gradients[name]).all()
                       or float(gradients[name].abs().max()) == 0.0]
            if missing:
                raise ValueError(f"no gradient on {missing} at update {update}: the "
                                 f"frozen reader's backward did not reach the window")
            training_frozen = [name for name, parameter in module.named_parameters()
                               if not parameter.requires_grad
                               and parameter.grad is not None]
            if training_frozen:
                raise ValueError(f"the frozen parameters {training_frozen} received a "
                                 f"gradient at update {update}: the baseline is being "
                                 f"trained, so it is not a baseline")
            norm = float(torch.nn.utils.clip_grad_norm_(
                [p for p in module.parameters() if p.requires_grad], args.grad_clip))
            optimizer.step()
            ce = float(measured["ce"].detach())
            reference = measured["reference"]
            row = dict(update=update, document=document["id"],
                       loss=float(measured["loss"].detach()), ce=ce,
                       protection=float(measured["protection"].detach()),
                       grad_norm=norm, seconds=time.monotonic() - step_started,
                       gate=measured["gate"], delta=measured["relative_delta"],
                       grad_out_proj=float(
                           gradients[OUTPUT_PROJECTION[args.variant]].norm())
                       if OUTPUT_PROJECTION[args.variant] in gradients else None,
                       grad_gate=float(gradients[GATE[args.variant]].norm())
                       if GATE[args.variant] in gradients else None)
            row[reference] = float(measured[reference].detach())
            row["removed" if reference == "graft_ce" else "removed_vs_anchor"] = (
                1.0 - ce / max(row[reference], 1e-9))
            # Variant-specific diagnostics, only where the module produced them.
            for extra in ("anchor_relative_delta", "max_weight", "entropy",
                          "novelty_ratio", "raw_rms_ratio", "rms_preserved"):
                if extra in measured:
                    row[extra] = measured[extra]
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
                # The anchor-fusion variant also scores the frozen stage, so the
                # validation curve says whether the new branch beat the baseline it
                # was built on -- not merely whether it beat the grafted window.
                if scores and "anchor_ce" in scores[0]:
                    summary["anchor_ce"] = float(np.mean([score["anchor_ce"]
                                                          for score in scores]))
                    summary["removed_vs_anchor"] = (
                        1.0 - summary["ce"] / summary["anchor_ce"]
                        if summary["anchor_ce"] else None)
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
    config = dict(hidden_size=manager.hidden, rank=rank,
                  max_ratio=args.max_ratio, gate_init=args.gate_init)
    if args.variant == "anchor-fusion":
        config["top_k"] = args.top_k
        frozen_after = {name: tensor_digest(parameter) for name, parameter
                        in module.named_parameters() if not parameter.requires_grad}
        unchanged = frozen_after == frozen_before
        report["frozen_anchor_check"] = dict(
            parameters=sorted(frozen_before), digest_before=frozen_before,
            digest_after=frozen_after, unchanged=unchanged, passed=unchanged)
        if not unchanged:
            raise ValueError("the frozen anchor module changed during training: the "
                             "baseline under the new branch is not the checkpoint it "
                             "was loaded from")
    report["status"] = "complete"
    payload = dict(
        schema=schema + "/checkpoint", state_dict=module.state_dict(),
        config=config,
        training=dict(variant=args.variant, task=args.train_task,
                      updates=args.updates, lr=args.lr,
                      seed=args.seed, protect_weight=args.protect_weight,
                      data_sha256=report["data"]["sha256"],
                      checkpoint=args.checkpoint,
                      checkpoint_sha256=report["reader"]["checkpoint_sha256"],
                      anchor_checkpoint=(args.anchor_checkpoint
                                         if args.variant == "anchor-fusion" else None),
                      anchor_checkpoint_sha256=(sha(args.anchor_checkpoint)
                                                if args.variant == "anchor-fusion"
                                                else None)))
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
                   help="weight of the KL guard against the variant's reference "
                        "window (graft, or anchor-only for --variant anchor-fusion); "
                        "0 disables it")
    p.add_argument("--variant", default="compaction",
                   choices=("compaction", "anchor-only", "anchor-fusion"),
                   help="compaction: each anchor reads its block's leftovers; "
                        "anchor-only: the same trainer with the candidate path "
                        "removed; anchor-fusion: a frozen anchor-only module plus a "
                        "sparse top-k candidate branch")
    p.add_argument("--anchor-checkpoint", dest="anchor_checkpoint", default=None,
                   help="the trained AnchorResidual to freeze under "
                        "--variant anchor-fusion")
    p.add_argument("--top-k", dest="top_k", type=int, default=4,
                   help="how many of a block's leftovers the fusion branch reads")
    p.add_argument("--rank", type=int, default=None,
                   help="bottleneck rank; the defaults are chosen so all three "
                        "variants have the same parameter count (compaction 32, "
                        "anchor-only 64, anchor-fusion 32)")
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
