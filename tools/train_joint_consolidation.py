#!/usr/bin/env python3
"""Round 6: train the V11 joint architecture once, against the AnchorOnly baseline.

The three mechanisms this project has arrived at are trained together, because
separately each one has an obvious way to look like a failure and together they are
where the capacity actually comes from:

    module 1  reader-utility slot selection   `methods/prlmc/reader_utility.py`
    module 2  same-block homogeneous folding  `methods/prlmc/homogeneous_merge.py`
    module 3  full-pool residual calibration  `FullPoolResidual` in `anchor_residual.py`

    carriers 9 x 256 -> 1,620 core (folded)   dropped 2,560 -> 684 graft (selected)
    2,304 slots, the same budget the baseline spends on 1,944 verbatim + 360 anchors

What trains
-----------
`FullPoolResidual` only: 4 x rank x hidden plus a 3 -> 32 -> 2 x rank FiLM, with the
graft half initialised from the finished AnchorOnly checkpoint and the core half at
zero. Reader, writer and the probe bank are frozen; the fold assignment and the
graft selection are index operations with no parameters at all.

The objective
-------------
    L = CE(answer) + w_cf * CF + w_protect * KL(calibrated || reference) + w_reg * R

CE        weighted answer cross-entropy; the first answer token (the entity) counts
          `--first-token-weight` times, because "the reader produced the right
          string but the wrong head noun" and "the reader produced nothing" are not
          the same failure and a plain mean treats them as one.
CF        the counterfactual hinge, `max(0, gamma - logP(y|M_true) + logP(y|M_irrel))`,
          where `M_irrel` is the SAME window with the 684 graft slots taken from the
          previous document instead of this one. It is computed only on rows where the
          compressed window is not answering on its own (`--cf-gate reference`, the
          default), because a hinge on a row the window already gets right has nothing
          to teach. The gate deliberately does NOT measure `logP(Full) - logP(window)`:
          that needs a 4,864-slot forward per row, twice the width of the forward the
          training step runs, for one boolean. `--cf-gate full` restores it, `none`
          disables the gate. The number of qualifying steps is recorded either way: a
          CF term that almost never fires is not evidence about the architecture.
          The donor window is scored under `no_grad` -- the negative side enters as a
          measured constant, so the hinge is one-sided pressure on the true window
          instead of a second differentiable branch.
protect   KL to the window the joint run STARTS at -- the V11 window with the initial
          module, i.e. the AnchorOnly calibration on the graft slots and an untouched
          core. KL is exactly zero there, so the term only stops the core half from
          buying graft accuracy with carrier knowledge; it cannot push the run to undo
          the calibration it is built on. Cached per document: the reference depends on
          the document and on frozen weights only, so it never changes.
R         mean over slots of `(||dh|| / rms(h))^2`; keeps the calibration from
          spending the whole bound allowance on a few slots.

Cost, and why the effective batch size is a flag
------------------------------------------------
The plan asks for 384 updates at an effective batch of 8. One document step is three
reader forwards (Full, reference, calibrated) plus the window build, so a batch of 8
multiplies both by eight. The default here is `--grad-accum 1`, which is the
384-update budget every earlier round used and which is enough to answer "does the
joint architecture produce a net gain at all"; `--grad-accum 8` is available and the
dry run prints what it costs.
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
from runtime import load_runtime, reset, tensor_digest    # noqa: E402
from methods.prlmc.anchor_residual import (               # noqa: E402
    FULL_POOL_SCHEMA, FullPoolResidual, SCHEMA as ANCHOR_SCHEMA)
from methods.prlmc.core_fusion import apply_delta         # noqa: E402
from methods.prlmc.joint_consolidation import (           # noqa: E402
    SCHEMA as JOINT_SCHEMA, QUOTA, JointConsolidation, conditions)
from methods.prlmc.probes import _checkpoint_func         # noqa: E402
from methods.prlmc.reader_utility import ReaderScorer     # noqa: E402
from objective import answer_logits                       # noqa: E402
from train_core_fusion import (                           # noqa: E402
    BUDGET_BLOCKS, TARGET_BLOCKS, WINDOW_BLOCKS, chunks_of, load_documents,
    write_blocks)
from train_source_compaction import installed             # noqa: E402

SLOTS_PER_BLOCK = gr.SLOTS_PER_BLOCK
TARGET_SLOTS = TARGET_BLOCKS * SLOTS_PER_BLOCK
# Update 1 can only reach the two output halves, the two gates and the FiLM output:
# `up_core` starts at zero, so nothing upstream of it has a gradient yet, and the FiLM
# input projection sits behind a zero matrix of its own. Update 2 must reach all of it.
STAGE1 = ("up_graft", "up_core", "gate_logit", "type_gate_logit", "cond_up")
STAGE2 = ("down", "up_graft", "up_core", "gate_logit", "type_gate_logit",
          "cond_up", "cond_down")


# ---------------------------------------------------------------------------
# module and probe bank
# ---------------------------------------------------------------------------
def load_anchor_state(path, hidden):
    """The finished AnchorOnly checkpoint: its state dict and its config."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != ANCHOR_SCHEMA + "/checkpoint":
        raise SystemExit(f"{path} is not a {ANCHOR_SCHEMA} checkpoint")
    config = dict(payload["config"])
    if int(config.get("hidden_size", -1)) != int(hidden):
        raise SystemExit(f"{path} was built for hidden {config.get('hidden_size')}, "
                         f"the reader is {hidden}")
    return payload["state_dict"], config


def build_module(args, hidden, anchor_state, anchor_config):
    """The one trainable module: graft half from AnchorOnly, core half at zero."""
    rank = int(args.rank or anchor_config.get("rank"))
    module = FullPoolResidual(hidden_size=hidden, rank=rank,
                              max_ratio=args.max_ratio, gate_init=args.gate_init,
                              cond_rank=args.cond_rank)
    module.load_anchor_only(anchor_state, anchor_config)
    return module, rank


def build_probes(args, tokenizer, documents):
    """The query bank the reading score is computed against. No test data in it."""
    if args.probe_text:
        ids = tokenizer(args.probe_text, add_special_tokens=False)["input_ids"]
        return [list(ids)], dict(kind="text", text=args.probe_text, count=1)
    wanted = min(int(args.probe_count), len(documents))
    if wanted <= 0:
        raise SystemExit("the reader-utility score needs at least one probe question")
    bank = [list(document["query_ids"]) for document in documents[:wanted]]
    return bank, dict(kind="training_questions", task=args.probe_task, count=len(bank),
                      documents=[document["id"] for document in documents[:wanted]])


# ---------------------------------------------------------------------------
# one document
# ---------------------------------------------------------------------------
def calibrate_with_reg(window, meta, module, layer_chunk=4):
    """Module 3 with the residual regulariser kept inside the graph.

    Each layer chunk is a `torch.utils.checkpoint` region: the module is 0.52 M
    parameters, so recomputing it on the backward is nearly free, while keeping its
    32-layer float32 activations alive is not. Measured on the smoke run, the
    un-checkpointed version peaked at 33.4 GiB and the next update's backward had no
    room for the 3.12 GiB memory gradient. The delta is also cast to the window's
    dtype before it meets the slot, so the sum and the RMS restore never promote a
    [layers, slots, hidden] tensor to float32 twice over.
    """
    cond = conditions(meta, window.shape[1])
    # Preallocated and filled slice by slice. `torch.cat(pieces)` needs the whole
    # 32-layer output twice alive at once (the list and the concatenation), which at
    # 2,304 slots is 576 MiB of pure transient; writing into a buffer that already
    # has the final shape costs nothing.
    calibrated_all = torch.empty_like(window)
    regulariser = torch.zeros((), device=window.device)
    delta_sq = slot_sq = 0.0

    def chunk_step(chunk, conditioning):
        delta = module(chunk, conditioning)
        calibrated = apply_delta(chunk, delta.to(chunk.dtype))
        rms = chunk.float().square().mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        penalty = (delta.float() / rms).square().mean() * int(chunk.shape[0])
        magnitude = delta.detach().float().square().sum()
        return calibrated, penalty, magnitude

    for start in range(0, window.shape[0], layer_chunk):
        stop = min(start + layer_chunk, window.shape[0])
        chunk = window[start:stop]
        if torch.is_grad_enabled() and chunk.requires_grad:
            calibrated, penalty, magnitude = torch.utils.checkpoint.checkpoint(
                chunk_step, chunk, cond, use_reentrant=False)
        else:
            calibrated, penalty, magnitude = chunk_step(chunk, cond)
        calibrated_all[start:stop] = calibrated
        regulariser = regulariser + penalty
        delta_sq += float(magnitude)
        slot_sq += float(chunk.detach().float().square().sum())
    regulariser = regulariser / max(int(window.shape[0]), 1)
    info = dict(slots=int(window.shape[1]),
                core_slots=int((meta["slot_type"] == 0).sum()),
                graft_slots=int((meta["slot_type"] == 1).sum()),
                folded_slots=int((meta["cluster_size"] > 1).sum()),
                relative_delta=(delta_sq / max(slot_sq, 1e-24)) ** 0.5,
                gate=float(torch.sigmoid(module.gate_logit.detach().float())))
    return calibrated_all, regulariser, info


@contextmanager
def read_window(model, manager, window, calibrated):
    """Install the calibrated window in the reader's pool with its graph intact."""
    offsets = torch.arange(int(window.shape[1]), device=window.device)
    with installed(model, manager, window.detach(), calibrated, offsets):
        yield


def answer_logp(logits, gold):
    """Sum of the gold answer tokens' log-probabilities."""
    return -F.cross_entropy(logits, gold, reduction="none").sum()


def answer_weight(gold, first_weight):
    """Weights that put `first_weight`x the average emphasis on the first token."""
    weight = torch.ones_like(gold, dtype=torch.float32)
    if float(first_weight) != 1.0:
        weight[0] = float(first_weight)
        weight = weight / weight.mean()
    return weight


def document_step(model, manager, merge, scorer, joint, module, reference, initial,
                  document, donor_graft, args, train, cache):
    """One document: window, calibration, reference, and the gold answer's CE."""
    qa = dict(prompt_ids=document["query_ids"], answer_ids=document["answer_ids"])
    device = model.memory.device
    key = (document["id"], document.get("context_sha256"))
    reset(manager, initial)
    with torch.no_grad():
        write_blocks(manager, merge, chunks_of(document))
        source, ages, picked = gr.select_carriers(manager, WINDOW_BLOCKS, TARGET_BLOCKS)
        full = None
        if args.cf_gate == "full":
            # The 19-block uncompressed forward. It exists only to measure what the
            # compression COST this row, and at 4,864 slots it is twice the width of
            # the window the training step itself runs -- the single largest
            # transient in the step, carried for one boolean.
            full = cache["full"].get(key)
            if full is None:
                full_logits, full_gold = answer_logits(model, qa, device)
                full = dict(logp=float(answer_logp(full_logits, full_gold)))
                cache["full"][key] = full
                del full_logits, full_gold
        window, meta, telemetry = joint.build(source, picked, manager.block)
        reference_payload = cache["reference"].get(key)
        if reference_payload is None:
            reference_window, _, _ = calibrate_with_reg(window, meta, reference)
            with read_window(model, manager, window, reference_window):
                logits, gold = answer_logits(model, qa, device)
            per_token = -F.cross_entropy(logits, gold, reduction="none")
            reference_payload = dict(logp=float(answer_logp(logits, gold)),
                                     logprob=F.log_softmax(logits.float(), dim=-1),
                                     mean_per_token=float(per_token.mean()))
            # Bounded: the cached tensor is a [answer tokens, 128k] float32 slice, and
            # holding one per document would be hundreds of MiB by the end of a run
            # for no benefit -- the cache only ever hits on a repeated document.
            cache["reference"][key] = reference_payload
            cache["reference_order"].append(key)
            while len(cache["reference_order"]) > 8:
                cache["reference"].pop(cache["reference_order"].pop(0), None)
            del logits, reference_window
    # Two ways to decide which rows the counterfactual term is worth computing on.
    # `full` is the original one and pays for a 4,864-slot forward per row; `reference`
    # gates on how well the compressed window ITSELF answers, which is already measured
    # for the protection term and costs nothing. `none` turns the gate off.
    if args.cf_gate == "full":
        cost = full["logp"] - reference_payload["logp"]
        qualifies = bool(cost > float(args.cf_gate_margin))
    elif args.cf_gate == "reference":
        cost = None
        qualifies = bool(reference_payload["mean_per_token"]
                         < float(args.cf_gate_logp))
    else:
        cost = None
        qualifies = True
    evidence = dict(reference_logp=reference_payload["logp"],
                    mean_per_token=reference_payload["mean_per_token"],
                    full_logp=(full["logp"] if full is not None else None),
                    cost=cost, qualifies=qualifies)

    calibrated, regulariser, info = calibrate_with_reg(window, meta, module)
    with read_window(model, manager, window, calibrated):
        logits, gold = answer_logits(model, qa, device)
    per_token = F.cross_entropy(logits, gold, reduction="none")
    weight = answer_weight(gold, args.first_token_weight).to(per_token.dtype)
    ce = (per_token * weight).sum() / weight.sum()
    protection = F.kl_div(F.log_softmax(logits.float(), dim=-1),
                          reference_payload["logprob"].exp().to(device),
                          reduction="batchmean")
    loss = (ce + args.protect_weight * protection + args.reg_weight * regulariser)
    record = dict(ce=ce, protection=protection, regulariser=regulariser, loss=loss,
                  relative_delta=info["relative_delta"],
                  folded_slots=info["folded_slots"], module_gate=info["gate"],
                  **evidence)
    if train and args.cf_weight > 0 and evidence["qualifies"] and donor_graft is not None:
        graft_offsets = torch.nonzero(meta["slot_type"] == 1, as_tuple=False).flatten()
        donor_window = window.clone()
        donor_window[:, graft_offsets] = donor_graft.to(window.dtype)
        with torch.no_grad():
            donor_calibrated, _, _ = calibrate_with_reg(donor_window, meta, module)
            with read_window(model, manager, donor_window, donor_calibrated):
                donor_logits, _ = answer_logits(model, qa, device)
            donor_logp = float(answer_logp(donor_logits, gold))
            del donor_logits, donor_calibrated, donor_window
        margin = answer_logp(logits, gold) - donor_logp
        counterfactual = F.relu(float(args.cf_gamma) - margin)
        loss = loss + args.cf_weight * counterfactual
        record.update(loss=loss, counterfactual=counterfactual,
                      margin=float(margin.detach()), donor_logp=donor_logp)
    if not train:
        record = {name: (float(value.detach()) if torch.is_tensor(value) else value)
                  for name, value in record.items()}
    carried = dict(id=document["id"])
    if args.cf_weight > 0:
        graft_offsets = torch.nonzero(meta["slot_type"] == 1, as_tuple=False).flatten()
        carried["grafts"] = window.index_select(1, graft_offsets).detach().clone()
    return record, carried, telemetry


# ---------------------------------------------------------------------------
# report pieces
# ---------------------------------------------------------------------------
def critical_manifest():
    return dict(
        joint=sha(ROOT / "methods/prlmc/joint_consolidation.py"),
        utility=sha(ROOT / "methods/prlmc/reader_utility.py"),
        merge=sha(ROOT / "methods/prlmc/homogeneous_merge.py"),
        residual=sha(ROOT / "methods/prlmc/anchor_residual.py"),
        trainer=sha(__file__))


def module_report(args, rank, hidden, anchor_path, anchor_config):
    return dict(
        schema=FULL_POOL_SCHEMA, hidden=hidden, rank=rank, max_ratio=args.max_ratio,
        cond_rank=args.cond_rank,
        trainable_parameters=int(4 * rank * hidden + 2 * rank * args.cond_rank
                                 + 3 * args.cond_rank + 3),
        conditions=["delta_position", "log_cluster_size", "is_graft"],
        initialisation=dict(
            graft_half="copied from the AnchorOnly checkpoint, so the measured "
                       "calibration is the starting point rather than a target",
            core_half="zero, so every carrier slot is bit-identical at update 0",
            film="zero, so the condition cannot move a slot before it is trained",
            anchor_checkpoint=anchor_path, anchor_sha256=sha(anchor_path),
            anchor_rank=anchor_config.get("rank"),
            anchor_max_ratio=anchor_config.get("max_ratio")),
        applies_to="all 2,304 output slots: 1,620 core and 684 graft")


def objective_report(args):
    return dict(
        loss="CE(answer) + w_cf*CF + w_protect*KL + w_reg*R",
        first_token_weight=args.first_token_weight,
        counterfactual=dict(
            weight=args.cf_weight, gamma=args.cf_gamma,
            gate=args.cf_gate,
            gate_definition=("logP(Full) - logP(window) > margin" if args.cf_gate == "full"
                             else "the compressed window's own mean gold log-prob per "
                                  "token is below the threshold, i.e. it is not "
                                  "answering this row"
                             if args.cf_gate == "reference" else "off, every row"),
            gate_logp=args.cf_gate_logp, gate_margin=args.cf_gate_margin,
            negative_window="the same window with the graft slots taken from the "
                            "previous document",
            negative_gradient="the donor window is scored under no_grad, so the hinge "
                              "is one-sided pressure on the true window"),
        protection=dict(weight=args.protect_weight,
                        reference="the V11 window with the initial module, i.e. the "
                                  "AnchorOnly calibration on the graft slots and an "
                                  "untouched core",
                        note="KL is zero there, so the term cannot push the run to "
                             "undo its own baseline"),
        regulariser=dict(weight=args.reg_weight,
                         definition="mean over slots of (||dh||/rms(h))^2"),
        answers_seen=True)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def train(args):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for flag, value, what in (("--checkpoint", args.checkpoint, "the frozen operator"),
                              ("--anchor", args.anchor, "the AnchorOnly checkpoint")):
        if not value or not Path(value).is_file():
            raise SystemExit(f"this run needs {what}: pass {flag}")
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
    probe_pool = load_documents(tokenizer, args.data_dir,
                                args.probe_task or args.train_task, None)[0]
    probes, probe_sources = build_probes(args, tokenizer, probe_pool)

    if args.dry_run:
        print(json.dumps(dict(
            dry_run=True, out=args.out, report=args.report, counts=counts,
            train_documents=len(training), validation_documents=len(validation),
            probes=len(probes),
            quota=dict(core_per_block=args.core_per_block,
                       graft_per_block=args.graft_per_block,
                       core_total=TARGET_BLOCKS * args.core_per_block,
                       graft_total=TARGET_BLOCKS * args.graft_per_block,
                       per_block_total=args.core_per_block + args.graft_per_block,
                       budget=TARGET_SLOTS),
            updates=args.updates, grad_accum=args.grad_accum,
            document_forwards=args.updates * args.grad_accum,
            cost_note="one document step is three reader forwards; --grad-accum 8 is "
                      "the plan's effective batch of 8 and costs 8x the wall clock"),
            indent=2, ensure_ascii=False))
        return None
    if args.core_per_block + args.graft_per_block != SLOTS_PER_BLOCK:
        raise SystemExit("core + graft per block must fill one 256-slot block")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.cuda_device))
    nv.seed_everything(args.seed)
    checkpoint_training = torch.load(args.checkpoint, map_location="cpu").get("training")
    model, tokenizer, manager, merge, initial, cfg, saved = load_runtime(
        args.model_path, checkpoint=args.checkpoint, train=False, architecture=None,
        budget_blocks=BUDGET_BLOCKS, strict_code_hash=args.strict_code_hash,
        training=checkpoint_training)
    merge.operator_mode = "native"
    for dropout in model.modules():
        if isinstance(dropout, nn.Dropout):
            dropout.p = 0.0
    model.train()
    if not args.no_reader_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.model.gradient_checkpointing = True
        model.model._gradient_checkpointing_func = _checkpoint_func()

    layers = tuple(int(x) for x in str(args.utility_layers).split(",") if x.strip())
    scorer = ReaderScorer(model, layers)
    bank = [scorer.probe_bank(ids) for ids in probes]
    joint = JointConsolidation(
        scorer, bank, core_per_block=args.core_per_block,
        graft_per_block=args.graft_per_block, policy=args.policy,
        mmr_penalty=args.mmr_penalty, novelty_weight=args.novelty_weight,
        merge_kwargs=dict(key_threshold=args.key_threshold,
                          value_threshold=args.value_threshold,
                          max_cluster=args.max_cluster,
                          blend_weight=args.blend_weight))
    anchor_state, anchor_config = load_anchor_state(args.anchor, manager.hidden)
    module, rank = build_module(args, manager.hidden, anchor_state, anchor_config)
    module = module.to(model.memory.device).float()
    reference = FullPoolResidual(hidden_size=manager.hidden, rank=rank,
                                 max_ratio=args.max_ratio, cond_rank=args.cond_rank)
    reference.load_state_dict(module.state_dict(), strict=True)
    reference = reference.to(model.memory.device).eval().requires_grad_(False)

    optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
    trainable = int(sum(p.numel() for p in module.parameters() if p.requires_grad))
    report = dict(
        schema=FULL_POOL_SCHEMA + "/training", status="running",
        data=dict(dir=args.data_dir, task=args.train_task,
                  sha256=sha(Path(args.data_dir) / (args.train_task + ".jsonl")),
                  counts=counts, train_documents=len(training),
                  validation_documents=len(validation),
                  validation_ids=[document["id"] for document in validation],
                  evaluation_task_never_written="hotpotqa"),
        reader=dict(model_path=args.model_path, frozen=True,
                    checkpoint=args.checkpoint, checkpoint_sha256=sha(args.checkpoint),
                    checkpoint_code_hash_verified=(saved or {}).get("code_hash_verified")),
        module=module_report(args, rank, manager.hidden, args.anchor, anchor_config),
        geometry=dict(core_per_block=args.core_per_block,
                      graft_per_block=args.graft_per_block,
                      core_total=TARGET_BLOCKS * args.core_per_block,
                      graft_total=TARGET_BLOCKS * args.graft_per_block,
                      budget=TARGET_SLOTS, policy=args.policy,
                      joint_schema=JOINT_SCHEMA, trainable_parameters=trainable),
        probe=dict(sources=probe_sources, layers=list(layers), count=len(probes),
                   digest=tensor_digest(torch.tensor(
                       [token for probe in probes for token in probe], dtype=torch.long)),
                   note="a fixed bank encoded before the first test row is read; no "
                        "test question and no reference answer enters the score"),
        objective=objective_report(args),
        optimizer=dict(name="AdamW", lr=args.lr, weight_decay=args.weight_decay,
                       grad_clip=args.grad_clip, updates=args.updates,
                       grad_accum=args.grad_accum),
        critical_manifest=critical_manifest(), history=[], validation=[],
        peak_allocated_bytes=0)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.report, report)

    # ---- identity: the initial module must reproduce its own baseline ----------
    with torch.no_grad():
        reset(manager, initial)
        write_blocks(manager, merge, chunks_of(training[0]))
        source, ages, picked = gr.select_carriers(manager, WINDOW_BLOCKS, TARGET_BLOCKS)
        window, meta, telemetry = joint.build(source, picked, manager.block)
        calibrated, regulariser, info = calibrate_with_reg(window, meta, module)
        core = meta["slot_type"] == 0
        graft = meta["slot_type"] == 1
        core_identical = bool(torch.equal(calibrated[:, core], window[:, core]))
        anchor = module.anchor_view().to(window.device).float()
        expected_graft = apply_delta(window[:, graft], anchor(window[:, graft]))
        # Compared with a tolerance, not bit for bit: the graft half is evaluated
        # here on a gathered copy and inside the training loop on a strided view of
        # the same slots, and the per-slot RMS reduction can round differently
        # between the two layouts. What the check has to rule out is the half not
        # having loaded, which would move these slots by O(10%). Measured on the
        # smoke run: 4.4e-3, i.e. fp16 storage noise on the delta, hence 1e-2.
        deviation = ((calibrated[:, graft].float() - expected_graft.float()).abs()
                     / window[:, graft].float().square().mean(dim=-1, keepdim=True)
                     .sqrt().clamp_min(1e-6))
        graft_deviation = float(deviation.max())
        graft_identical = graft_deviation <= 1e-2
        at_origin = float(regulariser.detach())
    if not core_identical:
        raise ValueError("the initial module moved a core slot: the core half is not "
                         "the identity, so the run does not start on its baseline")
    if not graft_identical:
        raise ValueError(f"the initial module does not reproduce AnchorOnly on the graft "
                         f"slots (max relative deviation {graft_deviation:.3e}, the "
                         f"tolerance is 1e-2), so the graft half did not load")
    identity = dict(document=training[0]["id"], core_bit_identical=core_identical,
                    graft_matches_anchor_only=graft_identical,
                    graft_max_relative_deviation=graft_deviation,
                    regulariser_at_origin=at_origin, core_slots=info["core_slots"],
                    graft_slots=info["graft_slots"], folded_slots=info["folded_slots"],
                    merge=telemetry["merge"], probe_passed=True)
    report["identity_check"] = identity
    print(json.dumps({"identity_check": identity}, ensure_ascii=False), flush=True)

    cache = dict(full={}, reference={}, reference_order=[])
    donor_graft = None
    started = time.monotonic()
    update = 0
    qualified = 0
    while update < args.updates:
        optimizer.zero_grad(set_to_none=True)
        recorded = []
        for accumulated in range(args.grad_accum):
            document = training[(update * args.grad_accum + accumulated) % len(training)]
            measured, carried, telemetry = document_step(
                model, manager, merge, scorer, joint, module, reference, initial,
                document, donor_graft, args, True, cache)
            # The 3.12 GiB memory gradient has to be one contiguous block, and the
            # three no_grad forwards above leave freed-but-reserved blocks in the
            # allocator's cache. Handing them back to the driver first is what turns
            # "262 MiB reserved but unallocated" into usable room.
            torch.cuda.empty_cache()
            (measured["loss"] / args.grad_accum).backward()
            donor_graft = carried["grafts"] if args.cf_weight > 0 else None
            recorded.append({name: (float(value.detach()) if torch.is_tensor(value)
                                    and value.dim() == 0 else value)
                             for name, value in measured.items()})
            if measured["qualifies"]:
                qualified += 1
        update += 1
        gradients = {name: parameter.grad for name, parameter in
                     module.named_parameters() if parameter.grad is not None}
        required = STAGE1 if update == 1 else STAGE2
        missing = [name for name in required
                   if name not in gradients or not torch.isfinite(gradients[name]).all()
                   or float(gradients[name].abs().max()) == 0.0]
        if missing:
            raise ValueError(f"no gradient on {missing} at update {update}: the call is "
                             f"not attached to the loss")
        norm = float(torch.nn.utils.clip_grad_norm_(
            [p for p in module.parameters() if p.requires_grad], args.grad_clip))
        optimizer.step()
        measured = recorded[-1]
        row = dict(update=update, loss=measured["loss"], ce=measured["ce"],
                   protection=measured["protection"],
                   regulariser=measured["regulariser"],
                   module_gate=measured["module_gate"],
                   delta=measured["relative_delta"], cost=measured["cost"],
                   qualifies=measured["qualifies"], grad_norm=norm,
                   grad_up_core=float(gradients["up_core"].norm()),
                   grad_cond_down=(float(gradients["cond_down"].norm())
                                   if "cond_down" in gradients else None),
                   seconds=time.monotonic() - started)
        if "counterfactual" in measured:
            row["counterfactual"] = measured["counterfactual"]
            row["margin"] = measured["margin"]
        report["history"].append(row)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                          for k, v in row.items()}), flush=True)
        if args.val_every and validation and (update % args.val_every == 0
                                              or update == args.updates):
            scores = []
            for held in validation:
                record, _, held_telemetry = document_step(
                    model, manager, merge, scorer, joint, module, reference, initial,
                    held, None, args, False, cache)
                scores.append((record, held_telemetry))
            summary = dict(
                update=update,
                ce=float(np.mean([record["ce"] for record, _ in scores])),
                reference_logp=float(np.mean([record["reference_logp"]
                                              for record, _ in scores])),
                mean_per_token=float(np.mean([record["mean_per_token"]
                                              for record, _ in scores])),
                merge_evicting_blocks=int(np.mean([
                    telemetry["merge"]["blocks_where_quota_came_from_eviction"]
                    for _, telemetry in scores])))
            if scores and scores[0][0]["full_logp"] is not None:
                summary["full_logp"] = float(np.mean([record["full_logp"]
                                                      for record, _ in scores]))
                summary["cost"] = summary["full_logp"] - summary["reference_logp"]
            report["validation"].append(summary)
            report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            if args.report:
                atomic_json(args.report, report)
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                              for k, v in summary.items()}), flush=True)

    report["cf_effective_steps"] = qualified
    report["cf_effective_fraction"] = round(qualified / max(args.updates, 1), 4)
    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    report["status"] = "complete"
    payload = dict(
        schema=FULL_POOL_SCHEMA + "/checkpoint",
        config=dict(hidden_size=manager.hidden, rank=rank, max_ratio=args.max_ratio,
                    gate_init=args.gate_init, cond_size=3, cond_rank=args.cond_rank),
        geometry=dict(policy=args.policy, core_per_block=args.core_per_block,
                      graft_per_block=args.graft_per_block,
                      key_threshold=args.key_threshold,
                      value_threshold=args.value_threshold,
                      max_cluster=args.max_cluster, blend_weight=args.blend_weight,
                      mmr_penalty=args.mmr_penalty,
                      novelty_weight=args.novelty_weight,
                      utility_layers=list(layers), probe_ids=probes,
                      probe_sources=probe_sources),
        state_dict={name: tensor.detach().cpu()
                    for name, tensor in module.state_dict().items()},
        training=dict(updates=args.updates, seed=args.seed, lr=args.lr,
                      elapsed_seconds=report["elapsed_seconds"],
                      cf_effective_steps=qualified, report=args.report,
                      manifest=report["critical_manifest"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    report["checkpoint"] = dict(path=args.out, sha256=sha(args.out))
    if args.report:
        atomic_json(args.report, report)
    print(json.dumps(dict(status="complete", out=args.out, report=args.report,
                          updates=args.updates, cf_effective_steps=qualified,
                          elapsed_seconds=round(report["elapsed_seconds"], 1),
                          peak_allocated_bytes=report["peak_allocated_bytes"]),
                     indent=1, ensure_ascii=False))
    return report


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--train-task", dest="train_task", default="musique")
    p.add_argument("--model-path", default=None)
    p.add_argument("--checkpoint", default=None,
                   help="the frozen V10 operator (runs/natural_merge_v9/full_v10.pt)")
    p.add_argument("--anchor", default=None,
                   help="the finished AnchorOnly checkpoint the graft half starts from")
    p.add_argument("--out", default=None, help="module checkpoint (.pt)")
    p.add_argument("--report", default=None, help="training report (.json)")
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--updates", type=int, default=384)
    p.add_argument("--grad-accum", dest="grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.0)
    p.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rank", type=int, default=None,
                   help="default: the AnchorOnly checkpoint's rank (64)")
    p.add_argument("--max-ratio", dest="max_ratio", type=float, default=0.5)
    p.add_argument("--gate-init", dest="gate_init", type=float, default=0.0)
    p.add_argument("--cond-rank", dest="cond_rank", type=int, default=32)
    p.add_argument("--protect-weight", dest="protect_weight", type=float, default=1.0)
    p.add_argument("--reg-weight", dest="reg_weight", type=float, default=0.05)
    p.add_argument("--cf-weight", dest="cf_weight", type=float, default=0.3)
    p.add_argument("--cf-gamma", dest="cf_gamma", type=float, default=0.5)
    p.add_argument("--cf-gate", dest="cf_gate", default="reference",
                   choices=("reference", "full", "none"),
                   help="which rows the counterfactual term is computed on; `full` "
                        "pays for a 4,864-slot forward per row, `reference` is free")
    p.add_argument("--cf-gate-logp", dest="cf_gate_logp", type=float, default=-0.5)
    p.add_argument("--cf-gate-margin", dest="cf_gate_margin", type=float, default=1.0)
    p.add_argument("--first-token-weight", dest="first_token_weight", type=float,
                   default=2.0)
    p.add_argument("--max-docs", dest="max_docs", type=int, default=None)
    p.add_argument("--val-docs", dest="val_docs", type=int, default=2)
    p.add_argument("--val-every", dest="val_every", type=int, default=48)
    p.add_argument("--policy", default="merge", choices=("merge", "evict"))
    p.add_argument("--core-per-block", dest="core_per_block", type=int,
                   default=QUOTA["core_per_block"])
    p.add_argument("--graft-per-block", dest="graft_per_block", type=int,
                   default=QUOTA["graft_per_block"])
    p.add_argument("--mmr-penalty", dest="mmr_penalty", type=float, default=0.5)
    p.add_argument("--novelty-weight", dest="novelty_weight", type=float, default=0.5)
    p.add_argument("--key-threshold", dest="key_threshold", type=float, default=0.72)
    p.add_argument("--value-threshold", dest="value_threshold", type=float, default=0.60)
    p.add_argument("--max-cluster", dest="max_cluster", type=int, default=3)
    p.add_argument("--blend-weight", dest="blend_weight", type=float, default=0.25)
    p.add_argument("--utility-layers", dest="utility_layers", default="12,16,20,24,28")
    p.add_argument("--probe-count", dest="probe_count", type=int, default=16)
    p.add_argument("--probe-task", dest="probe_task", default=None)
    p.add_argument("--probe-text", dest="probe_text", default=None)
    p.add_argument("--strict-code-hash", dest="strict_code_hash", default="true",
                   choices=("true", "false"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-reader-checkpointing", dest="no_reader_checkpointing",
                   action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.data_dir or not args.model_path:
        raise SystemExit("--data-dir and --model-path are required")
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
