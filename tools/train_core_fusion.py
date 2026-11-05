#!/usr/bin/env python3
"""Experiment 1B, training half: teach a small module to fuse the doomed slots.

What is trained
---------------
Exactly one module (`methods.prlmc.core_fusion.LearnedCoreFusion`, 0.52 M
parameters at rank 32). The reader, the writer and the V10 graft operator are
frozen; nothing else in the system has a gradient.

What the objective is
---------------------
Reader readout distillation, and it never sees an answer. For every training
document the driver writes the 19-block window, compresses it with the frozen V10
operator (native mode), fuses the 40 doomed slots of each carrier into that
carrier's 216 kept slots, and then asks the frozen reader's own attention: how
close is the readout from the COMPRESSED window to the readout from the
UNCOMPRESSED one, at the four V10 distillation taps, for the last eight tokens of
the QUESTION? The question's hidden states come from `prompt_hidden_states`, which
runs the prompt alone -- no gold answer, no answer tokens, no reference model.

This is the same loss V10 itself was trained with
(`tools.objective.compute_local_window_readout_loss`); the only change is that it
is applied to the fusion residual instead of to the operator.

Training data
-------------
A DISJOINT task, by default LongBench musique (196 of its 200 rows fill a 19-block
window). HotpotQA, the evaluation task, is never written during training: the
fusion is a zero-shot module on the 158 evaluation rows. The readout objective is
answer-free either way, but a module tuned on the test documents would make the
comparison unreadable, so the split is by task and recorded in the report.

Usage
-----
    python tools/train_core_fusion.py --dry-run \
        --data-dir /data/.../LongBench/extracted/data --train-task musique

    python tools/train_core_fusion.py --smoke \
        --data-dir /data/.../LongBench/extracted/data --train-task musique \
        --model-path /dev/shm/s1d_reader_fp16 \
        --checkpoint runs/natural_merge_v9/full_v10.pt \
        --out runs/core_fusion/smoke.pt --report runs/core_fusion/smoke.json

    python tools/train_core_fusion.py --updates 192 \
        --data-dir /data/.../LongBench/extracted/data --train-task musique \
        --model-path /dev/shm/s1d_reader_fp16 \
        --checkpoint runs/natural_merge_v9/full_v10.pt \
        --out runs/core_fusion/fusion_rank32.pt \
        --report runs/core_fusion/fusion_rank32.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_core as pe                                    # noqa: E402
import eval_graft_recovery as gr                          # noqa: E402
import native_eval as nv                                  # noqa: E402
from io_utils import atomic_json, digest, sha             # noqa: E402
from runtime import critical_manifest, load_runtime, reset  # noqa: E402
from stream_eval import StreamingMemory                   # noqa: E402
from methods.prlmc.core_fusion import (                   # noqa: E402
    LearnedCoreFusion, SCHEMA, apply_delta, prompt_hidden_states, readout_distance,
    split_core_and_deleted)
from tools.v9_geometry_50block import DISTILL_LAYERS      # noqa: E402

BLOCK_TOKENS = gr.BLOCK_TOKENS
SLOTS_PER_BLOCK = gr.SLOTS_PER_BLOCK
WINDOW_BLOCKS = gr.WINDOW_BLOCKS
TARGET_BLOCKS = gr.TARGET_BLOCKS
CORE_SLOTS = gr.CORE_SLOTS
BUDGET_BLOCKS = gr.BUDGET_BLOCKS
WRITE_TOKENS = gr.WRITE_TOKENS
TAPS = tuple(int(layer) for layer in DISTILL_LAYERS)


def write_blocks(manager, merge, chunks):
    """Append `len(chunks)` native blocks through the plain (non-compacting) writer."""
    stream = StreamingMemory(manager, merge, "Full", budget_blocks=BUDGET_BLOCKS)
    for index, chunk in enumerate(chunks):
        record = dict(role="write", token_ids=chunk, token_sha256=digest(chunk))
        if stream.write(record, index) is None:
            raise ValueError("the writer refused a block: the window does not fit the pool")
    return stream


def chunks_of(document):
    return [document["write_ids"][index:index + BLOCK_TOKENS]
            for index in range(0, len(document["write_ids"]), BLOCK_TOKENS)]


def load_documents(tokenizer, data_dir, task, limit=None):
    """Training documents that fill one 19-block window, with their question."""
    rows = nv.load_rows(data_dir, task)
    if limit is not None:
        rows = rows[:limit]
    documents, skipped = [], 0
    for row in rows:
        ids = tokenizer(row["context"], add_special_tokens=False)["input_ids"]
        if len(ids) // BLOCK_TOKENS < WINDOW_BLOCKS:
            skipped += 1
            continue
        documents.append(dict(
            id=str(row.get("_id")),
            write_ids=list(ids[:WRITE_TOKENS]),
            query_ids=tokenizer(gr.QUERY_SUFFIX.format(input=row["input"]),
                                add_special_tokens=False).input_ids,
            context_sha256=digest(row["context"])))
    return documents, dict(rows=len(rows), eligible=len(documents), too_short=skipped)


def fused_readout_loss(model, manager, merge, fusion, initial, document, train):
    """One document: write, compress, fuse, and measure the distillation loss.

    `train=True` keeps the graph on the fused window and returns the loss tensor.
    `train=False` additionally measures the UNFUSED window, so a validation row
    says how much of the readout distance the fusion removes rather than only
    what the fused loss is.
    """
    reset(manager, initial)
    with torch.no_grad():
        write_blocks(manager, merge, chunks_of(document))
        source, _, picked = gr.select_carriers(manager, WINDOW_BLOCKS, TARGET_BLOCKS)
        frozen, pinned = gr.window_locks(manager, WINDOW_BLOCKS)
        output, telemetry = gr.compress_graft(manager, merge, source, picked,
                                              WINDOW_BLOCKS, TARGET_BLOCKS,
                                              frozen, pinned)
    parts = split_core_and_deleted(telemetry["slot_source_positions"],
                                   telemetry["bridge_slot_offsets"], picked,
                                   SLOTS_PER_BLOCK, CORE_SLOTS, device=source.device)
    graft_offsets = torch.as_tensor(telemetry["bridge_slot_offsets"], dtype=torch.long,
                                    device=source.device)
    taps = torch.as_tensor(TAPS, dtype=torch.long, device=source.device)
    query_hidden = prompt_hidden_states(model, document["query_ids"], TAPS)
    deleted_index = torch.as_tensor(parts["deleted_positions"], dtype=torch.long,
                                    device=source.device)
    core = output.index_select(0, taps).index_select(1, parts["core_offsets"])
    doomed = source.index_select(0, taps).index_select(1, deleted_index)
    graft = output.index_select(0, taps).index_select(1, graft_offsets)
    pre = {layer: source[layer] for layer in TAPS}
    unfused = {layer: torch.cat([core[index], graft[index]], dim=0)
               for index, layer in enumerate(TAPS)}
    with torch.no_grad():
        baseline = float(readout_distance(model, pre, unfused, query_hidden, TAPS))
    # The module emits a RESIDUAL. Adding it and restoring the slot's RMS is the
    # one step that turns it into a window; skipping it trains the module to emit
    # a window-shaped tensor and silently replaces the memory with the residual.
    fused = apply_delta(core, fusion(core, doomed, parts["core_block_ids"],
                                     parts["deleted_block_ids"]))
    post = {layer: torch.cat([fused[index], graft[index]], dim=0)
            for index, layer in enumerate(TAPS)}
    if train:
        return dict(loss=readout_distance(model, pre, post, query_hidden, TAPS),
                    baseline=baseline,
                    delta=float((fused - core).detach().float().norm()
                                / core.detach().float().norm().clamp_min(1e-6)))
    with torch.no_grad():
        loss = float(readout_distance(model, pre, post, query_hidden, TAPS))
    return dict(loss=loss, baseline=baseline)


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
                      core_slots=CORE_SLOTS, deleted_per_block=SLOTS_PER_BLOCK - CORE_SLOTS,
                      deleted_total=TARGET_BLOCKS * (SLOTS_PER_BLOCK - CORE_SLOTS)),
        fusion=dict(rank=args.rank, max_ratio=args.max_ratio, gate_init=args.gate_init,
                    hidden=4096, parameters=4 * args.rank * 4096 + 1,
                    initial_residual="exactly zero (out_proj starts at zero)"),
        distillation_taps=list(TAPS),
        objective=("reader readout cosine + 0.5 * smooth_l1 from the compressed window "
                   "to the uncompressed one, at the last 8 tokens of the question only"),
        answer_leakage="none: the objective never reads an answer token",
        data_note=("training documents come from a different LongBench task than the "
                   "HotpotQA evaluation rows")), indent=2, ensure_ascii=False))
    return 0


def train(args):
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
    fusion = LearnedCoreFusion(hidden_size=manager.hidden, rank=args.rank,
                               max_ratio=args.max_ratio,
                               gate_init=args.gate_init).to(model.memory.device).float()
    optimizer = torch.optim.Adam(fusion.parameters(), lr=args.lr)

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
        fusion=dict(schema=SCHEMA, rank=args.rank, max_ratio=args.max_ratio,
                    gate_init=args.gate_init, hidden=manager.hidden,
                    parameters=int(sum(p.numel() for p in fusion.parameters())),
                    initialization="q/k/v orthogonal, out_proj zero: the fused window "
                                   "starts bit-identical to the unfused one"),
        objective=dict(loss="tools.objective.compute_local_window_readout_loss "
                            "(cosine + 0.5 * smooth_l1)",
                       taps=list(TAPS), query_tokens=8, slot_stride=1,
                       answers_seen=False),
        optimizer=dict(name="Adam", lr=args.lr, grad_clip=args.grad_clip),
        updates=args.updates, steps_per_doc=args.steps_per_doc,
        critical_manifest=critical_manifest(), history=[], validation=[],
        peak_allocated_bytes=0)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.report, report)

    # A zero output projection must reproduce the unfused window EXACTLY. This is
    # the whole "the two arms differ only by the fusion" claim, so it is measured
    # once before the first update instead of being assumed.
    identity = fused_readout_loss(model, manager, merge, fusion, initial,
                                  training[0], train=False)
    if abs(identity["loss"] - identity["baseline"]) > 1e-6:
        raise ValueError(f"the fresh module is not the identity: fused readout loss "
                         f"{identity['loss']} != unfused {identity['baseline']}")
    report["identity_check"] = dict(document=training[0]["id"], **identity,
                                    passed=True)

    started = time.monotonic()
    update = 0
    while update < args.updates:
        document = training[update % len(training)]
        for _ in range(args.steps_per_doc):
            if update >= args.updates:
                break
            step_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            measured = fused_readout_loss(model, manager, merge, fusion, initial,
                                          document, train=True)
            measured["loss"].backward()
            norm = float(torch.nn.utils.clip_grad_norm_(fusion.parameters(),
                                                        args.grad_clip))
            optimizer.step()
            update += 1
            row = dict(update=update, document=document["id"],
                       loss=float(measured["loss"].detach()),
                       baseline=measured["baseline"],
                       removed=float(1.0 - measured["loss"].detach()
                                     / max(measured["baseline"], 1e-9)),
                       grad_norm=norm, seconds=time.monotonic() - step_started,
                       gate=float(torch.sigmoid(fusion.gate_logit.detach())),
                       delta=measured.get("delta"))
            report["history"].append(row)
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                              for k, v in row.items()}), flush=True)
            if args.val_every and validation and (update % args.val_every == 0
                                                  or update == args.updates):
                scores = [fused_readout_loss(model, manager, merge, fusion, initial,
                                             document, train=False)
                          for document in validation]
                summary = dict(
                    update=update,
                    loss=float(np.mean([score["loss"] for score in scores])),
                    baseline=float(np.mean([score["baseline"] for score in scores])))
                summary["removed"] = (1.0 - summary["loss"] / summary["baseline"]
                                      if summary["baseline"] else None)
                report["validation"].append(summary)
                print(json.dumps({key: (round(value, 5) if isinstance(value, float)
                                        else value)
                                  for key, value in summary.items()}), flush=True)
        if args.report:
            report["elapsed_seconds"] = time.monotonic() - started
            atomic_json(args.report, report)

    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    report["final_gate"] = float(torch.sigmoid(fusion.gate_logit.detach()))
    report["status"] = "complete"
    payload = dict(
        schema=SCHEMA + "/checkpoint", state_dict=fusion.state_dict(),
        config=dict(hidden_size=manager.hidden, rank=args.rank,
                    max_ratio=args.max_ratio, gate_init=args.gate_init),
        training=dict(task=args.train_task, updates=args.updates, lr=args.lr,
                      seed=args.seed, taps=list(TAPS),
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
                   help="the frozen V10 operator, e.g. runs/natural_merge_v9/full_v10.pt")
    p.add_argument("--out", default=None, help="fusion checkpoint to write")
    p.add_argument("--report", default=None, help="training report to write")
    p.add_argument("--updates", type=int, default=192)
    p.add_argument("--steps-per-doc", dest="steps_per_doc", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--max-ratio", dest="max_ratio", type=float, default=0.5)
    p.add_argument("--gate-init", dest="gate_init", type=float, default=0.0)
    p.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0)
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


def main():
    args = build_parser().parse_args()
    if args.model_path is None:
        args.model_path = pe.MODEL_PATH
    if args.smoke:
        args.updates = min(args.updates, 4)
        args.val_docs = min(args.val_docs, 2)
        args.val_every = 2
    if args.dry_run:
        return dry_run(args)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
