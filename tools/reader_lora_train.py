#!/usr/bin/env python3
"""Reader LoRA for MemoryLLM-8B native latent memory.

Trains ONLY LoRA deltas on the last 8 decoder layers (q_proj / o_proj). Every
MemoryLLM weight and the whole memory write path stay frozen and bit-identical to
evaluation. Compression during training uses the Selection arm at the paper
geometry (50 blocks, 19 -> 9, trigger at full load), so the adapter only ever sees
the memory format it meets at 128k evaluation.

The question and the answer are never written into memory: each stream is built from
context text only, and the QA pair is asked after the stream has been read.

Data: JSONL records {"doc_id": str, "context": str, "question": str, "answers": [str]}
built from the official SQuAD 2.0 train and HotpotQA distractor train splits.
Split by doc_id; every evaluation benchmark (LongBench / LooGLE / InfiniteBench /
LongBench v2) is excluded upstream.
"""


import hashlib
import contextlib
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import longbench_prlmc_eval as pe                      # noqa: E402
from natural_merge_runtime import load_runtime, reset   # noqa: E402
from v9_geometry_50block import GEOMETRY, CHUNK_TOKENS  # noqa: E402
from longbench_memoryllm_eval import atomic_json, seed_everything   # noqa: E402


# ---------------------------------------------------------------- LoRA (no peft) ---
class LoRALinear(nn.Module):
    """Frozen base weight plus a rank-r update; `enabled` is the write/query switch."""

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        self.drop = nn.Dropout(dropout)
        dev = base.weight.device
        self.A = nn.Parameter(torch.zeros(r, base.in_features, dtype=torch.float32, device=dev))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=dev))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        self.enabled = True

    def forward(self, x):
        out = self.base(x)
        if not self.enabled:
            return out
        delta = self.drop(x).to(torch.float32) @ self.A.t()
        return out + (delta @ self.B.t()).to(out.dtype) * self.scale


def _decoder_layers(model):
    for path in ("model.layers", "model.model.layers", "transformer.h"):
        node, ok = model, True
        for part in path.split("."):
            if not hasattr(node, part):
                ok = False
                break
            node = getattr(node, part)
        if ok:
            return node
    raise AttributeError("cannot locate decoder layers; patch _decoder_layers()")


def _resolve_target(layer, name):
    parent, parts = layer, name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1], getattr(parent, parts[-1])


def attach_lora(model, last_layers=8, targets=("self_attn.q_proj", "self_attn.o_proj"),
                r=8, alpha=16, dropout=0.0):
    """Wrap the query/output projections of the last `last_layers` decoder layers.

    This reader ships some projections already wrapped (`.base_layer` +
    `.lora_dropout`), so a non-Linear target is adapted through its base layer and
    any pre-existing adapter stays in the graph untouched.
    """
    layers = list(_decoder_layers(model))
    names = []
    for index, layer in enumerate(layers):
        if index < len(layers) - last_layers:
            continue
        for name in targets:
            parent, attr, module = _resolve_target(layer, name)
            if isinstance(module, LoRALinear):
                continue
            if isinstance(module, nn.Linear):
                setattr(parent, attr, LoRALinear(module, r, alpha, dropout))
            elif isinstance(getattr(module, "base_layer", None), nn.Linear):
                module.base_layer = LoRALinear(module.base_layer, r, alpha, dropout)
            else:
                raise AttributeError(
                    f"{name} is neither nn.Linear nor a base_layer wrapper: {type(module).__name__}")
            names.append(f"layers.{index}.{name}")
    return names


def lora_modules(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def lora_named_state(model):
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[name + ".A"] = module.A.detach().cpu()
            state[name + ".B"] = module.B.detach().cpu()
    return state


def lora_parameters(model):
    return [p for m in lora_modules(model) for p in (m.A, m.B)]


def set_lora_enabled(model, flag: bool):
    for m in lora_modules(model):
        m.enabled = flag


def load_lora_state(model, payload):
    """Attach the adapter described by a saved payload and copy its weights in.

    Used by the V9 trainer so the fusion module is trained against exactly the
    read-out ability the main table will use.
    """
    attach_lora(model, payload["last_layers"], r=payload["rank"], alpha=payload["alpha"],
                dropout=float(payload.get("dropout", 0.0)))
    modules = {name: module for name, module in model.named_modules()
               if isinstance(module, LoRALinear)}
    for key, value in payload["state"].items():
        module_name, _, which = key.rpartition(".")
        if module_name not in modules:
            raise KeyError(f"LoRA target {module_name} not found in this reader")
        getattr(modules[module_name], which).data.copy_(value.to(torch.float32))
    for p in lora_parameters(model):
        p.requires_grad_(False)
    return model


# ------------------------------------------------------------------- memory stream ---
class AdapterHandle:
    """Trainable view of the attached LoRA deltas.

    The adapter is enabled only around the supervised QA read; every memory write
    runs the native reader, so the stored representation is unchanged.
    """

    def __init__(self, model):
        self.model = model
        self.names = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
        if not self.names:
            raise ValueError("no LoRA module attached")

    def active(self):
        model = self.model

        @contextlib.contextmanager
        def scope():
            set_lora_enabled(model, True)
            try:
                yield
            finally:
                set_lora_enabled(model, False)

        return scope()

    def parameters(self):
        return lora_parameters(self.model)

    def digest(self):
        handle = hashlib.sha256()
        for name, tensor in sorted(lora_named_state(self.model).items()):
            handle.update(name.encode())
            handle.update(tensor.float().numpy().tobytes())
        return handle.hexdigest()


def read_stream(manager, merge, initial, tokenizer, context, stream_tokens):
    """Write one 512-token-chunked stream at the paper geometry, as Selection.

    The write phase runs with LoRA disabled: it must reproduce the native
    injection exactly, otherwise the adapter trains on a memory state the frozen
    reader never produces at evaluation time.
    """
    reset(manager, initial)
    with torch.no_grad():
        manager._tile_free_region()
        manager._sync_active_mask()
    ids = tokenizer(context, add_special_tokens=False, truncation=False).input_ids
    if len(ids) < stream_tokens:
        return -1
    ids = ids[:stream_tokens - (stream_tokens % CHUNK_TOKENS)]
    chunks = [ids[i:i + CHUNK_TOKENS] for i in range(0, len(ids), CHUNK_TOKENS)]
    compactor = pe.build_consolidate_merge(merge, "Selection", manager)
    stream = pe.MergeStream(manager, compactor, "Selection",
                            budget_blocks=GEOMETRY["budget_blocks"],
                            window_blocks=GEOMETRY["window_blocks"],
                            target_blocks=GEOMETRY["target_blocks"],
                            trigger_blocks=GEOMETRY["trigger_blocks"])
    with torch.no_grad():
        pe.run_sample(stream, chunks)
    return len(stream.events)


def qa_ids(tokenizer, question, answer):
    prompt = f"Question: {question.strip()}\nAnswer:"
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(" " + answer.strip(), add_special_tokens=False).input_ids
    return prompt_ids, answer_ids


def answer_ce(model, prompt_ids, answer_ids, block_slots):
    """CE over the answer tokens only, conditioned on the current memory pool.

    NOTE the single integration seam: if this fork routes memory attention only
    through `inject_memory`, replace the two lines below with the call the eval
    path uses inside `pe.generate` (same attention mask, update_memory=False).
    """
    ids = torch.tensor([prompt_ids + answer_ids], dtype=torch.long, device="cuda:0")
    mask = torch.ones((1, block_slots + ids.shape[1]), dtype=torch.long, device=ids.device)
    out = model(input_ids=ids, attention_mask=mask)
    logits = out.logits if hasattr(out, "logits") else out
    lo = len(prompt_ids) - 1
    window = logits[:, lo:lo + len(answer_ids)]
    labels = torch.tensor(answer_ids, dtype=torch.long, device=logits.device)
    return F.cross_entropy(window.reshape(-1, window.shape[-1]), labels)


# --------------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/dev/shm/s1d_reader_fp16")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", default=None)
    ap.add_argument("--out", default=str(ROOT / "runs/reader_lora/lora_r8_a16.pt"))
    ap.add_argument("--stream-tokens", type=int, default=64000,
                    help="64000 -> ~8 merge events at trigger 50 / chunk 512")
    ap.add_argument("--updates", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--last-layers", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    model, tokenizer, manager, merge, initial, cfg, _ = load_runtime(
        args.model_path, checkpoint=None, train=False,
        architecture="grouped_gated_sparse_native_v9",
        budget_blocks=GEOMETRY["budget_blocks"], strict_code_hash=False)
    tokenizer.pad_token = tokenizer.eos_token
    targets = attach_lora(model, args.last_layers, r=args.rank, alpha=args.alpha,
                          dropout=args.dropout)
    params = lora_parameters(model)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    block_slots = int(model.num_blocks) * int(model.num_tokens)

    records = [json.loads(l) for l in Path(args.train_jsonl).read_text().splitlines() if l.strip()]
    random.Random(args.seed).shuffle(records)
    val = [json.loads(l) for l in Path(args.val_jsonl).read_text().splitlines() if l.strip()] \
        if args.val_jsonl else []

    report = dict(schema="reader_lora_train/1", args=vars(args), geometry=dict(GEOMETRY),
                  lora=dict(targets=targets, rank=args.rank, alpha=args.alpha,
                            dropout=args.dropout,
                            trainable_params=sum(p.numel() for p in params)),
                  updates=[], val=[], status="running")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = out_path.with_suffix(".report.json")
    atomic_json(report_path, report)

    started, step, cursor = time.monotonic(), 0, 0
    while step < args.updates:
        rec = records[cursor % len(records)]
        cursor += 1
        set_lora_enabled(model, False)
        merges = read_stream(manager, merge, initial, tokenizer, rec["context"], args.stream_tokens)
        if merges < 0:
            continue
        set_lora_enabled(model, True)
        prompt_ids, answer_ids = qa_ids(tokenizer, rec["question"], rec["answers"][0])
        loss = answer_ce(model, prompt_ids, answer_ids, block_slots)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1

        report["updates"].append(dict(step=step, loss=float(loss), merges=merges,
                                      doc_id=rec.get("doc_id"), stream_tokens=args.stream_tokens,
                                      seconds=time.monotonic() - started))
        if step % 10 == 0 or step == args.updates:
            atomic_json(report_path, report)
            print(json.dumps(report["updates"][-1]), flush=True)

        if val and step % args.val_every == 0:
            losses = []
            for vrec in val[:8]:
                set_lora_enabled(model, False)
                read_stream(manager, merge, initial, tokenizer, vrec["context"], args.stream_tokens)
                set_lora_enabled(model, True)
                p_ids, a_ids = qa_ids(tokenizer, vrec["question"], vrec["answers"][0])
                with torch.no_grad():
                    losses.append(float(answer_ce(model, p_ids, a_ids, block_slots)))
            report["val"].append(dict(step=step, ce=sum(losses) / max(1, len(losses)),
                                      n=len(losses)))
            print(json.dumps(report["val"][-1]), flush=True)

    torch.save(dict(state=lora_named_state(model), rank=args.rank, alpha=args.alpha,
                    dropout=args.dropout, last_layers=args.last_layers,
                    stream_tokens=args.stream_tokens, geometry=dict(GEOMETRY),
                    updates=args.updates, lr=args.lr), out_path)
    report["status"] = "complete"
    report["wall_seconds"] = time.monotonic() - started
    atomic_json(report_path, report)
    print("saved", out_path)


if __name__ == "__main__":
    main()
