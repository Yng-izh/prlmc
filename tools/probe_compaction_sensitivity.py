#!/usr/bin/env python3
"""Why did the candidate swap leave all 158 answers unchanged?

`tools/eval_counterfactual_candidates.py` replaced the 2,200 candidate slots with
another document's leftovers and got a byte-for-byte identical score: same QA F1
on all 158 rows, 0 rescued, 0 harmed. A flat ablation has two readings -- the
module does not need the candidates, or the swap never reached the module -- and
they are told apart by measuring the module's own input-output sensitivity
instead of its accuracy.

Three inputs are compared on each document, all through the same anchors:

    real      the sample's own leftovers                   (experiment 2)
    donor     another document's leftovers, same shape     (the ablation)
    shuffled  the same rows in a different order           (wiring check only)

`shuffled` is not a sensitivity measurement: a candidate row carries its key and
its value together, so permuting the rows permutes both and the attention output
is invariant by construction. It is here to prove the probe is wired correctly --
the module is a set function, and a variant that is NOT invariant would mean the
measurement is broken.

Six things are recorded per pair:

    input_change   RMS(alt - real) / RMS(real) over the candidate slots
    context_change the same ratio one stage earlier, on the attention output
    raw_change     the same ratio after `gelu` and `out_proj`
    output_change  RMS(delta_alt - delta_real) / RMS(delta_real)
    sign_agreement fraction of delta entries that keep their sign
    saturation     fraction of entries with |tanh(raw/rms)| > 0.99, read back as
                   |delta| / (gate * rms) so the module is not re-implemented

Attention statistics on the real candidates are recorded because they decide how
to read the attenuation: a softmax that is uniform over the block makes the
module a block average, a concentrated one makes it a slot lookup, and a `tanh`
already pinned at +-1 turns whatever it reads into `gate * rms * sign(...)`,
which carries the anchor's norm and almost none of the candidate's content. The
three stages are reported separately so the attenuation is attributed to a stage
instead of to the module as a whole.

No reader generation runs here: this is a property of the 0.52 M-parameter module
on real windows, not of the answers.

    python tools/probe_compaction_sensitivity.py \
        --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
        --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
        --model-path /dev/shm/s1d_reader_fp16 \
        --checkpoint runs/natural_merge_v9/full_v10.pt \
        --compaction runs/source_compaction/compaction_rank32.pt \
        --cuda-device 0 --max-samples 8 \
        --out reports/exp2_source_compaction/probe_candidate_sensitivity.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_counterfactual_candidates as cf                # noqa: E402
import eval_graft_recovery as gr                           # noqa: E402
import eval_source_compaction as es                        # noqa: E402
import native_eval as nv                                   # noqa: E402
from io_utils import atomic_json, sha                      # noqa: E402
from runtime import load_runtime, reset                    # noqa: E402
from methods.prlmc.source_compaction import (              # noqa: E402
    SCHEMA, SourceCompaction, compact_window)

BLOCK_TOKENS = gr.BLOCK_TOKENS
SCHEMA_RESULTS = "compaction_sensitivity/1"


def shuffle_within_block(candidates, parts, seed):
    """The same candidate rows in a different order inside each dropped block."""
    out = candidates.clone()
    generator = torch.Generator().manual_seed(int(seed))
    blocks = parts["candidate_blocks"].cpu()
    for block in parts["dropped_blocks"]:
        rows = torch.nonzero(blocks == int(block)).flatten()
        order = rows[torch.randperm(int(rows.numel()), generator=generator)]
        out[:, rows] = candidates[:, order.to(candidates.device)]
    return out


def _accumulate(store, prefix, values):
    for key, value in values.items():
        store.setdefault(prefix + key, []).append(float(value))


def measure(module, round1, donor, seed, sample_id, layer_chunk=8):
    """Input change against output change, per candidate variant."""
    parts = round1["parts"]
    anchors = round1["output"].index_select(1, parts["anchor_offsets"]).detach()
    real = round1["source"].index_select(1, parts["candidate_positions"])
    variants = dict(donor=donor["slots"].index_select(1, cf.donor_map(parts, donor, seed)),
                    shuffled=shuffle_within_block(real, parts, seed))
    gate = float(torch.sigmoid(module.gate_logit.detach().float())) * module.max_ratio
    mask = parts["same_block"]
    store = {}
    per_layer = {}
    with torch.no_grad():
        for start in range(0, anchors.shape[0], layer_chunk):
            stop = min(start + layer_chunk, anchors.shape[0])
            anchor = anchors[start:stop]
            rms = anchor.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            base = module(anchor, real[start:stop], mask).float()
            # Where the candidate content dies, if it dies: context -> raw -> delta.
            query = F.linear(anchor.float(), module.q_proj.float())

            def stages(candidates):
                key = F.linear(candidates.float(), module.k_proj.float())
                logits = torch.matmul(query, key.transpose(1, 2)) * module.scale
                logits = logits.masked_fill(~mask[None], -1e4)
                weights = torch.softmax(logits, dim=-1)
                value = F.linear(candidates.float(), module.v_proj.float())
                context = torch.matmul(weights, value)
                return context, F.linear(F.gelu(context), module.out_proj.float()), weights

            context_real, raw_real, _ = stages(real[start:stop])
            for name, candidates in variants.items():
                alt = module(anchor, candidates[start:stop], mask).float()
                context_alt, raw_alt, _ = stages(candidates[start:stop])
                _accumulate(store, name + "/", dict(
                    input_change=(candidates[start:stop].float() - real[start:stop].float())
                                 .square().sum().sqrt() / real[start:stop].float().square().sum().sqrt(),
                    context_change=(context_alt - context_real).square().sum().sqrt()
                                   / context_real.square().sum().sqrt().clamp_min(1e-12),
                    raw_change=(raw_alt - raw_real).square().sum().sqrt()
                               / raw_real.square().sum().sqrt().clamp_min(1e-12),
                    output_change=(alt - base).square().sum().sqrt()
                                  / base.square().sum().sqrt().clamp_min(1e-12),
                    sign_agreement=(alt.sign() == base.sign()).float().mean()))
            saturation = ((base.abs() / (gate * rms)).clamp(max=1.0) > 0.99).float().mean()
            _accumulate(store, "real/", dict(saturation=saturation,
                                             relative_delta=base.square().sum().sqrt()
                                             / anchor.float().square().sum().sqrt()))
            # Attention statistics of the real pass: is the softmax uniform, i.e.
            # does the module read a block average instead of a slot?
            weights = stages(real[start:stop])[2]
            entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1)
            _accumulate(store, "attention/", dict(
                entropy_ratio=(entropy / float(torch.tensor(weights.shape[-1]).log())).mean(),
                max_weight=weights.max(dim=-1).values.mean()))
            # How far apart the 220 candidates of a block are in the module's own
            # value space: permutation invariance requires this to be near zero,
            # because the attention weights are demonstrably not uniform.
            value = F.linear(real[start:stop].float(), module.v_proj.float())
            spreads = []
            for block in parts["dropped_blocks"]:
                rows = torch.nonzero(parts["candidate_blocks"] == int(block)).flatten()
                chunk = value[:, rows.to(value.device)]
                centre = chunk.mean(dim=1, keepdim=True)
                spreads.append(float((chunk - centre).square().sum().sqrt()
                                     / chunk.square().sum().sqrt().clamp_min(1e-12)))
            _accumulate(store, "value/", dict(
                spread_within_block=sum(spreads) / len(spreads)))
            for offset in range(stop - start):
                layer = start + offset
                single = module(anchor[offset:offset + 1],
                                variants["donor"][layer:layer + 1], mask)
                denominator = base[offset:offset + 1].square().sum().sqrt().clamp_min(1e-12)
                per_layer[layer] = dict(
                    output_change=float((single - base[offset:offset + 1]).square()
                                        .sum().sqrt() / denominator))

    fused, _ = compact_window(round1["output"], round1["source"], parts, module)
    hybrid = round1["source"].index_copy(1, parts["candidate_positions"],
                                         variants["donor"])
    counterfactual, _ = compact_window(round1["output"], hybrid, parts, module)
    offsets = parts["anchor_offsets"]
    left = fused.index_select(1, offsets)
    right = counterfactual.index_select(1, offsets)
    keep = torch.ones(fused.shape[1], dtype=torch.bool, device=offsets.device)
    keep[offsets] = False
    window = dict(
        entry_change=float((left != right).float().mean()),
        relative_change=float((left.float() - right.float()).square().sum().sqrt()
                              / left.float().square().sum().sqrt()),
        core_bit_identical=bool(torch.equal(fused[:, keep], counterfactual[:, keep])))
    summary = {key: round(sum(values) / len(values), 6) for key, values in store.items()}
    summary["per_layer_output_change"] = {str(k): round(v["output_change"], 6)
                                          for k, v in sorted(per_layer.items())}
    return dict(id=sample_id, donor_id=donor["id"], gate=round(gate, 6),
                metrics=summary, window=window)


def evaluate(args):
    facts = gr.load_supporting_facts(args.supporting_facts_glob)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    nv.seed_everything(args.seed)
    checkpoint_training = torch.load(args.checkpoint, map_location="cpu").get("training")
    model, tokenizer, manager, merge, initial, cfg, saved = load_runtime(
        args.model_path, checkpoint=args.checkpoint, train=False, architecture=None,
        budget_blocks=gr.BUDGET_BLOCKS, strict_code_hash=args.strict_code_hash,
        training=checkpoint_training)
    merge.operator_mode = "native"
    payload = torch.load(args.compaction, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA + "/checkpoint":
        raise ValueError(f"{args.compaction} is not a {SCHEMA} checkpoint")
    module_config = dict(payload["config"])
    module = SourceCompaction(**module_config)
    module.load_state_dict(payload["state_dict"], strict=True)
    module = module.to(model.memory.device).eval().requires_grad_(False)

    rows, prepared, excluded = gr.prepare_rows(args, tokenizer, facts)
    if len(prepared) < 2:
        raise ValueError("the probe needs two documents: the donor is the previous one")
    output = Path(args.out)
    if output.exists():
        raise ValueError(f"{output} exists; delete it first or choose another --out")
    report = dict(schema=SCHEMA_RESULTS, status="running", task=args.task,
                  counts=dict(rows=len(rows), eligible=len(prepared), excluded=len(excluded)),
                  protocol=dict(
                      experiment="2-counterfactual: module sensitivity to its candidates",
                      variants=dict(
                          real="the sample's own leftovers",
                          donor="another document's leftovers, same shape",
                          shuffled="the same rows in a different order; a row carries "
                                   "its key and its value, so this is exactly invariant "
                                   "by construction and only checks the wiring"),
                      readouts_per_document=3, reader_generation=False,
                      checkpoint=args.compaction, checkpoint_sha256=sha(args.compaction),
                      operator=args.checkpoint, seed=args.seed,
                      max_samples=args.max_samples,
                      entrypoint_sha256=sha(__file__)),
                  documents=[], training_performed=False)
    atomic_json(output, report)

    donor = None
    for position, sample in enumerate(prepared):
        nv.seed_everything(args.seed)
        reset(manager, initial)
        if donor is None:
            donor = cf.build_donor(manager, merge, module, prepared[1], args)
            reset(manager, initial)
            nv.seed_everything(args.seed)
        chunks = [sample["write_ids"][index:index + BLOCK_TOKENS]
                  for index in range(0, len(sample["write_ids"]), BLOCK_TOKENS)]
        with torch.no_grad():
            es.write_blocks(manager, merge, chunks[:args.window_blocks])
            round1 = es.compress_window(manager, merge, module, args.window_blocks,
                                        args.target_blocks)
            result = measure(module, round1, donor, args.seed * 100003 + position,
                             sample["id"])
        next_donor = dict(id=sample["id"],
                          slots=round1["source"].index_select(
                              1, round1["parts"]["candidate_positions"]),
                          blocks=round1["parts"]["candidate_blocks"],
                          dropped_blocks=round1["parts"]["dropped_blocks"])
        report["documents"].append(result)
        atomic_json(output, report)
        print(json.dumps(dict(document=position + 1, total=len(prepared), **result)),
              flush=True)
        donor = next_donor

    keys = sorted(report["documents"][0]["metrics"])
    report["summary"] = {key: round(sum(document["metrics"][key]
                                        for document in report["documents"])
                                    / len(report["documents"]), 6)
                         for key in keys if not isinstance(
                             report["documents"][0]["metrics"][key], dict)}
    report["summary"]["window_entry_change"] = round(
        sum(document["window"]["entry_change"] for document in report["documents"])
        / len(report["documents"]), 6)
    report["summary"]["window_relative_change"] = round(
        sum(document["window"]["relative_change"] for document in report["documents"])
        / len(report["documents"]), 6)
    report["status"] = "complete"
    atomic_json(output, report)
    print(json.dumps(dict(status="complete", out=str(output), summary=report["summary"]),
                     indent=1))
    return report


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--task", default="hotpotqa", choices=("hotpotqa",))
    p.add_argument("--supporting-facts-glob", dest="supporting_facts_glob", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--compaction", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--window-blocks", dest="window_blocks", type=int,
                   default=gr.WINDOW_BLOCKS)
    p.add_argument("--target-blocks", dest="target_blocks", type=int,
                   default=gr.TARGET_BLOCKS)
    p.add_argument("--sample-ids", dest="sample_ids", default=None)
    p.add_argument("--max-samples", dest="max_samples", type=int, default=8)
    p.add_argument("--strict-code-hash", dest="strict_code_hash", default="true",
                   choices=("true", "false"))
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.data_dir or not args.checkpoint or not args.compaction or not args.out:
        raise SystemExit("--data-dir, --checkpoint, --compaction and --out are required")
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
