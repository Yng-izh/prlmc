#!/usr/bin/env python3
"""Build the control checkpoint for experiment 2: same module, same residual size,
no training.

Why this exists
---------------
The trained compaction module is the only trained object in the comparison, so a
+10 pp gain could mean two very different things:

  * the module learned which direction to rotate each anchor in, or
  * rotating each anchor by ~26% of its norm in ANY direction helps, in which case
    the supervision is doing nothing and the result is an artefact of perturbing
    the graft slots.

This script produces the second case on purpose. It builds `SourceCompaction` with
the same config, random q/k/v (orthogonal, exactly as at initialisation) and a
random `out_proj`, then bisects the `out_proj` scale until the measured residual
matches the trained checkpoint's residual on a real document. The resulting
checkpoint is byte-for-byte the same architecture and the same intervention size,
with no learning in it.

Usage
-----
    python tools/randomize_compaction.py \
        --data-dir /data/.../LongBench/extracted/data --train-task musique \
        --model-path /dev/shm/s1d_reader_fp16 \
        --v10-checkpoint runs/natural_merge_v9/full_v10.pt \
        --trained runs/source_compaction/compaction_rank32.pt \
        --out runs/source_compaction/random_rank32.pt \
        --report runs/source_compaction/random_rank32.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_graft_recovery as gr                          # noqa: E402
import native_eval as nv                                  # noqa: E402
from io_utils import atomic_json, sha                     # noqa: E402
from runtime import load_runtime, reset                   # noqa: E402
from methods.prlmc.source_compaction import (             # noqa: E402
    SCHEMA, SourceCompaction, compact_anchors, split_source_anchors)
from train_core_fusion import (                           # noqa: E402
    BUDGET_BLOCKS, TARGET_BLOCKS, WINDOW_BLOCKS, chunks_of, load_documents,
    write_blocks)

SLOTS_PER_BLOCK = gr.SLOTS_PER_BLOCK


def measure(module, output, source, parts):
    with torch.no_grad():
        _, info = compact_anchors(output, source, parts, module)
    return float(info["relative_delta"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--train-task", dest="train_task", default="musique")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--v10-checkpoint", dest="v10_checkpoint", required=True)
    parser.add_argument("--trained", required=True,
                        help="the trained checkpoint whose residual size to match")
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--strict-code-hash", dest="strict_code_hash", default="true",
                        choices=("true", "false"))
    args = parser.parse_args(argv)
    if Path(args.out).exists():
        raise SystemExit(f"{args.out} exists; refuse overwrite")

    payload = torch.load(args.trained, map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    trained = SourceCompaction(**config)
    trained.load_state_dict(payload["state_dict"], strict=True)
    trained = trained.float().eval().requires_grad_(False)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    documents, _ = load_documents(tokenizer, args.data_dir, args.train_task, limit=1)
    if not documents:
        raise SystemExit("no usable document to calibrate on")
    document = documents[0]

    nv.seed_everything(args.seed)
    checkpoint_training = torch.load(args.v10_checkpoint, map_location="cpu").get("training")
    model, tokenizer, manager, merge, initial, cfg, saved = load_runtime(
        args.model_path, checkpoint=args.v10_checkpoint, train=False, architecture=None,
        budget_blocks=BUDGET_BLOCKS, strict_code_hash=args.strict_code_hash,
        training=checkpoint_training)
    merge.operator_mode = "native"
    # The trained module was built on the CPU before the reader was loaded; it has to
    # land on the same device as the window before `compact_anchors` can run.
    trained = trained.to(model.memory.device)
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
    # The target IS the trained module's residual on this document: matching it is
    # what makes the control "the same intervention size, no learning".
    target = measure(trained, output, source, parts)
    print(json.dumps(dict(document=document["id"], target_delta=target)), flush=True)

    control = SourceCompaction(**config).to(output.device).float().eval()
    with torch.no_grad():
        control.out_proj.normal_(0.0, 1.0 / max(config["rank"], 1) ** 0.5)
        # The residual saturates at `sigmoid(gate) * max_ratio`; at the trained
        # gate (0.5438) the ceiling is 0.272 and a bisection that needs 0.2636 can
        # only approach it from below. Opening the gate to 0.6 puts the ceiling at
        # 0.30, so the scale search can actually hit the target. The gate only sets
        # the magnitude, which is the thing being matched -- the direction is still
        # whatever `out_proj` says.
        control.gate_logit.fill_(0.4055)
    # `delta` saturates at `gate * max_ratio` because of the tanh, so the scale is
    # found by bisection between "far too small" and "definitely saturated" rather
    # than by dividing once.
    low, high = 0.0, 1.0
    with torch.no_grad():
        base = control.out_proj.clone()
        control.out_proj.copy_(base * high)
    while measure(control, output, source, parts) < target and high < 1e6:
        high *= 4.0
        with torch.no_grad():
            control.out_proj.copy_(base * high)
    for _ in range(24):
        mid = 0.5 * (low + high)
        with torch.no_grad():
            control.out_proj.copy_(base * mid)
        if measure(control, output, source, parts) < target:
            low = mid
        else:
            high = mid
    scale = high
    with torch.no_grad():
        control.out_proj.copy_(base * scale)
    achieved = measure(control, output, source, parts)
    control = control.cpu()
    print(json.dumps(dict(scale=scale, achieved_delta=achieved,
                          relative_error=abs(achieved - target) / max(target, 1e-9))),
          flush=True)

    report = dict(
        schema=SCHEMA + "/control", status="complete",
        purpose=("untrained module with the trained residual size: separates 'the "
                 "module learned the direction' from 'any rotation of this size helps'"),
        config=config, seed=args.seed, calibrated_on=document["id"],
        trained_checkpoint=args.trained,
        trained_checkpoint_sha256=sha(args.trained),
        trained_delta=target, target_delta=target, achieved_delta=achieved,
        out_proj_scale=scale,
        training_performed=False, optimizer_created=False,
        v10_checkpoint=args.v10_checkpoint,
        v10_checkpoint_sha256=sha(args.v10_checkpoint))
    checkpoint = dict(
        schema=SCHEMA + "/checkpoint", state_dict=control.state_dict(),
        config=config,
        training=dict(task="none (control)", updates=0, lr=None, seed=args.seed,
                      control="random out_proj, calibrated residual size",
                      target_delta=target, achieved_delta=achieved,
                      trained_checkpoint_sha256=report["trained_checkpoint_sha256"],
                      checkpoint=args.v10_checkpoint,
                      checkpoint_sha256=report["v10_checkpoint_sha256"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out)
    report["out"] = args.out
    report["out_sha256"] = sha(args.out)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.report, report)
    print(json.dumps(dict(status="complete", out=args.out, achieved_delta=achieved),
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
