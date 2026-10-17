"""Which blocks actually carry the probes' answers?

E6a's screen reported, on seven separate windows:

    no-mem 0/20   full 20/20   select 20/20   lost 0

The pool is required (0/20 with it zeroed) and selection keeps only 9 of 19 blocks,
yet nothing is lost. Two explanations fit that pattern and they call for opposite
responses:

  (a) the dropped blocks are still being read -- the compaction is not actually
      removing them, so `select` is measuring the uncompacted pool; or
  (b) the answers are duplicated in the kept blocks, so there is genuinely nothing
      to lose and the window simply cannot test recovery.

They are separable without touching the compactor, by zeroing one group of blocks
in place and re-scoring:

  drop-zeroed   zero the 10 blocks selection discards. If accuracy falls, the
                answers live in the dropped blocks -- and `select` scoring 20/20
                would be a measurement bug, not a fact about the compactor.
  keep-zeroed   zero the 9 blocks selection keeps. If accuracy holds, the answer
                is in the dropped blocks alone, confirming the same conclusion.

Both ablations read the *uncompacted* pool, so neither depends on `_compacted_pool`
or the active mask. That is deliberate: the point is to test the two explanations
without reusing the machinery under suspicion.
"""
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "9")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/share/latent/PRLM-C")
sys.path.insert(0, "/tmp")

import torch  # noqa: E402  (after CUDA_VISIBLE_DEVICES)

import e6a_overfit as E  # noqa: E402
import yaml  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM  # noqa: E402
from methods.prlmc.layout import PRLMCConfig, ProtectedResidualMemory  # noqa: E402

WINDOW = (3, 17890)
DEVICE = "cuda:0"


def main():
    raw = yaml.safe_load(open(E.CONFIG))
    config = dict(raw["prlmc"])
    config["device"] = DEVICE
    config["dtype"] = "float16"
    config["stats_dir"] = "./outputs/prlmc-diag"
    cfg = PRLMCConfig.from_dict(config)

    model = MemoryLLM.from_pretrained(E.MODEL, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = model.to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(E.MODEL)
    manager = ProtectedResidualMemory(model, tokenizer, cfg)

    row, start = WINDOW
    lines = E.read_fact_lines(row)
    window_slots = E.WINDOW_BLOCKS * manager.block
    covered = E.build_window(manager, lines, window_slots, start_line=start)
    indices = E.probe_indices(covered, window_slots)

    line_to_block = {}
    for block_index, (lo, hi, _before, _width) in enumerate(covered):
        for line in range(lo, hi):
            line_to_block[line] = block_index

    begin = manager.total - manager.used
    source = manager._memory[:, begin : begin + window_slots, :].to(device=DEVICE)
    weights = manager._window_importance(begin, window_slots)
    dropped = E.dropped_block_indices(manager, source, weights)
    kept = sorted(set(range(E.WINDOW_BLOCKS)) - set(dropped))
    del source, weights
    torch.cuda.empty_cache()

    probes = E.probes_from_lines(
        lines, indices, 20, dropped_blocks=set(dropped), line_to_block=line_to_block
    )
    print("[diag] row %d start %d: dropped=%s" % (row, start, dropped), flush=True)
    print("[diag] kept=%s  probes=%d" % (kept, len(probes)), flush=True)

    block = manager.block
    storage = manager.model.memory.data

    def zero_blocks(blocks):
        for b in blocks:
            lo = begin + b * block
            storage[:, lo : lo + block, :] = 0

    def score(label):
        torch.cuda.empty_cache()
        result = E.score_uncompacted(manager, probes, 64)
        print("[diag] %-14s acc %.3f (%d/%d)" % (
            label, result["accuracy"],
            sum(bool(v) for v in result["correct"].values()), result["n"]), flush=True)
        return result

    intact = score("intact")

    snapshot = storage[:, begin : begin + window_slots, :].clone()
    zero_blocks(dropped)
    drop_zeroed = score("drop-zeroed")
    storage[:, begin : begin + window_slots, :] = snapshot

    zero_blocks(kept)
    keep_zeroed = score("keep-zeroed")
    storage[:, begin : begin + window_slots, :] = snapshot
    del snapshot
    torch.cuda.empty_cache()

    summary = {
        "window": {"row": row, "start": start},
        "dropped": dropped, "kept": kept, "n_probes": len(probes),
        "intact": intact["correct"],
        "drop_zeroed": drop_zeroed["correct"],
        "keep_zeroed": keep_zeroed["correct"],
    }
    with open("/tmp/e6a_diag_blocks.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print("\n[diag] intact %d/%d, drop-zeroed %d/%d, keep-zeroed %d/%d" % (
        sum(intact["correct"].values()), len(probes),
        sum(drop_zeroed["correct"].values()), len(probes),
        sum(keep_zeroed["correct"].values()), len(probes)), flush=True)
    print("[diag] written to /tmp/e6a_diag_blocks.json", flush=True)


if __name__ == "__main__":
    main()
