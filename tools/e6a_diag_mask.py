"""Does the active mask actually remove the dropped blocks from the read?

`e6a_diag_blocks.py` established, on the uncompacted pool:

    intact 20/20   drop-zeroed 0/20   keep-zeroed 0/20

So zeroing the ten blocks selection discards destroys every answer. Yet the
compacted pool -- which keeps only the other nine -- scored 20/20. Those two
facts cannot both hold unless the compacted pool the scorer read still contained
the dropped blocks' content, i.e. the mask did not take effect.

This isolates the mask from the placement. The *same* compacted pool is scored
twice, once with the mask and once without:

  mask=on    what the run believes it measures
  mask=off   what the reader sees when the freed slots stay active

If the two agree, the mask is inert and `select` has been measuring the
uncompacted pool all along. The pooled values and the placement are identical
across the two calls, so the mask is the only variable.
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
from methods.prlmc.consolidate import CarrierResidualCompactor  # noqa: E402
from methods.prlmc.layout import PRLMCConfig, ProtectedResidualMemory  # noqa: E402

WINDOW = (3, 17890)
DEVICE = "cuda:0"


def main():
    raw = yaml.safe_load(open(E.CONFIG))
    config = dict(raw["prlmc"])
    config["device"] = DEVICE
    config["dtype"] = "float16"
    config["stats_dir"] = "./outputs/prlmc-diagmask"
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
    del source, weights
    torch.cuda.empty_cache()

    probes = E.probes_from_lines(
        lines, indices, 20, dropped_blocks=set(dropped), line_to_block=line_to_block
    )
    print("[mask] dropped=%s probes=%d" % (dropped, len(probes)), flush=True)

    compactor = CarrierResidualCompactor(
        num_layers=manager.num_layers,
        hidden=manager.hidden,
        block=manager.block,
        rank=int(cfg.carrier_rank),
        gamma_max=float(cfg.carrier_gamma_max),
    ).to(device=DEVICE, dtype=torch.float32)
    compactor.eval()

    print("[mask] manager.active_memory_mask=%s" % manager.active_memory_mask, flush=True)

    pool, mask = E.candidate_pool(manager, compactor, DEVICE)
    print("[mask] mask is None: %s" % (mask is None), flush=True)
    if mask is not None:
        active = int(mask.sum())
        print("[mask] active slots %d / %d (inactive %d)"
              % (active, mask.numel(), mask.numel() - active), flush=True)
        # Where the dropped blocks' content sits in the compacted pool, and
        # whether the mask covers it.
        new_start = manager.total - (manager.used - window_slots + E.TARGET_BLOCKS * manager.block)
        print("[mask] carriers placed at [%d, %d)" % (new_start, new_start + E.TARGET_BLOCKS * manager.block),
              flush=True)
        old_lo = begin + dropped[0] * manager.block
        old_hi = begin + (dropped[-1] + 1) * manager.block
        print("[mask] dropped-block slots in base pool: [%d, %d)" % (old_lo, old_hi), flush=True)
        if old_lo < manager.total:
            covered_flags = mask[old_lo : min(old_hi, manager.total)]
            print("[mask] those slots active in mask: %d / %d"
                  % (int(covered_flags.sum()), covered_flags.numel()), flush=True)

    with_mask = E.score_probes(manager, probes, pool, 64, mask=mask)
    without_mask = E.score_probes(manager, probes, pool, 64, mask=None)
    del pool, mask
    torch.cuda.empty_cache()

    def show(label, r):
        print("[mask] %-13s acc %.3f (%d/%d)" % (
            label, r["accuracy"], sum(bool(v) for v in r["correct"].values()), r["n"]), flush=True)

    show("compacted+on", with_mask)
    show("compacted+off", without_mask)

    summary = {
        "window": {"row": row, "start": start},
        "dropped": dropped, "n_probes": len(probes),
        "mask_used": bool(manager.active_memory_mask),
        "with_mask": with_mask["correct"],
        "without_mask": without_mask["correct"],
    }
    with open("/tmp/e6a_diag_mask.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print("[mask] written to /tmp/e6a_diag_mask.json", flush=True)


if __name__ == "__main__":
    main()
