"""Is the active mask read at all, and do the carriers carry the answers?

Two prior diagnostics came back uninformative, for opposite reasons:

  `e6a_diag_blocks.py` zeroed blocks in the *uncompacted* pool. Both ablations
  scored 0/20. Zeroing ten of nineteen blocks is far out of distribution -- the
  pool has a `_tile_free_region` specifically because uninitialised or zero
  space breaks the reader -- so both numbers measure OOD breakage, not where the
  information lives.

  `e6a_diag_mask.py` scored the compacted pool with the mask on and off and got
  20/20 both ways, while the mask correctly marked all 2560 dropped slots
  inactive. If the mask did anything, turning it off should have changed the
  read.

This settles both with two ablations that avoid zeros in live slots:

  empty-mask     mark *every* slot inactive. If accuracy survives, the mask is
                 not reaching attention at all and `select` has been scoring the
                 uncompacted pool.
  zero-carriers  overwrite only the carrier region [new_start, new_start+target)
                 with zeros. Those slots are supposed to be the entire output of
                 compaction. If accuracy survives, the answers are coming from
                 somewhere else in the pool -- the tiled free region, which
                 `_compacted_pool` copies verbatim from the base pool and which
                 holds copies of the dropped blocks.
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
    config["stats_dir"] = "./outputs/prlmc-diagmask2"
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
    print("[m2] dropped=%s probes=%d" % (dropped, len(probes)), flush=True)

    compactor = CarrierResidualCompactor(
        num_layers=manager.num_layers, hidden=manager.hidden, block=manager.block,
        rank=int(cfg.carrier_rank), gamma_max=float(cfg.carrier_gamma_max),
    ).to(device=DEVICE, dtype=torch.float32)
    compactor.eval()

    total = manager.total
    target = E.TARGET_BLOCKS * manager.block
    new_start = total - (manager.used - window_slots + target)

    def score(pool, mask, label):
        torch.cuda.empty_cache()
        r = E.score_probes(manager, probes, pool, 64, mask=mask)
        print("[m2] %-16s acc %.3f (%d/%d)" % (
            label, r["accuracy"], sum(bool(v) for v in r["correct"].values()), r["n"]), flush=True)
        return r

    pool, mask = E.candidate_pool(manager, compactor, DEVICE)
    print("[m2] carriers at [%d, %d); new_start=%d" % (new_start, new_start + target, new_start), flush=True)

    base = score(pool, mask, "baseline")
    empty = score(pool, torch.zeros_like(mask), "empty-mask")

    # Zero only the carrier region. The tiled free region is left untouched, so
    # this is not the blanket-zero OOD failure that made `drop-zeroed` useless.
    zero_pool = pool.clone()
    zero_pool[:, new_start : new_start + target, :] = 0
    nocarr = score(zero_pool, mask, "zero-carriers")
    del zero_pool
    torch.cuda.empty_cache()

    # And the converse: zero the free region (everything outside the carriers and
    # the anchor) while keeping the carriers. If the answers are in the tiled
    # filler, this is what destroys them.
    fill_pool = pool.clone()
    free = torch.ones(total, dtype=torch.bool, device=DEVICE)
    free[: manager.layout.k_anchor] = False
    free[new_start : new_start + target] = False
    fill_pool[:, free, :] = 0
    nofill = score(fill_pool, mask, "zero-freeregion")
    del fill_pool
    torch.cuda.empty_cache()

    summary = {
        "window": {"row": row, "start": start},
        "dropped": dropped, "n_probes": len(probes),
        "new_start": new_start, "target": target,
        "baseline": base["correct"], "empty_mask": empty["correct"],
        "zero_carriers": nocarr["correct"], "zero_free": nofill["correct"],
    }
    with open("/tmp/e6a_diag_mask2.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print("[m2] written to /tmp/e6a_diag_mask2.json", flush=True)


if __name__ == "__main__":
    main()
