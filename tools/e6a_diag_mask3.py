"""Why is `_memory_bias_for` returning None, making the mask inert?

`e6a_diag_mask2.py` scored the same compacted pool three ways:

    baseline (mask on)   20/20
    empty-mask (all -inf) 20/20   <-- impossible if the bias were applied
    zero-carriers         0/20

An all-masked pool must behave like no memory, and `no-mem` is 0/20. So the bias
never reaches attention. `_memory_bias_for` returns None on exactly two
conditions, checked here by wrapping `MemoryLLM.forward` and recording:

  * `is_injection` -- True skips the bias entirely, and also switches the read to
    the trailing `num_tokens` block only
  * `memory_active` -- None when the buffer is absent or all-True

The wrapper prints the first few calls, which is enough: if `is_injection` is True
for the scoring forward, the mask is unreachable by construction and the fix is to
publish the mask through a path the injection branch honours.
"""
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "9")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/share/latent/PRLM-C")
sys.path.insert(0, "/tmp")

import torch  # noqa: E402

import e6a_overfit as E  # noqa: E402
import yaml  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM  # noqa: E402
from methods.prlmc.consolidate import CarrierResidualCompactor  # noqa: E402
from methods.prlmc.layout import PRLMCConfig, ProtectedResidualMemory  # noqa: E402

WINDOW = (3, 17890)
DEVICE = "cuda:0"

RECORDS = []
_ORIGINAL = MemoryLLM.forward


def traced(self, *args, **kwargs):
    mask = getattr(self, "memory_active_mask", None)
    RECORDS.append({
        "is_injection": kwargs.get("is_injection"),
        "output_delta_memory": kwargs.get("output_delta_memory"),
        "input_ids_len": int(kwargs["input_ids"].shape[1]) if kwargs.get("input_ids") is not None else None,
        "mask_present": mask is not None,
        "mask_all": bool(mask.all().item()) if mask is not None else None,
        "mask_active": int(mask.sum().item()) if mask is not None else None,
        "mask_len": int(mask.numel()) if mask is not None else None,
    })
    return _ORIGINAL(self, *args, **kwargs)


MemoryLLM.forward = traced


def main():
    raw = yaml.safe_load(open(E.CONFIG))
    config = dict(raw["prlmc"])
    config["device"] = DEVICE
    config["dtype"] = "float16"
    config["stats_dir"] = "./outputs/prlmc-diagmask3"
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
    for block_index, (lo, hi, _b, _w) in enumerate(covered):
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

    compactor = CarrierResidualCompactor(
        num_layers=manager.num_layers, hidden=manager.hidden, block=manager.block,
        rank=int(cfg.carrier_rank), gamma_max=float(cfg.carrier_gamma_max),
    ).to(device=DEVICE, dtype=torch.float32)
    compactor.eval()

    pool, mask = E.candidate_pool(manager, compactor, DEVICE)
    total = manager.total
    target = E.TARGET_BLOCKS * manager.block
    new_start = total - (manager.used - window_slots + target)

    RECORDS.clear()
    with_mask = E.score_probes(manager, probes, pool, 64, mask=mask)
    masked_records = list(RECORDS)

    # The discriminator: with the mask as published, blank the dropped blocks
    # *inside the compacted pool*. If the mask really hides them, this changes
    # nothing; if the reader is seeing them, accuracy collapses.
    dropped_pool = pool.clone()
    lo = begin + dropped[0] * manager.block
    hi = begin + (dropped[-1] + 1) * manager.block
    dropped_pool[:, lo:hi, :] = 0
    RECORDS.clear()
    with_mask_dropped_zeroed = E.score_probes(manager, probes, dropped_pool, 64, mask=mask)
    del dropped_pool
    torch.cuda.empty_cache()

    def acc(r):
        return sum(bool(v) for v in r["correct"].values())

    print("[m3] dropped blocks [%d, %d); carriers [%d, %d)"
          % (lo, hi, new_start, new_start + target), flush=True)
    print("[m3] masked                %d/%d" % (acc(with_mask), with_mask["n"]), flush=True)
    print("[m3] masked + dropped=0    %d/%d" % (acc(with_mask_dropped_zeroed), with_mask_dropped_zeroed["n"]),
          flush=True)
    print("[m3] --- forward records for the masked scoring pass (first 3) ---", flush=True)
    for record in masked_records[:3]:
        print("[m3] %s" % (record,), flush=True)

    summary = {
        "dropped": dropped, "new_start": new_start, "target": target,
        "masked": with_mask["correct"],
        "masked_dropped_zeroed": with_mask_dropped_zeroed["correct"],
        "forward_records": masked_records[:5],
    }
    with open("/tmp/e6a_diag_mask3.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print("[m3] written to /tmp/e6a_diag_mask3.json", flush=True)


if __name__ == "__main__":
    main()
