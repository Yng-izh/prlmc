"""Does the memory bias reach scaled_dot_product_attention at all?

Three measurements on the same compacted pool:

    baseline (mask on, 2816 active)   20/20
    empty-mask (0 active)             20/20   <-- must equal `no-mem` (0/20)
    zero-carriers                     0/20

An all-inactive pool removes the pool from attention, so the reader should fall
back to its no-memory behaviour. It does not. Either `_memory_bias_for` returns
None, or the bias is dropped before SDPA.

This wraps `torch.nn.functional.scaled_dot_product_attention` -- which the fork
resolves by attribute at call time -- and reports, for each call, whether an
attn_mask arrived and how many pool positions it blocks. That distinguishes the
two explanations directly instead of inferring them.
"""
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "9")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/share/latent/PRLM-C")
sys.path.insert(0, "/tmp")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import e6a_overfit as E  # noqa: E402
import yaml  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM  # noqa: E402
from methods.prlmc.consolidate import CarrierResidualCompactor  # noqa: E402
from methods.prlmc.layout import PRLMCConfig, ProtectedResidualMemory  # noqa: E402

WINDOW = (3, 17890)
DEVICE = "cuda:0"
CALLS = []
_ORIGINAL_SDPA = F.scaled_dot_product_attention


def traced_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    record = {"attn_mask_none": attn_mask is None}
    if attn_mask is not None:
        record["shape"] = list(attn_mask.shape)
        record["dtype"] = str(attn_mask.dtype)
        record["is_bool"] = attn_mask.dtype == torch.bool
        last = attn_mask.reshape(-1, attn_mask.shape[-1])
        record["rows"] = int(last.shape[0])
        record["cols"] = int(last.shape[-1])
        if attn_mask.dtype == torch.bool:
            # A bool mask means "this key is blocked" where True. The shipped
            # branch builds exactly this, and it carries no room for a bias.
            blocked_rows = last.all(dim=-1)
            record["pool_all_blocked"] = int(
                last[:, 1:12801].all(dim=-1).sum().item()
            )
            record["blocked_frac"] = float((~last).float().mean().item())
        else:
            floor = torch.finfo(attn_mask.dtype).min
            record["min_rows"] = int((last == floor).all(dim=-1).sum().item())
            record["pool_all_blocked"] = int(
                (last[:, 1:12801] == floor).all(dim=-1).sum().item()
            )
            record["blocked_frac"] = float((last == floor).float().mean().item())
    CALLS.append(record)
    return _ORIGINAL_SDPA(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                          is_causal=is_causal, **kwargs)


F.scaled_dot_product_attention = traced_sdpa
torch.nn.functional.scaled_dot_product_attention = traced_sdpa


def main():
    raw = yaml.safe_load(open(E.CONFIG))
    config = dict(raw["prlmc"])
    config["device"] = DEVICE
    config["dtype"] = "float16"
    config["stats_dir"] = "./outputs/prlmc-diagmask4"
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

    for label, use_mask in (("mask-on", mask), ("mask-all-off", torch.zeros_like(mask))):
        CALLS.clear()
        result = E.score_probes(manager, probes, pool, 64, mask=use_mask)
        acc = sum(bool(v) for v in result["correct"].values())
        n_calls = len(CALLS)
        with_mask = sum(1 for c in CALLS if not c["attn_mask_none"])
        blocked = max((c.get("pool_all_blocked", -1) for c in CALLS), default=-1)
        print("[m4] %-13s acc %2d/%d  sdpa_calls %d  with_mask %d  max blocked rows %d"
              % (label, acc, result["n"], n_calls, with_mask, blocked), flush=True)
        if CALLS:
            print("[m4]     first call: %s" % (CALLS[0],), flush=True)

    summary = {"dropped": dropped, "calls": CALLS[:64], "n_calls": len(CALLS)}
    with open("/tmp/e6a_diag_mask4.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print("[m4] written to /tmp/e6a_diag_mask4.json", flush=True)


if __name__ == "__main__":
    main()
