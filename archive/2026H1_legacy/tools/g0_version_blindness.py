"""G0: version-blindness of similarity in the memory pool (S1-SDC gate 0).

The claim to test (S1-SDC.md section 2): a superseded fact and the current fact
for the same entity are near-duplicates in the pool's representation space, so
any similarity-based selection structurally cannot tell which one to keep. If
the cross-layer cosine between such blocks is *not* near 1, the SDC motivation
collapses and the whole plan stops.

Method
------
Scan every distinct context (row 0-3 of Conflict_Resolution; rows 4-7 are
byte-duplicates) and build one live 19-block window per context exactly the way
E6a/E6b did (anchor + append-only writes, no compaction). Window-local version
pairs (same template+subject, >= 2 occurrences inside the window) are pooled
across the four contexts, because a single window carries as few as 3 pairs.

For every (current, superseded) pair of the same subject, and for a matched
control set of different-subject pairs from the same window, compute the cosine
between the two blocks' mean-pooled pool states, per layer and averaged over
layers. Cosines are always computed inside one window's own block space.

PASS = same-subject mean cosine >= 0.95 (near-duplicate confirmed, similarity
selection is version-blind). FAIL = materially below 1 -> stop-loss.

Run on v100-2:
  /share/latent/mllm_env/bin/python /share/latent/PRLM-C/tools/g0_version_blindness.py \
      --rows 0,1,2,3 --out /share/latent/PRLM-C/runs/g0_report.json
"""

import argparse
import json
import os
import re
import subprocess
import sys

PARQUET = "/share/latent/MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet"
REPO = "/share/latent/PRLM-C"
MODEL = "/share/latent/MemoryLLM-8B"
CONFIG = os.path.join(REPO, "configs/agent_conf/prlmc_e6.yaml")

DEVICE = "cuda:0"
WINDOW_BLOCKS = 19
ANCHOR_LINES = 30
LINES_PER_CHUNK = 20


def pick_gpu(need_mb: int = 20000) -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    free = []
    for line in out.strip().splitlines():
        index, used = [int(v) for v in line.split(",")]
        free.append((32768 - used, index))
    free.sort(reverse=True)
    if free[0][0] < need_mb:
        raise SystemExit("no GPU with %d MB free; best is gpu%d with %d MB"
                         % (need_mb, free[0][1], free[0][0]))
    return free[0][1]


if __name__ == "__main__" and not os.environ.get("G0_GPU_SET"):
    _gpu = pick_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu)
    os.environ["G0_GPU_SET"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import torch  # noqa: E402

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

import e6a_overfit as E  # noqa: E402


def cosine_row(a, b):
    """Cosine between two equal-length vectors, as python floats."""
    dot = float((a * b).sum())
    na = float((a * a).sum()) ** 0.5
    nb = float((b * b).sum()) ** 0.5
    return dot / (na * nb + 1e-12)


def block_vecs(source, begin, window_slots, layer):
    """[n_blocks, d] mean-pooled vectors of each block at one layer."""
    n_blocks = window_slots // 256
    vecs = []
    for b in range(n_blocks):
        sl = source[layer, begin + b * 256: begin + (b + 1) * 256, :]
        vecs.append(sl.mean(dim=0))
    return torch.stack(vecs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="0,1,2,3",
                    help="comma-separated parquet rows; duplicate contexts auto-deduped")
    ap.add_argument("--start", type=int, default=30,
                    help="start line for every window")
    ap.add_argument("--min-subjects", type=int, default=8,
                    help="abort if fewer same-subject version pairs than this")
    ap.add_argument("--out", default="/tmp/g0_report.json")
    args = ap.parse_args()

    import yaml
    from transformers import AutoTokenizer
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory

    raw = yaml.safe_load(open(CONFIG))
    config = dict(raw["prlmc"])
    config["device"] = DEVICE
    config["dtype"] = "float16"
    config["stats_dir"] = "./outputs/prlmc-g0"
    cfg = PRLMCConfig.from_dict(config)

    model = MemoryLLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                      attn_implementation="sdpa")
    model = model.to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    manager = ProtectedResidualMemory(model, tokenizer, cfg)

    lines_row = {}
    line_to_block_row = {}
    row_index = {}
    pool_window_blocks = []   # [window][layer] -> [n_blocks, d] vectors
    pairs = []
    groups = {}
    seen_hash = set()
    n_windows = 0

    window_slots = WINDOW_BLOCKS * manager.block
    start = args.start
    for row in [int(r) for r in args.rows.split(",")]:
        import hashlib as _hl
        lines = E.read_fact_lines(row)
        digest = _hl.md5("\n".join(lines).encode()).hexdigest()[:8]
        if digest in seen_hash:
            print("[g0] row %d skipped (duplicate context %s)" % (row, digest), flush=True)
            continue
        seen_hash.add(digest)
        lines_row[row] = lines

        covered = E.build_window(manager, lines, window_slots, start_line=start)
        l2b = {}
        for block_index, (lo, hi, _before, _width) in enumerate(covered):
            for line in range(lo, hi):
                l2b[line] = block_index
        line_to_block_row[row] = l2b

        # capture this window's pooled block vectors before the next rebuild
        # resets the pool (every build_window re-anchors and zeroes it).
        begin = manager.total - manager.used
        source = manager._memory[:, begin: begin + window_slots, :].to(
            device=DEVICE, dtype=torch.float32)
        pool_window_blocks.append(
            [block_vecs(source, 0, window_slots, layer) for layer in range(manager.num_layers)])
        del source
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        row_groups = {}
        window_line_set = set()
        for lo, hi, _b, _w in covered:
            window_line_set.update(range(lo, hi))
        for index in sorted(window_line_set):
            found = E.first_match(lines[index])
            if found is None:
                continue
            slot, subject, _answer, _q = found
            row_groups.setdefault((slot, subject), []).append(index)
        groups[row] = row_groups
        row_index[row] = n_windows

        for key, idxs in sorted(row_groups.items()):
            if len(idxs) < 2:
                continue
            current = idxs[-1]
            for old in idxs[:-1]:
                pairs.append({
                    "key": "%s|%s" % key,
                    "row": row,
                    "current_line": current, "old_line": old,
                    "current_block": l2b[current],
                    "old_block": l2b[old],
                    "current_text": lines[current], "old_text": lines[old],
                })
        n_multi = sum(1 for v in row_groups.values() if len(v) >= 2)
        print("[g0] row %d start %d: %d multi-occurrence groups, cumulative %d pairs"
              % (row, start, n_multi, len(pairs)), flush=True)
        n_windows += 1

    if n_windows == 0:
        raise SystemExit("no windows built")
    if len(pairs) < args.min_subjects:
        raise SystemExit("only %d version pairs; scan more windows" % len(pairs))

    # control pairs: different subjects within the same window, count-matched
    control = []
    ctrl_by_window = {}
    same_by_window = {}
    for row, row_groups in groups.items():
        wi = row_index[row]
        same_by_window[wi] = [p for p in pairs if p["row"] == row]
        l2b = line_to_block_row[row]
        lines = lines_row[row]
        multi = sorted(k for k, v in row_groups.items() if len(v) >= 2)
        if len(multi) < 2:
            ctrl_by_window[wi] = []
            continue
        row_ctrl = []
        local = same_by_window[wi]
        for i in range(len(local)):
            ka, kb = multi[(i * 7) % len(multi)], multi[(i * 7 + 3) % len(multi)]
            if ka == kb:
                continue
            a_cur = row_groups[ka][-1]
            b_old = row_groups[kb][0]
            row_ctrl.append({
                "key": "%s|%s VS %s|%s" % (ka + kb),
                "row": row,
                "current_line": a_cur, "old_line": b_old,
                "current_block": l2b[a_cur],
                "old_block": l2b[b_old],
                "current_text": lines[a_cur], "old_text": lines[b_old],
            })
        ctrl_by_window[wi] = row_ctrl
        control.extend(row_ctrl)

    # --- pool-space cosine ---------------------------------------------------
    # Window-local: every cosine is computed between blocks of the same window.
    n_layers = manager.num_layers
    same_layers = [[] for _ in range(n_layers)]
    ctrl_layers = [[] for _ in range(n_layers)]
    for wi in range(n_windows):
        blocks_per_layer = pool_window_blocks[wi]
        for layer in range(n_layers):
            bv = blocks_per_layer[layer]
            for p in same_by_window.get(wi, []):
                same_layers[layer].append(
                    cosine_row(bv[p["current_block"]], bv[p["old_block"]]))
            for p in ctrl_by_window.get(wi, []):
                ctrl_layers[layer].append(
                    cosine_row(bv[p["current_block"]], bv[p["old_block"]]))

    def stats(xs):
        xs = sorted(xs)
        n = len(xs)
        return {
            "n": n,
            "mean": round(sum(xs) / n, 4),
            "min": round(xs[0], 4),
            "max": round(xs[-1], 4),
            "median": round(xs[n // 2], 4),
            "p10": round(xs[max(0, n // 10)], 4),
            "p90": round(xs[min(n - 1, (9 * n) // 10)], 4),
        }

    layer_stats = []
    for layer in range(n_layers):
        layer_stats.append({
            "layer": layer,
            "same_subject": stats(same_layers[layer]),
            "diff_subject": stats(ctrl_layers[layer]) if ctrl_layers[layer] else None,
        })

    # layer-averaged per-pair cosine
    same_pair_mean = [round(sum(same_layers[l][i] for l in range(n_layers)) / n_layers, 4)
                      for i in range(len(pairs))]
    ctrl_pair_mean = ([round(sum(ctrl_layers[l][i] for l in range(n_layers)) / n_layers, 4)
                       for i in range(len(control))] if any(ctrl_layers) else [])

    overall_same = stats(same_pair_mean)
    overall_ctrl = stats(ctrl_pair_mean) if ctrl_pair_mean else None

    # --- text-space cosine (E7a-sim style, optional sanity anchor) -----------
    text_same, text_ctrl = None, None
    if False:
        # Reuse the E7a-sim style encoder path: initialized=0 + lm_head Identity
        model.initialized.zero_()
        import torch.nn as nn
        model.lm_head = nn.Identity()
        first_device = next(model.parameters()).device

        def embed(texts):
            embs = []
            for t in texts:
                enc = tokenizer([t], return_tensors="pt", padding=True,
                                truncation=True, max_length=64,
                                add_special_tokens=False).to(first_device)
                with torch.no_grad():
                    out = model(input_ids=enc.input_ids,
                                attention_mask=enc.attention_mask,
                                return_dict=True, output_hidden_states=True)
                h = out.hidden_states[-1]
                m = enc.attention_mask.unsqueeze(-1).expand(h.size()).float()
                embs.append(((h * m).sum(1) / m.sum(1).clamp(min=1))[0].float().cpu())
            return embs

        uniq = sorted({p["current_line"] for p in pairs} | {p["old_line"] for p in pairs})
        emb = {i: e for i, e in zip(uniq, embed([lines[i] for i in uniq]))}
        text_same = [round(cosine_row(emb[p["current_line"]], emb[p["old_line"]]), 4)
                     for p in pairs]
        cu = sorted({p["current_line"] for p in control} | {p["old_line"] for p in control})
        cemb = {i: e for i, e in zip(cu, embed([lines[i] for i in cu]))}
        text_ctrl = [round(cosine_row(cemb[p["current_line"]], cemb[p["old_line"]]), 4)
                     for p in control]

    # --- verdict -------------------------------------------------------------
    same_mean = overall_same["mean"]
    ctrl_mean = overall_ctrl["mean"] if overall_ctrl else None
    verdict = "PASS" if same_mean >= 0.95 else "FAIL"
    if ctrl_mean is not None and abs(same_mean - ctrl_mean) < 0.005:
        verdict += " (but same-vs-diff indistinguishable: even stronger blindness)"

    report = {
        "design": {
            "rows": args.rows, "start": start,
            "n_windows": n_windows,
            "window_blocks": WINDOW_BLOCKS,
            "n_pairs": len(pairs), "n_control": len(control),
            "representation": "mean-pooled per-block pool states, per layer, window-local",
            "falsification": "same-subject mean cosine >= 0.95 => PASS (blindness confirmed)",
        },
        "pairs": pairs,
        "control": control,
        "layer_stats": layer_stats,
        "overall": {
            "same_subject_layer_avg": overall_same,
            "diff_subject_layer_avg": overall_ctrl,
            "same_pair_cosines": same_pair_mean,
            "ctrl_pair_cosines": ctrl_pair_mean,
        },
        "verdict": verdict,
    }
    with open(args.out, "w") as f:
        json.dump(report, f)
    print("[g0] same-subject layer-avg cosine: mean %.4f min %.4f max %.4f (n=%d)"
          % (overall_same["mean"], overall_same["min"], overall_same["max"],
             overall_same["n"]), flush=True)
    if overall_ctrl:
        print("[g0] diff-subject layer-avg cosine: mean %.4f min %.4f max %.4f (n=%d)"
              % (overall_ctrl["mean"], overall_ctrl["min"], overall_ctrl["max"],
                 overall_ctrl["n"]), flush=True)
    print("[g0] verdict: %s -> %s" % (verdict,
          "SDC motivation holds" if verdict.startswith("PASS") else "STOP-LOSS"),
          flush=True)
    print("[g0] report", args.out, flush=True)


if __name__ == "__main__":
    main()
