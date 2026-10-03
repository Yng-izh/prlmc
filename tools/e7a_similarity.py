"""E7a-sim: content-similarity vs functional-interference adjacency comparison.

Stage A's final read-out (E7_PLAN.md lines 63-70): does semantic similarity of
window *content* predict functional interference/compatibility? If yes, a
content router has a mechanistic basis; if not, the stop gate fires (no
semantic router).

Representation: mean-pooled last-hidden-state of the frozen MemoryLLM reader
over the window's fact lines, batched in chunks under no_grad. The reader is
the model whose behaviour we actually route for, it is already on disk, and
this mirrors the "source/carrier summary" idea from the plan. Also computed
per-fact-line embeddings so similarity is not dominated by one long string.

Signals compared (all 8x8, off-diagonal vectors of 56 pairs):
  1. content cosine (reader last-hidden mean-pool over window body lines)
  2. gradient cosine at the shared initial compactor (from e7a reports)
  3. functional delta = rescued - harmed per (source -> target) at 24 steps
  4. functional rescue-only and harmed-only matrices

Read-outs: Pearson + Spearman correlations between signals, k=2/k=3
agglomerative clustering agreement (ARI) between content and functional
adjacency, and the bandwidth (min/max) of the content similarity matrix --
the known risk is that all 8 windows come from one context (row 3), so the
similarity band is expected to be narrow.

No training, no structure change, no new modules. Pure measurement.
"""

import argparse
import json
import statistics
import subprocess
import sys
import os

# GPU pick before torch import, mirroring e7a_interference's guard.
if "CUDA_VISIBLE_DEVICES" not in os.environ and not os.environ.get("E7A_GPU_SET"):
    try:
        need = int(os.environ.get("E7A_NEED_MB", "26000"))
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"], text=True)
        for line in out.strip().splitlines():
            idx, used = (int(x) for x in line.split(","))
            if used < 4000:  # a card that is effectively free
                os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
                break
        else:
            raise SystemExit("no free GPU for e7a similarity pass")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("gpu pick failed: %s" % exc)

import torch  # noqa: E402

sys.path.insert(0, "/tmp")
sys.path.insert(0, "/share/latent/PRLM-C")

import e6a_overfit as E  # noqa: E402
import e6b_mini as B  # noqa: E402
from methods.prlmc.config import PRLMCConfig  # noqa: E402
from methods.prlmc.layout import ProtectedResidualMemory  # noqa: E402
from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM  # noqa: E402

DEVICE = "cuda:0"


def embed_texts(model, tokenizer, texts, first_device, batch_tokens=2000):
    """Mean-pool last hidden state over each text; batches by token budget.

    A all-zero memory_active_mask hides every pool slot from the reader's
    attention (the same mechanism `active_mask_override` uses to hide freed
    slots during compaction), which turns the forward into a pure text
    encoding: no memory content contributes to the hidden states.
    """
    embs = []
    batch, batch_len = [], 0
    def flush():
        nonlocal batch, batch_len
        if not batch:
            return
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=128,
                        add_special_tokens=False).to(first_device)
        with torch.no_grad():
            out = model(input_ids=enc.input_ids,
                        attention_mask=enc.attention_mask,
                        return_dict=True, output_hidden_states=True)
        h = out.hidden_states[-1]  # (B, T, D)
        mask = enc.attention_mask.unsqueeze(-1).expand(h.size()).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        embs.extend(pooled.cpu().tolist())
        del out, h, pooled, enc
        batch, batch_len = [], 0
    for t in texts:
        n = min(len(t.split()) * 2, 128) + 2
        if batch and batch_len + n > batch_tokens:
            flush()
        batch.append(t)
        batch_len += n
    flush()
    return embs


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb + 1e-12)


def pearson(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def spearman(xs, ys):
    def ranks(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        r = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(ranks(xs), ranks(ys))


def cluster_labels(sim, k):
    """Agglomerative single-link-ish clustering via average linkage on 1-sim."""
    import itertools
    nodes = list(range(len(sim)))
    # simple agglomerative: repeatedly merge closest pair of clusters by avg dist
    clusters = [[n] for n in nodes]
    dist = lambda a, b: 1 - sim[a][b]
    while len(clusters) > k:
        best, bi, bj = None, None, None
        for i, j in itertools.combinations(range(len(clusters)), 2):
            d = statistics.mean(dist(a, b) for a in clusters[i] for b in clusters[j])
            if best is None or d < best:
                best, bi, bj = d, i, j
        clusters[bi] = clusters[bi] + clusters[bj]
        del clusters[bj]
    lab = [0] * len(nodes)
    for ci, c in enumerate(clusters):
        for n in c:
            lab[n] = ci
    return lab


def ari(l1, l2):
    """Adjusted Rand index between two label vectors."""
    from collections import Counter
    n = len(l1)
    c1, c2 = Counter(l1), Counter(l2)
    pair_counts = Counter(zip(l1, l2))
    comb = lambda x: x * (x - 1) / 2
    sum_c = sum(comb(v) for v in c1.values())
    sum_k = sum(comb(v) for v in c2.values())
    sum_ij = sum(comb(v) for v in pair_counts.values())
    total = comb(n)
    expected = sum_c * sum_k / total if total else 0
    max_i = (sum_c + sum_k) / 2
    return (sum_ij - expected) / (max_i - expected) if max_i != expected else 1.0


def offdiag(m):
    return [m[i][j] for i in range(len(m)) for j in range(len(m)) if i != j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sim-report", default="/tmp/e7a_interference_seed0_s24.json")
    ap.add_argument("--windows-json", default="/tmp/e7a_window_content.json")
    ap.add_argument("--out", default="/tmp/e7a_similarity.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # 1. window content from the frozen fixture dump
    content = json.load(open(args.windows_json))
    labels = list(content.keys())
    print("[e7a-sim] windows:", labels, flush=True)

    # 2. reader as encoder: load once, embed all body lines of every window.
    # Every card on the host is ~18GB occupied by outside processes, so a
    # single-card fp16 load (16GB) does not fit; split the weights across two
    # cards with device_map="auto" (assign CUDA_VISIBLE_DEVICES=A,B outside).
    import yaml
    from transformers import AutoTokenizer
    raw = yaml.safe_load(open(E.CONFIG))
    device_map = "auto" if os.environ.get("E7A_SIM_SHARD") else None
    model = MemoryLLM.from_pretrained(
        E.MODEL, torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map=device_map)
    if device_map is None:
        model = model.to(DEVICE)
    model.eval()
    # The checkpoint ships with `initialized=1`, which makes every forward
    # prepend the 12824-slot memory pool to cache_position and crash against
    # short text batches (23 vs 12824). Resetting the flag puts the model in
    # its pre-initialization state where `cat_memory_and_hiddens` returns the
    # hidden states untouched: a pure text encoder, exactly what a content
    # embedding needs.
    model.initialized.zero_()
    # lm_head -> Identity: this pass only consumes `hidden_states[-1]`, and the
    # real lm_head would materialise a (B, T, 128k-vocab) float32 logits tensor
    # that does not fit next to the sharded weights on a shared card.
    import torch.nn as nn
    model.lm_head = nn.Identity()
    model._hf_hook.need_to_reduce = False if hasattr(model, "_hf_hook") else None
    tokenizer = AutoTokenizer.from_pretrained(E.MODEL)
    first_device = next(model.parameters()).device

    win_line_embs = {}
    for w in labels:
        texts = content[w]["anchor"] + content[w]["body"]
        embs = embed_texts(model, tokenizer, texts, first_device)
        win_line_embs[w] = embs
        print("[e7a-sim] embedded %s: %d lines" % (w, len(embs)), flush=True)

    # window vector = mean over its line embeddings
    win_vec = {}
    for w in labels:
        embs = win_line_embs[w]
        d = len(embs[0])
        win_vec[w] = [statistics.mean(e[di] for e in embs) for di in range(d)]

    sim = [[cosine(win_vec[a], win_vec[b]) for b in labels] for a in labels]
    del model
    torch.cuda.empty_cache()

    # 3. functional + gradient signals from the e7a report
    rep = json.load(open(args.sim_report))
    glabels = rep["gradient_cosine"]["labels"]
    assert glabels == labels, "window order mismatch between reports"
    gcos = rep["gradient_cosine"]["matrix"]
    fi = rep["functional_interference"]
    delta = fi["delta_n_correct"]["matrix"]
    rescued = fi["rescued"]["matrix"]
    harmed = fi["harmed"]["matrix"]

    # 4. pairwise signal correlations over the 56 off-diagonal pairs
    s_sim, s_grad, s_delta, s_resc, s_harm = (
        offdiag(sim), offdiag(gcos), offdiag(delta),
        offdiag(rescued), offdiag(harmed))
    rows = []
    for name, other in [("grad_cos", s_grad), ("func_delta", s_delta),
                        ("rescued", s_resc), ("harmed", s_harm)]:
        rows.append({
            "signal": name,
            "pearson_vs_content_sim": round(pearson(s_sim, other), 4),
            "spearman_vs_content_sim": round(spearman(s_sim, other), 4),
        })
    grad_delta = {
        "signal": "func_delta_vs_grad_cos",
        "pearson": round(pearson(s_grad, s_delta), 4),
        "spearman": round(spearman(s_grad, s_delta), 4),
    }

    # 5. clustering agreement: content clusters vs functional clusters
    # functional adjacency: symmetrise delta (i->j and j->i both matter)
    fadj = [[(delta[i][j] + delta[j][i]) / 2 for j in range(8)] for i in range(8)]
    cl = {}
    for k in (2, 3):
        c_lab = cluster_labels(sim, k)
        f_lab = cluster_labels(fadj, k)
        cl["k%d" % k] = {
            "content_clusters": c_lab,
            "functional_clusters": f_lab,
            "ari": round(ari(c_lab, f_lab), 4),
        }

    od = offdiag(sim)
    report = {
        "design": {
            "seed": args.seed,
            "representation": "reader last-hidden mean-pool over window lines",
            "sim_report": args.sim_report,
            "structure_changed": False,
            "loss_changed": False,
            "note": "content similarity vs functional interference; "
                    "stop gate for the semantic router decision",
        },
        "content_similarities": {"labels": labels, "matrix": sim},
        "bandwidth": {
            "offdiag_min": round(min(od), 4),
            "offdiag_max": round(max(od), 4),
            "offdiag_mean": round(statistics.mean(od), 4),
        },
        "correlations": rows,
        "grad_vs_func": grad_delta,
        "clustering": cl,
    }
    with open(args.out, "w") as f:
        json.dump(report, f)
    print("[e7a-sim] report", args.out, flush=True)
    print("[e7a-sim] bandwidth %.3f-%.3f (mean %.3f)"
          % (min(od), max(od), statistics.mean(od)), flush=True)
    for row in rows:
        print("[e7a-sim] corr(content, %s): pearson %.3f spearman %.3f"
              % (row["signal"], row["pearson_vs_content_sim"],
                 row["spearman_vs_content_sim"]), flush=True)
    print("[e7a-sim] ARI k=2 %.3f k=3 %.3f"
          % (cl["k2"]["ari"], cl["k3"]["ari"]), flush=True)


if __name__ == "__main__":
    main()
