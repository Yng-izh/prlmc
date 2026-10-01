"""E6e: catch the forgetting in the act -- which parameters move when it happens.

The question
------------
E6c's retention reading is `overwrites`: a window recovers inside its own
training, then later windows erase it. E6d (running in parallel) tests whether
the *schedule* is the cause. This experiment is the mechanism probe: while a
victim window's recovery decays, photograph the compactor's routing / residual
generation / residual strength, and report which of the three moved when the
recovery died.

The design follows the "train one window, then watch it die while later windows
train" protocol:

  1. Train window A (victim, first of the E6c fixture) for its full budget.
  2. Photograph: A's recovery count, and the compactor's parameter snapshot.
  3. For each later window X (B..H, in forward order), train X in `--chunk`
     step chunks; after *every* chunk, re-score the victim window A and
     photograph again.
  4. Correlate: between consecutive photographs, how much did routing /
     residual / strength parameters move, and how did they move relative to
     A's recovery change. Also record the conflict matrix (per-window effect
     on A's correct count) that the E6d report cannot give per-chunk.

What is measured, concretely
----------------------------
Parameter deltas (relative Frobenius distance, per group, vs the victim
photograph):

    routing   q, k, route_logit          "which block absorbs what"
    residual  down, up                   "what content gets injected"
    strength  gamma_logit                "how much gets injected"

Behaviour deltas on the *victim window's* pool (the thing A's probes actually
read): routing distribution (attention matrix) total-variation and per-carrier
JS divergence vs the victim photograph, and alpha / residual-ratio drift. These
are the "frozen check": if routing barely moves but A still dies, routing is
acquitted *on behaviour*, not merely on parameters.

Readings (pre-registered)
--------------------------
    routing moves most, behaviour follows  -> "content still there, delivered
    to the wrong block"; suspect q/k/route_logit interference.
    residual moves most, routing stable    -> "right place, corrupted content";
    suspect down/up interference.
    both move                              -> joint interference; needs
    per-group freeze ablation to separate.
    neither moves but A still dies         -> the damage is *inside* gamma_logit
    (strength) or in the reader-facing scale; check strength deltas first.

The conflict matrix (which later window kills how much of A) is recorded
regardless: universal decay vs specific-window conflict is a separate axis
from the mechanism question and both answers are needed to design E7.

Cost control: the victim is re-scored after every chunk of every later window
(--chunk 4 default: 8 windows x 24 chunks x ~2.5 min full-window scoring is
the dominant cost; --chunk 8 or 16 halves/quarters it). Parameters and
behaviour are cheap (one forward on the victim's pool, already needed for
scoring).

This probe deliberately does NOT modify the compactor, the loss, or the
schedule beyond chunking -- it is an instrument, not a treatment.

Smoke test:
    /share/latent/mllm_env/bin/python /tmp/e6e_probe.py \
        --victim-window 0 --later-windows 1 --chunk 2 --steps-b-per-window 8 \
        --out /tmp/e6e_smoke.json
"""

import argparse
import copy
import hashlib
import json
import os
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = "/share/latent/PRLM-C"
for path in ("/tmp", TOOLS, REPO):
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PARAM_GROUPS = {
    "routing": ("q.weight", "k.weight", "route_logit"),
    "residual": ("down.weight", "up.weight"),
    "strength": ("gamma_logit",),
}


def _reexec_on_free_gpu():
    import subprocess

    need_mb = int(os.environ.get("E6E_NEED_MB", "26000"))
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
        raise SystemExit(
            "no GPU with %d MB free; best is gpu%d with %d MB" % (need_mb, free[0][1], free[0][0])
        )
    print("[e6e] gpu%d, %d MB free" % (free[0][1], free[0][0]), flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(free[0][1])
    os.environ["E6E_GPU_SET"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


if __name__ == "__main__" and not os.environ.get("E6E_GPU_SET") and not os.environ.get("E6E_NO_GPU"):
    _reexec_on_free_gpu()

import torch  # noqa: E402

DEVICE = "cuda:0"


# --- pure helpers (locally testable without torch) -----------------------------

def group_param_md5s(state):
    """md5 per parameter group -- the cheap identity of a photograph."""
    out = {}
    for group, names in PARAM_GROUPS.items():
        h = hashlib.md5()
        for name in names:
            v = state[name]
            h.update(name.encode())
            h.update(v.detach().cpu().numpy().tobytes())
        out[group] = h.hexdigest()
    return out


def group_param_distances(state_a, state_b):
    """Symmetric relative distance per group: ||A-B|| / (||A||+||B||+eps).

    Symmetric on purpose: the victim photograph has `up.weight` at exactly zero
    (zero-init), so a ||A-B||/||B|| form explodes to 4e10 the moment training
    touches it. The symmetric form gives a bounded [0, 2] range where a
    zero-to-nonzero transition reads as ~1.0 ("completely new direction") --
    which is the honest description of what happened -- and an unchanged
    parameter reads exactly 0.0.
    """
    out = {}
    for group, names in PARAM_GROUPS.items():
        num, den = 0.0, 0.0
        for name in names:
            a = state_a[name].detach().float()
            b = state_b[name].detach().float()
            num += float((a - b).pow(2).sum())
            den += float(a.pow(2).sum()) + float(b.pow(2).sum())
        out[group] = (num ** 0.5) / (den ** 0.5 + 1e-12)
    return out


def routing_behaviour(ref_attn, cur_attn):
    """Total variation + mean per-carrier JS divergence between two routings.

    `attn` is [b_out, b]: one distribution over source blocks per carrier.
    Total variation answers "did the delivery plan change at all"; JS adds a
    bounded (0..ln 2) per-carrier view that is robust to one carrier blowing up.
    """
    import math

    a = ref_attn.clamp_min(1e-12)
    c = cur_attn.clamp_min(1e-12)
    tv = 0.5 * float((a - c).abs().sum(dim=-1).mean())
    m = 0.5 * (a + c)
    js = float((0.5 * (a * (a / m).log()) + 0.5 * (c * (c / m).log())).sum(dim=-1).mean())
    if js < 0:
        js = 0.0
    if js > math.log(2):
        js = float(math.log(2))
    return {"tv": tv, "js": js}


# --- main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--victim-window", type=int, default=0,
                        help="index into the E6c fixture trained first and watched")
    parser.add_argument("--later-windows", type=int, default=7,
                        help="how many of the remaining windows train afterwards")
    parser.add_argument("--steps-a-per-window", type=int, default=4)
    parser.add_argument("--steps-b-per-window", type=int, default=96)
    parser.add_argument("--chunk", type=int, default=4,
                        help="later-window chunk size; victim re-scored after each")
    parser.add_argument("--probes", type=int, default=20)
    parser.add_argument("--max-probe-tokens", type=int, default=256)
    parser.add_argument("--min-probes", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma-max", type=float, default=0.05)
    parser.add_argument("--out", default="/tmp/e6e_probe_report.json")
    parser.add_argument("--ckpt-prefix", default="/tmp/e6e_ckpt")
    args = parser.parse_args()

    import random

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        import numpy as np

        np.random.seed(args.seed)
    except ImportError:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    import e6a_overfit as E
    import e6b_mini as B

    from methods.prlmc.consolidate import CarrierResidualCompactor
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM

    import yaml
    from transformers import AutoTokenizer

    raw = yaml.safe_load(open(E.CONFIG))
    config = dict(raw["prlmc"])
    config["device"] = DEVICE
    config["dtype"] = "float16"
    config["stats_dir"] = "./outputs/prlmc-e6e"
    config["carrier_gamma_max"] = args.gamma_max
    cfg = PRLMCConfig.from_dict(config)

    model = MemoryLLM.from_pretrained(E.MODEL, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = model.to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(E.MODEL)
    manager = ProtectedResidualMemory(model, tokenizer, cfg)

    def new_compactor():
        return CarrierResidualCompactor(
            num_layers=manager.num_layers, hidden=manager.hidden,
            block=manager.block, rank=int(cfg.carrier_rank),
            gamma_max=float(cfg.carrier_gamma_max),
        ).to(device=DEVICE, dtype=torch.float32)

    # The victim is trained from the seeded init; its "photograph" is the
    # reference every later measurement is diffed against.
    compactor = new_compactor()
    initial_state = copy.deepcopy(compactor.state_dict())

    # E6c fixture (same as E6d): comparability across the whole E6 series.
    from e6d_schedule import DEFAULT_WINDOWS

    n_windows = 1 + args.later_windows
    windows = DEFAULT_WINDOWS[:n_windows]
    if len(windows) < n_windows:
        raise SystemExit("[e6e] fixture has %d windows, need %d" % (len(DEFAULT_WINDOWS), n_windows))
    victim_idx = args.victim_window
    if not 0 <= victim_idx < len(windows):
        raise SystemExit("[e6e] --victim-window %d out of range" % victim_idx)

    cache = {}
    for row in sorted({r for r, _ in windows}):
        cache[row] = E.read_fact_lines(row)

    print("[e6e] fixture: victim w%d, then %d later windows, %d B-steps each, chunk %d"
          % (victim_idx, len(windows) - 1, args.steps_b_per_window, args.chunk), flush=True)

    entries = []
    for row, start in windows:
        entry = B.measure_window(
            manager, cache[row], row, start, compactor, initial_state,
            args.probes, args, log_prefix="[e6e] fixture",
        )
        if entry is None:
            raise SystemExit("[e6e] fixture window %d:%d unusable" % (row, start))
        entries.append(entry)
        print("[e6e] fixture %d:%d probes %d eligible %d lost %d"
              % (row, start, entry["n_probes"], entry["n_eligible"], entry["n_lost"]), flush=True)
    torch.cuda.empty_cache()

    victim_entry = entries[victim_idx]

    def score_victim(cptr):
        """One forward-only scoring of the victim window; returns full result."""
        result = B.evaluate_window(manager, cache[victim_entry["row"]], victim_entry, cptr, args)
        B.release()
        return result

    def photograph(cptr, tag, cum):
        """Victim scoring + full parameter snapshot for later distance analysis.

        `_state` (the tensors) is popped before serialisation; md5s stay for a
        cheap identity check in the JSON.
        """
        result = score_victim(cptr)
        state = {k: v.detach().clone() for k, v in cptr.state_dict().items()}
        return {
            "tag": tag,
            "cum_steps": cum,
            "victim_n_correct": sum(bool(v) for v in result["after"]["correct"].values()),
            "victim_recovered": result["recovery"]["n_recovered"],
            "victim_lost": result["recovery"]["n_lost"],
            "correct": {q: bool(v) for q, v in result["after"]["correct"].items()},
            "signature": result["after_signature"],
            "param_md5s": group_param_md5s(state),
            "gamma": float(cptr.gamma.detach()),
            "route_scale": float(cptr.route_scale.detach()),
            "last_residual_ratio": float(cptr.last_residual_ratio),
            "last_routing_entropy": float(cptr.last_routing_entropy),
            "_state": state,
        }

    # --- Stage A over all fixture windows (victim included), forward order ----
    optimiser = torch.optim.AdamW(compactor.parameters(), lr=args.lr)
    cum = 0
    for i, entry in enumerate(entries):
        B.train_window(manager, cache[entry["row"]], entry, compactor, optimiser,
                       "A", args.steps_a_per_window, args)
        cum += args.steps_a_per_window
    print("[e6e] stage A done at cum %d" % cum, flush=True)

    # --- Train the victim to its full budget, then photograph ----------------
    B.train_window(manager, cache[victim_entry["row"]], victim_entry, compactor,
                   optimiser, "B", args.steps_b_per_window, args)
    cum += args.steps_b_per_window
    photos = [photograph(compactor, "victim-trained", cum)]
    print("[e6e] victim trained: correct %d recovered %d/%d"
          % (photos[0]["victim_n_correct"], photos[0]["victim_recovered"],
             photos[0]["victim_lost"]), flush=True)

    conflict = {}
    for j, entry in enumerate(entries):
        if j == victim_idx:
            continue
        key = "%d:%d" % (entry["row"], entry["start"])
        steps_left = args.steps_b_per_window
        first_chunk = True
        while steps_left > 0:
            chunk = min(args.chunk, steps_left)
            B.train_window(manager, cache[entry["row"]], entry, compactor,
                           optimiser, "B", chunk, args)
            steps_left -= chunk
            cum += chunk
            photo = photograph(compactor, "%s+%d" % (key, cum), cum)
            if first_chunk:
                conflict[key] = {"n_correct_before": photos[-1]["victim_n_correct"]}
                first_chunk = False
            conflict[key]["n_correct_after"] = photo["victim_n_correct"]
            photos.append(photo)
            print("[e6e]   %s chunk cum %-4d victim %d/%d recovered %d/%d"
                  % (key, cum, photo["victim_n_correct"], photo["victim_lost"],
                     photo["victim_recovered"], photo["victim_lost"]), flush=True)
        conflict[key]["delta"] = (
            conflict[key]["n_correct_after"] - conflict[key]["n_correct_before"]
        )

    # --- correlations: parameter-group movement vs victim decay --------------
    victim_ref = photos[0]
    correlations = []
    for i in range(1, len(photos)):
        prev, cur = photos[i - 1], photos[i]
        # Distance vs the victim photograph = cumulative drift since the victim
        # was healthy; distance vs the previous photo = movement in this chunk
        # alone. Both are needed: the first attributes the death, the second
        # separates "one violent chunk" from "slow steady erosion".
        from_victim = group_param_distances(cur["_state"], victim_ref["_state"])
        since_prev = group_param_distances(cur["_state"], prev["_state"])
        md5_changed = {
            g: cur["param_md5s"][g] != prev["param_md5s"][g] for g in PARAM_GROUPS
        }
        correlations.append({
            "tag": cur["tag"],
            "cum_steps": cur["cum_steps"],
            "victim_correct_prev": prev["victim_n_correct"],
            "victim_correct": cur["victim_n_correct"],
            "victim_delta": cur["victim_n_correct"] - prev["victim_n_correct"],
            "victim_delta_since_train": cur["victim_n_correct"] - victim_ref["victim_n_correct"],
            "param_dist_from_victim": from_victim,
            "param_dist_since_prev": since_prev,
            "param_md5_changed": md5_changed,
            "gamma_prev": prev["gamma"], "gamma": cur["gamma"],
            "route_scale_prev": prev["route_scale"], "route_scale": cur["route_scale"],
            "residual_ratio_prev": prev["last_residual_ratio"],
            "residual_ratio": cur["last_residual_ratio"],
            "routing_entropy_prev": prev["last_routing_entropy"],
            "routing_entropy": cur["last_routing_entropy"],
        })

    # tensors never enter the JSON; keep photographs lightweight
    for photo in photos:
        photo.pop("_state", None)

    # save
    report = {
        "design": {
            "question": "which compactor part (routing / residual / strength) moves when a recovered window decays?",
            "seed": args.seed,
            "victim_window": victim_idx,
            "victim_window_id": "%d:%d" % (victim_entry["row"], victim_entry["start"]),
            "later_windows": [e for i, e in enumerate(
                ["%d:%d" % (en["row"], en["start"]) for en in entries]) if i != victim_idx],
            "steps_a_per_window": args.steps_a_per_window,
            "steps_b_per_window": args.steps_b_per_window,
            "chunk": args.chunk,
            "lr": args.lr,
            "gamma_max": float(cfg.carrier_gamma_max),
            "fixture": "E6c's training windows (first %d)" % n_windows,
            "pre_registered_readings": [
                "routing moves most, behaviour follows -> delivery plan interference",
                "residual moves most, routing stable -> content corruption",
                "both move -> joint interference, needs freeze ablation",
                "neither moves but victim dies -> strength (gamma_logit) or scale",
            ],
        },
        "baseline": {
            "%d:%d" % (e["row"], e["start"]): {
                "n_eligible": e["n_eligible"], "n_lost": e["n_lost"],
                "lost_questions": e["lost_questions"],
                "uncompacted_n_correct": sum(bool(v) for v in e["uncompacted"]["correct"].values()),
                "before_n_correct": sum(bool(v) for v in e["before"]["correct"].values()),
            } for e in entries
        },
        "photographs": photos,
        "conflict_matrix": conflict,
        "correlations": correlations,
    }
    json.dump(report, open(args.out, "w"), indent=2, default=str)
    torch.save({"state_dict": compactor.state_dict()},
               "%s_final.pt" % args.ckpt_prefix)
    print("[e6e] ---- conflict matrix (victim correct count) ----", flush=True)
    for key, row in conflict.items():
        print("[e6e]   %-12s %d -> %d (%+d)" % (key, row["n_correct_before"],
                                                row["n_correct_after"], row["delta"]), flush=True)
    print("[e6e] report %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
