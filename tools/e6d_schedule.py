"""E6d: does the *schedule*, not the budget, cause the forgetting? (three-arm control)

The question, and the only variable
-----------------------------------
E6b-mini/E6c trained one shared compactor over 8 windows, each window getting a
contiguous block of steps:

    Stage A:  w1 x4,  w2 x4,  ...  w8 x4
    Stage B:  w1 x96, w2 x96, ...  w8 x96

E6c's retention read flipped to `overwrites`: windows recovered inside their own
steps and were zeroed by later windows. But that run had no control -- it
changed nothing and compared nothing, and its seed was unset. This experiment
runs the same 8 windows, same structure, same loss, same lr, same gamma, same
per-window step totals under three window *orders*:

    forward    w1 x96 -> w2 x96 -> ...  -> w8 x96        (E6c's order)
    reverse    w8 x96 -> w7 x96 -> ...  -> w1 x96
    interleave w1 x4, w2 x4, ... w8 x4, repeated 24 rounds (each window still
               totals 96 B-steps)

Stage A keeps its E6b/E6c shape (one contiguous 4-step pass per window, in the
arm's order; for interleave that pass is the forward order). If forgetting is
driven by "one window trained too long without revisiting", interleave must hold
recoveries that forward loses, and reverse must move the forgetting to the
other end of the window list. If all three arms land on the same final counts,
the schedule is not the cause and the constraint is capacity/optimisation.

Seed
----
`--seed` defaults to 0 and is applied before the model, the manager and the
compactor are built. The compactor's initial `state_dict` is snapshotted once
and *deep-copied into every arm*, so the three schedules start from bitwise
identical weights -- the seed fixes the init once, the copy makes the arms
share it, and the snapshot's md5 is recorded in the report. E6c's unseeded run
is therefore demoted from "the control" to a reference number: the forward arm
*is* the seeded control this time.

Determinism is best-effort and reported honestly: cudnn.deterministic is set,
benchmarking is off, CUBLAS_WORKSPACE_CONFIG is exported before CUDA init, and
`torch.use_deterministic_algorithms(True, warn_only=True)` is enabled unless
`--no-deterministic` says otherwise. CUDA-level nondeterminism cannot be fully
ruled out on this stack, so the run also proves what it can: the baseline
snapshot is evaluated twice and the two correct-signatures must match bitwise,
which is the scoring path's reproducibility check. The comparison between arms
does not depend on cross-process reproducibility anyway -- all three run in one
process, on one GPU, from one init.

Per-question records, not totals
--------------------------------
Every measurement node stores, for every one of the 8 windows x 20 probes, the
full `question -> correct` dict. Nodes:

    cum=0               baseline (initial weights), scored twice for determinism
    cum=steps_a_total   after Stage A
    cum=+k*probe_every  after every `--probe-every` B-steps (default 96 = one
                        forward/reverse segment = 3 interleave rounds)
    cum=total           end of training (is the last probe_every node)

From the node sequence each question gets a bit trace and derived events:

    learned_at   cum where the answer flipped False -> True
    forgotten_at cum where the answer flipped True -> False
    (lists; a question can learn and forget repeatedly under interleave)

Per node the report carries the aggregates the question asks for directly:

    newly_correct   was wrong at baseline, right now
    lost_correct    was right at baseline, wrong now
    n_correct       total correct over all 8 windows
    delta           n_correct - previous node's n_correct

Pre-registered readings (chosen before the numbers exist)
---------------------------------------------------------
    interleave forgets ~nothing and ends clearly ahead of forward
        -> contiguous training duration is the main cause of forgetting
    all three arms end within noise of each other
        -> schedule is not the cause; capacity/optimisation binds
    reverse mirrors forward (forgetting moved to the opposite end of the list)
        -> confirms order-dependence, sharpens the schedule diagnosis

Smoke test (small, verifies the pipeline end to end, ~1h on one GPU):

    /share/latent/mllm_env/bin/python /tmp/e6d_schedule.py \
        --schedules forward,interleave --train-windows 2 \
        --steps-a-per-window 1 --steps-b-per-window 8 --probe-every 4 \
        --out /tmp/e6d_smoke.json

Full run (three arms, ~11 measurement nodes each):

    /share/latent/mllm_env/bin/python /tmp/e6d_schedule.py \
        --out /tmp/e6d_schedule_report.json
"""

import argparse
import copy
import hashlib
import json
import os
import sys

# GPU pick before torch import (same pattern as e6b_mini; CUDA_VISIBLE_DEVICES
# must be set before the first CUDA call, and CUBLAS_WORKSPACE_CONFIG before
# cublas init or deterministic algorithms refuse to run).
TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = "/share/latent/PRLM-C"
for path in ("/tmp", TOOLS, REPO):
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _reexec_on_free_gpu():
    import subprocess

    need_mb = int(os.environ.get("E6D_NEED_MB", "26000"))
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
    print("[e6d] gpu%d, %d MB free" % (free[0][1], free[0][0]), flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(free[0][1])
    os.environ["E6D_GPU_SET"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


if __name__ == "__main__" and not os.environ.get("E6D_GPU_SET") and not os.environ.get("E6D_NO_GPU"):
    _reexec_on_free_gpu()

import torch  # noqa: E402

# Heavy imports live inside main(): the schedule constructors below are pure
# functions and stay importable/verifiable on a host without torch or the repo
# tree (this checkout is developed off-box and verified by py_compile + the
# --dry-schedule assertions before deployment).

DEVICE = "cuda:0"

# E6c's training windows, verbatim (runs/e6c_train_report.json partition).
# Freezing the fixture is deliberate: same windows, same contexts, same probes
# as the run whose retention flipped to `overwrites`, so the only moving part
# is the order. Re-screening would change the fixture and the comparison with
# E6c's numbers with it.
DEFAULT_WINDOWS = [
    (3, 17890), (3, 17130), (3, 16750), (3, 14470),
    (3, 17510), (3, 14850), (3, 16370), (3, 15230),
]


# --- schedule construction (pure; unit-testable without torch) ----------------

def build_schedules(n_windows, steps_a, steps_b, chunk):
    """The three window orders as event lists [(stage, window_index, steps)].

    Invariants, asserted rather than hoped:
      * every window sees exactly `steps_a` A-steps and `steps_b` B-steps in
        every schedule -- totals are identical, only the order differs;
      * interleave's chunking divides steps_b exactly (96/4 = 24 rounds);
      * forward and reverse are exact mirrors.
    """
    if steps_b % chunk != 0:
        raise ValueError(
            "interleave needs steps_b %% chunk == 0 (got %d %% %d)" % (steps_b, chunk)
        )
    order = list(range(n_windows))

    def stage_a(seq):
        return [("A", i, steps_a) for i in seq]

    def contiguous(seq):
        return stage_a(seq) + [("B", i, steps_b) for i in seq]

    interleave_a = stage_a(order)  # one 4-step pass per window == forward's A
    interleave_b = []
    for _ in range(steps_b // chunk):
        interleave_b += [("B", i, chunk) for i in order]

    schedules = {
        "forward": contiguous(order),
        "reverse": contiguous(order[::-1]),
        "interleave": interleave_a + interleave_b,
    }
    for name, events in schedules.items():
        a_total = sum(s for st, _, s in events if st == "A")
        b_total = sum(s for st, _, s in events if st == "B")
        if a_total != steps_a * n_windows or b_total != steps_b * n_windows:
            raise AssertionError("schedule %s broke the step-total invariant" % name)
        for i in range(n_windows):
            per_a = sum(s for st, w, s in events if st == "A" and w == i)
            per_b = sum(s for st, w, s in events if st == "B" and w == i)
            if per_a != steps_a or per_b != steps_b:
                raise AssertionError(
                    "schedule %s: window %d got A=%d B=%d, want %d/%d"
                    % (name, i, per_a, per_b, steps_a, steps_b)
                )
    return schedules


def probe_points(steps_a_total, steps_b_total, probe_every):
    """Measurement nodes as cum-steps values.

    Aligned to the *forward* schedule's segment boundaries so all three arms
    are probed at the same cum-steps and the same per-window B-budgets have
    been spent: 0 (baseline), after A, then after every full interleave round
    (probe_every B-steps = one forward window = chunk*(n_windows)/n_windows
    rounds of interleave), and the end.
    """
    points = {0}
    if steps_a_total > 0:
        points.add(steps_a_total)
    total = steps_a_total + steps_b_total
    k = steps_a_total + probe_every
    while k < total:
        points.add(k)
        k += probe_every
    points.add(total)
    return sorted(points)


# --- measurement ---------------------------------------------------------------

def snapshot(manager, cache, entries, compactor, args, note, cum, B):
    """Score every window with the current weights; return one timeline node.

    Forward-only, one window at a time, cache cleared between windows -- the
    same discipline as e6b's immediate/final measurements. An OOM retries once
    after a clear (no rollback needed: nothing is trained here).
    """
    windows = {}
    for entry in entries:
        key = "%d:%d" % (entry["row"], entry["start"])
        for attempt in (1, 2):
            try:
                result = B.evaluate_window(
                    manager, cache[entry["row"]], entry, compactor, args
                )
                break
            except torch.OutOfMemoryError:
                if attempt == 2:
                    raise
                print("[e6d]     snapshot OOM at %s; clearing and retrying" % key, flush=True)
                B.release()
        windows[key] = {
            "n_correct": sum(bool(v) for v in result["after"]["correct"].values()),
            "n": result["after"]["n"],
            "recovered": result["recovery"]["n_recovered"],
            "n_lost": result["recovery"]["n_lost"],
            "recovered_questions": result["recovery"]["recovered_questions"],
            "signature": result["after_signature"],
            "correct": {q: bool(v) for q, v in result["after"]["correct"].items()},
        }
        B.release()
    n_correct = sum(w["n_correct"] for w in windows.values())
    node = {
        "cum_steps": cum,
        "note": note,
        "n_correct": n_correct,
        "recovered_lost": sum(w["recovered"] for w in windows.values()),
        "n_lost": sum(w["n_lost"] for w in windows.values()),
        "windows": windows,
    }
    print("[e6d]   node cum=%-4d %-28s correct %3d/%3d  lost-recovered %d/%d"
          % (cum, note, n_correct, sum(w["n"] for w in windows.values()),
             node["recovered_lost"], node["n_lost"]), flush=True)
    return node


def derive_question_records(nodes, entries):
    """Per-question traces and learn/forget events from the node sequence.

    `learned_at`/`forgotten_at` are lists of cum-steps: a question may learn,
    forget and relearn, and the *sequence* is the evidence for or against the
    schedule hypothesis, so no event is collapsed away.
    """
    records = {}
    for entry in entries:
        key = "%d:%d" % (entry["row"], entry["start"])
        questions = [p.question for p in entry["probes"]]
        for q in questions:
            trace = [bool(node["windows"][key]["correct"].get(q, False)) for node in nodes]
            learned_at, forgotten_at = [], []
            for i in range(1, len(trace)):
                if not trace[i - 1] and trace[i]:
                    learned_at.append(nodes[i]["cum_steps"])
                if trace[i - 1] and not trace[i]:
                    forgotten_at.append(nodes[i]["cum_steps"])
            records.setdefault(key, {})[q] = {
                "trace": trace,
                "learned_at": learned_at,
                "forgotten_at": forgotten_at,
                "right_at_baseline": trace[0],
                "right_at_end": trace[-1],
            }
    return records


def arm_summary(nodes, records, entries):
    """The aggregates the question is asked in: learned / forgotten / net."""
    baseline = nodes[0]
    final = nodes[-1]
    learned, forgotten = [], []
    for key, per_q in records.items():
        for q, rec in per_q.items():
            if rec["right_at_baseline"] and not rec["right_at_end"]:
                forgotten.append("%s|%s" % (key, q))
            if not rec["right_at_baseline"] and rec["right_at_end"]:
                learned.append("%s|%s" % (key, q))
    per_node = []
    for i, node in enumerate(nodes):
        delta = node["n_correct"] - nodes[i - 1]["n_correct"] if i else 0
        per_node.append({
            "cum_steps": node["cum_steps"], "note": node["note"],
            "n_correct": node["n_correct"], "delta_vs_prev": delta,
            "recovered_lost": node["recovered_lost"], "n_lost": node["n_lost"],
        })
    return {
        "nodes": per_node,
        "final_n_correct": final["n_correct"],
        "baseline_n_correct": baseline["n_correct"],
        "learned_vs_baseline": learned,
        "forgotten_vs_baseline": forgotten,
        "net_vs_baseline": final["n_correct"] - baseline["n_correct"],
        "final_recovered_lost": final["recovered_lost"],
        "final_n_lost": final["n_lost"],
        "n_learn_events": sum(len(r["learned_at"]) for per_q in records.values() for r in per_q.values()),
        "n_forget_events": sum(len(r["forgotten_at"]) for per_q in records.values() for r in per_q.values()),
    }


# --- main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0,
                        help="fixed before model/manager/compactor construction")
    parser.add_argument("--schedules", default="forward,reverse,interleave",
                        help="comma-separated subset of the three arms")
    parser.add_argument("--steps-a-per-window", type=int, default=4)
    parser.add_argument("--steps-b-per-window", type=int, default=96)
    parser.add_argument("--interleave-chunk", type=int, default=4,
                        help="contiguous B-steps per window visit in the interleave arm")
    parser.add_argument("--probe-every", type=int, default=96,
                        help="B-steps between full 8-window measurement nodes")
    parser.add_argument("--train-windows", type=int, default=8,
                        help="use the first N windows of --windows")
    parser.add_argument("--windows", default=None,
                        help="row:start pairs, comma-separated; defaults to E6c's 8")
    parser.add_argument("--probes", type=int, default=20)
    parser.add_argument("--max-probe-tokens", type=int, default=256)
    parser.add_argument("--min-probes", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma-max", type=float, default=0.05)
    parser.add_argument("--candidates", default="/tmp/e6a_screen_data.json")
    parser.add_argument("--out", default="/tmp/e6d_schedule_report.json")
    parser.add_argument("--ckpt-prefix", default="/tmp/e6d_ckpt")
    parser.add_argument("--no-deterministic", action="store_true",
                        help="skip torch.use_deterministic_algorithms (debug only)")
    parser.add_argument("--dry-schedule", action="store_true",
                        help="print the three event lists, assert invariants, exit")
    args = parser.parse_args()

    schedules = build_schedules(
        args.train_windows, args.steps_a_per_window, args.steps_b_per_window,
        args.interleave_chunk,
    )
    if args.dry_schedule:
        for name, events in schedules.items():
            flat = " ".join("%s%dx%d" % (st.lower(), w, s) for st, w, s in events)
            print("%-10s (%d events) %s" % (name, len(events), flat))
        pts = probe_points(
            args.steps_a_per_window * args.train_windows,
            args.steps_b_per_window * args.train_windows, args.probe_every,
        )
        print("probe nodes: %s" % pts)
        return

    # --- determinism setup, before anything touches CUDA --------------------
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
    if not args.no_deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    wanted = [s.strip() for s in args.schedules.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in schedules]
    if unknown:
        raise SystemExit("[e6d] unknown schedule(s) %s; have %s" % (unknown, list(schedules)))

    import e6a_overfit as E  # window building, scoring, recovery
    import e6b_mini as B  # measure/evaluate/train_window machinery

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
    config["stats_dir"] = "./outputs/prlmc-e6d"
    config["carrier_gamma_max"] = args.gamma_max
    cfg = PRLMCConfig.from_dict(config)

    model = MemoryLLM.from_pretrained(E.MODEL, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = model.to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(E.MODEL)
    manager = ProtectedResidualMemory(model, tokenizer, cfg)

    compactor = CarrierResidualCompactor(
        num_layers=manager.num_layers,
        hidden=manager.hidden,
        block=manager.block,
        rank=int(cfg.carrier_rank),
        gamma_max=float(cfg.carrier_gamma_max),
    ).to(device=DEVICE, dtype=torch.float32)

    # The single init every arm starts from. The seed fixes it once; the
    # snapshot + deep-copy makes the three arms bitwise-identical at start,
    # which is stronger than re-seeding before each arm.
    initial_state = copy.deepcopy(compactor.state_dict())
    init_md5 = hashlib.md5(
        json.dumps(
            {k: hashlib.md5(v.detach().cpu().numpy().tobytes()).hexdigest()
             for k, v in initial_state.items()},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    print("[e6d] seed %d, initial_state md5 %s" % (args.seed, init_md5), flush=True)

    windows = DEFAULT_WINDOWS if not args.windows else [
        tuple(int(v) for v in pair.split(":")) for pair in args.windows.split(",")
    ]
    windows = windows[: args.train_windows]
    if len(windows) < args.train_windows:
        raise SystemExit("[e6d] --windows gave %d entries, need %d"
                         % (len(windows), args.train_windows))

    cache = {}
    for row in sorted({r for r, _ in windows}):
        cache[row] = E.read_fact_lines(row)

    # --- fixture: measure every window once, with the initial weights --------
    # Same machinery as E6b/E6c screening (no_memory / uncompacted / before),
    # so `lost` and `recovered` keep their E6b semantics and the numbers stay
    # comparable across experiments.
    entries = []
    for row, start in windows:
        entry = B.measure_window(
            manager, cache[row], row, start, compactor, initial_state,
            args.probes, args, log_prefix="[e6d] fixture",
        )
        if entry is None:
            raise SystemExit("[e6d] fixture window %d:%d unusable" % (row, start))
        entries.append(entry)
        print("[e6d] fixture %d:%d probes %d eligible %d lost %d"
              % (row, start, entry["n_probes"], entry["n_eligible"], entry["n_lost"]),
              flush=True)
    del compactor
    torch.cuda.empty_cache()

    steps_a_total = args.steps_a_per_window * len(entries)
    steps_b_total = args.steps_b_per_window * len(entries)
    points = probe_points(steps_a_total, steps_b_total, args.probe_every)

    report = {
        "design": {
            "question": "is the forgetting caused by contiguous per-window training?",
            "seed": args.seed,
            "initial_state_md5": init_md5,
            "schedules": wanted,
            "steps_a_per_window": args.steps_a_per_window,
            "steps_b_per_window": args.steps_b_per_window,
            "interleave_chunk": args.interleave_chunk,
            "probe_every": args.probe_every,
            "probe_nodes": points,
            "lr": args.lr,
            "gamma_max": float(cfg.carrier_gamma_max),
            "structure_changed": False,
            "loss_changed": False,
            "fixture": "E6c's 8 training windows (runs/e6c_train_report.json)",
            "deterministic_algorithms": not args.no_deterministic,
        },
        "windows": ["%d:%d" % (r, s) for r, s in windows],
        "probes": {
            "%d:%d" % (e["row"], e["start"]): [p.question for p in e["probes"]]
            for e in entries
        },
        "baseline": {
            "%d:%d" % (e["row"], e["start"]): {
                "n_eligible": e["n_eligible"], "n_lost": e["n_lost"],
                "lost_questions": e["lost_questions"],
                "uncompacted_n_correct": sum(
                    bool(v) for v in e["uncompacted"]["correct"].values()),
                "before_n_correct": sum(bool(v) for v in e["before"]["correct"].values()),
            } for e in entries
        },
        "arms": {},
    }

    # Baseline node + determinism proof: same weights, two independent full
    # scoring passes; the signatures must match or no trajectory comparison in
    # this report means anything.
    template = B.before_compactor_for(
        CarrierResidualCompactor(
            num_layers=manager.num_layers, hidden=manager.hidden, block=manager.block,
            rank=int(cfg.carrier_rank), gamma_max=float(cfg.carrier_gamma_max),
        ).to(device=DEVICE, dtype=torch.float32),
        initial_state,
    )
    base_a = snapshot(manager, cache, entries, template, args, "baseline (pass 1)", 0, B)
    base_b = snapshot(manager, cache, entries, template, args, "baseline (pass 2)", 0, B)
    sig_a = {k: v["signature"] for k, v in base_a["windows"].items()}
    sig_b = {k: v["signature"] for k, v in base_b["windows"].items()}
    scoring_deterministic = sig_a == sig_b
    print("[e6d] scoring path deterministic: %s" % scoring_deterministic, flush=True)
    if not scoring_deterministic:
        print("[e6d] WARNING: two identical-weight scoring passes disagreed; "
              "learn/forget events carry seed-level noise", flush=True)
    report["design"]["scoring_deterministic"] = scoring_deterministic
    del template, base_b
    torch.cuda.empty_cache()

    # --- the three arms ------------------------------------------------------
    def fresh_arm():
        """A compactor carrying the snapshot weights, sharing no storage."""
        return B.before_compactor_for(
            CarrierResidualCompactor(
                num_layers=manager.num_layers, hidden=manager.hidden,
                block=manager.block, rank=int(cfg.carrier_rank),
                gamma_max=float(cfg.carrier_gamma_max),
            ).to(device=DEVICE, dtype=torch.float32),
            initial_state,
        )

    def state_md5(module):
        return hashlib.md5(
            json.dumps(
                {k: hashlib.md5(v.detach().cpu().numpy().tobytes()).hexdigest()
                 for k, v in module.state_dict().items()},
                sort_keys=True,
            ).encode()
        ).hexdigest()

    for name in wanted:
        events = schedules[name]
        print("[e6d] === arm %s: %d events, %d A + %d B steps ==="
              % (name, len(events), steps_a_total, steps_b_total), flush=True)
        compactor_arm = fresh_arm()
        # The snapshot is the single source of truth; every arm must start
        # bitwise-identical to it, and that is verified rather than assumed.
        arm_md5 = state_md5(compactor_arm)
        if arm_md5 != init_md5:
            raise AssertionError(
                "arm %s initial weights diverged from the snapshot (%s != %s)"
                % (name, arm_md5, init_md5)
            )

        optimiser = torch.optim.AdamW(compactor_arm.parameters(), lr=args.lr)
        nodes = [base_a]
        trained_log = []
        cum = 0
        window_counts = [0] * len(entries)
        for ev_index, (stage, widx, steps) in enumerate(events):
            entry = entries[widx]
            trained = B.train_window(
                manager, cache[entry["row"]], entry, compactor_arm, optimiser,
                stage, steps, args,
            )
            cum += steps
            if stage == "B":
                window_counts[widx] += steps
            trained_log.append({
                "event": ev_index, "stage": stage,
                "window": "%d:%d" % (entry["row"], entry["start"]),
                "steps": steps, "cum_steps": cum,
                "initial_loss": trained["initial_loss"],
                "final_loss": trained["final_loss"],
            })
            print("[e6d]   %s %-10s w%d x%-3d cum %-4d loss %.4f -> %.4f"
                  % (name, stage, widx, steps, cum,
                     trained["initial_loss"], trained["final_loss"]), flush=True)
            if cum in points:
                if cum == steps_a_total:
                    note = "after stage A"
                elif cum == steps_a_total + steps_b_total:
                    note = "end of training"
                else:
                    note = "mid-B"
                nodes.append(snapshot(
                    manager, cache, entries, compactor_arm, args, note, cum, B
                ))

        # The end-of-training node must exist even if probe_every missed it.
        if nodes[-1]["cum_steps"] != steps_a_total + steps_b_total:
            nodes.append(snapshot(manager, cache, entries, compactor_arm, args,
                                  "end of training", cum, B))

        records = derive_question_records(nodes, entries)
        summary = arm_summary(nodes, records, entries)
        torch.save({"state_dict": compactor_arm.state_dict()},
                   "%s_%s.pt" % (args.ckpt_prefix, name))
        report["arms"][name] = {
            "schedule_events": len(events),
            "per_window_b_steps": window_counts,
            "per_window_b_steps_expected": [args.steps_b_per_window] * len(entries),
            "trained_log": trained_log,
            "nodes": nodes,
            "question_records": records,
            "summary": summary,
        }
        print("[e6d] arm %s done: baseline %d -> final %d correct "
              "(learned %d, forgotten %d, net %+d); lost-recovered %d/%d"
              % (name, summary["baseline_n_correct"], summary["final_n_correct"],
                 len(summary["learned_vs_baseline"]), len(summary["forgotten_vs_baseline"]),
                 summary["net_vs_baseline"], summary["final_recovered_lost"],
                 summary["final_n_lost"]), flush=True)
        del compactor_arm, optimiser
        torch.cuda.empty_cache()

    # --- cross-arm comparison ------------------------------------------------
    comparison = {
        name: {
            "final_n_correct": s["final_n_correct"],
            "net_vs_baseline": s["net_vs_baseline"],
            "learned_vs_baseline": len(s["learned_vs_baseline"]),
            "forgotten_vs_baseline": len(s["forgotten_vs_baseline"]),
            "n_forget_events": s["n_forget_events"],
            "n_learn_events": s["n_learn_events"],
            "final_recovered_lost": s["final_recovered_lost"],
            "final_n_lost": s["final_n_lost"],
        } for name, s in (
            (n, report["arms"][n]["summary"]) for n in report["arms"]
        )
    }
    report["comparison"] = comparison
    json.dump(report, open(args.out, "w"), indent=2, default=str)

    print("[e6d] ---- three-arm comparison (final, vs baseline) ----", flush=True)
    print("[e6d] %-11s %8s %6s %8s %10s %12s" %
          ("schedule", "correct", "net", "learned", "forgotten", "lost-recovered"),
          flush=True)
    for name, row in comparison.items():
        print("[e6d] %-11s %8d %+6d %8d %10d %12d/%d"
              % (name, row["final_n_correct"], row["net_vs_baseline"],
                 row["learned_vs_baseline"], row["forgotten_vs_baseline"],
                 row["final_recovered_lost"], row["final_n_lost"]), flush=True)
    print("[e6d] report %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
