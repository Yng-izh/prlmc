"""G2: version-stratified analysis of block selection (S1-SDC gate 2).

Pre-registered criterion (S1-SDC.md section 5): the drop rate of *superseded*
blocks must differ by >= 2x between saliency selection and similarity selection.

Mechanism being tested: G1 found sic systematically keeps the oldest window
blocks [0-3] that select discards. G2 asks what those blocks *are*. Three
line-level strata, from the same (slot, subject) grouping G0 used:

  superseded  a fact line whose subject has a LATER line in the window
  current     the latest line of each multi-occurrence subject
  singleton   a line whose subject occurs exactly once in the window

The predictions the SDC story makes:
  * sic should keep `current` lines' blocks (probe gradients land on the live
    version) and be at least as willing as select to drop `superseded` lines'
    blocks;
  * select, being recency-led but version-blind (G0), should waste budget on
    near-duplicate middle blocks and drop both old and new information
    indiscriminately;
  * the probes sic rescued in G1 (select-lost, sic-kept) should sit
    disproportionately on `singleton` lines -- information with no duplicate
    anywhere in the window, where a drop is unrecoverable.

This tool does NOT rerun the gradient pass. Window construction is
deterministic (same row, same start line), so the windows are rebuilt and the
G1 report's kept sets are projected onto them. Two guards make that sound:
  1. re-running `_select_block_indices` on the rebuilt window must reproduce
     the report's `select` kept set exactly (proves the pool bytes match);
  2. the regenerated probe set's questions must be a superset of the report's
     recorded lost questions for that row.

Run on v100-2 (rebuild only, no backward pass):
  /share/latent/mllm_env/bin/python /share/latent/PRLM-C/tools/g2_version_strata.py \
      --g1 /share/latent/PRLM-C/runs/g1_report.json \
      --out /share/latent/PRLM-C/runs/g2_report.json
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PARQUET = "/share/latent/MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet"
REPO = "/share/latent/PRLM-C"
MODEL = "/share/latent/MemoryLLM-8B"
CONFIG = os.path.join(REPO, "configs/agent_conf/prlmc_e6.yaml")

DEVICE = "cuda:0"


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


if __name__ == "__main__" and not os.environ.get("G2_GPU_SET"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(pick_gpu())
    os.environ["G2_GPU_SET"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import torch  # noqa: E402

import e6a_overfit as E  # noqa: E402
from methods.prlmc.consolidate import _select_block_indices  # noqa: E402


def strata_for_window(lines, window_lines, l2b):
    """Line-level strata from the G0 grouping rule.

    Returns (strata, groups) where strata maps line index -> 'superseded' |
    'current' | 'singleton' and groups maps (slot, subject) -> [line indices]
    in serial order. `window_lines` is the set of line indices inside the
    compaction window (probe-eligible region).
    """
    groups = {}
    for index in sorted(window_lines):
        found = E.first_match(lines[index])
        if found is None:
            continue
        slot, subject, _answer, _q = found
        groups.setdefault((slot, subject), []).append(index)

    strata = {}
    for key, idxs in groups.items():
        if len(idxs) == 1:
            strata[idxs[0]] = "singleton"
        else:
            strata[idxs[-1]] = "current"
            for old in idxs[:-1]:
                strata[old] = "superseded"
    # Lines the templates do not match at all: no version semantics, but they
    # still occupy slots. Counted as singleton for drop-rate purposes -- a
    # dropped unmatched line is as unrecoverable as a dropped singleton fact.
    for index in sorted(window_lines):
        strata.setdefault(index, "singleton")
    return strata, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1", default="/share/latent/PRLM-C/runs/g1_report.json")
    ap.add_argument("--out", default="/tmp/g2_report.json")
    ap.add_argument("--start", type=int, default=30)
    args = ap.parse_args()

    g1 = json.load(open(args.g1))
    if g1.get("verdict") not in ("PASS",):
        print("[g2] WARNING: analysing a G1 report whose verdict is %s"
              % g1.get("verdict"), flush=True)

    import yaml
    from transformers import AutoTokenizer
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory

    raw = yaml.safe_load(open(CONFIG))
    config = dict(raw["prlmc"])
    config.update(device=DEVICE, dtype="float16", stats_dir="./outputs/prlmc-g2")
    cfg = PRLMCConfig.from_dict(config)

    model = MemoryLLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                      attn_implementation="sdpa")
    model = model.to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    manager = ProtectedResidualMemory(model, tokenizer, cfg)

    window_slots = E.WINDOW_BLOCKS * manager.block

    # Line-level tallies: stratum -> {arm: [kept_flags]}
    tally = {"superseded": {"select": [], "sic": [], "oldest": [], "random": []},
             "current":    {"select": [], "sic": [], "oldest": [], "random": []},
             "singleton":  {"select": [], "sic": [], "oldest": [], "random": []}}
    # Probe-level: what the rescued / still-lost probes are supported by
    probe_strata = {"rescued": [], "still_lost": [], "select_lost_all": []}
    # Same-block version pairs (purity diagnostic)
    same_block_pairs = 0
    total_pairs = 0
    per_row_reports = []

    for entry in g1["per_row"]:
        row = entry["row"]
        lines = E.read_fact_lines(row)
        covered = E.write_until(manager, lines, window_slots, args.start)
        if covered is None:
            raise SystemExit("row %d window rebuild failed; G1 kept sets unusable"
                             % row)
        l2b = {}
        window_lines = set()
        for block_index, (lo, hi, _b, _w) in enumerate(covered):
            for line in range(lo, hi):
                l2b[line] = block_index
                window_lines.add(line)

        # Guard 1: the rebuilt pool must reproduce the report's select kept.
        begin = manager.total - manager.used
        source = manager._memory[:, begin: begin + window_slots, :].to(DEVICE)
        weights = manager._window_importance(begin, window_slots)
        rebuilt_kept = _select_block_indices(
            source, E.TARGET_BLOCKS * manager.block, manager.block,
            weights, manager.goal_direction, manager.select_weights,
        ).tolist()
        del source, weights
        torch.cuda.empty_cache()
        report_kept = sorted(entry["arms"]["select"]["kept"])
        if rebuilt_kept != report_kept:
            raise SystemExit(
                "row %d: rebuilt select kept %s != report %s; window construction "
                "is not deterministic against the G1 run"
                % (row, rebuilt_kept, report_kept))

        strata, groups = strata_for_window(lines, window_lines, l2b)
        kept = {"select": set(entry["arms"]["select"]["kept"]),
                "sic": set(entry["arms"]["sic"]["kept"]),
                "oldest": set(entry["arms"]["oldest"]["kept"]),
                "random": set(entry["arms"]["random"]["kept"])}

        for line, cls in strata.items():
            block = l2b[line]
            for arm in ("select", "sic", "oldest", "random"):
                tally[cls][arm].append(1 if block in kept[arm] else 0)

        for key, idxs in sorted(groups.items()):
            if len(idxs) < 2:
                continue
            total_pairs += 1
            if l2b[idxs[-1]] == l2b[idxs[0]]:
                same_block_pairs += 1

        # Guard 2 + probe-level strata. Regenerate the probe set with the same
        # rule G1 used (support inside select-dropped blocks), tracking lines.
        dropped_native = sorted(set(range(E.WINDOW_BLOCKS)) - kept["select"])
        indices = E.probe_indices(covered, window_slots)
        probes = E.probes_from_lines(
            lines, indices, g1["design"]["probes"],
            dropped_blocks=set(dropped_native), line_to_block=l2b,
        )
        question_line = {}
        for index in sorted(set(indices)):
            if l2b.get(index) not in set(dropped_native):
                continue
            found = E.first_match(lines[index])
            if found is None:
                continue
            slot, subject, _answer, question = found
            # probes_from_lines keeps the *latest* line per (slot, subject) only
            latest = None
            for j, text in enumerate(lines):
                f = E.first_match(text)
                if f and (f[0], f[1]) == (slot, subject):
                    latest = j
            if latest is not None and latest in window_lines:
                filled = question % subject
                question_line.setdefault(filled, latest)
        probe_line = {}
        for probe in probes:
            if probe.question in question_line:
                probe_line[probe.question] = question_line[probe.question]

        select_lost = set(entry["arms"]["select"]["lost"])
        sic_lost = set(entry["arms"]["sic"]["lost"])
        for q in select_lost:
            cls = strata.get(probe_line.get(q, -1), "unknown")
            probe_strata["select_lost_all"].append(cls)
            if q not in sic_lost:
                probe_strata["rescued"].append(cls)
            else:
                probe_strata["still_lost"].append(cls)

        n_by = {c: sum(1 for v in strata.values() if v == c)
                for c in ("superseded", "current", "singleton")}
        per_row_reports.append({
            "row": row,
            "lines_by_stratum": n_by,
            "n_version_groups": sum(1 for v in groups.values() if len(v) >= 2),
            "kept_overlap_select_sic": len(kept["select"] & kept["sic"]),
        })
        print("[g2] row %d: superseded %d / current %d / singleton %d lines; "
              "version groups %d; kept overlap select-sic %d/9"
              % (row, n_by["superseded"], n_by["current"], n_by["singleton"],
                 per_row_reports[-1]["n_version_groups"],
                 per_row_reports[-1]["kept_overlap_select_sic"]), flush=True)

    # --- drop rates -----------------------------------------------------------
    drop_rates = {}
    for cls in ("superseded", "current", "singleton"):
        drop_rates[cls] = {}
        for arm in ("select", "sic", "oldest", "random"):
            flags = tally[cls][arm]
            drop_rates[cls][arm] = {
                "n": len(flags),
                "kept": sum(flags),
                "drop_rate": round(1.0 - sum(flags) / len(flags), 4) if flags else None,
            }

    sic_sup = drop_rates["superseded"]["sic"]["drop_rate"]
    sel_sup = drop_rates["superseded"]["select"]["drop_rate"]
    ratio = (sic_sup / sel_sup) if sel_sup else (float("inf") if sic_sup else float("nan"))

    def dist(seq):
        out = {}
        for c in ("superseded", "current", "singleton", "unknown"):
            out[c] = sum(1 for x in seq if x == c)
        return out

    # Verdict per the pre-registration: superseded drop rate sic vs select >= 2x.
    verdict = "PASS" if (ratio == ratio and ratio >= 2.0) else "FAIL"
    # A secondary, unplanned-but-load-bearing read: singleton protection.
    sic_single = drop_rates["singleton"]["sic"]["drop_rate"]
    sel_single = drop_rates["singleton"]["select"]["drop_rate"]

    report = {
        "design": {
            "g1_report": args.g1,
            "start": args.start,
            "strata": "line-level, window-local, G0 grouping rule",
            "criterion": "superseded drop rate sic / select >= 2.0",
            "guards": ["rebuilt select kept == report kept (pool determinism)",
                       "probe regeneration from same rule"],
        },
        "per_row": per_row_reports,
        "drop_rates": drop_rates,
        "superseded_drop_ratio_sic_over_select": (round(ratio, 4)
                                                  if ratio == ratio else None),
        "version_pair_purity": {
            "total_pairs": total_pairs,
            "same_block_pairs": same_block_pairs,
            "note": "same-block pairs are pure for line-level strata (a block "
                    "mixing current and superseded lines still counts each line)",
        },
        "probe_level": {
            "select_lost": dist(probe_strata["select_lost_all"]),
            "rescued_by_sic": dist(probe_strata["rescued"]),
            "still_lost": dist(probe_strata["still_lost"]),
        },
        "secondary_singleton_drop": {
            "sic": sic_single, "select": sel_single,
            "note": "unplanned diagnostic; G1's rescued probes should sit here",
        },
        "verdict": verdict,
    }
    with open(args.out, "w") as f:
        json.dump(report, f)
    print("[g2] drop rates by stratum:", flush=True)
    for cls in ("superseded", "current", "singleton"):
        row = drop_rates[cls]
        print("[g2]   %-11s select %.3f  sic %.3f  oldest %.3f  random %.3f"
              % (cls, row["select"]["drop_rate"], row["sic"]["drop_rate"],
                 row["oldest"]["drop_rate"], row["random"]["drop_rate"]), flush=True)
    print("[g2] superseded drop ratio sic/select = %.3f -> %s"
          % (ratio if ratio == ratio else -1, verdict), flush=True)
    print("[g2] probe-level: select_lost %s | rescued %s | still_lost %s"
          % (dist(probe_strata["select_lost_all"]), dist(probe_strata["rescued"]),
             dist(probe_strata["still_lost"])), flush=True)
    print("[g2] report %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
