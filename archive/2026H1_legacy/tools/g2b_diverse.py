"""G2b: diverse, selection-independent probe evaluation for S1-SDC.

Why this exists
----------------
G1's probes were filtered to facts in blocks that native ``select`` drops.
G2 then found that 18/19 select-lost questions were singleton facts. That is
valid evidence for the original G1 protocol, but it mixes the method effect
with a narrow probe distribution. G2b removes that conditioning.

Protocol
--------
* Rebuild the same four 19-block windows as G1.
* Label valid latest-answer facts as ``current`` (a subject has multiple facts
  and this is the newest) or ``singleton`` (one fact in the window).
* For every row, sample equal numbers from both strata, without looking at
  select's kept/dropped blocks.
* Split by line: discovery probes are used for SIC gradients, held-out probes
  are never used for selection. Each split gets a deterministic mix of the
  existing relation templates and a lexical question variant.
* Evaluate select and SIC on the held-out probes using the existing full/no-
  memory eligibility rule.

Superseded lines are still reported at block level, exactly as G2. They are
not made into gold QA probes: Conflict_Resolution defines the newest fact as
the answer, so an old fact has no valid answer under the frozen B2 protocol.

The result is G2b, a robustness check rather than a replacement of the
pre-registered G2. It answers whether G1 survives when the probe set is not
selected by the native baseline and when the evaluation contains balanced
current/singleton facts.
"""

import argparse
import json
import os
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = "/share/latent/PRLM-C"
MODEL = "/share/latent/MemoryLLM-8B"
CONFIG = os.path.join(REPO, "configs/agent_conf/prlmc_e6.yaml")
DEVICE = "cuda:0"

if __name__ == "__main__" and not os.environ.get("G2B_GPU_SET"):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    free = sorted(
        [(32768 - int(used), int(index)) for index, used in
         (line.split(",") for line in out.strip().splitlines())],
        reverse=True,
    )
    if free[0][0] < 21000:
        raise SystemExit("no GPU with 21000 MB free")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(free[0][1])
    os.environ["G2B_GPU_SET"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import torch  # noqa: E402

import e6a_overfit as E  # noqa: E402
import sic_probe as S  # noqa: E402
from methods.prlmc.consolidate import _select_block_indices  # noqa: E402
from methods.prlmc.probes import Probe  # noqa: E402


def window_metadata(lines, covered):
    """Return line->block, valid latest facts, and block strata metadata."""
    l2b = {}
    window_lines = set()
    for block, (lo, hi, _before, _width) in enumerate(covered):
        for i in range(lo, hi):
            l2b[i] = block
            window_lines.add(i)

    groups = {}
    parsed = {}
    for i in sorted(window_lines):
        found = E.first_match(lines[i])
        if found is None:
            continue
        slot, subject, answer, question = found
        key = (slot, subject)
        groups.setdefault(key, []).append(i)
        parsed[i] = (answer, question, subject, key)

    strata = {}
    latest = {}
    for key, indices in groups.items():
        latest[key] = indices[-1]
        if len(indices) == 1:
            strata[indices[0]] = "singleton"
        else:
            strata[indices[-1]] = "current"
            for old in indices[:-1]:
                strata[old] = "superseded"
    return l2b, window_lines, groups, parsed, strata, latest


def question_variants(question, variant):
    """Two conservative B2-compatible forms, selected deterministically.

    The question remains the same factual query; only the surface wording
    changes. The original form is always available, while the second form
    tests a short natural-language paraphrase without introducing a new answer
    or changing the instruction template.
    """
    if variant % 2 == 0:
        return question
    replacements = (
        ("What is the country of citizenship of ", "Which country is "),
        ("What is the capital of ", "Which city is the capital of "),
        ("What continent is ", "On which continent is "),
        ("Who is ", "Which person is "),
        ("What language does ", "Which language does "),
        ("What religion is ", "Which religion is "),
        ("What sport is ", "Which sport is "),
        ("What position does ", "Which position does "),
        ("What is ", "Which fact describes "),
    )
    for old, new in replacements:
        if question.startswith(old):
            return new + question[len(old):]
    return question


def make_balanced_probes(lines, metadata, per_stratum, offset):
    """Create balanced discovery/held-out probes without select conditioning."""
    _l2b, _window_lines, groups, parsed, strata, latest = metadata
    by_class = {"current": [], "singleton": []}
    for key, index in sorted(latest.items(), key=lambda item: item[1]):
        answer, question, subject, _ = parsed[index]
        cls = strata[index]
        if cls not in by_class or not (1 <= len(answer.split()) <= 5):
            continue
        by_class[cls].append((index, answer, question, subject))

    # Interleaved deterministic selection keeps discovery and held-out line sets
    # disjoint. Small contexts get fewer samples in that stratum; the aggregate
    # still pools all four contexts without duplicating a fact as independent.
    discovery, heldout = [], []
    actual = {}
    for cls in ("current", "singleton"):
        candidates = by_class[cls]
        n_each = min(int(per_stratum), len(candidates) // 2)
        actual[cls] = n_each
        start = offset % 3
        for position, item in enumerate(candidates):
            if position < start:
                continue
            if len([x for x in discovery if x["stratum"] == cls]) < n_each:
                discovery.append({"item": item, "stratum": cls, "variant": position})
            elif len([x for x in heldout if x["stratum"] == cls]) < n_each:
                heldout.append({"item": item, "stratum": cls, "variant": position + 1})
            if (len([x for x in discovery if x["stratum"] == cls]) >= n_each and
                    len([x for x in heldout if x["stratum"] == cls]) >= n_each):
                break

    def render(entries):
        probes = []
        labels = {}
        for entry in entries:
            line, answer, question, subject = entry["item"]
            q = question_variants(question % subject, entry["variant"])
            probes.append(Probe(
                question=q, answers=[answer], split="train",
                prompt=E.query_stem(q),
            ))
            labels[q] = {"line": line, "stratum": entry["stratum"], "variant": entry["variant"]}
        return probes, labels

    return render(discovery), render(heldout), {
        "available": {cls: len(values) for cls, values in by_class.items()},
        "per_split": actual,
    }


def score_arm(manager, probes, blocks, args):
    pool, mask = S.pool_from_blocks(manager, blocks)
    scored = E.score_probes(manager, probes, pool, args.max_probe_tokens, mask=mask)
    del pool, mask
    torch.cuda.empty_cache()
    return scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="0,1,2,3")
    ap.add_argument("--start", type=int, default=30)
    ap.add_argument("--per-stratum", type=int, default=12)
    ap.add_argument("--max-probe-tokens", type=int, default=256)
    ap.add_argument("--basis-cap", type=int, default=8)
    ap.add_argument("--out", default="/tmp/g2b_report.json")
    args = ap.parse_args()

    import yaml
    from transformers import AutoTokenizer
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory

    raw = yaml.safe_load(open(CONFIG))
    cfg = PRLMCConfig.from_dict(dict(raw["prlmc"], device=DEVICE, dtype="float16",
                                     stats_dir="./outputs/prlmc-g2b"))
    model = MemoryLLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                      attn_implementation="sdpa").to(DEVICE)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    manager = ProtectedResidualMemory(model, tokenizer, cfg)

    rows = {int(row): E.read_fact_lines(int(row)) for row in args.rows.split(",")}
    pooled = {
        "discovery": {"select": [0, 0], "sic": [0, 0]},
        "heldout": {"select": [0, 0], "sic": [0, 0]},
    }
    per_row = []

    for row, lines in rows.items():
        covered = E.write_until(manager, lines, E.WINDOW_BLOCKS * manager.block, args.start)
        if covered is None:
            raise SystemExit("row %d could not build" % row)
        metadata = window_metadata(lines, covered)
        discovery, eval_set, counts = make_balanced_probes(
            lines, metadata, args.per_stratum, row,
        )
        (discovery_probes, discovery_labels), (eval_probes, eval_labels) = discovery, eval_set
        actual_n = counts["per_split"]
        if actual_n["current"] < 1 or actual_n["singleton"] < 1:
            raise SystemExit("row %d lacks both current and singleton probes: %s" % (row, counts))

        begin = manager.total - manager.used
        source = manager._memory[:, begin:begin + E.WINDOW_BLOCKS * manager.block, :].to(DEVICE)
        weights = manager._window_importance(begin, E.WINDOW_BLOCKS * manager.block)
        select_kept = _select_block_indices(
            source, E.TARGET_BLOCKS * manager.block, manager.block,
            weights, manager.goal_direction, manager.select_weights,
        ).tolist()
        del source, weights
        torch.cuda.empty_cache()

        state = {"probes": discovery_probes, "kept_native": select_kept}
        harvest_args = SimpleNamespace(arm="sicm", basis_cap=args.basis_cap,
                                       max_probe_tokens=args.max_probe_tokens)
        salience, bases = S.harvest_gradients(manager, state, harvest_args)
        sic_kept = S.salience_select(salience, E.TARGET_BLOCKS)

        arms = {"select": select_kept, "sic": sic_kept}
        row_out = {"row": row, "candidate_counts": counts,
                   "n_discovery": len(discovery_probes), "n_heldout": len(eval_probes),
                   "kept": arms, "strata": {"discovery": discovery_labels,
                                               "heldout": eval_labels},
                   "scores": {}}

        for split_name, probes, labels in (
            ("discovery", discovery_probes, discovery_labels),
            ("heldout", eval_probes, eval_labels),
        ):
            no_mem = E.score_no_memory(manager, probes, args.max_probe_tokens)
            full = E.score_uncompacted(manager, probes, args.max_probe_tokens)
            eligible = [q for q in labels
                        if full["correct"].get(q, False) and
                        not no_mem["correct"].get(q, False)]
            for arm, blocks in arms.items():
                scored = score_arm(manager, probes, blocks, args)
                lost = [q for q in eligible if not scored["correct"].get(q, False)]
                pooled[split_name][arm][0] += len(lost)
                pooled[split_name][arm][1] += len(eligible)
                by_stratum = {}
                for q in lost:
                    cls = labels[q]["stratum"]
                    by_stratum[cls] = by_stratum.get(cls, 0) + 1
                row_out["scores"].setdefault(split_name, {})[arm] = {
                    "n_eligible": len(eligible), "n_lost": len(lost),
                    "lost": lost, "lost_by_stratum": by_stratum,
                }
            del no_mem, full
            torch.cuda.empty_cache()
        per_row.append(row_out)
        print("[g2b] row %d: discovery select %d sic %d; heldout select %d sic %d"
              % (row, row_out["scores"]["discovery"]["select"]["n_lost"],
                 row_out["scores"]["discovery"]["sic"]["n_lost"],
                 row_out["scores"]["heldout"]["select"]["n_lost"],
                 row_out["scores"]["heldout"]["sic"]["n_lost"]), flush=True)

    def ratio(split):
        select_lost = pooled[split]["select"][0]
        sic_lost = pooled[split]["sic"][0]
        return {
            "select_lost": select_lost, "sic_lost": sic_lost,
            "sic_over_select": sic_lost / select_lost if select_lost else None,
            "reduction": 1.0 - sic_lost / select_lost if select_lost else None,
        }
    report = {
        "design": {"rows": args.rows, "start": args.start,
                   "per_stratum": args.per_stratum,
                   "strata": ["current", "singleton"],
                   "selection_independent": True,
                   "discovery_heldout_disjoint": True,
                   "superseded_probe_policy": "not gold-valid under newest-fact rule"},
        "per_row": per_row,
        "pooled": {split: ratio(split) for split in pooled},
        "verdict": "PASS" if (
            ratio("heldout")["reduction"] is not None and
            ratio("heldout")["reduction"] >= 0.30 and len(per_row) >= 3
        ) else "FAIL",
    }
    with open(args.out, "w") as f:
        json.dump(report, f)
    print("[g2b] pooled discovery:", report["pooled"]["discovery"], flush=True)
    print("[g2b] pooled heldout:", report["pooled"]["heldout"], flush=True)
    print("[g2b] verdict:", report["verdict"], flush=True)
    print("[g2b] report", args.out, flush=True)


if __name__ == "__main__":
    main()
