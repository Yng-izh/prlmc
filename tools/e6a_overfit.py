"""E6a: single-window overfit of the carrier-residual compactor.

The one question this answers: can the trained residual be made to beat native
block selection on the *same* compaction of the *same* window? It is the gate the
plan puts before any larger run -- "if one window cannot be overfit, do not run
any large experiment" -- so it is deliberately the smallest thing that can fail
informatively.

Why the probes are synthesised here
-----------------------------------
The benchmark's own questions in this file are multi-hop and paraphrased (every
one of the top-25 question patterns has count 1), so a fixed single-hop template
scores 0/0 against them. Rendering probes from the window's own fact lines is the
only way to get a gold answer that the pool actually contains.

Why the window is built through a live manager
----------------------------------------------
`CarrierResidualCompactor` is called with `goal_direction` and `select_weights`
supplied by the manager, and `_select_block_indices` is the same scorer the
`select_blocks` baseline uses. Building the window by writing real chunks keeps
every one of those inputs on the real code path, so the "before" number is the
baseline's number and not a re-derivation of it.

What is measured, in the plan's acceptance terms
------------------------------------------------
    uncompacted accuracy   upper bound: what the reader can do with no compaction
    select_blocks accuracy the baseline to beat (== carrier_residual at init)
    carrier_residual       after stage A, and after stage B
plus residual_ratio, gamma against its ceiling, and the per-component loss
trajectory (KD and reader-hidden must fall; carrier and stats must not run away).

Run on the host:
    /share/latent/mllm_env/bin/python /tmp/e6a_overfit.py --steps-a 16 --steps-b 24
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

WINDOW_BLOCKS = 19          # the real E6 window: 4864 slots
TARGET_BLOCKS = 9           # 2304 slots, compact_ratio 0.5
ANCHOR_LINES = 30
# Every write contributes exactly `num_tokens` = 256 slots (the delta is the
# trailing slice of the forward), so one write is one block and 19 blocks needs 19
# writes. The context has 455 numbered fact lines, so chunks must be ~20 lines
# for that many writes to fit; 60 lines/chunk ran out of context at 8 blocks.
LINES_PER_CHUNK = 20

# The reader is only in answer format when it is asked the way inference asks.
# Rendered by the driver rather than imported so the prompt is visible here.
QUERY_TEMPLATE = (
    "Pretend you are a knowledge management system. Each fact in the knowledge pool "
    "is provided with a serial number at the beginning, and the newer fact has larger "
    "serial number. \n You need to solve the conflicts of facts in the knowledge pool "
    "by finding the newest fact with larger serial number. You need to answer a "
    "question based on this rule. You should give a very concise answer without saying "
    "other words for the question **only** from the knowledge pool you have memorized "
    "rather than the real facts in real world. \n\nFor example:\n\n [Knowledge Pool] "
    "\n\n Question: Based on the provided Knowledge Pool, what is the name of the "
    "current president of Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: "
    "Based on the provided Knowledge Pool, {question} \nAnswer:"
)


# --- GPU pick, before torch is imported so CUDA_VISIBLE_DEVICES takes effect ---

def pick_gpu(need_mb: int = 21000) -> int:
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
        raise SystemExit("no GPU with %d MB free; best is gpu%d with %d MB" % (need_mb, free[0][1], free[0][0]))
    print("[e6a] gpu%d, %d MB free" % (free[0][1], free[0][0]), flush=True)
    return free[0][1]


if __name__ == "__main__" and not os.environ.get("E6A_GPU_SET"):
    _gpu = pick_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu)
    os.environ["E6A_GPU_SET"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "methods", "memoryllm_modeling"))

from methods.prlmc.config import PRLMCConfig  # noqa: E402
from methods.prlmc.consolidate import CarrierResidualCompactor, select_block_compact  # noqa: E402
from methods.prlmc.layout import ProtectedResidualMemory  # noqa: E402
from methods.prlmc.probes import (  # noqa: E402
    Probe,
    _probe_batches,
    differentiable_memory,
    score_candidate,
    train_carrier_compactor,
)


# --- fact lines -> probes ----------------------------------------------------

# Only templates whose question can be answered from the line alone. The regex is
# anchored, so a line matching none is simply not probed; nothing is guessed.
#
# A generic `^The (.+) of (.+) is (.+)$` template was removed: it asked "Who is the
# <relation> of <subject>?" for relations whose answer is not a person, producing
# unanswerable questions like "Who is the type of music that Dale Evans plays?".
TEMPLATES = [
    (r"^(.+) is a citizen of (.+)$", "What is the country of citizenship of %s?", "1"),
    (r"^The capital of (.+) is (.+)$", "What is the capital of %s?", "1"),
    (r"^(.+) is located in the continent of (.+)$", "What continent is %s located in?", "1"),
    (r"^(.+) is married to (.+)$", "Who is %s married to?", "1"),
    (r"^(.+) died in the city of (.+)$", "In what city did %s die?", "1"),
    (r"^(.+) was born in the city of (.+)$", "In what city was %s born?", "1"),
    (r"^(.+) was founded by (.+)$", "Who founded %s?", "1"),
    (r"^(.+) speaks the language of (.+)$", "What language does %s speak?", "1"),
    (r"^(.+) is affiliated with the religion of (.+)$", "What religion is %s affiliated with?", "1"),
    (r"^(.+) is associated with the sport of (.+)$", "What sport is %s associated with?", "1"),
    (r"^(.+) was performed by (.+)$", "Who performed %s?", "1"),
    (r"^(.+) was developed by (.+)$", "Who developed %s?", "1"),
    (r"^(.+) was created in the country of (.+)$", "In what country was %s created?", "1"),
    (r"^(.+) was written in the language of (.+)$", "In what language was %s written?", "1"),
    (r"^The author of (.+) is (.+)$", "Who is the author of %s?", "1"),
    (r"^(.+) author of (.+) is (.+)$", "Who is the author of %s?", "2"),
    (r"^The director of (.+) is (.+)$", "Who is the director of %s?", "1"),
    (r"^(.+) plays the position of (.+)$", "What position does %s play?", "1"),
    (r"^(.+) was founded in the city of (.+)$", "In what city was %s founded?", "1"),
    (r"^(.+) worked in the city of (.+)$", "In what city did %s work?", "1"),
    (r"^(.+) is employed by (.+)$", "Who is %s employed by?", "1"),
    (r"^(.+) is famous for (.+)$", "What is %s famous for?", "1"),
    (r"^(.+) was created by (.+)$", "Who created %s?", "1"),
]


def read_fact_lines():
    import pandas as pd

    frame = pd.read_parquet(PARQUET)
    context = frame.iloc[0]["context"]
    lines = [l for l in context.split("\n") if re.match(r"^[0-9]+\. ", l)]
    return [re.sub(r"^[0-9]+\. ", "", l).strip() for l in lines]


def query_stem(question):
    """The exact prompt the reader gets at inference for FactConsolidation.

    Probes must be scored through this, not a bare "Question: ... Answer:". The
    dataset's instruction block is what puts the reader in answer format; a bare
    prompt scored 0/16 even against the full uncompacted pool, so a before/after
    comparison on bare probes would report 0 vs 0 and prove nothing. The template
    carries the trailing "Answer:" that marks where `n_prefix` ends.
    """
    return QUERY_TEMPLATE.format(question=question)


def first_match(text):
    """The first template that matches `text`, or None.

    First-match is required, not cosmetic. The templates overlap: for
    "The capital of Great Britain is London.", both

        ^The capital of (.+) is (.+)$      -> "What is the capital of Great Britain?"
        ^The (.+) of (.+) is (.+)$         -> "Who is the capital of Great Britain?"

    apply. Rendering every match produced nonsense questions like "Who is the
    capital of Great Britain?", which no memory can answer, so a probe set built
    that way scores 0 even uncompacted. The specific templates are listed first and
    the generic `^The (.+) of (.+) is (.+)$` is last for exactly this reason.
    """
    stripped = text.rstrip(".")
    for slot, (pattern, question, groups) in enumerate(TEMPLATES):
        match = re.match(pattern, stripped)
        if not match:
            continue
        parts = match.groups()
        try:
            subject = tuple(parts[int(g) - 1] for g in groups.split(","))
        except (IndexError, TypeError):
            continue
        return slot, subject, parts[-1], question
    return None


def probes_from_lines(lines, indices, limit):
    """Render probes that the *compaction window* is responsible for answering.

    Three constraints, each of which would otherwise produce a probe that cannot
    detect a bad compaction:

    1. Conflict resolution. FactConsolidation's rule is "the newer fact has the
       larger serial number", so a question's correct answer is the *latest*
       matching fact, not the line it was rendered from. The pool contains real
       contradictions ("Chanel was founded by Coco Chanel" vs "... by Andy
       Warhol"), so rendering a probe from an early line would score a correct
       reader as wrong.
    2. The latest matching fact must lie inside the window. If a newer fact on the
       same subject sits in the retained region, the answer survives any
       compaction and the probe cannot move between before and after.
    3. The anchor region is excluded, since it is never compacted either.
    """
    window = set(indices)
    # Latest matching line per (slot, subject), scanned in serial order.
    latest = {}
    for index, text in enumerate(lines):
        found = first_match(text)
        if found is None:
            continue
        slot, subject, answer, _question = found
        latest[(slot, subject)] = (index, answer)

    probes = []
    seen = set()
    for index in sorted(window):
        found = first_match(lines[index])
        if found is None:
            continue
        slot, subject, _answer, question = found
        owner, answer = latest.get((slot, subject), (None, None))
        if owner != index:
            continue
        try:
            filled = question % subject
        except TypeError:
            continue
        if filled in seen or not (1 <= len(answer.split()) <= 5):
            continue
        seen.add(filled)
        probes.append(
            Probe(
                question=filled,
                answers=[answer],
                split="train",
                prompt=query_stem(filled),
            )
        )
        if len(probes) >= limit:
            break
    return probes


# --- uncompacted reference ---------------------------------------------------

@torch.no_grad()
def score_uncompacted(manager, probes, max_probe_tokens):
    """Reader accuracy with no compaction at all: the upper bound for any operator."""
    device = manager.model.memory.device
    batches = _probe_batches(manager.tokenizer, probes, str(device), max_length=max_probe_tokens)
    pool = manager._memory.to(device).detach()
    hits = total = 0
    loss = 0.0
    with differentiable_memory(manager.model, pool):
        for input_ids, n_prefix, _ in batches:
            logits = manager.model(input_ids=input_ids, return_dict=True).logits
            span = logits[:, n_prefix - 1 : -1, :]
            labels = input_ids[:, n_prefix:]
            if labels.numel() == 0:
                continue
            loss += float(F.cross_entropy(span.reshape(-1, span.shape[-1]), labels.reshape(-1)))
            hits += int(bool((span.argmax(dim=-1) == labels).all(dim=-1).item()))
            total += 1
    return {"accuracy": hits / total if total else float("nan"), "gold_loss": loss / total if total else float("nan"), "n": total}


# --- window construction -----------------------------------------------------

def build_window(manager, lines, target_slots):
    """Write real fact chunks until the oldest `target_slots` are worth probing."""
    manager.initialize_anchor("\n".join(lines[:ANCHOR_LINES]), max_tokens=512)
    covered = []
    offset = ANCHOR_LINES
    cumulative = 0
    writes = 0
    while cumulative < target_slots and offset < len(lines):
        chunk = lines[offset : offset + LINES_PER_CHUNK]
        if not chunk:
            break
        record = manager.write("\n".join(chunk), max_tokens=512)
        if record.get("dropped"):
            offset += LINES_PER_CHUNK
            continue
        width = int(record.get("tokens", 0))
        writes += 1
        if width:
            covered.append((offset, min(offset + LINES_PER_CHUNK, len(lines)), cumulative, width))
            cumulative += width
        if record.get("evicted_blocks") or record.get("compacted"):
            raise SystemExit("the window build compacted or evicted; E6a needs a pristine append-only window")
        offset += LINES_PER_CHUNK
    print("[e6a] %d writes, used=%d slots, window=%d slots" % (writes, manager.used, cumulative), flush=True)
    return covered


def probe_indices(covered, window_slots):
    """Fact lines that lie entirely inside the compaction window."""
    indices = []
    for lo, hi, before, width in covered:
        if before + width <= window_slots:
            indices.extend(range(lo, hi))
    return indices


# --- main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps-a", type=int, default=16)
    parser.add_argument("--steps-b", type=int, default=24)
    parser.add_argument("--probes", type=int, default=20)
    parser.add_argument("--max-probe-tokens", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma-max", type=float, default=0.05)
    parser.add_argument("--out", default="/tmp/e6a_report.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import yaml
    from transformers import AutoTokenizer

    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM

    raw = yaml.safe_load(open(CONFIG))
    config = dict(raw["prlmc"])
    config["device"] = "cuda:0"
    config["dtype"] = "float16"
    config["stats_dir"] = "./outputs/prlmc-e6a"
    config["carrier_gamma_max"] = args.gamma_max
    cfg = PRLMCConfig.from_dict(config)

    model = MemoryLLM.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = model.to("cuda:0").eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    manager = ProtectedResidualMemory(model, tokenizer, cfg)
    lines = read_fact_lines()
    window_slots = WINDOW_BLOCKS * manager.block
    covered = build_window(manager, lines, window_slots)

    indices = probe_indices(covered, window_slots)
    probes = probes_from_lines(lines, indices, args.probes)
    print("[e6a] %d probes from %d in-window fact lines" % (len(probes), len(indices)), flush=True)
    for probe in probes:
        print("        %-60s -> %s" % (probe.question, probe.answers[0]))
    if len(probes) < 8:
        raise SystemExit("not enough in-window probes; widen the window or the template set")

    compactor = CarrierResidualCompactor(
        num_layers=manager.num_layers,
        hidden=manager.hidden,
        block=manager.block,
        rank=int(cfg.carrier_rank),
        gamma_max=float(cfg.carrier_gamma_max),
    ).to(device="cuda:0", dtype=torch.float32)

    # Real-input version of the CPU structural test: a zero-initialised residual
    # must reproduce select_blocks bitwise, or the "before" number is not the
    # baseline's number. The weights come from the manager's own
    # `_window_importance`, which is exactly what `compact_with_report` feeds the
    # operator, so this compares the two paths on identical inputs.
    start = manager.total - manager.used
    source = manager._memory[:, start : start + window_slots, :].to("cuda:0")
    weights = manager._window_importance(start, window_slots)
    reference = select_block_compact(
        source, TARGET_BLOCKS * manager.block, block=manager.block,
        weights=weights, goal_direction=manager.goal_direction,
        select_weights=manager.select_weights,
    )
    # Under no_grad deliberately. At init the residual is identically zero, so the
    # comparison needs no graph -- and building one here leaked ~3.8 GB that stayed
    # resident into stage A: this call runs *before* the trainer, so its activations
    # were never freed, and the compactor's forward then had no headroom left.
    with torch.no_grad():
        got = compactor(
            source, weights=weights, target=TARGET_BLOCKS * manager.block, block=manager.block,
            goal_direction=manager.goal_direction, select_weights=manager.select_weights,
        )
    identity = bool(torch.equal(reference, got))
    print("[e6a] carrier_residual == select_blocks at init: %s (max |d| = %.3e)"
          % (identity, float((reference - got).abs().max())), flush=True)
    del got, reference
    torch.cuda.empty_cache()

    report = {
        "window_slots": window_slots,
        "target_slots": TARGET_BLOCKS * manager.block,
        "used_slots": manager.used,
        "n_probes": len(probes),
        "gamma_max": float(cfg.carrier_gamma_max),
        "identity_at_init": identity,
        "probes": [{"q": p.question, "a": p.answers[0]} for p in probes],
    }

    report["uncompacted"] = score_uncompacted(manager, probes, args.max_probe_tokens)
    report["before"] = score_candidate(
        manager, compactor, probes, WINDOW_BLOCKS, TARGET_BLOCKS, max_probe_tokens=args.max_probe_tokens
    )
    print("[e6a] uncompacted %s" % report["uncompacted"], flush=True)
    print("[e6a] before      %s" % report["before"], flush=True)

    if args.dry_run:
        json.dump(report, open(args.out, "w"), indent=2)
        print("[e6a] dry run written to %s" % args.out)
        return

    for stage, steps in (("A", args.steps_a), ("B", args.steps_b)):
        if steps <= 0:
            continue
        print("[e6a] training stage %s for %d steps" % (stage, steps), flush=True)
        trained = train_carrier_compactor(
            manager, probes, steps=steps, lr=args.lr, stage=stage,
            window_blocks=WINDOW_BLOCKS, target_blocks=TARGET_BLOCKS,
            max_probe_tokens=args.max_probe_tokens, compactor=compactor,
        )
        after = score_candidate(
            manager, compactor, probes, WINDOW_BLOCKS, TARGET_BLOCKS, max_probe_tokens=args.max_probe_tokens
        )
        report["stage_" + stage] = {"train": trained, "after": after}
        print("[e6a] stage %s after %s" % (stage, after), flush=True)

    final = report.get("stage_B", report.get("stage_A"))["after"]
    report["acceptance"] = {
        "improved_over_select": bool(final["accuracy"] > report["before"]["accuracy"]),
        "before_accuracy": report["before"]["accuracy"],
        "after_accuracy": final["accuracy"],
        "uncompacted_accuracy": report["uncompacted"]["accuracy"],
    }
    json.dump(report, open(args.out, "w"), indent=2, default=str)
    print("[e6a] acceptance: %s" % report["acceptance"], flush=True)
    print("[e6a] written to %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
