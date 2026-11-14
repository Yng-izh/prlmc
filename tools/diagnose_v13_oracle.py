#!/usr/bin/env python3
"""Diagnostic: how much of V13's 91 wrong rows is retrieval, and how much is the reader.

Two questions, both answered without training anything:

1. RETRIEVAL. Of the rows V13 answers wrong, how many have their supporting-fact blocks
   among the 10 DROPPED blocks and no evidence slot surviving into the window at all?
   Those rows are lost by the slot selection, not by the calibration.

2. CEILING. How many of the 158 rows does the frozen reader fail even when it is handed
   ONLY the golden blocks -- the raw blocks that contain the supporting facts, no
   compression, no calibration? That number is the hard ceiling this base model has on
   this row set, and no amount of memory engineering can cross it.

Both are measured on the same window V13 reads (`--v11-evict`'s geometry), so the pieces
are comparable: a row counted in (1) is a selection miss; a row counted in (2) is a
reader miss; the remainder is where a better calibration could still win.

Two additions (V17, evidence chain):

3. CANDIDATE SENSITIVITY (`--candidate-replacement {off,random,shuffle}`). On the rows
   "Ours right, Selection wrong" -- the rows the calibration rescued -- the graft slots'
   CONTENT is replaced and the question is asked again: `random` swaps every graft slot
   for a core slot of the same document, `shuffle` permutes the graft slots among
   themselves. Same window shape, same 2,304 slots, same module, same question; only
   whose content the graft slots hold changes. If the rescue survives, the gain never
   came from that content. The seed is derived from `--seed` and the row id, so a rerun
   reproduces the draw exactly. Written into the report as `candidate_sensitivity`.

4. ORACLE TABLE (`--export-oracle-table`). One row per arm -- Selection, AnchorOnly,
   NativeGraft, Full, Ours -- plus the reader ceiling this run measures (the golden-block
   readout), with the share of the recoverable space each arm captures, defined as
   `(arm - Selection) / (Oracle - Selection)` over the same rows. Every arm count comes
   from a finished report named on the command line; where a name is absent the cell says
   MISSING rather than borrowing a number from another arm.

    python tools/diagnose_v13_oracle.py \
        --data-dir ... --task hotpotqa --supporting-facts-glob '...' \
        --model-path /dev/shm/s1d_reader_fp16 \
        --checkpoint runs/natural_merge_v9/full_v10.pt \
        --anchor runs/anchor_residual/anchor_only_rank64.pt \
        --v11-evict runs/cross_block/cross_lr.pt \
        --v13-report reports/cross_block/eval_v16.json \
        --cuda-device 0 --out reports/cross_block/diagnose_v13_oracle.json

WHAT RUNS WHERE: the harness is not in this repository
------------------------------------------------------
This tool measures through `eval_cross_block_calibration`, `eval_graft_recovery`,
`eval_source_compaction`, `native_eval`, `io_utils` and `runtime`. Those files exist ONLY
in the server checkout; this repository ships the entrypoint and none of what it measures
through (nor `v9_geometry_50block`, which `eval_learned_fusion` imports). Consequences:

  * `--help` and `python3 -m py_compile` work anywhere: every harness import is deferred
    to the function that uses it, and a module that cannot be found is recorded in
    `MISSING` instead of aborting the import.
  * EVERY measurement mode needs the harness -- the golden-block ceiling, the slot
    survival count, the per-row numbers `--export-oracle-table` copies out of a report.
    There is no reduced run that produces them without it, so `main()` calls
    `require_harness()` and raises `MissingHarnessDependency` naming the absent files,
    instead of degrading silently or skipping a measurement quietly.
  * Only one import is flag-dependent: `eval_replace_reorder` is needed by
    `--candidate-replacement random|shuffle` and is NOT loaded for the `off` default, so
    that mode carries one dependency fewer. Writing the markdown table is plain Python
    and adds none.
"""
import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SCHEMA = "v13_oracle_diagnosis/1"
REPLACEMENT_MODES = ("off", "random", "shuffle")
_TOOLS = Path(__file__).resolve().parent     # where the harness would have to sit

# --------------------------------------------------------------------------------------
# The harness: what this file needs, and what it does when the harness is not here.
#
# (module, short name, what this file takes from it, needed by every run?). Short names
# are used at the call sites; the module objects are resolved by `_dep` on first use.
# --------------------------------------------------------------------------------------
HARNESS_DEPS = (
    ("eval_cross_block_calibration", "ev",
     "calibrate / load_v11 / joint_for / the arm names (V11_EVICT, FULL, SELECTION)", True),
    ("eval_graft_recovery", "gr",
     "BLOCK_TOKENS, BUDGET_BLOCKS, WINDOW_BLOCKS, TARGET_BLOCKS, load_supporting_facts, "
     "prepare_rows, select_carriers, readout, restore, snapshot", True),
    ("eval_source_compaction", "es", "write_blocks", True),
    ("native_eval", "nv", "seed_everything", True),
    ("io_utils", "io", "atomic_json, sha", True),
    ("runtime", "rt", "load_runtime, reset", True),
    # Only the replacement arms install a window back into the manager; the `off`
    # default asks the reader about the calibrated window in place and never needs it.
    ("eval_replace_reorder", "er", "install_output", False),
)


def _probe(name):
    """Try to import one harness module; never raises, so the module import can finish."""
    try:
        return importlib.import_module(name), None
    except Exception as exc:                  # ImportError, or a harness that is broken
        return None, dict(module=name, error=f"{type(exc).__name__}: {exc}",
                          cause=getattr(exc, "name", None))


_PROBED = {alias: _probe(name) for name, alias, _take, _core in HARNESS_DEPS}
_NAME_OF = {alias: name for name, alias, _take, _core in HARNESS_DEPS}
_TAKES_OF = {alias: take for name, alias, take, _core in HARNESS_DEPS}
_ALWAYS = tuple(alias for _name, alias, _take, core in HARNESS_DEPS if core)
#: alias -> why it could not be imported. Empty in the server checkout.
MISSING = {alias: info for alias, (module, info) in _PROBED.items() if module is None}

_STOP_HEAD = """\
this checkout cannot run the v13-oracle diagnosis: it needs evaluation-harness modules
that exist only in the server checkout.

absent:
{detail}

why this is a stop and not a degraded mode:
  every number this tool reports comes out of that harness -- the memory manager, the
  reader readout, the supporting-fact loader, the atomic json writer. No reduced run
  measures the golden-block ceiling, the slot survival or the candidate replacement
  without it, and a stubbed harness would print numbers that look like evidence and are
  not. So the run ends here, before a model is loaded or a report file is touched.

what still works in this checkout:
  python tools/diagnose_v13_oracle.py --help
  python3 -m py_compile tools/diagnose_v13_oracle.py

what to do:
  copy the files above into tools/ of this tree, or run this tool on the server checkout,
  where they sit next to it. `--candidate-replacement off` (the default) is the cheapest
  mode: it is the only one that does not also need tools/eval_replace_reorder.py.
"""


class MissingHarnessDependency(RuntimeError):
    """The evaluation harness this tool measures through is not in this checkout."""


def _detail(alias):
    """One entry of the 'absent:' list: the file, what is taken from it, why it failed."""
    name, info = _NAME_OF[alias], MISSING[alias]
    state = ("file is here but does not import"
             if (_TOOLS / f"{name}.py").exists() else "file is not in this checkout")
    return (f"  tools/{name}.py  [{state}]\n"
            f"        needs: {_TAKES_OF[alias]}\n"
            f"        {info['error']}")


def _hard_stop(aliases):
    return _STOP_HEAD.format(detail="\n".join(_detail(alias) for alias in aliases))


def _dep(alias):
    """The harness module for `alias`, or a complete error naming what is missing."""
    module, _info = _PROBED[alias]
    if module is None:
        raise MissingHarnessDependency(_hard_stop([alias]))
    return module


def _ev():
    """`eval_cross_block_calibration`: the V13/V17 arms and the window calibration."""
    return _dep("ev")


def _gr():
    """`eval_graft_recovery`: window geometry, row preparation, reader readout."""
    return _dep("gr")


def _es():
    """`eval_source_compaction`: `write_blocks`, the only writer of the window stream."""
    return _dep("es")


def _nv():
    """`native_eval`: `seed_everything`."""
    return _dep("nv")


def _io():
    """`io_utils`: `atomic_json`, `sha`."""
    return _dep("io")


def _rt():
    """`runtime`: `load_runtime`, `reset`."""
    return _dep("rt")


def _er():
    """`eval_replace_reorder`: `install_output`, the replacement arms only."""
    return _dep("er")


def block_tokens():
    """`gr.BLOCK_TOKENS`: write-stream tokens per block.

    Read from the harness rather than copied here: a local copy would silently disagree
    with the geometry every other tool in the chain uses, and every chunk boundary in this
    file would be off by the same amount.
    """
    return int(_gr().BLOCK_TOKENS)


def require_harness(args):
    """Refuse the run, once and completely, if a module this run needs is absent.

    Called from `main()` after argument validation, i.e. after `--help` has already been
    served and before a model, a checkpoint or an output file is opened.
    """
    needed = list(_ALWAYS)
    if args.candidate_replacement != "off":
        needed.append("er")
    absent = [alias for alias in needed if alias in MISSING]
    if absent:
        raise MissingHarnessDependency(_hard_stop(absent))


def table_arms():
    """The arms the exported table lines up, in the order the evidence chain reads them.

    The second element lists the names finished reports use for the same arm; the FIRST
    name that actually appears in the report wins and is printed next to the number, so no
    cell can come from an arm the file does not name. Built at run time because `Ours`'s
    primary name is `ev.V11_EVICT` and `ev` is resolved lazily.
    """
    return (
        ("Selection", ("Selection",)),
        ("AnchorOnly", ("AnchorOnly",)),
        # The native graft arm (`Graft` in this round's reports: V10's operator, 1,944 +
        # 360). `V11Evict` is the plain full-pool calibration arm, not this one.
        ("NativeGraft", ("NativeGraft", "Graft")),
        ("Full", ("Full",)),
        ("Ours", (_ev().V11_EVICT, "V11Evict", "V17", "Trained")),
    )


ORACLE_ROW = "Oracle"
TABLE_ROLE_ORDER = ("Selection", "AnchorOnly", "NativeGraft", "Full", "Ours", ORACLE_ROW)


def chunks_of(prepared, blocks, limit):
    """The prepared row's token chunks, restricted to `blocks`, in block order."""
    BLOCK_TOKENS = block_tokens()
    chunks = [prepared["write_ids"][index:index + BLOCK_TOKENS]
              for index in range(0, len(prepared["write_ids"]), BLOCK_TOKENS)]
    out = []
    for block in sorted(set(blocks)):
        if 0 <= int(block) < min(limit, len(chunks)):
            out.append(chunks[int(block)])
    return out


def blank_chunk(tokenizer, length):
    """A chunk of `length` neutral tokens -- same shape, no content."""
    ids = tokenizer(" .", add_special_tokens=False)["input_ids"]
    token = int(ids[-1])
    return [token] * int(length)


def golden_answer(model, tokenizer, manager, merge, initial, prepared, blocks, args):
    """Greedy answer with the golden evidence ALONE inside the normal 19-block pool.

    The pool keeps its usual shape and occupancy: all 19 blocks are written, the golden
    blocks verbatim, every other block filled with the same number of neutral tokens.
    Writing only the two golden blocks instead (the first version of this tool) changes
    the pool's size and the reader's regime, and it scored 18/158 -- below Full's 47 and
    below V13's 67, i.e. it was measuring the regime change, not the reader's ability.
    """
    gr, es, reset = _gr(), _es(), _rt().reset        # the harness, resolved on first use
    BLOCK_TOKENS = block_tokens()
    chunks = [prepared["write_ids"][index:index + BLOCK_TOKENS]
              for index in range(0, len(prepared["write_ids"]), BLOCK_TOKENS)]
    chunks = chunks[:args.window_blocks]
    # Positive control: with `--control-full` every block counts as golden, so no block
    # is blanked and this function must reproduce the Full arm row for row. If it does
    # not, the harness -- not the ceiling -- is what the 18/158 measured.
    wanted = (set(range(len(chunks))) if args.control_full
              else set(int(block) for block in blocks))
    if not wanted or not wanted <= set(range(len(chunks))):
        return dict(available=False, correct=False, qa_f1=0.0, prediction="",
                    blocks=[], slots=0, blanked=0)
    stream, blanked = [], 0
    for index, chunk in enumerate(chunks):
        if index in wanted:
            stream.append(chunk)
        else:
            stream.append(blank_chunk(tokenizer, len(chunk)))
            blanked += 1
    with torch.no_grad():
        reset(manager, initial)
        es.write_blocks(manager, merge, stream)
        answer = gr.readout(model, tokenizer, prepared["query_ids"], prepared["answers"],
                            args.f1_threshold, lambda: None)
    return dict(available=True, correct=bool(answer["qa_f1"] >= args.f1_threshold),
                qa_f1=float(answer["qa_f1"]), prediction=answer["prediction"],
                blocks=sorted(wanted), blanked_blocks=blanked,
                slots=int(manager.used))


def evidence_survival(meta, blocks):
    """How many slots of the evidence blocks survived, split by core / graft.

    `meta["source_block"]` carries the true block id, so this asks the direct question:
    is any slot OF THAT BLOCK in the window? (Deriving the block from `source_position`
    would be wrong -- that index addresses the carrier pool, not the 19-block window.)
    """
    source = meta["source_block"]
    mask = torch.zeros_like(source, dtype=torch.bool)
    for block_id in set(int(block) for block in blocks):
        mask |= source == block_id
    if not bool(mask.any()):
        return dict(total=0, core=0, graft=0)
    return dict(total=int(mask.sum()),
                core=int((mask & (meta["slot_type"] == 0)).sum()),
                graft=int((mask & (meta["slot_type"] == 1)).sum()))


def row_seed(seed, row_id):
    """A per-row seed derived from `--seed` and the row id.

    Derived rather than drawn from one running generator, so the same row gets the same
    replacement whether it is the 1st or the 158th row of the run, and whether or not the
    other rows were processed at all.
    """
    digest = hashlib.sha256(f"{int(seed)}:{row_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def replacement_window(window, meta, mode, seed):
    """The same window with the graft slots' CONTENT replaced, and nothing else moved.

    `random`  every graft slot receives a core slot of the same window -- one document's
              19 blocks, so the replacement is in-distribution content from the wrong
              place; `shuffle` the graft slots are permuted among themselves. Both keep
              the slot count, the slot order and the core half untouched, so a changed
              answer can only come from whose content the graft slots hold.
    """
    if mode not in REPLACEMENT_MODES or mode == "off":
        raise ValueError(f"replacement_window was asked for mode {mode!r}")
    graft = torch.nonzero(meta["slot_type"] == 1, as_tuple=False).flatten()
    core = torch.nonzero(meta["slot_type"] == 0, as_tuple=False).flatten()
    if int(graft.numel()) == 0:
        raise ValueError("the window holds no graft slot: there is nothing to replace")
    generator = torch.Generator().manual_seed(int(seed))
    replaced = window.clone()
    if mode == "random":
        if int(core.numel()) == 0:
            raise ValueError("the window holds no core slot: the same-document control "
                             "has nothing to draw from")
        draw = torch.randint(0, int(core.numel()), (int(graft.numel()),),
                             generator=generator)
        replaced[:, graft] = window.index_select(1, core[draw].to(window.device))
        origin = "core slots of the same window (one document, 19 blocks)"
    else:
        order = torch.randperm(int(graft.numel()), generator=generator)
        if bool(torch.equal(order, torch.arange(int(graft.numel())))):
            order = torch.roll(order, 1)
        replaced[:, graft] = window.index_select(1, graft[order].to(window.device))
        origin = "the window's graft slots, permuted among themselves"
    moved = float((replaced[:, graft] != window[:, graft]).any(dim=-1).float().mean())
    return replaced, dict(mode=mode, seed=int(seed), origin=origin,
                          replaced_slots=int(graft.numel()),
                          core_slots_on_offer=int(core.numel()),
                          moved_fraction=moved)


def candidate_analysis(model, tokenizer, manager, state, source, picked, window, meta,
                       module, prepared_row, reference, args):
    """Does the rescue survive when the graft slots hold other content?

    Both readouts are produced HERE -- the untouched calibrated window and the same window
    with the graft content replaced -- so a flip cannot be a difference between harnesses.
    Nothing is retrained: replace content, recalibrate with the same module, ask the same
    question. `state` is the pre-compression snapshot the caller took, i.e. the pool the
    compressed window is installed over. `agrees_with_report` says whether this harness
    reproduced the report's Ours answer on this row; when it does not, that row's flip is
    not evidence either way.
    """
    ev, gr, er = _ev(), _gr(), _er()                 # the harness, resolved on first use
    with torch.no_grad():
        calibrated, _ = ev.calibrate(window, meta, module, source=source)

        def ask(window_arm):
            def prepare():
                gr.restore(model, manager, state)
                er.install_output(manager, window_arm, picked, args.window_blocks)
            return gr.readout(model, tokenizer, prepared_row["query_ids"],
                              prepared_row["answers"], args.f1_threshold, prepare)

        ours = ask(calibrated)
        replaced, replacement = replacement_window(window, meta,
                                                   args.candidate_replacement,
                                                   row_seed(args.seed, prepared_row["id"]))
        replaced_calibrated, replaced_info = ev.calibrate(replaced, meta, module,
                                                          source=source)
        swapped = ask(replaced_calibrated)
        gr.restore(model, manager, state)
    ours_correct = bool(ours["qa_f1"] >= args.f1_threshold)
    swapped_correct = bool(swapped["qa_f1"] >= args.f1_threshold)
    return dict(
        mode=args.candidate_replacement,
        replacement=replacement,
        ours=dict(correct=ours_correct, qa_f1=float(ours["qa_f1"]),
                  prediction=ours["prediction"]),
        replaced=dict(correct=swapped_correct, qa_f1=float(swapped["qa_f1"]),
                      prediction=swapped["prediction"]),
        # `flipped`: the rescue is gone -- Ours answered it and the replaced window does
        # not. `changed_prediction`: the answer string differs at all, which also counts
        # a wrong answer that became a different wrong answer.
        flipped=bool(ours_correct and not swapped_correct),
        changed_prediction=bool(ours["prediction"] != swapped["prediction"]),
        agrees_with_report=bool(ours_correct == bool(reference["accuracy_correct"])),
        replaced_calibration=dict(
            core_relative_delta=replaced_info["core_relative_delta"],
            graft_relative_delta=replaced_info["graft_relative_delta"],
            relative_delta=replaced_info["relative_delta"]))


def _table_cell(count, rows):
    """`correct/rows` as a percentage, or MISSING -- never a number from another row set."""
    if count is None or not rows:
        return "MISSING"
    return f"{100.0 * count / rows:.1f}%"


def oracle_table(report, args):
    """Selection / AnchorOnly / NativeGraft / Full / Ours, against the reader's ceiling.

    The ceiling is this run's own golden readout; every other count is read, per row, from
    the finished report named by `--v13-report`. The row set is the rows where BOTH exist,
    so the denominators are the same; an arm whose report name is absent prints MISSING
    instead of a count taken from another arm. `captured` is
    `(arm - Selection) / (Oracle - Selection)`: the share of the space that is recoverable
    above the capacity baseline which this arm actually takes.
    """
    graded = [sample for sample in report["samples"]
              if sample.get("v13") and (sample.get("golden") or {}).get("available")]
    rows = {role: dict(arm=None, correct=None, n=0, reason=None) for role in TABLE_ROLE_ORDER}
    resolved = {}
    for role, aliases in table_arms():
        name = None
        for alias in aliases:
            if any(alias in (sample["v13"].get("arm_correct") or {}) for sample in graded):
                name = alias
                break
        if name is None:
            rows[role]["reason"] = ("the report names none of " + ", ".join(aliases))
            continue
        values = [sample["v13"]["arm_correct"][name] for sample in graded
                  if name in (sample["v13"].get("arm_correct") or {})]
        rows[role].update(arm=name, correct=int(sum(values)), n=len(values))
        resolved[role] = name
    rows[ORACLE_ROW].update(
        arm="golden", correct=int(sum(1 for sample in graded if sample["golden"]["correct"])),
        n=len(graded))

    base_rows = len(graded)
    selection = rows["Selection"]["correct"]
    oracle = rows[ORACLE_ROW]["correct"]
    recoverable = (oracle - selection
                   if (selection is not None and oracle is not None) else None)
    lines = [
        "| Arm | Correct | Rows | Accuracy | Δ vs Selection | Captured of recoverable "
        "space | Arm name in the report |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for role in TABLE_ROLE_ORDER:
        row = rows[role]
        if row["correct"] is None:
            lines.append(f"| {role} | MISSING | {base_rows} | MISSING | MISSING | MISSING "
                         f"| MISSING -- {row['reason']} |")
            continue
        delta = ("MISSING" if selection is None else
                 ("+" if row["correct"] - selection > 0 else "")
                 + f"{row['correct'] - selection}")
        captured = ("MISSING" if (recoverable is None or recoverable <= 0) else
                    f"{100.0 * (row['correct'] - selection) / recoverable:.1f}%")
        name = row["arm"] if row["arm"] else "golden (this run)"
        if role == ORACLE_ROW:
            name = "golden (this run)"
        lines.append(f"| {role} | {row['correct']} | {row['n']} | "
                     f"{_table_cell(row['correct'], row['n'])} | {delta} | {captured} | "
                     f"{name} |")
    lines += [
        "",
        f"Rows: {base_rows} (rows where this run measured the golden ceiling AND the "
        f"report has an Ours record).",
        "Recoverable space = Oracle − Selection = "
        + ("MISSING" if recoverable is None else str(recoverable))
        + " questions; captured = (arm − Selection) / (Oracle − Selection).",
        "Oracle = the frozen reader given ONLY the supporting-fact blocks, inside a "
        "normal 19-block pool (the other blocks filled with neutral tokens); it is the "
        "ceiling no memory mechanism can cross.",
        "Selection / AnchorOnly / NativeGraft / Full / Ours are read row for row from "
        f"{args.v13_report or 'MISSING (--v13-report not given)'}; "
        "a cell says MISSING when its arm is not named in that file.",
    ]
    return dict(base_rows=base_rows, rows=rows, resolved=resolved,
                recoverable=recoverable, markdown="\n".join(lines))


def write_oracle_table(table, args, out_path):
    """Write the markdown next to the report (nothing else is created)."""
    target = Path(args.oracle_table_out or Path(str(out_path)).with_suffix("").as_posix()
                  + "_oracle_table.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(table["markdown"] + "\n", encoding="utf-8")
    return target


def evaluate(args):
    # The harness is resolved HERE, not at import time (see HARNESS_DEPS): the locals keep
    # every call site below unchanged, so the measurement code reads exactly as it did
    # when these modules were imported at the top of the file.
    ev, gr, es, nv, io, rt = _ev(), _gr(), _es(), _nv(), _io(), _rt()
    reset, load_runtime = rt.reset, rt.load_runtime
    atomic_json, sha = io.atomic_json, io.sha
    BLOCK_TOKENS = block_tokens()
    if args.window_blocks is None:
        args.window_blocks = int(gr.WINDOW_BLOCKS)
    if args.target_blocks is None:
        args.target_blocks = int(gr.TARGET_BLOCKS)
    from methods.prlmc.reader_utility import ReaderScorer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    facts = gr.load_supporting_facts(args.supporting_facts_glob)
    nv.seed_everything(args.seed)
    checkpoint_training = torch.load(args.checkpoint, map_location="cpu").get("training")
    model, tokenizer, manager, merge, initial, cfg, saved = load_runtime(
        args.model_path, checkpoint=args.checkpoint, train=False, architecture=None,
        budget_blocks=gr.BUDGET_BLOCKS, strict_code_hash=args.strict_code_hash,
        training=checkpoint_training)
    merge.operator_mode = "native"
    module, geometry, config, _ = ev.load_v11(args.v11_evict, manager.hidden)
    module = module.to(model.memory.device).float()
    scorer = ReaderScorer(model, tuple(int(x) for x in geometry["utility_layers"]))
    joint = ev.joint_for(geometry, scorer, {}, manager.hidden)
    rows, prepared, excluded = gr.prepare_rows(args, tokenizer, facts)
    if args.max_samples:
        prepared = prepared[:args.max_samples]
    original = {}
    if args.v13_report:
        payload = json.load(open(args.v13_report))
        # Every arm the exported table can name, so `oracle_table` reads one dict per row
        # instead of re-parsing the report (and cannot silently mix two row sets).
        wanted = {alias for _, aliases in table_arms() for alias in aliases}
        for sample in payload["samples"]:
            arms = sample.get("arms", {})
            treatment = arms.get(ev.V11_EVICT)
            full = arms.get(ev.FULL)
            selection = arms.get(ev.SELECTION)
            if treatment:
                original[str(sample["id"])] = dict(
                    accuracy_correct=bool(treatment["qa_f1"] >= args.f1_threshold),
                    selection_correct=(bool(selection["qa_f1"] >= args.f1_threshold)
                                       if selection else None),
                    qa_f1=float(treatment["qa_f1"]), prediction=treatment["prediction"],
                    full_correct=(bool(full["qa_f1"] >= args.f1_threshold)
                                  if full else None),
                    arm_correct={name: bool(entry["qa_f1"] >= args.f1_threshold)
                                 for name, entry in arms.items() if name in wanted})

    report = dict(schema=SCHEMA, status="running", counts=dict(
        rows=len(rows), eligible=len(prepared), excluded=len(excluded)),
        identity=dict(model_path=args.model_path, checkpoint=args.checkpoint,
                      checkpoint_sha256=sha(args.checkpoint),
                      v11_evict=args.v11_evict, v11_evict_sha256=sha(args.v11_evict),
                      v13_report=args.v13_report,
                      geometry_core=geometry["core_per_block"],
                      geometry_graft=geometry["graft_per_block"],
                      entrypoint_sha256=sha(__file__)), samples=[])
    output = Path(args.out)
    if output.exists():
        raise ValueError(f"{output} exists; delete it first")
    atomic_json(output, report)
    started = time.monotonic()
    for position, prepared_row in enumerate(prepared):
        sample_started = time.monotonic()
        nv.seed_everything(args.seed)
        with torch.no_grad():
            reset(manager, initial)
            es.write_blocks(manager, merge,
                            [prepared_row["write_ids"][index:index + BLOCK_TOKENS]
                             for index in range(0, len(prepared_row["write_ids"]),
                                                BLOCK_TOKENS)][:args.window_blocks])
            source, ages, picked = gr.select_carriers(manager, args.window_blocks,
                                                      args.target_blocks)
            window, meta, telemetry = joint.build(source, picked, manager.block)
            # The snapshot has to be taken HERE: it is the pre-compression pool the
            # compressed windows are installed over, and `golden_answer` rewrites the
            # memory from scratch further down.
            state = gr.snapshot(model, manager)
        dropped = [int(block) for block in telemetry["dropped_blocks"]]
        blocks = [int(block) for block in (prepared_row.get("evidence", {}).get("blocks")
                                           or [])]
        inside = [block for block in blocks if block < args.window_blocks]
        outside = [block for block in blocks if block >= args.window_blocks]
        survival = evidence_survival(meta, inside)
        golden = golden_answer(model, tokenizer, manager, merge, initial, prepared_row,
                               inside, args)
        reference = original.get(str(prepared_row["id"]))
        # The rows the calibration rescued: the report says Ours right and Selection
        # wrong. Only those rows can show a flip; on any other row a changed answer is
        # just a changed answer.
        sensitivity = None
        rescued = bool(reference and reference["accuracy_correct"] is True
                       and reference.get("selection_correct") is False)
        if args.candidate_replacement != "off" and rescued:
            sensitivity = candidate_analysis(model, tokenizer, manager, state, source,
                                             picked, window, meta, module, prepared_row,
                                             reference, args)
        record = dict(
            id=prepared_row["id"], evidence_blocks=blocks,
            evidence_blocks_outside_window=outside,
            dropped_blocks=dropped,
            evidence_in_dropped=bool(inside) and set(inside) <= set(dropped),
            evidence_slots=survival,
            evidence_lost=bool(inside) and survival["total"] == 0,
            golden=golden,
            v13=reference,
            rescued_by_ours=rescued,
            candidate_sensitivity=sensitivity,
            seconds=time.monotonic() - sample_started)
        report["samples"].append(record)
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output, report)
        print(json.dumps(dict(sample=position + 1, total=len(prepared),
                              id=prepared_row["id"],
                              golden=golden["correct"],
                              slots=survival["total"],
                              in_dropped=record["evidence_in_dropped"],
                              rescued=rescued,
                              flip=(None if sensitivity is None
                                    else bool(sensitivity["flipped"])),
                              seconds=round(record["seconds"], 1))), flush=True)

    graded = [sample for sample in report["samples"] if sample["v13"]]
    wrong = [sample for sample in graded if not sample["v13"]["accuracy_correct"]]
    right = [sample for sample in graded if sample["v13"]["accuracy_correct"]]
    golden_fail = [sample for sample in report["samples"] if not sample["golden"]["correct"]]
    selection_loss = [sample for sample in wrong if sample["evidence_lost"]]
    both = [sample for sample in wrong
            if sample["evidence_lost"] and not sample["golden"]["correct"]]
    report["summary"] = dict(
        n=len(report["samples"]),
        v13_rows=len(graded), v13_correct=len(right), v13_wrong=len(wrong),
        v13_accuracy=(round(100.0 * len(right) / max(len(graded), 1), 2)),
        golden_ceiling_correct=len(report["samples"]) - len(golden_fail),
        golden_ceiling_accuracy=round(
            100.0 * (len(report["samples"]) - len(golden_fail))
            / max(len(report["samples"]), 1), 2),
        reader_fails_on_golden=len(golden_fail),
        wrong_and_evidence_in_dropped=len([s for s in wrong
                                           if s["evidence_in_dropped"]]),
        wrong_and_evidence_lost=len(selection_loss),
        wrong_and_reader_fails_on_golden=len([s for s in wrong
                                             if not s["golden"]["correct"]]),
        wrong_lost_and_reader_fails=len(both),
        wrong_explained=len([s for s in wrong
                             if s["evidence_lost"] or not s["golden"]["correct"]]),
        golden_over_full=len([s for s in report["samples"]
                              if s["golden"]["correct"]
                              and s.get("v13") and s["v13"].get("full_correct") is False]),
        full_correct_rows=len([s for s in report["samples"]
                               if s.get("v13") and s["v13"].get("full_correct")]),
        blanked_blocks=[s["golden"].get("blanked_blocks") for s in report["samples"][:1]],
        definition=("`golden_ceiling_accuracy` is the reader's ceiling with the golden "
                    "evidence alone in a normal 19-block pool (non-golden blocks filled "
                    "with the same number of neutral tokens). `full_correct_rows` is the "
                    "same reader given all 19 blocks raw, read from the finished report. "
                    "`wrong_and_evidence_lost`: V13 answers wrong AND no supporting-fact "
                    "slot survived into the 2,304-slot window -- lost by selection. "
                    "`wrong_and_reader_fails_on_golden`: V13 answers wrong AND the frozen "
                    "reader also fails when given only the golden blocks -- lost by the "
                    "reader itself. `golden_ceiling_accuracy` is that reader ceiling over "
                    "all rows."),
        per_row=[dict(id=sample["id"], v13=bool(sample["v13"]["accuracy_correct"]),
                      golden=bool(sample["golden"]["correct"]),
                      evidence_lost=bool(sample["evidence_lost"]),
                      in_dropped=bool(sample["evidence_in_dropped"]))
                 for sample in report["samples"]])
    # Candidate-content sensitivity: the rescues that survive the graft content being
    # replaced were never carried by that content.
    scope = [sample for sample in report["samples"]
             if sample["v13"] and sample["v13"]["accuracy_correct"] is True
             and sample["v13"].get("selection_correct") is False]
    attempted = [sample for sample in scope if sample.get("candidate_sensitivity")]
    flipped = [sample for sample in attempted if sample["candidate_sensitivity"]["flipped"]]
    changed = [sample for sample in attempted
               if sample["candidate_sensitivity"]["changed_prediction"]]
    agreed = [sample for sample in attempted
              if sample["candidate_sensitivity"]["agrees_with_report"]]
    report["candidate_sensitivity"] = dict(
        mode=args.candidate_replacement,
        performed=bool(args.candidate_replacement != "off"),
        rescue_definition=("a rescued row is one where the finished report says Ours ("
                           f"{ev.V11_EVICT}) is correct and Selection is wrong"),
        rescued_rows_available=len(scope), rescued_rows=len(attempted),
        flipped_rows=len(flipped),
        flip_rate=(round(len(flipped) / len(attempted), 4) if attempted else None),
        changed_rows=len(changed),
        change_rate=(round(len(changed) / len(attempted), 4) if attempted else None),
        ours_readout_agrees_with_report=len(agreed),
        note=(None if args.candidate_replacement != "off" else
              "off: no replacement was run, so rescued_rows / flipped_rows are 0 and "
              "flip_rate is null; rescued_rows_available says how many rows the flag "
              "would have covered"),
        seed_rule=("per-row seed = first 4 bytes of sha256('<--seed>:<row id>'), so the "
                   "draw is reproducible and independent of row order"),
        definition=("the graft slots' CONTENT is replaced (random = core slots of the same "
                    "document, shuffle = the graft slots permuted among themselves), the "
                    "window is recalibrated with the same trained module and the SAME "
                    "question is asked again; the 2,304-slot shape, the slot order and the "
                    "core half are untouched and nothing is retrained. `flipped` = the "
                    "row was correct before the replacement and is not afterwards; "
                    "`changed_prediction` = the answer string differs at all. `flip_rate` "
                    "is flipped / rescued (a fraction, not a percentage)"),
        per_row=[dict(id=sample["id"],
                      flipped=bool(sample["candidate_sensitivity"]["flipped"]),
                      changed_prediction=bool(
                          sample["candidate_sensitivity"]["changed_prediction"]),
                      agrees_with_report=bool(
                          sample["candidate_sensitivity"]["agrees_with_report"]),
                      moved_fraction=sample["candidate_sensitivity"]["replacement"][
                          "moved_fraction"],
                      ours_qa_f1=sample["candidate_sensitivity"]["ours"]["qa_f1"],
                      replaced_qa_f1=sample["candidate_sensitivity"]["replaced"]["qa_f1"])
                  for sample in attempted])
    table = oracle_table(report, args)
    if args.export_oracle_table:
        table["path"] = str(write_oracle_table(table, args, output))
    report["oracle_table"] = table
    report["status"] = "complete"
    report["elapsed_seconds"] = time.monotonic() - started
    atomic_json(output, report)
    if args.export_oracle_table:
        print(table["markdown"], flush=True)
    print(json.dumps({k: v for k, v in report["summary"].items()
                      if k not in ("definition", "per_row")}, indent=1), flush=True)
    return report


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--task", default="hotpotqa", choices=("hotpotqa",))
    p.add_argument("--supporting-facts-glob", dest="supporting_facts_glob", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--anchor", default=None)
    p.add_argument("--v11-evict", dest="v11_evict", default=None,
                   help="the V13 checkpoint whose window is being diagnosed")
    p.add_argument("--v13-report", dest="v13_report", default=None,
                   help="a finished report holding the V13 arm; its per-row correctness "
                        "decides which rows are 'wrong'")
    p.add_argument("--candidate-replacement", dest="candidate_replacement", default="off",
                   choices=REPLACEMENT_MODES,
                   help="content sensitivity on the rescued rows (Ours right, Selection "
                        "wrong): replace the graft slots' content, recalibrate and ask "
                        "again. random = core slots of the same document, shuffle = the "
                        "graft slots permuted among themselves; off = no extra readout")
    p.add_argument("--export-oracle-table", dest="export_oracle_table", action="store_true",
                   help="write the Selection / AnchorOnly / NativeGraft / Full / Ours / "
                        "Oracle markdown table (it is always recorded in the json)")
    p.add_argument("--oracle-table-out", dest="oracle_table_out", default=None,
                   help="where to write that table (default: <out>_oracle_table.md)")
    p.add_argument("--out", default=None)
    p.add_argument("--cuda-device", default="0")
    # These two defaults come from the harness geometry, so they are resolved in
    # `evaluate()` instead of here: touching `gr` at parser-build time would make
    # `--help` fail wherever the harness is absent (see HARNESS_DEPS).
    p.add_argument("--window-blocks", dest="window_blocks", type=int, default=None,
                   help="default: eval_graft_recovery.WINDOW_BLOCKS, resolved at run time")
    p.add_argument("--target-blocks", dest="target_blocks", type=int, default=None,
                   help="default: eval_graft_recovery.TARGET_BLOCKS, resolved at run time")
    p.add_argument("--f1-threshold", dest="f1_threshold", type=float, default=0.5)
    p.add_argument("--control-full", dest="control_full", action="store_true",
                   help="positive control: treat all 19 blocks as golden (nothing is "
                        "blanked) and check the harness reproduces the Full arm")
    p.add_argument("--sample-ids", dest="sample_ids", default=None,
                   help="comma separated row ids (gr.prepare_rows honours it)")
    p.add_argument("--max-samples", dest="max_samples", type=int, default=None)
    p.add_argument("--strict-code-hash", dest="strict_code_hash", default="true",
                   choices=("true", "false"))
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.data_dir or not args.model_path or not args.v11_evict:
        raise SystemExit("--data-dir, --model-path and --v11-evict are required")
    if args.candidate_replacement != "off" and not args.v13_report:
        # The rescued rows ARE the report's rows; without it there is nothing to replace
        # on, and silently replacing on every row would answer a different question.
        raise SystemExit("--candidate-replacement needs --v13-report: the rescued rows "
                         "('Ours right, Selection wrong') are defined by that file")
    # Last gate before anything is loaded or written: in a checkout without the harness
    # this raises `MissingHarnessDependency`, whose message lists the absent files, says
    # they exist only in the server checkout, and says to run the tool there. It never
    # falls through to a partial run.
    require_harness(args)
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
