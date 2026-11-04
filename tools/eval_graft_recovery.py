#!/usr/bin/env python3
"""Experiment 1 -- can native slot grafting recover knowledge that Selection dropped?

Three readouts from ONE pre-compression memory
----------------------------------------------
For every question this driver writes exactly `WINDOW_BLOCKS` native blocks
(19 x 512 = 9,728 tokens of the document), snapshots the pool, and then answers
the same question three times, each time restoring that one snapshot first:

    Full        the 19 written blocks, nothing compressed        19 blocks / 4,864 slots
    Selection   the 9 carrier blocks the importance selector keeps 9 blocks / 2,304 slots
    Graft       the same 9 carriers, 216 native core slots each, plus 40 native
                slots per block copied verbatim from the 10 dropped blocks
                (`operator_mode="native"`: no same-block condensation, no
                orthogonal context delta, no core reinforcement)  9 blocks / 2,304 slots

So the three arms differ in exactly one thing -- which native slots reach the
frozen reader -- and Selection/Graft share their carrier set by construction:
`select_window_carrier_blocks` runs ONCE, on the snapshot, and its output is
handed to both installers. No carrier schedule, no replay, no second selector.

The two questions the experiment answers
----------------------------------------
1. When Selection drops a block that holds the evidence, does grafting native
   slots back restore any answering ability?  (Rescued)
2. Grafting has to give up 40 of the carrier's own slots per block.  Does that
   destroy knowledge the compressed memory could still answer?  (Harmed)

Protocol notes (all recorded in the report)
-------------------------------------------
* The write stream is the DOCUMENT PREFIX ONLY. The official HotpotQA template
  is `instruction | passages | instruction | question | Answer:`; the leading
  instruction is dropped so that block b covers exactly tokens
  `[512b, 512(b+1))` of the passages and the char->token->block mapping of the
  evidence stays exact. The trailing instruction + question + `Answer:` is the
  query, so the reader still sees the task sentence. Same prompt for all arms.
* 19 -> 9 is the V10 geometry the checkpoint was trained at, and 2,304 =
  9 x 256 slots is the capacity both compared arms report.
* Rows whose context is shorter than 19 blocks cannot fill the window. They are
  EXCLUDED and counted, never padded: a padded window would be a different
  experiment.
* "Evidence" is the HotpotQA supporting sentence, not the answer string. The
  question is joined against the HotpotQA dev parquet to recover
  `supporting_facts` (title + sentence index), the sentence text is located in
  the context through a normalised index map, and its character span is mapped
  to native blocks through the tokenizer's offset mapping. A row without a
  locatable supporting sentence is graded but flagged, because for it the
  rescue/protection grouping is unknown.
* Screening uses the answer and the evidence location, so the screened-group
  accuracies are a MECHANISM DIAGNOSTIC, not a benchmark score. The report says
  so in `protocol.screening_caveat`.

Usage
-----
    # what would run, without loading the model or touching the GPU
    python tools/eval_graft_recovery.py --dry-run \
        --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
        --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
        --model-path /dev/shm/s1d_reader_fp16 \
        --out reports/diagnostics/graft_recovery_hotpotqa.json

    # the experiment
    python tools/eval_graft_recovery.py \
        --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
        --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
        --model-path /dev/shm/s1d_reader_fp16 \
        --checkpoint runs/natural_merge_v9/full_v10.pt \
        --out reports/diagnostics/graft_recovery_hotpotqa.json

    # re-group a finished report without a GPU
    python tools/eval_graft_recovery.py --analyze reports/diagnostics/graft_recovery_hotpotqa.json
"""
import argparse
import bisect
import glob
import json
import os
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_core as pe                                    # noqa: E402
import native_eval as nv                                  # noqa: E402
from io_utils import atomic_json, digest, sha             # noqa: E402
from runtime import critical_manifest, install, load_runtime, reset  # noqa: E402
from stream_eval import StreamingMemory                   # noqa: E402
from methods.prlmc.consolidate import select_window_carrier_blocks  # noqa: E402
from tools.v9_geometry_50block import GEOMETRY_50B        # noqa: E402

# Geometry: the V10 working point, read from the single source of truth.
BLOCK_TOKENS = int(GEOMETRY_50B.chunk_tokens)             # 512 tokens per native block
SLOTS_PER_BLOCK = int(GEOMETRY_50B.slots_per_block)       # 256 slots per native block
WINDOW_BLOCKS = int(GEOMETRY_50B.window_blocks)           # 19
TARGET_BLOCKS = int(GEOMETRY_50B.target_blocks)           # 9
CORE_SLOTS = int(GEOMETRY_50B.core_slots_per_block)       # 216
GRAFT_SLOTS = int(GEOMETRY_50B.graft_slots_per_block)     # 40
BUDGET_BLOCKS = int(GEOMETRY_50B.budget_blocks)           # 50 (physical pool)

WRITE_TOKENS = WINDOW_BLOCKS * BLOCK_TOKENS               # 9,728
TARGET_SLOTS = TARGET_BLOCKS * SLOTS_PER_BLOCK            # 2,304
CORE_SLOTS_TOTAL = TARGET_BLOCKS * CORE_SLOTS             # 1,944
GRAFT_SLOTS_TOTAL = TARGET_BLOCKS * GRAFT_SLOTS           # 360

# The three arms. "Graft" is not a streaming arm: it is the native-mode operator
# applied once to the Selection carrier set.
FULL, SELECTION, GRAFT = "Full", "Selection", "Graft"
ARMS = (FULL, SELECTION, GRAFT)

# Official LongBench HotpotQA prompt, split so the passages are the write stream.
QUERY_SUFFIX = ("\n\nAnswer the question based on the given passages. Only give me "
                "the answer and do not output any other words.\n\nQuestion: {input}"
                "\nAnswer:")
MAX_NEW = nv.MAX_NEW["hotpotqa"]

SCREENING_CAVEAT = (
    "the rescue/protection groups are selected using the answer and the evidence "
    "location, so their accuracies are a mechanism diagnostic and must never be "
    "reported as a dataset score; the full-split columns are the honest reference")


def _check_geometry():
    """Fail at import if the frozen geometry is not the one this experiment needs."""
    if (WINDOW_BLOCKS, TARGET_BLOCKS) != (19, 9):
        raise ValueError(f"this experiment is pinned to 19->9, got "
                         f"{WINDOW_BLOCKS}->{TARGET_BLOCKS}")
    if CORE_SLOTS + GRAFT_SLOTS != SLOTS_PER_BLOCK:
        raise ValueError("core+graft slots must tile a native block")
    if (CORE_SLOTS_TOTAL, GRAFT_SLOTS_TOTAL) != (1944, 360):
        raise ValueError("unexpected core/graft slot budget")


_check_geometry()


# ---------------------------------------------------------------------------
# text / evidence location
# ---------------------------------------------------------------------------
def normalise_with_map(text):
    """NFKD-folded word stream plus the raw character index of every output char.

    The map is what makes evidence location robust: the supporting sentence is
    searched in normalised space (so quotes, dashes and case differences do not
    matter) and the hit is mapped back to raw character offsets, which is what
    the tokenizer's offset mapping speaks.
    """
    out, raw = [], []
    for index, char in enumerate(text):
        for folded in unicodedata.normalize("NFKD", char.lower()):
            if unicodedata.combining(folded):
                continue
            if folded.isalnum():
                out.append(folded)
                raw.append(index)
            elif out and out[-1] != " ":
                out.append(" ")
                raw.append(index)
    while out and out[-1] == " ":
        out.pop()
        raw.pop()
    return "".join(out), raw


def normalise(text):
    return normalise_with_map(text or "")[0]


def occurrence_spans(context, needle, limit=32):
    """Raw character spans of every occurrence of a normalised needle."""
    if not needle:
        return []
    text, raw = normalise_with_map(context)
    spans, start = [], text.find(needle)
    while start >= 0 and len(spans) < limit:
        spans.append((raw[start], raw[start + len(needle) - 1] + 1))
        start = text.find(needle, start + 1)
    return spans


def evidence_char_spans(context, sentences):
    """Raw character spans of the supporting sentences inside `context`."""
    text, raw = normalise_with_map(context)
    spans = []
    for sentence in sentences:
        needle = normalise(sentence)
        # A very short sentence ("yes.") matches everywhere and is not evidence.
        if len(needle) < 8:
            continue
        start = text.find(needle)
        if start < 0:
            continue
        spans.append((raw[start], raw[start + len(needle) - 1] + 1))
    return spans


def spans_to_blocks(offsets, spans, write_tokens):
    """Character spans -> token positions -> native-block indices of the window.

    `offsets` is the tokenizer offset mapping of the whole context, so a span
    that starts past the write prefix simply contributes no block: the evidence
    is then outside the window, which is exactly the fact the caller needs.
    """
    kept = min(len(offsets), write_tokens)
    if kept <= 0:
        return [], []
    lefts = [offsets[index][0] for index in range(kept)]
    rights = [offsets[index][1] for index in range(kept)]
    if any(lefts[index] > lefts[index + 1] for index in range(kept - 1)):
        raise ValueError("tokenizer offsets are not monotonic")
    tokens, blocks = set(), set()
    for start, end in spans:
        first = max(0, bisect.bisect_right(rights, start))
        last = min(kept, bisect.bisect_left(lefts, end))
        for index in range(first, last):
            tokens.add(index)
            blocks.add(index // BLOCK_TOKENS)
    return sorted(tokens), sorted(blocks)


def load_supporting_facts(pattern):
    """normalised question -> supporting sentence texts, from the HotpotQA dev parquet.

    LongBench's `hotpotqa.jsonl` carries no `supporting_facts`, so the evidence
    has to be recovered from the upstream dev split. The join is by normalised
    question text; the caller reports the join rate instead of assuming it.
    """
    if not pattern:
        return {}
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ValueError(
            "--supporting-facts-glob needs pyarrow (pip install pyarrow); without "
            "it the evidence location and therefore the whole grouping is unknown"
        ) from exc
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(f"--supporting-facts-glob matched no file: {pattern}")
    table = {}
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                context = row.get("context") or {}
                sentences = {title: list(group) for title, group in
                             zip(context.get("title") or [], context.get("sentences") or [])}
                facts = row.get("supporting_facts") or {}
                indices = facts.get("sent_id")
                if indices is None:
                    indices = facts.get("sent_idx") or []
                picked = []
                for title, index in zip(facts.get("title") or [], indices):
                    group = sentences.get(title)
                    if group and 0 <= int(index) < len(group):
                        picked.append(group[int(index)])
                table[normalise(row.get("question", ""))] = picked
    return table


# ---------------------------------------------------------------------------
# sample preparation
# ---------------------------------------------------------------------------
def prepare_sample(tokenizer, row, facts, write_tokens=WRITE_TOKENS):
    """Write stream, query and evidence blocks for one row, or a named exclusion.

    Nothing here touches the reader: this is the part `--dry-run` exercises so a
    broken data path fails before a single GPU-second is spent.
    """
    context = row["context"]
    encoded = tokenizer(context, add_special_tokens=False, truncation=False,
                        return_offsets_mapping=True)
    context_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    available = len(context_ids) // BLOCK_TOKENS
    if available < WINDOW_BLOCKS:
        return None, dict(eligible=False,
                          reason=f"context is {len(context_ids)} tokens = {available} "
                                 f"blocks, the window needs {WINDOW_BLOCKS}",
                          context_tokens=len(context_ids), available_blocks=available)

    answers = [str(a) for a in row["answers"]]
    sentences = facts.get(normalise(row["input"])) or []
    spans = evidence_char_spans(context, sentences) if sentences else []
    tokens, blocks = spans_to_blocks(offsets, spans, write_tokens)
    # The answer string is recorded only as a cross-check: an answer can be a
    # common word and its presence is not evidence, which is why it never
    # decides the grouping.
    _, answer_blocks = spans_to_blocks(
        offsets, occurrence_spans(context, normalise(answers[0])), write_tokens)

    return dict(
        write_ids=context_ids[:write_tokens],
        query_ids=tokenizer(QUERY_SUFFIX.format(input=row["input"]),
                            add_special_tokens=False).input_ids,
        evidence=dict(
            supporting_sentences=len(sentences),
            supporting_sentences_located=len(spans),
            located=bool(blocks),
            blocks=blocks,
            token_span=[min(tokens), max(tokens)] if tokens else None,
            answer_string_in_context=bool(occurrence_spans(context, normalise(answers[0]), limit=1)),
            answer_string_blocks=answer_blocks,
        )), dict(eligible=True, context_tokens=len(context_ids), available_blocks=available)


# ---------------------------------------------------------------------------
# reader state
# ---------------------------------------------------------------------------
def snapshot(model, manager):
    """Everything the compression and the readout may mutate, kept on the device.

    A device-side clone is ~0.1 s for 3.3 GB and is what makes the three arms
    literally share one pre-compression memory instead of reconstructing it.
    """
    return dict(
        memory=model.memory.detach().clone(),
        active_mask=model.memory_active_mask.detach().clone(),
        initialized=model.initialized.detach().clone(),
        used=int(manager.used),
        n_consolidated=int(manager.n_consolidated),
        # `goal_direction` may be a tensor or None, `_active_mask_used` is a plain
        # int memo of the last masked `used` -- not a tensor.
        goal_direction=(manager.goal_direction.detach().clone()
                        if torch.is_tensor(manager.goal_direction)
                        else manager.goal_direction),
        active_mask_used=manager._active_mask_used,
        block_ages=manager._block_ages.detach().clone(),
        slot_fused=(None if manager.slot_fused is None
                    else manager.slot_fused.detach().clone()),
        pinned_blocks=(None if manager.pinned_blocks is None
                       else list(manager.pinned_blocks)),
    )


def restore(model, manager, state):
    with torch.no_grad():
        model.memory.copy_(state["memory"])
        model.memory_active_mask.copy_(state["active_mask"])
        model.initialized.copy_(state["initialized"])
    manager.used = state["used"]
    manager.n_consolidated = state["n_consolidated"]
    manager.goal_direction = (state["goal_direction"].clone()
                              if torch.is_tensor(state["goal_direction"])
                              else state["goal_direction"])
    manager._active_mask_used = state["active_mask_used"]
    manager._block_ages.copy_(state["block_ages"])
    manager.slot_fused = (None if state["slot_fused"] is None
                          else state["slot_fused"].clone())
    manager.pinned_blocks = (None if state["pinned_blocks"] is None
                             else list(state["pinned_blocks"]))


def active_fingerprint(manager):
    """Sampled mean of the resident memory: a cheap provenance cross-check.

    Not a hash. It is here so the report can show that Full, Selection and Graft
    were read from three different pools while Selection and Graft shared one
    pre-compression state.
    """
    with torch.no_grad():
        return float(manager._memory[:, -manager.used::13].float().mean())


# ---------------------------------------------------------------------------
# write phase and the two compressions
# ---------------------------------------------------------------------------
def write_window(manager, merge, chunks):
    """Append the whole 19-block window through the managed native write path.

    `StreamingMemory` with arm "Full" is the plain writer: it skips compaction
    and only enforces the physical pool, which is exactly what we want because
    the single compression is applied deliberately afterwards.
    """
    if len(chunks) != WINDOW_BLOCKS:
        raise ValueError(f"expected {WINDOW_BLOCKS} write blocks, got {len(chunks)}")
    stream = StreamingMemory(manager, merge, "Full", budget_blocks=BUDGET_BLOCKS)
    for index, chunk in enumerate(chunks):
        record = dict(role="write", token_ids=chunk, token_sha256=digest(chunk))
        if stream.write(record, index) is None:
            raise ValueError("the writer refused a block: the window does not fit the pool")
    if int(manager.used) != WINDOW_BLOCKS * manager.block:
        raise ValueError("the write phase did not leave exactly one window resident")
    return stream


def select_carriers(manager, window_blocks, target_blocks):
    """The 9 carrier blocks, chosen once and shared by Selection and Graft."""
    start = manager.total - manager.used
    source = manager._memory[:, start:start + window_blocks * manager.block].contiguous()
    ages = manager.window_block_ages(start, window_blocks * manager.block)
    picked = select_window_carrier_blocks(
        window_memory=source, target_blocks=target_blocks, block_size=manager.block,
        block_ages=ages, condensed_slot_mask=None)
    picked = [int(i) for i in picked]
    if (len(picked) != target_blocks or len(set(picked)) != target_blocks
            or min(picked) < 0 or max(picked) >= window_blocks):
        raise ValueError(f"the selector returned an invalid carrier set: {picked}")
    return source, ages, picked


def window_locks(manager, window_blocks):
    """Freeze mask and archive-pinned blocks of the window, as the V10 stream does."""
    if not hasattr(manager, "fused_window"):
        return None, None
    start = manager.total - manager.used
    frozen, pinned = manager.fused_window(start, window_blocks * manager.block)
    manager.pinned_blocks = pinned
    return frozen, pinned


def install_selection(manager, source, picked, window_blocks):
    """Arm A: keep the 9 carrier blocks whole, byte for byte."""
    window = window_blocks * manager.block
    selected = torch.as_tensor(picked, dtype=torch.long, device=source.device)
    blocks = source.reshape(manager.num_layers, window_blocks, manager.block, manager.hidden)
    output = blocks.index_select(1, selected).flatten(1, 2)
    install(manager, output, window, selected_blocks=selected)
    return selected


def install_graft(manager, merge, source, picked, window_blocks, target_blocks, frozen, pinned):
    """Arm B: native-mode V10 operator on the SAME carriers, then the same install."""
    window = window_blocks * manager.block
    target = target_blocks * manager.block
    selected = torch.as_tensor(picked, dtype=torch.long, device=source.device)
    allow = torch.ones(window_blocks, dtype=torch.bool, device=source.device)
    output = merge(source, target=target, block=manager.block, carrier_indices=selected,
                   source_allow_mask=allow, frozen_slots=frozen, pinned_blocks=pinned)
    install(manager, output, window, selected_blocks=selected, fused_offsets=None)
    return selected, dict(merge.last_fusion)


def graft_provenance(telemetry, window_blocks):
    """Which window blocks the 360 graft slots came from, and which carriers survived.

    The output slot order is chronological, so graft slots are interleaved with
    the carriers: the graft mask has to be `bridge_slot_offsets`, never "the last
    360 entries". Every claim below is checked against the selection instead of
    being inferred, because a provenance report that quietly disagrees with the
    compression is worse than no provenance report.
    """
    selected = [int(i) for i in telemetry["selected_blocks"]]
    source_blocks = [int(i) for i in telemetry["slot_source_blocks"]]
    offsets = [int(o) for o in telemetry["bridge_slot_offsets"]]
    total = len(source_blocks)
    core_count = int(telemetry["core_slot_count"])
    if total != TARGET_SLOTS or core_count != CORE_SLOTS_TOTAL:
        raise ValueError(f"unexpected output geometry: {total} slots, {core_count} core")
    if len(offsets) != GRAFT_SLOTS_TOTAL:
        raise ValueError(f"expected {GRAFT_SLOTS_TOTAL} graft slots, got {len(offsets)}")
    graft_offsets = set(offsets)
    core_sources = [source_blocks[index] for index in range(total) if index not in graft_offsets]
    graft_sources = [source_blocks[index] for index in offsets]
    kept = sorted(set(core_sources))
    dropped = sorted(set(range(window_blocks)) - set(selected))
    if kept != sorted(selected):
        raise ValueError(f"carrier provenance {kept} disagrees with the selection {selected}")
    if set(graft_sources) - set(dropped):
        raise ValueError("a graft slot claims a block that was not dropped")
    counts = Counter(graft_sources)
    return dict(
        selected_blocks=selected,
        dropped_blocks=dropped,
        core_slots=len(core_sources),
        graft_slots=len(graft_sources),
        graft_slots_by_source_block={str(block): int(counts.get(block, 0))
                                     for block in range(window_blocks)},
        graft_slots_per_dropped_block=[int(counts.get(block, 0)) for block in dropped],
        dropped_blocks_without_graft=[block for block in dropped if not counts.get(block)],
        operator_mode=telemetry.get("operator_mode"),
        mode=telemetry.get("mode"),
    )


# ---------------------------------------------------------------------------
# readout
# ---------------------------------------------------------------------------
def answer_logprob(model, tokenizer, query_ids, answer):
    """Teacher-forced log-likelihood of one gold answer after the query.

    `p_gold` is the geometric mean per-token probability, so it is comparable
    across answers of different length; `logprob_sum` is the joint log-likelihood
    and is reported next to it because a short answer can have a high mean while
    the joint probability is low.
    """
    answer_ids = tokenizer(" " + answer.strip(), add_special_tokens=False).input_ids
    if not answer_ids:
        return None
    ids = list(query_ids) + list(answer_ids)
    tensor = torch.tensor([ids], dtype=torch.long, device=model.memory.device)
    attention = torch.ones((1, model.memory.shape[1] + len(ids)), dtype=torch.long,
                           device=tensor.device)
    with torch.no_grad():
        logits = model(input_ids=tensor, attention_mask=attention,
                       use_cache=False).logits[0].float()
        logprobs = torch.log_softmax(logits, dim=-1)
    start = len(query_ids)
    picked = torch.stack([logprobs[position - 1, ids[position]]
                          for position in range(start, len(ids))])
    total = float(picked.sum())
    return dict(logprob_sum=total, n_tokens=len(picked),
                mean_logprob=total / len(picked),
                p_gold=float(np.exp(total / len(picked))))


def readout(model, tokenizer, query_ids, answers, threshold, prepare):
    """One greedy answer plus its official QA F1 / EM / gold likelihood.

    `prepare` restores the pre-compression snapshot and applies this arm's
    compression. It runs before BOTH passes, so the greedy answer and the gold
    likelihood are measured on the same memory even if generation were to touch
    the pool.
    """
    prepare()
    prediction, output_ids = pe.generate(model, tokenizer, query_ids, MAX_NEW)
    prepare()
    gold = answer_logprob(model, tokenizer, query_ids, answers[0])
    record = dict(prediction=prediction, output_tokens=output_ids,
                  qa_f1=float(nv.score(prediction, answers)),
                  qa_f1_truncated=float(nv.score(nv.truncated_prediction(prediction), answers)),
                  em=int(any(normalise(prediction) == normalise(answer) for answer in answers)),
                  correct=int(nv.score(prediction, answers) >= threshold),
                  f1_threshold=threshold)
    if gold is not None:
        record.update(gold)
    return record


# ---------------------------------------------------------------------------
# one sample
# ---------------------------------------------------------------------------
def run_sample(model, tokenizer, manager, merge, prepared, args):
    """Write 19 blocks once, then answer Full / Selection / Graft from one snapshot."""
    chunks = [prepared["write_ids"][index:index + BLOCK_TOKENS]
              for index in range(0, len(prepared["write_ids"]), BLOCK_TOKENS)]
    answers = prepared["answers"]
    fingerprints, telemetries = {}, []

    with torch.no_grad():
        write_window(manager, merge, chunks)
        state = snapshot(model, manager)
        source, ages, picked = select_carriers(manager, args.window_blocks, args.target_blocks)
        frozen, pinned = window_locks(manager, args.window_blocks)
    restore(model, manager, state)

    def prepare(arm):
        def run():
            restore(model, manager, state)
            if arm == SELECTION:
                install_selection(manager, source, picked, args.window_blocks)
            elif arm == GRAFT:
                _, telemetry = install_graft(manager, merge, source, picked,
                                             args.window_blocks, args.target_blocks,
                                             frozen, pinned)
                telemetries.append(telemetry)
            fingerprints[arm] = active_fingerprint(manager)
            fingerprints[arm + "/slots"] = int(manager.used)
        return run

    arms = {arm: readout(model, tokenizer, prepared["query_ids"], answers,
                         args.f1_threshold, prepare(arm)) for arm in ARMS}
    restore(model, manager, state)

    if len(telemetries) != 2:
        raise ValueError(f"expected two graft passes, got {len(telemetries)}")
    if telemetries[0]["bridge_slot_offsets"] != telemetries[1]["bridge_slot_offsets"]:
        raise ValueError("the graft compression is not deterministic across readouts")
    provenance = graft_provenance(telemetries[0], args.window_blocks)
    if provenance["selected_blocks"] != sorted(picked):
        raise ValueError("graft provenance disagrees with the shared carrier set")
    if fingerprints[FULL] == fingerprints[SELECTION]:
        raise ValueError("Full and Selection read the same pool: the compression did not run")
    if fingerprints[SELECTION] == fingerprints[GRAFT]:
        raise ValueError("Selection and Graft read the same pool: no native slot was grafted")
    return dict(arms=arms, fingerprints=fingerprints, provenance=provenance,
                carrier_ages=[int(ages[block]) for block in picked])


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def _mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _arm_summary(samples, arm):
    records = [sample["arms"][arm] for sample in samples if arm in sample.get("arms", {})]
    if not records:
        return dict(n=0)
    return dict(n=len(records),
                qa_f1=round(100.0 * _mean([r["qa_f1"] for r in records]), 2),
                em=round(100.0 * _mean([r["em"] for r in records]), 2),
                accuracy=round(100.0 * _mean([r["correct"] for r in records]), 2),
                p_gold=_mean([r.get("p_gold") for r in records]),
                mean_logprob=_mean([r.get("mean_logprob") for r in records]))


def _paired(samples, left, right, key):
    """Mean and sign counts of `left - right` over samples that have both."""
    deltas = [sample["arms"][left][key] - sample["arms"][right][key] for sample in samples
              if key in sample.get("arms", {}).get(left, {})
              and key in sample.get("arms", {}).get(right, {})]
    if not deltas:
        return dict(n=0)
    return dict(n=len(deltas), mean=float(np.mean(deltas)),
                up=int(sum(delta > 0 for delta in deltas)),
                down=int(sum(delta < 0 for delta in deltas)),
                equal=int(sum(delta == 0 for delta in deltas)))


def _evidence_graft_slots(sample):
    counts = sample["provenance"]["graft_slots_by_source_block"]
    return sum(int(counts.get(str(block), 0)) for block in sample["evidence"]["blocks"])


def analyse(samples, threshold):
    """Rescued / Harmed and the provenance question, on the graded samples only."""
    graded = [sample for sample in samples if sample.get("status") == "graded"]
    located = [sample for sample in graded if sample.get("evidence", {}).get("located")]

    def evidence_in_dropped(sample):
        blocks = set(sample["evidence"]["blocks"])
        return bool(blocks & set(sample["provenance"]["dropped_blocks"]))

    rescue = [sample for sample in located
              if sample["arms"][FULL]["correct"]
              and not sample["arms"][SELECTION]["correct"]
              and evidence_in_dropped(sample)]
    protect = [sample for sample in located
               if sample["arms"][FULL]["correct"]
               and sample["arms"][SELECTION]["correct"]]
    rescued = [sample for sample in rescue if sample["arms"][GRAFT]["correct"]]
    harmed = [sample for sample in protect if not sample["arms"][GRAFT]["correct"]]

    # The same two counters over every graded sample, with no evidence filter:
    # this is the number that cannot be accused of cherry-picking.
    rescued_all = [sample for sample in graded
                   if not sample["arms"][SELECTION]["correct"]
                   and sample["arms"][GRAFT]["correct"]]
    harmed_all = [sample for sample in graded
                  if sample["arms"][SELECTION]["correct"]
                  and not sample["arms"][GRAFT]["correct"]]

    evidence_block_slots = [_evidence_graft_slots(sample) for sample in located]
    all_slot_counts = [count for sample in located
                       for count in sample["provenance"]["graft_slots_by_source_block"].values()]

    return dict(
        counts=dict(
            attempted=len(samples), graded=len(graded),
            excluded=len(samples) - len(graded),
            evidence_located=len(located),
            evidence_in_window=len([sample for sample in located
                                    if sample["evidence"]["blocks"]]),
            full_correct=len([sample for sample in graded
                              if sample["arms"][FULL]["correct"]]),
            full_wrong=len([sample for sample in graded
                            if not sample["arms"][FULL]["correct"]]),
            selection_correct=len([sample for sample in graded
                                   if sample["arms"][SELECTION]["correct"]]),
            graft_correct=len([sample for sample in graded
                               if sample["arms"][GRAFT]["correct"]]),
            evidence_in_dropped=len([sample for sample in located
                                     if evidence_in_dropped(sample)])),
        arms={arm: _arm_summary(graded, arm) for arm in ARMS},
        arms_evidence_located={arm: _arm_summary(located, arm) for arm in ARMS},
        paired=dict(
            f1_graft_minus_selection=_paired(graded, GRAFT, SELECTION, "qa_f1"),
            f1_graft_minus_full=_paired(graded, GRAFT, FULL, "qa_f1"),
            f1_selection_minus_full=_paired(graded, SELECTION, FULL, "qa_f1"),
            p_gold_graft_minus_selection=_paired(graded, GRAFT, SELECTION, "p_gold"),
            logprob_graft_minus_selection=_paired(graded, GRAFT, SELECTION, "mean_logprob")),
        rescue_group=dict(
            definition=("Full correct, Selection wrong, and the dropped blocks hold "
                        "the supporting sentence"),
            n=len(rescue), rescued=len(rescued),
            rescued_rate=(len(rescued) / len(rescue) if rescue else None),
            evidence_block_grafted=len([sample for sample in rescue
                                        if _evidence_graft_slots(sample) > 0]),
            rescued_ids=[sample["id"] for sample in rescued],
            ids=[sample["id"] for sample in rescue]),
        protection_group=dict(
            definition="Full correct and Selection correct",
            n=len(protect), harmed=len(harmed),
            harmed_rate=(len(harmed) / len(protect) if protect else None),
            harmed_ids=[sample["id"] for sample in harmed],
            ids=[sample["id"] for sample in protect]),
        ungated=dict(
            definition="every graded sample, no evidence filter",
            rescued=len(rescued_all), harmed=len(harmed_all),
            selection_wrong=len([sample for sample in graded
                                 if not sample["arms"][SELECTION]["correct"]]),
            selection_correct=len([sample for sample in graded
                                   if sample["arms"][SELECTION]["correct"]])),
        graft_provenance=dict(
            graft_slots_total=GRAFT_SLOTS_TOTAL,
            mean_slots_per_block=float(np.mean(all_slot_counts)) if all_slot_counts else None,
            samples_with_a_dropped_block_unused=len(
                [sample for sample in graded
                 if sample["provenance"]["dropped_blocks_without_graft"]]),
            mean_evidence_block_graft_slots=_mean(evidence_block_slots),
            evidence_blocks_with_zero_graft=len([count for count in evidence_block_slots
                                                 if count == 0]),
            note=("`graft_slots_by_source_block` is keyed by window block index; a "
                  "dropped block with 0 slots contributed nothing to the merge, "
                  "which points at slot selection rather than at the reader")),
        screening_caveat=SCREENING_CAVEAT,
        f1_threshold=threshold)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def build_protocol(args, task):
    return dict(
        experiment="1 native slot grafting knowledge recovery",
        task=task, data_dir=args.data_dir,
        arms=list(ARMS),
        window_blocks=args.window_blocks, target_blocks=args.target_blocks,
        block_tokens=BLOCK_TOKENS, slots_per_block=SLOTS_PER_BLOCK,
        write_tokens=args.window_blocks * BLOCK_TOKENS,
        full_slots=args.window_blocks * SLOTS_PER_BLOCK,
        target_slots=args.target_blocks * SLOTS_PER_BLOCK,
        core_slots_total=args.target_blocks * CORE_SLOTS,
        graft_slots_total=args.target_blocks * GRAFT_SLOTS,
        compressions_per_sample=1,
        write_stream="document prefix only (the leading template instruction is dropped)",
        query=QUERY_SUFFIX,
        prompt_note=("all three arms answer the identical query tokens; the official "
                     "leading instruction is dropped so block b covers exactly tokens "
                     "[512b, 512(b+1)) of the passages and the evidence mapping is exact"),
        operator_mode="native",
        operator_note=("carrier and graft slots are copied verbatim; no same-block "
                       "condensation, no orthogonal context delta, no core reinforcement"),
        carrier_selection=("select_window_carrier_blocks runs once on the "
                           "pre-compression window; Selection and Graft share its output"),
        decoding="greedy, first non-blank line, max_new_tokens=%d" % MAX_NEW,
        scoring=("official LongBench HotpotQA QA F1 (max over golds); "
                 "EM; gold-answer log-likelihood"),
        f1_threshold=args.f1_threshold,
        supporting_facts=args.supporting_facts_glob,
        seed=args.seed, training_performed=False,
        screening_caveat=SCREENING_CAVEAT)


def prepare_rows(args, tokenizer, facts):
    rows = nv.load_rows(args.data_dir, args.task)
    if args.sample_ids:
        wanted = [str(x) for x in json.loads(Path(args.sample_ids).read_text())]
        table = {str(row.get("_id")): row for row in rows}
        missing = [w for w in wanted if w not in table]
        if missing:
            raise ValueError(f"--sample-ids not found in {args.data_dir}: {missing}")
        rows = [table[w] for w in wanted]
    if args.max_samples is not None:
        rows = rows[:args.max_samples]
    prepared, excluded = [], []
    for index, row in enumerate(rows):
        sample, status = prepare_sample(tokenizer, row, facts)
        if sample is None:
            excluded.append(dict(sample_index=index, id=row.get("_id"), **status))
            continue
        sample.update(sample_index=index, id=row.get("_id"),
                      answers=[str(answer) for answer in row["answers"]],
                      question=row["input"], context_sha256=digest(row["context"]),
                      source_row_sha256=digest(row), **status)
        prepared.append(sample)
    return rows, prepared, excluded


def dry_run(args, facts, tokenizer):
    rows, prepared, excluded = prepare_rows(args, tokenizer, facts)
    located = [sample for sample in prepared if sample["evidence"]["located"]]
    histogram = Counter(block for sample in located for block in sample["evidence"]["blocks"])
    plan = dict(
        dry_run=True, out=args.out,
        protocol=build_protocol(args, args.task),
        counts=dict(rows=len(rows), eligible=len(prepared), excluded=len(excluded),
                    supporting_facts_joined=len(
                        [sample for sample in prepared
                         if sample["evidence"]["supporting_sentences"]]),
                    sentence_located_in_context=len(
                        [sample for sample in prepared
                         if sample["evidence"]["supporting_sentences_located"]]),
                    evidence_in_window=len(located),
                    sentence_found_but_outside_window=len(
                        [sample for sample in prepared
                         if sample["evidence"]["supporting_sentences_located"]
                         and not sample["evidence"]["located"]])),
        evidence_block_histogram={str(block): histogram[block] for block in sorted(histogram)},
        excluded_examples=excluded[:5],
        eligible_ids=[sample["id"] for sample in prepared],
        evidence_preview=[dict(id=sample["id"], blocks=sample["evidence"]["blocks"],
                               tokens=sample["evidence"]["token_span"])
                          for sample in located[:10]])
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return plan


def evaluate(args):
    facts = load_supporting_facts(args.supporting_facts_glob)
    if not facts:
        print("WARNING: no supporting facts; the evidence grouping will be empty. "
              "Pass --supporting-facts-glob for the actual experiment.", file=sys.stderr)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    if args.dry_run:
        return dry_run(args, facts, tokenizer)
    if not args.checkpoint or not Path(args.checkpoint).is_file():
        raise ValueError("the Graft arm needs the trained operator: pass --checkpoint")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("output exists; refuse overwrite")

    rows, prepared, excluded = prepare_rows(args, tokenizer, facts)
    if not prepared:
        raise ValueError("no eligible sample: every context is shorter than the window")
    if args.seed != 0:
        raise ValueError("the fixed evaluation seed is 0")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.cuda_device))
    nv.seed_everything(args.seed)
    pe.install_generation_compatibility()

    checkpoint_training = torch.load(args.checkpoint, map_location="cpu").get("training")
    model, tokenizer, manager, merge, initial, cfg, saved = load_runtime(
        args.model_path, checkpoint=args.checkpoint, train=False,
        architecture=None, budget_blocks=BUDGET_BLOCKS,
        strict_code_hash=args.strict_code_hash, training=checkpoint_training)
    tokenizer.pad_token = tokenizer.eos_token
    if int(model.num_blocks) != BUDGET_BLOCKS or int(model.num_tokens) != SLOTS_PER_BLOCK:
        raise ValueError("expected the native 50-block / 256-slot MemoryLLM reader")
    merge.operator_mode = "native"       # experiment 1: native slots only

    report = dict(
        schema="graft_recovery_results/1", status="running",
        protocol=dict(build_protocol(args, args.task),
                      operator_mode_applied=merge.operator_mode),
        identity=dict(
            model_path=args.model_path, checkpoint=args.checkpoint,
            checkpoint_sha256=sha(args.checkpoint),
            checkpoint_code_hash_verified=saved.get("code_hash_verified") if saved else None,
            data_dir=args.data_dir,
            data_sha256=sha(Path(args.data_dir) / (args.task + ".jsonl")),
            supporting_facts_glob=args.supporting_facts_glob,
            entrypoint_sha256=sha(__file__),
            critical_manifest=critical_manifest(),
            reader_frozen=True),
        excluded=excluded, samples=[], training_performed=False,
        optimizer_created=False, full_split=(args.max_samples is None and args.sample_ids is None))
    atomic_json(output, report)

    started = time.monotonic()
    for position, sample in enumerate(prepared):
        sample_started = time.monotonic()
        nv.seed_everything(args.seed)
        reset(manager, initial)
        with torch.no_grad():
            manager._tile_free_region()
            manager._sync_active_mask()
        result = run_sample(model, tokenizer, manager, merge, sample, args)
        record = dict(sample_index=sample["sample_index"], id=sample["id"],
                      question=sample["question"], answers=sample["answers"],
                      context_sha256=sample["context_sha256"],
                      source_row_sha256=sample["source_row_sha256"],
                      context_tokens=sample["context_tokens"],
                      available_blocks=sample["available_blocks"],
                      evidence=sample["evidence"], status="graded",
                      seconds=time.monotonic() - sample_started, **result)
        report["samples"].append(record)
        report["progress"] = dict(sample=position + 1, total=len(prepared))
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output, report)
        print(json.dumps(dict(sample=position + 1, total=len(prepared), id=sample["id"],
                              full=result["arms"][FULL]["correct"],
                              selection=result["arms"][SELECTION]["correct"],
                              graft=result["arms"][GRAFT]["correct"],
                              f1=round(result["arms"][SELECTION]["qa_f1"], 3),
                              dropped=len(result["provenance"]["dropped_blocks"]),
                              evidence=sample["evidence"]["blocks"],
                              seconds=round(record["seconds"], 1))), flush=True)

    report["summary"] = analyse(report["samples"], args.f1_threshold)
    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["status"] = "complete"
    atomic_json(output, report)
    print(json.dumps(report["summary"], indent=1, ensure_ascii=False))


def analyze_existing(path):
    report = json.loads(Path(path).read_text())
    threshold = report.get("protocol", {}).get("f1_threshold", 0.5)
    summary = analyse(report.get("samples") or [], threshold)
    report["summary"] = summary
    atomic_json(Path(path), report)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None,
                   help="directory holding <task>.jsonl (LongBench v1 test files)")
    p.add_argument("--task", default="hotpotqa", choices=("hotpotqa",))
    p.add_argument("--supporting-facts-glob", dest="supporting_facts_glob", default=None,
                   help="HotpotQA dev parquet glob providing supporting_facts; without it "
                        "the evidence location (and therefore the grouping) is unknown")
    p.add_argument("--model-path", default=None,
                   help="fp16 MemoryLLM reader directory (default: the eval_core constant)")
    p.add_argument("--checkpoint", default=None,
                   help="trained V10 operator, e.g. runs/natural_merge_v9/full_v10.pt")
    p.add_argument("--out", default=None)
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--window-blocks", dest="window_blocks", type=int, default=WINDOW_BLOCKS)
    p.add_argument("--target-blocks", dest="target_blocks", type=int, default=TARGET_BLOCKS)
    p.add_argument("--f1-threshold", dest="f1_threshold", type=float, default=0.5,
                   help="QA F1 at or above this counts as a correct answer (0.5 default); "
                        "Rescued/Harmed are defined on this binary")
    p.add_argument("--sample-ids", dest="sample_ids", default=None,
                   help="JSON list of LongBench row ids; evaluates exactly these rows")
    p.add_argument("--max-samples", dest="max_samples", type=int, default=None)
    p.add_argument("--strict-code-hash", dest="strict_code_hash", default="true",
                   choices=("true", "false"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="resolve the sample set and evidence blocks, print the plan, "
                        "load no model and write no file")
    p.add_argument("--analyze", default=None,
                   help="re-group an existing report without inference")
    return p


def main():
    args = build_parser().parse_args()
    if args.analyze:
        analyze_existing(args.analyze)
        return
    args.strict_code_hash = args.strict_code_hash == "true"
    if args.model_path is None:
        args.model_path = pe.MODEL_PATH
    if not args.data_dir:
        raise SystemExit("--data-dir is required")
    if (args.window_blocks, args.target_blocks) != (WINDOW_BLOCKS, TARGET_BLOCKS):
        raise SystemExit(f"this experiment is pinned to {WINDOW_BLOCKS}->{TARGET_BLOCKS}; "
                         f"got {args.window_blocks}->{args.target_blocks}")
    if args.out is None and not args.dry_run:
        raise SystemExit("--out is required unless --dry-run")
    evaluate(args)


if __name__ == "__main__":
    main()
