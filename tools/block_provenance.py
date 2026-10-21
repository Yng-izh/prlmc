#!/usr/bin/env python3
"""Block-level provenance for individual LongBench samples.

Answers one question about a wrong answer: was the evidence even reachable?

It rebuilds the exact prompt for a sample (same tokenizer, same total/tail split),
locates the sentence that contains the gold answer, maps that sentence to the
512-token write chunk that produced a native block, and then replays each arm's
recorded compaction events to see whether that block was still resident when the
question was asked. It also locates the text behind each arm's own prediction, so
a wrong answer can be traced to a block that is or is not in memory.

Three very different failures get separated:

    evidence in the tail query window   -> both arms read it verbatim, memory can
                                           only hurt by interference
    evidence block dropped by an arm    -> real information loss
    evidence block kept by both arms    -> the arm corrupted retained content

The pure functions (`replay`, `absorbed`, `blocks_of`) take plain data and are
covered by tools/test_block_provenance.py; only `main` needs the tokenizer.

Usage:
    python tools/block_provenance.py --indices 24,35,68
    python tools/block_provenance.py --indices 24,35 --runs-dir runs/longbench_prlmc_v7_eval
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 512          # tokens per written block
TOTAL = 16384        # prompt budget
TAIL = 2048          # tokens answered verbatim, never written to memory


def _events(sample):
    return {int(e["before_write"]): e for e in sample["compression_events"]}


def replay(sample, window_blocks, n_blocks):
    """Resident source-block ids after the last write.

    Each write appends one block; a compaction rewrites the oldest
    `window_blocks` entries into the block ids named by `selected_blocks`.
    """
    events = _events(sample)
    resident = []
    for write in range(n_blocks):
        event = events.get(write)
        if event is not None:
            window = resident[:window_blocks]
            resident = [window[i] for i in event["selected_blocks"]] + resident[window_blocks:]
        resident.append(write)
    return resident


def absorbed(evidence, sample, window_blocks, n_blocks):
    """Writes at which evidence blocks sat inside a compaction window.

    The managed arms only name carriers in `selected_blocks`, but SharedMerge
    feeds every window block to the fusion (`allow` is all ones), so a block can
    be "absorbed" without appearing in the carrier list.
    """
    events = _events(sample)
    resident, seen = [], []
    for write in range(n_blocks):
        event = events.get(write)
        if event is not None:
            window = resident[:window_blocks]
            hit = evidence & set(window)
            if hit:
                seen.append((write, sorted(hit)))
            resident = [window[i] for i in event["selected_blocks"]] + resident[window_blocks:]
        resident.append(write)
    return seen


def blocks_of(span, write_tokens):
    """Block ids touched by a token span, clipped to the written prefix."""
    start, end = span
    start = max(0, min(start, write_tokens - 1))
    end = max(start, min(end, write_tokens))
    return set(range(start // BLOCK, (end - 1) // BLOCK + 1))


def sentence_spans(tokenizer, prompt, phrase, dropped):
    """Token spans (inside all_ids) of every sentence containing `phrase`."""
    def count(text):
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    out, start = [], 0
    while True:
        pos = prompt.find(phrase, start)
        if pos < 0:
            return out
        begin = prompt.rfind(". ", 0, pos)
        begin = 0 if begin < 0 else begin + 2
        end = prompt.find(". ", pos)
        end = len(prompt) if end < 0 else end
        out.append((count(prompt[:begin]) - dropped, count(prompt[:end]) - dropped))
        start = pos + 1


def answer_forms(answer):
    """The answer plus its suffixes, so 'Hakim Abdullah Jamal' also matches 'Hakim Jamal'."""
    words = answer.split()
    return [f for f in [answer] + [" ".join(words[c:]) for c in range(1, len(words))]
            if len(f) >= 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", required=True, help="comma separated sample_index list")
    ap.add_argument("--task", default="hotpotqa")
    ap.add_argument("--data-dir", default="/share/latent/benchmarks/LongBench/data")
    ap.add_argument("--runs-dir", default=str(ROOT / "runs/longbench_prlmc_v7_eval"))
    ap.add_argument("--model", default="/share/latent/MemoryLLM-8B")
    ap.add_argument("--window-blocks", type=int, default=13)
    args = ap.parse_args()
    wanted = [int(x) for x in args.indices.split(",")]

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tools"))
    from transformers import AutoTokenizer
    from longbench_memoryllm_eval import load_rows, split_prompt

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = load_rows(args.data_dir, args.task)
    runs = {}
    for arm in ("Selection", "Full", "SharedMerge"):
        path = Path(args.runs_dir, "%s_%s.json" % (arm, args.task))
        data = json.loads(path.read_text())
        runs[arm] = {(s["task"], s["id"], s["sample_index"]): s for s in data["samples"]}

    for index in wanted:
        row = rows[index]
        prompt, all_ids, write_ids, _ = split_prompt(
            tokenizer, args.task, row, TOTAL, tail_tokens=TAIL)
        n_blocks = len(range(0, len(write_ids), BLOCK))
        dropped = len(tokenizer(prompt, add_special_tokens=False).input_ids) - len(all_ids)
        gold = row["answers"][0]
        gold_spans = []
        for form in answer_forms(gold):
            gold_spans = sentence_spans(tokenizer, prompt, form, dropped)
            if gold_spans:
                break
        gold_blocks = set()
        for span in gold_spans:
            gold_blocks |= blocks_of(span, len(write_ids))

        print("=" * 96)
        print("idx %d  gold=%r  blocks=%d  evidence blocks=%s"
              % (index, gold, n_blocks, sorted(gold_blocks) if gold_spans else "NOT FOUND"))
        key = (args.task, row.get("_id"), index)
        for arm in ("Selection", "SharedMerge", "Full"):
            sample = runs[arm].get(key)
            if sample is None:
                print("  %-12s missing" % arm)
                continue
            if arm == "Full":            # no compaction: every block is resident
                resident, sample_resident = set(range(n_blocks)), n_blocks
            else:
                resident = set(replay(sample, args.window_blocks, n_blocks))
                sample_resident = len(resident)
            status = "evidence 保留" if gold_blocks & resident else (
                "evidence 已丢" if gold_blocks else "evidence 未定位")
            pred = (sample["prediction"] or "").split("\n")[0].strip()[:38]
            print("  %-12s f1=%.2f resident=%2d  %-14s | pred=%r"
                  % (arm, sample["official_qa_f1_truncated"], sample_resident, status, pred))
            if arm == "SharedMerge":
                seen = absorbed(gold_blocks, sample, args.window_blocks, n_blocks)
                print("               融合窗口吸收 evidence：%s"
                      % (["write%d:%s" % (w, b) for w, b in seen] or "从未进入任何窗口"))
            if arm == "Full":
                continue
            gold_kept = sum(1 for s in gold_spans if blocks_of(s, len(write_ids)) & resident)
            print("               保留的 gold 证据句 %d/%d 处" % (gold_kept, len(gold_spans)))
            pred_spans = []
            for form in answer_forms(pred) if pred else []:
                pred_spans = sentence_spans(tokenizer, prompt, form, dropped)
                if pred_spans:
                    break
            pred_kept = sum(1 for s in pred_spans if blocks_of(s, len(write_ids)) & resident)
            print("               保留的 pred 来源句 %d/%d 处" % (pred_kept, len(pred_spans)))
            if pred_spans:
                pred_blocks = set()
                for span in pred_spans:
                    pred_blocks |= blocks_of(span, len(write_ids))
                print("               pred 原文出现在 block %s，其中保留 %s"
                      % (sorted(pred_blocks), sorted(pred_blocks & resident)))


if __name__ == "__main__":
    main()
