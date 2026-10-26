#!/usr/bin/env python3
"""Build HotpotQA-train supervised episodes for the v8 plan (Edit 3, part 3).

Source: the official HotpotQA *distractor* TRAIN split, relayed via hf-mirror
(`train-0000{0,1}-of-00002.parquet`, already on the server at /tmp). The
LongBench hotpotqa.jsonl TEST file is never read here; nothing from any test
split enters training.

Rules honoured:
- questions are training supervision only: each episode writes its contexts
  through the same real long-text pipeline (`make_writes(..., "author512")`)
  and no question text ever enters the written memory;
- one training QA per context (the official answer; no invented answers);
- context text is rendered exactly like LongBench renders a hotpotqa context
  (titled paragraphs, "\n\n" between passages) so the same knowledge the
  evaluation-time reader sees is what training writes encode;
- context length is capped at 39 contexts per episode, mirroring the sealed
  corpus stream length (39 x ~170 tok writes ~= 20 blocks at trigger 20).

Output: data/natural_merge_v8/hotpot_train_episodes.json with
  episodes: list of {id, contexts, writes} in sealed-plan episode schema
  (QA fields: prompt/prompt_ids/answer/answer_ids/content_ids/... matching
  encode_context + the v4 leading-space answer transform).
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from prepare_public_retention import digest, sha  # noqa: E402


SCHEMA = "natural_merge_v8_hotpot_episodes/1"


def encode_hotpot_context(title, paragraphs, source_index, tokenizer, question, answer):
    """One original context: titled-paragraph rendering, one official QA."""
    context = "\n\n".join(
        f"{t}\n{''.join(sents)}" for t, sents in zip(title, paragraphs))
    context_sha256 = digest(context)
    prompt = "Questions: " + question + " Answer:"
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    content_ids = tokenizer(" " + answer, add_special_tokens=False).input_ids
    answer_ids = content_ids + [tokenizer.eos_token_id]
    if not prompt_ids or not answer_ids or len(prompt_ids) + len(answer_ids) > 512:
        raise ValueError("hotpot QA exceeds the fixed reader input budget")
    return dict(
        article=title[0], article_sha256=digest(title[0]),
        context=context, context_sha256=context_sha256,
        qas=[dict(
            id=None,  # assigned by the caller after dedup
            question=question, answer=answer,
            answers=[answer], original_answers=[dict(answer_start=-1, text=answer)],
            prompt=prompt, prompt_ids=prompt_ids,
            prompt_sha256=digest(prompt_ids),
            answer_ids=answer_ids, answer_sha256=digest(answer_ids),
            content_ids=content_ids, content_sha256=digest(content_ids),
            scoring_target=tokenizer.decode(answer_ids).replace("</s>", "").strip(),
            training_answer_text=" " + answer,
            source_index=source_index,
            context_sha256=context_sha256)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquets", nargs="+", required=True)
    p.add_argument("--model-path", default="/dev/shm/s1d_reader_fp16")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--contexts-per-episode", type=int, default=39)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from prepare_public_retention import make_writes

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

    rows = []
    for path in args.parquets:
        rows.extend(pq.read_table(path).to_pylist())
    print(f"loaded {len(rows)} official HotpotQA train rows")
    if len(rows) != 90447:
        raise ValueError("official distractor train split must have 90447 rows")

    # Deterministic sample: comparable multi-hop mix, no test contamination.
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    wanted = args.episodes * args.contexts_per_episode
    seen_ids, chosen = set(), []
    for row in rows:
        if len(chosen) >= wanted:
            break
        if row["id"] in seen_ids or not row["answer"].strip() or not row["question"].strip():
            continue
        title, sentences = row["context"]["title"], row["context"]["sentences"]
        if not title or any(not s for s in sentences):
            continue
        seen_ids.add(row["id"])
        chosen.append(row)
    if len(chosen) < wanted:
        raise ValueError("not enough valid official train rows sampled")

    episodes = []
    for e in range(args.episodes):
        chunk = chosen[e * args.contexts_per_episode:(e + 1) * args.contexts_per_episode]
        contexts, writes = [], []
        for i, row in enumerate(chunk):
            context = encode_hotpot_context(
                row["context"]["title"], row["context"]["sentences"], i, tokenizer,
                row["question"], row["answer"])
            context["qas"][0]["id"] = "hotpot-" + row["id"]
            qa = context["qas"][0]
            contexts.append(context)
            tokenize = lambda text: tokenizer(text, add_special_tokens=False).input_ids
            write = make_writes("Context: " + context["context"], "related",
                                tokenize, tokenizer.decode, "author512")[0]
            write.update(context_sha256=context["context_sha256"], source_index=i)
            writes.append(write)
        episodes.append(dict(id=f"hotpot-{e}", contexts=contexts, writes=writes))

    # QA id uniqueness against the sealed v4 corpus (the v8 plan splices both).
    sealed = json.loads((ROOT / "data" / "natural_merge_v7" / "train.json").read_text())
    sealed_ids = {q["id"] for ep in sealed["episodes"] for c in ep["contexts"] for q in c["qas"]}
    new_ids = {q["id"] for ep in episodes for c in ep["contexts"] for q in c["qas"]}
    if sealed_ids & new_ids:
        raise ValueError("hotpot QA ids collide with the sealed corpus")

    payload = dict(schema=SCHEMA,
                   dataset="official HotpotQA distractor train split (90447 rows; parquet sha256 recorded)",
                   parquet_sha256={Path(p).name: sha(p) for p in args.parquets},
                   seed=args.seed, episodes=episodes,
                   provenance=dict(source="hotpotqa/hotpot_qa distractor train via hf-mirror",
                                   usage="training supervision only; questions never pre-fed to"
                                         " writer/selection; LongBench test data excluded",
                                   rows_total=len(rows), episodes_used=args.episodes,
                                   contexts_per_episode=args.contexts_per_episode))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}: {len(episodes)} episodes x {args.contexts_per_episode} contexts,"
          f" unique QA ids {len(new_ids)}")


if __name__ == "__main__":
    main()
