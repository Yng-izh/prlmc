#!/usr/bin/env python3
"""Pack 64k QA streams for the Reader LoRA / V9 retrain.

Each stream is a concatenation of many natural records (SQuAD 2.0 train answerable
items and HotpotQA distractor train supporting paragraphs) until the packed text
reaches --pack-tokens, then one included record's QA is used as supervision. The
answer is guaranteed to occur inside the stream because its own paragraph is part of
it; questions and answers are never written into memory.

Output JSONL: {"doc_id", "context", "question", "answers": [str]}
"""
import argparse, glob, json, random
from pathlib import Path


def iter_squad(path):
    data = json.loads(Path(path).read_text())["data"]
    for article in data:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                if qa.get("is_impossible") or not qa["answers"]:
                    continue
                text = qa["answers"][0]["text"]
                if not text.strip():
                    continue
                yield dict(doc_id="squad::" + str(qa["id"]), context=para["context"],
                           question=qa["question"], answers=[text])


def _hotpot_paragraphs(row):
    ctx = row["context"]
    if isinstance(ctx, dict):
        return {t: "".join(s) for t, s in zip(ctx["title"], ctx["sentences"])}
    return {t: "".join(s) for t, s in ctx}


def _hotpot_titles(row):
    sf = row["supporting_facts"]
    if isinstance(sf, dict):
        return list(sf.get("title", []))
    return [item[0] if isinstance(item, (list, tuple)) else item for item in sf]


def iter_hotpot(pattern):
    import pyarrow.parquet as pq
    for path in sorted(glob.glob(pattern)):
        handle = pq.ParquetFile(path)
        for batch in handle.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                para = _hotpot_paragraphs(row)
                titles = [t for t in _hotpot_titles(row) if t in para]
                if not titles:
                    continue
                context = "\n\n".join(para[t] for t in dict.fromkeys(titles))
                answer = row["answer"]
                if not str(answer).strip() or str(answer).lower() not in context.lower():
                    continue
                rid = row.get("_id") or row.get("id")
                yield dict(doc_id="hotpot::" + str(rid), context=context,
                           question=row["question"], answers=[str(answer)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad", required=True)
    ap.add_argument("--hotpot-glob", required=True)
    ap.add_argument("--tokenizer", default="/dev/shm/s1d_reader_fp16")
    ap.add_argument("--pack-tokens", type=int, default=64000)
    ap.add_argument("--streams", type=int, default=700)
    ap.add_argument("--val-streams", type=int, default=20)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--val-out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rng = random.Random(args.seed)

    records = []
    for rec in iter_squad(args.squad):
        records.append(rec)
    for rec in iter_hotpot(args.hotpot_glob):
        records.append(rec)
        if len(records) > args.streams * 400:
            break
    rng.shuffle(records)

    streams, buf, buf_tokens, index = [], [], 0, 0
    for rec in records:
        if len(streams) >= args.streams + args.val_streams:
            break
        tokens = len(tok(rec["context"], add_special_tokens=False).input_ids)
        buf.append(rec)
        buf_tokens += tokens
        if buf_tokens < args.pack_tokens:
            continue
        context = "\n\n".join(r["context"] for r in buf)
        chosen = rng.choice(buf)
        if chosen["answers"][0].lower() not in context.lower():
            chosen = next(r for r in buf if r["answers"][0].lower() in context.lower())
        index += 1
        streams.append(dict(doc_id=f"stream::{index:05d}", context=context,
                            question=chosen["question"], answers=chosen["answers"],
                            packed_records=len(buf), packed_tokens=buf_tokens))
        buf, buf_tokens = [], 0

    val, train = streams[:args.val_streams], streams[args.val_streams:]
    for path, rows in ((args.train_out, train), (args.val_out, val)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(dict(train=len(train), val=len(val), pack_tokens=args.pack_tokens,
                          mean_packed_tokens=round(sum(r["packed_tokens"] for r in streams) / max(1, len(streams)), 1))))


if __name__ == "__main__":
    main()
