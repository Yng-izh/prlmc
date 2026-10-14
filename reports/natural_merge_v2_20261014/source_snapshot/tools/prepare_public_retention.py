#!/usr/bin/env python3
"""Prepare immutable original SQuAD/NQ retention inputs; never invent facts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

AUTHOR_REVISION = "5acaf52d2e60087f6bc38a06063d7558ab9a2c2c"
DATA_REPO = "YuWangX/KnowledgeRetention"
PROCESSED_REPO = "YuWangX/KnowledgeRetentionProcessed"
REVISIONS = {DATA_REPO: "c609067a4b7464dc4bf949cbb31996e47dd9be5a",
             PROCESSED_REPO: "03a8db0d2d9ffd2caecae8f5f7af09294ece6e96"}
FILES = {
    "squad": ("dev-v2.0.json", "train-v2.0.json", "indices_squad_3.npy"),
    "naturalqa": ("v1.0-simplified_nq-dev-all.jsonl",
                  "v1.0-simplified_simplified-nq-train.jsonl", "indices_nq_4.npy"),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def write_new(path, payload):
    # Exclusive creation keeps a prepared protocol from being silently replaced.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)


def tokenizer_manifest(path):
    path = Path(path)
    names = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json",
             "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt")
    result = {name: sha(path / name) for name in names if (path / name).is_file()}
    if not result or not any(name in result for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        raise ValueError("local tokenizer assets are required")
    return result


def squad_rows(path, evaluation):
    for article in json.loads(Path(path).read_text())["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            if not isinstance(context, str) or not context:
                raise ValueError("empty original paragraph")
            if not evaluation:
                yield dict(context=context)
                continue
            for qa in paragraph["qas"]:
                if qa["is_impossible"] or not qa["answers"] or not qa["answers"][0]["text"]:
                    continue  # Exactly the author's answerable-data filter.
                yield dict(id=str(qa["id"]), context=context, question=qa["question"],
                           answer=qa["answers"][0]["text"],
                           original_answers=qa["answers"], extraction="original paragraph")


def nq_span(row, span, mode):
    start, end = span["start_token"], span["end_token"]
    if "document_tokens" in row:
        tokens = row["document_tokens"]
        if not 0 <= start < end <= len(tokens):
            raise ValueError("invalid NQ token span")
        return " ".join(t["token"] for t in tokens[start:end] if not t["html_token"]), "document_tokens"
    if mode == "author512":
        raise ValueError("author NQ loader slices document_text by character using token offsets; "
                         "refuse to call this faithful original text. Use lossless mode.")
    tokens = row["document_text"].split(" ")
    if not 0 <= start < end <= len(tokens):
        raise ValueError("invalid NQ document_text token span")
    return " ".join(t for t in tokens[start:end] if not (t.startswith("<") and t.endswith(">"))), "document_text token-offset adapter"


def nq_rows(path, evaluation, mode):
    with Path(path).open() as f:
        for line_no, line in enumerate(f):
            row = json.loads(line)
            annotation = row["annotations"][0]
            if annotation["yes_no_answer"] == "None":
                continue  # Literal author predicate; not silently changed to NONE.
            if evaluation and not annotation["short_answers"]:
                continue
            context, extraction = nq_span(row, annotation["long_answer"], mode)
            if not context:
                raise ValueError("empty NQ original long-answer span")
            result = dict(context=context, extraction=extraction, source_line=line_no)
            if evaluation:
                answer, _ = nq_span(row, annotation["short_answers"][0], mode)
                if not answer:
                    raise ValueError("empty NQ gold span")
                result.update(id=str(row["example_id"]), question=row["question_text"],
                              answer=answer, original_annotations=row["annotations"])
            yield result


def choose_indices(rows, indices):
    wanted = set(indices)
    found = {}
    for i, row in enumerate(rows):
        if i in wanted:
            found[i] = row
        if len(found) == len(wanted):
            break
    if len(found) != len(wanted):
        raise ValueError("official index outside filtered dataset")
    return [dict(found[i], filtered_index=i, sample_index=j) for j, i in enumerate(indices)]


def group_distractors(rows, count, tokenize, task):
    if count == 0:
        return []
    groups = []
    limit = 768 if task == "squad" else 512
    for row in rows:
        text = row["context"]
        if groups and (len(tokenize(groups[-1]["text"])) < limit if task == "squad"
                       else len(tokenize(groups[-1]["text"])) <= limit):
            groups[-1]["text"] += " " + text
            groups[-1]["sources"].append(row)
        else:
            if len(groups) == count:
                break
            groups.append(dict(text=text, sources=[row]))
    if len(groups) != count:
        raise ValueError("not enough original training distractors")
    return groups


def processed_inputs(dev_path, train_path, num_samples, unrelated_contexts):
    """8B author release is already filtered; keep its order and exact fields."""
    dev = json.loads(Path(dev_path).read_text())
    train = json.loads(Path(train_path).read_text())
    if not isinstance(dev, list) or not 0 < num_samples <= len(dev):
        raise ValueError("invalid processed public QA count")
    if not isinstance(train, list) or not 0 <= unrelated_contexts <= len(train):
        raise ValueError("invalid processed public distractor count")
    selected = []
    for i, row in enumerate(dev[:num_samples]):
        if any(not isinstance(row.get(k), str) or not row[k] for k in ("context", "question", "answer")):
            raise ValueError("processed release is missing original text/QA fields")
        selected.append(dict(row, id=str(row.get("id", "published-index-" + str(i))),
            filtered_index=i, sample_index=i, extraction="unchanged author processed release fields"))
    groups = []
    for i, row in enumerate(train[:unrelated_contexts]):
        if not isinstance(row, str) or not row:
            raise ValueError("processed distractor must be an original published string")
        groups.append(dict(text=row, sources=[dict(published_index=i, text_sha256=digest(row))]))
    return selected, groups


def make_writes(text, role, tokenize, decode, mode):
    if mode == "author512":
        ids = tokenize(text)
        if role == "unrelated":
            # Author loader decode/re-encodes before collate truncation.
            ids = tokenize(decode(ids[:512]))
        used = ids[:512]
        return [dict(role=role, token_ids=used, token_sha256=digest(used),
                     raw_tokens=len(ids), truncated_tokens=max(0, len(ids) - 512))]
    from native_qa_compare import split_context
    chunks = split_context(text, lambda s: len(tokenize(s)), lambda s: s, 512)
    if "".join(chunks) != text:
        raise ValueError("lossless text coverage failed")
    writes = []
    offset = 0
    for chunk in chunks:
        ids = tokenize(chunk)
        if not ids or len(ids) > 512:
            raise ValueError("invalid original chunk")
        writes.append(dict(role=role, token_ids=ids, token_sha256=digest(ids),
                           char_start=offset, char_end=offset + len(chunk),
                           raw_tokens=len(ids), truncated_tokens=0))
        offset += len(chunk)
    return writes


def prepare(args):
    import numpy as np
    from transformers import AutoTokenizer
    root = Path(args.dataset_dir)
    provenance = json.loads((root / "public_dataset_manifest.json").read_text())
    if provenance["repo"] not in REVISIONS or not re.fullmatch(r"[0-9a-f]{40}", provenance["revision"]):
        raise ValueError("pinned public dataset provenance required")
    processed = provenance["repo"] == PROCESSED_REPO
    subdir = ("nqa" if processed else "nq") if args.task == "naturalqa" else "squad"
    names = ("val_data.json", "unrelated_contexts_from_train.json") if processed else FILES[args.task]
    paths = [root / subdir / name for name in names]
    for path in paths:
        if provenance["files"].get(str(path.relative_to(root))) != sha(path):
            raise ValueError("dataset file differs from pinned download")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenize = lambda s: tokenizer(s, add_special_tokens=False).input_ids
    if processed:
        selected, groups = processed_inputs(paths[0], paths[1], args.num_samples, args.unrelated_contexts)
        chosen = list(range(args.num_samples))
    else:
        indices = np.load(paths[2], allow_pickle=False)
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or not 0 < args.num_samples <= len(indices):
            raise ValueError("invalid fixed index array/sample count")
        chosen = [int(x) for x in indices[:args.num_samples]]
        if min(chosen) < 0 or len(set(chosen)) != len(chosen):
            raise ValueError("invalid/duplicate official indices")
        rows = (lambda p, ev: squad_rows(p, ev)) if args.task == "squad" else (lambda p, ev: nq_rows(p, ev, args.mode))
        selected = choose_indices(rows(paths[0], True), chosen)
        groups = group_distractors(rows(paths[1], False), args.unrelated_contexts, tokenize, args.task)
    unrelated = []
    for i, group in enumerate(groups):
        unrelated.extend([dict(w, distractor_group=i) for w in make_writes(
            group["text"], "unrelated", tokenize, tokenizer.decode, args.mode)])
    for sample in selected:
        sample["context_sha256"] = digest(sample["context"])
        prompt = "Questions: " + sample["question"] + ("? Answer:" if args.task == "naturalqa" else " Answer:")
        ids = tokenize(prompt)
        if not ids or len(ids) > 512 or len(tokenize(sample["answer"] + "</s>")) > 512:
            raise ValueError("QA would be truncated; refuse silent changes to original question/gold")
        sample.update(prompt=prompt, prompt_ids=ids,
                      related_writes=make_writes("Context: " + sample["context"], "related", tokenize,
                                               tokenizer.decode, args.mode))
    plan = dict(schema="public_retention/1", status="prepared", task=args.task, mode=args.mode,
        data_variant="author 8B answerability-filtered release" if processed else "author original fixed indices",
        dataset=provenance, original_files={str(p.relative_to(root)): sha(p) for p in paths},
        tokenizer=tokenizer_manifest(args.model_path), official_indices=chosen,
        author_revision=AUTHOR_REVISION, samples=selected, distractor_groups=groups,
        unrelated_writes=unrelated, protocol=dict(num_unrelated_contexts=args.unrelated_contexts,
            split="published filtered validation order; published train distractors" if processed else "official fixed development indices; train distractors in original order",
            max_new_tokens=10, do_sample=False, stop_strings=[], score="author case-sensitive answer containment",
            reset="restore checkpoint memory and initialized before every sample and arm",
            budget_blocks=20, window_blocks=19, target_blocks=9, anchor_blocks=0,
            arms=["Full", "Selection", "Recent", "SharedE6d", "NativeDefault"],
            gamma_max=.05, rank=32, training=False,
            input_policy="author512 truncation" if args.mode == "author512" else "complete extracted text; lossless character chunks",
            deviations=["explicit independent-sample checkpoint reset", "new managed-budget compression arms",
                        "readout after every chunk; no prompt/answer truncation"] +
                       (["complete text instead of author truncation", "NQ document_text spans use token offsets"] if args.mode == "lossless" else [])))
    plan["content_sha256"] = digest(plan)
    write_new(args.out, plan)
    print(json.dumps(dict(status="prepared", samples=len(selected), unrelated_writes=len(unrelated),
                          content_sha256=plan["content_sha256"])))


def download(args):
    from huggingface_hub import HfApi, hf_hub_download
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision or ""):
        raise ValueError("--revision must be an explicit dataset commit SHA; never floating main")
    repo = args.repo
    info = HfApi().dataset_info(repo, revision=args.revision)
    if info.sha != args.revision:
        raise ValueError("dataset revision resolution mismatch")
    names = {s.rfilename for s in info.siblings}
    expected = ([f"{'nqa' if task == 'naturalqa' else task}/{name}" for task in FILES
                for name in ("val_data.json", "unrelated_contexts_from_train.json")] if repo == PROCESSED_REPO else
                [f"{'nq' if task == 'naturalqa' else task}/{name}" for task in FILES for name in FILES[task]])
    # Some snapshots package all files under data/; record original repo paths.
    selected = {}
    for local_name in expected:
        matches = [n for n in names if n == local_name or n.endswith("/" + local_name)]
        if len(matches) != 1:
            raise ValueError("required official asset missing/ambiguous: " + local_name)
        selected[local_name] = matches[0]
    import shutil
    root = Path(args.dataset_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any((root / n).exists() for n in expected) or (root / "public_dataset_manifest.json").exists():
        raise ValueError("download destination contains existing assets; use an empty directory")
    hashes = {}
    for local_name, remote_name in selected.items():
        cached = hf_hub_download(repo, remote_name, repo_type="dataset", revision=args.revision,
                                 cache_dir=str(root / ".download_cache"))
        target = root / local_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, target)
        hashes[local_name] = sha(target)
    write_new(root / "public_dataset_manifest.json", dict(repo=repo, revision=args.revision,
                source_paths=selected, files=hashes, license=getattr(info, "card_data", None).get("license") if getattr(info, "card_data", None) else None))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("download", "prepare"))
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--revision")
    p.add_argument("--repo", choices=tuple(REVISIONS), default=DATA_REPO)
    p.add_argument("--task", choices=tuple(FILES))
    p.add_argument("--model-path")
    p.add_argument("--mode", choices=("author512", "lossless"), default="lossless")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--unrelated-contexts", type=int, default=30)
    p.add_argument("--out")
    a = p.parse_args()
    if a.unrelated_contexts < 0:
        p.error("unrelated-contexts must be nonnegative")
    if a.action == "prepare" and not all((a.task, a.model_path, a.out)):
        p.error("prepare requires task, model-path and out")
    (download if a.action == "download" else prepare)(a)


if __name__ == "__main__":
    main()
