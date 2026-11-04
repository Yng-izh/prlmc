#!/usr/bin/env python3
"""File identity and durable JSON writes shared by the training and eval paths.

`sha` hashes a file, `digest` hashes a JSON-serialisable value (sorted keys, no
whitespace) so a record can be sealed against later edits, `atomic_json` writes
a report without ever truncating a previous one, and `tokenizer_manifest` pins
the local tokenizer assets a run was prepared against.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, payload):
    # Serialize first: a non-serializable field never truncates an old result.
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def tokenizer_manifest(path):
    path = Path(path)
    names = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json",
             "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt")
    result = {name: sha(path / name) for name in names if (path / name).is_file()}
    if not result or not any(name in result for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        raise ValueError("local tokenizer assets are required")
    return result
