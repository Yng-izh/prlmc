#!/usr/bin/env python3
"""Download pinned original SQuAD assets, verifying upstream Git/LFS identities."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_public_retention import DATA_REPO, REVISIONS, sha, write_new

NAMES = ("squad/train-v2.0.json", "squad/dev-v2.0.json", "squad/indices_squad_3.npy")


def upstream(endpoint):
    revision = REVISIONS[DATA_REPO]
    base = endpoint.rstrip("/")
    with urllib.request.urlopen(base + "/api/datasets/" + DATA_REPO + "/revision/" + revision, timeout=30) as response:
        info = json.load(response)
    if info["sha"] != revision:
        raise ValueError("upstream resolved a different dataset revision")
    with urllib.request.urlopen(base + "/api/datasets/" + DATA_REPO + "/tree/" + revision + "/squad", timeout=30) as response:
        entries = json.load(response)
    metadata = {row["path"]: row for row in entries if row["path"] in NAMES}
    if set(metadata) != set(NAMES):
        raise ValueError("pinned original assets missing from upstream tree")
    return info, metadata


def verify_file(path, metadata):
    if path.stat().st_size != metadata["size"]:
        raise ValueError("upstream byte-size mismatch: " + str(path))
    if "lfs" in metadata:
        if sha(path) != metadata["lfs"]["oid"]:
            raise ValueError("upstream LFS SHA256 mismatch: " + str(path))
    else:
        h = hashlib.sha1(("blob " + str(path.stat().st_size) + "\0").encode())
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != metadata["oid"]:
            raise ValueError("upstream Git blob identity mismatch: " + str(path))
    return sha(path)


def run(args):
    root = Path(args.out_dir)
    info, metadata = upstream(args.endpoint)
    existing = root / "public_dataset_manifest.json"
    if existing.exists():
        manifest = json.loads(existing.read_text())
        if manifest["repo"] != DATA_REPO or manifest["revision"] != REVISIONS[DATA_REPO]:
            raise ValueError("destination holds a different dataset")
        for name in NAMES:
            if verify_file(root / name, metadata[name]) != manifest["files"][name]:
                raise ValueError("existing manifest differs from verified asset")
        print(json.dumps(dict(status="already_verified", files=manifest["files"])), flush=True)
        return
    hashes = {}
    for name in NAMES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            partial = target.with_suffix(target.suffix + ".download")
            url = args.endpoint.rstrip("/") + "/datasets/" + DATA_REPO + "/resolve/" + REVISIONS[DATA_REPO] + "/" + name
            failure = None
            for attempt in range(3):
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": "PRLM-C-pinned-data/1"})
                    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
                        total = 0
                        while True:
                            chunk = response.read(1 << 20)
                            if not chunk:
                                break
                            output.write(chunk)
                            total += len(chunk)
                            if total % (8 << 20) == 0:
                                print(json.dumps(dict(file=name, bytes=total, expected=metadata[name]["size"])), flush=True)
                    verify_file(partial, metadata[name])
                    partial.replace(target)
                    break
                except (OSError, ValueError) as exc:
                    failure = exc
                    print(json.dumps(dict(file=name, attempt=attempt + 1, error=str(exc))), flush=True)
                    if attempt == 2:
                        raise
                    time.sleep(2)
            else:
                raise RuntimeError(str(failure))
        hashes[name] = verify_file(target, metadata[name])
        if name.endswith(".json"):
            with target.open() as f:
                data = json.load(f)
            if data["version"] != "v2.0" or not data["data"]:
                raise ValueError("unexpected original SQuAD version/schema")
        print(json.dumps(dict(file=name, status="verified", sha256=hashes[name], bytes=target.stat().st_size)), flush=True)
    manifest = dict(repo=DATA_REPO, revision=REVISIONS[DATA_REPO], source_paths={name: name for name in NAMES},
        files=hashes, scope="original SQuAD train/dev and official indices; NQ transfer reuses separately pinned processed release",
        acquisition_endpoint=args.endpoint, upstream_file_metadata=metadata,
        license=(info.get("cardData") or {}).get("license"),
        verification="pinned commit resolved; byte sizes; Git blob SHA1 or LFS SHA256; local SHA256")
    write_new(existing, manifest)
    print(json.dumps(dict(status="complete", manifest=str(existing), files=hashes)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--endpoint", default="https://huggingface.co")
    run(p.parse_args())


if __name__ == "__main__":
    main()
