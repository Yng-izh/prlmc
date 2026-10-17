#!/usr/bin/env python3
"""Stage the exact float16 reader weights in shared RAM once for many ranks."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    source, destination = Path(args.source), Path(args.destination)
    marker = destination / "fp16_cache_manifest.json"
    if marker.exists():
        data = json.loads(marker.read_text())
        if data["source"] != str(source.resolve()):
            raise SystemExit("cache source mismatch")
        print("verified cache already present", flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    index = json.loads((source / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    manifest = {"source": str(source.resolve()), "target_dtype": "torch.float16",
                "equivalence": "torch.float16 conversion before storage; saved tensor digest rechecked",
                "shards": {}}
    for filename in shards:
        values, hashes = {}, {}
        with safe_open(source / filename, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            for key in handle.keys():
                original = handle.get_tensor(key)
                value = original.to(torch.float16) if original.is_floating_point() else original
                values[key] = value.contiguous()
                hashes[key] = digest(values[key])
        save_file(values, destination / filename, metadata=metadata)
        del values
        with safe_open(destination / filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if digest(handle.get_tensor(key)) != hashes[key]:
                    raise RuntimeError("converted weight mismatch: " + key)
        manifest["shards"][filename] = hashes
        print("verified FP16 cache shard " + filename, flush=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix in (".json", ".md"):
            shutil.copy2(path, destination / path.name)
    marker.write_text(json.dumps(manifest, indent=2))
    print("FP16 shared cache complete", flush=True)


if __name__ == "__main__":
    main()
