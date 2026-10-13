#!/usr/bin/env python3
"""Rehash cached FP16 tensors against the original staging certificate."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',default='/dev/shm/s1d_reader_fp16')
    p.add_argument('--out',required=True);args=p.parse_args()
    torch.set_num_threads(4)
    cache=Path(args.cache); marker=cache/'fp16_cache_manifest.json'
    manifest=json.loads(marker.read_text()); verified=0
    for name,expected in manifest['shards'].items():
        with safe_open(cache/name,framework='pt',device='cpu') as f:
            if set(f.keys())!=set(expected):raise ValueError('tensor keys mismatch: '+name)
            for key,sha in expected.items():
                tensor=f.get_tensor(key)
                actual=hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
                if actual!=sha:raise ValueError('tensor digest mismatch: '+key)
                verified+=1
        print('verified',name,flush=True)
    result=dict(valid=True,verified_tensors=verified,cache=str(cache),
                manifest_sha256=hashlib.sha256(marker.read_bytes()).hexdigest())
    Path(args.out).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
