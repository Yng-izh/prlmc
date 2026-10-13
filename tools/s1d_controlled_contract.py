"""Freeze/single-open and episode-level paired inference for future tests."""
import hashlib
import json
import os
from pathlib import Path
import random


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def claim_sealed_test(plan_path, frozen_path, current_manifest):
    """Called only after an explicit method freeze, before any test generation."""
    frozen=json.loads(Path(frozen_path).read_text())
    if frozen.get('status')!='frozen' or frozen.get('plan_sha256')!=file_sha(plan_path):
        raise ValueError('method/data freeze missing or changed')
    if frozen.get('manifest')!=current_manifest:
        raise ValueError('runtime differs from frozen method/protocol/code/model manifest')
    if not frozen.get('all_development_complete') or frozen.get('test_scored'):
        raise ValueError('invalid development/test freeze state')
    original=Path(plan_path).with_suffix('.test_opened.json')
    if original.exists():raise ValueError('SIC test set already scored; not independent')
    ledger=Path(plan_path).with_suffix('.s1d_test_opened.json')
    fd=os.open(ledger,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as f:
        json.dump(dict(plan_sha256=file_sha(plan_path),freeze_sha256=file_sha(frozen_path),
                       manifest=current_manifest,single_open=True),f)
        f.flush();os.fsync(f.fileno())
    return ledger


def episode_bootstrap(rows,baseline,replicates=10000,seed=20261012):
    if not rows or baseline not in ('D0','Selection'):
        raise ValueError('nonempty episode results and fixed baseline required')
    ids=[r['episode'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('duplicate episode')
    if any(r.get('status')!='complete' for r in rows):
        return dict(valid=False,reason='incomplete episodes: do not silently drop failures',
                    failed_episode_ids=[r['episode'] for r in rows if r.get('status')!='complete'])
    differences=[r['C_em']-r[baseline+'_em'] for r in rows]
    if any(not -1<=v<=1 for v in differences):raise ValueError('invalid episode EM delta')
    rng=random.Random(seed);n=len(differences)
    samples=sorted(sum(rng.choice(differences) for _ in range(n))/n for _ in range(replicates))
    return dict(valid=True,unit='episode',baseline=baseline,episodes=n,
                mean_delta=sum(differences)/n,ci95=[samples[int(.025*(replicates-1))],
                                                  samples[int(.975*(replicates-1))]],
                bootstrap_replicates=replicates,seed=seed)
