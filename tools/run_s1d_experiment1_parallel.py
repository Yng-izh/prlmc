#!/usr/bin/env python3
"""Eight isolated GPU jobs, smoke gate, strict cross-arm checks, durable status."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEVICES = (0, 1, 2, 3, 4, 6, 8, 9)
ASSIGNMENTS = [(arm, (3, start), device)
               for start, devices in ((16370, DEVICES[:4]), (17510, DEVICES[4:]))
               for arm, device in zip('ABCD', devices)]
FACT_CACHE = {
    16370: '/share/latent/PRLM-C/runs/s1d_exp1_fact_v1.smoke.C.w16370.fact_rows.pt',
    17510: '/share/latent/PRLM-C/runs/s1d_exp1_fact_v1.smoke.C.w17510.fact_rows.pt',
}


def atomic_json(path, value):
    # Both host schedulers write the shared GPFS state. A process-specific
    # temporary path prevents one writer from replacing another writer's tmp.
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def verify_reports(paths, steps):
    grouped = {}
    for path in paths:
        r = json.loads(Path(path).read_text())
        assert r['status'] == 'complete', (path, r.get('failure'))
        assert r['protocol']['steps'] == steps
        assert r['protocol']['evaluation_qa_used_for_training'] is False
        assert len(r['windows']) == 1
        w = r['windows'][0]; name = r['protocol']['exp1_arm']
        assert set(w['arms']) == {name}, 'empty/wrong arm'
        a = w['arms'][name]; f = a['final']
        assert a['trainable_dof'] == 13 and a['initial_identity_verified']
        assert f['step'] == steps and f['basis_unchanged']
        assert 0 < f['deviation'] <= r['protocol']['budget'] + 1e-6
        assert math.isfinite(f['sup_loss']) and len(f['alpha']) == 13
        assert f['net_vs_d0'] == f['new_recovery'] - f['d0_harmed']
        assert Path(a['residual_checkpoint']).exists()
        if name in 'BCD':
            assert len(a['fact_target_keys']) + len(a['protection_fact_keys']) <= 40
            for key in a['basis_fact_keys']:
                assert a['fact_full_ce'][key] <= r['protocol']['fact_full_ce_max']
                assert a['fact_d0_ce'][key] - a['fact_full_ce'][key] >= r['protocol']['fact_delta_threshold']
            assert set(a['fact_target_keys']).isdisjoint(a['protection_fact_keys'])
            assert [x['key'] for x in a['protection_fact_records']] == a['protection_fact_keys']
            assert [x['key'] for x in a['basis_fact_records']] == a['basis_fact_keys']
            assert all(x['block'] not in a['dropped_blocks'] for x in a['protection_fact_records'])
            assert all(x['block'] in a['dropped_blocks'] for x in a['basis_fact_records'])
            assert all(a['fact_d0_ce'][key] <= r['protocol']['fact_full_ce_max']
                       for key in a['protection_fact_keys'])
        if name in 'CD':
            assert a['fact_basis_provenance'] == 'fact_answer_gradients_cached'
        grouped.setdefault(tuple(w['window']), {})[name] = (r, w, a)
    assert set(grouped) == {(3, 16370), (3, 17510)}
    for window, arms in grouped.items():
        assert set(arms) == set('ABCD')
        base = arms['A'][1]
        for name, (_, w, _) in arms.items():
            assert w['sets'] == base['sets'], (window, name, 'cohort drift')
            for b in ('full', 'select', 'd0'):
                assert w['baselines'][b] == base['baselines'][b], (window, name, b)
            assert arms[name][0]['driver_sha256'] == arms['A'][0]['driver_sha256']
        assert arms['A'][2]['basis_sha256'] == arms['B'][2]['basis_sha256']
        assert arms['C'][2]['basis_sha256'] == arms['D'][2]['basis_sha256']
        for name in 'CD':
            assert arms[name][2]['fact_target_keys'] == arms['B'][2]['fact_target_keys']
            assert arms[name][2]['protection_fact_keys'] == arms['B'][2]['protection_fact_keys']
    return {'valid': True, 'windows': [list(w) for w in sorted(grouped)],
            'reports': [str(p) for p in paths], 'steps': steps}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--devices', default=','.join(map(str, DEVICES)))
    p.add_argument('--cache-dir', default='/dev/shm/s1d_reader_fp16')
    p.add_argument('--prefix', default='s1d_experiment1')
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--measure-every', type=int, default=10)
    p.add_argument('--smoke-first', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if tuple(map(int, args.devices.split(','))) != DEVICES:
        raise SystemExit('Use GPU 0,1,2,3,4,6,8,9; leave 5,7 alone')
    marker = Path(args.cache_dir) / 'fp16_cache_manifest.json'
    if not marker.exists():
        raise SystemExit('verified cache missing: ' + str(marker))
    directory = ROOT / 'runs'; directory.mkdir(exist_ok=True)
    state_path = directory / (args.prefix + '.launcher.json')
    if state_path.exists():
        raise SystemExit('launcher state already exists')
    state = {'status': 'starting', 'created_utc': datetime.now(timezone.utc).isoformat(),
             'driver_sha256': hashlib.sha256((ROOT/'tools/s1d_experiment1.py').read_bytes()).hexdigest(),
             'devices': list(DEVICES), 'pid': os.getpid(), 'stages': {}}
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    def run_stage(label, steps, cadence):
        prefix = args.prefix + ('.smoke' if label == 'smoke' else '')
        jobs = []
        for arm, window, gpu in ASSIGNMENTS:
            stem = '%s.%s.w%d' % (prefix, arm, window[1])
            out = directory / (stem+'.json'); log = directory / (stem+'.log')
            if out.exists() or log.exists():
                raise RuntimeError('output exists: ' + stem)
            command = [sys.executable, str(ROOT/'tools/s1d_experiment1.py'),
                '--exp1-arm', arm, '--window', '%d,%d' % window,
                '--cuda-device', str(gpu), '--model-path', args.cache_dir,
                '--out', str(out), '--steps', str(steps), '--measure-every', str(cadence),
                '--skip-generation', '--qa-max-tokens', '256', '--fact-max', '40',
                '--fact-delta-threshold', '0.10', '--fact-full-ce-max', '2.0']
            if arm in 'BCD':
                command.extend(['--fact-cache', FACT_CACHE[window[1]]])
            jobs.append({'arm': arm, 'window': list(window), 'gpu': gpu,
                         'report': str(out), 'log': str(log), 'command': command})
        if args.dry_run:
            print(json.dumps(jobs, indent=2)); return
        # Check actual free memory before allocating a replicated reader.
        usage = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                    '--format=csv,noheader,nounits'], text=True)
        free = {int(i): int(m) for i, m in (line.split(',') for line in usage.splitlines())}
        if any(free[g] < 30000 for g in DEVICES):
            raise RuntimeError('one of the requested GPUs is occupied: '+str(free))
        handles = []
        state['status'] = label + '_running'; state['stages'][label] = jobs
        for job in jobs:
            stream = open(job['log'], 'w')
            child = subprocess.Popen(job['command'], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
            handles.append((job, child, stream)); job['pid'] = child.pid
            print('launched %s %s GPU%d pid=%d' %
                  (label, job['arm']+str(job['window']), job['gpu'], child.pid), flush=True)
        atomic_json(state_path, state)
        pending = list(handles)
        while pending:
            for item in pending[:]:
                job, child, stream = item
                code = child.poll()
                if code is None:
                    continue
                stream.close(); job['exit_code'] = code; pending.remove(item)
                atomic_json(state_path, state)
                print('finished %s %s rc=%d' % (label, job['arm']+str(job['window']), code), flush=True)
            if pending:
                time.sleep(5)
        if any(j['exit_code'] != 0 for j in jobs):
            raise RuntimeError(label+' worker failure; see individual logs')
        certificate = verify_reports([j['report'] for j in jobs], steps)
        atomic_json(directory/(prefix+'.verification.json'), certificate)
        state['status'] = label+'_verified'; atomic_json(state_path, state)
    try:
        if args.smoke_first:
            run_stage('smoke', 1, 1)
        run_stage('formal', args.steps, args.measure_every)
        if not args.dry_run:
            state['status'] = 'complete'; atomic_json(state_path, state)
            print('all eight formal runs verified', flush=True)
    except Exception as exc:
        if not args.dry_run:
            state.update(status='failed', failure=repr(exc)); atomic_json(state_path, state)
        raise

if __name__ == '__main__':
    main()
