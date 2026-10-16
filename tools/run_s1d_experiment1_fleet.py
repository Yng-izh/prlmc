#!/usr/bin/env python3
"""Shared queue for isolated GPU experiment jobs on the two V100 servers."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from run_s1d_experiment1_parallel import FACT_CACHE, atomic_json, verify_reports

ROOT = Path(__file__).resolve().parents[1]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def jobs_for(prefix, stage, steps, cadence):
    jobs = []
    for arm in 'DCBA':
        for start in (16370, 17510):
            stem = '%s%s.%s.w%d' % (prefix, '.smoke' if stage == 'smoke' else '', arm, start)
            jobs.append({'id': stem, 'stage': stage, 'arm': arm, 'window': [3, start],
                         'steps': steps, 'cadence': cadence, 'status': 'queued',
                         'report': str(ROOT/'runs'/(stem+'.json')),
                         'log': str(ROOT/'runs'/(stem+'.log'))})
    return jobs


def command_for(job, gpu, cache_dir):
    command = [sys.executable, str(ROOT/'tools/s1d_experiment1.py'),
               '--exp1-arm', job['arm'], '--window', '%d,%d' % tuple(job['window']),
               '--cuda-device', str(gpu), '--model-path', cache_dir,
               '--out', job['report'], '--steps', str(job['steps']),
               '--measure-every', str(job['cadence']), '--skip-generation',
               '--qa-max-tokens', '256', '--fact-max', '40',
               '--fact-delta-threshold', '0.10', '--fact-full-ce-max', '2.0']
    if job['arm'] in 'BCD':
        command += ['--fact-cache', FACT_CACHE[job['window'][1]]]
    return command


def free_memory():
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
        text=True, timeout=30)
    return {int(i): int(m) for i, m in (line.split(',') for line in output.splitlines())}


class AdoptedProcess:
    """Minimal poll-compatible handle for a child started by a prior scheduler."""
    def __init__(self, pid):
        self.pid = pid

    def poll(self):
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return 0
        except PermissionError:
            return 1
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--host-label', required=True)
    parser.add_argument('--devices', required=True)
    parser.add_argument('--cache-dir', default='/dev/shm/s1d_reader_fp16')
    parser.add_argument('--min-free-mib', type=int, default=30000)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--measure-every', type=int, default=10)
    parser.add_argument('--init', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    devices = [int(x) for x in args.devices.split(',')]
    if len(set(devices)) != len(devices):
        parser.error('duplicate GPU indices')
    if Path(args.prefix).name != args.prefix:
        parser.error('prefix must be a simple filename')
    runs = ROOT/'runs'; runs.mkdir(exist_ok=True)
    state_path = runs/(args.prefix+'.fleet.json')
    lock_path = runs/(args.prefix+'.fleet.lock')
    driver_sha = hashlib.sha256((ROOT/'tools/s1d_experiment1.py').read_bytes()).hexdigest()

    @contextmanager
    def locked_state():
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(state_path.read_text())
            try:
                yield state
            finally:
                state['updated_utc'] = stamp()
                atomic_json(state_path, state)
                fcntl.flock(lock, fcntl.LOCK_UN)

    if args.dry_run:
        print(json.dumps([command_for(j, devices[i % len(devices)], args.cache_dir)
                          for i, j in enumerate(jobs_for(args.prefix, 'smoke', 1, 1))], indent=2))
        return
    if args.init:
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if state_path.exists():
                raise SystemExit('state already exists; reuse without --init')
            jobs = jobs_for(args.prefix, 'smoke', 1, 1)
            if any(Path(j[k]).exists() for j in jobs for k in ('report', 'log')):
                raise SystemExit('output already exists')
            atomic_json(state_path, {'status': 'smoke_running', 'created_utc': stamp(),
                'prefix': args.prefix, 'driver_sha256': driver_sha, 'hosts': {},
                'steps': args.steps, 'measure_every': args.measure_every, 'jobs': jobs})
    if not (Path(args.cache_dir)/'fp16_cache_manifest.json').exists():
        raise SystemExit('reader cache manifest missing')
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    active = {}
    print('worker started', args.host_label, devices, os.getpid(), flush=True)
    # Recover jobs orphaned by a scheduler restart. Their children are
    # intentionally left running; adopting them keeps the shared state live.
    with locked_state() as state:
        for job in state['jobs']:
            if job.get('status') != 'running' or job.get('host') != args.host_label:
                continue
            pid = job.get('pid')
            if not pid:
                continue
            try:
                os.kill(pid, 0)
            except (ProcessLookupError, PermissionError):
                continue
            stream = open(job['log'], 'a')
            active[job['id']] = (dict(job), AdoptedProcess(pid), stream)
            print('adopted', job['id'], args.host_label, 'GPU'+str(job.get('gpu')), pid, flush=True)
    while True:
        # Persist process completion before considering further placements.
        for job_id, (job, child, stream) in list(active.items()):
            code = child.poll()
            if code is None:
                continue
            stream.close()
            with locked_state() as state:
                current = next(j for j in state['jobs'] if j['id'] == job_id)
                current.update(exit_code=code, finished_utc=stamp(),
                               status='complete' if code == 0 else 'failed')
                if code != 0:
                    state.update(status='failed', failure='worker failure: '+job_id)
            del active[job_id]
            print('finished', job_id, 'rc='+str(code), flush=True)
        with locked_state() as state:
            if state['driver_sha256'] != driver_sha:
                raise RuntimeError('driver changed during batch')
            state['hosts'][args.host_label] = {'pid': os.getpid(), 'hostname': socket.gethostname(),
                    'devices': devices, 'heartbeat_utc': stamp(), 'active': list(active)}
            stage = 'smoke' if state['status'].startswith('smoke') else 'formal'
            stage_jobs = [j for j in state['jobs'] if j['stage'] == stage]
            if state['status'] != 'failed' and all(j['status'] == 'complete' for j in stage_jobs):
                try:
                    certificate = verify_reports([j['report'] for j in stage_jobs],
                                                 1 if stage == 'smoke' else state['steps'])
                    cert_prefix = args.prefix + ('.smoke' if stage == 'smoke' else '')
                    atomic_json(runs/(cert_prefix+'.verification.json'), certificate)
                    if stage == 'smoke':
                        formal = jobs_for(args.prefix, 'formal', state['steps'], state['measure_every'])
                        if any(Path(j[k]).exists() for j in formal for k in ('report', 'log')):
                            raise RuntimeError('formal output already exists')
                        state['jobs'].extend(formal); state['status'] = 'formal_running'
                        print('all eight smoke runs verified; formal queue opened', flush=True)
                    else:
                        state['status'] = 'complete'; state['finished_utc'] = stamp()
                        print('all eight formal runs verified', flush=True)
                except Exception as exc:
                    state.update(status='failed', failure=repr(exc))
            terminal = state['status'] in ('failed', 'complete')
        if terminal:
            if not active:
                return
            time.sleep(5); continue
        try:
            free = free_memory()
        except Exception as exc:
            print('GPU inventory unavailable', repr(exc), flush=True); time.sleep(10); continue
        for gpu in devices:
            if gpu in [j['gpu'] for j, _child, _stream in active.values()]:
                continue
            if free.get(gpu, 0) < args.min_free_mib:
                continue
            with locked_state() as state:
                if state['status'] in ('failed', 'complete'):
                    break
                if any(j['status'] == 'running' and j.get('host') == args.host_label
                       and j.get('gpu') == gpu for j in state['jobs']):
                    continue
                job = next((j for j in state['jobs'] if j['status'] == 'queued'), None)
                if job is None:
                    break
                command = command_for(job, gpu, args.cache_dir)
                stream = open(job['log'], 'x')
                try:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                        stdout=stream, stderr=subprocess.STDOUT)
                except Exception:
                    stream.close(); raise
                job.update(status='running', host=args.host_label, gpu=gpu, pid=child.pid,
                           command=command, started_utc=stamp())
                active[job['id']] = (dict(job), child, stream)
                print('launched', job['id'], args.host_label, 'GPU'+str(gpu), child.pid, flush=True)
        time.sleep(5)


if __name__ == '__main__':
    main()
