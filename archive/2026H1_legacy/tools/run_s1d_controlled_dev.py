#!/usr/bin/env python3
"""Launch only eight development episodes after a matching GPU smoke passes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from s1d_controlled_c13 import DEVELOPMENT_IDS, PROTOCOL, source_manifest


def aggregate(paths):
    from readout_probe import official_score
    from s1d_c_generation import paired_summary
    from s1d_controlled_contract import episode_bootstrap
    rows, failures, identity = [], [], None
    for episode, path in zip(DEVELOPMENT_IDS, paths):
        report = json.loads(Path(path).read_text()) if Path(path).exists() else {}
        if report.get('status') != 'complete' or report.get('identity',{}).get('smoke'):
            failures.append(dict(episode=episode, path=str(path), status=report.get('status','missing'),
                                 failure=report.get('failure')))
            continue
        if report['episode'] != episode or report['identity']['protocol'] != PROTOCOL:
            raise ValueError('episode/protocol mismatch')
        if report.get('training',{}).get('steps_completed') != 60 or not report.get('checkpoint_frozen'):
            raise ValueError('fixed step60 checkpoint required')
        common = {k:v for k,v in report['identity'].items() if k not in ('episode','context_sha256')}
        if identity is None:
            identity = common
        if common != identity or common['source_manifest'] != source_manifest():
            raise ValueError('mixed model/data/code/method provenance')
        from s1d_controlled_c13 import sha
        if sha(Path(path).with_suffix('.alpha.pt')) != report['checkpoint_sha256']:
            raise ValueError('frozen checkpoint hash mismatch')
        generation_path = Path(report['generation_file'])
        if sha(generation_path) != report['generation_sha256']:
            raise ValueError('generation hash mismatch')
        generation = json.loads(generation_path.read_text())
        if generation['status'] != 'complete' or len(generation['pairs']) != 64:
            raise ValueError('incomplete paired generation')
        if (generation['checkpoint_sha256'] != report['checkpoint_sha256']
                or generation.get('episode') != episode or not generation['source_memory_unchanged']
                or len({p['question'] for p in generation['pairs']}) != 64):
            raise ValueError('generation provenance/memory/question mismatch')
        if any(generation['arm_results'][name]['active_slots'] != report['active_slots']
               for name in ('Selection','D0','C')):
            raise ValueError('compressed budget mismatch')
        maps = {name:{} for name in PROTOCOL['arms']}
        for pair in generation['pairs']:
            for name in maps:
                scored = bool(official_score(pair[name]['prediction'],pair['answers'])[0])
                if scored != pair[name]['em']:
                    raise ValueError('independent scoring mismatch')
                maps[name][pair['question']] = scored
        summary = paired_summary(maps)
        if summary != report['summary'] or summary != generation['summary']:
            raise ValueError('paired summary mismatch')
        row = dict(episode=episode, status='complete', path=str(path),
            C_em=summary['arms']['C']['em'], D0_em=summary['arms']['D0']['em'],
            Selection_em=summary['arms']['Selection']['em'], Full_em=summary['arms']['Full']['em'],
            C_minus_Selection=summary['C_vs_Selection']['delta_em'],
            C_minus_D0=summary['C_vs_D0']['delta_em'], R=summary['R'],H=summary['H'],
            summary=summary,costs=report['costs'])
        rows.append(row)
    result = dict(status='complete' if not failures else 'incomplete', stage='development',
        expected_episode_ids=list(DEVELOPMENT_IDS), episodes=rows, failures=failures,
        independent_official_rescoring=True, inference_unit='episode', sealed_test_opened=False)
    if not failures:
        result['paired'] = {key:episode_bootstrap(rows,key) for key in ('Selection','D0')}
        result['positive_vs_Selection'] = sum(r['C_minus_Selection'] > 0 for r in rows)
        result['negative_vs_Selection'] = sum(r['C_minus_Selection'] < 0 for r in rows)
        result['mean_em'] = {k:sum(r[k+'_em'] for r in rows)/len(rows)
                             for k in ('Full','Selection','D0','C')}
        result['next_action'] = 'review development evidence before any explicit method freeze or sealed-test opening'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--devices', required=True, help='eight currently idle physical GPUs')
    p.add_argument('--smoke-report', required=True)
    p.add_argument('--prefix', default='runs/s1d_controlled_c13_v1')
    p.add_argument('--await-smoke', action='store_true', help='wait for matching running smoke; never launch after failure')
    args = p.parse_args()
    from native_qa_compare import atomic_json
    smoke_path=Path(args.smoke_report)
    deadline=time.monotonic()+3600
    while args.await_smoke:
        smoke=json.loads(smoke_path.read_text()) if smoke_path.exists() else {}
        if smoke.get('status') in ('complete','failed'):
            break
        if time.monotonic()>deadline:
            raise ValueError('smoke timed out; no development workers launched')
        time.sleep(10)
    smoke = json.loads(smoke_path.read_text())
    if (smoke.get('status') != 'complete' or smoke['identity']['smoke'] is not True
            or not smoke['initial_identity_verified'] or not smoke['source_memory_unchanged']
            or smoke['identity']['source_manifest'] != source_manifest()):
        raise ValueError('completed matching smoke required')
    devices = [int(v) for v in args.devices.split(',')]
    if len(devices) != 8 or len(set(devices)) != 8:
        raise ValueError('eight distinct idle GPUs required')
    # The completed smoke process may still be releasing its CUDA context.
    if args.await_smoke:
        release_deadline=time.monotonic()+120
        while time.monotonic()<release_deadline:
            occupied=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
                '--format=csv,noheader,nounits'],text=True)
            used={int(row.split(',')[0]):int(row.split(',')[1]) for row in occupied.splitlines()}
            if all(used.get(g,999999)<=256 for g in devices):
                break
            time.sleep(3)
    query = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.used',
                                      '--format=csv,noheader,nounits'],text=True)
    available = {int(row.split(',')[0]):[int(v.strip()) for v in row.split(',')[1:]]
                 for row in query.splitlines()}
    if any(g not in available or available[g][0] < 30000 or available[g][1] > 256 for g in devices):
        raise ValueError('GPU preflight: selected GPU occupied or <30GB free')
    fleet_path = Path(args.prefix+'.fleet.json')
    paths = [Path(args.prefix+'.ep%d.json'%i) for i in DEVELOPMENT_IDS]
    logs = [path.with_suffix('.log') for path in paths]
    if fleet_path.exists() or any(p.exists() for p in paths+logs):
        raise ValueError('existing fleet/results/logs; preserve original artifacts')
    jobs=[]
    fleet=dict(status='launching', stage='development', sealed_test_opened=False,
               smoke_report=args.smoke_report, jobs=jobs, protocol=PROTOCOL)
    atomic_json(fleet_path,fleet)
    workers=[]
    try:
        for episode,gpu,path,log in zip(DEVELOPMENT_IDS,devices,paths,logs):
            command=[sys.executable,'-u','tools/s1d_controlled_c13.py','--episode',str(episode),
                     '--cuda-device',str(gpu),'--out',str(path)]
            with log.open('w') as handle:
                worker=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,start_new_session=True)
            workers.append(worker)
            jobs.append(dict(episode=episode,gpu=gpu,pid=worker.pid,command=command,
                             out=str(path),log=str(log),exit_code=None))
            atomic_json(fleet_path,fleet)
        fleet['status']='running'
        atomic_json(fleet_path,fleet)
        print('LAUNCHED',json.dumps(jobs),flush=True)
        while any(w.poll() is None for w in workers):
            for worker,job in zip(workers,jobs):
                job['exit_code']=worker.poll()
            atomic_json(fleet_path,fleet)
            time.sleep(20)
        for worker,job in zip(workers,jobs):
            job['exit_code']=worker.returncode
        fleet['status']='complete' if all(w.returncode==0 for w in workers) else 'failed'
        atomic_json(fleet_path,fleet)
        summary=aggregate(paths)
        atomic_json(args.prefix+'.summary.json',summary)
        print('FINAL',json.dumps(summary.get('paired',summary['failures'])),flush=True)
    except Exception as error:
        fleet.update(status='failed',failure=dict(type=type(error).__name__,message=str(error)))
        atomic_json(fleet_path,fleet)
        raise


if __name__=='__main__':
    main()
