#!/usr/bin/env python3
"""Queue fixed native write/drop development evaluations; rescore and compare."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from s1d_native_policy_eval import ARMS, PROTOCOL, source_manifest
from s1d_shared_compactor_eval import DEVELOPMENT_IDS, choose_development, sha, atomic_json


def aggregate(paths, args):
    from readout_probe import official_score
    plan = json.loads(Path(args.plan).read_text())
    rows, failures = [], []
    expected_manifest = source_manifest()
    for ep, path in zip(DEVELOPMENT_IDS, paths):
        if not path.exists():
            failures.append(dict(episode=ep, status='missing')); continue
        report = json.loads(path.read_text())
        if report['status'] != 'complete':
            failures.append(dict(episode=ep, status=report['status'], failure=report.get('failure'))); continue
        episode = choose_development(plan, ep)
        identity = report['identity']
        if (report.get('smoke') or report.get('test_opened') is not False or report.get('training_performed') is not False
                or identity['source_manifest'] != expected_manifest or identity['protocol'] != PROTOCOL
                or identity['plan_sha256'] != sha(args.plan) or identity['episode'] != ep
                or identity['context_sha256'] != episode['context_sha256']):
            raise ValueError('native report integrity mismatch')
        reference_path = Path(args.reference_prefix+'.ep%d.json'%ep)
        if identity['reference_sha256'] != sha(reference_path):
            raise ValueError('shared reference changed')
        reference = json.loads(reference_path.read_text())
        if identity['model_manifest_sha256'] != reference['identity']['model_manifest_sha256']:
            raise ValueError('native/shared reader caches differ')
        answers = {q['question']:q['answer'] for q in episode['splits']['evaluation']}
        if len(answers) != 64 or set(report['generation_questions']) != set(answers):
            raise ValueError('native QA set differs')
        maps, metrics = {}, {}
        for name in ['Full','Selection','SharedE6d'] + list(ARMS):
            origin = reference if name not in ARMS else report
            predictions = origin['predictions'][name]
            if set(predictions) != set(answers):
                raise ValueError('prediction question set differs')
            maps[name] = {}
            for question, record in predictions.items():
                em, sub = official_score(record['prediction'], [answers[question]])
                if bool(record['em']) != bool(em) or bool(record['substring_em']) != bool(sub):
                    raise ValueError('score differs from raw prediction rescoring')
                maps[name][question] = bool(em)
            metrics[name] = dict(correct=sum(maps[name].values()), n=64, em=sum(maps[name].values())/64)
            if name in ARMS:
                metadata = report['arm_metadata'][name]
                if (metadata['active_slots'] != ARMS[name]*256 or metadata['physical_slots'] != ARMS[name]*256
                        or not metadata['source_memory_unchanged'] or not metadata['native_drop_per_layer']
                        or len(report['writes'][name]) != 20):
                    raise ValueError('native budget/write integrity failed')
                if metrics[name] != report['summary'][name]:
                    raise ValueError('native summary differs')
        paired = {}
        for baseline in ['Selection'] + list(ARMS):
            gained = [q for q in answers if maps['SharedE6d'][q] and not maps[baseline][q]]
            harmed = [q for q in answers if maps[baseline][q] and not maps['SharedE6d'][q]]
            paired[baseline] = dict(gained=gained, harmed=harmed, net=len(gained)-len(harmed),
                delta_em=(len(gained)-len(harmed))/64,
                equal_effective_slots=baseline != 'NativeDefault')
        rows.append(dict(episode=ep, metrics=metrics, shared_vs=paired, costs=report['costs']))
    result = dict(schema_version='s1d_native_policy_summary/1', status='complete' if not failures else 'incomplete',
        protocol=PROTOCOL, episodes=rows, failures=failures, sealed_test_opened=False,
        source_manifest=expected_manifest, native_default_slots=12800, matched_budget_slots=2816,
        interpretation='NativeDefault uses more slots; NativeBudget changes capacity only, preserving native write/drop policy')
    if not failures:
        result['totals'] = {name:dict(correct=sum(e['metrics'][name]['correct'] for e in rows), n=512,
            em=sum(e['metrics'][name]['correct'] for e in rows)/512)
            for name in ['Full','Selection','SharedE6d']+list(ARMS)}
        result['shared_vs'] = {}
        for baseline in ['Selection']+list(ARMS):
            deltas = [e['shared_vs'][baseline]['delta_em'] for e in rows]
            rng = random.Random(20261012)
            samples = sorted(sum(rng.choice(deltas) for _ in deltas)/len(deltas) for _ in range(10000))
            result['shared_vs'][baseline] = dict(mean_delta=sum(deltas)/8,
                positive_episodes=sum(d>0 for d in deltas), negative_episodes=sum(d<0 for d in deltas),
                gained=sum(len(e['shared_vs'][baseline]['gained']) for e in rows),
                harmed=sum(len(e['shared_vs'][baseline]['harmed']) for e in rows),
                bootstrap_episode_ci95=[samples[249],samples[9749]],
                equal_effective_slots=baseline!='NativeDefault')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--devices', default='5,9')
    p.add_argument('--prefix', default='runs/s1d_native_policy_dev_gpu59_v1')
    p.add_argument('--plan', default='runs/sic_paper_controlled_plan.json')
    p.add_argument('--reference-prefix', default='runs/s1d_shared_e6d_dev_gpu59_v1')
    p.add_argument('--model-path', default='/dev/shm/s1d_reader_fp16')
    args=p.parse_args()
    devices=[int(d) for d in args.devices.split(',')]
    if not 1<=len(devices)<=8 or len(set(devices))!=len(devices):
        raise ValueError('one to eight unique devices required')
    query=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.used','--format=csv,noheader,nounits'], text=True)
    available={int(line.split(',')[0]):[int(x) for x in line.split(',')[1:]] for line in query.splitlines()}
    if any(d not in available or available[d][0]<30000 or available[d][1]>256 for d in devices):
        raise ValueError('GPU preflight failed')
    prefix=Path(args.prefix); prefix.parent.mkdir(parents=True,exist_ok=True)
    fleet=Path(str(prefix)+'.fleet.json')
    paths=[Path(str(prefix)+'.ep%d.json'%ep) for ep in DEVELOPMENT_IDS]
    if fleet.exists() or any(p.exists() or p.with_suffix('.log').exists() for p in paths):
        raise ValueError('refuse to overwrite existing fleet artifacts')
    pending=list(zip(DEVELOPMENT_IDS,paths)); workers={}; jobs=[]
    def launch(device):
        ep,path=pending.pop(0)
        cmd=[sys.executable,'-u','tools/s1d_native_policy_eval.py','--episode',str(ep),
            '--cuda-device',str(device),'--out',str(path),'--plan',args.plan,
            '--reference-prefix',args.reference_prefix,'--model-path',args.model_path]
        log=path.with_suffix('.log').open('w')
        worker=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        job=dict(episode=ep,device=device,pid=worker.pid,command=cmd,out=str(path),log=str(path.with_suffix('.log')),exit_code=None)
        jobs.append(job); workers[device]=(worker,log,job)
        print('START',json.dumps(job),flush=True)
    def save(status):
        atomic_json(fleet,dict(status=status,devices=devices,jobs=jobs,pending_episodes=[ep for ep,_ in pending],protocol=PROTOCOL))
    for device in devices: launch(device)
    save('running')
    while workers:
        for device,(worker,log,job) in list(workers.items()):
            code=worker.poll()
            if code is None: continue
            log.close(); job['exit_code']=code; del workers[device]
            print('EXIT',job['episode'],device,code,flush=True)
            if code != 0:
                # Preserve unstarted episodes on a setup failure; do not burn the fleet.
                save('failed')
                while workers:
                    for d,(w,l,j) in list(workers.items()):
                        c=w.poll()
                        if c is not None:
                            l.close(); j['exit_code']=c; del workers[d]
                    save('failed')
                    if workers: time.sleep(5)
                raise RuntimeError('native worker failed; pending episodes left unstarted')
            if pending: launch(device)
        save('running' if workers else 'complete')
        if workers: time.sleep(5)
    summary=aggregate(paths,args)
    atomic_json(str(prefix)+'.summary.json',summary)
    print('FINAL',json.dumps({k:summary[k] for k in ['status','failures','totals','shared_vs'] if k in summary}),flush=True)

if __name__=='__main__': main()
