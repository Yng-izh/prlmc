#!/usr/bin/env python3
"""Three independent smoke/formal jobs alongside the existing weak M1."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from native_qa_compare import atomic_json
from verify_s1d_risk_protection import verify
from s1d_c_generation import paired_summary

ROOT = Path(__file__).resolve().parents[1]


def verify_generation(path, smoke):
    r = json.loads(Path(path).read_text())
    if r['status'] != 'complete' or r['smoke'] != smoke:
        raise ValueError('generation incomplete/wrong stage')
    maps = {k:v['correct'] for k,v in r['arm_results'].items()}
    if set(maps) != {'Full','Selection','D0','C'} or paired_summary(maps) != r['summary']:
        raise ValueError('generation metrics mismatch')
    if len(r['pairs']) != (2 if smoke else len(r['reference_tf']['C'])):
        raise ValueError('question count drift')
    if len({r['arm_results'][k]['active_slots'] for k in ('Selection','D0','C')}) != 1:
        raise ValueError('budget mismatch')
    return dict(valid=True, smoke=smoke, n=len(r['pairs']), summary=r['summary'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--devices',default='5,6,8')
    p.add_argument('--host-label',default='v100-1')
    p.add_argument('--prefix',default='s1d_two_lines_v1')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    devices=list(map(int,args.devices.split(',')))
    if len(devices)!=3 or len(set(devices))!=3:
        raise SystemExit('three distinct available devices required')
    if args.host_label=='v100-2' and set(devices)&{5,7}:
        raise SystemExit('v100-2 GPU5/7 excluded')
    runs=ROOT/'runs'; statepath=runs/(args.prefix+'.'+args.host_label+'.launcher.json')
    if statepath.exists() and not args.dry_run:
        raise SystemExit('state exists; inspect rather than relaunch')
    def now(): return datetime.now(timezone.utc).isoformat()
    modules=['s1d_c_runtime.py','s1d_c_generation.py','s1d_c_generate.py','s1d_c_risk_train.py',
             's1d_risk_protection.py','readout_probe.py','verify_s1d_risk_protection.py']
    initial={n:hashlib.sha256((ROOT/'tools'/n).read_bytes()).hexdigest() for n in modules}
    jobs=[dict(name=n,start=s,gpu=g,status='queued',stage='smoke')
          for (n,s),g in zip([('M2',16370),('A1',16370),('A2',17510)],devices)]
    state=dict(status='running',host=args.host_label,pid=os.getpid(),created_utc=now(),
               script_hashes=initial,jobs=jobs)
    def command(j):
        stem=args.prefix+'.'+j['name']+('.smoke' if j['stage']=='smoke' else '')
        j.update(report=str(runs/(stem+'.json')),log=str(runs/(stem+'.log')))
        method=j['name']=='M2'; smoke=j['stage']=='smoke'
        cmd=[sys.executable,str(ROOT/'tools'/('s1d_c_risk_train.py' if method else 's1d_c_generate.py')),
             '--exp1-arm','C','--window','3,%d'%j['start'],'--cuda-device',str(j['gpu']),
             '--model-path','/dev/shm/s1d_reader_fp16','--fact-cache',
             str(runs/('s1d_exp1_fact_v1.smoke.C.w%d.fact_rows.pt'%j['start'])),
             '--out',j['report'],'--skip-generation','--qa-max-tokens','256']
        if method:
            cmd+=['--risk-protection','--steps','1' if smoke else '60',
                  '--measure-every','1' if smoke else '10']
        else:
            cmd+=['--c-checkpoint',str(runs/('s1d_exp1_fact_v3.C.w%d.alpha.pt'%j['start'])),
                  '--c-reference',str(runs/('s1d_exp1_fact_v3.C.w%d.json'%j['start']))]
            if smoke: cmd+=['--generation-smoke']
        return cmd
    if args.dry_run:
        print(json.dumps([command(j) for j in jobs],indent=2));return
    atomic_json(statepath,state)
    active={}
    env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    while any(j['status'] in ('queued','running') for j in jobs):
        for name,(child,stream,j) in list(active.items()):
            code=child.poll()
            if code is None:continue
            stream.close();del active[name]
            stage=j['stage'];j.update(exit_code=code,finished_utc=now())
            try:
                if code!=0:raise RuntimeError('worker exit '+str(code))
                if j['name']=='M2':
                    baseline=json.loads((runs/('s1d_exp1_fact_v3.C.w%d.json'%j['start'])).read_text())
                    report=json.loads(Path(j['report']).read_text())
                    cert=verify(report,baseline,1 if stage=='smoke' else 60)
                    import torch
                    a=report['windows'][0]['arms']['C'];ba=baseline['windows'][0]['arms']['C']
                    saved=torch.load(a['residual_checkpoint'],map_location='cpu',weights_only=False)
                    old=torch.load(ba['residual_checkpoint'],map_location='cpu',weights_only=False)
                    if not torch.equal(saved['basis'],old['basis']):raise ValueError('saved C basis drift')
                    cert['saved_basis_identical_to_C']=True
                    cert['original_strict_gate_passed']=cert['gate_passed']
                    cert['latest_strong_gate']=dict(min_net=7,damage_reported_separately=True)
                    cert['gate_passed']=stage=='formal' and cert['net']>=7
                else:
                    cert=verify_generation(j['report'],stage=='smoke')
                atomic_json(Path(j['report']).with_suffix('.verification.json'),cert)
                j['verification']=cert
                if stage=='smoke':
                    j['smoke']=dict(report=j['report'],log=j['log'],exit_code=code,verification=cert)
                    j.update(stage='formal',status='queued')
                else:j['status']='complete'
                print(j['name'],stage,'verified',flush=True)
            except Exception as exc:
                j.update(status='failed',failure=repr(exc));print(j['name'],'FAILED',repr(exc),flush=True)
        usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                                      '--format=csv,noheader,nounits'],text=True)
        free={int(i):int(m) for i,m in (l.split(',') for l in usage.splitlines())}
        for j in jobs:
            if j['status']!='queued' or free.get(j['gpu'],0)<30000:continue
            if any(hashlib.sha256((ROOT/'tools'/n).read_bytes()).hexdigest()!=h for n,h in initial.items()):
                raise RuntimeError('code changed during frozen batch')
            cmd=command(j)
            if Path(j['report']).exists() or Path(j['log']).exists():
                raise RuntimeError('refusing to overwrite evidence')
            stream=open(j['log'],'x')
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            j.update(status='running',pid=child.pid,command=cmd,started_utc=now())
            active[j['name']]=(child,stream,j)
            print('launched',j['name'],j['stage'],'GPU',j['gpu'],'PID',child.pid,flush=True)
        state['updated_utc']=now();atomic_json(statepath,state)
        if active or any(j['status']=='queued' for j in jobs):time.sleep(10)
    state.update(status='complete' if all(j['status']=='complete' for j in jobs) else 'failed',finished_utc=now())
    atomic_json(statepath,state)


if __name__=='__main__':main()
