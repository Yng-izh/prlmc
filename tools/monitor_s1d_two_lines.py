#!/usr/bin/env python3
"""Adopt M1 without restarting; stop only its superseded serial dispatcher."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time
from native_qa_compare import atomic_json
from verify_s1d_risk_protection import verify

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',default='s1d_two_lines_v1')
    p.add_argument('--adopt-m1',action='store_true');args=p.parse_args()
    runs=ROOT/'runs';out=runs/(args.prefix+'.M1.monitor.json')
    if out.exists():raise SystemExit('monitor state already exists')
    old_path=runs/'s1d_c_risk_v1.launcher.json';old=json.loads(old_path.read_text())
    weak=next(s for s in old['stages'] if s['stage']=='weak')
    state=dict(status='running',pid=os.getpid(),host='v100-2',source_report=weak['report'],
               source_log=weak['log'],training_pid=weak['pid'],original_launcher=old,
               exit_code=None,exit_code_source='adopted process: original exit code unavailable')
    if args.adopt_m1:
        cmdline=Path('/proc/%d/cmdline'%old['pid']).read_bytes().decode().replace('\0',' ')
        if 'run_s1d_risk_protection.py' not in cmdline:raise RuntimeError('dispatcher PID mismatch')
        os.kill(old['pid'],signal.SIGTERM)
        old.update(status='superseded_parallel',replacement=str(out),
                   note='serial dispatcher stopped; existing M1 child retained; M2 runs independently')
        atomic_json(old_path,old)
        print('stopped old dispatcher only; retained M1 PID',weak['pid'],flush=True)
    atomic_json(out,state)
    while True:
        report_path=Path(weak['report'])
        r=json.loads(report_path.read_text()) if report_path.exists() else None
        if r and r['status']=='complete':
            try:
                baseline=json.loads((runs/'s1d_exp1_fact_v3.C.w17510.json').read_text())
                cert=verify(r,baseline,60)
                import torch
                a=r['windows'][0]['arms']['C'];ba=baseline['windows'][0]['arms']['C']
                saved=torch.load(a['residual_checkpoint'],map_location='cpu',weights_only=False)
                previous=torch.load(ba['residual_checkpoint'],map_location='cpu',weights_only=False)
                if not torch.equal(saved['basis'],previous['basis']):raise ValueError('saved basis changed')
                cert['saved_basis_identical_to_C']=True
                atomic_json(runs/(args.prefix+'.M1.verification.json'),cert)
                state.update(status='complete',verification=cert)
            except Exception as exc:
                state.update(status='failed',failure=repr(exc))
            break
        if r and r['status']=='failed':
            state.update(status='failed',failure=r.get('failure'));break
        if not Path('/proc/%d'%weak['pid']).exists():
            state.update(status='failed',failure='M1 process disappeared without complete report');break
        if r and r.get('windows'):
            a=r['windows'][0]['arms'].get('C',{})
            state['last_measured_step']=a.get('final',{}).get('step',0)
        state['updated_utc']=datetime.now(timezone.utc).isoformat();atomic_json(out,state)
        time.sleep(15)
    atomic_json(out,state)
    # Independent tasks continue on the other host. Aggregate once all finish;
    # this monitor never opens controlled test episodes or starts new training.
    fleet=runs/(args.prefix+'.v100-1.launcher.json')
    while True:
        other=json.loads(fleet.read_text()) if fleet.exists() else {}
        if other.get('status') in ('complete','failed'):break
        time.sleep(20)
    summary=dict(status='complete' if state['status']==other['status']=='complete' else 'failed',
                 method={},application={},sealed_test_run=False)
    if state['status']=='complete':summary['method']['M1']=state['verification']
    for j in other.get('jobs',[]):
        if j['status']!='complete':continue
        key='method' if j['name']=='M2' else 'application'
        summary[key][j['name']]=j['verification']
    m=summary['method']
    summary['method_gates_passed']=len(m)==2 and all(x['gate_passed'] for x in m.values())
    summary['decision']='review both lines before method freeze; no prompt/weight reselection; sealed tests remain unopened'
    atomic_json(runs/(args.prefix+'.summary.json'),summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
