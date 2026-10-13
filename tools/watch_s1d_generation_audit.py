#!/usr/bin/env python3
"""Audit generated reports with official source once workers finish."""
import argparse
import json
from pathlib import Path
import time
from audit_s1d_generation import audit
from native_qa_compare import atomic_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',default='s1d_two_lines_v2')
    args=p.parse_args();runs=ROOT/'runs'
    source=Path('/share/latent/MemoryAgentBench/utils/eval_other_utils.py').read_text()
    statepath=runs/(args.prefix+'.official_audit.json')
    state=dict(status='running',results={})
    while True:
        fleetpath=runs/(args.prefix+'.v100-1.launcher.json')
        fleet=json.loads(fleetpath.read_text()) if fleetpath.exists() else {}
        for name in ('A1','A2'):
            for suffix in ('.smoke',''):
                key=name+suffix
                if key in state['results']:continue
                path=runs/(args.prefix+'.'+key+'.json')
                if not path.exists():continue
                report=json.loads(path.read_text())
                if report.get('status')!='complete':continue
                try:
                    result=audit(report,source)
                    atomic_json(path.with_suffix('.official_verification.json'),result)
                    state['results'][key]=result
                    print('official verified',key,flush=True)
                except Exception as exc:
                    state['results'][key]=dict(valid=False,failure=repr(exc))
        atomic_json(statepath,state)
        if fleet.get('status') in ('complete','failed'):
            state['status']='complete' if all(k in state['results'] and state['results'][k]['valid']
                for k in ('A1.smoke','A2.smoke','A1','A2')) else 'failed'
            atomic_json(statepath,state);return
        time.sleep(15)

if __name__=='__main__':main()
