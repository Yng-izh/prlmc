#!/usr/bin/env python3
"""Sequential audited dimension runs on one idle GPU; hosts can run in parallel."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--gpu', required=True, type=int)
    ap.add_argument('--max-existing-mib', type=int, default=256)
    ap.add_argument('--wait-idle', action='store_true')
    ap.add_argument('--dimensions', required=True, nargs='+', type=int, choices=(20,24,28))
    args = ap.parse_args()
    if len(args.dimensions) != len(set(args.dimensions)):
        raise ValueError('duplicate dimensions')
    root = Path(__file__).resolve().parents[1]
    prefix = root/'runs'/args.prefix
    path = Path(str(prefix)+f'.gpu{args.gpu}.launcher.json')
    if path.exists():
        raise ValueError('launcher report already exists')
    baseline = root/'runs/s1d_exp1_fact_v3.C.w17510.json'
    facts = root/'runs/s1d_exp1_fact_v1.smoke.C.w17510.fact_rows.pt'
    state = dict(status='running', gpu=args.gpu, window=[3,17510],
                 started_utc=datetime.now(timezone.utc).isoformat(),
                 tasks={f'C{d}':dict(status='queued') for d in args.dimensions})

    def flush():
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2)+'\n')
        tmp.replace(path)

    def update(variant, **values):
        state['tasks'][variant].update(values)
        flush()
        print(variant, values, flush=True)

    def preflight(variant):
        while True:
            result = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
                '--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'], text=True)
            used, free = (int(x.strip()) for x in result.strip().split(','))
            if used <= args.max_existing_mib and free >= 30000:
                return
            if not args.wait_idle:
                raise RuntimeError(f'GPU {args.gpu} occupied: used={used}, free={free} MiB')
            update(variant,status='waiting_idle_gpu',memory_used_mib=used,memory_free_mib=free)
            time.sleep(30)

    def run(variant, stage, command, log):
        update(variant,status=stage,command=command,log=str(log))
        with log.open('x') as handle:
            child = subprocess.Popen(command,cwd=root,stdout=handle,stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL)
            update(variant,pid=child.pid)
            code = child.wait()
        update(variant,last_exit_code=code)
        if code:
            raise RuntimeError(f'{stage} exited {code}; see {log}')

    # Record the exact files used without modifying the active projection runs.
    state['script_hashes'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__),root/'tools/s1d_dimension_basis.py',root/'tools/s1d_dimension_train.py',
         root/'tools/verify_s1d_dimensions.py')}
    flush()
    for dimension in args.dimensions:
        variant = f'C{dimension}'
        try:
            for steps in (1,60):
                preflight(variant)
                suffix = '.smoke' if steps == 1 else ''
                output = Path(str(prefix)+f'.{variant}{suffix}.json')
                command = [sys.executable,'-u',str(root/'tools/s1d_dimension_train.py'),
                    '--basis-variant',variant,'--window','3,17510','--cuda-device',str(args.gpu),
                    '--model-path','/dev/shm/s1d_reader_fp16','--fact-cache',str(facts),
                    '--c-reference',str(baseline),'--out',str(output),'--steps',str(steps),
                    '--measure-every','1' if steps==1 else '10','--skip-generation','--qa-max-tokens','256']
                run(variant,'smoke' if steps==1 else 'training60',command,output.with_suffix('.log'))
                verify = [sys.executable,str(root/'tools/verify_s1d_dimensions.py'),
                    '--report',str(output),'--baseline',str(baseline),'--fact-cache',str(facts),
                    '--checkpoint',str(output.with_suffix('.alpha.pt')),'--steps',str(steps)]
                run(variant,f'verify_{steps}',verify,output.with_suffix('.verification.log'))
            update(variant,status='complete',report=str(output))
        except Exception:
            update(variant,status='failed',error=traceback.format_exc())
    state['status'] = 'complete' if all(t['status']=='complete' for t in state['tasks'].values()) else 'failed'
    state['ended_utc'] = datetime.now(timezone.utc).isoformat()
    flush()
    return 0 if state['status']=='complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
