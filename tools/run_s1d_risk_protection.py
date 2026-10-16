#!/usr/bin/env python3
"""Single fixed scheme: smoke -> weak60 -> conditional strong60 -> stop."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from verify_s1d_risk_protection import verify
from s1d_risk_protection import risk_protocol

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--prefix', default='s1d_c_risk_v1')
    p.add_argument('--cache-dir', default='/dev/shm/s1d_reader_fp16')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.gpu in (5, 7):
        raise SystemExit('v100-2 GPU5/7 excluded')
    runs = ROOT/'runs'; runs.mkdir(exist_ok=True)
    state_path = runs/(args.prefix+'.launcher.json')
    if state_path.exists() and not args.dry_run:
        raise SystemExit('state already exists; inspect it instead of relaunching')
    def now():
        return datetime.now(timezone.utc).isoformat()
    scripts = ['s1d_experiment1.py', 's1d_risk_protection.py',
               'verify_s1d_risk_protection.py', 'run_s1d_risk_protection.py']
    def hashes():
        return {n: hashlib.sha256((ROOT/'tools'/n).read_bytes()).hexdigest() for n in scripts}
    initial_hashes = hashes()
    state = dict(status='starting', pid=os.getpid(), gpu=args.gpu, created_utc=now(),
                 protocol=risk_protocol(), script_hashes=initial_hashes, stages=[])
    def flush():
        tmp = state_path.with_name(state_path.name+'.tmp.'+str(os.getpid()))
        tmp.write_text(json.dumps(state, indent=2)); tmp.replace(state_path)
    def command(start, steps, output):
        return [sys.executable, str(ROOT/'tools/s1d_experiment1.py'),
                '--exp1-arm', 'C', '--risk-protection', '--window', '3,%d'%start,
                '--cuda-device', str(args.gpu), '--model-path', args.cache_dir,
                '--fact-cache', str(runs/('s1d_exp1_fact_v1.smoke.C.w%d.fact_rows.pt'%start)),
                '--out', str(output), '--steps', str(steps),
                '--measure-every', '1' if steps == 1 else '10',
                '--skip-generation', '--qa-max-tokens', '256']
    if args.dry_run:
        for label, start, steps in [('smoke',17510,1), ('weak',17510,60), ('strong',16370,60)]:
            print(json.dumps(dict(stage=label, command=command(start, steps, runs/(args.prefix+'.'+label+'.json')))))
        return
    flush()
    try:
        if not (Path(args.cache_dir)/'fp16_cache_manifest.json').exists():
            raise RuntimeError('verified FP16 cache missing')
        for label, start, steps in [('smoke',17510,1), ('weak',17510,60), ('strong',16370,60)]:
            output = runs/(args.prefix+'.'+label+'.json')
            logfile = runs/(args.prefix+'.'+label+'.log')
            baseline = runs/('s1d_exp1_fact_v3.C.w%d.json'%start)
            if output.exists() or logfile.exists():
                raise RuntimeError('refusing to overwrite prior evidence')
            job = dict(stage=label, window=[3,start], steps=steps, report=str(output),
                       log=str(logfile), baseline=str(baseline), command=command(start, steps, output))
            state['stages'].append(job)
            state.update(status=label+'_waiting_for_gpu', updated_utc=now()); flush()
            while True:
                usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                                 '--format=csv,noheader,nounits'], text=True)
                free = dict((int(i),int(m)) for i,m in (r.split(',') for r in usage.splitlines()))
                if free.get(args.gpu,0) >= 30000:
                    break
                time.sleep(15)
            if hashes() != initial_hashes:
                raise RuntimeError('script changed after preregistration')
            env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
            with logfile.open('x') as stream:
                child = subprocess.Popen(job['command'], cwd=ROOT, env=env,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                job.update(pid=child.pid, started_utc=now())
                state.update(status=label+'_running', updated_utc=now()); flush()
                job['exit_code'] = child.wait()
            job['finished_utc'] = now(); flush()
            if job['exit_code'] != 0:
                raise RuntimeError(label+' failed; inspect its log')
            if hashes() != initial_hashes:
                raise RuntimeError('script changed during run')
            certificate = verify(json.loads(output.read_text()), json.loads(baseline.read_text()), steps)
            # Verify saved basis matches the published C binary exactly.
            import torch
            report = json.loads(output.read_text())
            a = report['windows'][0]['arms']['C']
            original = json.loads(baseline.read_text())['windows'][0]['arms']['C']
            binary = torch.load(a['residual_checkpoint'], map_location='cpu', weights_only=False)
            old_binary = torch.load(original['residual_checkpoint'], map_location='cpu', weights_only=False)
            if not torch.equal(binary['basis'], old_binary['basis']):
                raise RuntimeError('C saved basis changed')
            if not torch.equal(binary['alpha'], torch.tensor(a['final']['alpha'], dtype=binary['alpha'].dtype)):
                raise RuntimeError('saved alpha differs from final measurement')
            certificate['saved_basis_identical_to_C'] = True
            certpath = runs/(args.prefix+'.'+label+'.verification.json')
            certpath.write_text(json.dumps(certificate, indent=2)); job['verification'] = certificate
            flush()
            print(label+' verified '+json.dumps(certificate), flush=True)
            if steps == 60 and not certificate['gate_passed']:
                state.update(status='complete_no_go', decision='stop protection mechanism',
                             failed_gate=label, finished_utc=now()); flush()
                print('NO-GO: '+label+' gate failed; no further stage launched', flush=True)
                return
        state.update(status='complete_dev_gates_passed', finished_utc=now(),
                     decision='both dev gates passed; unseen contexts and official generation require separate protocol')
        flush()
    except Exception as exc:
        state.update(status='failed', failure=repr(exc), updated_utc=now()); flush()
        raise


if __name__ == '__main__':
    main()
