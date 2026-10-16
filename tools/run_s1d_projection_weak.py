#!/usr/bin/env python3
"""Run the three fixed weak-window variants with audited smoke gates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import threading
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--devices', default='6,8,9')
    args = parser.parse_args()
    devices = [int(x) for x in args.devices.split(',')]
    if len(devices) != 3 or len(set(devices)) != 3:
        raise ValueError('three unique physical GPUs required')
    root = Path(__file__).resolve().parents[1]
    prefix = root / 'runs' / args.prefix
    state_path = Path(str(prefix) + '.launcher.json')
    if state_path.exists():
        raise ValueError('launcher report exists')
    baseline = root / 'runs/s1d_exp1_fact_v3.C.w17510.json'
    facts = root / 'runs/s1d_exp1_fact_v1.smoke.C.w17510.fact_rows.pt'
    protection = Path(str(prefix) + '.protection_rows.pt')
    lock = threading.Lock()
    state = dict(status='running', window=[3, 17510],
                 started_utc=datetime.now(timezone.utc).isoformat(),
                 tasks={v: dict(gpu=g, status='queued')
                        for v, g in zip(('C32', 'P13', 'P32'), devices)})

    def update(variant, **values):
        with lock:
            state['tasks'][variant].update(values)
            temporary = state_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(state, indent=2) + '\n')
            temporary.replace(state_path)
        print(variant, values, flush=True)

    def preflight(gpu):
        result = subprocess.check_output([
            'nvidia-smi', '-i', str(gpu),
            '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits'], text=True)
        used, free = [int(x.strip()) for x in result.strip().split(',')]
        if used > 256 or free < 30000:
            raise RuntimeError(f'GPU {gpu} occupied: used={used}, free={free} MiB')

    def train_command(variant, gpu, output, steps):
        return [sys.executable, '-u', str(root / 'tools/s1d_projected_train.py'),
                '--basis-variant', variant, '--window', '3,17510',
                '--cuda-device', str(gpu), '--model-path', '/dev/shm/s1d_reader_fp16',
                '--fact-cache', str(facts), '--c-reference', str(baseline),
                '--protection-rows', str(protection), '--out', str(output),
                '--steps', str(steps), '--measure-every', '1' if steps == 1 else '10',
                '--skip-generation', '--qa-max-tokens', '256']

    def run(variant, stage, command, log):
        update(variant, status=stage, command=command, log=str(log))
        with log.open('x') as handle:
            process = subprocess.Popen(command, cwd=root, stdout=handle,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            update(variant, pid=process.pid)
            code = process.wait()
        update(variant, last_exit_code=code)
        if code:
            raise RuntimeError(f'{stage} exited {code}; see {log}')

    def verify(variant, output, steps):
        command = [sys.executable, str(root / 'tools/verify_s1d_projection.py'),
                   '--report', str(output), '--baseline', str(baseline),
                   '--fact-cache', str(facts), '--checkpoint', str(output.with_suffix('.alpha.pt')),
                   '--steps', str(steps)]
        if variant != 'C32':
            command += ['--protection-cache', str(protection)]
        run(variant, 'verify_' + str(steps), command, output.with_suffix('.verification.log'))

    preparation_done = threading.Event()
    preparation_error = []

    def worker(variant, gpu):
        try:
            preflight(gpu)
            if variant == 'P13':
                output = Path(str(prefix) + '.prepare.json')
                try:
                    run(variant, 'prepare_protection',
                        train_command(variant, gpu, output, 1) + ['--prepare-protection-cache'],
                        output.with_suffix('.log'))
                except Exception as error:
                    preparation_error.append(str(error))
                    raise
                finally:
                    preparation_done.set()
            elif variant == 'P32':
                update(variant, status='waiting_protection')
                preparation_done.wait()
                if preparation_error:
                    raise RuntimeError('protection preparation failed: ' + preparation_error[0])
            smoke = Path(str(prefix) + f'.{variant}.smoke.json')
            preflight(gpu)
            run(variant, 'smoke', train_command(variant, gpu, smoke, 1), smoke.with_suffix('.log'))
            verify(variant, smoke, 1)
            output = Path(str(prefix) + f'.{variant}.json')
            preflight(gpu)
            run(variant, 'training60', train_command(variant, gpu, output, 60), output.with_suffix('.log'))
            verify(variant, output, 60)
            update(variant, status='complete', report=str(output))
        except Exception:
            if variant == 'P13' and not preparation_done.is_set():
                preparation_error.append(traceback.format_exc())
                preparation_done.set()
            update(variant, status='failed', error=traceback.format_exc())

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(worker, v, g)
                   for v, g in zip(('C32', 'P13', 'P32'), devices)]
        for future in futures:
            future.result()
    state['status'] = ('complete' if all(t['status'] == 'complete' for t in state['tasks'].values())
                       else 'failed')
    state['ended_utc'] = datetime.now(timezone.utc).isoformat()
    update('C32')
    return 0 if state['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
