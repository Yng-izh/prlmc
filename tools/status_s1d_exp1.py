#!/usr/bin/env python3
"""Read experiment fleet or legacy launcher state without changing jobs."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', default='s1d_exp1_fact_v3')
    parser.add_argument('--runs-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'runs')
    parser.add_argument('--logs', action='store_true')
    args = parser.parse_args()
    path = args.runs_dir / (args.prefix + '.fleet.json')
    if not path.exists():
        path = args.runs_dir / (args.prefix + '.launcher.json')
    if not path.exists():
        parser.error('state missing: ' + str(path))
    state = json.loads(path.read_text())
    print('batch:', args.prefix, 'status:', state['status'],
          'updated:', state.get('updated_utc', 'unknown'))
    if state.get('failure'):
        print('failure:', state['failure'])
    for host, details in state.get('hosts', {}).items():
        print('scheduler:', host, 'pid:', details['pid'],
              'heartbeat:', details.get('heartbeat_utc'))
    jobs = state.get('jobs')
    if jobs is None:
        jobs = [dict(j, stage=stage) for stage, values in state['stages'].items()
                for j in values]
    for job in jobs:
        report_path = Path(job['report'])
        try:
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
        except (OSError, ValueError) as exc:
            report = {'failure': 'unreadable report: ' + str(exc)}
        windows = report.get('windows', [])
        arm = windows[0].get('arms', {}).get(job['arm'], {}) if windows else {}
        final = arm.get('final', {})
        print(job['stage'], job['arm'], job['window'],
              job.get('host', '-'), 'GPU' + str(job.get('gpu', '-')),
              'job=' + job.get('status', report.get('status', 'unknown')),
              'report=' + report.get('status', 'pending'),
              'step=' + str(final.get('step', '-')),
              'R/H/net=' + str((final.get('new_recovery'), final.get('d0_harmed'),
                                final.get('net_vs_d0'))),
              'exit=' + str(job.get('exit_code', '-')))
        if report.get('failure'):
            print(' ', report['failure'])
        if args.logs and job.get('log') and Path(job['log']).exists():
            tail = subprocess.run(['tail', '-n', '1', job['log']],
                                  text=True, capture_output=True, check=False)
            print(' ', tail.stdout.strip()[:300])


if __name__ == '__main__':
    main()
