#!/usr/bin/env python3
"""Recheck eight completed reports with the launcher's strict certificate rules."""
import argparse
import json
from pathlib import Path
from run_s1d_experiment1_parallel import verify_reports

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prefix', default='s1d_exp1_fact_v1')
    p.add_argument('--steps', type=int, default=60)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = [root/'runs'/('%s.%s.w%d.json' % (args.prefix, arm, start))
             for start in (16370, 17510) for arm in 'ABCD']
    print(json.dumps(verify_reports(paths, args.steps), indent=2))

if __name__ == '__main__':
    main()
