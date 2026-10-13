#!/usr/bin/env python3
"""Integrity-only preparation for C controlled episodes; no model/scoring."""
import argparse
import hashlib
import json
from pathlib import Path
from sic_paper_data import validate


def prepare(plan_path):
    path=Path(plan_path);plan=json.loads(path.read_text())
    validate(plan)
    dev=[e['episode'] for e in plan['episodes'] if e['partition']=='development']
    test=[e['episode'] for e in plan['episodes'] if e['partition']=='test']
    if len(dev)!=8 or len(test)!=16:raise ValueError('expected frozen 8/16 split')
    if path.with_suffix('.test_opened.json').exists():raise ValueError('original SIC sealed test already opened')
    if path.with_suffix('.s1d_test_opened.json').exists():raise ValueError('S1d sealed test already opened')
    return dict(status='prepared_unscored',plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        internal_plan_sha256=plan['plan_sha256'],context_type=plan['context_type'],
        development_episode_ids=dev,test_episode_count=len(test),
        structural_integrity_verified=True,test_predictions_generated=False,
        test_ledger_created=False,entity_isolation_verified=True,
        next_steps=['adapt C write-time basis/tail training on development episodes only',
                    'freeze method, protocol, source hashes and model fingerprint',
                    'single-open S1d ledger before all16 test episodes; no subsets/reselection'],
        freeze_required=['candidate C or C+RiskProtect selected before test',
                         'rank13, norm1, lr.003, weight_decay.01, steps60, seed0',
                         'fixed B2 FEWSHOT newline stop, max_new_tokens32, official EM',
                         'identical Full/Selection/D0/C contexts, questions and compression budget',
                         'record fact-rank<13 as explicit failure, no unplanned fallback',
                         'writer/basis/tail/protection access write-time statements only',
                         'evaluation questions and answers withheld until checkpoint is frozen'],
        inference=dict(unit='episode',primary='paired EM delta C-D0 and C-Selection',
                       bootstrap_replicates=10000,seed=20261012,confidence=.95,
                       report_failure_count=True,never_use_question_independence=True),
        limitations=['shared corpus/templates; controlled cross-entity generalization only',
                     'four unique natural contexts in Conflict_Resolution',
                     'this preparation performs structural validation including answer consistency, no model scoring'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out)
    if out.exists():raise SystemExit('preparation evidence exists; refuse overwrite')
    result=prepare(args.plan);out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
