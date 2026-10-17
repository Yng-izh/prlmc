#!/usr/bin/env python3
"""Independent official-source EM audit of frozen S1d generated predictions."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import string
from s1d_c_generation import paired_summary


def official_em_scorer(source):
    names={'normalize_answer','drqa_exact_match_score','substring_exact_match_score',
           'drqa_metric_max_over_ground_truths','parse_output'}
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=names:raise ValueError('official scoring functions missing')
    namespace=dict(re=re,string=string)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'official-source-em','exec'),namespace)
    def score(pred,aliases):
        parsed=namespace['parse_output'](pred)
        variants=[pred]+([parsed] if parsed is not None else [])
        return tuple(any(namespace['drqa_metric_max_over_ground_truths'](namespace[name],v,aliases)
                         for v in variants)
                     for name in ('drqa_exact_match_score','substring_exact_match_score'))
    return score


def audit(report,source):
    score=official_em_scorer(source)
    maps={name:{} for name in ('Full','Selection','D0','C')}
    for pair in report['pairs']:
        for name in maps:
            item=pair[name];em,sub=score(item['prediction'],pair['answers'])
            if bool(em)!=item['em'] or bool(sub)!=item['substring_em']:
                raise ValueError('official source score mismatch')
            maps[name][pair['question']]=bool(em)
    if paired_summary(maps)!=report['summary']:raise ValueError('summary mismatch')
    if any(maps[n]!=report['arm_results'][n]['correct'] for n in maps):
        raise ValueError('pair/map mismatch')
    return dict(valid=True,all_predictions_rescored=True,n=len(report['pairs']),
                official_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                summary=report['summary'])


def main():
    p=argparse.ArgumentParser();p.add_argument('report')
    p.add_argument('--official-source',default='/share/latent/MemoryAgentBench/utils/eval_other_utils.py')
    p.add_argument('--out',required=True);args=p.parse_args()
    r=json.loads(Path(args.report).read_text());source=Path(args.official_source).read_text()
    result=audit(r,source);result['report_sha256']=hashlib.sha256(Path(args.report).read_bytes()).hexdigest()
    Path(args.out).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
