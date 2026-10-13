#!/usr/bin/env python3
"""Verify C invariants, write-time selection and all measured QA metrics."""
import argparse
import json
import math
from pathlib import Path

from s1d_experiment1 import stage2_metric
from s1d_risk_protection import risk_protocol, select_risk_facts


def verify(report, baseline, steps=60):
    def check(condition, message):
        if not condition:
            raise ValueError(message)
    check(report['status'] == baseline['status'] == 'complete', 'incomplete report')
    p, bp = report['protocol'], baseline['protocol']
    check(report['schema_version'] == 's1d_c_risk/1', 'wrong schema')
    check(p['risk_protection'] == risk_protocol(), 'risk protocol drift')
    check(p['steps'] == steps and p['exp1_arm'] == 'C', 'steps/arm drift')
    for key in ('checkpoint_sha256', 'selfsup_rows_sha256', 'lr', 'budget',
                'qa_max_tokens', 'skip_generation', 'seed', 'optimizer',
                'weight_decay', 'fact_delta_threshold', 'fact_full_ce_max', 'fact_max'):
        check(p[key] == bp[key], 'C protocol drift: ' + key)
    check(p['evaluation_qa_used_for_training'] is False, 'QA training leak')
    check(len(report['windows']) == len(baseline['windows']) == 1, 'window count')
    w, bw = report['windows'][0], baseline['windows'][0]
    for key in ('window', 'sets', 'baselines', 'n_supervision_lines'):
        check(w[key] == bw[key], 'baseline/cohort drift: ' + key)
    check(set(w['arms']) == {'C'}, 'arm count')
    a, ba = w['arms']['C'], bw['arms']['C']
    for key in ('basis_type', 'loss_type', 'basis_sha256', 'basis_fact_keys',
                'basis_fact_records', 'fact_full_ce', 'fact_d0_ce', 'dropped_blocks'):
        check(a[key] == ba[key], 'frozen basis/input drift: ' + key)
    check(a['trainable_dof'] == 13 and a['initial_identity_verified'], 'alpha/init drift')
    # Check candidate completeness against the original C complete CE key
    # universe, and partition using recorded dropped blocks.
    rows = a['risk_candidate_records']
    all_rows = a['risk_all_fact_records']
    check({x['key'] for x in all_rows} == set(ba['fact_d0_ce']), 'incomplete fact universe')
    check(len(all_rows) == len(ba['fact_d0_ce']), 'duplicate fact universe')
    check(rows == [x for x in all_rows if x['block'] not in a['dropped_blocks']], 'kept candidate omission')
    check(len({x['key'] for x in rows}) == len(rows), 'duplicate candidate')
    check(all(x['block'] not in a['dropped_blocks'] for x in rows), 'dropped candidate')
    expected = select_risk_facts(rows, a['fact_d0_ce'], a['fact_d0_ce'])
    check(expected and a['protection_fact_keys'] == expected, 'risk selection drift')
    by_key = {x['key']: x for x in rows}
    check(a['protection_fact_records'] == [by_key[k] for k in expected], 'record mismatch')
    check(a['protection_fact_baseline_ce'] == {k: a['fact_d0_ce'][k] for k in expected}, 'D0 refs drift')
    cadence = 1 if steps == 1 else 10
    check([h['step'] for h in a['history']] == list(range(cadence, steps+1, cadence)), 'missing steps')
    check(a['final'] == a['history'][-1], 'wrong final')
    for h in a['history']:
        m = stage2_metric(h['correct'], w['sets'], w['baselines']['d0']['correct'])
        for k, v in m.items():
            check(h[k] == v, 'metric mismatch: ' + k)
        check(h['basis_unchanged'] and len(h['alpha']) == 13, 'basis/alpha drift')
        check(all(math.isfinite(x) for x in h['alpha']), 'nonfinite alpha')
        check(0 <= h['deviation'] <= 1.000001, 'budget violated')
        for key in ('sup_loss', 'tail_loss_sum', 'tail_loss_mean', 'protect_penalty'):
            check(math.isfinite(h[key]), 'nonfinite loss')
        check(h['protect_penalty'] >= 0, 'negative penalty')
        check(abs(h['sup_loss']-h['tail_loss_sum']-h['protect_penalty']) < 1e-6, 'loss reduction drift')
        check(abs(h['tail_loss_sum']-h['tail_loss_mean']*w['n_supervision_lines']) < 1e-5, 'tail scale drift')
        check(h['tf_correct_count'] == sum(h['correct'].values()), 'TF count mismatch')
    gate = risk_protocol()['weak_gate' if w['window'][1] == 17510 else 'strong_gate']
    f = a['final']
    passed = steps == 60 and f['net_vs_d0'] >= gate['min_net'] and f['d0_harmed'] <= gate['max_harmed']
    return dict(valid=True, steps=steps, window=w['window'], gate_passed=passed,
                R=f['new_recovery'], H=f['d0_harmed'], net=f['net_vs_d0'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('report'); p.add_argument('baseline'); p.add_argument('--steps', type=int, default=60)
    p.add_argument('--out')
    args = p.parse_args()
    result = verify(json.loads(Path(args.report).read_text()), json.loads(Path(args.baseline).read_text()), args.steps)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
