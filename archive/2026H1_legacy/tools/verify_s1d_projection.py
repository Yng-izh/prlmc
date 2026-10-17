#!/usr/bin/env python3
"""CPU report/checkpoint/cache audit for the three fixed projected-C variants."""
import argparse
import json
import math
from pathlib import Path

import torch

from s1d_experiment1 import sha256_file, stage2_metric
from s1d_projected_train import validate_c_reference
from s1d_projection import (VARIANTS, NUMERICAL_TOL, build_basis, projection_protocol,
                           select_protection, tensor_sha256, validate_protection_cache)


def verify(report, baseline, fact_cache, checkpoint, protection_cache=None, steps=60):
    def check(ok, message):
        if not ok:
            raise ValueError(message)
    check(report.get('status') == 'complete' and not report.get('preparation_only'), 'not completed training')
    check(report.get('schema_version') == 's1d_c_projected/1', 'schema mismatch')
    p = report['protocol']
    variant = p['projection']['variant']
    dimension, projected = VARIANTS[variant]
    check(p['projection'] == projection_protocol(variant), 'projection protocol drift')
    check(p['steps'] == steps and p['measure_every'] == (1 if steps == 1 else 10), 'step/cadence drift')
    check(p['evaluation_qa_used_for_training'] is False, 'QA training leakage')
    check(p['gradient_reduction'] == 'sum of per-line mean tail4 CE gradients', 'objective drift')
    check(len(report['windows']) == 1, 'window count')
    w = report['windows'][0]
    check(set(w['arms']) == {'C'}, 'wrong arm')
    a, ba = w['arms']['C'], baseline['windows'][0]['arms']['C']
    validate_c_reference(baseline, p, w['window'], w['sets'],
        {k: v['correct'] for k, v in w['baselines'].items() if k in ('full', 'select', 'd0')}, w['n_supervision_lines'])
    for k in ('baselines', 'sets', 'n_supervision_lines'):
        check(w[k] == baseline['windows'][0][k], 'original C drift: '+k)
    for k in ('basis_fact_keys', 'basis_fact_records', 'fact_full_ce', 'fact_d0_ce', 'dropped_blocks'):
        check(a[k] == ba[k], 'fact inputs drift: '+k)
    check(a['original_c13_basis_sha256'] == ba['basis_sha256'], 'original C13 basis mismatch')
    check(a['trainable_dof'] == dimension and a['initial_identity_verified'], 'DOF/init mismatch')
    check(a['initial_correct'] == w['baselines']['d0']['correct'], 'nonidentity init')
    all_records, kept = a['projection_all_fact_records'], a['projection_kept_fact_records']
    check(len(all_records) == len({r['key'] for r in all_records}) == len(ba['fact_d0_ce']), 'fact universe incomplete/duplicate')
    check({r['key'] for r in all_records} == set(ba['fact_d0_ce']), 'fact universe differs')
    check(kept == [r for r in all_records if r['block'] not in a['dropped_blocks']], 'kept partition mismatch')
    keys = select_protection(kept, ba['fact_d0_ce'], ba['fact_d0_ce']) if projected else []
    check(a['protection_fact_keys'] == keys, 'protection selection mismatch')
    check(a['protection_fact_baseline_ce'] == {k: ba['fact_d0_ce'][k] for k in keys}, 'protection CE references mismatch')
    raw = torch.load(fact_cache, map_location='cpu', weights_only=False)
    check(p['fact_cache_sha256'] == sha256_file(fact_cache), 'raw cache file digest mismatch')
    check(raw['basis_fact_keys'] == a['basis_fact_keys'] and raw['checkpoint_sha256'] == p['checkpoint_sha256'], 'raw cache provenance mismatch')
    protected_rows = None
    if projected:
        check(protection_cache is not None, 'missing protection cache')
        check(sha256_file(protection_cache) == report['protection_cache_sha256'] == a['protection_cache_sha256'], 'protection cache digest mismatch')
        saved = torch.load(protection_cache, map_location='cpu', weights_only=False)
        protected_rows = validate_protection_cache(saved, report['source_metadata'], keys, raw['rows'].shape[1])
        for k in ('fact_cache_sha256', 'c_reference_sha256', 'projection_module_sha256'):
            check(report['source_metadata'][k] == p[k], 'metadata mismatch: '+k)
    else:
        check(report['protection_cache_sha256'] is None and not a['protection_fact_keys'], 'C32 has protection')
    expected, audit = build_basis(raw['rows'], dimension, protected_rows)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    basis, alpha = saved['basis'], saved['alpha']
    check(tuple(basis.shape) == tuple(expected.shape) and alpha.shape == (dimension,), 'checkpoint dimensions')
    check(torch.isfinite(basis).all() and torch.isfinite(alpha).all(), 'nonfinite checkpoint')
    # Sign choices and rotations within exactly degenerate singular spaces do
    # not change the allowed update subspace; compare principal cosines.
    check(float(torch.linalg.svdvals(expected.double().T @ basis.double()).min()) > 1-NUMERICAL_TOL, 'basis subspace mismatch')
    check(float((basis.double().T@basis.double()-torch.eye(dimension)).abs().max()) <= NUMERICAL_TOL, 'nonorthonormal saved basis')
    check(tensor_sha256(basis) == saved['projection_basis_sha256'] == a['projection_basis_sha256'] == report['projection_audit']['basis_sha256'], 'basis tensor digest mismatch')
    check(hashlib_basis(basis) == a['basis_sha256'], 'legacy basis digest mismatch')
    check(sha256_file(checkpoint) == a['residual_checkpoint_sha256'], 'checkpoint file digest mismatch')
    for k, expected_value in dict(window=w['window'], arm='C', driver_sha256=report['driver_sha256'],
            checkpoint_sha256=p['checkpoint_sha256'], projection_variant=variant,
            projection_protocol=p['projection'], fact_cache_sha256=p['fact_cache_sha256'],
            protection_cache_sha256=report['protection_cache_sha256']).items():
        check(saved[k] == expected_value, 'checkpoint provenance: '+k)
    check(a['projection_audit'] == report['projection_audit'], 'conflicting basis audit')
    for k, v in audit.items():
        if k == 'basis_sha256':
            continue
        if isinstance(v, list):
            check(len(v) == len(a['projection_audit'][k]) and all(math.isclose(x,y,rel_tol=1e-5,abs_tol=1e-8) for x,y in zip(v,a['projection_audit'][k])), 'audit spectrum: '+k)
        else:
            check(math.isclose(v,a['projection_audit'][k],rel_tol=1e-5,abs_tol=1e-8), 'audit mismatch: '+k)
    check(a['explicit_basis_fp32_bytes'] == basis.numel()*4 and a['explicit_basis_fp16_bytes'] == basis.numel()*2, 'basis storage mismatch')
    cadence = 1 if steps == 1 else 10
    check([h['step'] for h in a['history']] == list(range(cadence,steps+1,cadence)), 'measurement sequence')
    check(a['final'] == a['history'][-1], 'checkpoint cherry-picking')
    for h in a['history']:
        check(set(h['correct']) == set(w['baselines']['d0']['correct']), 'incomplete QA maps')
        check(all(type(x) is bool for x in h['correct'].values()), 'QA values must be boolean')
        m = stage2_metric(h['correct'],w['sets'],w['baselines']['d0']['correct'])
        for k,v in m.items():
            check(h[k] == v, 'metric mismatch: '+k)
        check(h['basis_unchanged'] and len(h['alpha']) == dimension and all(math.isfinite(x) for x in h['alpha']), 'alpha/basis mismatch')
        check(math.isfinite(h['sup_loss']) and h['sup_loss'] >= 0, 'nonfinite loss')
        check(0 <= h['deviation'] <= 1.000001 and 0 <= h['down_delta_norm'] <= 1.00001, 'residual budget violation')
        check(abs(torch.tensor(h['alpha']).norm().item()-h['deviation']) < 1e-6, 'recorded alpha norm mismatch')
        check(h['tf_correct_count'] == sum(h['correct'].values()), 'TF count mismatch')
    check(torch.allclose(alpha,torch.tensor(a['final']['alpha']),atol=0,rtol=0), 'final alpha mismatch')
    check(a['tail_supervision_line_indices'] == ba['tail_supervision_line_indices'], 'supervision lines drift') if 'tail_supervision_line_indices' in ba else None
    f = a['final']
    passed = steps == 60 and f['net_vs_d0'] >= (5 if w['window'][1] == 17510 else 7)
    if w['window'][1] == 17510:
        passed = passed and f['d0_harmed'] <= 4
    return dict(valid=True, variant=variant, window=w['window'], steps=steps,
                gate_passed=passed, R=f['new_recovery'], H=f['d0_harmed'], net=f['net_vs_d0'],
                effective_rank=audit['effective_rank'], recovery_energy_retained=audit['recovery_energy_retained'])


def hashlib_basis(basis):
    import hashlib
    # Legacy C records float64 SVD basis bytes. This new driver records the
    # float32 basis actually passed to the parametrization.
    return hashlib.sha256(basis.contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('report', 'baseline', 'fact-cache', 'checkpoint'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--protection-cache'); parser.add_argument('--steps', type=int, choices=(1,60), default=60)
    args = parser.parse_args()
    baseline = json.loads(Path(args.baseline).read_text())
    report = json.loads(Path(args.report).read_text())
    if report['protocol']['c_reference_sha256'] != sha256_file(args.baseline):
        raise ValueError('baseline file digest mismatch')
    print(json.dumps(verify(report,baseline,args.fact_cache,args.checkpoint,args.protection_cache,args.steps),indent=2))


if __name__ == '__main__':
    main()
