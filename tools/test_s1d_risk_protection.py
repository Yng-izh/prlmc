#!/usr/bin/env python3
"""CPU tests of actual nested driver gradients and CE-only risk selection."""
import sys
import types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from methods.prlmc.probes import _pool_in_place, differentiable_memory, active_mask_override
from test_s1d_experiment1 import nested_function
from s1d_risk_protection import select_risk_facts


def test_selection():
    records = [{'key': 'k%d' % i} for i in range(30)]
    encoded = {r['key']: None for r in records}
    ce = {r['key']: i/20 for i, r in enumerate(records)}
    ce.update(k0=float('nan'), k1=float('inf'), k2=-1, k3=2.1)
    result = select_risk_facts(records, encoded, ce)
    assert result == ['k%d' % i for i in range(29, 9, -1)]
    ce['k28'] = ce['k29']
    assert select_risk_facts(list(reversed(records)), encoded, ce)[:2] == ['k28', 'k29']
    del encoded['k29']
    assert 'k29' not in select_risk_facts(records, encoded, ce)
    assert select_risk_facts(records, {}, ce) == []


def test_combined_update():
    class Reader(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.memory = torch.nn.Parameter(torch.arange(12.).reshape(1, 6, 2)/20, requires_grad=False)
            self.register_buffer('memory_active_mask', torch.ones(6, dtype=torch.bool))
            self.projection = torch.arange(20.).reshape(4, 5)/30
        def logits(self, ids):
            assert not self.memory_active_mask[:4].any()
            return (self.memory[:, -2:].reshape(4) @ self.projection).expand(1, ids.shape[1], 5)
        def forward(self, input_ids, **kw):
            return types.SimpleNamespace(logits=checkpoint(self.logits, input_ids, use_reentrant=False))
    model = Reader(); memory = model.memory; saved = memory.detach().clone()
    mask_saved = model.memory_active_mask
    mask = torch.tensor([False]*4+[True]*2)
    basis = torch.tensor([[1., 2.], [3., -1.], [2., 4.], [-1., 2.]])
    basis_saved = basis.clone()
    d0 = torch.tensor([.1, .3, -.2, .4])
    sup = [(1, torch.tensor([[0, 1, 2, 3, 4, 1]])),
           (2, torch.tensor([[1, 0, 3, 2, 1, 4]]))]
    encoded = {'active': (torch.tensor([[0, 2]]), 1), 'inactive': (torch.tensor([[1, 3]]), 1)}
    for refs in (None, {'active': 100., 'inactive': 100.}, {'active': 0., 'inactive': 100.}):
        alpha = torch.nn.Parameter(torch.tensor([.2, -.1]))
        score = (d0+basis@alpha) @ model.projection
        expected = sum(F.cross_entropy(score.expand(4, 5), ids[0, -4:]) for _, ids in sup)
        tail = expected.detach().item()
        if refs is not None:
            expected = expected + sum(torch.relu(F.cross_entropy(score.unsqueeze(0), ids[:, 1])-refs[k])
                                      for k, (ids, _) in encoded.items())/len(encoded)
        expected_grad = torch.autograd.grad(expected, alpha)[0]
        old_alpha = alpha.detach().clone()
        reference_alpha = torch.nn.Parameter(old_alpha.clone())
        ref_optimizer = torch.optim.AdamW([reference_alpha], lr=.003, weight_decay=.01)
        reference_alpha.grad = expected_grad.clone(); ref_optimizer.step()
        namespace = dict(torch=torch, F=F, model=model,
            manager=types.SimpleNamespace(model=model, total=6, used=4),
            candidate=lambda arm: (d0+basis@alpha).reshape(1, 2, 2),
            _pool_in_place=_pool_in_place, differentiable_memory=differentiable_memory,
            active_mask_override=active_mask_override, window_used=4, target_used=2,
            MASK_SPAN=4, args=types.SimpleNamespace(budget=1.0))
        namespace['backward_fact_protection'] = nested_function('backward_fact_protection', namespace)
        fn = nested_function('train_arm', namespace)
        optimizer = torch.optim.AdamW([alpha], lr=.003, weight_decay=.01)
        clamps = []
        displacement = types.SimpleNamespace(value=alpha, clamp=lambda b: clamps.append(b))
        kwargs = {} if refs is None else {'protection': (encoded, list(encoded), refs)}
        result = fn(None, displacement, optimizer, sup, mask, **kwargs)
        torch.testing.assert_close(alpha.grad, expected_grad)
        torch.testing.assert_close(alpha, reference_alpha)
        assert optimizer.state[alpha]['step'] == 1 and clamps == [1.0]
        if refs is None:
            assert abs(result-tail/len(sup)) < 1e-6
        else:
            assert abs(result['loss']-expected.item()) < 1e-6
            assert abs(result['tail_loss_sum']-tail) < 1e-6
            assert (result['protect_penalty'] > 0) == (refs['active'] == 0)
        assert model.memory is memory and torch.equal(memory, saved)
        assert model.memory_active_mask is mask_saved
        assert torch.equal(basis, basis_saved)
        assert all(p.grad is None for p in model.parameters())


def test_verifier_rejects_drift():
    import copy
    import json
    from s1d_experiment1 import stage2_metric
    from s1d_risk_protection import risk_protocol
    from verify_s1d_risk_protection import verify
    # A structurally complete synthetic risk report using archived C baselines;
    # this tests certificate rejection, not a claimed training result.
    base_path = Path(__file__).resolve().parents[1]/'reports/s1d_exp1_fact_v3/s1d_exp1_fact_v3.C.w17510.json'
    if not base_path.exists():
        base_path = Path(__file__).resolve().parents[1]/'runs/s1d_exp1_fact_v3.C.w17510.json'
    baseline = json.loads(base_path.read_text()); report = copy.deepcopy(baseline)
    report['schema_version'] = 's1d_c_risk/1'
    report['protocol']['risk_protection'] = risk_protocol()
    w = report['windows'][0]; a = w['arms']['C']
    drop = a['dropped_blocks'][0]
    kept = next(i for i in range(20) if i not in a['dropped_blocks'])
    basis_keys = set(a['basis_fact_keys'])
    all_rows = [{'key': k, 'block': drop if k in basis_keys else kept}
                for k in a['fact_d0_ce']]
    rows = [x for x in all_rows if x['block'] == kept]
    a['risk_all_fact_records'] = all_rows; a['risk_candidate_records'] = rows
    keys = select_risk_facts(rows, a['fact_d0_ce'], a['fact_d0_ce'])
    a['protection_fact_keys'] = keys
    a['protection_fact_records'] = [next(x for x in rows if x['key'] == k) for k in keys]
    a['protection_fact_baseline_ce'] = {k: a['fact_d0_ce'][k] for k in keys}
    for h in a['history']:
        h.update(tail_loss_mean=1., tail_loss_sum=float(w['n_supervision_lines']),
                 protect_penalty=0., sup_loss=float(w['n_supervision_lines']))
    a['final'] = a['history'][-1]
    result = verify(report, baseline)
    assert result['valid'] and not result['gate_passed']
    mutations = [lambda r: r['protocol']['risk_protection'].update(weight=2),
        lambda r: r['windows'][0]['arms']['C']['protection_fact_keys'].reverse(),
        lambda r: r['windows'][0]['arms']['C']['risk_candidate_records'].pop(),
        lambda r: r['windows'][0]['arms']['C']['final'].update(net_vs_d0=100),
        lambda r: r['windows'][0]['arms']['C'].update(basis_sha256='bad')]
    for mutate in mutations:
        changed = copy.deepcopy(report); mutate(changed)
        try:
            verify(changed, baseline)
        except ValueError:
            continue
        raise AssertionError('verifier accepted modified evidence')


if __name__ == '__main__':
    for test in (test_selection, test_combined_update, test_verifier_rejects_drift):
        test(); print('PASS '+test.__name__, flush=True)
