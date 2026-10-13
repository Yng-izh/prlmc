"""CPU checks: genuine subspace projection, rank rejection and driver invariants."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import types
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from s1d_projection import (build_basis, select_protection, tensor_sha256,
                           validate_protection_cache, save_new_cache, projection_protocol,
                           PROTECTION_MAX)
from s1d_projected_train import validate_projection_arguments


def rejects(fn):
    try:
        fn()
    except (ValueError, FileExistsError):
        return
    raise AssertionError('invalid input accepted')


def test_projection_and_dimension():
    generator = torch.Generator().manual_seed(11)
    rows = torch.randn(40,96,generator=generator,dtype=torch.float64)
    protected = torch.randn(15,96,generator=generator,dtype=torch.float64)
    b13,a13 = build_basis(rows,13,protected)
    b32,a32 = build_basis(rows,32,protected)
    raw,araw = build_basis(rows,32)
    assert b32.shape == (96,32) and a32['effective_rank'] == 40
    torch.testing.assert_close(b13,b32[:,:13],atol=1e-7,rtol=1e-6)
    assert a13['recovery_energy_retained'] == a32['recovery_energy_retained'] < 1
    assert a32['retained_energy_in_selected_basis'] >= a13['retained_energy_in_selected_basis']
    assert a32['protection_max_abs_directional_ce'] < 1e-6
    assert (protected@b32.double()).abs().max() < 1e-5
    assert (protected@raw.double()).abs().max() > .01
    assert araw['recovery_energy_retained'] == 1
    # This projection really changes the span, rather than rotating C's basis.
    q = torch.linalg.qr(rows.T).Q
    assert (b32.double()-q@(q.T@b32.double())).norm() > .1
    from s1c_wash import make_arm_displacement
    d = make_arm_displacement(torch.zeros(12,8),b32,torch)
    assert d.value.numel() == 32 and torch.equal(d(torch.zeros(12,8)),torch.zeros(12,8))
    with torch.no_grad(): d.value.fill_(1)
    d.clamp(1)
    assert d.value.norm() <= 1.000001 and d(torch.zeros(12,8)).norm() <= 1.000001


def test_rank_and_numerical_rejection():
    eye = torch.eye(80,dtype=torch.float64)
    rejects(lambda: build_basis(eye[:20],32))
    rejects(lambda: build_basis(eye[:40],32,eye[:15]))
    rejects(lambda: build_basis(eye[:40],13,eye[:40]))
    rejects(lambda: build_basis(torch.zeros(40,80),13))
    rejects(lambda: build_basis(eye[:40],13,torch.zeros(2,80)))
    rejects(lambda: build_basis(eye[:40],13,torch.eye(81)[:2]))
    bad=eye[:40].clone();bad[0,0]=float('nan')
    rejects(lambda: build_basis(bad,13))
    rejects(lambda: build_basis(eye[:40],64))


def test_selection_and_cache():
    rows=[dict(key=k) for k in 'abcdef']
    ce=dict(a=0,b=2,c=2.01,d=-1,e=float('nan'),f=.5)
    assert select_protection(rows,'abcde',ce)==['a','b']
    assert select_protection(list(reversed(rows)),'abcdef',ce)==['a','b','f']
    many = [dict(key='x%03d' % i) for i in range(PROTECTION_MAX + 5)]
    many_ce = {r['key']:.5 for r in many}
    assert len(select_protection(many, many_ce, many_ce)) == PROTECTION_MAX
    rejects(lambda: select_protection(rows+rows[:1],'abcdef',ce))
    tensor=torch.eye(5)[:3].double();keys=['a','b','f'];meta=dict(window=[3,17510],carrier='x')
    saved=dict(schema='s1d_protection_rows/1',metadata=meta,keys=keys,rows=tensor,rows_sha256=tensor_sha256(tensor))
    torch.testing.assert_close(validate_protection_cache(saved,meta,keys,5),tensor)
    rejects(lambda: validate_protection_cache(saved,dict(window=[3,16370]),keys,5))
    rejects(lambda: validate_protection_cache(saved,meta,list(reversed(keys)),5))
    damaged=copy.deepcopy(saved);damaged['rows'][0,0]=2
    rejects(lambda: validate_protection_cache(damaged,meta,keys,5))
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'shared.pt';save_new_cache(saved,path)
        before=path.read_bytes();rejects(lambda: save_new_cache(damaged,path))
        assert path.read_bytes()==before and len(list(Path(tmp).iterdir()))==1


def test_driver_invariants():
    parent=ast.parse(Path(__file__).with_name('s1d_experiment1.py').read_text())
    new=ast.parse(Path(__file__).with_name('s1d_projected_train.py').read_text())
    for name in ('candidate','fresh_window','encode','encode_fact_records','fact_records',
                 'supervision_lines','reader_evaluation','evaluate','train_arm','fact_basis',
                 'window_sets','stage2_metric','base'):
        oldnode=next(n for n in ast.walk(parent) if isinstance(n,ast.FunctionDef) and n.name==name)
        newnode=next(n for n in ast.walk(new) if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(oldnode,include_attributes=False)==ast.dump(newnode,include_attributes=False),name
    # Preparation returns before the only active arm can construct an optimizer.
    main=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    preparation=next(n for n in ast.walk(main) if isinstance(n,ast.If) and
        isinstance(n.test,ast.Attribute) and n.test.attr=='prepare_protection_cache')
    assert any(isinstance(n,ast.Return) for n in ast.walk(preparation))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
                   n.func.attr in ('step','backward','AdamW') for n in ast.walk(preparation))
    assert preparation.lineno < next(n.lineno for n in ast.walk(main) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Attribute) and n.func.attr=='AdamW')
    # Collector shares the exact differentiated CE/pool binding from C's original collector.
    original=next(n for n in ast.walk(parent) if isinstance(n,ast.FunctionDef) and n.name=='fact_basis')
    collector=next(n for n in ast.walk(new) if isinstance(n,ast.FunctionDef) and n.name=='fact_gradient_rows')
    assert ast.dump(original.body[1],include_attributes=False)==ast.dump(collector.body[1],include_attributes=False)
    for a,b in zip(original.body[1:5],collector.body[1:5]):
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)


def test_protocol_rejection():
    args=types.SimpleNamespace(exp1_arm='C',risk_protection=False,fact_cache='raw.pt',c_reference='c.json',
        skip_generation=True,lr=.003,budget=1.,qa_max_tokens=256,fact_delta_threshold=.1,
        fact_full_ce_max=2.,fact_max=40,steps=60,measure_every=10,window='3,17510',
        prepare_protection_cache=False,basis_variant='P32',protection_rows='kept.pt',out='new.json')
    validate_projection_arguments(args)
    for key,value in [('risk_protection',True),('exp1_arm','D'),('budget',2),('steps',20),
                      ('measure_every',5),('window','3,17890'),('protection_rows',None),('out','same.pt')]:
        bad=copy.copy(args);setattr(bad,key,value)
        rejects(lambda: validate_projection_arguments(bad))


def test_new_collector_actual_gradients():
    from methods.prlmc.probes import _pool_in_place, differentiable_memory, active_mask_override
    from torch.utils.checkpoint import checkpoint
    import torch.nn.functional as F
    class Reader(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.memory=torch.nn.Parameter(torch.arange(12.).reshape(1,6,2)/20,requires_grad=False)
            self.register_buffer('memory_active_mask',torch.ones(6,dtype=torch.bool))
            self.projection=torch.arange(20.).reshape(4,5)/30
        def logits(self,ids):
            return (self.memory[:,-2:].reshape(4)@self.projection).expand(1,ids.shape[1],5)
        def forward(self,input_ids,**kw):
            return types.SimpleNamespace(logits=checkpoint(self.logits,input_ids,use_reentrant=False))
    model=Reader();memory=model.memory;snapshot=memory.detach().clone();oldmask=model.memory_active_mask
    arm=types.SimpleNamespace(down=torch.nn.Linear(2,2,bias=False))
    arm.down.weight.requires_grad_(False)
    enc={'a':(torch.tensor([[0,1,2]]),1),'b':(torch.tensor([[1,2,4]]),2)}
    expected=[]
    arm.down.weight.requires_grad_(True)
    for ids,boundary in enc.values():
        score=(arm.down.weight.reshape(4)@model.projection).expand(ids.shape[1]-boundary,5)
        loss=F.cross_entropy(score,ids[0,boundary:])
        expected.append(torch.autograd.grad(loss,arm.down.weight)[0].reshape(-1).double())
    arm.down.weight.requires_grad_(False)
    ns=dict(torch=torch,F=F,model=model,manager=types.SimpleNamespace(model=model,total=6,used=4),
        candidate=lambda a:a.down.weight.reshape(1,2,2),_pool_in_place=_pool_in_place,
        differentiable_memory=differentiable_memory,active_mask_override=active_mask_override,
        window_used=4,target_used=2)
    tree=ast.parse(Path(__file__).with_name('s1d_projected_train.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='fact_gradient_rows')
    for child in ast.walk(node):
        if isinstance(child,ast.Constant) and child.value=='cuda:0':child.value='cpu'
    exec(compile(ast.Module(body=[node],type_ignores=[]),'collector','exec'),ns)
    actual=ns['fact_gradient_rows'](arm,enc,list(enc),torch.tensor([False]*4+[True]*2))
    torch.testing.assert_close(actual,torch.stack(expected))
    assert not arm.down.weight.requires_grad and arm.down.weight.grad is None
    assert model.memory is memory and torch.equal(memory,snapshot) and model.memory_active_mask is oldmask


if __name__=='__main__':
    for test in (test_projection_and_dimension,test_rank_and_numerical_rejection,test_selection_and_cache,
                 test_driver_invariants,test_protocol_rejection,test_new_collector_actual_gradients):
        test();print('PASS '+test.__name__,flush=True)
