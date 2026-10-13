"""Pairing, scorer, checkpoint restoration and no-retraining entry tests."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import types
import torch
from s1d_c_generation import paired_summary, load_verified_c
from readout_probe import FEWSHOT, official_score


def test_metrics():
    maps = dict(Full=dict(a=True,b=True,c=False),
                Selection=dict(a=False,b=True,c=False),
                D0=dict(a=False,b=True,c=False), C=dict(a=True,b=False,c=True))
    result = paired_summary(maps)
    assert result['R'] == result['H'] == 1 and result['net_target'] == 0
    assert result['C_vs_D0']['net'] == 1
    bad = copy.deepcopy(maps); del bad['C']['a']
    try:
        paired_summary(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('unpaired maps accepted')
    assert official_score('Answer: The Beatles\nQuestion: next', ['Beatles'])[0]
    assert not official_score('London and Paris', ['London'])[0]
    assert official_score('London and Paris', ['London'])[1]
    prompt = FEWSHOT.format(question='Synthetic question?')
    assert prompt.endswith('Question: Synthetic question?\nAnswer:')


def test_checkpoint():
    from s1d_experiment1 import sha256_file
    with tempfile.TemporaryDirectory() as d:
        carrier = Path(d)/'carrier'; carrier.write_bytes(b'base')
        ckpt = Path(d)/'alpha.pt'
        basis = torch.eye(13)
        alpha = torch.arange(13.)/100
        ra = dict(basis_sha256='basis',final=dict(alpha=alpha.tolist()))
        w = dict(window=[3,17510],sets={},baselines={},n_supervision_lines=40,arms={'C':ra})
        reference = dict(status='complete',protocol=dict(exp1_arm='C'),driver_sha256='driver',windows=[w])
        args = types.SimpleNamespace(carrier_ckpt=str(carrier),c_checkpoint=str(ckpt))
        displacement = types.SimpleNamespace(basis=basis, value=torch.nn.Parameter(torch.zeros(13)))
        saved = dict(window=w['window'],arm='C',checkpoint_sha256=sha256_file(carrier),
                     driver_sha256='driver',basis=basis,alpha=alpha)
        torch.save(saved,ckpt)
        load_verified_c(args,reference,w,ra,displacement)
        assert torch.equal(displacement.value,alpha) and not displacement.value.requires_grad
        saved['alpha'] = alpha+1; torch.save(saved,ckpt)
        try:
            load_verified_c(args,reference,w,ra,displacement)
        except ValueError:
            pass
        else:
            raise AssertionError('wrong final alpha accepted')


def test_runtime_branch():
    tree = ast.parse(Path(__file__).with_name('s1d_c_runtime.py').read_text())
    generation_branch = next(n for n in ast.walk(tree) if isinstance(n,ast.If)
        and isinstance(n.test,ast.Attribute) and n.test.attr=='c_checkpoint')
    calls = [n for n in ast.walk(generation_branch) if isinstance(n,ast.Call)]
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('step','backward') for n in calls)
    assert any(isinstance(n,ast.Return) for n in ast.walk(generation_branch))
    original=ast.parse(Path(__file__).with_name('s1d_experiment1.py').read_text())
    for name in ('train_arm','backward_fact_protection','fact_basis','candidate','fresh_window'):
        old=next(n for n in ast.walk(original) if isinstance(n,ast.FunctionDef) and n.name==name)
        new=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False),name


def test_generation_restores_each_question():
    import s1d_c_generation as module
    source=Path(module.__file__).read_text()
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_four_arms')
    for child in ast.walk(node):
        if isinstance(child,ast.Constant) and child.value=='cuda:0':child.value='cpu'
    namespace=dict(module.__dict__)
    exec(compile(ast.Module(body=[node],type_ignores=[]),source,'exec'),namespace)
    fn=namespace['evaluate_four_arms']
    class Reader(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.memory=torch.nn.Parameter(torch.zeros(1,40,2),requires_grad=False)
            self.register_buffer('memory_active_mask',torch.tensor([False]*20+[True]*20))
            self.seen=[]
        def generate(self,input_ids,**kwargs):
            assert kwargs['stop_strings']==['\n'] and kwargs['max_new_tokens']==32
            self.seen.append(self.memory.detach().clone())
            # Writes through .data evade Tensor._version; the real evaluator
            # must still reset the candidate before the following question.
            self.memory.data.add_(1)
            return torch.cat([input_ids,torch.tensor([[2]])],dim=1)
    class Tokenizer:
        pad_token_id=0;eos_token_id=0
        def __call__(self,prompt,**kwargs):
            assert prompt.endswith('Answer:')
            return types.SimpleNamespace(input_ids=torch.tensor([[0,1]]))
        def decode(self,ids,**kwargs):return 'Paris'
    model=Reader();original=model.memory;mask=model.memory_active_mask
    manager=types.SimpleNamespace(used=20,block=1,layout=types.SimpleNamespace(k_anchor=0),tokenizer=Tokenizer())
    probes=[types.SimpleNamespace(question='q%d'%i,answers=['Paris']) for i in range(2)]
    with tempfile.TemporaryDirectory() as d:
        args=types.SimpleNamespace(out=str(Path(d)/'r.json'),generation_smoke=True)
        result=fn(args=args,manager=manager,model=model,probes=probes,
            arms=dict(Full=None,Selection=1,D0=2,C=3),
            candidate_pool=lambda a:(torch.ones(1,40,2)*a,torch.tensor([False]*30+[True]*10)),
            window_report=dict(window=[3,17510],baselines={}),
            reference=dict(windows=[dict(arms={'C':dict(final=dict(new_recovery_questions=['q0']))})]),
            covered=[],lines=['written fact'],checkpoint_sha='c',carrier_sha='d',driver_sha='s',
            loaded_tf={'q0':True,'q1':True})
        assert result['status']=='complete' and result['source_memory_unchanged']
        assert model.memory is original and not torch.count_nonzero(original)
        assert model.memory_active_mask is mask
        for lo in range(0,8,2):assert torch.equal(model.seen[lo],model.seen[lo+1])


if __name__ == '__main__':
    for test in (test_metrics,test_checkpoint,test_runtime_branch,test_generation_restores_each_question):
        test(); print('PASS '+test.__name__,flush=True)
