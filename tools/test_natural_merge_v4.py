"""Meaningful CPU contracts for independent EOS, QA coverage and native skip."""
import copy
import json
from pathlib import Path
import sys
import types
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from methods.prlmc.native_skip_merge import SupervisedDSlotMerge,NativeSkipSlotMerge
from prepare_natural_merge_v3 import build_plan as build_v3
from prepare_natural_merge_v4 import (build_plan,validate_v4_plan,check_answer_encoding,
                                    event_qas,TRAIN_V4)
from natural_merge_v4_objective import (NativeOrigins,full_targets,train_event,
    make_optimizer,balanced_coefficients,objective_terms)
from natural_merge_runtime import MergeStream,reset,state_digest,build_merge,validate_merge_checkpoint
from test_natural_merge_v2 import tiny_manager,tiny_qa
from test_public_retention import write_record


class Contracts(unittest.TestCase):
    def test_native_skip_identity_gradients_mask_and_stored_budget(self):
        torch.manual_seed(12)
        a=SupervisedDSlotMerge(4,8,block=2,rank=3,num_groups=2,reference_layer=2,topk=3)
        b=NativeSkipSlotMerge(4,8,block=2,rank=3,num_groups=2,reference_layer=2,topk=3)
        b.load_state_dict(a.state_dict(),strict=False)
        source=torch.randn(4,38,8);kwargs=dict(target=18,carrier_indices=list(range(10,19)))
        self.assertTrue(torch.equal(a(source,**kwargs),source[:,20:]))
        self.assertTrue(torch.equal(a(source,**kwargs),b(source,**kwargs)))
        (b(source,**kwargs)*torch.randn(4,18,8)).sum().backward()
        self.assertEqual(b.native_gain.shape,(2,))
        self.assertGreater(b.native_gain.grad.abs().sum().item(),0.)
        self.assertGreater(b.up_weight.grad.abs().sum().item(),0.)
        with torch.no_grad():
            b.native_gain.fill_(2);b.up_weight.normal_();b.slot_gate.bias.fill_(8)
        for dtype in (torch.float32,torch.float16):
            s=source.to(dtype);out=b(s,**kwargs)
            ratio=(out.float()-s[:,20:].float()).norm()/s[:,20:].float().norm()
            self.assertLessEqual(ratio.item(),.050001)
        mass=b.last_fusion["per_discarded_block_attention_mass"]
        self.assertEqual(set(mass),{str(i) for i in range(10)})
        self.assertAlmostEqual(sum(mass.values()),1.,places=5)
        mask=torch.zeros(19,dtype=torch.bool);mask[10:]=True
        self.assertTrue(torch.equal(b(source,source_allow_mask=mask,**kwargs),source[:,20:]))
        mask[:2]=True
        changed=source.clone();changed[:,4:20]*=100
        self.assertTrue(torch.equal(b(source,source_allow_mask=mask,**kwargs),
                                    b(changed,source_allow_mask=mask,**kwargs)))
        out=b(source,**kwargs);b.zero_grad();out.square().sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in b.parameters() if p.grad is not None))

    def test_content_teacher_and_EOS_are_independent(self):
        logits=torch.tensor([[3.,0.],[0.,0.]],requires_grad=True);gold=torch.tensor([0,1])
        coef=dict(recovery=1.,protect=0.,teacher=0.,eos=1.)
        ce,kd,rec,dist,protect,eos_ce,eos=objective_terms(logits,gold,0.,None,coef)
        logits2=logits.detach().clone();logits2[-1]=torch.tensor([9.,-9.])
        result=objective_terms(logits2,gold,0.,None,coef)
        self.assertEqual(ce.item(),result[0].item());self.assertGreater(result[-1].item(),eos.item())
        (rec+eos).backward();self.assertGreater(logits.grad[-1].abs().sum().item(),0.)

    def test_context_balancing_and_multiple_QAs_origin_cohorts(self):
        qs=[dict(source_index=0),dict(source_index=0),dict(source_index=1),dict(source_index=2)]
        cohorts=[("recover","current_window")]*3+[("protect","current_window")]
        c=balanced_coefficients(qs,cohorts,[True,True,True,False],[1.]*4,[2.]*4)
        self.assertAlmostEqual(c[0]["recovery"]+c[1]["recovery"],c[2]["recovery"])
        self.assertAlmostEqual(sum(x["eos"] for x in c),1.)
        self.assertAlmostEqual(sum(x["teacher"] for x in c),1.)
        origins=NativeOrigins()
        for i in range(20):origins.append(i)
        qas=[dict(source_index=i) for i in range(20)]+[dict(source_index=0)]
        rows=origins.cohorts(qas,list(range(10,19)))
        self.assertEqual(rows[0],rows[-1]);self.assertEqual(rows[19][0],"protect")

    def test_immutable_plan_full_QA_coverage_and_actual_EOS(self):
        p=ROOT/"reports/natural_merge_v2_20261014/train_plan.json"
        if not p.exists():self.skipTest("archived original plan required")
        class Tokens:
            eos_token_id=99999
            def __call__(self,text,**kwargs):return types.SimpleNamespace(input_ids=[ord(c) for c in text])
        original=build_v3(json.loads(p.read_text()),Tokens(),"v2")
        before=copy.deepcopy(original);plan=build_plan(original,Tokens(),"v3")
        self.assertEqual(original,before);validate_v4_plan(plan);check_answer_encoding(plan,Tokens())
        all_ids={q["id"] for e in plan["episodes"] for c in e["contexts"] for q in c["qas"]}
        seen={i for e in plan["qa_events"] for i in e["qa_ids"]}
        self.assertEqual(all_ids,seen);self.assertEqual(len(all_ids),1402)
        for event in plan["qa_events"]:
            ep=next(e for e in plan["episodes"] if e["id"]==event["stream"])
            key="window_start" if event["stage"]=="single_merge" else "before_write"
            qas=event_qas(plan,event["round"],ep,event["stage"],event[key])
            self.assertEqual([q["id"] for q in qas],event["qa_ids"])
            self.assertTrue(all(q["answer_ids"][-1]==99999 and q["content_ids"][0]==ord(" ") for q in qas))

    def test_reader_gradient_bridge_both_variants_multiple_QAs(self):
        for variant,cls in [("Dsup",SupervisedDSlotMerge),("Dnative",NativeSkipSlotMerge)]:
            manager=tiny_manager();merge=cls(1,4,block=2,rank=2,num_groups=1,reference_layer=0,topk=2)
            initial=manager._memory.clone();reset(manager,initial)
            stream=MergeStream(manager,merge,"Full");origins=NativeOrigins()
            for i in range(19):
                stream.write(write_record(i),i);origins.append(i)
            qas=[]
            for i in range(19):
                q=dict(tiny_qa(i),source_index=i,content_ids=[1],answer_ids=[1,0])
                qas.append(q)
            qas.append(dict(qas[0],id="additional-original-qa"))
            targets=full_targets(manager,qas);opt=make_optimizer(merge);before=state_digest(merge)
            event=train_event(manager,merge,opt,qas,targets,variant,origins)
            self.assertNotEqual(before,state_digest(merge));self.assertGreater(event["eos_loss"],0.)
            self.assertEqual(event["original_qa_count"],20)
            self.assertLessEqual(event["realized_residual"]["stored_residual_rms_ratio"],.050001)
            self.assertFalse(any(p.requires_grad or p.grad is not None for p in manager.model.parameters()))

    def test_runtime_checkpoint_architecture_disambiguation(self):
        for v,spec in TRAIN_V4["variants"].items():
            module=build_merge(spec["architecture"],32,8,2)
            self.assertEqual(module.architecture,spec["architecture"])
            saved=dict(variant=v,training=TRAIN_V4,module_configuration=spec,effective_gamma_max=.05,
                       global_rms_budget=.05,rank=32,evaluation_eligible=True)
            validate_merge_checkpoint(saved,spec["architecture"])
            bad=copy.deepcopy(saved);bad["variant"]="D"
            with self.assertRaises(ValueError):validate_merge_checkpoint(bad,spec["architecture"])

if __name__=="__main__":unittest.main()
