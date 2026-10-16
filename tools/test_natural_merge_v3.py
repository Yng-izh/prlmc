"""CPU C/D execution contracts; miniature readers are not research evidence."""
import copy
import json
from pathlib import Path
import sys
import types
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from methods.prlmc.gated_slot_merge import BudgetedSlotMemoryMerge, GraftedSlotMemoryMerge
from natural_merge_runtime import state_digest, build_merge, validate_merge_checkpoint, MergeStream, reset
from natural_merge_v3_objective import NativeOrigins, objective_terms, make_optimizer, train_event
from natural_merge_v2_train import full_targets, recurrent_targets
from prepare_natural_merge_v3 import TRAIN_V3, build_plan, validate_v3_plan, check_answer_encoding
from prepare_natural_merge_v2 import selected_qas
from prepare_public_retention import digest
from run_natural_merge_v3 import command, validate_scopes
from summarize_natural_merge_v3 import summarize_variants
from test_natural_merge_v2 import tiny_manager, tiny_qa
from test_public_retention import write_record
import test_natural_merge as legacy_tests


def modules():
    return [("C", BudgetedSlotMemoryMerge(4, 8, block=2, rank=3)),
            ("D", GraftedSlotMemoryMerge(4, 8, block=2, rank=3, num_groups=2, reference_layer=2, topk=3))]


class ModuleContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.source = torch.randn(4, 38, 8)
        self.carriers = list(range(10, 19))

    def test_exact_identity_and_zero_init_gradient(self):
        for _, module in modules():
            for dtype in (torch.float32, torch.float16):
                source = self.source.to(dtype)
                out = module(source, target=18, carrier_indices=self.carriers)
                self.assertTrue(torch.equal(out, source[:, 20:]))
                self.assertTrue(torch.equal(module.compact(source, target=18, carrier_indices=self.carriers),out))
                self.assertEqual(module.state_report()["gamma"],module.last_gamma)
                self.assertTrue(torch.equal(module(source, target=38), source))
            module.zero_grad()
            out = module(self.source, target=18, carrier_indices=self.carriers)
            (out*torch.randn_like(out)).sum().backward()
            up = module.up_weight if hasattr(module, "up_weight") else module.up.weight
            self.assertGreater(float(up.grad.abs().sum()), 0.)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None))

    def test_stored_global_budget_and_active_gradients(self):
        for _, module in modules():
            with torch.no_grad():
                if hasattr(module, "up_weight"):
                    module.up_weight.normal_(); module.slot_gate.bias.fill_(10.)
                else:
                    module.up.weight.normal_(); module.gamma_logit.fill_(10.)
            for dtype in (torch.float32, torch.float16):
                source = self.source.to(dtype)
                out = module(source, target=18, carrier_indices=self.carriers)
                ratio = (out.float()-source[:,20:].float()).square().sum().sqrt()/source[:,20:].float().square().sum().sqrt()
                self.assertLessEqual(float(ratio.detach()), .050001)
                self.assertGreater(float(ratio.detach()), 0.)
            out = module(self.source, target=18, carrier_indices=self.carriers)
            (out*torch.randn_like(out)).sum().backward()
            for p in module.parameters():
                self.assertIsNotNone(p.grad); self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(module.q.weight.grad.abs().sum()), 0.)
            self.assertGreater(float(module.k.weight.grad.abs().sum()), 0.)

    def test_ablation_removes_disabled_sources_before_routing(self):
        for _, module in modules():
            with torch.no_grad():
                (module.up_weight if hasattr(module, "up_weight") else module.up.weight).normal_()
            mask = torch.zeros(19, dtype=torch.bool); mask[10:] = True
            out = module(self.source, target=18, carrier_indices=self.carriers, source_allow_mask=mask)
            self.assertTrue(torch.equal(out, self.source[:,20:]))
            mask[:3] = True
            changed = self.source.clone(); changed[:,6:20] *= 1000.
            a = module(self.source, target=18, carrier_indices=self.carriers, source_allow_mask=mask)
            b = module(changed, target=18, carrier_indices=self.carriers, source_allow_mask=mask)
            self.assertTrue(torch.equal(a, b))

    def test_grouped_sparse_router_and_checkpoint_roundtrip(self):
        module = modules()[1][1]
        probabilities, indices = module._sparse_attention(torch.randn(9,3), torch.randn(20,3))
        self.assertEqual(indices.shape, (9,3)); self.assertTrue(torch.allclose(probabilities.sum(1), torch.ones(9)))
        with torch.no_grad(): module.up_weight.normal_()
        other = modules()[1][1]; other.query_chunk_size = 5
        other.load_state_dict(module.state_dict(), strict=True)
        a = module(self.source, target=18, carrier_indices=self.carriers)
        b = other(self.source, target=18, carrier_indices=self.carriers)
        self.assertTrue(torch.allclose(a,b,atol=1e-6))
        self.assertEqual(state_digest(module),state_digest(other))
        self.assertNotIn("down.weight", module.state_dict())

    def test_runtime_builds_fixed_C_D_and_rejects_mixed_checkpoint(self):
        for variant in ("C", "D"):
            spec = TRAIN_V3["variants"][variant]
            module = build_merge(spec["architecture"], 32, 8, 2)
            self.assertEqual(module.gamma_max, spec["local_gate_max"])
            saved = dict(variant=variant, training=TRAIN_V3, module_configuration=spec,
                effective_gamma_max=.05, global_rms_budget=.05, rank=32, evaluation_eligible=True)
            validate_merge_checkpoint(saved, spec["architecture"])
            bad = copy.deepcopy(saved); bad["global_rms_budget"] = .2
            with self.assertRaises(ValueError): validate_merge_checkpoint(bad, spec["architecture"])


class ObjectiveContracts(unittest.TestCase):
    def test_recovery_and_protection_do_not_share_CE_objective(self):
        logits = torch.tensor([[1.,2.]],requires_grad=True); gold = torch.tensor([0])
        ce,kd,rec,dist,prot = objective_terms(logits,gold,0.,"recover",1.,None,2,3,0)
        self.assertEqual(float(prot),0.); self.assertAlmostEqual(float(rec.detach()),float(ce.detach())/2)
        ce,kd,rec,dist,prot = objective_terms(logits,gold,2.,"protect",0.,None,2,3,0)
        self.assertEqual(float(rec),0.); self.assertEqual(float(prot.detach()),0.)
        ce,kd,rec,dist,prot = objective_terms(logits,gold,0.,"protect",0.,None,2,3,0)
        self.assertAlmostEqual(float(prot.detach()),float(ce.detach())/3)
        prot.backward(); self.assertGreater(float(logits.grad.abs().sum()),0.)

    def test_recurrent_origin_cohorts_include_old_losses_and_untouched_tail(self):
        origins = NativeOrigins()
        for i in range(20): origins.append(i)
        qas = [dict(source_index=i) for i in range(20)]
        cohorts = origins.cohorts(qas, list(range(10,19)))
        self.assertEqual(sum(c=="recover" for c,_ in cohorts),10)
        self.assertEqual(cohorts[19],("protect","untouched_tail"))
        origins.compact(list(range(10,19)))
        for i in range(20,30): origins.append(i)
        qas = [dict(source_index=i) for i in range(30)]
        cohorts = origins.cohorts(qas,list(range(10,19)))
        self.assertEqual(sum(c=="recover" for c,_ in cohorts),20)
        self.assertEqual(cohorts[0],("recover","previously_discarded"))
        self.assertEqual(cohorts[29],("protect","untouched_tail"))

    def test_common_optimizer_groups(self):
        for _, module in modules():
            optimizer=make_optimizer(module)
            self.assertEqual([g["lr"] for g in optimizer.param_groups],[1e-3,1e-2])
            parameters=[p for g in optimizer.param_groups for p in g["params"]]
            self.assertEqual(len(parameters),len({id(p) for p in module.parameters()}))
            self.assertEqual(len(parameters),len({id(p) for p in parameters}))

    def test_reader_gradient_bridge_and_recurrent_cohorts_both_variants(self):
        for variant in ("C", "D"):
            m=tiny_manager()
            cls=BudgetedSlotMemoryMerge if variant=="C" else GraftedSlotMemoryMerge
            kwargs={} if variant=="C" else dict(num_groups=1,reference_layer=0,topk=2)
            module=cls(1,4,block=2,rank=2,**kwargs)
            initial=m._memory.clone()
            qas=[dict(tiny_qa(i),source_index=i) for i in range(39)]
            ep=dict(contexts=[dict(qas=[q]) for q in qas],writes=[write_record(i) for i in range(39)])
            references=recurrent_targets(m,module,initial,ep,2)
            reset(m,initial); stream=MergeStream(m,module,"Full"); origins=NativeOrigins()
            optimizer=make_optimizer(module); before=state_digest(module); events=[]
            for i,write in enumerate(ep["writes"]):
                if m.used+m.block>20*m.block:
                    event=train_event(m,module,optimizer,qas[:i],references[i],variant,origins)
                    events.append(event)
                with torch.no_grad(): stream.write(write,i)
                origins.append(i)
            self.assertNotEqual(state_digest(module),before)
            self.assertEqual([e["cohorts"]["recover"]["count"] for e in events],[10,20])
            self.assertEqual([e["cohorts"]["protect"]["count"] for e in events],[10,10])
            self.assertEqual(m.used,38); self.assertEqual(len(origins.blocks),19)
            self.assertTrue(all(e["realized_residual"]["stored_residual_rms_ratio"]<=.050001 for e in events))
            self.assertFalse(any(p.requires_grad or p.grad is not None for p in m.model.parameters()))


class PlanAndQueueContracts(unittest.TestCase):
    def test_answer_encoding_preserves_originals_and_handles_fused_suffix(self):
        candidates = (ROOT/"data/natural_merge_v2/train.json",
                      ROOT/"reports/natural_merge_v2_20261014/train_plan.json")
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            self.skipTest("requires the fixed v2 training plan or its archived copy")
        original=json.loads(path.read_text())
        before=digest(original)
        class Tokens:
            def __call__(self,text,**kwargs): return types.SimpleNamespace(input_ids=[ord(c) for c in text])
        plan=build_plan(original,Tokens(),"file-sha")
        self.assertEqual(digest(original),before)
        validate_v3_plan(plan); check_answer_encoding(plan,Tokens())
        for ep in plan["episodes"]:
            for c in ep["contexts"]:
                for q in c["qas"]:
                    self.assertEqual(q["answer_ids"],[ord(x) for x in q["answer"]])
        bad=copy.deepcopy(plan); q=bad["episodes"][0]["contexts"][0]["qas"][0]
        q["answer_ids"] += [99]; q["answer_sha256"]=digest(q["answer_ids"])
        bad.pop("content_sha256"); bad["content_sha256"]=digest(bad)
        with self.assertRaises(ValueError): check_answer_encoding(bad,Tokens())

    def test_queue_and_development_comparison(self):
        args=types.SimpleNamespace(model_path="model",train_plan="train",dev_plan="dev")
        for variant in ("C","D"):
            self.assertIn("--smoke-updates",command(args,Path("run"),variant,1,"smoke"))
            self.assertIn("run/"+variant+".train.pt",command(args,Path("run"),variant,1,"eval"))
            self.assertIn(str(ROOT/"tools/natural_merge_v3_training_fit.py"),command(args,Path("run"),variant,1,"fit"))
        result,plan=legacy_tests.EvidenceContracts()._run_fixture()
        results={v:copy.deepcopy(result) for v in ("C","D")}
        for v,r in results.items():
            r.update(variant=v,architecture=TRAIN_V3["variants"][v]["architecture"],
                training_configuration=TRAIN_V3,train_plan_sha256="train",train_content_sha256="content")
        self.assertEqual(summarize_variants(results,plan)["D_vs_C"]["net"],0)
        results["D"]["train_content_sha256"]="other"
        with self.assertRaises(ValueError): summarize_variants(results,plan)


if __name__ == "__main__": unittest.main()
