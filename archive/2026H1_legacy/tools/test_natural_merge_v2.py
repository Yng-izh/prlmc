"""CPU execution contracts; fake readers are never used as experiment data."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(Path(__file__).resolve().parent))
from methods.prlmc.slot_merge import SlotMemoryMerge
from methods.prlmc.merge import SharedMemoryMerge
from methods.prlmc.config import PRLMCConfig
from methods.prlmc.layout import ProtectedResidualMemory
from natural_merge_runtime import MergeStream, state_digest, reset
from natural_merge_v2_train import (full_targets, recovery_weights, teacher_kl,
    train_event, recurrent_targets, realized_residual)
from prepare_natural_merge_v2 import TRAIN_V2, selected_qas, validate_v2_plan, prepare
from prepare_natural_merge import PROTOCOL, TRAIN_PROTOCOL, seal, encode_context
from prepare_public_retention import digest, sha
from run_natural_merge_v2 import validate_scopes, command
from natural_merge_v2_training_fit import verify_training_predictions
from summarize_natural_merge_v2 import summarize_variants
from test_public_retention import Reader, Tokenizer, write_record
import test_natural_merge as legacy_tests


class TinyFrozenReader(Reader):
    def __init__(self):
        super().__init__()
        self.num_blocks=40
        self.memory=torch.nn.Parameter(torch.randn(1,80,4),requires_grad=False)
        self.memory_active_mask=torch.ones(80,dtype=torch.bool)

    def forward(self,input_ids,**kwargs):
        features=self.memory[:,self.memory_active_mask].mean((0,1))
        logits=torch.stack((features[0]-features[1],features[2]+features[3]))
        return types.SimpleNamespace(logits=logits[None,None].expand(1,input_ids.shape[1],2))


def tiny_manager():
    model=TinyFrozenReader()
    cfg=PRLMCConfig(k_anchor=0,anchor_source="none",k_max=40,compact_mode="carrier_residual",
        active_memory_mask=True,native_filler_only=False,trigger="none",projection="none",basis_mode="none")
    return ProtectedResidualMemory(model,Tokenizer(),cfg)


def tiny_qa(i=0):
    return dict(id="qa"+str(i),prompt_ids=[0],answer_ids=[1],context_sha256="context"+str(i))


class SlotContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.module=SlotMemoryMerge(2,8,block=2,rank=3,gamma_max=.05,query_chunk_size=3)
        self.source=torch.randn(2,38,8)
        self.indices=list(range(9))

    def test_identity_init_and_budget(self):
        for dtype in (torch.float32,torch.float16):
            source=self.source.to(dtype)
            out=self.module(source,target=18,carrier_indices=self.indices)
            self.assertEqual(out.shape,(2,18,8)); self.assertEqual(out.dtype,dtype)
            self.assertTrue(torch.equal(out,source[:,:18]))
            self.assertTrue(torch.equal(self.module(source,target=38),source))

    def test_carrier_only_is_exact_selection_even_after_training(self):
        with torch.no_grad(): self.module.up.weight.normal_(std=.2)
        mask=torch.zeros(19,dtype=torch.bool); mask[self.indices]=True
        out=self.module(self.source,target=18,carrier_indices=self.indices,source_allow_mask=mask)
        self.assertTrue(torch.equal(out,self.source[:,:18]))
        self.assertEqual(self.module.last_fusion["fusion_source_indices"],[])

    def test_disabled_sources_cannot_change_routing_or_output(self):
        with torch.no_grad(): self.module.up.weight.normal_(std=.2)
        mask=torch.ones(19,dtype=torch.bool);mask[12:]=False
        changed=self.source.clone(); changed[:,24:]*=1000
        a=self.module(self.source,target=18,carrier_indices=self.indices,source_allow_mask=mask)
        b=self.module(changed,target=18,carrier_indices=self.indices,source_allow_mask=mask)
        self.assertTrue(torch.equal(a,b))

    def test_discarded_slot_permutation_preserves_content_fusion(self):
        with torch.no_grad(): self.module.up.weight.normal_(std=.2)
        changed=self.source.clone(); changed[:,18:]=changed[:,18:].flip(1)
        with torch.no_grad():
            a=self.module(self.source,target=18,carrier_indices=self.indices)
            b=self.module(changed,target=18,carrier_indices=self.indices)
        self.assertTrue(torch.allclose(a,b,atol=1e-6,rtol=1e-6))

    def test_chunked_attention_matches_unchunked_and_has_finite_gradient(self):
        with torch.no_grad(): self.module.up.weight.normal_(std=.2)
        other=SlotMemoryMerge(2,8,block=2,rank=3,gamma_max=.05,query_chunk_size=100)
        other.load_state_dict(self.module.state_dict())
        a=self.module(self.source,target=18,carrier_indices=self.indices)
        b=other(self.source,target=18,carrier_indices=self.indices)
        self.assertTrue(torch.allclose(a,b,atol=1e-6))
        a.square().sum().backward()
        for name in ("q.weight","k.weight","down.weight","up.weight","gamma_logit"):
            g=dict(self.module.named_parameters())[name].grad
            self.assertIsNotNone(g); self.assertTrue(torch.isfinite(g).all())
            self.assertGreater(float(g.abs().sum()),0.)

    def test_zero_init_backward_is_finite_and_updates_output_projection(self):
        out=self.module(self.source,target=18,carrier_indices=self.indices)
        out.square().sum().backward()
        self.assertGreater(float(self.module.up.weight.grad.abs().sum()),0.)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.module.parameters() if p.grad is not None))

    def test_residual_stays_inside_relative_rms_bound(self):
        with torch.no_grad(): self.module.up.weight.normal_(std=100.)
        out=self.module(self.source,target=18,carrier_indices=self.indices)
        carriers=self.source[:,:18].reshape(2,9,2,8)
        delta=(out.reshape_as(carriers)-carriers).square().mean((-1,-2)).sqrt()
        rms=carriers.square().mean((-1,-2)).sqrt()
        self.assertTrue((delta<=float(self.module.gamma.detach())*rms+1e-6).all())

    def test_rejects_invalid_carriers_mask_and_target(self):
        for indices in ([0]*9,list(range(8,-1,-1)),list(range(10,19))+[19]):
            with self.assertRaises(ValueError): self.module(self.source,target=18,carrier_indices=indices)
        with self.assertRaises(ValueError): self.module(self.source,target=17)
        with self.assertRaises(ValueError): self.module(self.source,target=18,carrier_indices=self.indices,
            source_allow_mask=torch.ones(19,dtype=torch.float32))


class ObjectiveContracts(unittest.TestCase):
    def test_recovery_weights_are_capped_and_mean_one(self):
        weights=recovery_weights([1.,1.,1.],[0.,2.,100.])
        self.assertAlmostEqual(sum(weights)/3,1.)
        self.assertEqual(weights,[.5,1.,1.5])
        self.assertEqual(recovery_weights([1.,2.],[0.,1.]),[1.,1.])
        with self.assertRaises(ValueError): recovery_weights([],[])

    def test_distillation_renormalizes_cached_distribution_and_backpropagates(self):
        logits=torch.tensor([[1.,2.,3.],[2.,0.,1.]],requires_grad=True)
        teacher=torch.log_softmax(logits.detach(),-1).half()
        loss=teacher_kl(logits,teacher);self.assertLess(abs(float(loss.detach())),1e-5)
        teacher_kl(logits,torch.zeros_like(logits)).backward()
        self.assertGreater(float(logits.grad.abs().sum()),0.)
        with self.assertRaises(ValueError): teacher_kl(logits,teacher[:1])

    def test_stored_residual_detects_fp16_rounding_to_identity(self):
        carriers=torch.ones(1,4,3,dtype=torch.float16)
        unchanged=(carriers.float()+1e-5).half()
        result=realized_residual(unchanged,carriers)
        self.assertEqual(result["stored_changed_fraction"],0.)
        self.assertEqual(result["stored_residual_rms_ratio"],0.)

    def test_gradient_bridge_updates_each_variant_and_preserves_frozen_reader(self):
        for variant,cls in (("A",SharedMemoryMerge),("B",SharedMemoryMerge),("C",SlotMemoryMerge)):
            m=tiny_manager(); module=cls(1,4,block=2,rank=2,gamma_max=.05)
            with torch.no_grad():
                stream=MergeStream(m,module,"Full")
                for i in range(19): stream.write(write_record(i),i)
                m._memory[:,-38:]+=torch.randn_like(m._memory[:,-38:])
            qas=[tiny_qa()];targets=full_targets(m,qas)
            before=state_digest(module); opt=torch.optim.AdamW(module.parameters(),lr=1e-4)
            event=train_event(m,module,opt,qas,targets,variant)
            self.assertNotEqual(state_digest(module),before)
            self.assertEqual(m.used,18);self.assertFalse(any(p.requires_grad or p.grad is not None for p in m.model.parameters()))
            self.assertTrue(torch.isfinite(torch.tensor(event["objective"])))
            self.assertEqual(event["teacher_weight"],0. if variant=="A" else .5)
            self.assertEqual(event["original_qa_count"],1)

    def test_teacher_mismatch_is_rejected_before_training(self):
        m=tiny_manager(); module=SharedMemoryMerge(1,4,block=2,rank=2,gamma_max=.05)
        with self.assertRaisesRegex(ValueError,"provenance"):
            train_event(m,module,torch.optim.AdamW(module.parameters()),[tiny_qa()],{},"B")

    def test_full_teacher_uses_uncompressed_raw_prefixes(self):
        m=tiny_manager(); module=SharedMemoryMerge(1,4,block=2,rank=2,gamma_max=.05)
        initial=m._memory.clone()
        ep=dict(contexts=[dict(qas=[tiny_qa(i)]) for i in range(39)],
                writes=[write_record(i) for i in range(39)])
        references=recurrent_targets(m,module,initial,ep,2)
        self.assertEqual(sorted(references),[20,30]);self.assertEqual(m.used,78)
        self.assertEqual(len(references[20]),20);self.assertEqual(len(references[30]),30)
        self.assertEqual(module.forward_calls,0)
        self.assertEqual(len(m.model.inputs),39)

    def test_recurrent_student_trains_twice_without_replacing_history_by_full(self):
        for variant,cls in (("B",SharedMemoryMerge),("C",SlotMemoryMerge)):
            m=tiny_manager(); module=cls(1,4,block=2,rank=2,gamma_max=.05)
            initial=m._memory.clone()
            ep=dict(contexts=[dict(qas=[tiny_qa(i)]) for i in range(39)],
                writes=[write_record(i) for i in range(39)])
            references=recurrent_targets(m,module,initial,ep,2)
            reset(m,initial); stream=MergeStream(m,module,"Full")
            optimizer=torch.optim.AdamW(module.parameters(),lr=1e-4)
            events=[]
            for i,write in enumerate(ep["writes"]):
                if m.used+m.block>20*m.block:
                    qas=selected_qas(ep["contexts"][:i],2)
                    event=train_event(m,module,optimizer,qas,references[i],variant)
                    events.append((i,event["original_qa_count"],m.used))
                with torch.no_grad(): stream.write(write,i)
            self.assertEqual(events,[(20,20,20),(30,30,20)])
            self.assertEqual(m.used,38)
            self.assertFalse(any(p.requires_grad or p.grad is not None for p in m.model.parameters()))

    def test_rotating_qas_preserves_originals(self):
        contexts=[dict(qas=[dict(id="a"),dict(id="b")]),dict(qas=[dict(id="c")])]
        self.assertEqual([q["id"] for q in selected_qas(contexts,0)],["a","c"])
        self.assertEqual([q["id"] for q in selected_qas(contexts,1)],["b","c"])


class DataAndQueueContracts(unittest.TestCase):
    def test_all_39_contexts_enter_single_merge_supervision_before_recurrent_stage(self):
        seen=set()
        for starts in TRAIN_V2["single_window_starts"]:
            for start in starts: seen.update(range(start,start+19))
        self.assertEqual(seen,set(range(39)))
        self.assertEqual(8*2*4,TRAIN_V2["planned_updates"])

    def test_prepare_restores_raw_qas_without_modifying_original_plan(self):
        import prepare_natural_merge_v2 as preparation
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"squad").mkdir()
            rows=[]; raw=[]
            for i in range(312):
                context="original paragraph "+str(i)+" A"
                qas=[dict(id="q"+str(i)+suffix,question="original question "+suffix,
                    answers=[dict(text="A",answer_start=len(context)-1)],is_impossible=False) for suffix in ("a","b")]
                raw.append(dict(title="train-"+str(i),paragraphs=[dict(context=context,qas=qas)]))
                rows.append(dict(context=context,context_sha256=digest(context),article="train-"+str(i),
                    article_sha256=digest("train-"+str(i)),qas=[dict(id=q["id"],question=q["question"],answers=["A"],
                    original_answers=q["answers"]) for q in qas]))
            raw_path=root/"squad/train-v2.0.json";raw_path.write_text(json.dumps(dict(data=raw)))
            manifest=dict(repo="pinned-original",revision="fixed",files={"squad/train-v2.0.json":sha(raw_path)})
            (root/"public_dataset_manifest.json").write_text(json.dumps(manifest))
            episodes=[];tok=Tokenizer()
            for ei in range(8):
                contexts=[encode_context(r,i,tok,True) for i,r in enumerate(rows[ei*39:(ei+1)*39])]
                writes=[dict(write_record(i),truncated_tokens=0) for i in range(39)]
                episodes.append(dict(id="stream"+str(ei),contexts=contexts,writes=writes))
            base=seal(dict(schema="natural_merge_plan/1",protocol=PROTOCOL,partition="train",experiment="train",
                tokenizer={},training=TRAIN_PROTOCOL,dataset=manifest,split_audit=dict(original_files=manifest["files"]),episodes=episodes))
            base_path=root/"base.json";base_path.write_text(json.dumps(base));before=sha(base_path)
            args=types.SimpleNamespace(base_train_plan=str(base_path),dataset_dir=str(root),model_path="fake",out=str(root/"v2.json"))
            transformers=types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**kw:tok))
            import io
            with patch.dict(sys.modules,{"transformers":transformers}),patch.object(preparation,"tokenizer_manifest",return_value={}),patch("sys.stdout",new=io.StringIO()):
                prepare(args)
            self.assertEqual(sha(base_path),before)
            plan=json.loads((root/"v2.json").read_text());validate_v2_plan(plan)
            self.assertEqual(sum(len(c["qas"]) for e in plan["episodes"] for c in e["contexts"]),624)
            bad=copy.deepcopy(plan);bad["episodes"][0]["contexts"][0]["qas"][0]["answer_ids"]=[99]
            bad.pop("content_sha256");bad["content_sha256"]=digest(bad)
            with self.assertRaises(ValueError):validate_v2_plan(bad)
            bad=copy.deepcopy(plan);bad["base_plan"]["partition"]="development"
            bad.pop("content_sha256");bad["content_sha256"]=digest(bad)
            with self.assertRaises(ValueError):validate_v2_plan(bad)

    def test_launch_commands_keep_smoke_ineligible_and_evaluation_frozen(self):
        args=types.SimpleNamespace(model_path="model",train_plan="train",dev_plan="dev")
        smoke=command(args,Path("run"),"C",2,"smoke")
        train=command(args,Path("run"),"C",2,"train")
        evaluation=command(args,Path("run"),"C",2,"eval")
        fit=command(args,Path("run"),"C",2,"fit")
        self.assertIn("--smoke-updates",smoke);self.assertNotIn("--smoke-updates",train)
        self.assertIn("run/C.train.pt",evaluation);self.assertNotIn("--legacy-diagnostic",evaluation)
        self.assertNotIn("--freeze",evaluation)
        self.assertIn("train",fit);self.assertIn("run/C.train.pt",fit)
        self.assertIn("run/C.training_fit.json",fit)

    def test_training_fit_prediction_checks_do_not_issue_development_certificate(self):
        result,plan=legacy_tests.EvidenceContracts()._run_fixture()
        verify_training_predictions(result["episodes"][0],plan["episodes"][0],Tokenizer())
        bad=copy.deepcopy(result["episodes"][0])
        bad["arms"]["SharedMerge"]["predictions"][0]["scores"]["author_containment"]=False
        with self.assertRaises(ValueError):verify_training_predictions(bad,plan["episodes"][0],Tokenizer())

    def test_summary_rejects_mixed_controls_and_variant(self):
        # Existing fake-reader fixture already exercises complete E1 rescoring.
        helper=legacy_tests.EvidenceContracts()
        result,plan=helper._run_fixture()
        results={v:copy.deepcopy(result) for v in ("A","B","C")}
        for v,r in results.items():
            r.update(variant=v,architecture=TRAIN_V2["variants"][v]["architecture"])
        summary=summarize_variants(results,plan)
        self.assertEqual(summary["C_vs_B"]["net"],0)
        bad=copy.deepcopy(results);bad["C"]["architecture"]="block_v1"
        with self.assertRaises(ValueError):summarize_variants(bad,plan)
        bad=copy.deepcopy(results);bad["C"]["model"]={"different":True}
        with self.assertRaises(ValueError):summarize_variants(bad,plan)


if __name__=="__main__": unittest.main()
