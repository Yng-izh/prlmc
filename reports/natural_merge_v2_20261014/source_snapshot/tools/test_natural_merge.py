"""CPU tests of causal source exclusion, recurrent placement and gradient path.

Tiny fake readers test execution contracts only. They are never experiment data.
"""
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
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from methods.prlmc.consolidate import CarrierResidualCompactor
from methods.prlmc.merge import SharedMemoryMerge
from prepare_natural_merge import (PROTOCOL,TRAIN_PROTOCOL,raw_squad,split_rows,encode_context,episode,seal,validate_plan)
from prepare_public_retention import digest
from natural_merge_runtime import (MergeStream,install,native_capacity,scores,state_digest,reset)
from natural_merge_train import train_event
from natural_merge_eval import e1_episode,retention_episode,validate_checkpoint_scope,validate_e2_gate
from verify_natural_merge import verify,summarize,paired
from run_natural_merge_parallel import merge_reports
from test_public_retention import manager,Tokenizer,write_record,Reader


def qa_fixture(index=0):
    context_hash=digest("original-context-"+str(index))
    return dict(id="q"+str(index),question="original question?",answer="A",answers=["A"],
        prompt="Q",prompt_ids=[ord("Q")],prompt_sha256=digest([ord("Q")]),answer_ids=[ord("A")],
        scoring_target="A",context_sha256=context_hash,source_index=index)


def fixture_plan(e1=True,episodes=1,writes=31):
    eps=[]
    for ei in range(episodes):
        contexts=[]
        for i in range(19 if e1 else 1):
            q=qa_fixture(ei*19+i);q["source_index"]=i
            text="original-context-"+str(ei*19+i)
            contexts.append(dict(context=text,context_sha256=digest(text),article="article"+str(ei*19+i),
                                 article_sha256=digest(["article",ei,i]),qas=[q]))
        records=[dict(write_record(i),truncated_tokens=0) for i in range(19 if e1 else writes)]
        eps.append(dict(id="episode"+str(ei),contexts=contexts,writes=records))
    return seal(dict(schema="natural_merge_plan/1",protocol=PROTOCOL,partition="development" if e1 else "formal",
                     experiment="E1" if e1 else "E2",episodes=eps))


class MergeCausality(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.old=CarrierResidualCompactor(2,8,block=2,rank=3,gamma_max=.05)
        with torch.no_grad():self.old.up.weight.normal_(std=.1)
        self.new=SharedMemoryMerge(2,8,block=2,rank=3,gamma_max=.05)
        self.new.load_state_dict(self.old.state_dict(),strict=True)
        self.source=torch.randn(2,38,8)

    def test_old_checkpoint_full_fusion_exact_parity(self):
        with torch.no_grad():
            expected=self.old(self.source,target=18)
            actual=self.new(self.source,target=18)
        self.assertTrue(torch.equal(expected,actual))
        self.assertEqual(self.old.state_dict().keys(),self.new.state_dict().keys())

    def test_mask_eliminates_unselected_values_keys_and_selection_dependence(self):
        indices=list(range(10,19));mask=torch.zeros(19,dtype=torch.bool);mask[indices]=True
        changed=self.source.clone();changed[:,:20]=torch.randn_like(changed[:,:20])*100
        with torch.no_grad():
            a=self.new(self.source,target=18,carrier_indices=indices,source_allow_mask=mask)
            b=self.new(changed,target=18,carrier_indices=indices,source_allow_mask=mask)
            full_a=self.new(self.source,target=18,carrier_indices=indices)
            full_b=self.new(changed,target=18,carrier_indices=indices)
        self.assertTrue(torch.equal(a,b))
        self.assertFalse(torch.equal(full_a,full_b))
        self.assertEqual(self.new.last_fusion["carrier_indices"],indices)

    def test_masked_sources_receive_zero_value_gradient(self):
        src=self.source.clone().requires_grad_(True)
        indices=list(range(10,19));mask=torch.zeros(19,dtype=torch.bool);mask[indices]=True
        self.new(src,target=18,carrier_indices=indices,source_allow_mask=mask).sum().backward()
        self.assertEqual(float(src.grad[:,:20].abs().sum()),0.)
        self.assertGreater(float(src.grad[:,20:].abs().sum()),0.)

    def test_rejects_incoherent_source_mask_or_carriers(self):
        with self.assertRaises(ValueError):self.new(self.source,target=18,carrier_indices=[0]*9)
        with self.assertRaises(ValueError):self.new(self.source,target=18,carrier_indices=list(range(9)),source_allow_mask=torch.zeros(19,dtype=torch.bool))
        with self.assertRaises(ValueError):self.new(self.source,target=18,carrier_indices=list(range(8,-1,-1)))


class NaturalDataIsolation(unittest.TestCase):
    def test_complete_prepare_seals_raw_train_and_formal_suite_without_synthetic_fallback(self):
        import prepare_natural_merge as preparation
        from public_retention_eval import ARMS
        from prepare_public_retention import DATA_REPO,REVISIONS,sha
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);(root/"raw/squad").mkdir(parents=True)
            def make_article(title,index):
                text="natural original paragraph "+str(index)+" A"
                return dict(title=title,paragraphs=[dict(context=text,qas=[dict(id="original-"+str(index),
                    question="What is the original answer?",is_impossible=False,
                    answers=[dict(text="A",answer_start=len(text)-1)])])])
            train=[make_article("train-"+str(i),i) for i in range(380)]
            observed_article=dict(title="observed-article",paragraphs=[])
            for i in range(100):
                observed_article["paragraphs"].extend(make_article("unused",10000+i)["paragraphs"])
            dev=[observed_article]+[make_article("dev-"+str(i),20000+i) for i in range(100)]
            train_path=root/"raw/squad/train-v2.0.json";dev_path=root/"raw/squad/dev-v2.0.json"
            train_path.write_text(json.dumps(dict(data=train)));dev_path.write_text(json.dumps(dict(data=dev)))
            provenance=dict(repo=DATA_REPO,revision=REVISIONS[DATA_REPO],files={
                "squad/train-v2.0.json":sha(train_path),"squad/dev-v2.0.json":sha(dev_path)})
            (root/"raw/public_dataset_manifest.json").write_text(json.dumps(provenance))
            old=root/"old-distractors.json";old.write_text(json.dumps([train[0]["paragraphs"][0]["context"]]))
            public_protocol=dict(budget_blocks=20,window_blocks=19,target_blocks=9,anchor_blocks=0,
                max_new_tokens=10,rank=32,gamma_max=.05,training=False,arms=list(ARMS),do_sample=False,stop_strings=[])
            samples=[]
            for i,paragraph in enumerate(observed_article["paragraphs"]):
                samples.append(dict(sample_index=i,filtered_index=i,context=paragraph["context"],
                    context_sha256=digest(paragraph["context"]),answer="A",related_writes=[write_record(i)]))
            observed=dict(schema="public_retention/1",status="prepared",protocol=public_protocol,task="squad",mode="author512",
                official_indices=list(range(100)),samples=samples,unrelated_writes=[],
                dataset=dict(files={"squad/unrelated_contexts_from_train.json":sha(old)}))
            observed["content_sha256"]=digest(observed)
            observed_path=root/"observed.json";observed_path.write_text(json.dumps(observed))
            nq_sample=dict(sample_index=0,filtered_index=0,id="nq-original",context="original NQ natural paragraph",
                context_sha256=digest("original NQ natural paragraph"),question="NQ?",answer="A",prompt="NQ? Answer:",
                prompt_ids=[78,81],related_writes=[write_record(0)])
            nq_writes=[dict(write_record(i),distractor_group=i) for i in range(30)]
            nq=dict(schema="public_retention/1",status="prepared",task="naturalqa",mode="author512",tokenizer={},
                protocol=public_protocol,official_indices=[0],samples=[nq_sample],unrelated_writes=nq_writes,
                distractor_groups=[dict(text="unchanged public distractor "+str(i)) for i in range(30)],dataset={})
            nq["content_sha256"]=digest(nq);nq_path=root/"nq.json";nq_path.write_text(json.dumps(nq))
            module=types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer()))
            args=types.SimpleNamespace(dataset_dir=str(root/"raw"),observed_squad_plan=str(observed_path),
                published_distractors=str(old),naturalqa_plan=str(nq_path),model_path="fake-tokenizer-only",
                formal_contexts=2,out_dir=str(root/"suite"))
            import io
            with patch.dict(sys.modules,{"transformers":module}),patch.object(preparation,"tokenizer_manifest",return_value={}),patch("sys.stdout",new=io.StringIO()):
                preparation.prepare(args)
            index=json.loads((root/"suite/suite.json").read_text())
            self.assertEqual(len(index["plans"]),6)
            plans={name:json.loads((root/"suite"/entry["path"]).read_text()) for name,entry in index["plans"].items()}
            for plan in plans.values():validate_plan(plan)
            self.assertEqual(sum(len(e["contexts"]) for e in plans["train"]["episodes"]),312)
            train_articles={c["article"] for e in plans["train"]["episodes"] for c in e["contexts"]}
            for name in ("E1-development","E1-formal","E2-squad30","E3-squad60"):
                evaluation_articles={c["article"] for e in plans[name]["episodes"] for c in e["contexts"]}
                self.assertFalse(train_articles&evaluation_articles)
                self.assertNotIn("observed-article",evaluation_articles)
            for ep in plans["E2-squad30"]["episodes"]:
                self.assertFalse(train_articles&{w.get("article") for w in ep["writes"]})
            self.assertEqual(plans["E3-squad60"]["episodes"][0]["writes"][:31],plans["E2-squad30"]["episodes"][0]["writes"])

    def test_original_spans_and_article_identity(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/"raw.json"
            row=dict(data=[dict(title="Original article",paragraphs=[dict(context="natural text A",qas=[
                dict(id="id",question="Original?",is_impossible=False,answers=[dict(text="A",answer_start=13)])])])])
            p.write_text(json.dumps(row));got=raw_squad(p)[0]
            self.assertEqual(got["article"],"Original article")
            self.assertEqual(got["qas"][0]["answers"],["A"])
            row["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["answer_start"]=12
            p.write_text(json.dumps(row))
            with self.assertRaises(ValueError):raw_squad(p)

    def test_article_disjoint_and_previously_observed_context_exclusion(self):
        def row(article,text):return dict(article=article,context=text,context_sha256=digest(text),qas=[])
        train=[row("T0","prior distractor"),row("T1","unused training")]
        dev=[row("D0","seen"),row("D0","same article other paragraph")]+[row("D"+str(i),"dev"+str(i)) for i in range(1,9)]
        splits,audit=split_rows(train,dev,dict(samples=[dict(context_sha256=digest("seen"))]),["prefix prior distractor suffix"])
        self.assertEqual([r["article"] for r in splits["train"]],["T1"])
        articles=[{r["article"] for r in splits[k]} for k in ("train","development","formal")]
        self.assertFalse(articles[0]&articles[1] or articles[0]&articles[2] or articles[1]&articles[2])
        self.assertNotIn("D0",articles[1]|articles[2])

    def test_context_changes_or_QA_omission_cannot_change_sealed_plan(self):
        plan=fixture_plan();validate_plan(plan)
        plan["episodes"][0]["contexts"][0]["context"]+=" edited"
        with self.assertRaises(ValueError):validate_plan(plan)

    def test_no_train_articles_or_contexts_including_distractors_at_test(self):
        plan=fixture_plan(e1=False)
        saved=dict(schema="natural_merge_checkpoint/1",evaluation_eligible=True,protocol=PROTOCOL,
            tokenizer={},train_context_hashes=[plan["episodes"][0]["contexts"][0]["context_sha256"]],train_articles=[])
        plan["tokenizer"]={}
        with self.assertRaises(ValueError):validate_checkpoint_scope(plan,saved)
        saved["train_context_hashes"]=[];saved["train_articles"]=["training-article"]
        plan["episodes"][0]["writes"][1]["article"]="training-article"
        with self.assertRaises(ValueError):validate_checkpoint_scope(plan,saved)


class RuntimeContracts(unittest.TestCase):
    def test_isolated_hooks_precede_next_injection_and_recurrent_count(self):
        m,_=manager();merge=SharedMemoryMerge(1,4,block=2,rank=2,gamma_max=.05).eval()
        hooks=[]
        def hook(phase,index,selected):hooks.append((phase,index,len(m.model.inputs),m.used,selected.tolist()))
        stream=MergeStream(m,merge,"SharedMerge",hook)
        with torch.no_grad():
            for i in range(61):stream.write(write_record(i),i)
        self.assertEqual([e["before_write"] for e in stream.events],[20,30,40,50,60])
        for pre,post in zip(hooks[::2],hooks[1::2]):
            self.assertEqual(pre[1:3],post[1:3])
            self.assertEqual(pre[3]-post[3],20)
        self.assertLessEqual(stream.peak,40)
        self.assertEqual(len(m.model.inputs),61)

    def test_selection_replays_same_carriers_even_when_own_memory_ranks_differ(self):
        m,_=manager();merge=SharedMemoryMerge(1,4,block=2,rank=2,gamma_max=.05)
        schedule=[list(range(9)),list(range(10,19))]
        stream=MergeStream(m,merge,"Selection",carrier_schedule=schedule)
        with torch.no_grad():
            for i in range(31):stream.write(write_record(i),i)
        self.assertEqual([e["selected_blocks"] for e in stream.events],schedule)
        self.assertTrue(torch.equal(m._memory[:,-2:],torch.full((1,2,4),31.)))

    def test_native_capacity_checkpoint_initialization_and_exception_restore(self):
        m,_=manager();model=m.model
        model.config=types.SimpleNamespace(num_blocks=30,num_memory_tokens=60)
        original=model.memory;mask=model.memory_active_mask;before=original.clone()
        with self.assertRaisesRegex(RuntimeError,"test failure"):
            with native_capacity(model,20):
                self.assertEqual(tuple(model.memory.shape),(1,40,4))
                self.assertTrue(torch.equal(model.memory,before[:,-40:]))
                self.assertEqual(int(model.initialized),1)
                model.memory.data.fill_(9);raise RuntimeError("test failure")
        self.assertIs(model.memory,original);self.assertIs(model.memory_active_mask,mask)
        self.assertTrue(torch.equal(model.memory,before));self.assertEqual(model.num_blocks,30)
        self.assertEqual(model.config.num_memory_tokens,60)

    def test_gradient_bridge_updates_shared_weights_and_restores_reader_source(self):
        m,_=manager();merge=SharedMemoryMerge(1,4,block=2,rank=2,gamma_max=.05)
        class DifferentiableReader(Reader):
            def forward(self,input_ids,**kwargs):
                # A minimal differentiable frozen reader; uses active suffix.
                mask=self.memory_active_mask
                feature=self.memory[:,mask].mean((0,1))
                logits=torch.stack((feature[0]-feature[1],feature[2]+feature[3]))
                return types.SimpleNamespace(logits=logits[None,None].expand(1,input_ids.shape[1],2))
        new=DifferentiableReader();m.model=new
        stream=MergeStream(m,merge,"Full")
        with torch.no_grad():
            for i in range(20):stream.write(write_record(i),i)
            new.memory[:,-40:]+=torch.randn_like(new.memory[:,-40:])
        reader_before={k:v.clone() for k,v in new.state_dict().items() if k!="memory" and k!="memory_active_mask"}
        initial_hash=state_digest(merge)
        optim=torch.optim.AdamW(merge.parameters(),lr=1e-4)
        qa=dict(prompt_ids=[0],answer_ids=[1],context_sha256="ctx")
        event=train_event(m,merge,optim,[qa])
        self.assertTrue(torch.isfinite(torch.tensor(event["gold_ce"])))
        self.assertNotEqual(state_digest(merge),initial_hash)
        self.assertEqual(m.used,20)  # 20 native slots after 19->9 plus newest block.
        self.assertFalse(any(p.requires_grad or p.grad is not None for p in new.parameters()))
        for k,v in reader_before.items():self.assertTrue(torch.equal(new.state_dict()[k],v))


class EvidenceContracts(unittest.TestCase):
    def test_author_and_auxiliary_metrics_separate(self):
        q=dict(scoring_target="The Eiffel Tower",answers=["The Eiffel Tower","Eiffel Tower"])
        score=scores("eiffel tower",q)
        self.assertFalse(score["author_containment"])
        self.assertTrue(score["normalized_em"])
        self.assertEqual(score["normalized_f1"],1.)

    def _run_fixture(self,e1=True,episodes=1):
        plan=fixture_plan(e1,episodes,writes=31)
        m,_=manager();m.model.config=types.SimpleNamespace(num_blocks=30,num_memory_tokens=60)
        merge=SharedMemoryMerge(1,4,block=2,rank=2,gamma_max=.05).eval().requires_grad_(False)
        initial=m._memory.clone();entries=[]
        with torch.no_grad():
            for i,ep in enumerate(plan["episodes"]):
                reset(m,initial)
                entry=e1_episode(m,merge,Tokenizer(),ep,lambda *a:None) if e1 else retention_episode(m,merge,Tokenizer(),initial,ep,lambda *a:None)
                entry["episode_index"]=i;entries.append(entry)
        frozen=state_digest(merge)
        r=dict(schema="natural_merge_results/1",status="complete",protocol=PROTOCOL,
            plan_content_sha256=plan["content_sha256"],experiment=plan["experiment"],partition=plan["partition"],
            plan_sha256="plan",checkpoint_sha256="ckpt",sources={},model={},suite_sha256=None,freeze_sha256=None,
            diagnostic_only=False,training_performed=False,optimizer_created=False,shared_state_sha256_before=frozen,
            shared_state_sha256_after=frozen,resources=dict(effective_budget_slots=40,physical_slots=60),
            sharding=dict(index=0,count=1,episode_indices=list(range(episodes))),episodes=entries,elapsed_seconds=1.)
        r["summary"]=summarize(r)
        return r,plan

    def test_e1_complete_original_QA_coverage_and_causal_certificate(self):
        result,plan=self._run_fixture()
        self.assertEqual(verify(result,plan)["original_qa_count"],19)
        bad=copy.deepcopy(result);bad["episodes"][0]["arms"]["CarrierOnlyFusion"]["carrier_indices"]=[0]*9
        with self.assertRaises(ValueError):verify(bad,plan)
        bad=copy.deepcopy(result);bad["episodes"][0]["arms"]["Full"]["predictions"].pop()
        with self.assertRaises(ValueError):verify(bad,plan)

    def test_continuous_retention_certificate_and_full_capacity_honesty(self):
        result,plan=self._run_fixture(False)
        self.assertEqual(verify(result,plan)["status"],"passed")
        self.assertEqual(result["episodes"][0]["arms"]["Full"]["status"],"capacity_exceeded")
        self.assertEqual(result["summary"]["arms"]["Full"]["n"],0)
        bad=copy.deepcopy(result);bad["episodes"][0]["arms"]["SharedMerge"]["compressions"][0]["phase"]="after"
        with self.assertRaises(ValueError):verify(bad,plan)

    def test_e3_requires_positive_formal_e2_same_checkpoint_and_complete_contexts(self):
        from prepare_public_retention import sha
        result,plan=self._run_fixture(False,episodes=2)
        with tempfile.TemporaryDirectory() as root:
            ckpt=Path(root)/"checkpoint.pt";ckpt.write_bytes(b"frozen natural checkpoint")
            result["checkpoint_sha256"]=sha(ckpt)
            freeze=dict(allowed_plan_content_sha256=[plan["content_sha256"]])
            with self.assertRaisesRegex(ValueError,"progression gate"):
                validate_e2_gate(result,plan,ckpt,freeze)
            # Construct a paired scoring fixture, not a claimed experimental result.
            for ep in result["episodes"]:
                row=ep["arms"]["Selection"]["trace"][-1]["predictions"][0]
                row["prediction"]="B";row["auxiliary_prediction"]="B";row["token_ids"]=[ord("B")]
                row["scores"]=scores("B",ep["arms"]["SharedMerge"]["trace"][-1]["predictions"][0] | {"answers":["A"]})
            result["summary"]=summarize(result)
            validate_e2_gate(result,plan,ckpt,freeze)
            ckpt.write_bytes(b"other checkpoint")
            with self.assertRaisesRegex(ValueError,"frozen formal E2"):
                validate_e2_gate(result,plan,ckpt,freeze)

    def test_token_prediction_and_claimed_score_are_independently_audited(self):
        result,plan=self._run_fixture()
        self.assertEqual(verify(result,plan,Tokenizer())["status"],"passed")
        row=result["episodes"][0]["arms"]["SharedMerge"]["predictions"][0]
        row["token_ids"]=[ord("B")]
        with self.assertRaisesRegex(ValueError,"generated tokens"):
            verify(result,plan,Tokenizer())

    def test_shard_merge_rejects_checkpoint_mixing_and_missing_context(self):
        r,plan=self._run_fixture(False,episodes=2)
        shards=[]
        for i in range(2):
            s=copy.deepcopy(r);s["episodes"]=[s["episodes"][i]]
            s["sharding"]=dict(index=i,count=2,episode_indices=[i]);s["summary"]=summarize(s);shards.append(s)
        combined=merge_reports(shards,plan)
        self.assertEqual(len(combined["episodes"]),2)
        shards[1]["checkpoint_sha256"]="other"
        with self.assertRaises(ValueError):merge_reports(shards,plan)

if __name__=="__main__":unittest.main()
