"""Pure CPU tests: no pandas, torch, model imports, or dataset downloads."""
import copy
import random
import tempfile
import unittest
from pathlib import Path

import sic_paper_data as D

SPECS = [(r"^(.+) likes (.+)$", "What does %s like?", "1"),
         (r"^The author of (.+) is (.+)$", "Who wrote %s?", "1")]


def fixture(count=4000):
    old = ["%d. Old%d likes A." % (i+1,i) for i in range(440)]
    lines = list(old)
    for i in range(count):
        relation = i%2
        text = "Entity%d likes %s." % (i,("A","B","C")[i%3]) if not relation else "The author of Entity%d is %s." % (i,("A","B","C")[i%3])
        lines.append("%d. %s" % (len(lines)+1,text))
    lines.append("%d. The author of Old0 is B." % (len(lines)+1))
    return ["\n".join(old)]*3+["\n".join(lines)]


def reseal(plan):
    for episode in plan["episodes"]:
        episode["context_sha256"] = D.hashlib.sha256("\n".join(episode["lines"]).encode()).hexdigest()
        episode.pop("episode_sha256",None)
        episode["episode_sha256"] = D.digest(episode)
    plan.pop("plan_sha256",None)
    plan["plan_sha256"] = D.digest(plan)


class ControlledDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contexts = fixture()
        cls.plan = D.generate(cls.contexts,SPECS)

    def test_full_fixed_shape(self):
        D.validate(self.plan)
        self.assertEqual(self.plan["actual_counts"],dict(episodes=24,development=8,test=16,
            probe_keys=3072,anchor_entities=720,entity_count=3792,lines=9840))
        self.assertFalse(self.plan["protocol"]["test_scored"])
        self.assertIn("not_natural_contexts",self.plan["context_type"])

    def test_deterministic_generation(self):
        self.assertEqual(self.plan,D.generate(self.contexts,SPECS))
        self.assertNotEqual(self.plan["plan_sha256"],D.generate(self.contexts,SPECS,seed=1)["plan_sha256"])

    def test_global_entity_and_key_isolation(self):
        identities = []
        for episode in self.plan["episodes"]:
            subjects = {f["subject"] for f in episode["facts"]}
            self.assertEqual(len(subjects),158)
            identities.extend(subjects)
        self.assertEqual(len(identities),len(set(identities)))
        self.assertFalse(any(s.startswith("Old") for s in identities))

    def test_latest_gold_and_distinct_versions(self):
        for episode in self.plan["episodes"]:
            for split in episode["splits"].values():
                for probe in split:
                    facts = [f for f in episode["facts"] if f["subject"]==probe["subject"]]
                    self.assertEqual(probe["answer"],facts[-1]["answer"])
                    self.assertEqual(probe["serial"],max(f["serial"] for f in facts))
                    self.assertEqual(len(facts),len({f["answer"] for f in facts}))
                    self.assertEqual(probe["block"],(probe["line"]-30)//20)

    def test_source_serial_preserved(self):
        for fact in self.plan["episodes"][0]["facts"]:
            original = next(f for f in D.extract(self.contexts[3],SPECS)
                            if f["source_serial"]==fact["source_serial"])
            self.assertEqual(fact["source_text"],original["source_text"])
            self.assertEqual(fact["subject"],original["subject"])

    def test_shortage_never_reduces_request(self):
        plan = D.generate(fixture(158),SPECS)
        self.assertEqual(plan["status"],"insufficient_data")
        self.assertEqual(plan["requested_episodes"],24)
        self.assertEqual(plan["capacity"]["maximum_episodes"],1)
        self.assertEqual(plan["episodes"],[])
        self.assertEqual(plan["feasible_adjustments"]["additional_unique_subjects_required"],3634)

    def test_long_answers_not_in_bank(self):
        contexts = fixture(158)
        contexts[3] += "\n99999. Other likes one two three four five six."
        plan = D.generate(contexts,SPECS,episodes=1,dev=1)
        self.assertEqual(plan["capacity"]["available_subjects"],158)

    def test_insufficient_distinct_values(self):
        contexts = [c.replace(" likes B."," likes A.").replace(" likes C."," likes A.")
                    for c in fixture(158)]
        plan = D.generate(contexts,SPECS,episodes=1,dev=1)
        self.assertEqual(plan["status"],"insufficient_data")
        self.assertEqual(plan["capacity"]["available_subjects"],79)

    def test_balanced_relation_sampling(self):
        items = [dict(relation=r,id=i) for r in range(4) for i in range(20)]
        chosen = D.balanced(items,32,random.Random(3))
        self.assertEqual(D.Counter(x["relation"] for x in chosen),{r:8 for r in range(4)})

    def test_template_ast_does_not_execute_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"templates.py"
            path.write_text("raise RuntimeError('must not execute')\nTEMPLATES = "+repr(SPECS))
            self.assertEqual(D.templates(path),SPECS)

    def test_hash_detects_corruption(self):
        plan = copy.deepcopy(self.plan)
        plan["episodes"][0]["lines"][0] += "x"
        with self.assertRaisesRegex(ValueError,"hash"):
            D.validate(plan)

    def test_semantic_latest_guard(self):
        plan = copy.deepcopy(self.plan)
        plan["episodes"][0]["splits"]["evaluation"][0]["answer"] = "Wrong"
        reseal(plan)
        with self.assertRaisesRegex(ValueError,"gold"):
            D.validate(plan)

    def test_semantic_split_overlap_guard(self):
        plan = copy.deepcopy(self.plan)
        ep = plan["episodes"][0]
        ep["splits"]["evaluation"][0] = copy.deepcopy(ep["splits"]["discovery"][0])
        reseal(plan)
        with self.assertRaisesRegex(ValueError,"overlap"):
            D.validate(plan)

    def test_token_budget_success_and_overflow(self):
        plan = D.generate(fixture(158),SPECS,episodes=1,dev=1)
        D.check_token_budget(plan,lambda text:list(range(100)))
        self.assertTrue(plan["token_budget_verified"])
        self.assertEqual(len(plan["episodes"][0]["token_lengths"]["blocks"]),19)
        D.validate(plan)
        D.check_token_budget(plan,lambda text:list(range(513)))
        self.assertEqual(plan["status"],"token_budget_exceeded")
        self.assertEqual(len(plan["overflows"]),20)
        self.assertEqual(len(plan["episodes"][0]["lines"]),410)

    def test_answer_bank_provenance_guard(self):
        plan = copy.deepcopy(self.plan)
        plan["answer_banks"]["0"] = ["X","Y","Z"]
        reseal(plan)
        with self.assertRaisesRegex(ValueError,"source relation bank"):
            D.validate(plan)

    def test_tokenizer_manifest_and_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,"vocabulary"):
                D.tokenizer_manifest(directory)
            (Path(directory)/"tokenizer.json").write_text("{}")
            manifest = D.tokenizer_manifest(directory)
            self.assertEqual(manifest["files"]["tokenizer.json"],D.file_hash(Path(directory)/"tokenizer.json"))
            plan = D.generate(fixture(158),SPECS,episodes=1,dev=1)
            D.check_token_budget(plan,lambda text:[0]*100)
            audit = D.token_audit(plan,"plan-file-hash",manifest)
            self.assertEqual(audit["plan_file_sha256"],"plan-file-hash")
            self.assertEqual(audit["tokenizer"],manifest)
            self.assertTrue(audit["verified"])
            self.assertFalse(audit["test_scored"])
            self.assertFalse(audit["model_loaded"])
            self.assertFalse(audit["truncation_performed"])
            self.assertEqual(len(audit["episodes"]),1)

    def test_top_level_hash_guard(self):
        plan = copy.deepcopy(self.plan)
        plan["seed"] += 1
        with self.assertRaisesRegex(ValueError,"plan hash"):
            D.validate(plan)

    def test_bad_arguments(self):
        for episodes,dev in ((0,0),(1,2),(1,-1)):
            with self.assertRaises(ValueError):
                D.generate(self.contexts,SPECS,episodes=episodes,dev=dev)
        with self.assertRaises(ValueError):
            D.generate(self.contexts[:3],SPECS)


if __name__=="__main__":
    unittest.main()
