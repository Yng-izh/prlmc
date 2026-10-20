"""Item 1: verify data/natural_merge_v7/train.json (plan_sha256 + split audit)."""
import hashlib, json, os, sys
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from prepare_public_retention import digest, sha, tokenizer_manifest
from prepare_natural_merge_v4 import validate_v4_plan
from prepare_natural_merge_v7 import validate_v7_plan, TRAIN_V7, make_schedule_v7

PLAN = "data/natural_merge_v7/train.json"
V4 = "data/natural_merge_v4/train_v5.json"
plan = json.load(open(PLAN))
v4 = json.load(open(V4))
checks = {}
validate_v7_plan(plan); checks["validate_v7_plan"] = "ok"
checks["schema"] = plan["schema"]
checks["content_sha256"] = plan["content_sha256"]
checks["content_sha256_matches_digest"] = plan["content_sha256"] == digest(
    {k: v for k, v in plan.items() if k != "content_sha256"})
checks["original_v4_file_sha256"] = plan["original_v4_file_sha256"]
checks["original_v4_file_sha256_matches_file"] = plan["original_v4_file_sha256"] == sha(V4)
validate_v4_plan(plan["original_v4_plan"])
checks["embedded_v4_plan_valid"] = True
checks["embedded_v4_content_sha256"] = plan["original_v4_plan"]["content_sha256"]

audit = plan["base_plan"]["split_audit"]
train_ctx = {c["context_sha256"] for e in plan["episodes"] for c in e["contexts"]}
train_art = {c["article"] for e in plan["episodes"] for c in e["contexts"]}
checks["audit_identical_to_v4"] = audit == v4["base_plan"]["split_audit"]
checks["train_contexts"] = len(train_ctx)
checks["train_articles"] = len(train_art)
checks["train_contexts_disjoint_from_observed"] = not (train_ctx & set(audit["observed_context_hashes"]))
checks["train_articles_disjoint_from_observed"] = not (train_art & set(audit["observed_articles"]))
checks["train_articles_disjoint_from_development"] = not (train_art & set(audit["development_articles"]))
checks["train_articles_disjoint_from_formal"] = not (train_art & set(audit["formal_articles"]))
checks["train_articles_disjoint_from_excluded_distractors"] = not (
    train_art & set(audit["excluded_previous_distractor_articles"]))
checks["original_files_sha256_verified"] = {
    name: hashlib.sha256(open("data/public_retention_original_c609067/" + name, "rb").read()).hexdigest() == value
    for name, value in audit["original_files"].items()}
checks["distractors_sha256_verified"] = hashlib.sha256(
    open("data/public_retention_processed_03a8db0/squad/unrelated_contexts_from_train.json", "rb").read()
).hexdigest() == audit["published_distractors_sha256"]
checks["observed_plan_sha256"] = audit["observed_plan_sha256"]
checks["observed_plan_file_present"] = False  # external --observed-squad-plan input, not in repo
checks["tokenizer_matches_reader"] = plan["tokenizer"] == tokenizer_manifest("/dev/shm/s1d_reader_fp16")
checks["eos_token_id"] = plan["eos_token_id"]
checks["episodes"] = len(plan["episodes"])
checks["contexts_per_episode"] = sorted({len(e["contexts"]) for e in plan["episodes"]})
checks["qa_events"] = len(plan["qa_events"])
checks["qa_event_stages"] = {s: sum(1 for e in plan["qa_events"] if e["stage"] == s) for s in ("single_merge", "recurrent")}
checks["recurrent_positions"] = sorted({e["before_write"] for e in plan["qa_events"] if e["stage"] == "recurrent"})
checks["single_window_starts"] = sorted({e["window_start"] for e in plan["qa_events"] if e["stage"] == "single_merge"})
checks["qa_slots"] = sum(len(e["qa_ids"]) for e in plan["qa_events"])
checks["unique_qa"] = len({i for e in plan["qa_events"] for i in e["qa_ids"]})
checks["all_episode_qa"] = len({q["id"] for e in plan["episodes"] for c in e["contexts"] for q in c["qas"]})
checks["schedule_matches_constants"] = plan["qa_events"] == make_schedule_v7(plan["episodes"])
checks["geometry"] = TRAIN_V7["geometry"]
checks["variants"] = sorted(TRAIN_V7["variants"])
checks["recovery_gate"] = TRAIN_V7["recovery_gate"]
checks["teacher_max_gold_ce"] = TRAIN_V7["teacher_max_gold_ce"]
checks["planned_updates"] = TRAIN_V7["planned_updates"]
checks["protocol"] = plan["base_plan"]["protocol"]
checks["dataset_repo"] = plan["base_plan"]["dataset"].get("repo")
checks["dataset_revision"] = plan["base_plan"]["dataset"].get("revision")
checks["file_bytes"] = os.path.getsize(PLAN)
checks["train_plan_file_sha256"] = sha(PLAN)
checks["episode_ids"] = [e["id"] for e in plan["episodes"]]
print(json.dumps(checks, indent=1, default=str))
