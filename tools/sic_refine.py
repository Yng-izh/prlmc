"""SIC-O2 development-only refinement of a completed frozen SIC-O1 report.

Run only after the O1 GPU job has exited. No confirmation scoring is exposed.
Actual compressed CE reranking uses discovery alone: it is an empirical training
criterion, not a guarantee of harmless behavior on future questions.
"""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import sic_optimize as O

SOURCE_ARMS = ("select", "raw", "normalized", "coverage", "taylor", "taylor_coverage")
ELIGIBLE_ARMS = ("eligible_normalized", "eligible_coverage", "eligible_taylor_coverage")
SELECTION_ALGO = "training-only eligibility and actual compressed CE rerank"


def finite_loss(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def checked_blocks(blocks):
    if (not isinstance(blocks, list) or len(blocks) != 9
            or any(type(b) is not int or not 0 <= b < 19 for b in blocks)
            or blocks != sorted(set(blocks))):
        raise ValueError("kept must contain nine unique temporal-order indices in [0,19)")
    return list(blocks)


def validate_inputs(plan_bytes, source_bytes):
    """Validate the complete development artifact before importing GPU libraries.

    The source plan hash binds matrix row length/order to frozen discovery.
    Confirmation remains unopened: only discovery/development fields are accessed.
    """
    plan, source = json.loads(plan_bytes), json.loads(source_bytes)
    digest = hashlib.sha256(plan_bytes).hexdigest()
    if source.get("plan_sha256") != digest:
        raise ValueError("source.plan_sha256 differs from frozen plan input sha256")
    if source.get("stage") != "develop" or source.get("selected") not in SOURCE_ARMS[1:]:
        raise ValueError("source must be a completed O1 development report")
    plans = plan["plans"]
    if len(plans) != 4 or len({p["row"] for p in plans}) != 4:
        raise ValueError("expected four unique frozen rows")
    if any(type(p["row"]) is not int or p["start"] != 30 for p in plans):
        raise ValueError("frozen rows must use start=30")
    if (sum(len(p["discovery"]) for p in plans) != 97
            or sum(len(p["development"]) for p in plans) != 200):
        raise ValueError("expected inherited 97 discovery and 200 development probes")
    if len({p["context_sha256"] for p in plans}) != 4:
        raise ValueError("expected four distinct frozen contexts")
    questions, keys = set(), set()
    for p in plans:
        if not p["discovery"] or not p["development"]:
            raise ValueError("each frozen row requires discovery and development")
        for split in ("discovery", "development"):
            for x in p[split]:
                if not x["question"] or not x["answer"]:
                    raise ValueError("empty frozen probe")
                if x["question"] in questions or x["key"] in keys:
                    raise ValueError("duplicate discovery/development question or key")
                questions.add(x["question"])
                keys.add(x["key"])
    windows, records = source.get("windows", []), source.get("records", [])
    rows = {p["row"] for p in plans}
    if (len(windows) != 4 or {w["row"] for w in windows} != rows
            or len(records) != 200):
        raise ValueError("source requires four complete windows and 200 records")
    by_window = {w["row"]: w for w in windows}
    by_record = {}
    for r in records:
        key = (r["row"], r["question"])
        if key in by_record:
            raise ValueError("duplicate source record")
        by_record[key] = r
    expected_records = {(p["row"], x["question"])
                        for p in plans for x in p["development"]}
    if set(by_record) != expected_records:
        raise ValueError("source development identities differ from frozen plan")
    for p in plans:
        w = by_window[p["row"]]
        if set(w["kept"]) != set(SOURCE_ARMS) or w.get("repeat_equal") is not True:
            raise ValueError("source window requires all six baseline arms and repeated scoring")
        for matrix in ("energy", "taylor"):
            values = w[matrix]
            if (len(values) != len(p["discovery"])
                    or any(not isinstance(v, list) or len(v) != 19
                           or not all(finite_loss(s) for s in v) for v in values)):
                raise ValueError("source %s shape/value differs from frozen discovery" % matrix)
        if (len(w["discovery_losses"]) != len(p["discovery"])
                or not all(finite_loss(v) for v in w["discovery_losses"])):
            raise ValueError("source discovery losses incomplete")
        signatures = {}
        for name in SOURCE_ARMS:
            blocks = checked_blocks(w["kept"][name])
            if not finite_loss(w["gold_loss"][name]):
                raise ValueError("source development CE incomplete/nonfinite")
            outcomes = []
            for x in p["development"]:
                r = by_record[(p["row"], x["question"])]
                if any(r.get(k) != v for k, v in x.items()):
                    raise ValueError("source probe metadata differs from frozen development")
                if (any(type(r.get(k)) is not bool for k in ("full", "no_memory", "eligible"))
                        or r["eligible"] != (r["full"] and not r["no_memory"])
                        or set(r["arms"]) != set(SOURCE_ARMS)
                        or any(type(v) is not bool for v in r["arms"].values())):
                    raise ValueError("source baseline/reference outcomes incomplete")
                outcomes.append(r["arms"][name])
            signature = (outcomes, w["gold_loss"][name])
            key = tuple(blocks)
            if key in signatures and signatures[key] != signature:
                raise ValueError("source identical kept sets have inconsistent outcomes/CE")
            signatures[key] = signature
    for category, eligible in (("eligible", True), ("all", False)):
        for name in SOURCE_ARMS:
            if source.get(category, {}).get(name) != O.stats(records, name, eligible):
                raise ValueError("source final statistics missing or inconsistent")
    return plan, source, digest, hashlib.sha256(source_bytes).hexdigest(), by_window, by_record


def eligible_candidates(energy, taylor, indices):
    from methods.prlmc.sic import normalize_probe_scores, coverage_select, salience_select
    if len(indices) < 4:
        return {}
    e, t = energy[indices], taylor[indices]
    return dict(eligible_normalized=salience_select(normalize_probe_scores(e).sum(0), 9),
                eligible_coverage=coverage_select(e, 9),
                eligible_taylor_coverage=coverage_select(t, 9))


def validate_score(result, probes, allow_empty=False):
    expected = {p.question for p in probes}
    if (set(result["correct"]) != expected or result["n"] != len(probes)
            or any(type(v) is not bool for v in result["correct"].values())):
        raise ValueError("scoring returned incomplete probe outcomes")
    if expected and not finite_loss(result["gold_loss"]):
        raise ValueError("scoring returned nonfinite CE")
    if not expected and not allow_empty:
        raise ValueError("empty scoring population")


def score_blocks(manager, blocks, probes, E, S, torch, repeat=True):
    with torch.no_grad():
        pool, mask = S.pool_from_blocks(manager, blocks)
        try:
            first = E.score_probes(manager, probes, pool, 256, mask)
            validate_score(first, probes)
            if repeat:
                second = E.score_probes(manager, probes, pool, 256, mask)
                validate_score(second, probes)
                if first["correct"] != second["correct"]:
                    raise RuntimeError("scoring changed on the same compressed pool")
        finally:
            del pool, mask
            torch.cuda.empty_cache()
    return dict(first, repeat_equal=repeat)


def verify_reference(live, expected, label):
    if live != expected:
        raise RuntimeError("development reference differs from source: " + label)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "source", "out", "choice"):
        ap.add_argument("--" + name, required=True)
    args = ap.parse_args()
    paths = [Path(getattr(args, name)).resolve() for name in ("plan", "source", "out", "choice")]
    if len(set(paths)) != 4:
        raise ValueError("plan, source, out and choice must use distinct paths")
    plan, source, plan_hash, source_hash, source_windows, source_records = validate_inputs(
        paths[0].read_bytes(), paths[1].read_bytes())
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "9")
    import torch
    import e6a_overfit as E
    import sic_probe as S
    import yaml
    from transformers import AutoTokenizer
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from methods.prlmc.probes import Probe
    from methods.prlmc.consolidate import _select_block_indices
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory

    random.seed(plan["seed"])
    torch.manual_seed(plan["seed"])
    torch.cuda.manual_seed_all(plan["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    cfg = PRLMCConfig.from_dict(dict(yaml.safe_load(Path(E.CONFIG).read_text())["prlmc"],
                                    device="cuda:0", active_memory_mask=True))
    model = MemoryLLM.from_pretrained(E.MODEL, torch_dtype=torch.float16,
                                     attn_implementation="sdpa").to("cuda:0")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
        if hasattr(module, "attention_dropout"):
            module.attention_dropout = 0.
    model.eval()
    manager = ProtectedResidualMemory(model, AutoTokenizer.from_pretrained(E.MODEL), cfg)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("reader parameter requires_grad must remain false")

    def probes(items):
        result = []
        for x in items:
            stem = E.query_stem(x["question"])
            tokens = manager.tokenizer(stem + " " + x["answer"], add_special_tokens=False).input_ids
            prefix = manager.tokenizer(stem, add_special_tokens=False).input_ids
            if len(tokens) > 256 or not 0 < len(prefix) < len(tokens):
                raise ValueError("invalid or truncated answer tail")
            result.append(Probe(question=x["question"], answers=[x["answer"]], prompt=stem))
        return result

    started = time.monotonic()
    report = dict(stage="develop", experiment="SIC-O2", plan_sha256=plan_hash,
                  source_sha256=source_hash, selected=None, records=[], windows=[],
                  protocol=plan["protocol"], selection_algo=SELECTION_ALGO,
                  runtime=dict(torch=torch.__version__, cuda=torch.version.cuda,
                               gpu=torch.cuda.get_device_name(0), seed=plan["seed"],
                               deterministic_warn_only=True, cudnn_benchmark=False,
                               dropout_zero=True, attention_dropout_zero=True,
                               model_eval=True, reader_parameters_require_grad=False,
                               gradient_checkpointing=True, use_cache=False,
                               cross_process_reproduction_tested=False,
                               cached_reference=True, base_pool_reproduction_tested=False),
                  cross_process_kept_equal=True, cached_reference=True)
    O.save(args.out, report)
    for p in plan["plans"]:
        row = p["row"]
        w = source_windows[row]
        row_started = time.monotonic()
        with torch.no_grad():
            lines = E.read_fact_lines(row)
            if hashlib.sha256("\n".join(lines).encode()).hexdigest() != p["context_sha256"]:
                raise ValueError("context changed after plan freeze")
            model.eval()
            covered = E.build_window(manager, lines, 19 * manager.block, 30)
            if covered != [(60 + 20*j, 80 + 20*j, 256*j, 256) for j in range(19)]:
                raise RuntimeError("unexpected written block layout")
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise RuntimeError("reader parameter unexpectedly requires grad")
            begin = manager.total - manager.used
            memory = manager._memory[:, begin:begin + 19 * manager.block]
            native = _select_block_indices(memory, 9 * manager.block, manager.block,
                                          manager._window_importance(begin, 19 * manager.block),
                                          manager.goal_direction, manager.select_weights).tolist()
            del memory
            if native != w["kept"]["select"]:
                raise RuntimeError("native kept differs from source for row %d" % row)
            energy = torch.tensor(w["energy"], dtype=torch.float32, device="cpu")
            taylor = torch.tensor(w["taylor"], dtype=torch.float32, device="cpu")
            if (not bool(torch.isfinite(energy).all()) or not bool(torch.isfinite(taylor).all())):
                raise ValueError("source matrices overflowed fp32")
            kept = {name: list(w["kept"][name]) for name in SOURCE_ARMS}
            discovery = probes(p["discovery"])
            full = E.score_uncompacted(manager, discovery, 256)
            no_mem = E.score_no_memory(manager, discovery, 256)
            validate_score(full, discovery)
            validate_score(no_mem, discovery)
            indices = [i for i, probe in enumerate(discovery)
                       if full["correct"][probe.question] and not no_mem["correct"][probe.question]]
            train_probes = [discovery[i] for i in indices]
            kept.update(eligible_candidates(energy, taylor, indices))
            # No development/confirmation inputs enter candidate construction or rerank.
            # Actual compressed CE is measured, not inferred from gradient proxies;
            # it gives no no-harm guarantee outside these discovery training probes.
            train_cache, candidate_scores = {}, {}
            for name, blocks in kept.items():
                key = tuple(checked_blocks(blocks))
                if key not in train_cache:
                    if train_probes:
                        train_cache[key] = score_blocks(manager, blocks, train_probes, E, S, torch, repeat=False)
                    else:
                        train_cache[key] = dict(n=0, gold_loss=None, correct={},
                                                status="no_eligible_training_questions")
                candidate_scores[name] = dict(train_cache[key])
            if len(indices) >= 4:
                joint_source = min(kept, key=lambda name: (candidate_scores[name]["gold_loss"],
                                                          name != "select", name))
                joint_status = "actual_compressed_discovery_ce_rerank"
            else:
                joint_source = "select"
                joint_status = "insufficient_eligible_training_questions_joint_select"
            kept["joint"] = list(kept[joint_source])
            candidate_scores["joint"] = dict(candidate_scores[joint_source])
            training = dict(full=full, no_memory=no_mem,
                            eligible={probe.question: i in indices for i, probe in enumerate(discovery)},
                            eligible_indices=indices, n_eligible=len(indices), n_discovery=len(discovery),
                            eligible_status="available" if len(indices) >= 4 else "insufficient_minimum_4",
                            actual_candidate_ce=candidate_scores, joint_source=joint_source,
                            joint_status=joint_status, distinct_kept_scored=len(train_cache) if train_probes else 0,
                            cached_reference=False)
            print("row", row, "training eligible", len(indices), "/", len(discovery),
                  "joint", joint_source, joint_status, flush=True)

            # Open development only after every training-only kept set is frozen.
            dev_probes = probes(p["development"])
            live_full = E.score_uncompacted(manager, dev_probes, 256)
            live_no_mem = E.score_no_memory(manager, dev_probes, 256)
            live_native = score_blocks(manager, native, dev_probes, E, S, torch, repeat=False)
            for result, field, label in ((live_full,"full","full"),(live_no_mem,"no_memory","no_memory")):
                validate_score(result,dev_probes)
                verify_reference(result["correct"],{x["question"]:source_records[(row,x["question"])][field]
                                                     for x in p["development"]},label)
            verify_reference(live_native["correct"],{x["question"]:source_records[(row,x["question"])]["arms"]["select"]
                                                     for x in p["development"]},"select")
            dev_cache, outcome_sources = {}, {}
            for name in SOURCE_ARMS:
                key = tuple(kept[name])
                dev_cache[key] = dict(correct={x["question"]: source_records[(row, x["question"])]["arms"][name]
                                               for x in p["development"]},
                                      gold_loss=w["gold_loss"][name], n=len(dev_probes),
                                      cached_reference=True, cached_source_arm=name)
            outcomes, ce = {}, {}
            for name, blocks in kept.items():
                key = tuple(blocks)
                if name in SOURCE_ARMS:
                    scored = dict(correct={x["question"]: source_records[(row, x["question"])]["arms"][name]
                                           for x in p["development"]}, gold_loss=w["gold_loss"][name],
                                  cached_reference=True, cached_source_arm=name)
                else:
                    if key not in dev_cache:
                        dev_cache[key] = dict(score_blocks(manager, blocks, dev_probes, E, S, torch),
                                              cached_reference=False)
                    scored = dev_cache[key]
                outcomes[name], ce[name] = scored["correct"], scored["gold_loss"]
                outcome_sources[name] = dict(cached_reference=scored["cached_reference"],
                                             cached_source_arm=scored.get("cached_source_arm"),
                                             repeat_equal=scored.get("repeat_equal", w["repeat_equal"]))
                print("row", row, "development", name, sum(outcomes[name].values()), "/", len(dev_probes),
                      "CE", ce[name], "cached", scored["cached_reference"], flush=True)
            for x in p["development"]:
                r = source_records[(row, x["question"])]
                report["records"].append(dict(x, row=row, full=r["full"], no_memory=r["no_memory"],
                                              eligible=r["eligible"], cached_reference=True,
                                              arms={name: answers[x["question"]] for name, answers in outcomes.items()},
                                              arm_cached_reference={name: info["cached_reference"]
                                                                    for name, info in outcome_sources.items()}))
            report["windows"].append(dict(row=row, kept=kept, energy=w["energy"], taylor=w["taylor"],
                                           discovery_losses=w["discovery_losses"], gold_loss=ce,
                                           training=training, repeat_equal=True,
                                           development_outcome_sources=outcome_sources,
                                           native_kept_equal=True, development_reference_equal=True, cached_reference=True,
                                           elapsed_seconds=time.monotonic() - row_started))
            del energy, taylor
            torch.cuda.empty_cache()
        O.save(args.out, report)

    common = set.intersection(*(set(w["kept"]) for w in report["windows"]))
    names = [name for name in SOURCE_ARMS + ELIGIBLE_ARMS + ("joint",) if name in common]
    report["names"] = names
    report["excluded_pooled_arms"] = {name: [w["row"] for w in report["windows"] if name not in w["kept"]]
                                     for name in ELIGIBLE_ARMS if name not in common}
    report["eligible"] = {name: O.stats(report["records"], name) for name in names}
    report["all"] = {name: O.stats(report["records"], name, False) for name in names}
    report["per_context"] = {str(w["row"]): {name: O.stats([r for r in report["records"] if r["row"] == w["row"]], name)
                                             for name in w["kept"]} for w in report["windows"]}
    mean_ce = {name: sum(w["gold_loss"][name] for w in report["windows"]) / 4 for name in names}
    choices = [name for name in names if name != "select"]
    selected = min(choices, key=lambda name: (report["eligible"][name]["wrong"],
                                            report["all"][name]["wrong"], mean_ce[name], name))
    report["selected"] = selected
    report["development_improved"] = report["eligible"][selected]["net"] > 0
    report["mean_ce"] = mean_ce
    report["runtime"]["elapsed_seconds"] = time.monotonic() - started
    statistics = dict(eligible=report["eligible"], all=report["all"], mean_ce=mean_ce,
                      per_context=report["per_context"], excluded_pooled_arms=report["excluded_pooled_arms"])
    O.save(args.choice, dict(source_sha256=source_hash, plan_sha256=plan_hash, selected=selected,
                             perrow_kept={str(w["row"]): w["kept"][selected] for w in report["windows"]},
                             baseline_perrow_kept={str(w["row"]): {name: w["kept"][name] for name in ("select", "raw")}
                                                  for w in report["windows"]},
                             statistics=statistics, eligible=report["eligible"], all=report["all"],
                             development_improved=report["development_improved"], selection_algo=SELECTION_ALGO))
    O.save(args.out, report)
    print("FINAL", json.dumps(dict(selected=selected, eligible=report["eligible"], all=report["all"])), flush=True)


if __name__ == "__main__":
    main()
