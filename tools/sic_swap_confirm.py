"""SIC swap confirmation scorer; selection is frozen before confirmation."""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def save(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(p)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_blocks(value):
    return (isinstance(value, list) and len(value) == 9 and
            value == sorted(set(value)) and
            all(type(x) is int and 0 <= x < 19 for x in value))


def window_stats(records, losses):
    eligible = [r for r in records if r["eligible"]]
    select_wrong = sum(not r["arms"]["select"] for r in eligible)
    finite_wrong = sum(not r["arms"]["finite_ce_swap"] for r in eligible)
    rescued = sum((not r["arms"]["select"]) and r["arms"]["finite_ce_swap"] for r in eligible)
    harmed = sum(r["arms"]["select"] and not r["arms"]["finite_ce_swap"] for r in eligible)
    return {
        "n": len(eligible),
        "correct": {name: sum(r["arms"][name] for r in records)
                    for name in ("select", "finite_ce_swap")},
        "gold_loss": {name: losses[name] for name in ("select", "finite_ce_swap")},
        "eligible_lost": {"select": select_wrong, "finite_ce_swap": finite_wrong},
        "rescued": rescued,
        "harmed": harmed,
        "net": rescued - harmed,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--choice", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cuda-device", default="cuda:0")
    args = ap.parse_args()

    plan_path, choice_path = Path(args.plan), Path(args.choice)
    plan_bytes = plan_path.read_bytes()
    choice_bytes = choice_path.read_bytes()
    plan = json.loads(plan_bytes)
    choice = json.loads(choice_bytes)
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    if choice.get("plan_sha256") != plan_sha:
        raise ValueError("choice plan_sha256 differs from plan")
    if "discovery" not in str(choice.get("selection_rule", "")).lower():
        raise ValueError("choice selection_rule must contain discovery")
    plans = plan.get("plans", [])
    if len(plans) != 4:
        raise ValueError("expected four plan windows")
    frozen = choice.get("perrow_kept")
    if not isinstance(frozen, dict) or set(frozen) != {str(p["row"]) for p in plans}:
        raise ValueError("choice must contain perrow_kept for all four windows")
    for row in frozen:
        if not valid_blocks(frozen[row]):
            raise ValueError("invalid perrow_kept for row %s" % row)

    import torch
    import yaml
    import e6a_overfit as E
    import sic_probe as S
    from transformers import AutoTokenizer
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory
    from methods.prlmc.probes import Probe
    from methods.prlmc.consolidate import _select_block_indices

    device = args.cuda_device
    random.seed(plan.get("seed", 20261009))
    torch.manual_seed(plan.get("seed", 20261009))
    torch.cuda.manual_seed_all(plan.get("seed", 20261009))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    cfg = PRLMCConfig.from_dict(dict(yaml.safe_load(Path(E.CONFIG).read_text())["prlmc"],
                                     device=device, active_memory_mask=True))
    model = MemoryLLM.from_pretrained(E.MODEL, torch_dtype=torch.float16,
                                      attn_implementation="sdpa").to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.config.use_cache = False
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if hasattr(module, "attention_dropout"):
            module.attention_dropout = 0.0
    manager = ProtectedResidualMemory(model, AutoTokenizer.from_pretrained(E.MODEL), cfg)

    def probes(items):
        return [Probe(question=x["question"], answers=[x["answer"]],
                      prompt=E.query_stem(x["question"])) for x in items]

    report = {
        "stage": "confirm",
        "experiment": "SIC-swap-confirm",
        "plan_sha256": plan_sha,
        "choice_sha256": hashlib.sha256(choice_bytes).hexdigest(),
        "choice_source_sha256": choice.get("source_sha256"),
        "selected": choice.get("selected", "finite_ce_swap"),
        "selection_rule": choice["selection_rule"],
        "confirmation_scoring_only": True,
        "records": [], "windows": [],
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "cuda_device": device, "deterministic_warn_only": True,
                    "gpu": torch.cuda.get_device_name(0),
                    "warnings": ["confirmation questions were not used for selection"]},
        "source_hashes": {"plan": plan_sha, "choice": hashlib.sha256(choice_bytes).hexdigest(),
                          "script": digest(__file__)},
    }
    started = time.monotonic()
    for p in plans:
        row = p["row"]
        lines = E.read_fact_lines(row)
        context_sha = hashlib.sha256("\n".join(lines).encode()).hexdigest()
        if context_sha != p["context_sha256"]:
            raise ValueError("context changed for row %d" % row)
        covered = E.build_window(manager, lines, 19 * manager.block, 30)
        expected = [(60 + 20 * j, 80 + 20 * j, 256 * j, 256) for j in range(19)]
        if covered != expected:
            raise RuntimeError("unexpected window layout for row %d" % row)
        begin = manager.total - manager.used
        source = manager._memory[:, begin:begin + 19 * manager.block]
        native = _select_block_indices(source, 9 * manager.block, manager.block,
                                       manager._window_importance(begin, 19 * manager.block),
                                       manager.goal_direction, manager.select_weights).tolist()
        del source
        if "baseline_perrow_kept" in choice:
            expected_native = choice["baseline_perrow_kept"].get(str(row), {}).get("select")
            if expected_native is not None and native != expected_native:
                raise RuntimeError("native mismatch for row %d" % row)
        frozen_blocks = frozen[str(row)]
        qs = probes(p["confirmation"])
        full = E.score_uncompacted(manager, qs, 256)["correct"]
        no_mem = E.score_no_memory(manager, qs, 256)["correct"]
        eligible = {x["question"]: bool(full[x["question"]] and not no_mem[x["question"]])
                    for x in p["confirmation"]}
        outcomes, losses = {}, {}
        for name, blocks in (("select", native), ("finite_ce_swap", frozen_blocks)):
            with torch.no_grad():
                pool, mask = S.pool_from_blocks(manager, blocks, device=device)
                try:
                    scored = E.score_probes(manager, qs, pool, 256, mask)
                finally:
                    del pool, mask
            outcomes[name] = scored["correct"]
            losses[name] = scored["gold_loss"]
        row_records = []
        for x in p["confirmation"]:
            q = x["question"]
            record = dict(x, row=row, full=full[q], no_memory=no_mem[q], eligible=eligible[q],
                          arms={name: outcomes[name][q] for name in outcomes},
                          gold_loss={name: losses[name] for name in losses})
            report["records"].append(record)
            row_records.append(record)
        report["windows"].append({"row": row, "context_sha256": context_sha,
                                  "native": native, "frozen_kept": frozen_blocks,
                                  "stats": window_stats(row_records, losses)})
        save(args.out, report)
    report["elapsed_seconds"] = time.monotonic() - started
    save(args.out, report)


if __name__ == "__main__":
    main()
