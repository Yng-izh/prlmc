"""Development-only exhaustive 19-choose-9 CE oracle (bounded by default)."""
import argparse
import hashlib
import itertools
import random
import time
from pathlib import Path

import sic_optimize as O
import sic_refine as R


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "source", "out"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--max-combos", type=int, default=100)
    args = ap.parse_args()
    if args.max_combos < 0:
        ap.error("--max-combos must be nonnegative")
    paths = [Path(getattr(args, name)).resolve() for name in ("plan", "source", "out")]
    if len(set(paths)) != 3:
        ap.error("plan, source and out must use distinct paths")
    plan, source, plan_hash, source_hash, windows, records = R.validate_inputs(
        paths[0].read_bytes(), paths[1].read_bytes())
    matches = [p for p in plan["plans"] if p["row"] == args.row]
    if len(matches) != 1:
        ap.error("--row must identify one frozen plan row")
    p = matches[0]

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

    lines = E.read_fact_lines(args.row)
    if hashlib.sha256("\n".join(lines).encode()).hexdigest() != p["context_sha256"]:
        raise ValueError("context changed after plan freeze")
    started = time.monotonic()
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
    covered = E.build_window(manager, lines, 19 * manager.block, 30)
    if covered != [(60 + 20*j, 80 + 20*j, 256*j, 256) for j in range(19)]:
        raise RuntimeError("unexpected written block layout")
    begin = manager.total - manager.used
    window = manager._memory[:, begin:begin + 19 * manager.block]
    native = _select_block_indices(
        window, 9 * manager.block, manager.block,
        manager._window_importance(begin, 19 * manager.block),
        manager.goal_direction, manager.select_weights).tolist()
    del window
    if native != windows[args.row]["kept"]["select"]:
        raise RuntimeError("native select differs from O1 source")
    probes = []
    for item in p["development"]:
        stem = E.query_stem(item["question"])
        tokens = manager.tokenizer(stem + " " + item["answer"], add_special_tokens=False).input_ids
        prefix = manager.tokenizer(stem, add_special_tokens=False).input_ids
        if len(tokens) > 256 or not 0 < len(prefix) < len(tokens):
            raise ValueError("invalid or truncated development answer tail")
        probes.append(Probe(question=item["question"], answers=[item["answer"]], prompt=stem))

    def evaluate(blocks):
        with torch.no_grad():
            pool, mask = S.pool_from_blocks(manager, blocks)
            try:
                score = E.score_probes(manager, probes, pool, 256, mask)
                R.validate_score(score, probes)
            finally:
                del pool, mask
                torch.cuda.empty_cache()
        return dict(kept=list(blocks), gold_loss=score["gold_loss"],
                    correct=sum(score["correct"].values()), n=score["n"],
                    eligible_wrong=sum(
                        records[(args.row, q)]["eligible"] and not correct
                        for q, correct in score["correct"].items()),
                    outcomes=score["correct"])

    select = evaluate(native)
    expected = {item["question"]: records[(args.row, item["question"])]["arms"]["select"]
                for item in p["development"]}
    if select["outcomes"] != expected:
        raise RuntimeError("select development outcomes differ from O1 source")
    select["source_gold_loss"] = windows[args.row]["gold_loss"]["select"]
    select["source_ce_delta"] = select["gold_loss"] - select["source_gold_loss"]
    report = dict(experiment="SIC-oracle", stage="develop", row=args.row, start=30,
                  plan_sha256=plan_hash, source_sha256=source_hash,
                  context_sha256=p["context_sha256"], development=p["development"],
                  objective="minimum mean development answer CE", total_combos=92378,
                  max_combos=args.max_combos, done=0, elapsed=0., oracle_exact=False,
                  select=select, top20=[])

    def checkpoint():
        report["elapsed"] = time.monotonic() - started
        report["oracle_exact"] = report["done"] == 92378
        if report["top20"]:
            report["best_ce_minus_select"] = report["top20"][0]["gold_loss"] - select["gold_loss"]
        O.save(paths[2], report)

    checkpoint()
    try:
        for blocks in itertools.islice(itertools.combinations(range(19), 9), args.max_combos):
            result = evaluate(blocks)
            report["done"] += 1
            report["top20"].append(result)
            report["top20"].sort(key=lambda x: (x["gold_loss"], x["kept"]))
            del report["top20"][20:]
            if report["done"] % 100 == 0:
                checkpoint()
                print("done", report["done"], "best CE", report["top20"][0]["gold_loss"],
                      "elapsed", round(report["elapsed"], 1), flush=True)
    finally:
        checkpoint()


if __name__ == "__main__":
    main()
