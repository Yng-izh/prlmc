#!/usr/bin/env python3
"""Zero-training evaluation of one frozen shared E6d carrier residual.

The episode supplies new memory contents only. The E6d parameters are loaded
once, remain frozen, and are reused unchanged for every development episode.
This entry point intentionally has no basis construction, alpha, optimizer, or
backward path.
"""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEVELOPMENT_IDS = (0, 2, 4, 8, 14, 15, 17, 18)
TEST_IDS = (1, 3, 5, 6, 7, 9, 10, 11, 12, 13, 16, 19, 20, 21, 22, 23)
WINDOW_BLOCKS = 19
TARGET_BLOCKS = 9
COMPARISON_ARMS = ("Full", "Selection", "SharedE6d", "Recent9", "Random9")
PROTOCOL = dict(
    name="S1d-shared-compactor-zero-training-v1",
    method="frozen E6d interleave CarrierResidualCompactor",
    arms=["Full", "Selection", "SharedE6d"],
    training=False,
    optimizer=None,
    backward=False,
    basis_construction=False,
    alpha_adaptation=False,
    window_blocks=WINDOW_BLOCKS,
    target_blocks=TARGET_BLOCKS,
    anchor_lines=30,
    written_lines=380,
    block_lines=20,
    evaluation_questions=64,
    prompt="fixed B2 FEWSHOT",
    stop_strings=["\n"],
    max_new_tokens=32,
    scorer="readout_probe.official_score",
    primary="episode paired SharedE6d-Selection generation EM",
    test_policy="development only; sealed test is never opened",
)


def evaluation_protocol(arms=None, random_seed=0, stage="development"):
    arms = list(arms or PROTOCOL["arms"])
    if len(set(arms)) != len(arms) or not set(PROTOCOL["arms"]).issubset(arms) or not set(arms).issubset(COMPARISON_ARMS):
        raise ValueError("unique arms must include Full, Selection, SharedE6d")
    protocol = dict(PROTOCOL) if arms == PROTOCOL["arms"] else dict(PROTOCOL, name="S1d-shared-baselines-v1", arms=arms,
                random_seed=random_seed, random_policy="one fixed sample; shared across layers; chronological placement")
    if stage == "test":
        protocol.update(stage="test", test_policy="all 16 sealed episodes; frozen manifest and single-open ledger required")
    elif stage != "development":
        raise ValueError("invalid evaluation stage")
    return protocol


def baseline_indices(kind, seed=0, window_blocks=WINDOW_BLOCKS):
    if kind == "Recent9":
        return list(range(window_blocks - TARGET_BLOCKS, window_blocks))
    if kind == "Random9":
        return sorted(random.Random(seed).sample(range(window_blocks), TARGET_BLOCKS))
    raise ValueError("unknown block baseline")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def choose_development(plan, episode):
    if plan.get("status") != "completed":
        raise ValueError("completed controlled plan required")
    development = [e for e in plan.get("episodes", [])
                   if e.get("partition") == "development"]
    if tuple(e.get("episode") for e in development) != DEVELOPMENT_IDS:
        raise ValueError("development split changed")
    if sum(e.get("partition") == "test" for e in plan.get("episodes", [])) != 16:
        raise ValueError("sealed test split changed")
    matches = [e for e in development if e.get("episode") == episode]
    if len(matches) != 1:
        raise ValueError("episode is not one of the eight development episodes")
    return matches[0]


def choose_episode(plan, episode, stage="development"):
    if stage == "development":
        return choose_development(plan, episode)
    choose_development(plan, DEVELOPMENT_IDS[0])
    tests = [item for item in plan["episodes"] if item["partition"] == "test"]
    if tuple(item["episode"] for item in tests) != TEST_IDS or episode not in TEST_IDS:
        raise ValueError("sealed test identity changed")
    return next(item for item in tests if item["episode"] == episode)


def freeze_manifest(plan, checkpoint, model_path, protocol, gamma_max):
    return dict(plan_sha256=sha(plan), checkpoint_sha256=sha(checkpoint),
                model_manifest_sha256=sha(Path(model_path) / "fp16_cache_manifest.json"),
                source_manifest=source_manifest(), protocol=protocol, gamma_max=gamma_max)


def validate_test_access(plan_path, freeze_path, manifest):
    frozen = json.loads(Path(freeze_path).read_text())
    if (frozen.get("status") != "frozen" or frozen.get("manifest") != manifest or
            frozen.get("plan_sha256") != sha(plan_path) or
            frozen.get("all_development_complete") is not True or frozen.get("test_scored") is not False):
        raise ValueError("sealed evaluation requires a matching complete method freeze")
    ledger = Path(plan_path).with_suffix(".s1d_test_opened.json")
    if not ledger.exists():
        raise ValueError("sealed test must be launched through the single-open fleet launcher")
    record = json.loads(ledger.read_text())
    if record.get("manifest") != manifest or record.get("freeze_sha256") != sha(freeze_path):
        raise ValueError("sealed test ledger does not match method freeze")


def validate_write_plan(episode):
    lines = episode.get("lines")
    if not isinstance(lines, list) or len(lines) != 410:
        raise ValueError("controlled episode must contain 410 lines")
    if hashlib.sha256("\n".join(lines).encode()).hexdigest() != episode.get("context_sha256"):
        raise ValueError("episode context hash mismatch")
    chunks = [lines[:30]] + [lines[30 + i * 20:50 + i * 20] for i in range(19)]
    if any(len(chunk) != expected for chunk, expected in
           zip(chunks, [30] + [20] * 19)):
        raise ValueError("controlled write chunk shape changed")
    return lines, chunks


def source_manifest():
    files = [
        "tools/s1d_shared_compactor_eval.py",
        "tools/s1d_c_generation.py",
        "tools/readout_probe.py",
        "tools/s1d_experiment1.py",
        "tools/run_s1d_shared_compactor_eval.py",
        "tools/s1d_controlled_contract.py",
        "tools/freeze_s1d_shared_method.py",
        "tools/s1d_shared_natural_eval.py",
        "tools/run_s1d_shared_natural_eval.py",
        "methods/prlmc/consolidate.py",
        "methods/prlmc/layout.py",
        "methods/prlmc/probes.py",
        "methods/memoryllm_modeling/modeling_memoryllm.py",
        "configs/agent_conf/prlmc_e6.yaml",
    ]
    return {item: sha(ROOT / item) for item in files}


def arm_summary(maps):
    questions = list(maps["Full"])
    if not questions:
        raise ValueError("nonempty paired question universe required")
    if any(set(values) != set(questions) for values in maps.values()):
        raise ValueError("arm question universes differ")
    full, selection, shared = (maps[name] for name in ("Full", "Selection", "SharedE6d"))
    lost = [q for q in questions if full[q] and not selection[q]]
    gained = [q for q in questions if shared[q] and not selection[q]]
    harmed = [q for q in questions if selection[q] and not shared[q]]
    return dict(
        n=len(questions),
        arms={name: dict(correct=sum(values.values()), em=sum(values.values()) / len(values))
              for name, values in maps.items()},
        full_selection_lost=lost,
        shared_gained_vs_selection=gained,
        shared_harmed_vs_selection=harmed,
        shared_vs_selection=dict(gained=gained, harmed=harmed,
                                 net=len(gained) - len(harmed),
                                 delta_em=(len(gained) - len(harmed)) / len(questions)),
        shared_recovers_lost=[q for q in gained if q in lost],
        **({"paired_comparisons": {
            name: dict(gained=[q for q in questions if shared[q] and not values[q]],
                       harmed=[q for q in questions if values[q] and not shared[q]],
                       net=sum(shared.values()) - sum(values.values()),
                       delta_em=(sum(shared.values()) - sum(values.values())) / len(questions))
            for name, values in maps.items() if name != "SharedE6d"}}
           if set(maps) != set(PROTOCOL["arms"]) else {}),
    )


def _make_compactor(manager, state, gamma_max, device):
    from methods.prlmc.consolidate import CarrierResidualCompactor
    arm = CarrierResidualCompactor(
        num_layers=manager.num_layers, hidden=manager.hidden,
        block=manager.block, rank=32, gamma_max=gamma_max,
    ).to(device=device, dtype=__import__("torch").float32)
    arm.load_state_dict(state, strict=True)
    arm.eval()
    for parameter in arm.parameters():
        parameter.requires_grad_(False)
    return arm


def _write_episode(manager, chunks):
    import torch
    writes = []
    with torch.no_grad():
        for index, chunk in enumerate(chunks):
            text = "\n".join(chunk)
            token_count = len(manager.tokenizer(text, add_special_tokens=False).input_ids)
            if token_count > 512:
                raise ValueError("controlled chunk would be truncated")
            record = (manager.initialize_anchor(text, max_tokens=512)
                      if index == 0 else manager.write(text, max_tokens=512))
            if record.get("dropped") or record.get("evicted_blocks") or record.get("compacted"):
                raise ValueError("write changed the append-only controlled protocol")
            writes.append(dict(record, input_tokens=token_count))
    if manager.used != WINDOW_BLOCKS * manager.block:
        raise ValueError("episode did not produce nineteen native blocks")
    return writes


def _candidate_pool(manager, compactor, kind, device, random_seed=0, window_blocks=WINDOW_BLOCKS):
    import torch
    from methods.prlmc.consolidate import select_block_compact
    from methods.prlmc.probes import _compacted_mask, _compacted_pool, compressed_pool
    begin = manager.total - manager.used
    source = manager._memory[:, begin:begin + window_blocks * manager.block, :].to(device)
    weights = manager._window_importance(begin, window_blocks * manager.block)
    if kind == "SharedE6d":
        pool, returned_mask = compressed_pool(manager, compactor, window_blocks, TARGET_BLOCKS, device=device)
    elif kind == "Selection":
        selected = select_block_compact(
            source, TARGET_BLOCKS * manager.block, block=manager.block,
            weights=weights, goal_direction=manager.goal_direction,
            select_weights=manager.select_weights,
        )
        pool = _compacted_pool(manager, device, selected,
                               window_blocks * manager.block, TARGET_BLOCKS * manager.block)
        # The plain selection path must be exactly the same carrier input used
        # by E6d at zero residual; this catches a changed placement or mask.
        with torch.no_grad():
            zero = compactor.__class__(num_layers=manager.num_layers, hidden=manager.hidden,
                block=manager.block, rank=compactor.rank, gamma_max=compactor.gamma_max).to(
                    device=device, dtype=torch.float32)
            zero.load_state_dict(compactor.state_dict())
            zero.up.weight.zero_()
            zero_pool, zero_mask = compressed_pool(manager, zero, window_blocks, TARGET_BLOCKS, device=device)
            if not torch.equal(pool, zero_pool):
                raise ValueError("Selection and zero-residual carrier pool diverged")
            del zero_pool, zero_mask, zero
    elif kind in ("Recent9", "Random9"):
        if kind == "Recent9":
            from methods.prlmc.consolidate import recent_block_compact
            selected = recent_block_compact(source, TARGET_BLOCKS * manager.block, manager.block)
        else:
            indices = torch.tensor(baseline_indices(kind, random_seed, window_blocks), device=device)
            selected = source.reshape(manager.num_layers, window_blocks, manager.block,
                                      manager.hidden).index_select(1, indices).reshape(
                                          manager.num_layers, TARGET_BLOCKS * manager.block, manager.hidden)
        pool = _compacted_pool(manager, device, selected,
                               window_blocks * manager.block, TARGET_BLOCKS * manager.block)
    else:
        raise ValueError("unknown compressed arm")
    mask = _compacted_mask(manager, manager.used - window_blocks * manager.block
                           + TARGET_BLOCKS * manager.block, device)
    if mask is None or int(mask.sum()) != manager.layout.k_anchor + TARGET_BLOCKS * manager.block:
        raise ValueError("compressed active-slot budget changed")
    if kind == "SharedE6d" and not torch.equal(mask, returned_mask):
        raise ValueError("shared active mask mismatch")
    return pool, mask


def _generate(model, manager, items, pool, mask, on_progress=None, prompt_template=None, scorer=None, max_new_tokens=32):
    import torch
    from readout_probe import FEWSHOT, official_score
    from methods.prlmc.probes import active_mask_override, differentiable_memory
    from s1d_experiment1 import reader_evaluation
    results = {}
    original = model.memory
    original_mask = model._buffers.get("memory_active_mask")
    # Full also needs a disposable pool: native generation can modify memory.
    if pool is None:
        pool = original.detach().clone()
    template = pool.detach().clone()
    with reader_evaluation(model, verify_memory=True), differentiable_memory(model, pool), active_mask_override(model, mask):
        if hasattr(model, "memory_attn_scale") and not torch.all(model.memory_attn_scale == 1):
            raise ValueError("reader attention scale differs from frozen B2")
        for index, item in enumerate(items):
            prompt = (prompt_template or FEWSHOT).format(question=item["question"])
            ids = manager.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(pool.device)
            try:
                with torch.inference_mode():
                    generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                        do_sample=False, use_cache=True, max_new_tokens=max_new_tokens, stop_strings=["\n"],
                        tokenizer=manager.tokenizer,
                        pad_token_id=(manager.tokenizer.pad_token_id if manager.tokenizer.pad_token_id is not None
                                      else manager.tokenizer.eos_token_id))
                if model.memory is not pool:
                    raise RuntimeError("generation replaced memory binding")
                prediction = manager.tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True)
                em, substring = (scorer or official_score)(prediction, item.get("answers", [item.get("answer")]))
                results[item["question"]] = dict(prediction=prediction, em=bool(em),
                    substring_em=bool(substring), generated_tokens=int(generated.shape[1] - ids.shape[1]))
            finally:
                if not torch.equal(pool, template):
                    with torch.no_grad():
                        pool.copy_(template)
            if on_progress is not None:
                on_progress(results)
            if index % 8 == 0 or index + 1 == len(items):
                print("[shared-eval] generation %d/%d" % (index + 1, len(items)), flush=True)
    if model.memory is not original or model._buffers.get("memory_active_mask") is not original_mask:
        raise RuntimeError("memory/mask restoration failed")
    return results


def state_digest(module):
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def atomic_json(path, payload):
    from native_qa_compare import atomic_json as write_json
    write_json(path, payload)


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    os.environ["S0B_GPU_SET"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    import yaml
    from transformers import AutoTokenizer
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory
    from s1d_experiment1 import memory_digest

    plan_path = Path(args.plan)
    stage = getattr(args, "stage", "development")
    protocol = evaluation_protocol(getattr(args, "arms", None), getattr(args, "random_seed", 0), stage)
    plan = json.loads(plan_path.read_text())
    if stage == "test":
        if not args.freeze:
            raise ValueError("method freeze required before test evaluation")
        validate_test_access(plan_path, args.freeze, freeze_manifest(plan_path, args.checkpoint, args.model_path, protocol, args.gamma_max))
    episode = choose_episode(plan, args.episode, stage)
    lines, chunks = validate_write_plan(episode)
    out = Path(args.out)
    if out.exists():
        raise ValueError("output exists; refuse to overwrite frozen evaluation")
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise ValueError("shared checkpoint missing")
    model_manifest = Path(args.model_path) / "fp16_cache_manifest.json"
    identity = dict(stage=stage, episode=args.episode, plan_sha256=sha(plan_path),
                    context_sha256=episode["context_sha256"], checkpoint_sha256=sha(ckpt),
                    model_manifest_sha256=sha(model_manifest), source_manifest=source_manifest(),
                    protocol=protocol, gamma_max=args.gamma_max, test_opened=stage == "test")
    report = dict(schema_version="s1d_shared_compactor_eval/1", status="running", identity=identity,
                  evaluation_opened=False, training_performed=False, optimizer_created=False,
                  backward_performed=False, basis_constructed=False, alpha_adapted=False,
                  episode=args.episode)
    atomic_json(out, report)
    started = time.monotonic()
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.cuda.manual_seed_all(0)
    raw = yaml.safe_load((ROOT / "configs/agent_conf/prlmc_e6.yaml").read_text())
    config = dict(raw["prlmc"])
    config.update(device="cuda:0", dtype="float16", active_memory_mask=True, trigger="none",
                  mode="ours", write_mode="append", gate_mode="ones", tau_alloc=-1.,
                  projection="none", basis_mode="none", functional_control="none",
                  compact_probe=False, stats_dir=None, carrier_gamma_max=args.gamma_max)
    cfg = PRLMCConfig.from_dict(config)
    model = MemoryLLM.from_pretrained(args.model_path, torch_dtype=torch.float16,
                                      attn_implementation="sdpa").to("cuda:0").eval()
    model.initialized.zero_()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("reader is not frozen")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    manager = ProtectedResidualMemory(model, tokenizer, cfg)
    saved = torch.load(ckpt, map_location="cuda:0", weights_only=True)
    if saved.get("evaluation_eligible") is False:
        raise ValueError("smoke checkpoint is ineligible for formal evaluation")
    if "effective_gamma_max" in saved and saved["effective_gamma_max"] != args.gamma_max:
        raise ValueError("checkpoint gamma_max differs from evaluation configuration")
    state = saved["state_dict"]
    compactor = _make_compactor(manager, state, args.gamma_max, "cuda:0")
    if any(parameter.requires_grad for parameter in compactor.parameters()):
        raise ValueError("shared compactor is not frozen")
    frozen_hash = state_digest(compactor)
    if saved.get("final_state_sha256", frozen_hash) != frozen_hash:
        raise ValueError("checkpoint state hash mismatch")
    report["checkpoint_training_protocol"] = saved.get("protocol")
    report["shared_parameter_count"] = sum(parameter.numel() for parameter in compactor.parameters())
    report["shared_state_bytes"] = sum(tensor.numel() * tensor.element_size() for tensor in compactor.state_dict().values())
    print("[shared-eval] checkpoint loaded, writing episode", args.episode, flush=True)
    writes = _write_episode(manager, chunks)
    source_hash = memory_digest(model.memory)
    report.update(writes=writes, source_memory_sha256=source_hash,
                  written_slots=manager.used, compressed_slots=manager.layout.k_anchor + TARGET_BLOCKS * manager.block)
    atomic_json(out, report)
    evaluation = episode["splits"]["evaluation"]
    if len(evaluation) != 64 or len({item["question"] for item in evaluation}) != 64:
        raise ValueError("evaluation split must contain 64 unique questions")
    # Evaluation questions enter only after the shared checkpoint and memory are fixed.
    report["evaluation_opened"] = True
    atomic_json(out, report)
    predictions = {}
    items = list(evaluation)
    full_mask = model._buffers.get("memory_active_mask")
    if full_mask is None or int(full_mask.sum()) != manager.layout.k_anchor + manager.used:
        raise ValueError("full active memory mask is invalid")
    report["predictions"] = predictions
    report["compactor_state_sha256_before"] = frozen_hash
    report["arm_metadata"] = {}
    # Build and free one arm at a time to fit the 32 GiB reader GPU.
    for name in protocol["arms"]:
        print("[shared-eval] arm", name, flush=True)
        torch.cuda.synchronize()
        arm_started = time.monotonic()
        with torch.no_grad():
            if name == "Full":
                pool, mask = None, full_mask.detach().clone()
            else:
                pool, mask = _candidate_pool(manager, compactor, name, "cuda:0", getattr(args, "random_seed", 0))
        report["arm_metadata"][name] = dict(active_slots=int(mask.sum()))
        if name in ("Recent9", "Random9"):
            report["arm_metadata"][name]["kept_blocks"] = baseline_indices(name, getattr(args, "random_seed", 0))
        torch.cuda.synchronize()
        report["arm_metadata"][name]["compaction_seconds"] = time.monotonic() - arm_started
        def save_progress(values):
            predictions[name] = values
            report["progress"] = dict(arm=name, completed_questions=len(values), total_questions=len(items))
            atomic_json(out, report)
        predictions[name] = _generate(model, manager, items, pool, mask, save_progress)
        torch.cuda.synchronize()
        report["arm_metadata"][name]["total_seconds"] = time.monotonic() - arm_started
        if memory_digest(model.memory) != source_hash:
            raise ValueError("generation changed source memory")
        if state_digest(compactor) != frozen_hash:
            raise ValueError("shared compactor changed during evaluation")
        del pool, mask
        torch.cuda.empty_cache()
    maps = {name: {question: row["em"] for question, row in values.items()}
            for name, values in predictions.items()}
    summary = arm_summary(maps)
    if memory_digest(model.memory) != source_hash:
        raise ValueError("generation changed source memory")
    report.update(status="complete", predictions=predictions, summary=summary,
        generation_questions=[item["question"] for item in items], source_memory_unchanged=True, compactor_unchanged=True,
        compactor_state_sha256_after=state_digest(compactor),
        costs=dict(elapsed_seconds=time.monotonic() - started,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved()),
        runtime=dict(torch=torch.__version__, cuda=torch.version.cuda,
                     gpu=torch.cuda.get_device_name(0), physical_device=args.cuda_device))
    atomic_json(out, report)
    print("FINAL", json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="runs/sic_paper_controlled_plan.json")
    parser.add_argument("--checkpoint", default="runs/e6d_interleave_ckpt.pt")
    parser.add_argument("--model-path", default="/dev/shm/s1d_reader_fp16")
    parser.add_argument("--gamma-max", type=float, default=.05)
    parser.add_argument("--arms", type=lambda text: text.split(","), default=None)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--stage", choices=("development", "test"), default="development")
    parser.add_argument("--freeze")
    parser.add_argument("--cuda-device", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise ValueError("output exists; refuse overwrite frozen evaluation")
    try:
        run(args)
    except Exception as exc:
        output = Path(args.out)
        if output.exists():
            report = json.loads(output.read_text())
            if report.get("status") == "running":
                report.update(status="failed", failure=dict(type=type(exc).__name__, message=str(exc)))
                atomic_json(output, report)
        raise


if __name__ == "__main__":
    main()
