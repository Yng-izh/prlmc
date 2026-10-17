#!/usr/bin/env python3
"""Train one shared rank32 E6d on independent scenes, never development/test QA."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prepare_s1d_shared_training import validate_training_plan
from s1d_shared_compactor_eval import (atomic_json, sha, source_manifest, state_digest,
                                     validate_write_plan, _write_episode)


def training_schedule(n, seed, steps_a=4, steps_b=96, chunk=4):
    if n < 1 or steps_a < 0 or steps_b <= 0 or chunk <= 0 or steps_b % chunk:
        raise ValueError("invalid interleave budget")
    order = list(range(n))
    return [("A", index, steps_a) for index in order if steps_a] + [
        ("B", index, chunk) for _ in range(steps_b // chunk) for index in order]


def train_events(events, train_visit, on_event):
    """Reusable executor; caller owns the single module and optimizer."""
    cumulative = 0
    for index, (stage, episode, steps) in enumerate(events):
        result = train_visit(stage, episode, steps)
        cumulative += steps
        on_event(dict(event=index, stage=stage, episode=episode, steps=steps,
                      cumulative_steps=cumulative, training=result))


def run(args):
    # Data isolation and output checks happen before any CUDA/model allocation.
    plan_path = Path(args.training_plan)
    reserved_path = Path(args.reserved_plan)
    plan = json.loads(plan_path.read_text())
    reserved = json.loads(reserved_path.read_text())
    validate_training_plan(plan, reserved)
    if plan.get("token_budget_verified") is not True:
        raise ValueError("training token audit required")
    out, checkpoint = Path(args.out), Path(args.checkpoint_out)
    if out.exists() or checkpoint.exists() or out.resolve() == checkpoint.resolve():
        raise ValueError("distinct new report/checkpoint paths required")
    events = training_schedule(len(plan["episodes"]), args.seed, args.steps_a, args.steps_b, args.chunk)
    if not args.smoke and (args.steps_a, args.steps_b, args.chunk) != (4, 96, 4):
        raise ValueError("formal training fixes A4/B96/chunk4; alternate budgets are smoke only")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    os.environ["S0B_GPU_SET"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    import yaml
    from transformers import AutoTokenizer
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory
    from methods.prlmc.consolidate import CarrierResidualCompactor
    from methods.prlmc.probes import Probe, train_carrier_compactor, clear_teacher_cache
    from readout_probe import FEWSHOT
    from s1d_experiment1 import memory_digest
    torch.manual_seed(args.seed)
    __import__("random").seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    protocol = dict(name="S1d-independent-shared-E6d-training-v1", rank=32, gamma_max=.05,
        window_blocks=19, target_blocks=9, optimizer="AdamW", learning_rate=.001, weight_decay=.01,
        steps_a=args.steps_a, steps_b=args.steps_b, interleave_chunk=args.chunk, seed=args.seed,
        supervision="20 predetermined latest write-time training facts per scene; fixed B2",
        objective="unchanged probes._stage_weights A/B", initialization="fresh; zero up projection",
        order="fixed episode order in every round; seed controls initialization",
        checkpoint_selection="final scheduled step only", development_scored=False, sealed_test_opened=False,
        smoke=args.smoke)
    manifest = source_manifest()
    for name in ("tools/s1d_shared_compactor_train.py", "tools/prepare_s1d_shared_training.py",
                 "tools/sic_paper_data.py", "methods/prlmc/probes.py"):
        manifest[name] = sha(ROOT / name)
    report = dict(status="running", protocol=protocol, training_plan_sha256=sha(plan_path),
        reserved_plan_sha256=sha(reserved_path), source_manifest=manifest,
        model_manifest_sha256=sha(Path(args.model_path) / "fp16_cache_manifest.json"),
        history=[], training_episodes=len(plan["episodes"]), development_scored=False, test_opened=False)
    atomic_json(out, report)
    started = time.monotonic()
    raw = yaml.safe_load((ROOT / "configs/agent_conf/prlmc_e6.yaml").read_text())["prlmc"]
    raw.update(device="cuda:0", dtype="float16", active_memory_mask=True, trigger="none",
               mode="ours", write_mode="append", gate_mode="ones", tau_alloc=-1., projection="none",
               basis_mode="none", functional_control="none", compact_probe=False, stats_dir=None,
               carrier_gamma_max=.05, carrier_rank=32)
    model = MemoryLLM.from_pretrained(args.model_path, torch_dtype=torch.float16,
                                     attn_implementation="sdpa").to("cuda:0").eval()
    model.requires_grad_(False)
    model.initialized.zero_()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if plan["tokenizer_manifest"] != __import__("sic_paper_data").tokenizer_manifest(args.model_path):
        raise ValueError("training tokenizer differs from audited tokenizer")
    manager = ProtectedResidualMemory(model, tokenizer, PRLMCConfig.from_dict(raw))
    compactor = CarrierResidualCompactor(manager.num_layers, manager.hidden, manager.block,
                                        rank=32, gamma_max=.05).to("cuda:0")
    optimizer = torch.optim.AdamW(compactor.parameters(), lr=.001, weight_decay=.01)
    report["initial_state_sha256"] = state_digest(compactor)
    report["shared_parameter_count"] = sum(parameter.numel() for parameter in compactor.parameters())
    report["shared_state_bytes"] = sum(tensor.numel() * tensor.element_size() for tensor in compactor.state_dict().values())
    memory_hashes = {}
    clear_teacher_cache()

    def visit(stage, index, steps):
        episode = plan["episodes"][index]
        _, chunks = validate_write_plan(episode)
        # initialize_anchor resets manager + mask + statistics on every visit.
        _write_episode(manager, chunks)
        digest = memory_digest(model.memory)
        if memory_hashes.setdefault(index, digest) != digest:
            raise ValueError("rebuilt training scene changed; cached teacher invalid")
        probes = [Probe(question=item["question"], answers=[item["answer"]], split="train",
                        prompt=FEWSHOT.format(question=item["question"])) for item in episode["training_probes"]]
        if any(len(tokenizer(probe.prompt + " " + probe.answers[0], add_special_tokens=False).input_ids) > 256 for probe in probes):
            raise ValueError("training prompt/answer exceeds256 tokens; refuse truncation")
        probe_hash = __import__("sic_paper_data").digest(episode["training_probes"])
        result = train_carrier_compactor(manager, probes, compactor=compactor, optimiser=optimizer,
            steps=steps, lr=.001, stage=stage, window_blocks=19, target_blocks=9, max_probe_tokens=256,
            teacher_cache_key=(sha(plan_path), index, digest, probe_hash))
        if memory_digest(model.memory) != digest:
            raise ValueError("training mutated source memory")
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("reader became trainable")
        return result

    def log_event(event):
        report["history"].append(event)
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(out, report)
        print("[shared-train]", json.dumps(event), flush=True)

    train_events(events[:1] if args.smoke else events, visit, log_event)
    clear_teacher_cache()
    final_hash = state_digest(compactor)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    torch.save(dict(state_dict=compactor.state_dict(), protocol=protocol,
                    training_plan_sha256=sha(plan_path), effective_gamma_max=.05,
                    final_state_sha256=final_hash, evaluation_eligible=not args.smoke), temporary)
    temporary.replace(checkpoint)
    report.update(status="smoke_complete" if args.smoke else "complete",
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), final_state_sha256=final_hash,
        source_memory_hashes=memory_hashes,
        costs=dict(elapsed_seconds=time.monotonic() - started,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved()),
        runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)))
    atomic_json(out, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-plan", required=True)
    parser.add_argument("--reserved-plan", default="runs/sic_paper_controlled_plan.json")
    parser.add_argument("--model-path", default="/dev/shm/s1d_reader_fp16")
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--cuda-device", type=int, required=True)
    parser.add_argument("--steps-a", type=int, default=4)
    parser.add_argument("--steps-b", type=int, default=96)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint-out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists() or Path(args.checkpoint_out).exists():
        raise ValueError("training artifacts exist; refuse overwrite")
    try:
        run(args)
    except Exception as exc:
        path = Path(args.out)
        if path.exists():
            report = json.loads(path.read_text())
            if report.get("status") == "running":
                report.update(status="failed", failure=dict(type=type(exc).__name__, message=str(exc)))
                atomic_json(path, report)
        raise


if __name__ == "__main__":
    main()
