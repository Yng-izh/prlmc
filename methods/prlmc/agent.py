"""MemoryAgentBench adapter for PRLM-C.

Loaded from AgentWrapper._initialize_agent_by_type and AgentWrapper.send_message
through a `prlmc` branch, which must be tested before the `rag` branch because
_is_agent_type does substring matching.

Contract, inherited from the MemoryLLM baseline
----------------------------------------------
Memorization calls send_message(chunk, memorizing=True) once per chunk; querying
calls send_message(query, memorizing=False) and reads a dict back. There is no
retriever and no external store: the answer comes from the memory pool alone.

What this adapter adds over the plain MemoryLLM baseline
--------------------------------------------------------
The first memorized chunk is treated as the episode's goal plus hard constraints
and is encoded into the read-only Anchor region once (section 3.1). Every later
chunk goes through the residual write path, which measures feature novelty,
applies the write gate and the optional Anchor projection, and consults the
capacity trigger before consolidating (sections 3.2 and 3.3). Per-episode
statistics and the cost axes of section 5.2 are written next to the benchmark
output so a run can be audited without re-executing it.

Scope note on the Anchor source
-------------------------------
`anchor_source=first_chunk` is the default because MemoryAgentBench streams
context chunks and does not expose a separate goal field. The note (section 4.1)
forbids building the Anchor from the final question, its answer, or any future
information; using only the first chunk respects that, since it is visible before
any query. A run whose research claim depends on the Anchor being a *task
instruction* rather than the first context chunk must set `anchor_source=explicit`
with `anchor_text`, which is why that path exists.
"""

import json
import os
import time
from typing import Any, Dict, Optional

import torch

from methods.prlmc.baselines import config_overrides_for_mode, get_mode
from methods.prlmc.budget import BudgetTracker
from methods.prlmc.config import PRLMCConfig
from methods.prlmc.layout import MIN_INJECT_TOKENS, ProtectedResidualMemory
from utils.templates import get_template

MODELING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "memoryllm_modeling")
MODELING_DIR = os.path.normpath(MODELING_DIR)

_model = None
_tokenizer = None

_STATS_FILENAME = "prlmc_stats.jsonl"


def _load_modeling_module():
    """Import the frozen MemoryLLM modeling fork by path, not by package name."""
    import importlib
    import sys

    if MODELING_DIR not in sys.path:
        sys.path.insert(0, MODELING_DIR)
    return importlib.import_module("modeling_memoryllm")


def get_backbone(model_path: str, dtype: str = "float16", device: str = "cuda:0"):
    """Load the frozen backbone once per process.

    float16 is the default because the checkpoint ships float32 (39.1 GB) and a
    V100 has 32 GB. bfloat16 is not used: sm_70 has no hardware bf16.

    attn_implementation="sdpa" is required rather than preferred. The shipped
    modeling code overrides forward with a prefix_token_length argument that
    offsets RoPE on q only; the flash-attention path is unreachable here and V100
    is sm_70 in any case.
    """
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    MemoryLLM = _load_modeling_module().MemoryLLM
    from transformers import AutoTokenizer

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    model = MemoryLLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
    )
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    _model, _tokenizer = model, tokenizer
    return _model, _tokenizer


def build_config(agent_config: Dict[str, Any]) -> PRLMCConfig:
    """Assemble the PRLM-C config from the agent YAML.

    Precedence is: mode defaults < explicit prlmc block. The mode block supplies
    the values that define an ablation arm (anchor on/off, compaction kind,
    trigger), so a config that only names a mode still produces a well-defined
    arm rather than a partially specified one.
    """
    raw = dict(agent_config.get("prlmc") or {})

    mode_code = raw.pop("mode_code", None) or agent_config.get("ablation_mode", "OURS")
    overrides = config_overrides_for_mode(mode_code)

    structural = raw.pop("structural_ablation", None)
    if structural:
        from methods.prlmc.baselines import apply_structural_ablation

        overrides = apply_structural_ablation(overrides, structural)

    overrides.update(raw)

    if "device" not in overrides:
        overrides["device"] = agent_config.get("device", "cuda:0")
    if "dtype" not in overrides:
        overrides["dtype"] = agent_config.get("dtype", "float16")
    if "stats_dir" not in overrides:
        overrides["stats_dir"] = agent_config.get("output_dir")

    config = PRLMCConfig.from_dict(overrides)
    config.mode_code = mode_code
    return config


def initialize_prlmc_agent(agent, agent_config: Dict[str, Any]) -> None:
    """Wire a PRLM-C manager into an AgentWrapper.

    Called once per benchmark context, so the pool is rebuilt per context and
    contexts stay independent. The Anchor is not written here: it is encoded from
    the first memorized chunk, which is the first information visible in the
    episode.
    """
    config = build_config(agent_config)

    if config.seed is not None:
        torch.manual_seed(int(config.seed))

    model, tokenizer = get_backbone(
        agent_config["model_path"], dtype=config.dtype, device=config.device
    )

    manager = ProtectedResidualMemory(model, tokenizer, config)

    agent.prlmc_config = config
    agent.prlmc_manager = manager
    agent.prlmc_model = model
    agent.prlmc_tokenizer = tokenizer
    agent.prlmc_device = config.device
    agent.prlmc_inject_max_tokens = agent_config.get("inject_max_tokens", 512)
    agent.prlmc_max_new_tokens = agent_config.get("max_new_tokens", 32)
    agent.prlmc_budget = BudgetTracker(manager, context_max_length=agent_config.get("input_length_limit"))
    agent.prlmc_anchored = False
    agent.prlmc_writes = 0
    agent.prlmc_queries = 0
    agent.prlmc_write_seconds = 0.0
    agent.prlmc_cumulative_probe_loss = 0.0
    agent.agent_start_time = time.time()

    if config.verbose:
        print(
            "\nPRLM-C: mode=%s anchor=%s k_anchor=%d projection=%s compact=%s trigger=%s\n"
            % (
                config.mode_code,
                config.anchor_source,
                manager.layout.k_anchor,
                config.projection,
                config.compact_mode,
                config.trigger,
            )
        )


def _anchor_text(agent, message: str) -> Optional[str]:
    """Determine the text that seeds the Anchor for this episode."""
    config = agent.prlmc_config
    if config.anchor_source == "none":
        return None
    if config.anchor_source == "explicit":
        return config.anchor_text
    return message


def handle_prlmc_agent(agent, message: str, memorizing: bool):
    """Memorize one chunk, or answer one query from the pool.

    Returns "" while memorizing and a standard response dict while querying, so
    the benchmark's metrics_summarization receives the same shape it does from
    every other baseline.
    """
    manager: ProtectedResidualMemory = agent.prlmc_manager
    config: PRLMCConfig = agent.prlmc_config

    if memorizing:
        return _memorize_chunk(agent, manager, config, message)

    return _answer_query(agent, manager, config, message)


def _memorize_chunk(agent, manager, config, message: str) -> str:
    """One write into the pool: the Anchor on the first chunk, a residual after."""
    template = get_template(agent.sub_dataset, "memorize", agent.agent_name)
    formatted = template.format(
        context=message,
        **(
            {"time_stamp": time.strftime("%Y-%m-%d %H:%M:%S")}
            if "{time_stamp}" in template
            else {}
        ),
    )

    if not agent.prlmc_anchored:
        anchor_text = _anchor_text(agent, formatted)
        info = manager.initialize_anchor(
            anchor_text or "", max_tokens=agent.prlmc_inject_max_tokens
        )
        agent.prlmc_anchored = True
        agent.prlmc_anchor_info = info
        agent.prlmc_budget.mark("anchor")
        if info.get("anchor_skipped"):
            # The episode's first chunk was too short to inject safely. It still
            # has to be written somewhere or it is silently lost, so it falls
            # through to the residual path instead of being dropped.
            record = manager.write(formatted, max_tokens=agent.prlmc_inject_max_tokens)
            _record_write(agent, record)
        return ""

    record = _write_with_probes(agent, manager, config, formatted)
    _record_write(agent, record)
    return ""


def _write_with_probes(agent, manager, config, formatted: str) -> Dict[str, Any]:
    """Residual write, optionally preceded by a functional probe of the candidate.

    Section 3.4 requires compaction to be judged by probe loss on a *copy* of the
    state. The same machinery is reused here so that a compaction triggered by
    this write is measured before it is committed; the live pool seen by the agent
    is unchanged until the write itself.
    """
    if config.compact_probe and manager.used >= 2 * manager.block:
        probes = getattr(agent, "prlmc_probes", None)
        if probes:
            from methods.prlmc.probes import evaluate_compaction

            try:
                report = evaluate_compaction(manager, probes)
                agent.prlmc_cumulative_probe_loss += max(0.0, report.degradation)
                agent.prlmc_last_probe_report = report.to_dict()
            except Exception as error:  # noqa: BLE001 - diagnostics must not stop the run
                agent.prlmc_last_probe_report = {"error": repr(error)}

    return manager.write(formatted, max_tokens=agent.prlmc_inject_max_tokens)


def _record_write(agent, record: Dict[str, Any]) -> None:
    if record.get("dropped"):
        return
    agent.prlmc_writes += 1
    agent.prlmc_budget.record_write(record.get("encode_seconds", 0.0), record.get("tokens", 0))


def _answer_query(agent, manager, config, message: str):
    """Generate from the pool alone and package the standard response."""
    memory_construction_time = time.time() - agent.agent_start_time

    max_new_tokens = getattr(agent, "max_tokens", None) or agent.prlmc_max_new_tokens

    input_ids = agent.prlmc_tokenizer(
        message, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(agent.prlmc_device)

    started = time.time()
    with torch.no_grad():
        outputs = agent.prlmc_model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=(
                agent.prlmc_tokenizer.pad_token_id or agent.prlmc_tokenizer.eos_token_id
            ),
        )
    elapsed = time.time() - started
    generated = outputs[0][input_ids.shape[1]:]
    text = agent.prlmc_tokenizer.decode(generated, skip_special_tokens=True)

    agent.prlmc_queries += 1
    agent.prlmc_budget.record_query(elapsed, int(input_ids.shape[1]))

    output = agent._create_standard_response(
        text,
        int(input_ids.shape[1]),
        int(generated.shape[0]),
        memory_construction_time,
        elapsed,
    )

    # Reset the clock so the next query's construction time excludes this read.
    agent.agent_start_time = time.time()

    _write_stats(agent, output)
    return output


def _write_stats(agent, output: Dict[str, Any]) -> None:
    """Append one JSON line of PRLM-C internals next to the benchmark output.

    Recorded per query so that a capacity plot can be reconstructed without
    re-running; the note asks for accuracy-cost curves rather than a single score
    (section 6.2).
    """
    config: PRLMCConfig = agent.prlmc_config
    directory = config.stats_dir
    if not directory:
        return

    try:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, _STATS_FILENAME)
        manager: ProtectedResidualMemory = agent.prlmc_manager
        payload = {
            "sub_dataset": agent.sub_dataset,
            "agent_name": agent.agent_name,
            "mode": getattr(config, "mode_code", config.mode),
            "query_index": agent.prlmc_queries,
            "writes": agent.prlmc_writes,
            "output_len": output.get("output_len"),
            "query_time_len": output.get("query_time_len"),
            "memory_construction_time": output.get("memory_construction_time"),
            "manager": manager.stats(),
            "budget": agent.prlmc_budget.snapshot().to_dict(),
            "cumulative_probe_loss": agent.prlmc_cumulative_probe_loss,
            "last_probe_report": getattr(agent, "prlmc_last_probe_report", None),
            # Per-write signals since the previous query. Calibration needs the
            # novelty / occupancy / rank trajectory of every write, not just the
            # final aggregate, and draining here keeps each line bounded.
            "writes_since_last_query": manager.drain_history(),
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - stats must never break an evaluation run
        pass


def register_probes(agent, question_answer_pairs) -> int:
    """Attach training-split probes so compaction can be checked functionally.

    Called by the caller that owns the episode, not by the benchmark loop, so the
    test question and its answer never enter the probe set (section 3.4). Returns
    the number of probes registered.
    """
    from methods.prlmc.probes import make_probes

    probes = make_probes(question_answer_pairs, split="train")
    agent.prlmc_probes = probes
    return len(probes)


def mode_summary() -> Dict[str, Any]:
    """Expose the implemented arms and their guarantees for a results table."""
    from methods.prlmc.baselines import GUARANTEES, implemented_modes

    return {
        "implemented": implemented_modes(),
        "guarantees": GUARANTEES,
        "min_inject_tokens": MIN_INJECT_TOKENS,
        "modeling_dir": MODELING_DIR,
        "modes": {code: get_mode(code).name for code in implemented_modes()},
    }
