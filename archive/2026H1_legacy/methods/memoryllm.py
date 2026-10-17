"""MemoryLLM as a MemoryAgentBench baseline method.

MemoryLLM (Wang et al., 2024) is a Llama-3-8B continuously pre-trained with a
per-layer memory pool. `inject_memory(context_ids, update_memory=True)` runs a
forward pass in *injection* mode and writes the resulting delta memory into the
pool; `generate(...)` then answers from the pool alone, with no context passed at
query time. That is exactly the write-then-query contract MemoryAgentBench
expects, so this baseline needs no retriever and no external store.

Three details are load-bearing and easy to get wrong.

  1. attn_implementation="sdpa". The shipped modeling code subclasses
     transformers' Llama and overrides forward with a prefix_token_length
     argument that offsets RoPE on q only. flash_attention_2 is unreachable here
     (the pool is prepended as embeddings, not as a KV cache) and V100 is sm_70
     besides; eager takes a different path inside the same file.
  2. The memory pool ships initialized=1 in the checkpoint, carrying whatever was
     in the pool when the authors exported it. MemoryAgentBench runs many
     independent contexts in one process, so the pool is reset before each
     context. See reset_memory for the two defensible choices.
  3. Injection has a hard floor of 16 tokens. Below that the model disturbs
     rather than updates the pool, and the model card says so explicitly.

Import isolation: the modeling file is a frozen fork of transformers ~4.43 and is
imported from methods/memoryllm_modeling/, never from site-packages. The
mllm_env venv pins transformers 4.48.2 for it.
"""

import importlib
import os
import sys
import time

from utils.templates import get_template

MODELING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memoryllm_modeling")

# Minimum injection length. Shorter contexts disturb the pool rather than update
# it, so they are dropped instead of injected.
MIN_INJECT_TOKENS = 16

_model = None
_tokenizer = None


def _load_modeling_module():
    """Import the frozen MemoryLLM modeling fork by path, not by package name."""
    if MODELING_DIR not in sys.path:
        sys.path.insert(0, MODELING_DIR)
    return importlib.import_module("modeling_memoryllm")


def get_memoryllm(model_path, dtype="float16", device="cuda:0"):
    """Load the MemoryLLM model and tokenizer once per process.

    Weights ship as float32 (39.1 GB), which does not fit a 32 GB V100, so the
    default is float16 (about 19.6 GB). bfloat16 is deliberately not used: sm_70
    has no hardware bf16.

    Returns:
        (model, tokenizer): the shared MemoryLLM instance and its tokenizer.
    """
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    import torch

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


def reset_memory(model):
    """Clear the pool so the context about to be memorized starts cold.

    The checkpoint ships initialized=1 over a populated pool, so without this
    every context would inherit the previous context's document plus the
    authors' export-time leftovers. Setting initialized back to 0 makes
    update_memory_with_delta_memory take its first-write branch: the first
    injected chunk overwrites the pool instead of concatenating onto it.

    The alternative, snapshotting the shipped pool and restoring that snapshot
    per context, would keep the authors' warm state. We do not use it, because
    the shipped pool content is arbitrary and a cold start is what makes contexts
    comparable across the benchmark.
    """
    model.initialized.zero_()


def inject(model, tokenizer, text, device="cuda:0", max_tokens=512):
    """Inject one chunk of context into the memory pool.

    Args:
        text: the chunk to memorize.
        max_tokens: truncate to this many tokens. MemoryLLM was trained at
            max_length=512 and the reference evaluation truncates contexts to
            512, so longer chunks are cut here rather than summarized. Set the
            benchmark chunk_size to 512 or less and no truncation happens.

    Returns:
        dict with token counts and elapsed seconds, or None when the chunk is
        below MIN_INJECT_TOKENS.
    """
    import torch

    ids = tokenizer(text, add_special_tokens=False).input_ids
    n_raw = len(ids)

    if n_raw < MIN_INJECT_TOKENS:
        return None

    truncated = n_raw > max_tokens
    ids = ids[:max_tokens]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    started = time.time()
    with torch.no_grad():
        model.inject_memory(input_ids, update_memory=True)

    return {
        "tokens": len(ids),
        "raw_tokens": n_raw,
        "truncated": truncated,
        "seconds": time.time() - started,
    }


def answer(model, tokenizer, question, device="cuda:0", max_new_tokens=32):
    """Generate an answer from the memory pool alone.

    MemoryAgentBench applies each dataset's query template before calling this,
    so the question already carries the task instruction.

    Returns:
        dict with the decoded answer, token counts and elapsed seconds.
    """
    import torch

    input_ids = tokenizer(question, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    started = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = outputs[0][input_ids.shape[1]:]

    return {
        "output": tokenizer.decode(generated, skip_special_tokens=True),
        "input_tokens": int(input_ids.shape[1]),
        "output_tokens": int(generated.shape[0]),
        "seconds": time.time() - started,
    }


def initialize_memoryllm_agent(agent, agent_config):
    """Wire a MemoryLLM instance into an AgentWrapper.

    Called from AgentWrapper._initialize_agent_by_type. A new AgentWrapper is
    built for every benchmark context, so resetting the pool here is what keeps
    contexts independent.
    """
    device = agent_config.get("device", "cuda:0")
    dtype = agent_config.get("dtype", "float16")

    model, tokenizer = get_memoryllm(agent_config["model_path"], dtype=dtype, device=device)
    reset_memory(model)

    agent.mllm_model = model
    agent.mllm_tokenizer = tokenizer
    agent.mllm_device = device
    agent.mllm_max_new_tokens = agent_config.get("max_new_tokens", 32)
    agent.mllm_inject_max_tokens = agent_config.get("inject_max_tokens", 512)
    agent.mllm_injected = 0
    agent.mllm_skipped = 0
    agent.agent_start_time = time.time()


def handle_memoryllm_agent(agent, message, memorizing):
    """Memorize one chunk, or answer one query from the pool.

    Memorization returns an empty string: the benchmark only reads the return
    value of a query, and MemoryLLM has no acknowledgement to give.

    Args:
        agent: the AgentWrapper holding the model, tokenizer and bookkeeping.
        message: the chunk to memorize, or the fully templated query.
        memorizing: True to write into the pool, False to read from it.

    Returns:
        str when memorizing, dict when answering.
    """
    if memorizing:
        # Wrap the chunk in the dataset's memorize template before injecting, the
        # same way _handle_long_context_agent and _handle_rag_agent do. Injecting
        # the bare chunk would put this row on a different input format from every
        # other baseline in the benchmark.
        memorize_template = get_template(agent.sub_dataset, "memorize", agent.agent_name)
        formatted = memorize_template.format(
            context=message,
            **(
                {"time_stamp": time.strftime("%Y-%m-%d %H:%M:%S")}
                if "{time_stamp}" in memorize_template
                else {}
            ),
        )

        result = inject(
            agent.mllm_model,
            agent.mllm_tokenizer,
            formatted,
            device=agent.mllm_device,
            max_tokens=agent.mllm_inject_max_tokens,
        )

        # Below 16 tokens the model disturbs the pool instead of updating it, so
        # the chunk is dropped and counted rather than injected.
        if result is None:
            agent.mllm_skipped += 1
            return ""

        agent.mllm_injected += 1
        if result["truncated"]:
            print(
                "MemoryLLM: chunk truncated %d -> %d tokens"
                % (result["raw_tokens"], result["tokens"])
            )
        return ""

    memory_construction_time = time.time() - agent.agent_start_time

    # generation_max_length comes from the dataset config and is what every other
    # baseline in this benchmark honours; the agent-config value is only a
    # fallback for datasets that omit the key. Using the fallback unconditionally
    # truncates tasks that need long answers: detective_qa asks for 2000 tokens and
    # was capped at 32, which cut generation off inside the prompt echo and
    # produced a meaningless 0.00.
    max_new_tokens = getattr(agent, "max_tokens", None) or agent.mllm_max_new_tokens

    result = answer(
        agent.mllm_model,
        agent.mllm_tokenizer,
        message,
        device=agent.mllm_device,
        max_new_tokens=max_new_tokens,
    )

    output = agent._create_standard_response(
        result["output"],
        result["input_tokens"],
        result["output_tokens"],
        memory_construction_time,
        result["seconds"],
    )

    # Reset the clock so the next query's construction time excludes this read.
    agent.agent_start_time = time.time()
    return output
