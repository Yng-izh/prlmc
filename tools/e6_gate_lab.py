"""E6 gate lab: five single-window branches, judged by lost-probe recovery.

The one question: is the information native selection throws away still *readable*
after SVD-style compaction, once the reader is given a per-layer memory-read
adapter? Branch table (advisor spec, frozen):

    frozen_full    uncompacted pool, no operator          -- upper bound
    native_select  select_blocks candidate, frozen reader -- lower bound / before
    svd_frozen     svd_tokens candidate, frozen reader    -- failure control (E4)
    svd_adapter    same SVD candidate + trained reader-A  -- is SVD info readable?
    overkill       CarrierResidualCompactor rank 128/256, gamma_max 0.2,
                   gold-dominant stage B, 19->9           -- deliberate overfit:
                   if ~520K params cannot recover the 3 lost probes, the formula
                   is dead regardless of what the adapter does.

Gate: recovered == lost (>= 3) on the pinned window. Everything else (EM,
capacity, triggers, generalisation) is out of scope until the gate passes.

Read-path mechanism (the part the first draft got wrong): the model reads
`model.memory` directly inside attention, so an adapter that returns a new
tensor from a standalone call has no effect. The adapter is therefore applied
*before* the forward and handed to the reader through `differentiable_memory`:

    adapted = adapter(pool)          # [L, T, d] fp16, per-layer residual
    with differentiable_memory(model, adapted):
        model(input_ids=...)         # attention reads adapted values

Training keeps the compacted pool fixed; the only trainable path is the adapter
(zero-init `up` => bit-identical pass-through at init), so no `_pool_in_place`
or candidate-leaf plumbing is needed here -- the compactor branches reuse
`train_carrier_compactor` unchanged.

Run on the host (one GPU per branch group):
    /share/latent/mllm_env/bin/python /tmp/e6_gate_lab.py \
        --branches svd_frozen,svd_adapter --out /tmp/e6_gate_g1.json
"""

import argparse
import json
import os
import subprocess
import sys

REPO = "/share/latent/PRLM-C"


# --- GPU pick, before torch is imported so CUDA_VISIBLE_DEVICES takes effect ---

def pick_gpu(need_mb=None, forced=None):
    if forced is not None and str(forced).strip() != "":
        return int(forced)
    if need_mb is None:
        need_mb = int(os.environ.get("E6A_NEED_MB", "21000"))
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    free = []
    for line in out.strip().splitlines():
        index, used = [int(v) for v in line.split(",")]
        free.append((32768 - used, index))
    free.sort(reverse=True)
    if free[0][0] < need_mb:
        raise SystemExit("no GPU with %d MB free; best is gpu%d with %d MB" % (need_mb, free[0][1], free[0][0]))
    print("[gate] gpu%d, %d MB free" % (free[0][1], free[0][0]), flush=True)
    return free[0][1]


if __name__ == "__main__" and not os.environ.get("E6A_GPU_SET"):
    _gpu = pick_gpu(forced=os.environ.get("E6A_GPU_ID"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu)
    os.environ["E6A_GPU_SET"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.utils.checkpoint  # noqa: E402,F401  (MemoryReadAdapter calls it)


def _mem_trace(label: str):
    """Phase-boundary memory trace (PRLMC_MEM_TRACE=1), mirroring probes._mem."""
    if not os.environ.get("PRLMC_MEM_TRACE") or not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 2**30
    reserved = torch.cuda.memory_reserved() / 2**30
    print("[mem] %-26s alloc=%.2fGB reserved=%.2fGB" % (label, alloc, reserved), flush=True)

sys.path.insert(0, "/tmp")
import e6a_overfit as e6a  # noqa: E402  (brings REPO onto sys.path)

from methods.prlmc.consolidate import CarrierResidualCompactor, select_block_compact, svd_tokens  # noqa: E402
from methods.prlmc.probes import (  # noqa: E402
    CompactObjective,
    _checkpoint_func,
    _compacted_mask,
    _compacted_pool,
    _probe_batches,
    _forward_with_hidden,
    active_mask_override,
    differentiable_memory,
    gold_loss_from_logits,
    reader_hidden_loss,
    teacher_kd_loss,
    train_carrier_compactor,
)

BRANCHES = ("frozen_full", "native_select", "svd_frozen", "svd_adapter", "overkill")

DEVICE = "cuda:0"
WINDOW_BLOCKS = 19          # 4864 slots, the E6a window
TARGET_BLOCKS = 9           # 2304 slots, compact ratio 0.5


# --- Reader-A: per-layer memory-read adapter ---------------------------------

class MemoryReadAdapter(nn.Module):
    """M~_l = M_l + alpha_l * W_up^(l) sigma(W_down^(l) M_l), per layer.

    `up` is zero-initialised, so the adapter is the identity at init and the
    first scored step reproduces the frozen-reader number bitwise. Parameters
    stay fp32 and are cast to the pool dtype per use (same discipline as the
    compactor), so AdamW updates keep fp32 precision while every activation
    stays fp16.

    Each layer's transform is wrapped in per-layer checkpointing: without it the
    per-layer [T, d] intermediates of all 32 layers would sit in the graph at
    once (~3.4 GB on top of the adapted pool); with it only the adapted pool
    itself and the checkpoint inputs are live. The transform is two matmuls --
    recomputing it on backward is noise next to the reader's own recompute.
    """

    def __init__(self, num_layers: int, hidden: int, rank: int = 64):
        super().__init__()
        self.num_layers = int(num_layers)
        self.hidden = int(hidden)
        self.rank = int(rank)
        self.down = nn.ModuleList(
            [nn.Linear(hidden, rank, bias=False) for _ in range(self.num_layers)]
        )
        self.up = nn.ModuleList(
            [nn.Linear(rank, hidden, bias=False) for _ in range(self.num_layers)]
        )
        for up in self.up:
            nn.init.zeros_(up.weight)
        # Per-layer learnable amplitude. 1.0 with a zero `up` still yields the
        # identity, so init behaviour is governed by `up` alone.
        self.alpha = nn.Parameter(torch.ones(self.num_layers))

    def _layer(self, index: int, x: torch.Tensor) -> torch.Tensor:
        down = self.down[index]
        up = self.up[index]
        h = F.gelu(F.linear(x, down.weight.to(x.dtype)))
        delta = F.linear(h, up.weight.to(x.dtype))
        return x + self.alpha[index].to(x.dtype) * delta

    def forward(self, pool: torch.Tensor, active_index: torch.Tensor = None) -> torch.Tensor:
        # Sparse path (active_index given): run the residual only on the rows
        # that survive `_compacted_mask` (~2560 of 12800). The per-layer
        # checkpoint input is the *CPU* slice, so nothing full-width is ever
        # saved on the GPU for recompute; returns [L, A, d] (~176 MiB).
        device = self.down[0].weight.device
        if active_index is not None:
            ai = active_index.to(pool.device)
            rows = []
            for index in range(self.num_layers):
                rows.append(
                    torch.utils.checkpoint.checkpoint(
                        lambda x, index=index: self._layer(
                            index, x.to(device, non_blocking=True)
                        ),
                        pool[index].index_select(0, ai),
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                )
            return torch.stack(rows, dim=0)
        # Full path: pool may live on pinned CPU; one layer's slice moves to the
        # GPU inside its checkpoint call, so the GPU only holds one working
        # layer. Kept for inspection; the gate trains through the sparse path.
        results = []
        for index in range(self.num_layers):
            results.append(
                torch.utils.checkpoint.checkpoint(
                    lambda x, index=index: self._layer(index, x.to(device, non_blocking=True)),
                    pool[index],
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            )
        return torch.stack(results, dim=0)


class _MemRowsInject(torch.autograd.Function):
    """Write adapter rows into the model's own memory storage, differentiably.

    Mirror of `_InjectSlice` (probes.py:799), host-verified on the overkill
    branch: an in-place write into a tensor that does not require grad silently
    drops the RHS graph, so the write lives inside a custom `Function` whose
    forward mutates under no_grad and whose backward routes the gradient of
    exactly the written rows back to the adapter graph. The returned tensor
    aliases the parameter storage itself -- this is the overkill `_pool_in_place`
    trick, and it is why no second full-width pool (the g2j canvas, 3.12 GiB,
    plus its 3.12 GiB backward output-grad buffer) is ever allocated.

    The forward receives `board = model.memory.data.view(-1, dim)`; the caller
    installs the returned alias through `differentiable_memory`, so the reader's
    `cat_memory_and_hiddens` reads exactly these rows and the gradient flows
    back into the adapter.

    Inactive slots keep whatever the restored pool held. That is numerically
    safe: `memory_active_mask` gives them an additive `finfo.min` attention bias
    (modeling_memoryllm.py:1920+), so their attention weight is exp(min) == 0
    and their *values* cannot reach the output. Only the ~2560 active rows carry
    information and gradient.
    """

    @staticmethod
    def forward(ctx, board, rows, global_index):
        ctx.rows_shape = rows.shape
        layers, total = ctx.rows_shape[0], board.shape[0] // ctx.rows_shape[0]
        ctx.global_index = global_index
        board.index_copy_(0, global_index, rows.reshape(-1, board.shape[-1]).to(board.dtype))
        return board.view(layers, total, board.shape[-1])

    @staticmethod
    def backward(ctx, grad_output):
        grad_rows = grad_output.reshape(-1, grad_output.shape[-1]).index_select(
            0, ctx.global_index
        ).reshape(ctx.rows_shape)
        return None, grad_rows, None


def _mem_rows_inject(board: torch.Tensor, rows: torch.Tensor, global_index: torch.Tensor):
    """One fresh inject node per probe, exactly like `_PoolInjection.reinject()`.

    A shared node would force `retain_graph=True` across the probe loop; a fresh
    node lets each probe's reader graph die with its own backward while the
    adapter accumulates the summed gradient in its parameters.
    """
    return _MemRowsInject.apply(board, rows, global_index)


def _enable_reader_checkpointing(manager):
    """The fork's per-layer checkpoint branch, exactly as `train_carrier_compactor`
    sets it up: flag and callable must land on the *inner* LlamaModel, and
    `train()` must be on for the branch to fire (no dropout in this fork, so
    train mode adds no stochasticity)."""
    manager.model.gradient_checkpointing_enable()
    inner = getattr(manager.model, "model", None)
    if inner is not None:
        inner.gradient_checkpointing = True
        inner._gradient_checkpointing_func = _checkpoint_func()
    manager.model.train()


def _disable_reader_checkpointing(manager):
    manager.model.gradient_checkpointing_disable()
    inner = getattr(manager.model, "model", None)
    if inner is not None:
        inner.gradient_checkpointing = False
    manager.model.eval()


def train_read_adapter(
    manager,
    adapter,
    probes,
    pool,
    mask,
    steps: int,
    lr: float,
    kd_weight: float = 0.5,
    hidden_weight: float = 0.1,
    reader_layers=(8, 16, 24, 31),
    max_probe_tokens: int = 256,
):
    """Adapter-only training against a FIXED compacted pool.

    The compacted pool never gets a standalone canvas: injection writes the
    adapter rows straight into the model's own memory storage (`_pool_in_place`
    style), and every forward installs the injected alias through
    `differentiable_memory`. The mask hides the freed slots, and the backbone stays frozen, so
    the only gradient path is pool -> adapter -> loss. Per-probe graphs are
    rebuilt and freed one at a time; adapter grads accumulate across the probe
    loop and the optimiser steps once per pass over the probes. AMP is not
    used: the model is loaded fp16 and a bf16 autocast region made the g2i
    attempt OOM *worse* (autocast caches fp32->bf16 weight copies on top of
    the fp16 originals).

    Teacher logits/hidden come from the intact (uncompacted) pool, matching
    stage-B semantics: gold primary, KD auxiliary, hidden-state tether last.
    """
    device = manager.model.memory.device
    reader_layers = list(reader_layers)
    batches = _probe_batches(manager.tokenizer, probes, str(device), max_length=max_probe_tokens)
    if not batches:
        raise ValueError("every probe was truncated away; raise max_probe_tokens")

    # Teacher cache: one pass over the intact pool, detached for good. The
    # intact pool is `manager._memory` moved onto the reader's device, the same
    # source `train_carrier_compactor` uses for its own teacher cache.
    with torch.no_grad(), differentiable_memory(manager.model, manager._memory.to(device).detach()):
        teacher_logits = []
        teacher_hidden = []
        for input_ids, n_prefix, _ in batches:
            logits, hidden = _forward_with_hidden(
                manager.model, input_ids, reader_layers, enable_grad=False
            )
            # KD targets only need the answer span; parking the full cache on CPU
            # keeps ~0.6 GiB off the GPU during training.
            teacher_logits.append(logits[:, n_prefix - 1 : -1, :].detach().cpu())
            teacher_hidden.append([h.detach().cpu() for h in hidden])
    torch.cuda.empty_cache()

    trainable = [p for p in adapter.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=lr)

    # Stage the SVD pool on pinned host memory: the adapter moves one layer's
    # slice to the GPU per checkpoint call, so the decoder's saved recompute
    # inputs stay on the host too and the GPU never holds a full-width copy.
    pool = pool.to("cpu").detach()
    if not pool.is_pinned():
        pool = pool.pin_memory()
    # Slots the reader may attend to (k_anchor + compacted tail); only these
    # carry adapter gradients. Inactive slots keep the restored pool values --
    # their `finfo.min` attention bias makes their values irrelevant (see
    # `_MemRowsInject`), so the source pool never has to be H2D-copied at all.
    active_index = mask.to(device=device, dtype=torch.long).nonzero(as_tuple=False).view(-1)
    layers, total, dim = pool.shape
    # Inject into the model's own memory storage: the overkill `_pool_in_place`
    # mechanism. No second [L, T, d] buffer exists, so the g2j bookkeeping
    # (3.12 GiB canvas + 3.12 GiB transient backward buffer) is gone by
    # construction. Rows not written keep their restored values; the `finfo.min`
    # bias keeps those slots invisible to the reader.
    board = manager.model.memory.data.view(-1, dim)
    if board.shape[0] != layers * total:
        raise RuntimeError(
            "model memory %r does not match pool %r" % (tuple(board.shape), (layers, total, dim))
        )
    # Rows this branch is about to overwrite, parked on CPU; `finally` restores
    # them under no_grad so later branches (score pass, overkill) see an intact
    # pool.
    global_index = (
        active_index.view(1, -1)
        + torch.arange(layers, device=device, dtype=torch.long).view(-1, 1) * total
    ).reshape(-1)
    saved_rows = board[global_index].detach().to("cpu", copy=True)
    checkpointing_was = bool(getattr(manager.model, "is_gradient_checkpointing", False))
    training_was = bool(manager.model.training)
    _enable_reader_checkpointing(manager)
    # Freeze the backbone and the reader. Without this the differentiable forward
    # retains an 8B-parameter gradient graph on every step (~32 GiB fp32) and the
    # first two attempts OOMed; same discipline as `train_carrier_compactor`
    # (probes.py:1170). Only the adapter's parameters may require grad.
    frozen = []
    for parameter in manager.model.parameters():
        if parameter.requires_grad:
            parameter.requires_grad_(False)
            frozen.append(parameter)
    # The compacted pool is CPU-pinned and re-materialised per probe by the
    # adapter; the model's own storage is the injection board, so nothing needs
    # parking or stashing any more.
    _mem_trace("train: setup done")
    history = []
    try:
        for step in range(steps):
            optimiser.zero_grad(set_to_none=True)
            step_losses = []
            # A fresh adapted graph per probe: gradients accumulate into the
            # adapter parameters and each graph dies with its own backward, so
            # nothing ever needs retain_graph.
            for index, (input_ids, n_prefix, _) in enumerate(batches):
                rows = adapter(pool, active_index=active_index)
                # Write the fresh rows into the model's own memory storage: one
                # fresh autograd node per probe, zero extra full-width
                # allocation, no H2D copy of the source pool. `differentiable_memory`
                # then hands the aliased board to the reader.
                adapted = _mem_rows_inject(board, rows, global_index)
                with active_mask_override(manager.model, mask), differentiable_memory(
                    manager.model, adapted
                ):
                    student_logits, student_hidden = _forward_with_hidden(
                        manager.model, input_ids, reader_layers,
                        enable_grad=True, pool=None,
                    )
                student_span = [student_logits[:, n_prefix - 1 : -1, :]]
                loss = gold_loss_from_logits([batches[index]], student_span, device)
                # Teacher caches live on CPU; the span/hidden slices are tiny, so
                # the transfer back is noise next to the forward pass.
                loss = loss + kd_weight * teacher_kd_loss(
                    [teacher_logits[index].to(device)], [batches[index]], student_span
                ).to(device)
                if hidden_weight > 0.0:
                    loss = loss + hidden_weight * reader_hidden_loss(
                        [h.to(device) for h in teacher_hidden[index]],
                        student_hidden, layers=reader_layers,
                    ).to(device)
                _mem_trace("train: s%d p%d pre-backward" % (step + 1, index + 1))
                loss.backward()
                step_losses.append(float(loss.detach()))
                del adapted, student_logits, student_hidden, loss, rows
                if index == 0 and step == 0:
                    _mem_trace("train: first probe done")
            optimiser.step()
            mean_loss = sum(step_losses) / len(step_losses)
            history.append(mean_loss)
            print("[gate] adapter step %d/%d mean loss %.4f" % (step + 1, steps, mean_loss), flush=True)
    finally:
        # Restore the overwritten rows first: the board is the live pool
        # storage, and every later branch reads it.
        with torch.no_grad():
            board[global_index] = saved_rows.to(device=board.device, dtype=board.dtype)
        del saved_rows
        for parameter in frozen:
            parameter.requires_grad_(True)
        _disable_reader_checkpointing(manager)
        if checkpointing_was:
            _enable_reader_checkpointing(manager)
        if training_was:
            manager.model.train()
        torch.cuda.empty_cache()

    return {
        "steps": steps,
        "lr": lr,
        "kd_weight": kd_weight,
        "hidden_weight": hidden_weight,
        "n_batches": len(batches),
        "loss": history,
        "final_loss": history[-1] if history else None,
    }


def score_with_adapter(manager, adapter, probes, pool, mask, max_probe_tokens):
    """Frozen scoring pass through the trained adapter.

    Same `_pool_in_place` discipline as training: the adapter's output is
    written straight into the model's memory storage, rows are saved/restored
    around the call, and the pool never gets a second full-width copy. The pool
    may live on pinned CPU (evicted for the training window); the sparse adapter
    path moves one layer's slice at a time, so no full-width H2D copy exists.
    Index math runs on the board's device regardless of where the pool sits.
    """
    index_device = manager.model.memory.device
    layers, total, dim = pool.shape
    board = manager.model.memory.data.view(-1, dim)
    active_index = mask.to(device=index_device, dtype=torch.long).nonzero(as_tuple=False).view(-1)
    global_index = (
        active_index.view(1, -1)
        + torch.arange(layers, device=index_device, dtype=torch.long).view(-1, 1) * total
    ).reshape(-1)
    with torch.no_grad():
        # No `pool.to(device)` here: the sparse adapter path moves one layer's
        # slice at a time, so no full-width H2D copy ever exists.
        rows = adapter(pool, active_index=active_index)
        saved_rows = board[global_index].detach().to("cpu", copy=True)
        adapted = _mem_rows_inject(board, rows, global_index)
    try:
        return e6a.score_probes(manager, probes, adapted, max_probe_tokens, mask=mask)
    finally:
        with torch.no_grad():
            board[global_index] = saved_rows.to(device=board.device, dtype=board.dtype)
        del adapted, rows, saved_rows
        torch.cuda.empty_cache()


# --- candidate builders and installation -------------------------------------

def native_candidate(manager, source, target_slots, weights):
    """The select_blocks candidate on the exact inputs the baseline uses."""
    return select_block_compact(
        source, target_slots, block=manager.block,
        weights=weights, goal_direction=manager.goal_direction,
        select_weights=manager.select_weights,
    )


def install_candidate(manager, candidate, window_slots, target_slots, device):
    """Rearrange the full-width pool around `candidate` and sync the mask.

    `_compacted_pool` puts the carriers flush against the trailing end with the
    protected remainder after them; `_compacted_mask` hides the freed slots so
    the reader cannot attend to both the carriers and the blocks they replaced.
    """
    pool = _compacted_pool(
        manager, device, candidate, window_used=window_slots, target=target_slots
    )
    mask = _compacted_mask(manager, manager.used - window_slots + target_slots, device)
    return pool, mask


def _gate_pass(recovery, min_lost):
    """The gate: every probe the pool was required for and selection lost is back."""
    return bool(recovery["n_lost"] >= min_lost and recovery["n_recovered"] == recovery["n_lost"])


# --- main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branches", default=",".join(BRANCHES),
                        help="comma-separated subset of: %s" % ", ".join(BRANCHES))
    parser.add_argument("--out", default="/tmp/e6_gate_report.json")
    parser.add_argument("--window", default="3,17890", help="pinned window as ROW,START")
    parser.add_argument("--probes", type=int, default=20)
    parser.add_argument("--min-probes", type=int, default=4)
    parser.add_argument("--min-lost", type=int, default=3)
    parser.add_argument("--max-probe-tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=8, help="svd_adapter training steps")
    parser.add_argument("--overkill-steps", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--adapter-rank", type=int, default=64)
    parser.add_argument("--rank", type=int, default=128, help="overkill carrier rank")
    parser.add_argument("--gamma-max", type=float, default=0.2)
    parser.add_argument("--kd-weight", type=float, default=0.5)
    parser.add_argument("--hidden-weight", type=float, default=0.1)
    args = parser.parse_args()

    branches = [b.strip() for b in args.branches.split(",") if b.strip()]
    bad = [b for b in branches if b not in BRANCHES]
    if bad:
        raise SystemExit("unknown branches: %s (choose from %s)" % (bad, ", ".join(BRANCHES)))

    # Recovery needs the frozen sibling as the "before" measurement, so pull it
    # in whenever a trained branch is requested.
    needed = set(branches)
    if "svd_adapter" in needed:
        needed.add("svd_frozen")
    if "overkill" in needed:
        needed.add("native_select")

    import yaml
    from transformers import AutoTokenizer

    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory

    raw = yaml.safe_load(open(e6a.CONFIG))
    config = dict(raw["prlmc"])
    config["device"] = DEVICE
    config["dtype"] = "float16"
    config["stats_dir"] = "./outputs/prlmc-e6-gate"
    config["carrier_gamma_max"] = args.gamma_max
    cfg = PRLMCConfig.from_dict(config)

    model = MemoryLLM.from_pretrained(
        e6a.MODEL, torch_dtype=torch.float16, attn_implementation="sdpa"
    ).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(e6a.MODEL)
    manager = ProtectedResidualMemory(model, tokenizer, cfg)

    window_slots = WINDOW_BLOCKS * manager.block
    target_slots = TARGET_BLOCKS * manager.block
    row, start = [int(v) for v in args.window.split(",")]
    lines = e6a.read_fact_lines(row)
    covered = e6a.build_window(manager, lines, window_slots, start_line=start)
    indices = e6a.probe_indices(covered, window_slots)
    line_to_block = {}
    for block_index, (lo, hi, _before, _width) in enumerate(covered):
        for line in range(lo, hi):
            line_to_block[line] = block_index
    begin = manager.total - manager.used
    source = manager._memory[:, begin : begin + window_slots, :].to(device=DEVICE)
    weights = manager._window_importance(begin, window_slots)
    dropped = e6a.dropped_block_indices(manager, source, weights)
    probes = e6a.probes_from_lines(
        lines, indices, args.probes,
        dropped_blocks=set(dropped), line_to_block=line_to_block,
    )
    print("[gate] pinned window row %d start %d, dropped blocks: %s" % (row, start, dropped), flush=True)
    for probe in probes:
        print("        %-60s -> %s" % (probe.question, probe.answers[0]), flush=True)
    if len(probes) < args.min_probes:
        raise SystemExit(
            "only %d probes in dropped blocks; need >= %d" % (len(probes), args.min_probes)
        )

    device = manager.model.memory.device
    report = {
        "window": {"row": row, "start": start},
        "window_slots": window_slots,
        "target_slots": target_slots,
        "used_slots": manager.used,
        "dropped_blocks": dropped,
        "n_probes": len(probes),
        "gamma_max": float(cfg.carrier_gamma_max),
        "probes": [{"q": p.question, "a": p.answers[0]} for p in probes],
        "branches_requested": branches,
    }

    # Shared baselines: no-memory floor and the intact-pool ceiling. The ceiling
    # doubles as the frozen_full branch -- same pool, same forward set -- so it
    # is measured once and reported under both names.
    report["no_memory"] = e6a.score_no_memory(manager, probes, args.max_probe_tokens)
    full_pool = manager._memory.to(device).detach()
    report["frozen_full"] = e6a.score_probes(manager, probes, full_pool, args.max_probe_tokens)
    del full_pool
    torch.cuda.empty_cache()
    report["uncompacted"] = report["frozen_full"]
    for label in ("no_memory", "frozen_full"):
        entry = report[label]
        print("[gate] %-12s acc %.3f (%d/%d)" % (
            label, entry["accuracy"],
            sum(bool(v) for v in entry["correct"].values()), entry["n"]), flush=True)

    def run_frozen(branch, candidate):
        pool, mask = install_candidate(manager, candidate, window_slots, target_slots, device)
        try:
            entry = e6a.score_probes(manager, probes, pool, args.max_probe_tokens, mask=mask)
        finally:
            del pool, mask, candidate
            torch.cuda.empty_cache()
        print("[gate] %-12s acc %.3f (%d/%d)" % (
            branch, entry["accuracy"],
            sum(bool(v) for v in entry["correct"].values()), entry["n"]), flush=True)
        report[branch] = entry

    if "native_select" in needed:
        run_frozen("native_select", native_candidate(manager, source, target_slots, weights))

    if "svd_frozen" in needed:
        run_frozen("svd_frozen", svd_tokens(source, target_slots))

    if "svd_adapter" in needed:
        candidate = svd_tokens(source, target_slots)
        pool, mask = install_candidate(manager, candidate, window_slots, target_slots, device)
        del candidate
        torch.cuda.empty_cache()
        adapter = MemoryReadAdapter(
            manager.num_layers, manager.hidden, rank=args.adapter_rank
        ).to(device=DEVICE, dtype=torch.float32)
        # Pass-through proof: with `up` zero-init the adapted pool must equal the
        # frozen pool bitwise, so any later change is the adapter's alone.
        with torch.no_grad():
            adapted0 = adapter(pool, active_index=None)
        identity = bool(torch.equal(adapted0, pool))
        print("[gate] adapter == identity at init: %s" % identity, flush=True)
        del adapted0
        torch.cuda.empty_cache()
        # Evict the full-width GPU pool before training: the g2k [mem] trace
        # showed pre-backward 25.60 GB with the pool resident, and the backward
        # then needs one more full-width fp16 block (3.12 GiB) -> 31.7+ OOM.
        # Training and scoring re-materialise rows per probe from the CPU-pinned
        # copy instead; the board (model.memory storage) already holds the
        # installed values for restoration.
        pool = pool.to("cpu").pin_memory()
        torch.cuda.empty_cache()
        train = train_read_adapter(
            manager, adapter, probes, pool, mask,
            steps=args.steps, lr=args.lr,
            kd_weight=args.kd_weight, hidden_weight=args.hidden_weight,
            max_probe_tokens=args.max_probe_tokens,
        )
        after = score_with_adapter(manager, adapter, probes, pool, mask, args.max_probe_tokens)
        print("[gate] %-12s acc %.3f (%d/%d)" % (
            "svd_adapter", after["accuracy"],
            sum(bool(v) for v in after["correct"].values()), after["n"]), flush=True)
        report["svd_adapter"] = {"identity_at_init": identity, "train": train, "after": after}
        del pool, mask, adapter
        torch.cuda.empty_cache()

    if "overkill" in needed:
        compactor = CarrierResidualCompactor(
            num_layers=manager.num_layers,
            hidden=manager.hidden,
            block=manager.block,
            rank=args.rank,
            gamma_max=args.gamma_max,
        ).to(device=DEVICE, dtype=torch.float32)
        # Deliberate overfit: gold-dominant objective, the native_select number is
        # the "before" (the compactor is bit-identical to select_blocks at init,
        # verified in E6a; zero-init `up` makes that independent of rank/gamma).
        trained = train_carrier_compactor(
            manager, probes,
            objective=CompactObjective(gold=1.0, kd=0.05, reader_hidden=0.0, carrier=0.05),
            steps=args.overkill_steps, lr=args.lr,
            window_blocks=WINDOW_BLOCKS, target_blocks=TARGET_BLOCKS,
            max_probe_tokens=args.max_probe_tokens, compactor=compactor,
            use_checkpoint=True,
        )
        pool, mask = e6a.candidate_pool(manager, compactor, device)
        try:
            after = e6a.score_probes(manager, probes, pool, args.max_probe_tokens, mask=mask)
        finally:
            del pool, mask
            torch.cuda.empty_cache()
        print("[gate] %-12s acc %.3f (%d/%d)" % (
            "overkill", after["accuracy"],
            sum(bool(v) for v in after["correct"].values()), after["n"]), flush=True)
        report["overkill"] = {"train": trained, "after": after}
        del compactor
        torch.cuda.empty_cache()

    # Lost-probe recovery, only for branches this process actually ran.
    # native_select/svd_frozen have after == before (frozen readers recover
    # nothing); svd_adapter is judged against svd_frozen, overkill against
    # native_select.
    recoveries = {}
    if "native_select" in report:
        recoveries["native_select"] = e6a.recovery_report(
            report["frozen_full"], report["native_select"], report["native_select"],
            no_memory=report["no_memory"],
        )
    if "svd_frozen" in report:
        recoveries["svd_frozen"] = e6a.recovery_report(
            report["frozen_full"], report["svd_frozen"], report["svd_frozen"],
            no_memory=report["no_memory"],
        )
    if "svd_adapter" in report and "svd_frozen" in report:
        recoveries["svd_adapter"] = e6a.recovery_report(
            report["frozen_full"], report["svd_frozen"], report["svd_adapter"]["after"],
            no_memory=report["no_memory"],
        )
    if "overkill" in report and "native_select" in report:
        recoveries["overkill"] = e6a.recovery_report(
            report["frozen_full"], report["native_select"], report["overkill"]["after"],
            no_memory=report["no_memory"],
        )
    for name, rec in recoveries.items():
        print("[gate] %-12s recovery %d/%d" % (name, rec["n_recovered"], rec["n_lost"]), flush=True)
    report["recovery"] = recoveries

    gate = {"min_lost": args.min_lost, "pass": False}
    if "svd_adapter" in recoveries:
        passed = _gate_pass(recoveries["svd_adapter"], args.min_lost)
        gate["svd_adapter"] = {
            "n_lost": recoveries["svd_adapter"]["n_lost"],
            "n_recovered": recoveries["svd_adapter"]["n_recovered"],
            "pass": passed,
        }
        gate["pass"] = gate["pass"] or passed
        print("[gate] VERDICT svd_adapter: %s (%d/%d)%s" % (
            "PASS" if passed else "FAIL",
            recoveries["svd_adapter"]["n_recovered"], recoveries["svd_adapter"]["n_lost"],
            " -- SVD information is READABLE; E4 was a reader-compatibility failure, "
            "not information destruction" if passed else "",
        ), flush=True)
    if "overkill" in recoveries:
        passed = _gate_pass(recoveries["overkill"], args.min_lost)
        gate["overkill"] = {
            "n_lost": recoveries["overkill"]["n_lost"],
            "n_recovered": recoveries["overkill"]["n_recovered"],
            "pass": passed,
        }
        gate["pass"] = gate["pass"] or passed
        print("[gate] VERDICT overkill: %s (%d/%d)" % (
            "PASS" if passed else "FAIL",
            recoveries["overkill"]["n_recovered"], recoveries["overkill"]["n_lost"],
        ), flush=True)
    report["gate"] = gate

    json.dump(report, open(args.out, "w"), indent=2, default=str)
    print("[gate] written to %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
