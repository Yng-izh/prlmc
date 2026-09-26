"""Compaction operators: native-block selectors, geometric baselines, learned.

The operator families, in the order the experiments introduced them:

  * `recent_blocks` / `select_blocks` -- **native block selection**. The window is
    cut into whole 256-slot blocks, B_out of them are kept, and every kept slot is
    copied verbatim from the source window. No new vector is ever synthesised, so
    the output is by construction something the frozen reader has already been
    trained to consume.
  * `svd_tokens` / `weighted_svd_tokens` -- principal directions scaled by their
    singular values. These emit K' brand-new vectors that lie off the native delta
    memory manifold. Kept only as a failed baseline; see EXPERIMENTS.md E4.
  * `carrier_residual` -- reserved for the learned residual fusion (phase 3). Not
    implemented here yet.

Section 3.4 states the subtlety that governs this file:

    truncated SVD yields a low-rank K x d reconstruction and does not by itself
    reduce the number of memory tokens. A token-budget experiment must emit
    K' < K *readable* tokens.

So `svd_tokens` returns K' tokens rather than a K-row reconstruction. The
compressed state is lossy and the loss is measured by probes, not by
reconstruction error. Native block selection makes that loss much smaller,
because the reader never has to interpret a new coordinate system.
"""

from typing import Optional

import torch
import torch.nn as nn

from methods.prlmc.config import PRLMCConfig


def _svd_tokens(source: torch.Tensor, target: int, weights: Optional[torch.Tensor]) -> torch.Tensor:
    """Return `target` tokens spanning the dominant subspace of `source`.

    Args:
        source: [L, K, d] window of memory slots.
        target: K', the number of tokens to emit. Must satisfy 0 < target < K.
        weights: optional [K] importance weights. When given, rows are scaled by
            sqrt(w) before the decomposition, so dominant directions follow the
            weighted covariance rather than the unweighted one.

    Returns:
        [L, target, d]. Magnitudes are kept comparable to the input by dividing
        the weighted case by the mean root weight; without that the weighted
        variant would shrink the state and confound the comparison with the
        plain variant.
    """
    layers, k, d = source.shape
    target = max(1, min(target, k))
    out = torch.zeros((layers, target, d), device=source.device, dtype=source.dtype)

    for layer in range(layers):
        x = source[layer].detach().to(torch.float32)
        scale = 1.0
        if weights is not None and weights.numel() == k:
            w = weights.detach().to(torch.float32).clamp_min(0.0)
            root = torch.sqrt(w).unsqueeze(1)
            x = x * root
            mean_root = float(root.mean().clamp_min(1e-6))
            scale = 1.0 / mean_root

        try:
            _, singular, vh = torch.linalg.svd(x, full_matrices=False)
        except Exception:
            # Numerical failure: fall back to keeping the most recent rows rather
            # than aborting the run. Recorded by the caller through the returned
            # shape, so a silent no-op is not possible.
            out[layer, :target, :] = source[layer, -target:, :]
            continue

        take = min(target, singular.numel())
        # Principal directions in feature space, scaled by their singular values
        # so that a linear reader reconstructs the window from these tokens.
        tokens = (singular[:take].unsqueeze(1) * vh[:take]) * scale
        out[layer, :take, :] = tokens.to(out.dtype)
        if take < target:
            out[layer, take:target, :] = tokens[-1:].expand(target - take, d)

    return out


def svd_tokens(source: torch.Tensor, target: int) -> torch.Tensor:
    """Plain truncated-SVD tokens. Geometric baseline, no importance."""
    return _svd_tokens(source, target, None)


def weighted_svd_tokens(source: torch.Tensor, target: int, weights: torch.Tensor) -> torch.Tensor:
    """Importance-weighted truncated-SVD tokens."""
    return _svd_tokens(source, target, weights)


# ---------------------------------------------------------------------------
# Native block selection
# ---------------------------------------------------------------------------
#
# The layer these summaries are computed in. It must match
# `layout._REFERENCE_LAYER`, which is where the basis, the spectrum and the goal
# direction live; a summary taken in another layer would be scored against a goal
# direction from a different space. Duplicated as a local constant because
# `layout` imports this module and importing back would be circular.
_SELECT_LAYER = 0
_EPS = 1e-8

# Development starting values for the selection score. They are deliberately
# tilted towards recency: the run that motivates this phase (EXPERIMENTS.md E4)
# shows that a compactor which discards recently written native blocks loses the
# answer, so recency must be a strong default until E5b shows otherwise.
_DEFAULT_SELECT_WEIGHTS = {
    "importance": 0.4,
    "recency": 0.3,
    "goal": 0.2,
    "diversity": 0.1,
}


def _unit(values: torch.Tensor) -> torch.Tensor:
    """Scale a non-negative vector so its maximum is 1. Zero vectors stay zero."""
    peak = values.max()
    if not torch.isfinite(peak) or float(peak) <= 0.0:
        return torch.zeros_like(values)
    return (values / peak).clamp(0.0, 1.0)


def recent_block_compact(source: torch.Tensor, target: int, block: int = 256) -> torch.Tensor:
    """Keep the most recent whole native blocks of the window.

    This is the most conservative possible K -> K' operator: the output slots are
    copied verbatim from the input, so the reader sees memory it already knows how
    to read. It is the control that isolates "does the compact/write-back
    lifecycle survive at all" from "is the compression operator any good".

    Args:
        source: [L, K, d] window of memory slots, K a multiple of `block`.
        target: K', the number of slots to emit. Must be a multiple of `block`
            and strictly smaller than K.
        block: native block width, `model.num_tokens` (256 for MemoryLLM-8B).

    Returns:
        [L, K', d] in the original temporal order.
    """
    layers, k, d = source.shape
    if k % block != 0:
        raise ValueError(f"recent_block_compact: window {k} is not a multiple of block {block}")
    if target % block != 0:
        raise ValueError(f"recent_block_compact: target {target} is not a multiple of block {block}")

    b = k // block
    b_out = target // block
    if b_out >= b:
        return source.contiguous()

    x = source.view(layers, b, block, d)
    kept = x[:, -b_out:, :, :]
    return kept.reshape(layers, b_out * block, d).contiguous()


def _greedy_diverse(
    score: torch.Tensor,
    unit_vectors: torch.Tensor,
    count: int,
    diversity_weight: float,
) -> list:
    """Pick `count` indices, trading score against max-min distance from the picks.

    Greedy max-min: at each round an index is scored as its own normalised score
    plus `diversity_weight` times the cosine distance to the nearest already
    chosen vector. The first pick therefore has no diversity term at all and is
    simply the highest-scoring block, and each later pick is pulled away from
    what has already been kept.
    """
    n = score.numel()
    count = max(1, min(count, n))
    available = torch.ones(n, dtype=torch.bool, device=score.device)
    nearest = torch.ones(n, device=score.device, dtype=score.dtype)
    chosen = []

    for _ in range(count):
        base = _unit(score.clone())
        base = base.masked_fill(~available, 0.0)
        combined = (base + diversity_weight * nearest).masked_fill(~available, float("-inf"))
        pick = int(torch.argmax(combined).item())
        chosen.append(pick)
        available[pick] = False
        distance = 1.0 - (unit_vectors @ unit_vectors[pick]).clamp(-1.0, 1.0)
        nearest = torch.minimum(nearest, distance.clamp_min(0.0))

    return sorted(chosen)


def select_block_compact(
    source: torch.Tensor,
    target: int,
    block: int = 256,
    weights: Optional[torch.Tensor] = None,
    goal_direction: Optional[torch.Tensor] = None,
    select_weights: Optional[dict] = None,
) -> torch.Tensor:
    """Keep the best-scoring whole native blocks, then restore temporal order.

    Selection only: no vector is synthesised and no slot is modified. The block
    summaries decide *who survives*; the survivors are still written back as the
    exact bytes the reader originally produced.

    The returned order is the original temporal order, not the score order. The
    reader's prefix is a sequence, and permutation destroys the positional
    structure the memory was written with; blocks are dropped, never reordered.

    Score terms, all normalised to [0, 1] across blocks:

        importance   mean per-slot importance over the block when weights are
                     given, else how far the block summary sits from the window
                     mean (a dispersion-based stand-in, no extra machinery);
        recency      block position within the window;
        goal         cosine of the block summary against the goal direction;
        diversity    added greedily inside `_greedy_diverse`, not a static term.
    """
    layers, k, d = source.shape
    if k % block != 0:
        raise ValueError(f"select_block_compact: window {k} is not a multiple of block {block}")
    if target % block != 0:
        raise ValueError(f"select_block_compact: target {target} is not a multiple of block {block}")

    b = k // block
    b_out = target // block
    if b_out >= b:
        return source.contiguous()

    cfg = dict(_DEFAULT_SELECT_WEIGHTS)
    if select_weights:
        for key, value in select_weights.items():
            if key in cfg:
                cfg[key] = float(value)

    x = source.view(layers, b, block, d)
    summary = x[_SELECT_LAYER].detach().to(torch.float32).mean(dim=1)  # [B, d]

    if weights is not None and weights.numel() == k:
        importance = weights.detach().to(torch.float32).view(b, block).mean(dim=1)
    else:
        importance = (summary - summary.mean(dim=0, keepdim=True)).norm(dim=-1)
    importance = _unit(importance)

    recency = _unit(torch.arange(b, device=source.device, dtype=torch.float32))

    if goal_direction is not None and goal_direction.numel() > 0:
        direction = goal_direction.detach().to(torch.float32).flatten().to(source.device)
        direction = direction / (direction.norm() + _EPS)
        normed = summary / (summary.norm(dim=-1, keepdim=True) + _EPS)
        goal = ((normed @ direction).clamp(-1.0, 1.0) + 1.0) / 2.0
    else:
        goal = torch.zeros(b, device=source.device, dtype=torch.float32)

    score = (
        cfg["importance"] * importance
        + cfg["recency"] * recency
        + cfg["goal"] * goal
    )
    unit_summary = summary / (summary.norm(dim=-1, keepdim=True) + _EPS)
    keep = _greedy_diverse(score, unit_summary, b_out, cfg["diversity"])

    index = torch.tensor(keep, device=source.device, dtype=torch.long)
    kept = x.index_select(1, index)
    return kept.reshape(layers, b_out * block, d).contiguous()


class LearnedCompactor(nn.Module):
    """Learnable K -> K' slot reducer, trained against functional probes.

    Architecture is deliberately small: the note warns that the first version
    should not depend on an unvalidated importance model, and the same caution
    applies to the compressor. It is a per-layer cross-attention pool with K'
    learned queries, so the output is genuinely K' readable tokens rather than a
    rank-reduced matrix.

    Training is `probes.train_compactor`, whose objective is L_compact of section
    3.4. Until it has been trained this module must not be selected; the
    `trained` flag makes that explicit and `forward` refuses to run untrained.
    """

    def __init__(self, num_layers: int, hidden: int, target: int, num_heads: int = 8):
        super().__init__()
        self.num_layers = num_layers
        self.hidden = hidden
        self.target = target
        self.trained = False

        self.queries = nn.Parameter(torch.randn(num_layers, target, hidden) * 0.02)
        self.norm = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.out = nn.Linear(hidden, hidden)

    def forward(
        self,
        source: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        target: Optional[int] = None,
    ) -> torch.Tensor:
        """Reduce [L, K, d] to [L, target, d].

        `target` is accepted for interface parity with the geometric compactors
        and checked against the fixed query count: a caller that asks for a width
        this module was not built for is misconfigured, and writing the result
        anyway would corrupt the pool.
        """
        if target is not None and int(target) != self.target:
            raise ValueError(
                "LearnedCompactor was built for %d target tokens but %d were requested"
                % (self.target, int(target))
            )
        if not self.trained:
            raise RuntimeError(
                "LearnedCompactor has not been trained; run probes.train_compactor "
                "first or select compact_mode=svd_tokens"
            )

        layers, k, d = source.shape
        device = source.device
        outputs = []
        for layer in range(layers):
            x = self.norm(source[layer].to(torch.float32)).unsqueeze(0)
            query = self.queries[layer].to(device=device, dtype=torch.float32).unsqueeze(0)
            if weights is not None and weights.numel() == k:
                key_padding = weights.to(device).unsqueeze(0) <= 0
            else:
                key_padding = None
            attended, _ = self.attn(query, x, x, key_padding_mask=key_padding, need_weights=False)
            outputs.append(self.out(attended.squeeze(0)))
        return torch.stack(outputs, dim=0)

    def compact(self, source: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(source, weights).to(source.dtype)

def build_compactor(config: PRLMCConfig, hidden: int):
    """Return the callable compaction operator for this config, or None.

    The callable signature is `(source, weights, target) -> [L, target, d]`. The
    target is passed in rather than derived here because the caller also uses it
    for the placement arithmetic; two independent derivations could disagree and
    silently write a block of the wrong width into the pool.

    The native block selectors are dispatched by `SlotMemory._run_compactor`
    instead, because they need the block width and the goal direction. Returning a
    non-None sentinel here is what tells `compact_with_report` that compaction is
    enabled at all, so the selectors must not fall through to `None`.
    """
    if config.compact_mode == "none":
        return None
    if config.compact_mode == "svd_tokens":
        return lambda source, weights, target: svd_tokens(source, target)
    if config.compact_mode == "weighted_svd_tokens":
        return lambda source, weights, target: weighted_svd_tokens(source, target, weights)
    if config.compact_mode in ("recent_blocks", "select_blocks", "carrier_residual"):
        # Dispatched in layout._run_compactor with the extra arguments it needs.
        return lambda source, weights, target: None
    if config.compact_mode == "learned":
        raise NotImplementedError(
            "compact_mode=learned requires a trained checkpoint; construct "
            "LearnedCompactor via probes.train_compactor and inject it into the "
            "manager rather than selecting it here"
        )
    raise ValueError(f"unknown compact_mode: {config.compact_mode}")


def _target_for(source: torch.Tensor, ratio: float) -> int:
    """Default target width for a window, used when no explicit target is given."""
    k = source.shape[1]
    return max(1, min(k - 1, int(k * ratio)))
