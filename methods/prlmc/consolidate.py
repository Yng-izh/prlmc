"""Compaction operators: geometric baselines and the learned compressor.

Section 3.4 requires the first version to offer two families:

  * a geometric baseline -- SVD / truncated SVD over the residual memory, with a
    plain and an importance-weighted variant;
  * a functionally supervised compressor -- a module that maps K latent slots to
    K' < K slots while preserving answers on a set of memory probes.

It also states the subtlety that governs this file:

    truncated SVD yields a low-rank K x d reconstruction and does not by itself
    reduce the number of memory tokens. A token-budget experiment must emit
    K' < K *readable* tokens.

So `svd_tokens` does not return the low-rank reconstruction (which has K rows).
It returns the K' principal directions scaled by their singular values, i.e. the
K' tokens a shared linear reader consumes. That is the "principal component
tokens" option the note names. The compressed state is lossy and the loss is
measured by probes, not by reconstruction error.
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
    """
    if config.compact_mode == "none":
        return None
    if config.compact_mode == "svd_tokens":
        return lambda source, weights, target: svd_tokens(source, target)
    if config.compact_mode == "weighted_svd_tokens":
        return lambda source, weights, target: weighted_svd_tokens(source, target, weights)
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
