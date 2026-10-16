"""Sensitivity-directed consolidation operators (S1-SDC section 3).

Three components, each a pure tensor function so the driver owns every
interaction with the frozen reader:

1. ``block_salience``       s_j = sum_p ||dL_p/dB_j||_F^2 from one probe's
                            window gradient; accumulate over probes.
2. ``salience_select``      top-k by salience, temporal order preserved.
3. ``merge_into_survivors`` B' = B + (I - U U^T) f(A) per layer per survivor,
                            U = span of the probe gradients at B.
4. ``RollbackGate``         probe-flip detector; degrades to pure selection.

The first-order guarantee is the point of component 3: because the merge
displacement lives in the orthogonal complement of the span of the probe
gradients at B, <dL_p/dB, dB> = 0 for every probe p, so the merge's effect on
answering behaviour is zero to first order. E6-gate's no-harm property moves
from an after-the-fact check to a constructive constraint.

``nullspace_project`` implements the complement without materialising the
basis U. The caller supplies the raw gradient rows G [P, N] (N = block * d per
layer); the projection runs through the P x P Gram matrix, so memory is O(P*N)
for the rows themselves -- the same tensors the backward already produced.

Run-time convention: everything here is dtype-agnostic fp32 math on whatever
device the inputs live on. The driver keeps the pool fp16 and casts at the
boundary, mirroring how ``CarrierResidualCompactor`` handles its fp32 island.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "block_salience",
    "block_taylor_salience",
    "normalize_probe_scores",
    "coverage_select",
    "salience_select",
    "nullspace_project",
    "IncrementalBasis",
    "merge_into_survivors",
    "RollbackGate",
]


def block_salience(grad: torch.Tensor, block: int = 256) -> torch.Tensor:
    """Per-block functional saliency of ONE probe's window gradient.

    ``grad`` is dL_p/d(window) with shape [L, K, d]; the result is
    [b] with b = K // block, entry j = ||grad[:, j*block:(j+1)*block, :]||_F^2
    summed over layers and slots -- the s_j of S1-SDC section 3.1 for one
    probe. Callers accumulate over probes in fp32.

    Kept separate from the accumulation on purpose: the same helper serves the
    per-probe diagnostic path (which block each probe lights up) and the
    selection path (the sum).
    """
    layers, slots, d = grad.shape
    if slots % block != 0:
        raise ValueError("window %d is not a multiple of block %d" % (slots, block))
    b = slots // block
    g = grad.detach().to(torch.float32).reshape(layers, b, block, d)
    return g.pow(2).sum(dim=(0, 2, 3))


def salience_select(salience: torch.Tensor, target_blocks: int) -> List[int]:
    """Top-`target_blocks` blocks by salience, returned in temporal order.

    Ties break toward the *newer* block (higher index): under the
    version-semantics of Conflict_Resolution the newer duplicate is the one the
    gold answer names, and a tie means the functional signal cannot separate
    them -- recency is the correct, and only, remaining tiebreak.

    Temporal order on output is not cosmetic: the reader's prefix is a
    sequence and permutation destroys positional structure (the same invariant
    ``select_block_compact`` maintains).
    """
    b = int(salience.numel())
    if not 0 < target_blocks < b:
        raise ValueError("target %d must be in (0, %d)" % (target_blocks, b))
    order = sorted(range(b), key=lambda j: (-float(salience[j]), -j))
    return sorted(order[:target_blocks])


def normalize_probe_scores(scores: torch.Tensor) -> torch.Tensor:
    """L1-normalize nonnegative finite scores [P, B], keeping zero rows zero.

    Returns detached fp32 scores on the input device. Empty probe or block
    axes are supported. Row-max scaling prevents overflow in the row sum.
    """
    if scores.ndim != 2 or scores.is_complex():
        raise ValueError("scores must be a real tensor [P, B]")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite")
    if bool((scores < 0).any()):
        raise ValueError("scores must be nonnegative")
    dtype = torch.float64 if scores.dtype == torch.float64 else torch.float32
    values = scores.detach().to(dtype)
    if values.shape[0] == 0 or values.shape[1] == 0:
        return values.to(torch.float32)
    peak = values.amax(dim=1, keepdim=True)
    scaled = values / torch.where(peak > 0, peak, torch.ones_like(peak))
    total = scaled.sum(dim=1, keepdim=True)
    return (scaled / torch.where(total > 0, total, torch.ones_like(total))).to(
        torch.float32
    )


def coverage_select(scores: torch.Tensor, target_blocks: int) -> List[int]:
    """Greedily maximize sum_p sqrt(sum_j normalized[p, j]) over kept j.

    Marginal-gain ties prefer newer blocks; the result is in temporal order.
    With no probes (or all-zero scores), selection falls back to recency.
    Budgets include zero and all blocks: 0 <= target_blocks <= B.
    """
    normalized = normalize_probe_scores(scores)
    b = normalized.shape[1]
    if (
        isinstance(target_blocks, bool)
        or not isinstance(target_blocks, int)
        or not 0 <= target_blocks <= b
    ):
        raise ValueError("target_blocks must be an integer in [0, %d]" % b)
    covered = normalized.new_zeros(normalized.shape[0])
    available = torch.ones(b, dtype=torch.bool, device=normalized.device)
    kept: List[int] = []
    for _ in range(target_blocks):
        gains = (
            (covered[:, None] + normalized).sqrt() - covered[:, None].sqrt()
        ).sum(dim=0)
        gains = gains.masked_fill(~available, -float("inf"))
        # argmax picks the first maximum; reversing makes that the newest.
        j = b - 1 - int(gains.flip(0).argmax().item())
        kept.append(j)
        available[j] = False
        covered = covered + normalized[:, j]
    return sorted(kept)


def block_taylor_salience(
    grad: torch.Tensor, memory: torch.Tensor, block: int = 256
) -> torch.Tensor:
    """Per-block abs(sum(grad * memory)) for matching [L, K, d] tensors.

    Deleting a block has direction -memory, so this is the magnitude of its
    first-order directional loss proxy, not a guarantee of actual deletion
    damage. Signed contributions cancel before the absolute value is taken.
    Only one block at a time is cast to fp32, never the entire window.
    """
    if (
        grad.ndim != 3
        or memory.shape != grad.shape
        or grad.is_complex()
        or memory.is_complex()
        or grad.device != memory.device
    ):
        raise ValueError("grad and memory must be matching real [L, K, d] tensors on one device")
    if isinstance(block, bool) or not isinstance(block, int) or block <= 0:
        raise ValueError("block must be a positive integer")
    layers, slots, d = grad.shape
    if layers == 0 or d == 0 or slots % block != 0:
        raise ValueError("nonempty layers/features and a block-aligned window are required")
    result = torch.empty(slots // block, dtype=torch.float32, device=grad.device)
    for j in range(slots // block):
        start = j * block
        g = grad[:, start:start + block, :].detach()
        m = memory[:, start:start + block, :].detach()
        if not bool(torch.isfinite(g).all()) or not bool(torch.isfinite(m).all()):
            raise ValueError("grad and memory must be finite")
        value = (g.to(torch.float32) * m.to(torch.float32)).sum().abs()
        if not bool(torch.isfinite(value)):
            raise ValueError("Taylor salience overflowed fp32")
        result[j] = value
    return result


def nullspace_project(
    f: torch.Tensor, rows: torch.Tensor, tol: float = 1e-6
) -> Tuple[torch.Tensor, int]:
    """Return (f - proj_rowspace(f), rank).

    ``rows`` [P, N] spans the sensitive subspace (probe gradients at one
    survivor block, one layer). The rowspace projector is
    G^T (G G^T)^+ G, evaluated through the P x P eigendecomposition so no
    [N, r] basis is ever materialised:

        G G^T = V diag(lam) V^T          (P x P)
        proj  = G^T V diag(1/lam) V^T G f

    Eigenvalues below ``tol * max`` are dropped, which is what makes the
    routine safe when the probes' gradients are linearly dependent -- the
    Gram of E7a's probe pairs is far from full rank.
    """
    if rows.ndim != 2:
        raise ValueError("rows must be [P, N]")
    if f.ndim != 1 or f.numel() != rows.shape[1]:
        raise ValueError("f must be [N] matching rows")
    f = f.to(torch.float32)
    rows = rows.to(torch.float32)
    p = rows.shape[0]
    if p == 0:
        return f, 0
    gram = rows @ rows.T
    evals, evecs = torch.linalg.eigh(gram)
    floor = tol * max(1.0, float(evals[-1]) if evals.numel() else 0.0)
    keep = evals > floor
    rank = int(keep.sum().item())
    if rank == 0:
        return f, 0
    lam = evals[keep].clamp_min(1e-20)
    v = evecs[:, keep]
    c = rows @ f
    coef = (v * (1.0 / lam)) @ (v.T @ c)
    return f - rows.T @ coef, rank


class IncrementalBasis:
    """Growable orthonormal record of the probe gradients at one (block, layer).

    The driver backprops one probe at a time and cannot hold the [P, N]
    gradient stack for all survivors at once (24 probes x 32 layers x 9
    survivors x 1M dims is ~15 GB even in fp16). This class absorbs one
    gradient vector per call and keeps only an orthonormal basis, appending a
    direction only when its residual against the current basis exceeds
    ``tol * ||g||``. Linearly dependent gradients (E7a: probe-pair gradient
    cosines 0.05-0.32, and blocks inside one window are near-duplicates by G0)
    are absorbed for free, so the stored rank is the *effective* sensitivity
    rank, bounded above by ``cap``.

    Storage is fp16 with fp32 math per call; the basis lives wherever the
    caller put it (GPU is fine: cap x 1M x 2 B).
    """

    def __init__(self, cap: int, tol: float = 1e-3):
        self.cap = int(cap)
        self.tol = float(tol)
        self.basis: Optional[torch.Tensor] = None  # [r, N] fp16
        self.rejected = 0

    def add(self, g: torch.Tensor) -> bool:
        """Absorb one gradient vector [N]; True if the basis grew."""
        g = g.detach().to(torch.float32).reshape(-1)
        if self.basis is None or self.basis.shape[0] == 0:
            norm = float(g.norm())
            if norm <= 0.0:
                self.rejected += 1
                return False
            self.basis = (g / norm).to(torch.float16).unsqueeze(0)
            return True
        q = self.basis.to(torch.float32)
        residual = g - q.T @ (q @ g)
        r_norm = float(residual.norm())
        g_norm = float(g.norm())
        if g_norm <= 0.0:
            self.rejected += 1
            return False
        if r_norm <= self.tol * g_norm or self.basis.shape[0] >= self.cap:
            self.rejected += 1
            return False
        self.basis = torch.cat(
            [self.basis, (residual / r_norm).to(torch.float16).unsqueeze(0)], dim=0
        )
        return True

    @property
    def rank(self) -> int:
        return 0 if self.basis is None else int(self.basis.shape[0])

    def rows(self) -> Optional[torch.Tensor]:
        """The stored orthonormal rows, for ``nullspace_project``."""
        return None if self.basis is None else self.basis.to(torch.float32)


def _route_dropped(
    source: torch.Tensor, kept: Sequence[int], dropped: Sequence[int], block: int,
    reference_layer: int = 0,
) -> Dict[int, List[int]]:
    """Assign each dropped block to the survivor it is cosine-nearest to.

    Layer-0 block means, the same summary `_select_block_indices` scores, so
    routing and selection agree on what a block *is*. This is a container
    choice only: version semantics live in the salience selection (G0 showed
    similarity cannot carry them), and the merge is orthogonal anyway.

    Greedy nearest, not balanced assignment: a survivor that is similar to
    many dropped blocks absorbs all of them. The orthogonal projection, not
    the routing, is what bounds the behavioural damage.
    """
    layers, slots, d = source.shape
    b = slots // block
    x = source.detach().to(torch.float32).reshape(layers, b, block, d)
    ref = x[min(max(reference_layer, 0), layers - 1)].mean(dim=1)  # [b, d]
    ref = ref / (ref.norm(dim=-1, keepdim=True) + 1e-12)
    kept = sorted(int(j) for j in kept)
    assign: Dict[int, List[int]] = {j: [] for j in kept}
    if not dropped:
        return assign
    # One matmul over all dropped blocks; indexing (not fancy-list indexing,
    # which CUDA asserts on when the list lives on another device) maps each
    # row back to its survivor.
    dropped = sorted(int(i) for i in dropped)
    sims = ref[kept] @ ref[dropped].T  # [n_kept, n_dropped]
    nearest = sims.argmax(dim=0).tolist()  # [n_dropped]
    for row, i in zip(nearest, dropped):
        assign[kept[int(row)]].append(i)
    return assign


def merge_into_survivors(
    source: torch.Tensor,
    kept: Sequence[int],
    bases: Dict[Tuple[int, int], IncrementalBasis],
    block: int = 256,
    reference_layer: int = 0,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """First-order-orthogonal merge of dropped blocks into the survivors.

    For every survivor j and layer l:

        B'_{j,l} = B_{j,l} + (I - U_{j,l} U_{j,l}^T) f(A)_{j,l}

    where ``bases[(j, l)]`` holds the orthonormalised probe gradients at that
    (survivor, layer) -- U_{j,l} -- and f(A)_{j,l} is the mean over the dropped
    blocks routed to j, at layer l. Mean pooling is the S1-SDC "lightweight
    mixer" in its first version; a learned f is the projected-cone ablation,
    not the mainline.

    Returns (candidate [L, target, d], report). The candidate is assembled in
    the source's own dtype and device; survivors stay in temporal order.
    """
    layers, slots, d = source.shape
    if slots % block != 0:
        raise ValueError("window %d is not a multiple of block %d" % (slots, block))
    b = slots // block
    kept = sorted(int(j) for j in kept)
    if not kept:
        raise ValueError("merge needs at least one survivor")
    out_of_range = [j for j in kept if not 0 <= j < b]
    if out_of_range:
        # A readable error here instead of a device-side assert from the
        # fancy-index kernel, which reports nothing about which index was bad.
        raise IndexError("survivor blocks %s outside [0, %d)" % (out_of_range, b))
    dropped = sorted(set(range(b)) - set(kept))
    x = source.reshape(layers, b, block, d)

    routing = _route_dropped(source, kept, dropped, block, reference_layer)

    out = torch.empty(
        layers, len(kept), block, d, dtype=source.dtype, device=source.device
    )
    ranks: Dict[str, int] = {}
    orthogonal_energy = 0.0
    total_energy = 0.0
    for out_index, j in enumerate(kept):
        absorbed = routing[j]
        if absorbed:
            donors = x[:, absorbed, :, :].to(torch.float32).mean(dim=1)  # [L, block, d]
        for layer in range(layers):
            carrier = x[layer, j]  # [block, d]
            if not absorbed:
                out[layer, out_index] = carrier
                continue
            f_vec = donors[layer].reshape(-1)
            basis = bases.get((j, layer))
            rows = None if basis is None else basis.rows()
            delta, rank = nullspace_project(f_vec, rows)
            ranks["b%d_l%d" % (j, layer)] = rank
            orthogonal_energy += float(delta.dot(delta))
            total_energy += float(f_vec.dot(f_vec))
            merged = carrier.to(torch.float32).reshape(-1) + delta
            out[layer, out_index] = merged.reshape(block, d).to(source.dtype)

    candidate = out.reshape(layers, len(kept) * block, d).contiguous()
    report = {
        "kept": kept,
        "dropped": dropped,
        "routing": {str(j): v for j, v in routing.items()},
        "mean_basis_rank": (
            sum(ranks.values()) / len(ranks) if ranks else 0.0
        ),
        "orthogonal_fraction": (
            orthogonal_energy / total_energy if total_energy > 0 else 0.0
        ),
    }
    return candidate, report


class RollbackGate:
    """Component 3: refuse a merge that flips any probe's answer.

    The gate compares the probe set's correctness *after* the merge against the
    same probes on the pre-compaction pool, and reports the flipped questions.
    The driver decides the degradation: a flip means the first-order guarantee
    was violated by second-order effects, and the touched survivors fall back
    to pure selection (the merge is simply not applied; survivors' original
    bytes are what selection would have kept anyway).
    """

    def __init__(self, probes, pre: Dict[str, bool]):
        self.probes = probes
        self.pre = {str(q): bool(v) for q, v in pre.items()}
        self.flipped: List[str] = []

    def check(self, post: Dict[str, bool]) -> bool:
        """Record flips; return True when the merge may stand."""
        self.flipped = sorted(
            q
            for q in self.pre
            if self.pre[q] and not bool(post.get(q, False))
        )
        return not self.flipped

    def report(self) -> Dict[str, object]:
        return {
            "n_probes": len(self.pre),
            "n_flipped": len(self.flipped),
            "flipped": self.flipped,
            "rolled_back": bool(self.flipped),
        }
