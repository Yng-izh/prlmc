"""Working-memory basis, feature novelty, and Anchor projection.

Section 3.2. The candidate observation z_t is a column vector in R^d; the working
memory is stored row-wise as [K, d], so the orthonormal basis derived from it is
U_t in R^{d x r} and novelty is

    e_t = (I - U_t U_t^T) z_t
    rho_t = ||e_t||^2 / (||z_t||^2 + eps)

Two warnings from the note are load-bearing and are enforced here:

  * rho_t is a *feature novelty proxy*, not importance and not value. It only
    allocates capacity; it must never decide whether a fact correction is
    accepted.
  * The projection matrix Q_A = I - U_A Lambda U_A^T right-multiplies a row-wise
    delta. Getting this backwards silently produces a no-op or a shape error
    depending on d, so the row/column convention is asserted rather than assumed.

The hard projection is deliberately not the default: task-relevant new knowledge
can share directions with the Anchor, and freezing those directions costs
plasticity (risk 3 in the note).
"""

from typing import Optional, Tuple

import torch


def estimate_basis(
    memory: torch.Tensor,
    max_rank: int = 256,
    energy: float = 0.95,
    subtract_mean: bool = True,
    eps: float = 1e-8,
    max_rows: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Estimate an orthonormal working-memory basis from memory activations.

    Args:
        memory: [K, d] actual memory slots, not a basis.
        max_rank: statistical budget r_max. Kept separate from the token budget
            K_max on purpose; the note requires them to be configured and
            reported separately.
        energy: keep the smallest number of directions covering this share of
            the centred energy, capped by max_rank.
        subtract_mean: whether to centre. When True the mean is returned so the
            caller can account for its storage, as section 3.3 requires.
        max_rows: when > 0 and K exceeds it, decompose an evenly spaced row
            subsample instead of the full matrix. A full SVD of [12800, 4096] is
            seconds of GPU time on every write; the basis is a coarse redundancy
            estimate, so a bounded subsample gives the same decisions. 0 disables
            the cap.

    Returns:
        (basis, mean, singular_values) where basis is [d, r] with orthonormal
        columns, mean is [d] or None, and singular_values is the kept spectrum.
    """
    if memory is None or memory.numel() == 0 or memory.shape[0] == 0:
        width = memory.shape[1] if memory is not None and memory.dim() > 1 else 0
        return memory.new_zeros((width, 0)), None, memory.new_zeros(0)

    x = memory.detach().to(torch.float32)
    if max_rows > 0 and x.shape[0] > max_rows:
        step = x.shape[0] / float(max_rows)
        index = torch.arange(max_rows, device=x.device, dtype=torch.float64) * step
        x = x[index.long().clamp_(0, x.shape[0] - 1)]

    mean = None
    if subtract_mean:
        mean = x.mean(dim=0)
        x = x - mean.unsqueeze(0)

    # full_matrices=False keeps this at O(K * d * min(K, d)) rather than also
    # materialising the K x K left factor we do not need.
    try:
        _, singular_values, vh = torch.linalg.svd(x, full_matrices=False)
    except Exception:
        return x.new_zeros((x.shape[1], 0)), mean, x.new_zeros(0)

    if singular_values.numel() == 0:
        return x.new_zeros((x.shape[1], 0)), mean, singular_values

    energy_total = singular_values.pow(2).sum()
    if float(energy_total) <= 0.0:
        return x.new_zeros((x.shape[1], 0)), mean, singular_values

    cumulative = torch.cumsum(singular_values.pow(2), dim=0) / energy_total
    reached = torch.nonzero(cumulative >= energy, as_tuple=False)
    rank = int(reached[0].item()) + 1 if reached.numel() > 0 else singular_values.numel()
    rank = max(1, min(rank, max_rank, singular_values.numel()))

    # vh rows are right singular vectors, i.e. directions in feature space.
    basis = vh[:rank].transpose(0, 1).contiguous()
    return basis, mean, singular_values[:rank]


def project_out(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Remove the components of row vectors x that lie in span(basis).

    x is [..., d] and basis is [d, r], so this is x - (x U) U^T.
    """
    if basis is None or basis.numel() == 0:
        return x
    coefficients = x @ basis
    return x - coefficients @ basis.transpose(0, 1)


def feature_novelty(z: torch.Tensor, basis: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """rho_t for one or more candidate vectors.

    Args:
        z: [d] or [n, d] candidate representations.
        basis: [d, r] orthonormal basis, or an empty tensor.

    Returns:
        scalar or [n] tensor in [0, 1]. Vectorised so a whole window of
        candidates can be scored at once, which is what the note asks for when a
        single observation carries several new factors.
    """
    z = z.detach().to(torch.float32)
    lone = z.dim() == 1
    if lone:
        z = z.unsqueeze(0)

    denominator = z.pow(2).sum(dim=-1) + eps
    if basis is None or basis.numel() == 0:
        result = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
    else:
        residual = project_out(z, basis.to(z.device))
        result = residual.pow(2).sum(dim=-1) / denominator

    result = result.clamp(0.0, 1.0)
    return result[0] if lone else result


class NoveltyEMA:
    """Exponentially weighted mean of the novelty proxy (rho_bar_t)."""

    def __init__(self, beta: float = 0.9):
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        self.beta = beta
        self.value: Optional[float] = None
        self.count = 0

    def update(self, novelty: torch.Tensor) -> float:
        scalar = float(novelty.reshape(-1).mean()) if isinstance(novelty, torch.Tensor) else float(novelty)
        self.count += 1
        if self.value is None:
            self.value = scalar
        else:
            self.value = self.beta * self.value + (1.0 - self.beta) * scalar
        return self.value

    def reset(self) -> None:
        self.value = None
        self.count = 0


class AnchorProjection:
    """Q_A = I - U_A Lambda U_A^T held in factored form, never materialised.

    Storing Q_A as a dense [d, d] matrix would cost 67 MB at d = 4096 in fp32 and
    force every write to upcast the whole [L, num_tokens, d] block before a
    matmul whose result is only rank-r away from the identity. Right-multiplying
    by the factored form uses the same rank-r inner product on a much smaller
    operand:

        x Q_A = x - ((x @ U_A) * lambda) @ U_A^T

    `numel` and `element_size` exist because the byte accounting in `budget.py`
    reads them uniformly off whatever object is stored here; the reported size is
    the factored footprint, which is the honest one.
    """

    __slots__ = ("u", "lam")

    def __init__(self, u: Optional[torch.Tensor], lam: Optional[torch.Tensor]):
        self.u = u
        self.lam = lam

    @classmethod
    def empty(cls, device=None, dtype=None) -> "AnchorProjection":
        return cls(None, None)

    @property
    def active(self) -> bool:
        return self.u is not None and self.u.numel() > 0 and self.lam is not None

    @property
    def shape(self) -> Tuple[int, int]:
        """Factored shape [d, r], not the [d, d] operator it represents."""
        if not self.active:
            return (0, 0)
        return (int(self.u.shape[0]), int(self.u.shape[1]))

    @property
    def rank(self) -> int:
        return 0 if not self.active else int(self.u.shape[1])

    def numel(self) -> int:
        if not self.active:
            return 0
        return int(self.u.numel() + self.lam.numel())

    def element_size(self) -> int:
        if not self.active:
            return 4
        return int(self.u.element_size())

    def project(self, delta: torch.Tensor) -> torch.Tensor:
        """Apply Q_A to a row-wise delta of shape [..., d]."""
        if not self.active:
            return delta
        d = delta.shape[-1]
        if self.u.shape[0] != d:
            raise ValueError(
                "projection basis is [%d, %d] but delta has feature dim %d; slots "
                "are stored row-wise, so U_A must be [d, r] for a right-multiply"
                % (self.u.shape[0], self.u.shape[1], d)
            )
        flat = delta.reshape(-1, d).to(torch.float32)
        coeff = (flat @ self.u) * self.lam.unsqueeze(0)
        projected = flat - coeff @ self.u.transpose(0, 1)
        return projected.reshape(delta.shape).to(delta.dtype)


def build_projection(
    anchor_basis: torch.Tensor,
    lambdas: torch.Tensor,
    mode: str,
    d: int,
    device,
    dtype,
) -> AnchorProjection:
    """Build Q_A = I - U_A Lambda U_A^T in factored form.

    mode "none" returns an inactive projection so callers skip the multiply
    entirely; the alternative is a rank-r inner product on every slot of every
    write.
    """
    if mode == "none" or anchor_basis is None or anchor_basis.numel() == 0:
        return AnchorProjection.empty(device=device, dtype=dtype)

    u = anchor_basis.to(device=device, dtype=torch.float32)
    if mode == "hard":
        lam = torch.ones(u.shape[1], device=device, dtype=torch.float32)
    elif mode == "soft":
        lam = lambdas.to(device=device, dtype=torch.float32).flatten()
        if lam.numel() < u.shape[1]:
            lam = torch.cat([lam, lam[-1:].repeat(u.shape[1] - lam.numel())])
        lam = lam[: u.shape[1]]
    else:
        raise ValueError(f"unknown projection mode: {mode}")

    return AnchorProjection(u.contiguous(), lam.contiguous())


def apply_projection(delta: torch.Tensor, projection) -> torch.Tensor:
    """Apply Q_A to a row-wise delta.

    Accepts the factored operator from `build_projection`. A plain dense [d, d]
    tensor is still handled so an externally supplied matrix cannot silently
    become a no-op, but the manager always stores the factored form.
    """
    if projection is None:
        return delta
    if isinstance(projection, AnchorProjection):
        return projection.project(delta)
    if projection.numel() == 0:
        return delta
    if projection.shape[0] != delta.shape[-1]:
        raise ValueError(
            "projection is [%d, %d] but delta has feature dim %d; "
            "slots are stored row-wise, so Q_A must be [d, d] on the right"
            % (projection.shape[0], projection.shape[1], delta.shape[-1])
        )
    return (delta.to(torch.float32) @ projection).to(delta.dtype)
