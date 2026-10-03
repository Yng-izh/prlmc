"""Functional subspace controls for residual-compactor experiments.

This module is deliberately independent of the reader and memory layout.  A
caller supplies end-to-end Jacobian rows with respect to the trainable
parameter vector (normally ``D``).  The sketch stores a bounded local estimate
of functional sensitivity and transforms the final optimizer displacement,
including AdamW preconditioning and weight decay, before parameters are
committed.
"""

from dataclasses import dataclass
from typing import Iterable, Optional

import torch


@dataclass
class SketchReport:
    rows: int
    rank: int
    trace: float
    discarded_energy: float


class FunctionalSensitivitySketch:
    """Small PSD Jacobian sketch ``H ~= V diag(mu) V.T``."""

    def __init__(self, dimension: int, rank: int = 16, dtype=torch.float32):
        if dimension < 1 or rank < 1:
            raise ValueError("dimension and rank must be positive")
        self.dimension = int(dimension)
        self.rank = int(rank)
        self.dtype = dtype
        self.v = torch.empty(self.dimension, 0, dtype=dtype)
        self.mu = torch.empty(0, dtype=dtype)
        self.rows = 0

    def clear(self):
        self.v = torch.empty(self.dimension, 0, dtype=self.dtype)
        self.mu = torch.empty(0, dtype=self.dtype)
        self.rows = 0

    def fit(self, rows: torch.Tensor, weights: Optional[torch.Tensor] = None):
        """Fit from rows shaped ``[n, p]`` without constructing a ``p x p`` matrix."""
        if rows.ndim != 2 or rows.shape[1] != self.dimension:
            raise ValueError("rows must have shape [n, dimension]")
        if rows.shape[0] == 0:
            self.clear()
            return SketchReport(0, 0, 0.0, 0.0)
        x = rows.detach().to(dtype=self.dtype, device="cpu")
        if not torch.isfinite(x).all():
            raise ValueError("non-finite sensitivity row")
        if weights is not None:
            weights = weights.detach().to(dtype=self.dtype, device="cpu").reshape(-1)
            if weights.numel() != x.shape[0] or (weights < 0).any():
                raise ValueError("weights must be non-negative and match row count")
            x = x * weights.clamp_min(0).sqrt().unsqueeze(1)
        gram = x @ x.T
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order].clamp_min(0)
        eigenvectors = eigenvectors[:, order]
        keep = min(self.rank, int((eigenvalues > 1e-12).sum()))
        if keep == 0:
            self.clear()
            self.rows = int(rows.shape[0])
            return SketchReport(self.rows, 0, 0.0, 0.0)
        values = eigenvalues[:keep]
        basis = x.T @ eigenvectors[:, :keep]
        basis = basis / values.sqrt().clamp_min(1e-12).unsqueeze(0)
        self.v = basis.contiguous()
        self.mu = values.contiguous()
        self.rows = int(rows.shape[0])
        total = float(eigenvalues.sum())
        discarded = float(eigenvalues[keep:].sum())
        return SketchReport(self.rows, keep, total, discarded / max(total, 1e-12))

    def to(self, device=None, dtype=None):
        self.v = self.v.to(device=device, dtype=dtype or self.v.dtype)
        self.mu = self.mu.to(device=device, dtype=dtype or self.mu.dtype)
        return self

    def risk(self, displacement: torch.Tensor) -> torch.Tensor:
        """Return ``u.T H u`` for a flat displacement."""
        u = displacement.reshape(-1)
        if u.numel() != self.dimension:
            raise ValueError("displacement dimension mismatch")
        if self.mu.numel() == 0:
            return u.new_zeros(())
        projection = self.v.T.to(u) @ u.to(self.v.device)
        return (self.mu.to(projection) * projection.square()).sum().to(u)

    def project(self, displacement: torch.Tensor) -> torch.Tensor:
        """Hard projection onto the complement of the estimated sensitive space."""
        if self.mu.numel() == 0:
            return displacement
        flat = displacement.reshape(-1)
        if flat.numel() != self.dimension:
            raise ValueError("displacement dimension mismatch")
        v = self.v.to(device=flat.device, dtype=flat.dtype)
        return (flat - v @ (v.T @ flat)).reshape_as(displacement)

    def soft_filter(
        self,
        displacement: torch.Tensor,
        strength: float,
        trace_normalized: bool = True,
    ) -> torch.Tensor:
        """Apply ``(I + lambda H)^-1`` to a final optimizer displacement."""
        if strength < 0:
            raise ValueError("strength must be non-negative")
        if self.mu.numel() == 0 or strength == 0:
            return displacement
        if trace_normalized:
            strength = strength / float(self.mu.sum().clamp_min(1e-12))
        flat = displacement.reshape(-1)
        if flat.numel() != self.dimension:
            raise ValueError("displacement dimension mismatch")
        v = self.v.to(device=flat.device, dtype=flat.dtype)
        mu = self.mu.to(device=flat.device, dtype=flat.dtype)
        coeff = strength * mu / (1.0 + strength * mu)
        return (flat - v @ (coeff * (v.T @ flat))).reshape_as(displacement)

    def report(self) -> SketchReport:
        trace = float(self.mu.sum()) if self.mu.numel() else 0.0
        return SketchReport(self.rows, int(self.mu.numel()), trace, 0.0)


class CoefficientOracle(torch.nn.Module):
    """Temporary S0b coefficient transform ``C = I + E``.

    The caller owns the optimization lifetime and must discard this module after
    writing the resulting carriers. It intentionally has no episode cache or
    deployment path.
    """

    def __init__(self, rank: int, max_deviation: Optional[float] = None):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.rank = int(rank)
        self.max_deviation = max_deviation
        self.residual = torch.nn.Parameter(torch.zeros(rank, rank))

    def matrix(self) -> torch.Tensor:
        matrix = torch.eye(self.rank, device=self.residual.device, dtype=self.residual.dtype)
        matrix = matrix + self.residual
        if self.max_deviation is not None:
            if self.max_deviation <= 0:
                raise ValueError("max_deviation must be positive")
            # Scale E as a whole: norm(E) = min(norm(E_raw), max_deviation).
            # The previous form subtracted E * min(1, max/norm), which at
            # norm=20, max=0.1 left 19.9 -- it shrank E by 0.5% instead of
            # clamping it. Scaling keeps the direction and enforces the bound.
            scale = self.residual.norm().clamp_min(1e-12)
            shrink = (self.max_deviation / scale).clamp_max(1.0)
            matrix = torch.eye(
                self.rank, device=matrix.device, dtype=matrix.dtype
            ) + self.residual * shrink
        return matrix

    def apply(self, coefficients: torch.Tensor) -> torch.Tensor:
        if coefficients.shape[-1] != self.rank:
            raise ValueError("coefficient rank mismatch")
        return coefficients @ self.matrix().transpose(-1, -2)


def flatten_named_tensors(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> torch.Tensor:
    """Flatten tensors in stable name order for reproducible Jacobian rows."""
    values = sorted(named_tensors, key=lambda item: item[0])
    if not values:
        raise ValueError("empty tensor collection")
    return torch.cat([value.reshape(-1).to(torch.float32) for _, value in values])


def scalar_jacobian_rows(
    scalars: torch.Tensor,
    named_parameters: Iterable[tuple[str, torch.Tensor]],
) -> torch.Tensor:
    """Build one flattened end-to-end Jacobian row per scalar output.

    ``scalars`` must be a one-dimensional tensor whose elements are individual
    gold-minus-foil margins. Unused parameters receive explicit zero blocks;
    rows are never summed before the sketch is fitted. The graph is retained
    until the final scalar so callers can keep other non-probe diagnostics.
    """
    if scalars.ndim != 1:
        raise ValueError("scalars must be a one-dimensional tensor")
    parameters = sorted(named_parameters, key=lambda item: item[0])
    if not parameters:
        raise ValueError("empty parameter collection")
    rows = []
    for index, scalar in enumerate(scalars):
        gradients = torch.autograd.grad(
            scalar,
            [parameter for _, parameter in parameters],
            retain_graph=index < scalars.numel() - 1,
            create_graph=False,
            allow_unused=True,
        )
        pieces = [
            (gradient.detach() if gradient is not None else torch.zeros_like(parameter)).reshape(-1)
            for gradient, (_, parameter) in zip(gradients, parameters)
        ]
        row = torch.cat([piece.to(torch.float32) for piece in pieces])
        if not torch.isfinite(row).all():
            raise ValueError("non-finite Jacobian row")
        rows.append(row)
    return torch.stack(rows) if rows else torch.empty((0, sum(p.numel() for _, p in parameters)))


def transform_displacement(
    displacement: torch.Tensor,
    sketch: FunctionalSensitivitySketch,
    control: str,
    strength: float = 0.0,
) -> torch.Tensor:
    """Select the hard, soft, or disabled control without touching gradients."""
    if control == "none":
        return displacement
    if control == "hard":
        return sketch.project(displacement)
    if control == "soft":
        return sketch.soft_filter(displacement, strength)
    raise ValueError(f"unknown functional control: {control}")


def displacement_metrics(before: torch.Tensor, after: torch.Tensor, sketch=None) -> dict:
    """Record norm, descent proxy, and functional risk for audit logs."""
    delta = after.reshape(-1) - before.reshape(-1)
    result = {
        "norm_before": float(before.reshape(-1).norm()),
        "norm_after": float(after.reshape(-1).norm()),
        "displacement_norm": float(delta.norm()),
    }
    if sketch is not None:
        result["risk"] = float(sketch.risk(delta))
        result["sensitive_fraction"] = float(
            (sketch.v.to(delta).T @ delta).norm() / delta.norm().clamp_min(1e-12)
        ) if sketch.mu.numel() else 0.0
    return result
