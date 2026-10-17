"""Offline S0/S0b diagnostics for residual-subspace experiments.

The module accepts saved tensors and never loads the reader by itself.  This
keeps local validation cheap and makes it impossible for importing the tool to
silently launch a GPU job.  A host-side driver can call ``run_s0`` after it has
constructed end-to-end margin Jacobian rows.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from methods.prlmc.functional_subspace import (
    CoefficientOracle,
    FunctionalSensitivitySketch,
)


def orthogonal_directions(sketch: FunctionalSensitivitySketch, count: int, seed: int = 0):
    """Return sensitive, low-sensitive, and random equal-norm directions."""
    if count < 1:
        raise ValueError("count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random = torch.randn(count, sketch.dimension, generator=generator)
    if sketch.mu.numel():
        v = sketch.v.to(random)
        sensitive = v[:,:count].T if v.shape[1] >= count else v.T
        if sensitive.shape[0] < count:
            sensitive = torch.cat([sensitive, random[: count - sensitive.shape[0]]], dim=0)
        residual = random - (random @ v) @ v.T
    else:
        sensitive = random.clone()
        residual = random.clone()
    low = residual / residual.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sensitive = sensitive / sensitive.norm(dim=1, keepdim=True).clamp_min(1e-12)
    random = random / random.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return {"sensitive": sensitive, "low_sensitive": low, "random": random}


def run_s0(rows: torch.Tensor, scales, directions: int = 8, seed: int = 0) -> Dict:
    """Build the local sensitivity sketch and return reproducible perturbations."""
    sketch = FunctionalSensitivitySketch(rows.shape[1], rank=min(32, rows.shape[0]))
    report = sketch.fit(rows)
    direction_set = orthogonal_directions(sketch, directions, seed)
    result = {
        "sketch": {
            "rows": report.rows,
            "rank": report.rank,
            "trace": report.trace,
            "discarded_energy": report.discarded_energy,
        },
        "directions": {},
    }
    for name, vectors in direction_set.items():
        result["directions"][name] = [
            (vector * float(scale)).tolist()
            for vector in vectors
            for scale in scales
        ]
    return result


def run_s0b(coefficients: torch.Tensor, rank: int, steps: int = 0) -> Dict:
    """Return an identity-initialized temporary coefficient oracle snapshot."""
    if coefficients.shape[-1] != rank:
        raise ValueError("coefficients last dimension must equal rank")
    oracle = CoefficientOracle(rank)
    with torch.no_grad():
        transformed = oracle.apply(coefficients)
    return {
        "rank": rank,
        "steps": int(steps),
        "identity_error": float((transformed - coefficients).norm()),
        "parameter_count": sum(p.numel() for p in oracle.parameters()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, help="CPU tensor file containing [n, p] Jacobian rows")
    parser.add_argument("--out", required=True)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    args = parser.parse_args()
    rows = torch.load(args.rows, map_location="cpu", weights_only=True)
    result = run_s0(rows, args.scales, args.directions)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
