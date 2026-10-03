"""Training-side helpers for residual-subspace experiments.

The reader driver supplies the actual AdamW candidate displacement after
preconditioning, momentum and weight decay. This module filters that vector and
provides an explicit parameter/optimizer transaction; it never projects a raw
gradient hook.
"""

from contextlib import contextmanager

import torch

from methods.prlmc.functional_subspace import (
    FunctionalSensitivitySketch,
    transform_displacement,
)


@contextmanager
def parameter_transaction(module: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """Yield ``commit``; rollback parameters and optimizer moments unless called."""
    parameters = [p for p in module.parameters() if p.requires_grad]
    values = [p.detach().clone() for p in parameters]
    state = {
        parameter: {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in optimizer.state[parameter].items()
        }
        for parameter in parameters
        if parameter in optimizer.state
    }
    committed = False

    def commit():
        nonlocal committed
        committed = True

    try:
        yield commit
    finally:
        if not committed:
            with torch.no_grad():
                for parameter, value in zip(parameters, values):
                    parameter.copy_(value)
            for parameter in parameters:
                optimizer.state.pop(parameter, None)
            for parameter, snapshot in state.items():
                optimizer.state[parameter] = snapshot


def filter_actual_displacement(
    candidate: torch.Tensor,
    sketch: FunctionalSensitivitySketch,
    control: str,
    strength: float,
) -> tuple[torch.Tensor, dict]:
    """Filter a flat AdamW displacement and return audit metrics.

    ``candidate`` must already include the optimizer's preconditioner, momentum,
    and decay. A caller commits the returned vector to parameters inside
    ``parameter_transaction`` and only calls ``commit`` after finite-step audit.
    """
    if candidate.ndim != 1:
        raise ValueError("candidate displacement must be flat")
    filtered = transform_displacement(candidate, sketch, control, strength)
    return filtered, {
        "candidate_norm": float(candidate.norm()),
        "filtered_norm": float(filtered.norm()),
        "risk_before": float(sketch.risk(candidate)),
        "risk_after": float(sketch.risk(filtered)),
    }
