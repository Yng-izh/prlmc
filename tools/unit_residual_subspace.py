"""CPU-only regression checks for the residual-subspace scaffolding.

This file intentionally constructs tiny tensors and a tiny compactor. It never
loads a checkpoint, touches CUDA, or starts an experiment.
"""

import torch

from methods.prlmc.consolidate import (
    CarrierResidualCompactor,
    FixedBasisCarrierResidualCompactor,
)
from methods.prlmc.functional_subspace import (
    CoefficientOracle,
    FunctionalSensitivitySketch,
    scalar_jacobian_rows,
)
from methods.prlmc.repeated import RepeatedCompactionLedger


def test_functional_sketch():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    unused = torch.nn.Parameter(torch.tensor([3.0]))
    scalars = torch.stack([parameter[0] ** 2, parameter[1] ** 2])
    rows = scalar_jacobian_rows(
        scalars,
        [("parameter", parameter), ("unused", unused)],
    )
    assert rows.shape == (2, 3)
    assert torch.equal(rows[:, 2], torch.zeros(2))
    assert torch.equal(rows[0, :2], torch.tensor([2.0, 0.0]))
    assert torch.equal(rows[1, :2], torch.tensor([0.0, 4.0]))

    probe_rows = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    sketch = FunctionalSensitivitySketch(3, rank=1)
    report = sketch.fit(probe_rows)
    before = torch.tensor([1.0, 0.0, 0.0])
    after = sketch.soft_filter(before, 10.0)
    assert report.rank == 1
    assert after.norm() < before.norm()
    assert sketch.risk(after) < sketch.risk(before)
    assert torch.allclose(CoefficientOracle(3).apply(torch.eye(3)), torch.eye(3))


def test_fixed_basis_compactor():
    warmup = CarrierResidualCompactor(num_layers=2, hidden=4, block=2, rank=2)
    with torch.no_grad():
        warmup.up.weight.normal_()
        warmup.down.weight.normal_()
    arm = FixedBasisCarrierResidualCompactor.from_warmup(warmup)
    assert [name for name, p in arm.named_parameters() if p.requires_grad] == ["down.weight"]
    output = arm(torch.randn(2, 4, 4), target=2, block=2)
    assert output.shape == (2, 2, 4)


def test_repeated_ledger():
    ledger = RepeatedCompactionLedger("initial", 8)
    ledger.append_write("write-1", 8, "write-1")
    ledger.compact(lambda state: state + ":compact", 4, "compact-1", "fixed")
    ledger.append_write("write-2", 8, "write-2")
    ledger.compact(lambda state: state + ":compact", 4, "compact-2", "fixed")
    report = ledger.report()
    assert report["events"][1]["source_ids"] == ["write-2"]
    assert report["state_id"] == "compact-2"


def main():
    test_functional_sketch()
    test_fixed_basis_compactor()
    test_repeated_ledger()
    print("residual-subspace CPU checks passed")


if __name__ == "__main__":
    main()
