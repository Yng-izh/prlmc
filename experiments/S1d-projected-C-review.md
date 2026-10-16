# S1d projected-C design review

This document records the implementation prepared after the C+RiskProtect
experiment. No GPU run is authorized by this document.

## Fixed variants

| Variant | Basis | Coefficients | Purpose |
|---|---|---:|---|
| `C32` | original damaged-fact gradient rows, SVD | 32 | test capacity alone |
| `P13` | damaged rows projected away from retained-fact gradients, SVD | 13 | test protected geometry at the old dimension |
| `P32` | same projected rows, SVD | 32 | primary candidate: capacity plus protected geometry |

All three keep the original C tail-4-token summed gradient, carrier, reader,
optimizer, learning rate, norm budget, seed, 60 steps and measurement cadence.
The protection mechanism is part of basis construction only. It adds no hinge
loss and does not use benchmark QA questions, answers, correctness, or
post-training scores.

## Construction

For damaged write-time fact gradient rows `G` and retained write-time fact
gradient rows `K`, the protected rows are

`G_safe = G (I - U_K U_K^T)`

where `U_K` is the float64 thin-SVD basis of up to 40 eligible retained rows.
Eligibility is finite D0 answer CE in `[0, 2]`; selection is deterministic by
fact key. The final basis is the SVD basis of `G_safe`, truncated to 13 or 32
columns. The implementation rejects a requested dimension when effective rank
is insufficient, and never fills missing dimensions with random directions.

The projection is a first-order gradient constraint. It does not guarantee
token-exact preservation after the finite AdamW trajectory. Basis diagnostics
record effective rank, projected-to-original gradient energy, protection
directional overlap, orthogonality error and tensor hashes.

## Why 32D is a hypothesis, not a free improvement

The existing C setting has no evidence that 13 is optimal. More dimensions may
recover more facts, but also expose more directions that can alter retained
knowledge. `C32` separates capacity from protection; `P13` separates geometry
from capacity; `P32` is the combined candidate. These three comparisons are
the minimum needed for attribution.

The repository already contains an earlier S1-local hard-nullspace experiment.
Its protected subspace had effective rank 6 and was a NO-GO because recovery
collapsed. That result is a warning about the recovery/protection trade-off,
not a direct verdict on this implementation: the new construction uses the
window's write-time fact gradients and explicitly tests 32D when rank permits.

## Artifacts and checks

- `tools/s1d_projection.py`: deterministic projection, rank and cache API.
- `tools/s1d_projected_train.py`: frozen-C training wrapper; preparation mode
  writes only a protection-gradient cache and returns before optimizer setup.
- `tools/verify_s1d_projection.py`: report, cache, basis and checkpoint audit.
- `tools/test_s1d_projection.py`: CPU projection, rank, cache, driver and
  actual-gradient tests.

The protection cache is published atomically and refuses overwrite. Every
formal report must reproduce the published C window cohorts, baselines, raw C
basis hash and fact cache before training. The new script has not been copied
to the V100 servers and no new GPU process has been started.

## Proposed run order after review

Prepare one protected-gradient cache per window, then run `C32`, `P13` and
`P32` with the same 60-step protocol. Use `P32` as the primary candidate only
if its effective rank is at least 32 and its projected gradient energy is
reported. Evaluate the weak window first; only a pre-registered passing result
should justify the strong-window run.
