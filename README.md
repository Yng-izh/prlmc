# PRLM-C — Protected Residual Latent Memory with Capacity-aware Consolidation

Implementation of the design note *残差记忆设计_方案整理* as a method for
MemoryAgentBench on top of the frozen MemoryLLM-8B backbone.

## Layout

```
methods/prlmc/        the method (11 modules, ~3.3k lines)
  layout.py           ProtectedResidualMemory: Anchor slots, residual pool, compaction
  capacity.py         occupancy / effective rank / energy rank, CompactTrigger
  consolidate.py      compaction arms (svd_tokens, weighted, learned, fixed-basis residual)
  functional_subspace.py  bounded Jacobian sketch and displacement controls
  repeated.py          real-state ancestry ledger for repeated compaction
  subspace.py         AnchorProjection, novelty probe
  budget.py           byte / token / latency accounting (note section 3.5)
  baselines.py        section 5.1 control arms B5-B8 and OURS
  probes.py           behaviour probes for probe-guided compaction
configs/agent_conf/   agent configs (prlmc_8b, prlmc_obs, prlmc_smoke)
configs/data_conf_obs/ dev splits used for threshold calibration
configs/prlmc_8b.yaml duplicate of configs/agent_conf/prlmc_8b.yaml
tools/residual_subspace_diag.py  offline S0/S0b tensor diagnostics
tools/residual_subspace_train.py  transactional actual-displacement helpers
tools/repeated_compaction_eval.py  manifest validator for frozen horizons
tools/unit_residual_subspace.py  CPU-only regression checks
configs/prlmc_residual_subspace.yaml  opt-in fixed-basis profile
apply_patch.py        installs the prlmc branch into a MemoryAgentBench copy
run_obs.sh            observation run (compact_mode: none)
run_smoke.sh          smoke test on factconsolidation_sh_32k
monitor_obs.sh        polling monitor for the observation run
```

## Deploying

Copy the tree over a MemoryAgentBench checkout, then run `apply_patch.py` from
that root. It edits two host files idempotently (`agent.py`, `utils/templates.py`)
and writes `.bak` copies.

The agent name must contain `prlmc` and must not contain `rag`:
`AgentWrapper._is_agent_type` does substring matching and the rag branch would
otherwise claim the call.

## Status

- Baseline Tier 1 (MemoryAgentBench, unmodified) has been evaluated: 5 datasets.
- The observation run has been done: internal signals measured, thresholds calibrated.
- S1d development experiments are complete: original C generation EM is
  46.04% / 44.93% on the strong/weak windows; Full is 64.53% / 63.04%.
  New protection/dimension variants did not pass the weak-window net≥5 gate.
  These are development fact-derived questions scored with official EM rules,
  not unseen-context or native official benchmark results. See
  [the complete round report](experiments/S1d-round-20261013-results.md) and
  [raw reports and logs](reports/s1d_round_20261013/).

## Known issues

- The threshold trigger `tau_k AND tau_n` did not fire in the smoke test: the two
  signals are anti-correlated (novelty falls as occupancy rises), so the windows
  are disjoint. Consolidation was driven by the `headroom` fallback path in
  `_place` instead. See the smoke test findings.
- `_headroom_handled` is set on every write, so the `over_budget` branch is dead.
  It does not affect correctness (`_place` already guarantees the budget) but the
  flag does not mean what its comment says.
- `compact_probe: true` is inert: `register_probes` has no call site in the
  benchmark path, so section 3.4's probe-guided compaction is not exercised.
- `BudgetReport.byte_saving_ratio` returns 0.0 by construction: the pool is a
  fixed-shape parameter read in full, so compaction reduces addressable tokens,
  not allocated bytes.
