# SIC-P1: controlled entity-disjoint paper protocol

## Status

Data prepared and frozen; development smoke (episode 0) completed; full
development run and sealed test pending. SIC stability candidate did **not**
beat native `select` on the smoke episode. Test episodes remain unopened.

## What was audited and corrected first

- The 82-question O2 confirmation set was already opened; later swap and
  local-oracle runs on the same questions are exploratory reuse, not
  independent confirmation.
- Support-oracle values (`11/14/20/12`) are specified-text-support omissions,
  not reader eligible-loss lower bounds. Actual reader development losses
  were `9/9/4/3`, already below the purported bounds on rows 1-3.
- Swap accounting split: whole-set `5/4` net +1 vs eligible `3/4` net -1.
- Teacher-forced answer-token exact match is not official generation EM.
- Only 4 unique Conflict_Resolution contexts exist in the dataset
  (`runs/sic_paper_inventory.json`); rows 4-7 duplicate rows 0-3.

## Controlled data (not natural contexts)

`runs/sic_paper_controlled_plan.json` (SHA256
`1444f62a1197a8663736f9f4d767aa3923f0432ab98507dde99de3d3bd64b9e5`):

- 24 episodes, 8 development / 16 test, entity-disjoint by construction.
- Source: row3 template facts (14,637), 6,802 usable subjects after excluding
  subjects from old rows 0-3; 3,792 used, 3,010 remaining.
- Each episode: 410 lines = 30 anchor + 380 fact lines (19 blocks x 20 lines),
  128 probe keys (124x3 versions + 4x2), episode serials 1-410.
- Splits 32/32/64 (discovery/calibration/evaluation), relation-cycled,
  globally disjoint entities/keys/questions.
- Token audit `runs/sic_paper_token_audit.json`: anchors 370-428 tokens,
  blocks 244-304, all <= 512, no truncation, no scoring during preparation.
- This is a controlled stress setting under a shared source corpus and answer
  banks; it is not natural-context generalization and must not be claimed as
  a native benchmark score.

## Protocol (frozen in `tools/sic_paper.py` PROTOCOL)

- Arms: native `select`, `raw`, `eligible_normalized`, `stability`,
  `gated_stability` (with native fallback), `recent`, `random`.
- Primary metric: all-question greedy generation EM (no gold in the query,
  stop at newline, official-style two-tier scoring from `readout_probe`).
- Structural metric: specified-text-support coverage; support oracle (92,378
  combinations) is a diagnostic target, not a reader EM upper bound.
- Acceptance gates compare **ungated SIC** against prespecified non-oracle
  baselines and against native on oracle gap; native fallback is operational
  protection and is never counted as SIC superiority.
- Episode-level inference: sign-flip test (symmetry assumption) + episode
  bootstrap 95% CI; question-level McNemar is inadequate when questions share
  a pool.
- Zero-pool ablation defines an operational eligibility subset only, not
  causal memory dependence.
- Sealed test: one frozen choice, single-open ledger file, no subsets, no
  reselection from test.

## Implementation notes

- Full-pool gradients OOM on 32GB V100 (fp16 weights ~28.7GB + full pool
  autograd). Fixed by keeping full pool geometry in the forward while
  differentiating only the 19-block working window; source pool offloaded to
  CPU. Recorded in costs as `pool_offload`.
- Active-mask verification: first scored candidate perturbs masked-out slots
  and requires unchanged reader logits (`max_logit_delta`); compressed
  active-slot budget must equal `k_anchor + 9*256 = 2816`.
- Every episode resets via `initialize_anchor -> reset()`; cross-episode pool
  carryover is prevented by the reset path, not by constructing a new manager.
- Runtime nondeterminism warnings (cumsum, memory-efficient attention
  backward) persist under `warn_only=True`; recorded, not suppressed.

## Smoke result (development episode 0, NOT a confirmation)

| arm | generation EM /64 |
|---|---|
| select | 23 |
| recent | 23 |
| gated_stability | 23 (native fallback; gate rejected candidate) |
| eligible_normalized / stability | 22 |
| random | 21 |
| raw | 18 |

Structural coverage: oracle 0.8125, select 0.78125, stability 0.640625.

The stability candidate regressed on both behavior and structure on this
episode; the fallback matched native by construction. This is a development
observation driving method iteration, not a paper claim.

Cost (episode 0): 32 window backwards in 112s; total episode 1598s including
7-arm generation evaluation; peak allocated 33.1GB (fits with expandable
segments).

## Reproduction

```bash
# CPU regression tests
python tools/test_sic_stability.py
python tools/test_sic_paper_data.py
python tools/test_sic_selection.py
python tools/test_sic_refine.py

# Remote prepare (CPU, no scoring)
python tools/sic_paper_data.py --dataset <conflict parquet> \
  --templates tools/e6a_overfit.py --out runs/sic_paper_controlled_plan.json

# Development run (remote, one GPU)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python tools/sic_paper.py \
  --stage development --plan runs/sic_paper_controlled_plan.json \
  --out runs/sic_paper_dev.json --cuda-device 9

# Freeze then sealed test (16 test episodes, one shot)
python tools/sic_paper.py --stage freeze --plan runs/sic_paper_controlled_plan.json \
  --development runs/sic_paper_dev.json --out runs/sic_paper_freeze.json
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python tools/sic_paper.py \
  --stage test --plan runs/sic_paper_controlled_plan.json \
  --freeze runs/sic_paper_freeze.json --out runs/sic_paper_test.json --cuda-device 9
```
