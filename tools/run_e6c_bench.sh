#!/bin/bash
# E6c, step 2 of 2: the three-arm benchmark under one tightened memory budget.
#
#   Eviction            compact_mode none            -- never compresses, drops oldest
#   Native selection    compact_mode select_blocks   -- trained-free block selection
#   Ours                compact_mode carrier_residual -- the trained operator
#
# Everything else is held fixed across the three arms: same dataset, same
# k_anchor 512, same k_max 6144, same compact_min_slots 4864, gate_mode ones,
# trigger occupancy at tau_k 0.80, active_memory_mask on, native_filler_only on.
# The only per-arm differences are compact_mode, output_dir/stats_dir, and the
# carrier checkpoint.
#
# The question the round answers: with capacity genuinely tight, does learned
# compaction keep more correct answers than not compressing at all? It does not
# answer anything about the trigger, the gate, SVD, or the selector -- none of
# those are exercised differently between arms.
#
# Each arm goes through retry_run.sh, which picks a GPU with >= 21 GB free and
# retries on contention. Wall clock is much larger than E5/E6: the pool now caps
# at 6144 slots instead of 12288, so writes hit the budget far earlier and every
# subsequent write compacts or evicts.
set -uo pipefail

BENCH=/share/latent/PRLM-C
CKPT=/tmp/e6c_ckpt.pt

if [ ! -f "$CKPT" ]; then
  echo "[e6c] no checkpoint at $CKPT; run tools/run_e6c_train.sh first"
  exit 1
fi

# retry_run.sh passes `--agent_config configs/agent_conf/prlmc_${RUN}.yaml` as a
# relative path, so the repository root has to be the working directory.
cd "$BENCH" || exit 1

for RUN in e6c-evict e6c-select e6c-carrier; do
  echo "[e6c] === $RUN ($(date -u)) ==="
  RUN="$RUN" bash "$BENCH/retry_run.sh" || { echo "[e6c] $RUN FAILED"; exit 1; }
done

echo "[e6c] all three arms done ($(date -u))"
