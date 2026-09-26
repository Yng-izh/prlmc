#!/usr/bin/env bash
# Observation run for PRLM-C threshold calibration.
#
# Runs on the PRLM-C tree (not the baseline tree): the point is to record the
# write path's capacity signals, and only PRLM-C has them.
#
# Two dev splits, neither of which is in the Tier 1 table:
#   icl_nlu                 333 chunks at chunk_size 400
#   factconsolidation_sh_64k 169 chunks
# The second is included because it exceeds the 12800-slot working capacity, so
# it shows what occupancy and rank do once the pool is full.
#
# --queries is small on purpose. Memorization covers the whole context before
# the first query, so all per-write records land in the first stats line; the
# queries only serve to flush that line.
#
# compact_mode is none in prlmc_obs.yaml, so a pool that fills falls through to
# eviction. That is expected here and is exactly the condition the real run has
# to avoid -- the stats file records every eviction.

set -uo pipefail

BENCH=/share/latent/PRLM-C
PY=/share/latent/mllm_env/bin/python
GPU="${GPU:-6}"
QUERIES="${QUERIES:-8}"

cd "$BENCH"
export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$BENCH/logs" "$BENCH/outputs/prlmc-obs/stats"
rm -rf "$BENCH/methods/prlmc/__pycache__"

echo "=== observation run gpu=${GPU} queries=${QUERIES} ==="
date

"$PY" /share/latent/mllm_patch/batch_run.py \
  configs/data_conf_obs/Factconsolidation_sh_64k.yaml \
  configs/data_conf_obs/ICL_nlu.yaml \
  --agent_config configs/agent_conf/prlmc_obs.yaml \
  --chunk 400 \
  --queries "$QUERIES" \
  --bench "$BENCH"

echo "=== observation run finished ==="
date
