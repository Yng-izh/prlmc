#!/usr/bin/env bash
# PRLM-C smoke test: does the trigger actually fire and consolidation actually
# free slots, now that the rank veto is gone and over_budget uses >=.
#
# factconsolidation_sh_32k, 83 chunks: the pool fills at chunk 48 and compaction
# has to carry the rest. --queries 2 only flushes the stats file; the evidence is
# in the per-write records, not in the answers.
set -uo pipefail

BENCH=/share/latent/PRLM-C
PY=/share/latent/mllm_env/bin/python

cd "$BENCH"
export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${GPU:-6}"
export TOKENIZERS_PARALLELISM=false

rm -rf "$BENCH/outputs/prlmc-smoke" "$BENCH/methods/prlmc/__pycache__"
mkdir -p "$BENCH/outputs/prlmc-smoke" "$BENCH/logs"

echo "=== prlmc smoke gpu=${GPU:-6} ==="
date

"$PY" /share/latent/mllm_patch/batch_run.py \
  configs/data_conf/Conflict_Resolution/Factconsolidation_sh_32k.yaml \
  --agent_config configs/agent_conf/prlmc_smoke.yaml \
  --chunk 400 \
  --queries "${QUERIES:-2}" \
  --bench "$BENCH"

echo "=== prlmc smoke finished ==="
date
