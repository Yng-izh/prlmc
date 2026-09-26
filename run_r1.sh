#!/usr/bin/env bash
# PRLM-C Round 1: occupancy-only trigger, full accuracy measurement.
#
# factconsolidation_sh_32k at chunk_size 400 = 83 chunks, 100 queries. The pool
# holds 48 blocks (12288 slots), so writes 1-48 fill it and 49-83 must be served
# by compaction. That is exactly the regime the occupancy trigger is supposed to
# handle, and it is the same dataset the baseline scored 22.0 EM / 25.99 F1 on,
# so the accuracy loss is directly comparable.
#
# --queries 0 means no cap: the baseline result contains 100 queries and a
# truncated run would not be comparable. Cost is roughly 35 min of writes
# (83 x ~25 s), ~6 min of queries, plus the 235 s one-time model load.
set -uo pipefail

BENCH=/share/latent/PRLM-C
PY=/share/latent/mllm_env/bin/python

cd "$BENCH"
export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${GPU:-6}"
export TOKENIZERS_PARALLELISM=false

rm -rf "$BENCH/outputs/prlmc-r1" "$BENCH/methods/prlmc/__pycache__"
mkdir -p "$BENCH/outputs/prlmc-r1" "$BENCH/logs"

echo "=== prlmc round1 gpu=${GPU:-6} ==="
date

"$PY" /share/latent/mllm_patch/batch_run.py \
  configs/data_conf/Conflict_Resolution/Factconsolidation_sh_32k.yaml \
  --agent_config configs/agent_conf/prlmc_r1.yaml \
  --chunk 400 \
  --queries "${QUERIES:-0}" \
  --bench "$BENCH"

echo "=== prlmc round1 finished ==="
date
