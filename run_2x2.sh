#!/usr/bin/env bash
# One cell of the 2x2 gate x compact isolation matrix.
#
#   usage:  CELL=a GPU=6 bash run_2x2.sh
#
# Every cell is prlmc_r1.yaml with exactly two switches changed, so cell D
# (gate=novelty, compact=svd_tokens) is byte-identical to the already-run E3.
set -uo pipefail

BENCH=/share/latent/PRLM-C
PY=/share/latent/mllm_env/bin/python
CELL="${CELL:?set CELL to a|b|c}"
GPU="${GPU:-6}"
QUERIES="${QUERIES:-0}"

cd "$BENCH"
export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false

OUT="$BENCH/outputs/prlmc-$CELL"
rm -rf "$OUT" "$BENCH/methods/prlmc/__pycache__"
mkdir -p "$OUT" "$BENCH/logs"

echo "=== 2x2 cell=$CELL gpu=$GPU queries=$QUERIES ==="
date
"$PY" /share/latent/mllm_patch/batch_run.py \
  configs/data_conf/Conflict_Resolution/Factconsolidation_sh_32k.yaml \
  --agent_config "configs/agent_conf/prlmc_${CELL}.yaml" \
  --chunk 400 \
  --queries "$QUERIES" \
  --bench "$BENCH"
echo "=== 2x2 cell=$CELL finished ==="
date
