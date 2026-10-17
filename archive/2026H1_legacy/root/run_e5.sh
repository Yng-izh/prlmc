#!/usr/bin/env bash
# One run of the E5 phase: native block selection, gate pinned to ones.
#
#   usage:  RUN=e5a GPU=6 bash run_e5.sh
#
# e5a = recent_blocks, e5b = select_blocks, e5c = select_blocks + active mask.
# All three are E4's cell A with exactly one switch changed, so every result is
# directly comparable to "gate=ones, compact=none -> EM 27.00".
set -uo pipefail

BENCH=/share/latent/PRLM-C
PY=/share/latent/mllm_env/bin/python
RUN="${RUN:?set RUN to e5a|e5b|e5c}"
GPU="${GPU:-6}"
QUERIES="${QUERIES:-0}"

cd "$BENCH" || exit 1
export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false

OUT="$BENCH/outputs/prlmc-$RUN"
rm -rf "$OUT" "$BENCH/methods/prlmc/__pycache__"
mkdir -p "$OUT" "$BENCH/logs"

echo "=== $RUN gpu=$GPU queries=$QUERIES ==="
date
"$PY" /share/latent/mllm_patch/batch_run.py \
  configs/data_conf/Conflict_Resolution/Factconsolidation_sh_32k.yaml \
  --agent_config "configs/agent_conf/prlmc_${RUN}.yaml" \
  --chunk 400 \
  --queries "$QUERIES" \
  --bench "$BENCH"
echo "=== $RUN finished (exit $?) ==="
date
