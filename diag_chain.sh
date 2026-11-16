#!/bin/bash
cd /data/zhangjiayang/yyz/PRLM-C || exit 1
PY=/data/zhangjiayang/yyz/env/bin/python
echo "=== eval: neighbour-choice arms $(date) ==="
$PY tools/eval_cross_block_calibration.py \
  --data-dir /data/zhangjiayang/yyz/datasets/LongBench/extracted/data --task hotpotqa \
  --supporting-facts-glob '/data/zhangjiayang/yyz/datasets/hotpot_qa/distractor/validation-*.parquet' \
  --model-path /dev/shm/s1d_reader_fp16 \
  --checkpoint runs/natural_merge_v9/full_v10.pt \
  --anchor runs/anchor_residual/anchor_only_rank64.pt \
  --v11 runs/cross_block/plain.pt --v11-evict runs/cross_block/cross_lr.pt \
  --v13-neighbor-modes random,bottom \
  --verify-against reports/anchor_fusion/eval_anchor_fusion_hotpotqa.json \
  --cuda-device 0 --out reports/cross_block/eval_v13_neighbors.json \
  > reports/cross_block/logs/eval_v13_neighbors.log 2>&1
echo "EVAL_EXIT=$? $(date)"
echo "=== oracle diagnosis $(date) ==="
$PY tools/diagnose_v13_oracle.py \
  --data-dir /data/zhangjiayang/yyz/datasets/LongBench/extracted/data --task hotpotqa \
  --supporting-facts-glob '/data/zhangjiayang/yyz/datasets/hotpot_qa/distractor/validation-*.parquet' \
  --model-path /dev/shm/s1d_reader_fp16 \
  --checkpoint runs/natural_merge_v9/full_v10.pt \
  --anchor runs/anchor_residual/anchor_only_rank64.pt \
  --v11-evict runs/cross_block/cross_lr.pt \
  --v13-report reports/cross_block/eval_v16.json \
  --cuda-device 0 --out reports/cross_block/diagnose_v13_oracle.json \
  > reports/cross_block/logs/diagnose_v13_oracle.log 2>&1
echo "ORACLE_EXIT=$? $(date)"
echo CHAIN_DONE
