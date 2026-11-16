#!/bin/bash
cd /data/zhangjiayang/yyz/PRLM-C || exit 1
PY=/data/zhangjiayang/yyz/env/bin/python
COMMON="--data-dir /data/zhangjiayang/yyz/datasets/LongBench/extracted/data --train-task musique --model-path /dev/shm/s1d_reader_fp16 --checkpoint runs/natural_merge_v9/full_v10.pt --anchor runs/anchor_residual/anchor_only_rank64.pt --updates 384 --cuda-device 0 --seed 0"
echo "=== train cross (--cross-lr 3e-3) $(date) ==="
$PY tools/train_cross_block_calibration.py $COMMON --cross on --cross-lr 3e-3 \
  --out runs/cross_block/cross_lr.pt --report runs/cross_block/cross_lr.json \
  > reports/cross_block/logs/train_cross_lr.log 2>&1
echo "TRAIN_EXIT=$? $(date)"
echo "=== eval $(date) ==="
$PY tools/eval_cross_block_calibration.py \
  --data-dir /data/zhangjiayang/yyz/datasets/LongBench/extracted/data --task hotpotqa \
  --supporting-facts-glob '/data/zhangjiayang/yyz/datasets/hotpot_qa/distractor/validation-*.parquet' \
  --model-path /dev/shm/s1d_reader_fp16 \
  --checkpoint runs/natural_merge_v9/full_v10.pt \
  --anchor runs/anchor_residual/anchor_only_rank64.pt \
  --v11 runs/cross_block/plain.pt --v11-evict runs/cross_block/cross_lr.pt \
  --verify-against reports/anchor_fusion/eval_anchor_fusion_hotpotqa.json \
  --cuda-device 0 --out reports/cross_block/eval_v13_lr.json \
  > reports/cross_block/logs/eval_v13_lr.log 2>&1
echo "EVAL_EXIT=$? $(date)"
echo CHAIN_DONE
