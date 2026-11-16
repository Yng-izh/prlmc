# 工具入口

当前主线:V10(基线)、V11Evict、V13、V17-density。实验总表见 [../docs/06-experiment-ledger.md](../docs/06-experiment-ledger.md)。

## 主线入口

| 文件 | 用途 |
| --- | --- |
| `train.py` | V10 训练(704 更新,`--resume`、原子保存) |
| `train_cross_block_calibration.py` | V11Evict / V13 训练。`--cross off` = V11Evict 校准器,`--cross on` = V13。`--architecture v13/v15/v16/v17` 与 `--neighbor-mode topk/complement` 保留了 V14 到 V17 的开关 |
| `eval_cross_block_calibration.py` | HotpotQA 158 题多臂评测(主评测),含 donor 对照、邻居消融、损伤分组、`--verify-against` 逐题复现检查 |
| `eval_lbv2.py` | LongBench v2 评测(128k,四选一),V10 的 Ours/Selection 两臂 |
| `native_eval.py` | MemoryLLM-8B 原生基线(50 块池,不压缩) |
| `run_free_v17_evals.sh` | 不训练的 V17 评测(density / absorb / attn_q50 / attn_q90) |
| `postcheck_free_evals.py` | 检查 V17 评测:V13 必须逐行复现参考报告 |
| `wrap_v13_as_v17.py` | 把 V13 checkpoint 包成 V17 格式(attn 臂用) |
| `diagnose_ab.py` | 读已有 LongBench v2 报告,对比逐题选项概率(不跑模型) |
| `diagnose_v13_oracle.py` | V13 的 oracle 诊断 |

## 基础设施

`runtime.py`(流式运行时、checkpoint 校验)、`stream_eval.py`(参考臂 Selection/Recent/Full)、
`objective.py`(V10 训练目标)、`plan.py`(sealed 训练计划)、`io_utils.py`、`lbv2_data.py`、
`eval_core.py`(LongBench 8k/16k,v2 复用其运行时)、`v9_geometry_50block.py`(50 块几何常量,
被哈希锁定的算子直接 import,文件名不能改)。

## 被 V13 依赖的早期实验工具(保留是因为被 import,不单独使用)

`eval_graft_recovery.py`、`eval_replace_reorder.py`、`eval_learned_fusion.py`、`eval_source_compaction.py`、
`train_core_fusion.py`、`train_source_compaction.py`。它们提供 V13 评测用的臂构造、McNemar 统计和训练辅助函数。
`test_key_safe.py`、`test_band_relation.py` 是 CPU 数学自检。

## 复现 V13 = 67/158

```bash
bash v13_lr_chain.sh
```

等价于:

```bash
PY=/data/zhangjiayang/yyz/env/bin/python
COMMON="--data-dir /data/zhangjiayang/yyz/datasets/LongBench/extracted/data --train-task musique \
  --model-path /dev/shm/s1d_reader_fp16 --checkpoint runs/natural_merge_v9/full_v10.pt \
  --anchor runs/anchor_residual/anchor_only_rank64.pt --updates 384 --cuda-device 0 --seed 0"

# 训练 V13(--cross on);--cross off 得到 V11Evict 校准器
$PY tools/train_cross_block_calibration.py $COMMON --cross on --cross-lr 3e-3 \
  --out runs/cross_block/cross_lr.pt --report runs/cross_block/cross_lr.json

# 评测:前四臂必须逐题复现参考报告,否则整轮作废
$PY tools/eval_cross_block_calibration.py \
  --data-dir /data/zhangjiayang/yyz/datasets/LongBench/extracted/data --task hotpotqa \
  --supporting-facts-glob '/data/zhangjiayang/yyz/datasets/hotpot_qa/distractor/validation-*.parquet' \
  --model-path /dev/shm/s1d_reader_fp16 \
  --checkpoint runs/natural_merge_v9/full_v10.pt \
  --anchor runs/anchor_residual/anchor_only_rank64.pt \
  --v11 runs/cross_block/plain.pt --v11-evict runs/cross_block/cross_lr.pt \
  --verify-against reports/anchor_fusion/eval_anchor_fusion_hotpotqa.json \
  --cuda-device 0 --out reports/cross_block/eval_v13_lr.json
```

`--v13-neighbor-modes random,bottom` 加邻居消融(见 `diag_chain.sh`)。

## V10 训练与 LongBench v2 评测

```bash
python tools/train.py --plan data/natural_merge_v9_50block/train.json \
  --model-path /dev/shm/s1d_reader_fp16 --cuda-device 0 --variant DnativeV9 --qa-batch 1 \
  --out reports/main_table/full_v10.json --checkpoint-out runs/natural_merge_v9/full_v10.pt

python tools/eval_lbv2.py --arm SharedMerge --checkpoint runs/natural_merge_v9/full_v10.pt \
  --data-json <lbv2/data.json> --budget-blocks 50 --cuda-device 0 --out <v10.json>
python tools/eval_lbv2.py --arm Selection --carrier-schedule <v10.json> \
  --data-json <lbv2/data.json> --budget-blocks 50 --cuda-device 0 --out <selection.json>
```

常用开关:`--max-samples N` 子集;`--rescore` 只重算评分;`--strict-code-hash` 默认开启,
算子文件与 checkpoint 记录不一致会直接报错;`--smoke-updates N`、`--preflight`、`--save-every N`。
