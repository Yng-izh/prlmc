# 工具入口（V10 单一版本）

| 文件 | 用途 |
| --- | --- |
| `io_utils.py` | `sha` / `digest` / `atomic_json` / `tokenizer_manifest` |
| `plan.py` | sealed 训练计划的构建与校验；`sealed_training()` 供 checkpoint 校验读取封存记录 |
| `objective.py` | V10 训练目标：恢复 CE、Full 教师 KL、保护 hinge、首 token 锚定、EOS；含可微注入用到的共享辅助 |
| `train.py` | 训练入口（704 更新，`--resume`、原子保存、滚动 `_latest.pt`） |
| `runtime.py` | 流式运行时：`MergeStream`、`install`、`load_runtime`、几何解析、checkpoint 校验 |
| `stream_eval.py` | 流式记忆与参考臂（`Selection` / `Recent` / `Full` / `NativeDefault`） |
| `v9_geometry_50block.py` | 50 块几何常量（`GEOMETRY_50B`、`TRIGGER_BLOCKS`、`DISTILL_LAYERS`） |
| `eval_lbv2.py` | LongBench v2 双口径评测（主表入口） |
| `eval_core.py` | LongBench 8k/16k 评测（v2 复用其运行时与协议记录） |
| `lbv2_data.py` | LBv2 数据加载、官方模板与官方答案抽取 |
| `native_eval.py` | MemoryLLM-8B 原生基线评测（50 块池、不压缩） |

`v9_geometry_50block.py` 的文件名是历史遗留：`methods/prlmc/consolidate.py` 被 checkpoint 哈希锁定，
它直接 `from tools.v9_geometry_50block import GEOMETRY_50B`，因此该文件名不能改。

## 训练

```bash
python tools/train.py --plan data/natural_merge_v9_50block/train.json \
  --model-path /dev/shm/s1d_reader_fp16 --cuda-device 0 \
  --variant DnativeV9 --qa-batch 1 \
  --out reports/main_table/full_v10.json \
  --checkpoint-out runs/natural_merge_v9/full_v10.pt
```

常用开关：`--smoke-updates N`（冒烟）、`--preflight`（6 个事件的结构检查）、
`--save-every N`（小 checkpoint 间隔）、`--resume <ckpt>`（权重+优化器+池状态续跑）、
`--no-reader-checkpointing`（显存够时关掉重算）。

## 评测

```bash
# V10 臂
python tools/eval_lbv2.py --arm SharedMerge \
  --checkpoint runs/natural_merge_v9/full_v10.pt \
  --data-json /data/.../LongBench-v2/data.json \
  --budget-blocks 50 --cuda-device 0 --out reports/main_table/<v10>.json

# Selection 臂：复现 V10 的载体序列，唯一差别是没有融合
python tools/eval_lbv2.py --arm Selection \
  --carrier-schedule reports/main_table/<v10>.json \
  --data-json /data/.../LongBench-v2/data.json \
  --budget-blocks 50 --cuda-device 0 --out reports/main_table/<selection>.json
```

`--max-samples N` 跑子集；`--rescore` 只重算已有报告的评分（不跑模型）；
`--strict-code-hash` 默认开启，算子文件与 checkpoint 记录不一致会直接报错。

## 问题定位（A/B/C/D）

```bash
# C：只有最后一次压缩融合（前面全部 Selection，且沿用 B 的载体序列）
python tools/eval_lbv2.py --arm SharedMerge --fusion-policy last \
  --carrier-schedule reports/main_table/<B>.json --checkpoint runs/natural_merge_v9/full_v10.pt \
  --data-json <lbv2/data.json> --budget-blocks 50 --max-samples 20 --cuda-device 0 --out <C>.json

# D：同 C，但算子只做原生槽位替换（不做凝聚/正交/Core 强化）
python tools/eval_lbv2.py --arm SharedMerge --fusion-policy last --operator-variant native ...

# 不跑模型，直接对比已有报告的逐题选项概率
python tools/diagnose_ab.py <A>.json <B>.json --labels Selection V10
```

每个样本会额外记录 `last_merge`（最后一次融合丢弃/保留了哪些写入块、覆盖文档的哪一段、
正确选项的措辞是否出现在被丢弃的原文里）。设计说明见 [docs/04-diagnostics.md](../docs/04-diagnostics.md)。
