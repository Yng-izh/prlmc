# LongBench PRLMC v5 协议

评测口径与 [LongBench-PRLMC-v4.md](LongBench-PRLMC-v4.md) 完全相同：6 任务、输入 8192 / 16384 tokens、尾部 2048 作 query、greedy、每样本重置 seed 0 与初始 memory、模块冻结、`official_qa_f1_truncated` 为汇总主指标。v5 只换 checkpoint 与显式传参方式。

## 与 v4 的三点差异

1. **架构名**：`grouped_gated_sparse_slot_v5`（v4 名已从运行时注册表移除，旧 checkpoint 加载即报错）。
2. **checkpoint 显式传入**：`--arm SharedMerge` 必须给 `--checkpoint <Dsup.train.pt>`；`--arm Selection` 传 checkpoint 直接报错。协议里记录 `checkpoint_sha256` 与 `architecture`/`method`/`variant`，Selection 臂这三项为 `None`。
3. **Selection 臂必须重跑**：v5 把 `consolidate._SELECT_LAYER` 与 `layout._REFERENCE_LAYER` 由 0 改为 16，与路由层对齐；沿用 v4 的 Selection 结果就与 Dsup 臂不同轴。

## 命令

```bash
cd /share/latent/PRLM-C

# Selection 臂（不传 checkpoint）
nohup /share/latent/mllm_env/bin/python tools/longbench_prlmc_eval.py \
  --data-dir /share/latent/benchmarks/LongBench/data \
  --out runs/longbench_prlmc_v5_16k/<task>__Selection \
  --total-tokens 16384 --cuda-device N --arm Selection --tasks <task> --resume &

# SharedMerge 臂
nohup /share/latent/mllm_env/bin/python tools/longbench_prlmc_eval.py \
  --data-dir /share/latent/benchmarks/LongBench/data \
  --out runs/longbench_prlmc_v5_16k/<task>__SharedMerge \
  --total-tokens 16384 --cuda-device N --arm SharedMerge \
  --checkpoint runs/<v5 train>/Dsup.train.pt --tasks <task> --resume &
```

`--resume` 依据 `(checkpoint, merge_arm)` 比对身份，不再依赖硬编码 sha。

## 判读

- 主表：`official_qa_f1_truncated`，n=200，两臂同轴相减；
- **必须同时报告融合子集**：压缩仅在逐样本写前缀 ≥ 10240 tokens（20 block × 512）时触发，16k 下多数样本文档不够长，根本不进融合路径；总表 Δ 因此被大量「恒等样本」稀释。融合子集（`compression_count>0`）的分数、样本数、两臂触发数是否一致，都是必报项；
- 配对胜负按题逐条统计，Qasper/NarrativeQA 是 v4 的抵消点，v5 是否把它们变成净正收益是主判据；
- 8k 只作无融合对照：该长度下 `compression_count` 恒为 0，两臂 prediction 逐字相同（已实测），跑两臂对照没有信息量。要扩大融合覆盖面应提高输入上限，而不是重跑 8k。

## 当前状态

代码就绪，GPU 未跑。v4 的 8k/16k 数字（docs/04）仍是现行结果记录，v5 结果出来后另行更新。
