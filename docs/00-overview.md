# 00 — 导览：PRLM-C 研究时间线

**目标**：在冻结的 MemoryLLM reader 上，学习有界、内容相关的共享残差压缩模块，将多个记忆块中的信息融合进保留的载体；在公开原始任务上以相同有效记忆槽位预算改善知识保留与读取；评测时模块完全冻结，不进行逐场景训练。

## 研究时间线（按文档编号阅读）

| 阶段 | 索引 | 内容 | 结局 |
|---|---|---|---|
| 问题发现（2026-09 下旬–10-03） | 01-problem-discovery.md | MemoryAgentBench E0–E7、SIC、Native-QA/Readout 诊断线 | 定位压缩与读出瓶颈；压缩算子侧变体停止 |
| 方法探索（2026-10-03–10-13） | 02-method-exploration.md | S0/S0b、S1-local/S1-soft-local、S1c 系列、S1d 两线、G0–G2c | 全部 NO-GO / FAIL / 停线 |
| 自然融合主线（2026-10-13 起） | 03-natural-merge.md | v1 → v2 A/B/C → v3 C/D → v4 Dsup | v1/v2/v3 未过门槛或未启动；v4 Dsup 为当前方案 |
| LongBench 评测（当前公开基准） | 04-longbench-eval.md | native50 基线 + PRLMC-v4 Dsup/Selection 臂，8k/16k | 8k 全量完成；16k 进行中 |
| public retention | 05-public-retention.md | 冻结 SharedE6d 公开原始知识保持 | 协议冻结，GPU 未启动 |

## 当前主线（活跃）

- 自然融合 v4 Dsup（grouped_gated_sparse_slot_v4）：experiments/Natural-merge-v4-D-supervision-native.md
- LongBench PRLMC v4：experiments/LongBench-PRLMC-v4.md
- LongBench native50 基线协议：experiments/LongBench-MemoryLLM-8B-native50.md

v1/v2 A/B/C、v3 C/D、S 系列、public retention 均为历史线，不再活跃。

## 证据与归档

- 证据原位不动（不入 git）：runs/、runs_results/、reports/、logs/、outputs/、data/。
- 旧线代码与文档归档：archive/2026H1_legacy/（不入 git，磁盘保留）。
- git 仓库只保留当前主线：v4 训练链、LongBench 评测链、公共依赖、docs 00–05 与 experiments 协议。
