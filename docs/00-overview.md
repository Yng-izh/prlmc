# 00 — 导览：PRLM-C 研究时间线

**目标**：在冻结的 MemoryLLM reader 上，学习有界、内容相关的共享残差压缩模块，将多个记忆块中的信息融合进保留的载体；在公开原始任务上以相同有效记忆槽位预算改善知识保留与读取；评测时模块完全冻结，不进行逐场景训练。

## 研究时间线（按文档编号阅读）

| 阶段 | 索引 | 内容 | 结局 |
|---|---|---|---|
| 问题发现（2026-09 下旬–10-03） | 01-problem-discovery.md | MemoryAgentBench E0–E7、SIC、Native-QA/Readout 诊断线 | 定位压缩与读出瓶颈；压缩算子侧变体停止 |
| 方法探索（2026-10-03–10-13） | 02-method-exploration.md | S0/S0b、S1-local/S1-soft-local、S1c 系列、S1d 两线、G0–G2c | 全部 NO-GO / FAIL / 停线；S1c 13D 跨窗口主线停线 |
| 自然融合主线（2026-10-13 起） | 03-natural-merge.md | v1 → v2 A/B/C → v3 C/D | v1/v2 未过推进门槛；v3 C/D 代码完成待运行 |
| LongBench 评测 | 04-longbench-eval.md | native50 基线 + PRLMC-v4 Dsup/Selection 三臂 | 8k 结果与协议对齐问题记录（issue #14） |
| public retention | 05-public-retention.md | 冻结 SharedE6d 公开原始知识保持 | 协议冻结，GPU 未启动 |

## 当前主线（活跃）

- 自然融合 v3 C/D 对照（待运行 GPU）：experiments/Natural-merge-v3-CD.md
- LongBench PRLMC v4：experiments/LongBench-PRLMC-v4.md
- LongBench native50 基线协议：experiments/LongBench-MemoryLLM-8B-native50.md
- public retention 冻结协议：experiments/Public-retention-frozen-SharedE6d.md

## 证据与归档

- 证据原位不动：runs/、runs_results/、reports/、logs/、outputs/、data/。
- 旧线代码与文档归档：archive/2026H1_legacy/（root/、experiments/、tools/ 等子目录）。
- 人工 episode 与逐窗口 C13 已退出主方法，仅保留历史诊断证据。
- 重组记录：REORG-PLAN.md（本目录）。
