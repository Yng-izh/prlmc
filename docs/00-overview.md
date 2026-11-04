# 00 — 目标、主张与当前版本

## 冻结的目标与主张

- **目标**：不大改论文主线方案，通过调节内部架构实现具有强泛化性的稳定涨点。
- **主张**：在相同、有限的潜在记忆容量下，通过跨块信息融合，提高长文档中知识的保留和可读取能力。
- **核心指标**：`Ours − Selection`。两臂必须共享写入流、载体块、容量、解码与评分，唯一差别是压缩窗口是否经过融合算子。
- **评测方式**：算子训练后完全冻结，不做逐场景训练；在公开原始任务上逐题生成并评分。

## 当前版本：V10 保流形槽位嫁接

读侧（MemoryLLM-8B）冻结，只有压缩算子可训练。压缩窗口内保留 9 个载体块，
每个输出块 = 216 个原生核心槽 + 40 个嫁接槽；嫁接槽是同块内被丢弃原生槽的
测地压缩（保范数），每次 merge 重新推导。设计见 [01-method-v10.md](01-method-v10.md)。

## 仓库里只有这一版

v1–v9 各代实现（`natural_merge_v2/v3/v4/v6/v7/v8` 训练器、目标、计划构建器、
S0/S1/S1c/S1d 诊断线、reader LoRA / bridge K/V adapter、public retention 协议、
旧 LongBench 8k/16k 报告）已从仓库删除，不再维护，也不再保留兼容入口。
需要历史实现时从 git 历史取回。

## 命名说明

V10 的算子文件是 `methods/prlmc/manifold_graft.py`，但训练计划 schema、架构名和
checkpoint 记录里仍是 `natural_merge_v9_training_plan/1`、`grouped_gated_sparse_native_v9`、
`DnativeV9`。这些字符串被 sealed plan 的 `content_sha256` 和 checkpoint 的 `training`
记录钉死，改名会让 `full_v10.pt` 无法通过校验，所以保持原样。
