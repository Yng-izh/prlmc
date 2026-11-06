# 实验报告索引

仓库里只保留**当前版本（V10）**的结果与证据：主表在 [main_table](main_table/)，
定位实验（1A）在 [diagnostics](diagnostics/)，最新一轮（实验 2，来源端浓缩）在
[exp2_source_compaction](exp2_source_compaction/)。所有数字的汇总表在
[docs/03-results.md](../docs/03-results.md)。

| 文件 | 内容 |
| --- | --- |
| [memoryllm8b_lbv2_all_128k.json](main_table/memoryllm8b_lbv2_all_128k.json) | MemoryLLM-8B 原生基线，LongBench v2 全 503 题，50 块池不压缩，官方 judge 18.7% |
| [eval_v10_50b_n20.json](main_table/eval_v10_50b_n20.json) | V10，20 题，cue/option 口径 20.0% |
| [eval_selection_50b_n20.json](main_table/eval_selection_50b_n20.json) | Selection（同载体），20 题，cue/option 口径 20.0% |
| [full_v10.training.json](main_table/full_v10.training.json) | V10 训练记录（704 更新的逐更新标量 + 计划/模型/资源元数据） |
| [logs/full_v10.log](main_table/logs/full_v10.log) | 训练日志 |
| [diagnostics/graft_span_hotpotqa.json](diagnostics/graft_span_hotpotqa.json) | 实验 1A（HotpotQA 158 题配对，也是后续实验的 `--verify-against` 参考） |
| [exp2_source_compaction/eval_trained_hotpotqa.json](exp2_source_compaction/eval_trained_hotpotqa.json) | **实验 2 主结果**：来源端浓缩 vs 原生嫁接，158 题四臂。`Fusion` 36.08 vs `Graft` 25.95（23 救 7 伤，p = 0.005） |
| [exp2_source_compaction/eval_random_control_hotpotqa.json](exp2_source_compaction/eval_random_control_hotpotqa.json) | **随机对照**：同结构、同残差幅度、不训练。26.58 vs 25.95（1 救 0 伤，p = 1.00） |
| [exp2_source_compaction/training_compaction.json](exp2_source_compaction/training_compaction.json) | 训练记录：384 更新逐更新标量 + 8 次验证（验证集 CE 消掉 20.5% 见顶） |
| [exp2_source_compaction/training_random_control.json](exp2_source_compaction/training_random_control.json) | 随机对照的标定记录（目标 0.26362831 / 实际 0.26362832、缩放因子） |
| [exp2_source_compaction/logs/train_compaction.log](exp2_source_compaction/logs/train_compaction.log) | 训练控制台日志 |

更早的报告（v2 A/B/C、v4 8k/16k、v6/v7、S1d 系列、public retention、定位实验 1/2/3，以及实验
1B 的全部原始文件）已从仓库移出，数字在 [docs/03-results.md](../docs/03-results.md)。
定位实验用 `git show d366d91:reports/diagnostics/<name>_hotpotqa.json` 取回，1B 用
`git show 32d970d:reports/exp1b_fusion/<name>` 取回。

报告里保留了当时的 `identity`（checkpoint sha256、数据 sha256、entrypoint sha256）与
`protocol`（预算、几何、提示词、评分口径）。**评测结论只以报告内记录的口径为准**，
跨报告的百分比不能直接相减。
