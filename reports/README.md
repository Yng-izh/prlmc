# 实验报告索引

仓库里只保留**当前版本（V10）**的结果与证据，全部在 [main_table](main_table/)。

| 文件 | 内容 |
| --- | --- |
| [memoryllm8b_lbv2_all_128k.json](main_table/memoryllm8b_lbv2_all_128k.json) | MemoryLLM-8B 原生基线，LongBench v2 全 503 题，50 块池不压缩，官方 judge 18.7% |
| [eval_v10_lbv2_50b_SharedMerge.partial.json](main_table/eval_v10_lbv2_50b_SharedMerge.partial.json) | V10，50 块预算，157/503 题（中断），cue 口径 18.5% |
| [eval_v10_50b_n20.json](main_table/eval_v10_50b_n20.json) | V10，20 题，cue/option 口径 20.0% |
| [eval_selection_50b_n20.json](main_table/eval_selection_50b_n20.json) | Selection（同载体），20 题，cue/option 口径 20.0% |
| [full_v10.training.json](main_table/full_v10.training.json) | V10 训练记录（704 更新的逐更新标量 + 计划/模型/资源元数据） |
| [logs/full_v10.log](main_table/logs/full_v10.log) | 训练日志 |
| [logs/eval_v10_lbv2_50b.log](main_table/logs/eval_v10_lbv2_50b.log) | 157 题那次评测的日志 |

更早的报告（v2 A/B/C、v4 8k/16k、v6/v7、S1d 系列、public retention）已从仓库删除，
需要时从 git 历史取回。

报告里保留了当时的 `identity`（checkpoint sha256、数据 sha256、entrypoint sha256）与
`protocol`（预算、几何、提示词、评分口径）。**评测结论只以报告内记录的口径为准**，
跨报告的百分比不能直接相减。
