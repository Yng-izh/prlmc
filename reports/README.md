# 实验报告索引

仓库里只保留**当前版本（V10）**的结果与证据：主表在 [main_table](main_table/)，
最新的定位实验（1A）在 [diagnostics](diagnostics/)。所有数字的汇总表在
[docs/03-results.md](../docs/03-results.md)。

| 文件 | 内容 |
| --- | --- |
| [memoryllm8b_lbv2_all_128k.json](main_table/memoryllm8b_lbv2_all_128k.json) | MemoryLLM-8B 原生基线，LongBench v2 全 503 题，50 块池不压缩，官方 judge 18.7% |
| [eval_v10_50b_n20.json](main_table/eval_v10_50b_n20.json) | V10，20 题，cue/option 口径 20.0% |
| [eval_selection_50b_n20.json](main_table/eval_selection_50b_n20.json) | Selection（同载体），20 题，cue/option 口径 20.0% |
| [full_v10.training.json](main_table/full_v10.training.json) | V10 训练记录（704 更新的逐更新标量 + 计划/模型/资源元数据） |
| [logs/full_v10.log](main_table/logs/full_v10.log) | 训练日志 |
| [diagnostics/graft_span_hotpotqa.json](diagnostics/graft_span_hotpotqa.json) | 实验 1A（HotpotQA 158 题配对，最新一轮定位实验，也是后续实验的 `--verify-against` 参考） |

更早的报告（v2 A/B/C、v4 8k/16k、v6/v7、S1d 系列、public retention，以及定位实验 1/2/3 的
原始 json）已从仓库移出，数字在 [docs/03-results.md](../docs/03-results.md)，
文件用 `git show d366d91:reports/diagnostics/<name>_hotpotqa.json` 或到服务器取。

报告里保留了当时的 `identity`（checkpoint sha256、数据 sha256、entrypoint sha256）与
`protocol`（预算、几何、提示词、评分口径）。**评测结论只以报告内记录的口径为准**，
跨报告的百分比不能直接相减。
