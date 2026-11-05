# 实验报告索引

仓库里只保留**当前版本（V10）**的结果与证据：主表在 [main_table](main_table/)，
定位实验（1A）在 [diagnostics](diagnostics/)，最新一轮（1B）在
[exp1b_fusion](exp1b_fusion/)。所有数字的汇总表在 [docs/03-results.md](../docs/03-results.md)。

| 文件 | 内容 |
| --- | --- |
| [memoryllm8b_lbv2_all_128k.json](main_table/memoryllm8b_lbv2_all_128k.json) | MemoryLLM-8B 原生基线，LongBench v2 全 503 题，50 块池不压缩，官方 judge 18.7% |
| [eval_v10_50b_n20.json](main_table/eval_v10_50b_n20.json) | V10，20 题，cue/option 口径 20.0% |
| [eval_selection_50b_n20.json](main_table/eval_selection_50b_n20.json) | Selection（同载体），20 题，cue/option 口径 20.0% |
| [full_v10.training.json](main_table/full_v10.training.json) | V10 训练记录（704 更新的逐更新标量 + 计划/模型/资源元数据） |
| [logs/full_v10.log](main_table/logs/full_v10.log) | 训练日志 |
| [diagnostics/graft_span_hotpotqa.json](diagnostics/graft_span_hotpotqa.json) | 实验 1A（HotpotQA 158 题配对，也是后续实验的 `--verify-against` 参考） |
| [exp1b_fusion/eval_fusion_hotpotqa.json](exp1b_fusion/eval_fusion_hotpotqa.json) | **1B 主结果**：4 层融合 vs 原生嫁接，HotpotQA 158 题六臂 + 32 层逐层读出距离。`Fusion` 23.42 vs `Graft` 25.95（1 救 5 伤，p = 0.22） |
| [exp1b_fusion/eval_perlayer_all32.json](exp1b_fusion/eval_perlayer_all32.json) | 32 层 checkpoint 的 2 题逐层诊断：前 13 层被放大 1.1–4.8 倍（发散证据） |
| [exp1b_fusion/training/fusion_rank32.json](exp1b_fusion/training/fusion_rank32.json) | 1B 融合模块，只监督 8/16/24/31 四层：384 更新逐更新标量 + 8 次验证 |
| [exp1b_fusion/training/fusion_all32.json](exp1b_fusion/training/fusion_all32.json) | 1B 融合模块，监督全部 32 层：**发散**（验证集 −28%） |
| [exp1b_fusion/logs/](exp1b_fusion/logs/) | 上面两次训练的控制台日志 |

更早的报告（v2 A/B/C、v4 8k/16k、v6/v7、S1d 系列、public retention，以及定位实验 1/2/3 的
原始 json）已从仓库移出，数字在 [docs/03-results.md](../docs/03-results.md)，
文件用 `git show d366d91:reports/diagnostics/<name>_hotpotqa.json` 或到服务器取。

报告里保留了当时的 `identity`（checkpoint sha256、数据 sha256、entrypoint sha256）与
`protocol`（预算、几何、提示词、评分口径）。**评测结论只以报告内记录的口径为准**，
跨报告的百分比不能直接相减。
