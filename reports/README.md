# 实验报告索引

仓库里只保留**当前版本（V10）**的结果与证据：主表在 [main_table](main_table/)，
定位实验（1A）在 [diagnostics](diagnostics/)，实验 2 及其反事实消融在
[exp2_source_compaction](exp2_source_compaction/)，Anchor-Only 在 [anchor_only](anchor_only/)，
最新一轮（冻结校准 + 稀疏候选分支）在 [anchor_fusion](anchor_fusion/)。所有数字的汇总表在
[docs/03-results.md](../docs/03-results.md)。

实验 2 的**训练版报告同时是反事实消融和 Anchor-Only 的 `--verify-against` 参考**，所以和 1A 一样
留在仓库里。

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
| [exp2_source_compaction/eval_counterfactual_hotpotqa.json](exp2_source_compaction/eval_counterfactual_hotpotqa.json) | **实验 2 反事实消融**：候选槽位换成另一篇文档的，158 题五臂。反事实 57 vs 训练版 57（0 救 0 伤，p = 1.00），四臂与训练版报告逐题一致 |
| [exp2_source_compaction/probe_candidate_sensitivity.json](exp2_source_compaction/probe_candidate_sensitivity.json) | **机制探针**：7 篇真实文档上把候选 → `context` → `raw` → 残差 → 窗口的衰减逐级量出来（130% → 19% → 16% → 5.0% → 1.5%） |
| [anchor_only/eval_anchor_only_hotpotqa.json](anchor_only/eval_anchor_only_hotpotqa.json) | **Anchor-Only 主结果**：候选路径删掉后的 158 题五臂。`AnchorOnly` 56 vs `Fusion` 57（6 救 7 伤，p = 1.00），对 `Graft` +15（p = 0.011） |
| [anchor_only/training_anchor_only.json](anchor_only/training_anchor_only.json) | Anchor-Only 训练记录：384 更新逐更新标量 + 8 次验证（峰值 19.3%，收在 17.2%） |
| [anchor_only/logs/train_anchor_only.log](anchor_only/logs/train_anchor_only.log) | Anchor-Only 训练控制台日志 |
| [anchor_fusion/eval_anchor_fusion_hotpotqa.json](anchor_fusion/eval_anchor_fusion_hotpotqa.json) | **实验 3 主结果**：冻结 `AnchorOnly` + 只读 Top-4 候选的分支，158 题六臂。`AnchorFusion` 50 vs `AnchorOnly` 56（10 救 16 伤，p = 0.33）；读**别人**候选的 `Donor` 反而 53（丢弃块证据 40 题上 40.0% vs 32.5%） |
| [anchor_fusion/training_fusion_top4.json](anchor_fusion/training_fusion_top4.json) | 实验 3 训练记录：384 更新逐更新标量 + 8 次验证；含初始逐位一致、冻结 digest 前后比对、梯度到达检查 |
| [anchor_fusion/logs/train_fusion_top4.log](anchor_fusion/logs/train_fusion_top4.log) | 实验 3 训练控制台日志 |
| [anchor_fusion/eval_anchor_fusion_key_safe_projected_hotpotqa.json](anchor_fusion/eval_anchor_fusion_key_safe_projected_hotpotqa.json) | **实验 4**：实验 3 权重 + Key-Safe 投影（冻结 Reader 的 key 子空间，不重训），158 题六臂。key 变化 5.3e-4（原来 1.63e-1）；`AnchorFusion` 54 vs `AnchorOnly` 56（5 救 7 伤，p = 0.77），真候选与无关候选**完全等价**（0 救 0 伤，预测只差 3/158） |
| [anchor_fusion/eval_anchor_fusion_crossover_hotpotqa.json](anchor_fusion/eval_anchor_fusion_crossover_hotpotqa.json) | **实验 5**：Key/Value 通道交换，158 题八臂。只换 key 复现 34/38 条预测变更、只换 value 22 条；p_gold 主效应 key:value ≈ 15:1。构造误差与恒等对照同为 1.9e-3（fp16 地板） |

更早的报告（v2 A/B/C、v4 8k/16k、v6/v7、S1d 系列、public retention、定位实验 1/2/3，以及实验
1B 的全部原始文件）已从仓库移出，数字在 [docs/03-results.md](../docs/03-results.md)。
定位实验用 `git show d366d91:reports/diagnostics/<name>_hotpotqa.json` 取回，1B 用
`git show 32d970d:reports/exp1b_fusion/<name>` 取回。

报告里保留了当时的 `identity`（checkpoint sha256、数据 sha256、entrypoint sha256）与
`protocol`（预算、几何、提示词、评分口径）。**评测结论只以报告内记录的口径为准**，
跨报告的百分比不能直接相减。
