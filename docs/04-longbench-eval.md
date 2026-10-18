# 04 — LongBench 评测（native50 基线 + PRLMC v5 双/三臂）

## 协议文档（活跃，原位）

- 基线：experiments/LongBench-MemoryLLM-8B-native50.md——只评测 MemoryLLM-8B checkpoint 原生 50-block 容量（`--capacity-blocks` 仅接受 50），官方 QA F1 主指标，逐题结果完整保存。
- 方法：experiments/LongBench-PRLMC-v5.md（当前口径，架构 grouped_gated_sparse_slot_v5）与 experiments/LongBench-PRLMC-v4.md（结果记录，v4 checkpoint 已不可被新代码加载）——20-block 托管预算内 19→9 slot 融合（Dsup，逐槽位门控上限 0.20、初始 0.02，全局 RMS 预算 0.05），与 Selection 臂同轴对照；入口 tools/longbench_prlmc_eval.py。
- checkpoint 必须显式传入：`--arm SharedMerge --checkpoint <Dsup.train.pt>`；`--arm Selection` 传 checkpoint 直接报错。协议里记录 `architecture`/`method`/`variant` 与 `checkpoint_sha256`，Selection 臂这三项为 None。
- 两者共用的固定协议：6 任务（HotpotQA、2WikiMQA、MuSiQue、Qasper、MultiFieldQA-en、NarrativeQA）、输入 8192/16384 tokens、尾部 2048 作 query、greedy、每样本重置 seed 0 与初始 memory、不训练。

## 评分口径

每题同时记录 official_qa_f1 与 official_qa_f1_truncated（第一个换行前截断）。MemoryLLM 解码器会在答案后继续生成追问文本，raw F1 被解码伪影压低；汇总以 truncated 为主指标，qa_f1_raw 附带。**跨论文比较必须确认对方口径。**

## 8k 首轮结果与问题（2026-10-16）

reports/longbench_prlmc_v4_8k_analysis.md（official_qa_f1_truncated，%）：

| 任务 | native50 8k | Dsup 8k | Δ |
|---|---|---|---|
| hotpotqa | 8.83 | 35.20 | +26.4 |
| 2wikimqa | 10.82 | 24.97 | +14.2 |
| musique | 5.85 | 20.55 | +14.7 |
| qasper | 10.72 | 24.76 | +14.0 |
| multifieldqa_en | 18.17 | 35.56 | +17.4 |
| narrativeqa | 3.16 | 19.81 | +16.7 |

**关键更正（2026-10-18，逐题证据复核）：8k 长度下不发生任何融合。** `--total-tokens` 是输入上限，逐样本写前缀为 `min(total_tokens−2048, 该样本长度)`，压缩仅在写前缀 ≥ 10240 tokens（20 block × 512）时触发。8k 的前缀上限是 6144，因此 `compression_count` 恒为 0：runs_results/ 与 runs/longbench_prlmc_v4/ 下全部 35 个 8k 文件、3151 条样本，`compression_count` 最大值为 0。直接比对逐题输出：hotpotqa 100/100、2wikimqa 200/200、musique 200/200 的 `prediction` 在两臂间**逐字相同**，F1 也相同（34.51 / 24.97 / 20.55）。

因此上表的 Δ 列**不是 Dsup 增益**，而是「托管预算内存协议（不融合）」与 native50 写入协议的差异；「16k − 8k」的差值是融合的真实效果（8k 侧等于无融合对照）。原「问题 3（Selection 与 Dsup 几乎一致）」到此结案：不是打分无区分度，而是 8k 下两臂走完全相同的代码路径。

三个核心问题（数字不改写）：

1. f1_raw 只有 truncated 的约 1/3（9–19）：生成 verbose/hallucination，截断指标掩盖退化。
2. **native50 绝对水平异常偏低**（hotpotqa 8.83 vs 公开 MemoryLLM-8B 报告 ≈33+）：tail2048_native_write_v1 写入方式与 native50 定义（每 50 token native write）不符，且指向 **memory pool 初始化/attention mask 协议 bug**，不是压缩方法失败。此协议对齐问题即 **issue #14** 的主题；复核动作：用 native50 原始协议重跑 100 样本 hotpotqa 对齐公开数字。注意该表两侧都发生在无融合条件下，因此差距不能归给压缩方法。
3. Selection 臂与 Dsup 几乎完全一致：已由上面的证据结案为「8k 不触发融合」。

## 下一步（报告原文优先级序）

1. audit 单样本 Selection vs Dsup 的 merge block 集合，确认两臂真正分化；
2. 协议复核：native50 原始协议重跑 100 样本 hotpotqa（预期 ≈33）；
3. 生成退化检查：抽 20 个低分样本看 raw 输出；
4. 三臂全量对比表。

## v5 重跑范围（规划中，未执行）

v5 把载体选择层从 0 改到 16，Selection 臂与 Dsup 臂必须同时重跑，否则不同轴：

1. native50 基线按原协议重跑 hotpotqa（`--reset-memory --total-tokens 16384`），确认绝对水平异常偏低是协议问题还是方法问题；
2. v5 训练（`tools/natural_merge_v4_train.py`，变体 `Dsup`，产出 `Dsup.train.pt`）；
3. 16k 的 Selection / SharedMerge 双臂全量重跑，口径与 v4 表一致（`official_qa_f1_truncated`），并额外报告融合子集（`compression_count>0`）的分数与样本数；
4. 8k 不再重跑两臂对照：该长度下 `compression_count` 恒为 0，两臂逐字相同，只需保留作为「无融合对照」。若要扩大融合覆盖面，应提高 `--total-tokens`（例如 32768，需要先放开工具的长度选择项），而不是重跑 8k。

## 证据位置

runs/longbench_native50_20261015_082937/、runs/longbench_prlmc_v4/、runs/longbench_prlmc_v4_16k/、runs_results/（逐题 JSON，dsup8k__* 等 27 项）。

## 16k 结果（2026-10-17，12 run 全部完成）

runs/longbench_prlmc_v4_16k/（official_qa_f1_truncated，%，n=200；g7 混入的 2 条 hotpotqa 重启样本不计入）：

| 任务 | Selection 16k | Dsup 16k | Δ(Dsup−Sel) | 8k Dsup |
|---|---|---|---|---|
| hotpotqa | 35.01 | 37.84 | +2.83 | 35.20 |
| 2wikimqa | 26.33 | 27.17 | +0.85 | 24.97 |
| musique | 21.03 | 24.61 | +3.58 | 20.55 |
| qasper | 25.44 | 25.52 | +0.07 | 24.76 |
| narrativeqa | 21.85 | 22.01 | +0.15 | 19.81 |

融合触发比例（16k，按 shard 统计 `compression_count>0` 的样本数）：触发条件是写前缀 ≥ 10240 tokens，观测到的最小触发前缀 10339、未触发样本的最大前缀 10205，边界与 20 block × 512 完全吻合；同一任务的两臂触发数一致（如 g0 Selection 25 与 g7 SharedMerge 25、g4 Selection 4 与 g0 SharedMerge 4）。16k 下多数 LongBench 样本文档不足 10240 write tokens，**根本不进融合路径**——这是 Δ 偏小的直接原因，也是后续必须额外报告「融合子集」指标的原因。

逐题配对（Dsup−Selection）：musique +3.58（21 胜/14 负）、hotpotqa +2.83（22/18）、narrativeqa +0.15（27/25）、2wikimqa +0.85（3/2，多数并列）、qasper +0.07。16k 相对 8k：musique +4.06、narrativeqa +2.20、hotpotqa +2.64、2wikimqa +2.20、qasper +0.76（Dsup 全线高于 8k）；多跳任务收益最大，Qasper/NarrativeQA 接近饱和。

## 后续评测方案（规划中，未执行）

- experiments/Evaluation-plan-benchmarks-baselines.md——三 benchmark（LongBench 6-QA / LooGLE / LongBench v2 Medium）× 五外部 baseline（MemoryLLM-8B、M+、H₂O、KIVI、ReST-KV）的统一预算（50 blocks / 3.125 GiB 真实历史存储）主表与消融副表方案；M+ 与 ReST-KV 的容量公平性验证未完成。
