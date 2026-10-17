# 04 — LongBench 评测（native50 基线 + PRLMC v4 三臂）

## 协议文档（活跃，原位）

- 基线：experiments/LongBench-MemoryLLM-8B-native50.md——只评测 MemoryLLM-8B checkpoint 原生 50-block 容量（`--capacity-blocks` 仅接受 50），官方 QA F1 主指标，逐题结果完整保存。
- 方法：experiments/LongBench-PRLMC-v4.md——20-block 托管预算内 19→9 slot 融合（Dsup，grouped_gated_sparse_slot_v4，Dsup.train.pt sha256 70733390…，局部门控上限 0.20、全局 RMS 预算 0.05），与 Selection 臂同轴对照；入口 tools/longbench_prlmc_eval.py。
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

三个核心问题（详见报告原文，数字不改写）：

1. f1_raw 只有 truncated 的约 1/3（9–19）：生成 verbose/hallucination，截断指标掩盖退化。
2. **native50 绝对水平异常偏低**（hotpotqa 8.83 vs 公开 MemoryLLM-8B 报告 ≈33+）：tail2048_native_write_v1 写入方式与 native50 定义（每 50 token native write）不符，且指向 **memory pool 初始化/attention mask 协议 bug**，不是压缩方法失败。此协议对齐问题即 **issue #14** 的主题；复核动作：用 native50 原始协议重跑 100 样本 hotpotqa 对齐公开数字。
3. Selection 臂（733/1150 部分）与 Dsup 几乎完全一致（同任务差 <1pt）：方法区分度缺失——importance 打分无区分度，或两臂走了相同代码路径，需 audit merge block 集合。

## 下一步（报告原文优先级序）

1. audit 单样本 Selection vs Dsup 的 merge block 集合，确认两臂真正分化；
2. 协议复核：native50 原始协议重跑 100 样本 hotpotqa（预期 ≈33）；
3. 生成退化检查：抽 20 个低分样本看 raw 输出；
4. 三臂全量对比表。

## 证据位置

runs/longbench_native50_20261015_082937/、runs/longbench_prlmc_v4/、runs_results/（逐题 JSON，dsup8k__* 等 27 项）。

## 后续评测方案（规划中，未执行）

- experiments/Evaluation-plan-benchmarks-baselines.md——三 benchmark（LongBench 6-QA / LooGLE / LongBench v2 Medium）× 五外部 baseline（MemoryLLM-8B、M+、H₂O、KIVI、ReST-KV）的统一预算（50 blocks / 3.125 GiB 真实历史存储）主表与消融副表方案；M+ 与 ReST-KV 的容量公平性验证未完成。
