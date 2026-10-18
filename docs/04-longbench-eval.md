# 04 — LongBench 评测（native50 基线 + PRLMC v5 双/三臂）

## 协议文档（活跃，原位）

- 基线：experiments/LongBench-MemoryLLM-8B-native50.md——只评测 MemoryLLM-8B checkpoint 原生 50-block 容量（`--capacity-blocks` 仅接受 50），官方 QA F1 主指标，逐题结果完整保存。
- 方法：experiments/LongBench-PRLMC-v5.md（当前口径，架构 grouped_gated_sparse_slot_v5）与 experiments/LongBench-PRLMC-v4.md（结果记录，v4 checkpoint 已不可被新代码加载）——20-block 托管预算内 19→9 slot 融合（Dsup，逐槽位门控上限 0.20、初始 0.02，全局 RMS 预算 0.05），与 Selection 臂同轴对照；入口 tools/longbench_prlmc_eval.py。
- checkpoint 必须显式传入：`--arm SharedMerge --checkpoint <Dsup.train.pt>`；`--arm Selection` 传 checkpoint 直接报错。协议里记录 `architecture`/`method`/`variant` 与 `checkpoint_sha256`，Selection 臂这三项为 None。
- 两者共用的固定协议：6 任务（HotpotQA、2WikiMQA、MuSiQue、Qasper、MultiFieldQA-en、NarrativeQA）、输入 8192/16384 tokens、尾部 2048 作 query、greedy、每样本重置 seed 0 与初始 memory、不训练。

## 评分口径

每题同时记录 official_qa_f1 与 official_qa_f1_truncated（第一个换行前截断）。MemoryLLM 解码器会在答案后继续生成追问文本，raw F1 被解码伪影压低；汇总以 truncated 为主指标，qa_f1_raw 附带。**跨论文比较必须确认对方口径。**

## 8k 首轮结果（2026-10-16，口径已更正）

reports/longbench_prlmc_v4_8k_analysis.md 的原表把 native50 列写成了 `official_qa_f1_truncated`，实际存的是 raw `official_qa_f1`：082937 那次基线用的工具版本逐题不记录 truncated（200 条样本的键里根本没有该字段，hotpotqa 8.83 恰好等于 raw 均值）。**2026-10-18 用当前工具的截断函数对已存 `prediction` 重新打分，得到同口径数字**（n 与原表一致，按 id 去重）：

| 任务 | native50 8k | Dsup 8k | Δ(Dsup−native50) | 原表写的 native50（raw） |
|---|---|---|---|---|
| hotpotqa | 33.63 | 35.20 | +1.57 | 8.83 |
| 2wikimqa | 27.74 | 24.97 | −2.77 | 10.82 |
| musique | 17.99 | 20.55 | +2.56 | 5.85 |
| qasper | 25.90 | 24.76 | −1.14 | 10.72 |
| multifieldqa_en | 33.82 | 35.56 | +1.74 | 18.17 |
| narrativeqa | 20.47 | 19.81 | −0.66 | 3.16 |

即：**原表「+26.4 / +14～17」全是 raw 与 truncated 的口径差**，同口径下 8k 是 ±2 以内、且有正有负。

**8k 长度下不发生任何融合。** `--total-tokens` 是输入上限，逐样本写前缀为 `min(total_tokens−2048, 该样本长度)`，压缩仅在写前缀 ≥ 10240 tokens（20 block × 512）时触发。8k 的前缀上限是 6144，因此 `compression_count` 恒为 0：runs_results/ 与 runs/longbench_prlmc_v4/ 下全部 35 个 8k 文件、3151 条样本，`compression_count` 最大值为 0。逐题比对：hotpotqa 100/100、2wikimqa 200/200、musique 200/200 的 `prediction` 在两臂间**逐字相同**。所以 8k 上 Selection 与 Dsup 只差在原「问题 3」描述的现象，原因是两臂走同一代码路径，而非打分无区分度。

## native50 基线协议复核（2026-10-18，issue #14 结案）

原「问题 2：native50 绝对水平异常偏低（hotpotqa 8.83 vs 公开 ≈33+）」是口径误读，不是协议 bug。复核证据（`runs/longbench_native50_v5metric_20261018_1050/`，hotpotqa，16k，n=200，greedy，seed 0）：

| 运行 | truncated | raw |
|---|---|---|
| 082937 旧 run（用当前截断函数重新打分） | 35.80 | 9.44 |
| 新 run，默认重置协议 | 35.80 | 10.34 |
| 新 run，`--reset-memory`（`initialized.zero_()`） | 34.32 | 8.91 |

- truncated 35.80 与公开 MemoryLLM-8B LongBench 区间一致，**基线没有异常**；raw 8.83/9.44 之所以低，是 MEMORYLLM 解码器在答案后继续生成追问文本（见「评分口径」）。
- `--reset-memory` **不提升**（−1.48），memory pool 初始化假设否定；该 flag 对协议无必要，也不再作为复核动作。
- 重跑与旧 run 逐题比对：raw 预测 184/200 相同，**truncated 打分逐题 200/200 相同、配对差 0.00**。基线生成存在轻微非确定性（raw 层面 ~0.9 点），但主指标 truncated 稳定；200 样本下同协议重跑的 truncated 波动视为 <1 点，Δ<1 点应结合配对胜负计数解读。

同法重算 native50 全部 6 任务的 truncated（8k / 16k，n=mfq 150、其余 200）：

| 任务 | native50 8k | native50 16k |
|---|---|---|
| hotpotqa | 33.63 | 35.80 |
| 2wikimqa | 27.74 | 27.71 |
| musique | 17.99 | 20.00 |
| qasper | 25.90 | 25.77 |
| multifieldqa_en | 33.82 | 33.63 |
| narrativeqa | 20.47 | 20.43 |

## 16k 三臂对照（真实增益口径）

runs/longbench_prlmc_v4_16k/ 与上面的 native50 16k 同口径（`official_qa_f1_truncated`，%，n=200；g7 混入的 2 条 hotpotqa 重启样本不计入）：

| 任务 | native50 | Selection | Dsup | Dsup−native50 | Dsup−Selection |
|---|---|---|---|---|---|
| hotpotqa | 35.80 | 35.01 | 37.84 | +2.04 | +2.83 |
| 2wikimqa | 27.71 | 26.33 | 27.17 | −0.54 | +0.85 |
| musique | 20.00 | 21.03 | 24.61 | +4.61 | +3.58 |
| qasper | 25.77 | 25.44 | 25.52 | −0.25 | +0.07 |
| narrativeqa | 20.43 | 21.85 | 22.01 | +1.58 | +0.15 |
| 均值 | 25.94 | 25.93 | 27.43 | **+1.49** | **+1.50** |

读法：**20-block 托管预算下 Selection 与不压缩的 native50 打平（25.93 vs 25.94）**，即预算压缩本身几乎无损；Dsup 相对两者各 +1.5 均值，收益集中在 musique（+4.6/+3.6）与 hotpotqa（+2.0/+2.8），2wikimqa 与 qasper 在噪声内或略负。这是当前方法可主张的真实增益量级，不能引用 +26 一类数字。

融合子集（16k，`compression_count>0`，直接由已存 run 计算，无需重跑）：

| 任务 | 触发样本 / 200 | Sel 子集 | Dsup 子集 | Δ 子集 | Δ 全集 |
|---|---|---|---|---|---|
| hotpotqa | 124 | 35.76 | 40.32 | +4.56 | +2.83 |
| 2wikimqa | 25 | 21.89 | 28.67 | +6.78 | +0.85 |
| musique | 191 | 20.96 | 24.71 | +3.75 | +3.58 |
| qasper | 4 | 46.26 | 49.95 | +3.69 | +0.07 |
| narrativeqa | 151 | 21.63 | 21.83 | +0.20 | +0.15 |
| 合计 | 495 | — | — | **+3.02（加权）** | — |

结论修正（此前一版写「16k 多数样本不足 10240 前缀、根本不进融合」是错的）：hotpotqa / musique / narrativeqa 的大多数样本确实进融合（124/191/151），只有 2wikimqa（25）和 qasper（4）覆盖很低。**两臂在未触发融合的样本上 `prediction` 逐字相同**（hotpotqa 76/76、2wikimqa 175/175、musique 9/9、qasper 196/196、narrativeqa 49/49），两臂的 `compression_count` 逐样本一致，因此全集 Δ 完全由融合子集贡献；按融合子集加权，Dsup 相对同预算 Selection 的真实效果是 **+3.02**。

逐题配对（Dsup−Selection）：musique +3.58（21 胜/14 负）、hotpotqa +2.83（22/18）、narrativeqa +0.15（27/25）、2wikimqa +0.85（3/2，多数并列）、qasper +0.07。16k 相对 8k（Dsup）：musique +4.06、narrativeqa +2.20、hotpotqa +2.64、2wikimqa +2.20、qasper +0.76；因为 8k 不触发融合，这组差值才是融合的真实效果。

## v5 重跑范围（规划中，未执行）

v5 把载体选择层从 0 改到 16，Selection 臂与 Dsup 臂必须同时重跑，否则不同轴：

1. ~~native50 基线按原协议重跑~~——已完成（上一节），基线无异常，`--reset-memory` 无效，不再需要基线侧动作；
2. v5 训练（`tools/natural_merge_v4_train.py --plan data/natural_merge_v4/train_v5.json`，变体 `Dsup`，产出 `Dsup.train.pt`）——v5 协议变了，旧的 `data/natural_merge_v4/train.json`（sha256 `0f78a212…`）会被 `validate_v4_plan` 拒绝，必须用按 v5 `TRAIN_V4` 重新封装的 plan（`train_v5.json`，content_sha256 `958ef644…`）；
3. 16k 的 Selection / SharedMerge 双臂全量重跑，口径与 v4 表一致（`official_qa_f1_truncated`），并额外报告融合子集（`compression_count>0`）的分数与样本数，与 native50 16k 同表；
4. 8k 不再重跑两臂对照：该长度下 `compression_count` 恒为 0，两臂逐字相同，只需保留作为「无融合对照」。若要扩大融合覆盖面，应提高 `--total-tokens`（例如 32768，需要先放开工具的长度选择项），而不是重跑 8k。

## 下一步

1. 扫描 Selection / Dsup 在**融合子集**上的分数（现有 16k run 已存 `compression_count`，可直接算，不必重跑）；
2. 生成退化检查：抽 20 个低分样本看 raw 输出，确认 verbose 模式；
3. v5 训练完成后按上面第 3 项重跑双臂，与 native50 16k 同表出数。

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
