# PRLM-C 文档重组方案（t3，待队长确认后由 t5 执行）

原则：只移动/归档，不删除 runs/reports 证据；旧文档正文不改写，只做搬移与索引；原始数字与结论逐字保留。

## 一、目标结构

```
docs/
  REORG-PLAN.md                  本文件（执行后保留作重组记录）
  00-overview.md                 导览：研究时间线 + 现主线入口（新建）
  01-problem-discovery.md        问题发现线索引（E0–E7、SIC，只留结论索引，新建）
  02-method-exploration.md       方法探索线索引（S0/S0b/S1/S1c/S1d/G0–G2，全部 NO-GO/停线，新建）
  03-natural-merge.md            自然融合主线 v1→v2→v3 C/D（新建，指向活跃协议）
  04-longbench-eval.md           LongBench 评测：native50 基线 + PRLMC-v4 三臂 + 基线协议对齐（含 issue #14 引用，新建）
  05-public-retention.md         public retention 主线（新建）
archive/docs/                    旧线正文归档（git mv，内容不动）
  EXPERIMENTS.md                 根目录旧索引（E0–E7/SIC/S1 线）
  BASELINE.md                    MemoryAgentBench baseline 冻结记录
  S1-soft-local.md … S1d-* 等 16 篇旧线协议（自 experiments/ 移入）
```

新结构放 `docs/`，README.md 更新"文档导航"指向 docs/00–05 与 archive/docs/。

## 二、重要事实约束（执行前必须知晓）

1. **旧索引引用的正文大部分不存在**：EXPERIMENTS.md 目录表引用 `experiments/E0.md`、`E4.md`、`S0.md`、`S1-SDC.md`、`G0.md`、`E7_PLAN.md`、`experiments/params.md`、`fixes.md`、`open-questions.md`、`reproduce.md` 等，**磁盘上均不存在**（experiments/ 现存仅 21 篇 .md，全部列于 §四）。01/02 两个索引文档的"正文"列只能写「正文缺失，证据见 runs/ 与 reports/」，不得伪造链接。
2. `experiments_check_tmp.md/`（目录）内容与根目录 EXPERIMENTS.md 逐字节相同（已 diff 确认 SAME），属 t1 删除类，由 t5 按删除清单处置，不参与归档。
3. runs/（含 natural_merge_initial_v1.results.md）、reports/（含 longbench_prlmc_v4_8k_analysis.md、s1d_round_20261013/、natural_merge_implementation_20261013/）一律原位不动，索引里以只读引用提及。

## 三、新文档内容要点（摘要级，全部结论摘自原文，不改写）

- **00-overview**：一句话目标（冻结 MemoryLLM reader 上的共享残差压缩）；时间线：问题发现（E0–E7/SIC，2026-09）→ 方法探索（S0–S1c/G 系列，10 月上旬，全部停线）→ 自然融合 v1/v2/v3（10 月 13–14）→ LongBench v4（10 月 15）→ public retention；现主线 = 自然融合 v3 C/D（待运行）+ LongBench PRLMC v4 + public retention SharedE6d。
- **01-problem-discovery**：E1 触发器不触发 → E2 标定 → E3 accuracy 全 0 → E4 compactor 是充分原因 → E5a–d 原生 block 选择恢复至 EM 26 → E6/E6a-c 训练链 → E6-gate FAIL（3/6）→ E6d interleave +30 → E6e 漂移 → E7a-sim 停止门槛 1；SIC 线（SIC-O1/O2/P1）与 Native-QA/Readout-probe（B2 few-shot+stop EM 30→42）只列结论；每条注明「正文缺失，证据见 runs/、outputs/」。
- **02-method-exploration**：S0 PASS → S0b INCONCLUSIVE → S1-local NO-GO → S1-soft-local NO-IMPROVEMENT → S1c-wash PASS（单窗口）→ S1c-geometry 共向 → S1c-global-oracle FAIL（停线）→ S1c-lowo-coverage 窗口差异机制解释 → S1d 两线 v1/v2 结果 → G0–G2c 各门结论；现有正文（S1-soft-local.md、S1c-global-oracle.md、S1c-lowo-coverage.md、S1d-* 11 篇）移 archive/docs/ 后从索引指向归档路径。
- **03-natural-merge**：v1 未过门（runs/natural_merge_initial_v1.results.md）→ v2 A/B/C 全部未过推进门槛（reports/natural_merge_v2 已有结果引文在 README）→ v3 C/D 为当前待运行方案（experiments/Natural-merge-v3-CD.md 保持原位活跃）。
- **04-longbench**：experiments/LongBench-MemoryLLM-8B-native50.md（基线协议）+ experiments/LongBench-PRLMC-v4.md（三臂）+ reports/longbench_prlmc_v4_8k_analysis.md（8k 分析）+ 基线协议对齐结论与 issue #14 引用；runs/longbench_native50_20261015_082937、runs/longbench_prlmc_v4、runs_results/ 为证据原位。
- **05-public-retention**：experiments/Public-retention-frozen-SharedE6d.md 保持原位活跃，索引页给入口与证据位置。

## 四、文件移动清单（全部 git mv，保持历史）

移入 `archive/docs/`：
- EXPERIMENTS.md（根目录）→ archive/docs/EXPERIMENTS.md
- BASELINE.md（根目录）→ archive/docs/BASELINE.md
- experiments/ 下 16 篇旧线（自 t1 §四归档类）：Natural-cross-block-merge-design.md、Natural-merge-v2-ABC.md、S1-soft-local.md、S1c-global-oracle.md、S1c-lowo-coverage.md、S1d-C-dimensions-v1.md、S1d-C-risk-protection-v1.md、S1d-controlled-C13-development.md、S1d-experiment1-fact-cross.md、S1d-experiment1-fact-v3.md、S1d-experiment1-v3-fleet.md、S1d-native-write-drop-development.md、S1d-projected-C-review.md、S1d-shared-E6d-baselines-training.md、S1d-shared-compactor-zero-training.md、S1d-two-lines-v1.md、S1d-two-lines-v2-results.md（17 篇，实际按现存清单执行）
- 根目录 LongBench-MemoryLLM-8B-native50.md（与 experiments/ 同名重复，t1 判归档/删除）→ archive/docs/LongBench-MemoryLLM-8B-native50-root-copy.md（diff 确认后如与 experiments/ 版本相同则按删除清单处理）

保持原位（活跃）：README.md、experiments/{Natural-merge-v3-CD,LongBench-PRLMC-v4,LongBench-MemoryLLM-8B-native50,Public-retention-frozen-SharedE6d}.md、tools/README.md、reports/ 全部、runs/ 全部。

## 五、执行顺序（t5）

1. `git mv` §四清单 → archive/docs/。
2. 新建 docs/00–05 六篇索引（只写摘要与指针，不复制大段正文）。
3. 改 README.md：加"文档导航"段（指向 docs/00–05、archive/docs/、reports/README 证据边界）；更新"仓库结构"里 experiments/ 描述为"当前主线协议 + docs/ 索引 + archive/docs/ 历史"。
4. 校验：grep 全仓库对 `experiments/S1-`、`experiments/S1d-`、`experiments/Natural-merge-v2`、`EXPERIMENTS.md E` 的引用，逐条更新指向新路径（archive/docs/… 或结论索引节锚点）；确认无死链（相对路径存在性检查）。
5. 不得动：runs/、runs_results/、reports/、logs/、outputs/、data/、work/。

## 六、待确认（沿用 t1 §七）

- 根目录 LongBench-MemoryLLM-8B-native50.md 与 experiments/ 版本 diff 后：相同→删，不同→归档。
- EXPERIMENTS.md 中被引用但缺失的正文（E0–E7/S0/G 系列等）：确认是否存在于其他机器/备份；如确认丢失，01/02 索引中标注「正文缺失」即可，不重建。
