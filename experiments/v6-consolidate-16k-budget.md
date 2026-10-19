# 方案A（Consolidate）16k 三任务 + 预算压力曲线评测

日期：2026-10-19 | 协议：tail2048_native_write_v1，16k，seed 0，greedy，n=200/任务
臂：Selection（无合并）vs Consolidate（方案A免训练槽位级择优重排+凸组合合并，零训练参数）
指标：official_qa_f1_truncated；融合子集 = Consolidate 侧 compression_count>0 的样本
配对统计：bootstrap 95% CI，2000 resamples，seed 0（融合子集 paired Δ）

## 主对比（20-block 默认预算）

| 任务 | Selection full | Consolidate full | 融合 n | Sel fused | Con fused | Δfused | W/L/T | CI95 |
|---|---|---|---|---|---|---|---|---|
| hotpotqa | 35.01 | 35.57 | 124 | 35.76 | 36.67 | +0.91 | 17/14/93 | [-4.95,+6.49] |
| musique  | 21.03 | 21.48 | 191 | 20.96 | 21.43 | +0.47 | 25/24/142 | [-3.33,+4.13] |
| narrativeqa | 21.85 | 21.43 | 151 | 21.63 | 21.07 | -0.57 | 18/25/108 | [-3.18,+2.11] |

## 预算压力曲线（hotpotqa）

| budget-blocks | 臂 | full | 融合 n | Sel fused | Con fused | Δfused | W/L/T | CI95 |
|---|---|---|---|---|---|---|---|---|
| 20 | Sel / Con | 35.01 / 35.57 | 124 | 35.76 | 36.67 | +0.91 | 17/14/93 | [-4.95,+6.49] |
| 12 | Sel / Con | 31.42 / 31.79 | 171 | 30.00 | 30.43 | +0.43 | 31/26/114 | [-5.03,+5.76] |
| 8  | Sel / Con | 32.43 / 31.23 | 178 | 30.90 | 29.55 | -1.35 | 27/25/126 | [-5.57,+2.84] |

预算压力现象：预算 20→12 时两臂 full 均下降约 3.5-4 分；8-block 时 Selection 反而回升至 32.43
（压缩次数减少意味着更少的信息被丢弃，budget geometry 8=(9,4) 窗口更保守），Consolidate 在极低预算下
融合子集 Δ 转负（-1.35），τ=0.99 凸组合合并未能在强记忆危机下提供保护。

## 决策门判定

判据：Consolidate 相对同预算 Selection 的融合子集 Δ 是否优于 v4 的 +3.02？
**结论：否。** 三任务 + 两个压力点共 5 组对比，融合子集 Δ 为 +0.91 / +0.47 / -0.57 / +0.43 / -1.35，
全部 95% CI 跨 0，无一显著，均远低于 v4 的 +3.02。方案A 在 16k 协议下未复现 v4 级别的融合子集优势：
hotpotqa/musique 小幅正、narrativeqa 小幅负，预算压力下低预算段反而退化。

与 v5（t2）对照：v5 SharedMerge 融合子集 Δ hotpotqa +2.74、narrativeqa +0.99（均不显著）；
方案A 免训练路线 Δ 更小，方向一致性弱于 v5。

## 产物
- runs/longbench_prlmc_v6_eval/16k__Consolidate_{hotpotqa,musique,narrativeqa}.json
- runs/longbench_prlmc_v6_eval/16k__Selection_musique.json（20-block 基线）
- runs/longbench_prlmc_v6_eval/16k__{Selection,Consolidate}_hotpotqa_b{12,8}.json
- runs/longbench_prlmc_v6_eval/t7_stats.json（机读统计）
- 每个输出文件独立，无共享写（t2 教训落实）
