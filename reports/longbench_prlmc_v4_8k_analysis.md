# LongBench PRLM-C v4 8k — 结果与问题分析 (2026-10-16)

## 结果总表 (official_qa_f1_truncated, %)

| 任务 | native50 8k | Dsup 8k | Δ |
|---|---|---|---|
| hotpotqa | 8.83 | 35.20 | +26.4 |
| 2wikimqa | 10.82 | 24.97 | +14.2 |
| musique | 5.85 | 20.55 | +14.7 |
| qasper | 10.72 | 24.76 | +14.0 |
| multifieldqa_en | 18.17 | 35.56 | +17.4 |
| narrativeqa | 3.16 | 19.81 | +16.7 |

Selection 臂（部分, 733/1150）与 Dsup 几乎完全一致（同任务差 <1pt）——方法区分度缺失。

## 主线核心问题：指标"上不去"的根本原因

### 1. 绝对水平远低于 LongBench 正常区间
MemoryLLM/Losing 原论文 8k 下 hotpotqa ≈ 30-45, 我们 Dsup = 35.2 看似正常；
但 **f1_raw 只有 9-19**（truncated 的 1/3），说明大量输出是长废话后被截断救回。
模型在协议下生成不受控（verbose/hallucination），截断指标掩盖了真实退化。

### 2. 为什么 native50（不压缩基线）这么低 → 关键异常
native50 8k hotpotqa 只有 **8.83**，而公开 LongBench MemoryLLM-8B 报告 ≈ 33+。
这**不是我们方法的失败，而是协议/加载路径的问题**：
- tail2048_native_write_v1 把 prefix 全部"写进" memory pool，与 native50 定义（每 50 token native write 一次）不符；
- 检查点加载路径（Dsup.train.pt gamma_max=0.05）与 native 权重不同但 native50 也低，指向 **memory pool 初始化/attention mask 协议 bug**，而非压缩策略问题。

### 3. Selection ≈ Dsup（差 <1pt）→ 选择机制未生效
Selection 臂应只在重要 block 上做 merge；结果几乎不变，说明：
- 20-block 预算下 Selection 的 importance 打分没有区分度（全部 block 同权）；
- 或 Selection 与 Dsup 走了相同代码路径（需要 audit arm 分支是否真正被调用）。

### 4. 下一步（优先级序）
1. audit: 在 1 个样本上 print Selection vs Dsup 的 merge block 集合，确认两臂真正分化；
2. 协议复核: 用 native50 原始协议重跑 100 样本 hotpotqa，对齐公开数字（预期 ≈33）；
3. 生成退化检查: 抽 20 个低分样本看 raw 输出，确认 verbose/重复模式；
4. 三臂全量对比表（Selection 跑完后自动出）。
