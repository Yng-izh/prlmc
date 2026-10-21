# v7 逐题归因：同路径 Full / Selection / SharedMerge 三臂对比（hotpotqa 96 题）

日期：2026-10-21 | 代码基线：`ee0a939` + 本次 Full 臂接入
产物：
- `runs/longbench_prlmc_v7_eval/Full_hotpotqa.json`（新增，同路径无压缩参照，96 题全量完成）
- `runs/longbench_prlmc_v7_eval/per_sample_diff_hotpotqa.json`（逐题报告，阈值 F1>0）
- `runs/longbench_prlmc_v7_eval/per_sample_diff_hotpotqa_exact.json`（逐题报告，阈值 F1>0.99）
- 工具：`tools/per_sample_diff.py`（本文件所有逐题结论由它生成）

---

## 0. 结论

1. **同路径 Full 臂 = 35.30%，Selection = 33.25%，headroom 点估计 +2.05pp**（配对 bootstrap 95% CI **[−3.59pp, +8.36pp]**，W/L/T = 9/11/76，n=96）。
2. **这不是"天花板为零"，而是"n=96 测不出来"**：点估计 +2.05pp 与 v5 宣称的 +1.70pp 同量级。此前 `experiments/v7-ceiling-finding.md` 的 +0.10pp 结论是**跨路径参照**（native writer + default 内存初始化）造成的，那条路径本身比同路径 Full 低 **1.95pp**，详见该文件新增的 D7。
3. **逐题层面，SharedMerge 只救回 6 题可恢复信息中的 1 题**（idx 23），另外 5 题全部漏掉（idx 12/36/44/58/87）。
4. **同时它损伤 11 题**（Selection 本来有分、SharedMerge 变 0），其中 9 题 Full 也答对。**按"F1>0 算答对"计，净效果 −10 题**。
5. **失效形态不是"保留的文本变差"，而是"编出一个来自别的段落的实体"**：损伤题的 SharedMerge 输出多为另一个真实存在的人名/地名/公司名（`Ayrton Senna`、`Suryanar Kovil`、`North American Company`、`Spain`、`Christmas`），不是语法崩坏。这与 v7 训练遥测一致（`residual_ratio` 长期贴在 5% 上限、`similarity_reject_fraction` 0.607）。
6. **Full 自身也会错**：5 题 Selection 有分而 Full 为 0（idx 29/52/74/81 及 1 例被归入 damaged），说明"全保留"不是严格上界，28 块的上下文本身带来干扰。

---

## 1. 协议

| 项 | 值 |
|---|---|
| 任务 | hotpotqa，LongBench test 前 96 题（与 Selection/SharedMerge 同一批 `sample_index` 0–95） |
| 输入 | 16384 token 上限，tail 2048 直查，写前缀 ≤ 14336 token |
| 内存初始化 | 每样本 `reset()`：checkpoint 值 + `used=0` + 全 active mask（三臂完全一致） |
| Selection / SharedMerge | budget 20，几何 13→9，触发 `used>=14` |
| **Full** | `--arm Full`：**不压缩**，容量为原生 50 块物理池；`compression_count=0` 全部 96 题 |
| 配对键 | `(task, id, sample_index)`；三臂 `input_token_sha256` 逐题一致（已抽查） |
| 口径 | `official_qa_f1_truncated` |

同路径证据：Selection 与 Full 的 `context_sha256`、`input_token_sha256`、`input_tokens`、`write_count` 逐题相同；唯一差异是 `final_active_slots`（例：idx 0 Full 4864 vs Selection 2816）。

---

## 2. 三臂总体

| 臂 | truncated F1 | 相对 Selection | W/L/T（F1 级） |
|---|---|---|---|
| Selection | 33.25% | — | — |
| **Full（同路径无压缩）** | **35.30%** | **+2.05pp**，CI [−3.59, +8.36] | 9/11/76 |
| SharedMerge v7 | 32.78% | −0.47pp，CI [−6.37, +5.52] | 11/16/69 |

headroom 配对 sd = 0.3049 ⇒ 80% power 检测 1.7pp 需 **n ≈ 2525**。96 题的 CI 宽度 ±6pp，**不足以判定**。

---

## 3. 逐题：Full 对而 Selection 错（可恢复信息集合，6 题）

| idx | 问题（截断） | 标准答案 | Selection | Full | SharedMerge |
|---|---|---|---|---|---|
| 23 | South Park 第 18 季最后两集的客串是谁 | PewDiePie | `Mira Jantunen` 0.00 | `PewDiePie` 1.00 | **`PewDiePie` 1.00 → 救回** |
| 12 | 1964 年某片女主角（导演也拍过 Vera Cruz / Kiss Me Deadly） | Mary Astor | `Joan Crawford` 0.00 | `Mary Astor` 1.00 | `Shirley Temple` 0.00 → 漏 |
| 36 | Rich Lee 做过 3-D 预可视化的片中，Francis Lawrence 导演的是哪部 | Constantine | `Mockingjay – Part 1` 0.00 | `Constantine` 1.00 | `Mockingjay – Part 1` 0.00 → 漏 |
| 44 | Allen Donaldson 采用的、与 60s 黑豹运动相关的名字 | Hakim Abdullah Jamal | `Maulana Ndabezitha Karenga` 0.00 | `Hakim Jamal` 0.80 | 同上 0.00 → 漏 |
| 58 | Luther: The Calling 所依据的 BBC 剧（六集）播出年份 | 2010 | `2015` 0.00 | `2010` 1.00 | 空串 0.00 → 漏 |
| 87 | Primetime Race Group 所在迈阿密都会区城市 | Hollywood, Florida | `Miami` 0.00 | `Hollywood, Florida` 1.00 | `Miami` 0.00 → 漏 |

**救回率 1/6**。6 题中 5 题的压缩次数 ≥3（idx 12/36/44/58/87 均为 4 次，idx 23 为 2 次），可恢复信息确实集中在被反复压缩的样本上。

---

## 4. 逐题：SharedMerge 损伤（Selection 有分、SharedMerge 为 0，11 题）

| idx | 标准答案 | Selection F1 | Full F1 | SharedMerge |
|---|---|---|---|---|
| 6 | Tharangambadi | 0.17 | 0.17 | `Suryanar Kovil` 0.00 |
| 8 | 2013 | 1.00 | 1.00 | `2005` 0.00 |
| 24 | It's Always Sunny in Philadelphia | 1.00 | 1.00 | `Robbie Tucker` 0.00 |
| 34 | Troy, Montana | 0.15 | 0.40 | `Siletz` 0.00 |
| 35 | Mika Häkkinen | 1.00 | 1.00 | `Ayrton Senna` 0.00 |
| 38 | Xcel Energy Inc. | 0.29 | 0.80 | `North American Company` 0.00 |
| 39 | Summer Magic | 0.21 | 0.21 | `Around the World in 80 Days` 0.00 |
| 41 | seasonal TV specials / stop motion | 0.21 | 0.21 | `Christmas` 0.00 |
| 57 | Floyd Casey Stadium | 0.21 | 0.00 | `Air Force` 0.00 |
| 68 | India | 1.00 | 1.00 | `Spain` 0.00 |
| 92 | no | 0.13 | 0.20 | `Yes` 0.00 |

其中 **9 题 Full 也答对**（6/8/24/34/35/38/39/41/68）——这些是"两个参照都对、只有融合错"的纯损伤。
另外 **1 题**（idx 80，Lev Yilmaz / Pamela B. Green）Selection 与 Full 都错而 SharedMerge 对，是唯一 gain。

---

## 5. 逐题：压缩反而更好（Selection 对而 Full 错，5 题，互斥分类后 4 题）

| idx | 标准答案 | Selection | Full | SharedMerge |
|---|---|---|---|---|
| 29 | University of Southern California | 0.36 | `the Coliseum, ... USC` 0.00 | 0.62 |
| 52 | yes | 1.00 | `No` 0.00 | 1.00 |
| 74 | Morgan Llywelyn | 0.19 | `Robert Jordan` 0.00 | 1.00 |
| 81 | Man Haron | 0.80 | `Chris Reason` 0.00 | 0.80 |

Full 在这些题上被 28 块里的干扰段落带偏。**"无压缩"是参照，不是上界**：任何"效应不得超过 headroom"的判据都必须考虑这一点（本文件用配对 CI 表述，而非硬上界）。

第 5 例是 **idx 57**（已列入 §4 损伤表）：Selection 0.21、Full 0.00、SharedMerge 0.00，同时满足 `compression_helped` 与 `damaged`；互斥分类把一题只记一次，所以 §5 表里是 4 行、§4 里是 11 行。

---

## 6. 两个阈值下的对照

| 判对标准 | both 对 | both 错 | 可恢复（Full 对/Sel 错） | 救回 | 漏掉 | gain | damage | 净题数 |
|---|---|---|---|---|---|---|---|---|
| `F1 > 0`（任何词重叠） | 37 | 37 | 6 | 1 | 5 | 1 | 11 | **−10** |
| `F1 > 0.99`（近似完全匹配） | 10 | 67 | 5 | 1 | 4 | 7 | 6 | **+2** |

两种口径方向相反：SharedMerge 的赢多是**部分匹配**（0.00 → 0.5），输多是**从 1.00 直接掉到 0**。所以"净题数"和"平均 F1"必须一起看：平均 F1 −0.47pp，近似完全匹配净 +2 题。

---

## 7. 对主线的含义

- **"可恢复信息为零"这个此前的判断不成立**：同路径 Full 比 Selection 高 2.05pp（点估计），且可恢复题集中出现在高压缩样本上。真实障碍是**评测功效**（需 n≈2525）与**算子实现**（救回率 1/6、损伤 11 题）。
- **下一步最有信息量的实验不是继续在 n=96 上改架构**，而是：
  1. 把 96 题扩到能判定 2pp 的样本量（hotpotqa 全量 200 题 + 2wikimqa/musique 补足 ~2500 配对样本），先用 Full/Selection 定住 headroom 的真值；
  2. 在这批配对样本上做**损伤/救回的定向消融**：损伤 9 题（两参照都对、融合独错）说明融合写入了错误实体，应优先检查 `carrier_indices` 选对但 `source_allow_mask`/`residual` 写错的情形；
  3. 把"救回率 + 损伤率"作为算子的一级指标，而不是只看任务均值 Δ。

---

## 8. 复现

```bash
# 同路径无压缩参照（不加载任何 merge 模块，不压缩）
python tools/longbench_prlmc_eval.py --arm Full \
  --data-dir /share/latent/benchmarks/LongBench/data \
  --tasks hotpotqa --max-samples 96 \
  --total-tokens 16384 --tail-tokens 2048 --cuda-device 8 \
  --out runs/longbench_prlmc_v7_eval/Full_hotpotqa.json

# 天花板裁决
python tools/ceiling_report.py \
  --selection runs/longbench_prlmc_v7_eval/Selection_hotpotqa.json \
  --ceiling   runs/longbench_prlmc_v7_eval/Full_hotpotqa.json \
  --treatment runs/longbench_prlmc_v7_eval/SharedMerge_hotpotqa.json \
  --allow-partial

# 逐题
python tools/per_sample_diff.py \
  --selection runs/longbench_prlmc_v7_eval/Selection_hotpotqa.json \
  --ceiling   runs/longbench_prlmc_v7_eval/Full_hotpotqa.json \
  --treatment runs/longbench_prlmc_v7_eval/SharedMerge_hotpotqa.json \
  --data-dir /share/latent/benchmarks/LongBench/data --allow-partial \
  --out runs/longbench_prlmc_v7_eval/per_sample_diff_hotpotqa.json
```
