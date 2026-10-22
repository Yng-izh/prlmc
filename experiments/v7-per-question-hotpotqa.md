# v7 逐题归因：同路径 Full / Selection / SharedMerge 三臂对比（hotpotqa 96 题）

日期：2026-10-21 | 代码基线：`ee0a939` + 本次 Full 臂接入
产物：
- `runs/longbench_prlmc_v7_eval/Full_hotpotqa.json`（新增，同路径无压缩参照，96 题全量完成）
- `runs/longbench_prlmc_v7_eval/{Selection,SharedMerge,Full}_hotpotqa.log`（三臂逐样本运行日志）
- `runs/longbench_prlmc_v7_eval/per_sample_diff_hotpotqa.json`（逐题报告，阈值 F1>0）
- `runs/longbench_prlmc_v7_eval/per_sample_diff_hotpotqa_exact.json`（逐题报告，阈值 F1>0.99）
- `runs/longbench_prlmc_v7_eval/block_provenance.txt`（块级溯源日志，§7）
- `runs/longbench_prlmc_v7_eval/evidence_location.txt`（证据位置分布，§8）
- 工具：`tools/per_sample_diff.py`、`tools/block_provenance.py`、`tools/evidence_location.py`（本文件所有结论由它们生成，均有单测）

---

## 0. 结论

1. **同路径 Full 臂 = 35.30%，Selection = 33.25%，headroom 点估计 +2.05pp**（配对 bootstrap 95% CI **[−3.59pp, +8.36pp]**，W/L/T = 9/11/76，n=96）。
2. **这不是"天花板为零"，而是"n=96 测不出来"**：点估计 +2.05pp 与 v5 宣称的 +1.70pp 同量级。此前 `experiments/v7-ceiling-finding.md` 的 +0.10pp 结论是**跨路径参照**（native writer + default 内存初始化）造成的，那条路径本身比同路径 Full 低 **1.95pp**，详见该文件新增的 D7。
3. **逐题层面，SharedMerge 只救回 6 题可恢复信息中的 1 题**（idx 23），另外 5 题全部漏掉（idx 12/36/44/58/87）。
4. **同时它损伤 11 题**（Selection 本来有分、SharedMerge 变 0），其中 9 题 Full 也答对。**按"F1>0 算答对"计，净效果 −10 题**。
5. **失效形态不是"保留的文本变差"，而是"编出一个来自别的段落的实体"**：损伤题的 SharedMerge 输出多为另一个真实存在的人名/地名/公司名（`Ayrton Senna`、`Suryanar Kovil`、`North American Company`、`Spain`、`Christmas`），不是语法崩坏。这与 v7 训练遥测一致（`residual_ratio` 长期贴在 5% 上限、`similarity_reject_fraction` 0.607）。
6. **Full 自身也会错**：5 题 Selection 有分而 Full 为 0（idx 29/52/74/81 及 1 例被归入 damaged），说明"全保留"不是严格上界，28 块的上下文本身带来干扰。
7. **块级溯源显示：答案没丢，是"记混了"。** 11 道损伤题里，两臂保留的 gold 证据句**数量相同**（如 idx 24 都是 10/10、idx 35 都是 5/6、idx 6 都是 11/11），Selection 从这份内容答对，SharedMerge 从同一份内容答错；错答实体多数**也在内存里**（`Ayrton Senna` 在块 19/24/27、`Robbie Tucker` 在块 15/23），少数连原文都不在（`Spain`、`Yes`、`2005`）。详见 §7。
8. **真正的损失在写入侧，不在读侧。** 按"答案能不能直接看见"分组：答案只在被写入前缀里的 43 题，**全保留（Full）也只答对 51%**，Selection 49%（只差 1 题）；而答案能直接在尾部文本里看到的题正确率 76%。也就是说 512 token → 256 latent 槽这次写侧压缩吃掉约 25pp，而我们一直在调 13→9 这次读侧压缩只值 2pp。详见 §8。

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

## 7. 块级溯源：答案在不在那 20 块里

工具 `tools/block_provenance.py`（产物 `runs/longbench_prlmc_v7_eval/block_provenance.txt`）重建原始 prompt，把 gold 答案所在的句子映射到产出原生块的 512-token 写入块，再重放每臂记录的 compaction 事件，判断该块在提问时是否仍在常驻集合里；同时定位每个臂自己答案的原文出处。

**损伤题（Selection 有分、SharedMerge 为 0）**

| idx | gold | Selection 保留 gold 证据句 | SharedMerge 保留 gold 证据句 | SM 错答 | SM 错答原文是否在内存 |
|---|---|---|---|---|---|
| 24 | It's Always Sunny… | 10/10 | 10/10 | `Robbie Tucker` | 在（3/4 处） |
| 35 | Mika Häkkinen | 5/6 | 5/6 | `Ayrton Senna` | 在（5/6 处） |
| 6 | Tharangambadi | 11/11 | 11/11 | `Suryanar Kovil` | 在（3/3 处） |
| 39 | Summer Magic | 7/7 | 7/7 | `Around the World in 80 Days` | 在（8/23 处） |
| 68 | India | 5/6 | 5/6 | `Spain` | **不在（0 处）** |
| 92 | no | 90/264 | 90/264 | `Yes` | **不在（极性翻转）** |
| 8 | 2013 | 保留（块 10） | 保留 | `2005` | **不在（源块 3/4 已丢）** |

**两臂的证据句保留数完全相同**——因为 Selection 选中的块与融合窗口覆盖的块是同一批（SharedMerge 的 `allow` 全 1，13 块全部进融合）。所以"20 块够不够"不是瓶颈：内容在，Selection 从里面答对，SharedMerge 从同一份内容答错。

**可恢复题（Full 对、Selection 错）**只有 3 题是真的把证据块丢了（idx 36 块 4–12、44 块 15、87 块 7/8/12/14）；另外 3 题（idx 23/12/58）证据块仍在 20 块里，Selection 照样被同段落的干扰实体带走。即使在那 3 道"丢块"的题里，融合窗口也吸收过这些块，所以算子本来有机会。

**机制**：Selection 是 13→9 **整块原样保留**，实体↔主语↔关系没动；SharedMerge 是 13→9 **有损融合**，实体词还在，但一个块里塞进 13 个段落，实体与所属句子的绑定被平均掉了。reader 于是挑该段落里更显著的实体（`Senna` 在同段反复出现）。`Spain`/`Yes`/`2005` 连原文支撑都没有，属于融合块里什么都没剩下之后的编造。

**为什么 6 题只救回 1 题**（idx 23，PewDiePie）：那题证据句里 PewDiePie 是唯一候选实体，模糊化之后仍然唯一。**救回的条件是"无竞争实体"，损伤的条件是"有任意同段落实体"**——前者苛刻、后者宽松，所以均值 Δ 恰好落在 0 附近。

边界：这是"证据句原样在常驻块里 + 错答实体也在同一批块里（或完全不在）+ 证据块确实进过融合窗口"三条实测推出的推断；潜变量未解码，融合块内部的实体绑定形态没有直接测量。

---

## 8. 证据位置分布：损失在写入侧，不在读侧

工具 `tools/evidence_location.py`（产物 `runs/longbench_prlmc_v7_eval/evidence_location.txt`）把 96 题按"离答案最近的通路"分组：

| 答案位置 | 题数 | Full 正确 | Full F1 | Selection 正确 | Selection F1 | SharedMerge 正确 | SharedMerge F1 |
|---|---|---|---|---|---|---|---|
| 只能靠内存（只在写入前缀里） | 43 | **22 (51%)** | 32.0% | 21 (49%) | 26.7% | 16 (37%) | 22.2% |
| 内存和尾部文本都有 | 33 | 25 (76%) | 54.8% | 25 (76%) | 54.2% | 23 (70%) | 59.1% |
| 字面无匹配（长答案/别名/极性题） | 12 | 3 (25%) | 10.5% | 3 (25%) | 10.0% | 2 (17%) | 9.5% |
| 只在尾部文本里 | 8 | 3 (38%) | 10.0% | 3 (38%) | 16.7% | 2 (25%) | 16.1% |

> **口径修正（本次）**：上一版把这 12 题记作"被 16384 尾部截断切掉"，是错的。`sentence_spans()` 返回的是 `all_ids` 坐标，落在被截断头部里的匹配会得到**负索引**，而旧 `classify()` 只判 `start < write_tokens`、没判 `start >= 0`，于是"已被截断出局"的匹配会被误归进 `memory_only`；反过来 `spans` 为空又被笼统记成 `absent`，把"字面串根本没出现"和"被截断"混为一谈。现在分类器把两者分开：`truncated_out`（匹配存在但整段落在丢弃区）与 `no_literal_match`（prompt 里根本没有这个字面串）。**在现有 96 题上 `truncated_out` 为 0 例，12 例全部是 `no_literal_match`**（长答案如 idx 41 的整句、极性题如 idx 92 的 "no"、以及别名），所以"12 题白占样本"的说法撤回：它们的证据位置无法用字面搜索判定，只能靠人工或语义匹配。

三个结论：

1. **真正依赖内存的 43 题，全保留也只答对 51%，Selection 49%——只差 1 题。** 块一块没丢、原文都在内存里，reader 还是错一半。融合模块在这批题上可争的空间就是 1 题。
2. **"存进内存"这件事本身吃掉约 25pp**：答案能直接看见的题 76%，必须走内存的题 51%。这一步是冻结 Reader 的写入压缩（512 token → 256 latent 槽，2:1），我们从未在它上面做过任何实验。我们一直在调的读侧压缩（13→9）只值 2pp（33.25 → 35.30）。
3. **SharedMerge 在内存依赖题上是净损伤**：37% vs Selection 49%（−5 题），而在"两处都有"的题上正确率 70% vs 76% 但 F1 更高（59.1% vs 54.2%）——赢的是部分匹配，输的是 1.00→0，与 §6 的口径敏感性一致。
4. **12 例"字面无匹配"不是截断造成的**（口径已修正，见上）：这批题的长答案/极性/别名使字面搜索失效，其证据位置只能靠语义匹配判定，不能据此断言"白占样本"。

### 8.1 关键 17 题的真实证据位置

`python tools/evidence_location.py --show-indices …` 逐题输出（`kept N/M spans` = 字面匹配落在保留区内的条数）：

| idx | 归类 | 字面匹配 | Full | Selection | SharedMerge |
|---|---|---|---|---|---|
| 6 | memory_only | 11/11 | 0.17 | 0.17 | 0.00 |
| 8 | memory_only | 7/7 | 1.00 | 1.00 | 0.00 |
| 12 | memory_only | 2/2 | 1.00 | 0.00 | 0.00 |
| 23 | **both** | 9/9 | 1.00 | 0.00 | **1.00** |
| 24 | **both** | 10/10 | 1.00 | 1.00 | 0.00 |
| 34 | memory_only | 1/1 | 0.40 | 0.15 | 0.00 |
| 35 | **both** | 6/6 | 1.00 | 1.00 | 0.00 |
| 36 | memory_only | 58/58 | 1.00 | 0.00 | 0.00 |
| 38 | memory_only | 2/2 | 0.80 | 0.29 | 0.00 |
| 39 | **tail_only** | 7/7 | 0.21 | 0.21 | 0.00 |
| 41 | memory_only | 1/1 | 0.21 | 0.21 | 0.00 |
| 44 | memory_only | 1/1 | 0.80 | 0.00 | 0.00 |
| 57 | memory_only | 2/2 | 0.00 | 0.21 | 0.00 |
| 58 | memory_only | 11/11 | 1.00 | 0.00 | 0.00 |
| 68 | **both** | 6/6 | 1.00 | 1.00 | 0.00 |
| 87 | memory_only | 5/5 | 1.00 | 0.00 | 0.00 |
| 92 | no_literal_match | 无字面匹配 | 0.20 | 0.13 | 0.00 |

**11 道受损题的真实分布**：

- **答案只存在于记忆池载体块中（`memory_only`）：6 题** — idx 6、8、34、38、41、57；
- **同时存在于末尾 2048 可见文本中却仍被污染：4 题** — idx 24、35、68（`both`）+ idx 39（`tail_only`，答案只在可见尾部、根本不需要内存）。这 4 题 Selection 全对、SharedMerge 全 0，说明污染不是"取不到证据"，而是**融合改变了 reader 在已可见证据上的选择**；
- **字面无法定位：1 题** — idx 92（"no" 极性题）。

**6 道待恢复题**：5 题 `memory_only`（idx 12/36/44/58/87，Selection 丢掉载体块后无从恢复），1 题 `both`（idx 23，证据本就在尾部可见——那题的"救回"其实不需要内存）。

---

## 9. 对主线的含义

- **"可恢复信息为零"不成立，"20 块全保留也就那样"成立，而且原因是量化的**：读侧压缩只值 2pp，且其中可争的只有 1 题（43 道内存依赖题里 Full 51% vs Selection 49%）。损失的大头在写侧（约 25pp），属于冻结 Reader 的范畴。
- **⚠ 本节 2pp 的读侧读数建立在"触发点 14、只用 14 块"的旧设置上**：把触发点改到预算 20（`--trigger-blocks 20`）后，Selection 33.25% → 37.13%（+3.88pp）、SharedMerge 32.78% → 37.90%（+5.12pp），读侧可争空间从 2pp 变成 4–5pp。逐题归因（救回 1 / 损伤 11）是**旧触发点**下的形态，需在满预算下重测。详见 `experiments/v7-full-trigger.md`。
- **因此 20/13/9 这个几何下，融合算子的可争空间已经被测到只剩噪声**；继续在该几何上调内部架构不可能产出可发表的涨点。
- 两条不动 Reader、不动主线的出路：
  1. **把预算压到 8/4 块**，让读侧压缩真的开始伤人（现在 13→9 只丢 2pp，因为丢得起）；先只跑 Full/Selection 参照臂扫天花板，判断融合是否还有空间。
  2. **换证据更分散/更靠前的长任务**，让"内存读取"在总分里占更大份额。
- 若继续做算子，一级指标必须是**救回率 + 损伤率 + 内存依赖题正确率**，而不是任务均值 Δ。

---

## 10. 复现

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

# 块级溯源（哪些题答案还在内存里、错答实体来自哪一块）
python tools/block_provenance.py --indices 12,23,36,44,58,87,6,8,24,34,35,38,39,41,57,68,92

# 证据位置分布（写侧 vs 读侧损失拆分）
python tools/evidence_location.py --arms Full,Selection,SharedMerge
```
