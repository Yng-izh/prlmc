# v7 天花板实测（已修正）：同路径 Full 比 Selection 高 2.05pp，但 n=96 判不出显著性

日期：2026-10-21 | 代码基线：`ee0a939`（`tools/ceiling_report.py`、`PRLMC_READER_PATH`、`--arm Full`）
状态：**§1.1 的 +0.10pp 已被 §1.5 的同路径 Full 修正为 +2.05pp**。旧数字来自跨路径参照（native writer + default 初始化，低 1.95pp）；§0 已按修正版重写，逐题归因见 `experiments/v7-per-question-hotpotqa.md`
产物：`runs/longbench_prlmc_v7_eval/`（Selection / SharedMerge / **Full**）、`runs/longbench_prlmc_v7_oracle/`（旧 Full50，已被同路径 Full 取代）

---

## 0. 结论

1. **同路径无压缩 Full = 35.30%，同预算 Selection = 33.25%，headroom 点估计 +2.05pp**，配对 bootstrap 95% CI **[−3.59pp, +8.36pp]**，W/L/T = 9/11/76（n=96）。
2. **headroom 不是零，只是 n=96 测不出来**：CI 覆盖 0，80% power 检测 1.7pp 需 **n ≈ 2525**。v7 实测 −0.47pp（CI [−6.37pp, +5.52pp]），W/L/T = 11/16/69，即**未检测到效应**——但这不等于"没有可恢复信息"。
3. **旧口径的 +0.10pp 天花板是跨路径参照造成的**（§1.5 / D7）：旧 Full50 走 native writer + default 内存初始化，比同路径 Full 低 **1.95pp**。用错参照会把 headroom 系统性低估到 0，进而把 v5 的 +1.70pp 误判为"逻辑上不可能"。
4. **逐题看，SharedMerge 在 6 题可恢复信息中只救回 1 题，同时损伤 11 题**（其中 9 题两个参照都答对）。按 `F1>0` 计净 −10 题；按 `F1>0.99` 计净 +2 题（赢的是部分匹配，输的是 1.00→0）。详见 `experiments/v7-per-question-hotpotqa.md`。
5. **v7 的架构工作（分代配额、tanh 软界、稀疏高 SNR 门控、原生支路解耦）不是"在零空间里优化"**，而是缺一个能判定 2pp 效应的评测：当前所有结论都受功效限制。
6. **评测设置本身有九个缺陷**（§2），其中"缺少同路径无压缩参照"、"统计功效不足"与"名义预算没花掉"是导致此前所有结论不可信的根本原因。

---

## 1. 实测数据

### 1.1 hotpotqa（96 配对样本，13/9 几何，16k / tail 2048 / reset 协议）

| 臂 | `official_qa_f1_truncated` | 相对 Selection |
|---|---|---|
| Selection（同预算选择器） | 33.25% | — |
| **Full（同路径无压缩，天花板）** | **35.30%** | **+2.05pp**，CI [−3.59, +8.36]，W/L/T 9/11/76 |
| Full50（旧，跨路径 native writer） | 33.35% | +0.10pp，CI [−6.21, +6.53]，W/L/T 14/17/65 |
| SharedMerge v7（DsupV7 ckpt `11d36751…`） | 32.78% | −0.47pp，CI [−6.37, +5.52]，W/L/T 11/16/69 |

同路径 Full 的 `compression_count` 在 96 题上全为 0；与 Selection 的 `input_token_sha256`、`context_sha256`、`input_tokens`、`write_count` 逐题一致，唯一差异是 `final_active_slots`（Full 保留全部写入块）。

### 1.1b 四项互补口径（`ceiling_report.py` schema/2，同一批 96 题）

| 臂 | mean F1 | 非零率 F1>0 | 高精度率 F1≥0.80 | EM |
|---|---|---|---|---|
| Selection | 33.25% | 54.2% | 22.9% | 16.7% |
| Full（参照） | 35.30% | 55.2% | 25.0% | 20.8% |
| SharedMerge v7 | 32.78% | **44.8%** | 25.0% | 19.8% |

配对 Δ（SharedMerge − Selection）：

| 口径 | Δ | 95% CI | W/L/T |
|---|---|---|---|
| mean F1 | −0.47pp | [−6.37, +5.52] | 11/16/69 |
| 非零率 | **−9.38pp** | **[−16.67, −2.08]** | 2/11/83 |
| 高精度率 | +2.08pp | [−5.21, +9.38] | 7/5/84 |
| EM | +3.12pp | [−4.17, +10.42] | 8/5/83 |

配对 Δ（Full − Selection，headroom）：

| 口径 | Δ | 95% CI | W/L/T |
|---|---|---|---|
| mean F1 | +2.05pp | [−3.59, +8.36] | 9/11/76 |
| 非零率 | +1.04pp | [−5.21, +8.33] | 6/5/85 |
| 高精度率 | +2.08pp | [−5.21, +9.38] | 7/5/84 |
| EM | +4.17pp | [−1.04, +9.38] | 5/1/90 |

**这是本次最重要的口径发现**：SharedMerge 的 mean F1 是 −0.47pp 且 CI 覆盖 0（"无效应"），但非零率 **−9.38pp 且 CI 不含 0（显著损伤）**，同时高精度率/EM 为正。也就是"能答对的实体少了 9pp，但凡答对就更完整"。单看 mean F1 会把这两个反向效应抵消成噪声。EM 口径上 headroom 也是四项里最接近显著的（+4.17pp，CI 下界 −1.04pp）。

### 1.2 历史无压缩读数（hotpotqa，200 样本全量）

`runs/longbench_native50_v5metric_20261018_1050/`，同一套 truncated F1 口径：

| 协议 | F1 |
|---|---|
| `default/`（样本间不重置内存） | 35.80% |
| `reset/`（每样本重置，与 v5/v7 同协议） | **34.32%** |

对照 v5 `t2_final_stats.json`：Selection 35.01%、SharedMerge 36.70%。**reset 口径下 Selection 反而比无压缩高 0.7pp**——压缩损失在噪声内为零。

### 1.3 narrativeqa（部分，90 样本）

Full50 = 22.76%；v5 全量 Selection 21.85%、SharedMerge 22.60%。**不同 run、未配对**，仅作方向参考：同样没有显示出可恢复空间。

### 1.4 统计功效

hotpotqa 配对差值 per-sample sd = **0.3256**。要在 80% power、α=0.05 下检测 **1.7pp** 效应，需要 **n ≈ 2880 样本**。既有 200 样本评测的 CI 宽度约 ±5–8pp，**从来没有能力证伪或证实 1–2pp 级效应**。

### 1.5 参照路径不同轴：同一批 96 题，同路径 Full 比跨路径 Full50 高 1.95pp

| 参照 | 实现路径 | 内存初始化 | F1（96 配对题） |
|---|---|---|---|
| 旧 Full50 | `tools/longbench_memoryllm_eval.py`：native writer `inject_memory(update_memory=True)`，绕过 manager | checkpoint 值 + `initialized` 标志，`used` 不归零 | 33.35% |
| **新 Full** | `tools/longbench_prlmc_eval.py --arm Full`：同一 manager/MergeStream 写路径，仅关闭压缩 | checkpoint 值 + `used=0` + 全 active mask | **35.30%** |

差 **+1.95pp**。这不是"两个都是无压缩所以应该相等"的小事：写路径（native 随机逐出 vs manager 放置）和内存初始化（D5）都变了。**天花板必须与被评测的臂走同一条代码路径**，否则 headroom 被低估，算子被判为"无空间"。

---

## 2. 评测设置的八个缺陷

| # | 缺陷 | 后果 | 修正 |
|---|---|---|---|
| D1 | **没有无压缩天花板参照** | 无法区分"算子弱"与"根本没有可恢复信息"；v5 的 +1.70pp 就是这样被当成涨点 | 每个任务必须同时跑 Full（`--arm Full`，同 entrypoint） |
| D2 | **统计功效不足** | 200 样本 CI ±5–8pp，1–2pp 效应不可判定 | 报告 `required_n`；n 不足时只能声明"未检测到" |
| D3 | **两套评分口径混用** | 同一 run 的 hotpotqa：raw F1 9.44% vs truncated 35.80%，差 3.8 倍 | 统一 truncated；raw 只作附注 |
| D4 | **几何不同轴** | v5 19/9 每样本最多压 1 次；v7 13/9 压 2–4 次（分布 {0:15,1:10,2:12,3:21,4:37}），绝对分不可比 | 只比同几何的 Δ；跨几何只比相对 Selection 的增益 |
| D5 | **内存初始化协议不统一** | `default` vs `reset` 在 hotpotqa 上差 1.5pp（35.80 vs 34.32），足以伪造或抹掉一个"涨点" | 固定 `--reset-memory`，并写入 identity |
| D6 | **reader 加载路径硬编码 GPFS** | `MODEL_PATH` 指向 37GB FP32，每行白耗 ~12 分钟；10 行 = 2 小时 | 新增 `PRLMC_READER_PATH`，指向 `/dev/shm/s1d_reader_fp16`（加载 ~4 秒），路径写入 identity |
| D7 | **天花板参照走了另一条代码路径** | 旧 Full50 用 native writer + default 初始化，同一批 96 题比同路径 Full 低 **1.95pp**，把 headroom 从 +2.05pp 压成 +0.10pp | 用 `--arm Full` 在同一个 entrypoint 内跑参照（复用已有 `Full` streaming 臂，不另造 oracle） |
| D8 | **名义预算 20 块、实际只用 14 块；训/评触发点错位** | `compaction_trigger = min(budget, window+1) = 14`：写满 14 块就压最老 13→9、只留 1 块未压缩尾巴，常驻宽度只在 10–14 之间循环，第 15–20 块（6 块 = 1536 槽）从未被使用；而训练侧前 32 步先堆到 19 块再压 13，留 6 块未压缩尾巴 | `--trigger-blocks`（评测侧控制，不重训）：触发点=预算 20，压最老 13→9 后保留最新 7 块未压缩，常驻宽度 16–20，与独立窗口训练布局一致；trigger 写入 protocol/identity |
| D9 | **口径单一 + 用"不可能效应"硬否决** | 只看 mean F1 会把"答错实体"与"答对但不完整"混成同一个 0.47pp；且旧工具用 `effect_exceeds_headroom` 把超过参照的效应直接判死，可参照本身并非单调上界（9 胜 / 11 负 / 76 平） | `ceiling_report.py` schema/2：每臂同时报 mean F1 / 非零率(F1>0) / 高精度率(F1≥0.80) / EM 四项 + 各自配对 CI；删除硬否决，改为 `notes` 提示 |

---

## 3. 修正后的评测协议（强制）

1. **三臂同跑**：每个任务必须同时产出 Selection / Treatment / Ceiling（`--arm Full`，同 entrypoint），同 seed、同几何、同 `--reset-memory`。缺 Ceiling 的行不得进入结论。
2. **只认配对 CI**：所有效应量报 paired bootstrap 95% CI（`tools/ceiling_report.py`，resamples 2000, seed 0）。
3. **四项互补口径同时报**：mean F1、非零率（F1>0）、高精度率（F1 ≥ 0.80）、EM。单一 mean F1 不得作为结论依据——v7 实测里 SharedMerge 的 mean F1 是 −0.47pp（CI 覆盖 0），但非零率 **−9.38pp，CI [−16.67, −2.08]** 已经显著为负，而高精度率 +2.08pp / EM +3.12pp 为正：同一均值掩盖了"答错实体 9pp、答对时更完整"这两个反向效应。
4. **参照不是单调上界，不做硬否决**：Full 与 Selection 逐题 9 胜 / 11 负 / 76 平，Full 自己在 5 题上比 Selection 差，所以"效应大于 headroom 均值"只作 `notes` 提示，不判为不可信。
5. **headroom 覆盖 0 不等于 headroom 为零**：只能声明"未检测到"；继续改算子的前提是把 n 提到 `required_n` 以上，或换到 headroom 更大的设置。
6. **样本量**：以 `required_n` 为准；达不到就只声明"未检测到效应"，不声明涨点。
7. **口径唯一**：`official_qa_f1_truncated`；raw F1 只作附注，不得跨口径比较。
8. **逐题归因**：均值 Δ 之外必须报"救回/漏掉/损伤"题数（`tools/per_sample_diff.py`），因为同一均值可以来自"救回 10 题、损伤 10 题"。
9. **预算必须真花掉**：`--trigger-blocks` 记录在 protocol/identity，名义预算与实际常驻宽度必须一致（D8）。

---

## 4. 复现命令

```bash
# 天花板（同路径无压缩，--arm Full 复用已有 Full streaming 臂）
python tools/longbench_prlmc_eval.py --arm Full \
  --data-dir /share/latent/benchmarks/LongBench/data \
  --tasks hotpotqa --total-tokens 16384 --tail-tokens 2048 --cuda-device 8 \
  --out runs/longbench_prlmc_v7_eval/Full_hotpotqa.json

# 同预算选择器 / 融合臂（默认触发点 min(budget,window+1)=14）
PRLMC_READER_PATH=/dev/shm/s1d_reader_fp16 \
python tools/longbench_prlmc_eval.py --arm Selection \
  --budget-blocks 20 --window-blocks 13 --target-blocks 9 \
  --tasks hotpotqa --total-tokens 16384 --tail-tokens 2048 --cuda-device 8 \
  --out runs/longbench_prlmc_v7_eval/Selection_hotpotqa.json \
  --data-dir /share/latent/benchmarks/LongBench/data

# 满预算控制（触发点=预算 20，常驻宽度 16–20，保留 7 块未压缩尾巴）
python tools/longbench_prlmc_eval.py --arm Selection \
  --budget-blocks 20 --window-blocks 13 --target-blocks 9 --trigger-blocks 20 \
  --tasks hotpotqa --total-tokens 16384 --tail-tokens 2048 --cuda-device 8 \
  --out runs/longbench_prlmc_v7_eval/Selection_fulltrigger_hotpotqa.json \
  --data-dir /share/latent/benchmarks/LongBench/data

# 天花板裁决
python tools/ceiling_report.py \
  --selection runs/longbench_prlmc_v7_eval/Selection_hotpotqa.json \
  --ceiling   runs/longbench_prlmc_v7_eval/Full_hotpotqa.json \
  --treatment runs/longbench_prlmc_v7_eval/SharedMerge_hotpotqa.json \
  --allow-partial

# 逐题归因
python tools/per_sample_diff.py \
  --selection runs/longbench_prlmc_v7_eval/Selection_hotpotqa.json \
  --ceiling   runs/longbench_prlmc_v7_eval/Full_hotpotqa.json \
  --treatment runs/longbench_prlmc_v7_eval/SharedMerge_hotpotqa.json \
  --data-dir /share/latent/benchmarks/LongBench/data --allow-partial
```

---

## 5. 对主线的影响

- **"恢复被丢弃信息"这个命题重新成立，但当前评测证明不了**：同路径 headroom 点估计 +2.05pp（n=96 的 CI 覆盖 0，需 n≈2525 才能判定）。此前"信息量为零"的判断来自跨路径参照（D7），已撤回。
- **逐题证据比均值更有信息量**：6 题可恢复中救回 1、损伤 11（9 题两参照都对），净 −10 题（`F1>0` 口径）。算子的失效形态是"写入另一个真实实体"，不是保留文本退化。
- **损失的大头在写入侧，不在读侧**：答案只在被写入前缀里的 43 题，全保留（Full）也只答对 **51%**，Selection 49%（只差 1 题）；答案能直接在尾部文本里看到的题正确率 **76%**。即冻结 Reader 的写入压缩（512 token → 256 latent 槽）吃掉约 **25pp**，而我们一直在调的读侧压缩（13→9）只值 **2pp**。工具 `tools/evidence_location.py`，详见 `experiments/v7-per-question-hotpotqa.md` §8。
- 可选的重构方向（需另行立项，不在本文件决定）：
  1. **先把功效补上**：hotpotqa 全量 200 + 2wikimqa/musique 补足到 ~2500 配对样本，用同路径 Full/Selection 定住 headroom 真值，再谈算子。
  2. **定向修损伤**：对"两参照都对、融合独错"的 9 题做因果消融（carrier 选择 vs 融合写入），把"救回率 / 损伤率"做成一级指标。
  3. **提高预算压力**：在 budget 10 / 8 / 4 上重做 headroom 扫描，确定"从第几块开始压缩才真的伤"。
- **当前不建议**在 n=96 上继续做架构消融：该样本量下任何 2pp 级消融都只能测到噪声。
