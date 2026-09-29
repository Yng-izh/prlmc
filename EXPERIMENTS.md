# PRLM-C 实验日志

实现 *残差记忆设计_方案整理* 的完整实验记录。每条记录给出配置、命令、原始数字与结论。

基线为未修改的 MemoryLLM-8B on MemoryAgentBench，参数见 `_mllm_patch/BASELINE.md`。
本文件中的一切数字都在同一环境（`/share/latent/mllm_env`，V100-32GB，`chunk_size 400`，
`inject_max_tokens 512`，`do_sample False`）下取得，可直接对比。

## 目录

| 编号 | 运行 | 目的 | 状态 |
|---|---|---|---|
| E0 | Baseline Tier 1 | 冻结参照分 | 已完成，5 数据集 |
| E1 | `prlmc-smoke` | 端到端链路验证 | 已完成 |
| E2 | `prlmc-obs` | 阈值标定（`compact_mode: none`） | 已完成，500 次写入 |
| E3 | `prlmc-r1` | occupancy-only 结构消融 | 已完成，**accuracy 全 0** |
| E3b | 纯代码分析 | 归零的机制根因（不运行） | 已完成 |
| E4 | 2×2 矩阵 | 分离 gate 幅值与 compactor | 已完成，**compactor 是充分原因** |
| E5a | `recent_blocks` | 只保留原生 block 时 reader 是否正常 | 已完成，**EM 26.00** |
| E5b | `select_blocks` | 打分选择能否胜过纯 recency | 已完成，**与 E5a 逐条相同** |
| E5c | E5b + active mask | 真正 K→K' 的无训练 compaction baseline | 已完成，**EM 23.00**（与 E5a 不再相同） |
| E5d | E5c，但 filler 不受限 | 排除 E5c 的两个开关混淆 | 已完成，**EM 24.00**，与 E5c 逐条 97/100 相同 |
| E6 | `carrier_residual` | learned fusion 是否超过 selection | 已完成，**EM 24.00**；无 ckpt，初始化即恒等，故仅验证接线 |
| E6a | 离线单窗口 overfit | 训练链能否跑通（不读 MemoryAgentBench 指标） | 已完成，单窗口 220 步恢复 2/3 |
| E6b-mini | 离线多窗口共享 checkpoint | 单窗口恢复能否泛化到未见窗口 | 已完成（**2026-09-29 重跑，前一轮作废**），训练 7/30 = 0.233、held-out 2/16 = 0.125，判为 `partial` |
| E7 | E6 + adaptive trigger | 恢复完整 Ours | 未开始，依赖 E6 的训练 |

线性读法：E1 发现触发器不触发 → E2 标定阈值 → E3 修触发器后首次测 accuracy → **发现 accuracy 归零，根因未分离** → E4 定位到 compactor → E5a/E5b 用原生 block 选择把 accuracy 恢复到 26.00 → E5c 打开 active mask，free region 真正移出 reader 视野，EM 23.00 → E5d 放开 filler 限制以排除开关混淆，与 E5c 逐条 97/100 相同 → E6 把 compactor 换成 `carrier_residual`，因未训练而与 E5c 逐条 98/100 相同 → E6a 用离线单窗口 overfit 验证训练链；首轮打分跨度错误使全部指标作废，修复后筛出 `lost = 3` 的硬窗口并训练到恢复 2/3 → E6b-mini 把单窗口扩到 8 训练 + 2 held-out；**首轮因度量缺陷把 held-out 分子结构性固定为 0，`window_overfit` 与 0/16 全部作废；修复后重跑得训练 7/30、held-out 2/16，两侧无统计差异（Fisher p = 0.4628），计划预设的「训练强 / held-out 弱」对比在本轮数据里检不出**。

---

---

## 总结：目前能得出什么结论

### 一条硬结论

**合成 latent 向量是读取失败的充分原因。**

E4 A→C 只改 `compact_mode` 一个变量：`none` 27.00 → `svd_tokens` **0.00**。
27 分的落差是 n=100 标准误（≈6.1）的 4.4 倍，远超噪声；E4 B→D 在 novelty gate 上复现同一结果。
这是整套实验里唯一统计上稳固的结论，也正是方案设计最初的诊断。

### 两条软结论

**原生 block 选择让压缩生命周期存活。** 同样的 5 次压缩、同样的内部状态，
0.00 → 26.00。压缩这个动作本身不再是杀手。

**active mask 是唯一真正改变读取路径的机制。** E5c 相对 E5b 有 15/100 条答案改变；
E5a vs E5b 是 100/100 逐条完全相同。

### 三条未确立

| 要证明的 | 实测 | 问题 |
|---|---|---|
| 压缩优于淘汰 | E4 A 淘汰 27.00 vs E5a 压缩 26.00 | 差 1 分，噪声内 |
| 打分选择优于纯 recency | E5a vs E5b 逐条 100/100 相同 | 选择策略从未被检验 |
| 残差融合有效 | E6 与 E5c 98/100 相同 | 未训练，`up` 零初始化 |

### 决定性缺口

**没有任何一次运行显示"压缩比不压缩更好"。**

| 运行 | 机制 | EM |
|---|---|---|
| E4 A | 淘汰（34 次驱逐） | 27.00 |
| E5a–E6 | 压缩（5 次，驱逐 0） | 23–26 |
| baseline | 无管理 | 22.00 |

三者全部落在 22–27 之间，而 n=100 的 SE 是 6.1，**该区间内的一切排序都不成立**。

压缩的价值主张是"同等预算下保住更多功能"。现有证据只支持**"不比淘汰差"**，
**不支持"比淘汰好"**。

### 一句话判定

方向没被证伪，处方也没出错；但方案的核心卖点——压缩带来的功能保持——**到目前为止没有被测到**。

原因具体：所有运行都在 `factconsolidation_sh_32k`（83 chunks），两种机制都装得下，
压缩没有被逼到发挥。要测出来需要一个**预算真正紧张**的配置
（更小的 `K_max`，或 `sh_262k` 的 688 chunks），并且 E6 必须带**训练过的** checkpoint。

### 全部结果一览

`factconsolidation_sh_32k`，n=100；除 E4 B/D 外均为 `gate_mode: ones`。

| 运行 | compactor | mask | EM | F1 | rougeL_f1 | rougeL_recall | 压缩次数 |
|---|---|---|---|---|---|---|---|
| baseline | — | — | 22.00 | 25.990 | 26.355 | 29.167 | — |
| E4 A | `none` | – | **27.00** | 32.019 | 32.101 | — | 0 |
| E4 B | `none`（gate=novelty） | – | 18.00 | 20.863 | 20.841 | — | 0 |
| E4 C | `svd_tokens` | – | **0.00** | 0.000 | 0.000 | — | 5 |
| E3 = E4 D | `svd_tokens`（gate=novelty） | – | **0.00** | 0.000 | 0.000 | — | 5 |
| E5a | `recent_blocks` | – | 26.00 | 31.819 | 31.901 | 34.333 | 5 |
| E5b | `select_blocks` | – | 26.00 | 31.819 | 31.901 | 34.333 | 5 |
| E5c | `select_blocks` | on | 23.00 | 29.249 | 29.577 | 33.833 | 5 |
| E5d | `select_blocks` | on | 24.00 | 30.249 | 30.377 | 33.833 | 5 |
| E6 | `carrier_residual` | on | 24.00 | 30.249 | 30.377 | 33.833 | 5 |

E5a/E5b/E5c/E5d/E6 的内部状态完全一致：`used_slots 8192`、`peak_used 9984`、
`compaction_count 5`、`evicted_blocks 0`、`occupancy 0.667`、`n_consolidated 2304`；
后三者另有 `active_memory_mask True`、`active_mask_slots 8704`。

---

## E0 — Baseline Tier 1（冻结参照）

命令：`/share/latent/mllm_patch/batch_run.py`，agent 配置 `configs/agent_conf/MemoryLLM/memoryllm_8b.yaml`。
输出：`/share/latent/MemoryAgentBench/outputs/memoryllm-8b/`。

| 数据集 | n | exact_match | f1 | rougeL_f1 | output_len |
|---|---|---|---|---|---|
| `icl_banking77` | 100 | **40.00** | 40.000 | 40.000 | 20 |
| `factconsolidation_sh_32k` | 100 | **22.00** | 25.990 | 26.355 | 10 |
| `eventqa_65536` | 100 | **18.00** | 40.128 | 43.580 | 40 |
| `factconsolidation_mh_6k` | 100 | **7.00** | 8.571 | 8.971 | 10 |
| `detective_qa` | **6** | 0.00 | 0.621 | 0.578 | 2000 |

`factconsolidation_sh_32k` 的完整 `averaged_metrics`，作为后续对比的基准：

```
exact_match 22.0            f1 25.990476
substring_exact_match 24.0  rougeL_f1 26.354762
rougeL_recall 29.166667     rougeLsum_f1 26.354762
rougeLsum_recall 29.166667  input_len 162.65
output_len 10.0             memory_construction_time 0.236619
query_time_len 1.315295
```

**`detective_qa` 只有 6 条查询，`output_len 2000`，且 `substring_exact_match` 与
`rougeL_recall` 退化为 100 —— 不要汇总此行的分数。**

---

## E1 — `prlmc-smoke`：端到端链路验证

| 项 | 值 |
|---|---|
| 配置 | `configs/agent_conf/prlmc_smoke.yaml` |
| 脚本 | `run_smoke.sh`（`QUERIES=2`） |
| 数据集 | `factconsolidation_sh_32k`，83 chunk |
| 时间 | 2026-09-26 07:00:08 UTC 结束，**3483.6 s** |
| 输出 | `outputs/prlmc-smoke/` |

配置要点：`trigger occupancy_novelty_rank`，`tau_k 0.80`，`tau_n 0.885`，
`tau_k_exit 0.60`，`tau_n_exit 0.880`，`trigger_consecutive 5`，
`compact_mode svd_tokens`，`write_mode append`，`gate_mode novelty`，`k_anchor 512`。

### 结果

**触发器 0/82 次触发。** 3 次压缩全部来自 `_place` 的兜底路径：

```
reason headroom  12288 -> 9216  freed 3072  window 6144  target 3072
```

每个兜底事件都是 `12288 → 9216`，即池满后压缩。配置的阈值从未被满足。

| 信号 | min | p50 | max |
|---|---|---|---|
| `novelty` | 0.8559 | 0.8788 | 1.0000 |
| `gate` | 0.8414 | 0.8726 | 1.0000 |
| `occupancy` | 0.0208 | 0.7917 | 1.0000 |

`novelty_ema >= 0.885` 只维持到第 24 次写入；`occupancy >= 0.80` 从第 38 次写入开始。
两个窗口相差 14 次写入，**没有任何一次同时满足**，连续满足 5 次的窗口为 0。

### 结论

`tau_k AND tau_n` 不是"阈值调得不好"，是**结构性不可达**：occupancy 每次写入单调上升，
novelty 则随子空间增长而下降，两者反相关。据此设计了 E2 的标定与 E3 的顺序修复。

### 附带发现（当时未修）

- `_headroom_handled` 在 `while` 循环体内无条件置真，而循环条件本身就要求超容量，
  因此 `over_budget` **恒为假**。
- `compact_probe: true` 是死代码：`register_probes` 在 benchmark 路径无调用点。

---

## E2 — `prlmc-obs`：阈值标定（`compact_mode: none`）

| 项 | 值 |
|---|---|
| 配置 | `configs/agent_conf/prlmc_obs.yaml` |
| 脚本 | `run_obs.sh`（`QUERIES=8`） |
| 数据集 | `Factconsolidation_sh_64k`（169 chunk）、`ICL_nlu`（333 chunk） |
| 时间 | 2026-09-26 01:42:04 → 05:49 UTC，**14780.7 s**（5407.0 + 9373.7） |
| 输出 | `outputs/prlmc-obs/prlmc_stats.jsonl`（500 次写入） |

配置要点：**`compact_mode: none`**，`tau_k 0.0`，`tau_k_exit 0.0`，
`trigger occupancy_novelty_rank`，`trigger_consecutive 5`。压缩全程关闭，只采信号。

### 内部信号（500 次写入，2 个 episode）

| 信号 | min | p10 | p50 | max |
|---|---|---|---|---|
| `novelty` | 0.8802 | 0.8844 | **0.8942** | 1.0000 |
| `occupancy` | 0.0208 | — | 1.0000 | 1.0000 |
| `effective_rank` | 25.77 | — | 110.29 | 280.74 |
| `energy_rank` | 105 | — | 427 | 600 |
| `rank_utilisation` | — | — | — | 1.097 |

`encode_seconds` min 15.40 / p50 25.73 / max 44.34 秒每 chunk。
`tokens` 恒为 256。健康度：0 丢弃，404/500 驱逐，**0 压缩**。

### 关键观察

`novelty` 的下界是 **0.8802**，p10 为 0.8844 —— 这个量与 `tau_n 0.885` 几乎重合。
标定后取了 `tau_n 0.885`。

**但这是基于错误数学得到的数。** 当时的 `summarise_block` 对 `[layer, token, d]` 同时跨
layer 与 token 求平均，而 `_refresh_basis` 与 `_spectrum` 只用 layer 0，
且 candidate 未减 `basis_mean`。三方口径不一致，0.88 这个量级没有几何意义。
E3 修正后同一数据上的 `novelty` p50 从 0.8942 掉到 **0.1277**（见 E3）。

### 本运行的 accuracy（`compact_mode: none`）

**这是 PRLM-C 首次得分**，且是在压缩完全关闭的情况下：

| 数据集 | n | EM | F1 | rougeL_f1 | 样例输出 |
|---|---|---|---|---|---|
| `factconsolidation_sh_64k` | 8 | 0.00 | 6.944 | 6.667 | `' English \n <Assistant> I have learned the facts'` |
| `icl_nlu` | 8 | **12.50** | 12.500 | 12.500 | `' 0\n\nQuestion:...label: 1'` |

输出是**合法自然语言**。这一点在 E3 之后成为关键证据（见 E3 的根因分析）。

---

## E3 — `prlmc-r1`：occupancy-only 结构消融

| 项 | 值 |
|---|---|
| 配置 | `configs/agent_conf/prlmc_r1.yaml` |
| 脚本 | `run_r1.sh`（`QUERIES=0`，即不截断，与基线同为 100 条查询） |
| 数据集 | `factconsolidation_sh_32k`，83 chunk，100 查询 |
| 时间 | 2026-09-26 08:06:52 → 08:28:45 UTC，**1273.7 s** |
| 输出 | `outputs/prlmc-r1/` |

配置要点：`structural_ablation: trigger_occupancy_only`，`trigger occupancy`，
`tau_k 0.80`，`tau_k_exit 0.0`，`trigger_consecutive 1`，`compact_mode svd_tokens`，
`compact_ratio 0.5`，`write_mode append`（刻意固定，隔离 compactor 与 trigger），
`gate_mode novelty`，`k_anchor 512`，`stats_dir` 位于 `prlmc:` 块内。

### 三个问题的答案

**1. occupancy 到 0.80 能否触发 —— 能。**

| 运行 | `compacted=true` | `trigger_reason` | 触发原因 |
|---|---|---|---|
| E1 smoke | **0/82** | 全 `None` | 3 次全为 `headroom` 兜底 |
| E3 r1 | **5/82** | 5 次 `capacity_trigger` | 5 次全为触发器主动决策 |

`over_budget` 与 `made_room_for_write` 均为 0 —— 5 次压缩没有一次是兜底路径。

**2. compact 后 `used` 能否下降 —— 能。** 5 次事件数字完全一致：

```
reason capacity_trigger  used_before 9984 -> used_after 7424
freed 2560  window_slots 4864  target_slots 2304  weighted False
```

`window 4864` = 19 块（`max(compact_min_slots 256, used//2)` 取整到块），
`target 2304` = 9 块（`4864 × 0.5` 取整到块），与 `compact_ratio 0.5` 一致。
触发点在第 40、51、62、73、84 次写入，`occupancy` 在 0.8125 越过 0.80。

**3. accuracy 掉多少 —— 归零。**

| 指标 | 基线 | E3 r1 |
|---|---|---|
| `exact_match` | **22.00** | **0.00** |
| `f1` | 25.990 | 0.000 |
| `substring_exact_match` | 24.00 | 0.000 |
| `rougeL_f1` | 26.355 | 0.000 |
| `rougeL_recall` | 29.167 | 0.000 |
| `output_len` | 10.0 | 10.0 |
| `memory_construction_time` | 0.2366 | **4.9943** |

100 条查询中只有 **16 个不同输出**（基线 65 个）。样例：

```
r1  : 'šíSSI-git\\Bridge471\ufffd性460aaaaaaaa_REFdef'
基线: ' baseball \n <Assistant> I have learned the facts'
```

### 内部信号（82 次写入）

| 信号 | E1 smoke | E3 r1 |
|---|---|---|
| `novelty` p50 | 0.8788 | **0.1277** |
| `gate` p50 | 0.8726 | **0.0967** |
| `occupancy` p50 | 0.7917 | 0.6667 |
| `used_slots` max | 12288 | 9984 |
| `encode_seconds` p50 | 22.74 | **0.1665** |

`encode_seconds` 从 22.74 秒降到 0.1665 秒是测量口径变化，不是真实提速（见下文"未解问题"）。

### 根因分析

**先作废一个错误推理。** 曾据"压缩发生在第 40 次写入，而前 40 条查询也是 0"推断压缩无关。
**该推理无效**：benchmark 的结构是先把 83 个 chunk 全部写完，再跑 100 条查询，
5 次压缩（第 40/51/62/73/84 次写入）全部发生在**任何查询之前**。
不存在"压缩前的查询"。

**决定性证据：E2 的运行是好的。** E2 用**同一套 PRLM-C 写入/读取代码**，
唯一差别是 `compact_mode: none`，输出为合法英文并拿到 `icl_nlu` 12.5。
因此"PRLM-C 的读取路径本来就坏"这一假设被排除。

由此得到两个**互相独立、未被分离**的候选根因：

**根因一：`svd_tokens` 破坏状态。**

`methods/prlmc/consolidate.py` 返回 `singular × vh`，即按奇异值缩放的右奇异向量，
是一组**正交主方向**，不是低秩重构。两个破坏机制：

- **分布外。** `cat_memory_and_hiddens` 把 `self.memory[idx][-num_tokens:]` 作为 prefix
  喂入 attention。backbone 期望读到的是它自己写出的 delta token 分布；正交主方向
  即使量级对齐，方向结构也不对。
- **信息丢失。** SVD 保留最大方差方向，而 `factconsolidation` 需要的恰是
  "最新序列号对应哪条事实"这类低方差细节。19 块压成 9 个平均方向后，具体事实不在其中。

**根因二：gate 幅值下降约 9 倍。**

E3 的 novelty 数学修正本身是对的，但它同时改变了 gate：

| 运行 | 数据集 | 代码 | `novelty` p50 | `gate` p50 |
|---|---|---|---|---|
| E1 smoke | `sh_32k` | 旧 | 0.8788 | 0.8726 |
| E3 r1 | `sh_32k` | 新，减 `basis_mean` | **0.1277** | **0.0967** |
| E2 obs | `sh_64k`+`icl_nlu` | 旧 | 0.8942 | 0.8918 |

`gate = novelty`，注入幅值被削弱约 9 倍。gate 是**唯一的写入幅值控制**。

**E1 与 E3 是同一数据集（`sh_32k`）、同一 chunk 数（83）、同一 `gate_mode`，
唯一差别是 novelty 数学修正**，因此 0.8726 → 0.0967 的落差可归因于该修正本身，
不是数据集差异。E2 使用不同数据集，仅作旁证。

**注意 E1 的反证：** E1 的 gate p50 是 0.8726（接近满幅），accuracy 同样是 0.0。
但 E1 只有 **n=2** 条查询，不足以单独定罪 compactor，也不足以给 gate 脱罪。

### 分离状态表

| | 有压缩 | 无压缩 |
|---|---|---|
| 旧代码（gate ≈ 0.87） | E1：0.0（n=2，弱） | **E2：正常（n=8，icl_nlu 12.5）** |
| 新代码（gate ≈ 0.10） | E3：0.0（n=100，强） | **未测** |

两个变量（compactor 与 gate 幅值）在 E3 中同时变化，无法归因。

**（E4 跑完后回填）** 空缺的那一格由 E4 的 B 格填上，结果为 **18.00**，不是 0.0。
完整的 2×2 矩阵与结论见 E4 一节。

### 结论

按设计文档"occupancy-only 跑不好就先修 compactor"的判据，**下一步是修 compactor，
不是继续调 trigger**。触发机制本身已验证可用（问题 1、2 均为正面结果）。

---

## E3b — 纯代码层面的根因分析（不运行）

在跑 E4 之前，先从代码读出的机制解释。以下每一条都给出文件与行号，可独立复核。
本节不含新数据，全部结论来自 E0–E3 的已有数字加上源码。

### 结论

**`svd_tokens` 写进池子的东西在数学上不是可读记忆，这一条足以单独解释归零。**
gate 是真实但次要的第二个问题，且它很可能是 compactor 的后果而非独立原因。

### 问题一：压缩产物不是记忆（高置信，主因）

`methods/prlmc/consolidate.py:46-79` 的 `_svd_tokens` 返回：

```python
tokens = (singular[:take].unsqueeze(1) * vh[:take]) * scale
```

即 **S·Vᵀ**。完整重构需要 **U·S·Vᵀ**，U 被丢弃。函数 docstring 声称
"so that a linear reader reconstructs the window from these tokens" —— **该断言不成立**。
没有 U，这批 token 重构不出任何东西；而且读池子的是 attention，不是线性解码器。

**更严重的是 `take` 的取值：**

| 量 | 值 | 来源 |
|---|---|---|
| 压缩窗口 K | 4864 | E3 的 `window_slots` |
| 特征维 d | 4096 | 模型 `hidden_size` |
| 请求的 target | **2304** | E3 的 `target_slots` |
| obs 实测窗口 `effective_rank` p50 | **110.29** | E2 |

`take = min(target, singular.numel()) = min(2304, 4096) = 2304`，而窗口的**实际有效秩
只有约 110**。第 111～2304 个 token 是噪声方向：`vh` 各行互相正交，前几个占掉数据
真实使用的高方差方向后，其余被迫落在数据几乎不用的方向上。
**这不是"压缩后的事实"，是 2200 多个相互正交的噪声向量。**

量级上 `σᵢ · vhᵢ` 的范数恰为 σᵢ，而 σ₁ 是窗口矩阵的谱范数，远大于单行范数。
fp16 存得下，但作为 attention 的 key/value 是强离群值，softmax 会被它吸走。

### 问题二：free region 平铺把污染放大（高置信，放大机制）

`methods/prlmc/layout.py:753` 的 `_tile_free_region` 用 `_distinct()` 平铺
`[k_anchor, total - used)`。

| 运行 | `used` | free width | 是否生效 |
|---|---|---|---|
| E2 obs（`compact_mode: none`） | 长期钉在 12288 | **0** | 几乎从不生效 |
| E3 r1（压缩后） | 7424 | **4864** | 每次压缩后生效 |

平铺的源头 `_distinct()` 以刚压过的 SVD 块开头（压缩块落在 distinct suffix 最前）。
结果：12800 个位置里 **4864 个（约 38%）是同一批 SVD 噪声向量的重复**，
且紧贴工作区之前。查询时 `is_injection=False`，`cat_memory_and_hiddens` 读的是
**整个池子**，这批内容同时"离群"且"高频重复"。

**这正是 E2 与 E3 差别被放大的地方**：E2 里这条路径是空操作，E3 里它是主要贡献者。

### 问题三：gate 真实但不是决定因素（中置信）

`methods/prlmc/layout.py:469-475`：

```python
delta = apply_projection(delta, self.projection)
gate = self._gate(novelty)
gate_value = float(gate)
delta = delta * gate
```

基线 MemoryLLM 的写入没有 gate，幅值即 1.0。E3 的 gate p50 为 0.0967，
写入比基线弱约 10 倍。

**但 E1 的反证同样有力**：E1 用旧数学，gate p50 0.8726（接近基线幅值），accuracy 也是 0.0。
E1 只有 n=2，不足以给 compactor 定罪，但**足以说明 gate 单独解释不了归零**。
反向也成立：低 gate + 无压缩这一格从未测过，所以 gate 也脱不了罪。

### 问题四：gate 与 compactor 通过 basis 耦合（中置信）

`_refresh_basis()` 与 `_spectrum()`（`layout.py:923`）都建立在
`_distinct()[_REFERENCE_LAYER]` 之上。

压缩后 distinct suffix 以 SVD 噪声块开头 → basis 描述的是噪声子空间 →
新数学下的 novelty 是**相对量**（被解释能量占比），对 basis 高度敏感。
所以 E3 的 0.0967 里有一部分是 compactor 的后果，不全是数学修正的后果。

**推论**：E4 若也归零，只能证明"低 gate 也致命"，**不能**证明 gate 是独立的第二因。
若要真正分离，需要第三个格子：新代码 + `compact_mode: none` + `gate_mode: ones`。

**（E4 跑完后回填）** 该第三个格子就是 E4 的 **C 格**（`gate=ones` + `svd_tokens`），
它拿到 0.00，与 D 格相同；而 B 格（`gate=novelty` + `none`）拿到 18.00。
结论：**compactor 是充分原因，gate 是次要因素**。E3b 原先"E4 归零即证明 gate 独立"
的判据已被证伪，详见 E4 一节。

### 两个数据异常的代码解释（已定性）

**rank 全为 0 —— 不是 bug，是没算。** `trigger: occupancy` 时 `_snapshot` 不计算 spectrum
（`needs_spectrum` 仅在 `trigger == "occupancy_novelty_rank"` 时为真），
因此 `effective_rank` 与 `energy_rank` 在 E3 中恒为 0。E1/E2 的 0 同理（且 E1/E2 的
p50 为 0 是因为 `used < block` 的前期写入返回 None）。

**`encode_seconds` p50 0.1665 —— 无法由代码路径解释。** `_encode` 走
`is_injection=True`，前缀只取尾部 256 slot，成本与 `used` 几乎无关。
150 倍落差指向环境（GPU 争用）或测量点不同，仍需核对计时起止点。

### Round 2 代码在 E3 中完全没执行

E3 用 `write_mode: append`，因此 `addressed = False`、`allocated = True` 恒成立
（`layout.py:483-497`）。`_merge_into_block`、`_address`、`tau_alloc` **一行都没跑**。
**E3 的 0.0 不能归给合并逻辑。**

### 代码层面定不下来的

- **是否有第二个独立杀手。** ~~必须靠 E4~~ **已由 E4 的 C 格定下**：gate=ones 时
  compactor 单独仍打 0.00，故 compactor 是充分原因；gate 是次要因素（见 E4）。
- 从代码下注的结果：`svd_tokens` 是主因（**高置信，已实证**）；gate 是次要放大
  （**中置信，已实证掉 9.0 EM 但不致命**）；两者经 basis 耦合（中置信，未单独验证）。

### 一句话机制

**压缩把 19 块真实 delta 换成了 9 块含约 90% 噪声方向的半截 SVD 因子，
再把这批内容平铺复制到池子的 38%，让 attention 优先锁定它们。**

---

## E4 — 2×2 隔离矩阵：gate × compact

**设计**：代码、数据、seed 全部固定，只改两个开关。四格中的 D 就是 E3，
所以只需新跑 A、B、C 三格。

|  | `gate=ones` | `gate=novelty` |
|---|---|---|
| **`compact=none`** | **A**（EM 27.00） | **B**（EM 18.00） |
| **`compact=svd_tokens`** | **C**（EM 0.00） | **D = E3**（EM 0.00） |

每格 = `prlmc_r1.yaml` 改 `gate_mode` 与 `compact_mode` 两行（外加 `output_dir`/`stats_dir`），
共 3～4 行差异，逐行核验过。数据集 `factconsolidation_sh_32k`，82 次写入，100 条查询。

### 结果

| 格 | `gate_mode` | `compact_mode` | EM | F1 | rougeL_f1 | 基线参照 |
|---|---|---|---|---|---|---|
| **A** | `ones` | `none` | **27.00** | 32.019 | 32.101 | EM 22.00 |
| **B** | `novelty` | `none` | **18.00** | 20.863 | 20.841 | F1 25.990 |
| **C** | `ones` | `svd_tokens` | **0.00** | 0.000 | 0.000 | rougeL 26.355 |
| **D**（= E3） | `novelty` | `svd_tokens` | **0.00** | 0.000 | 0.000 | |

A 的其他指标：`substring_exact_match 29.00`，`rougeL_recall 34.333`，
`input_len 162.65`，`output_len 10.0`。

### 支撑数字

| 格 | compacts | 触发原因 | evictions | `gate` p50 | `novelty` p50 | `mct` s/chunk |
|---|---|---|---|---|---|---|
| A | 0 | — | **34** | 1.0000 | 0.4372 | 0.309 |
| B | 0 | — | **34** | **0.0577** | 0.0839 | 0.279 |
| C | **5** | `capacity_trigger` | 0 | 1.0000 | 0.3793 | 25.218 |
| D | 5 | `capacity_trigger` | 0 | 0.0967 | 0.1277 | 4.994 |

A、B 两格 `over_budget_request` 各 34 次、`made_room_for_write` 各 34 次 —— 即
`compact=none` 时全部走 `_place` 的 `_evict_oldest` 兜底，行为退化为滑动窗口。
C、D 两格的 5 次压缩均为 `capacity_trigger`，`used` 9984 → 7424，与 E3 完全一致。

### 时间

| 格 | 开始 | 结束 | 耗时 |
|---|---|---|---|
| A | 11:52:25 | 12:09:12 | **955.3 s** |
| B | 12:09:45 | 12:22:13 | **702.9 s** |
| C | 12:22:13 | — | **853.0 s** |

三格在同一张 GPU 6 上串行执行（`chain_2x2.sh`），同进程外无并发。

### 结论

**1. compactor 是充分原因。** A → C 只动 `compact_mode`，gate 恒为 1.0，
accuracy 从 27.00 掉到 0.00。E3b 的代码分析（S·Vᵀ 非可读记忆 + free region 平铺）
得到实证。

**2. gate 是真实但次要的伤害，单独不致命。** A → B 只动 `gate_mode`：
27.00 → 18.00，**掉 9.0 个 EM**，且**低于基线 22.00**。gate p50 从 1.0000 降到 0.0577。

**3. 无交互效应。** 两个 compactor 格都是 0.00，compaction 完全主导，
gate 在其之上没有可测量的额外影响。

### 判据对照

E3b 原先给出的判据是"若 E4 也归零 → gate 是第二个独立杀手"。**该判据已被证伪**：
E4 的 B 格（新代码 + `compact=none`）拿到 18.00，不是 0.0。
真正决定性的是新增的 **C 格**：gate 取满幅时 compactor 单独仍把 accuracy 打到 0.00。
E3b 中"需要第三个格子才能分离"的判断由此兑现 —— C 就是那个格子。

### 不确定项（不要从本矩阵读出的结论）

- **A 高于基线 5 个 EM 不构成结论。** n=100 下差值标准误约 6.1 个百分点，
  5 个点是 0.8 个标准误，**不显著**。
- **seed**：配置无 seed 旋钮。确定性来自 `do_sample=False` 贪心解码与固定数据顺序，
  因此四格可比。
- **C 的 `mct` 25.218 s/chunk 与 D 的 4.994 s/chunk 差 5.05 倍**，
  两格压缩次数与参数完全相同，该差异无代码解释，怀疑是运行时 GPU 争用。

### Round 2 待办（已被 carrier-residual 路线取代）

按依赖顺序：

1. 修 novelty 数学定义 —— **已完成**：`_REFERENCE_LAYER = 0` 统一 basis/spectrum/candidate
   口径，candidate 减同一个 `basis_mean`。
2. 实现 low-novelty 更新已有 block / high-novelty 才 append —— **代码已写**：
   `_merge_into_block`、`_address`、`allocated_new_slots`、`write_mode: additive`。
   判据为 `rho < tau_alloc OR cosine >= address_threshold`。
3. 重新设计 adaptive trigger —— 围绕 allocation pressure，而非 occupancy ∧ novelty。
4. 接 functional probes —— `register_probes` 需要调用点，否则 `compact_probe` 永远惰性。

> 第 2、3 项已被 carrier-residual 阶段方案取代：真正的长期方法是
> carrier-residual functional compactor，而不是修 append/additive 二选一。
> 代码保留在上表中，但不再作为主线推进。

---

## E5a — `recent_blocks`：原生 block 纯 recency 选择

### 动机

E4 已经把范围收窄到一个开关。同一 `gate=ones` 下：

| cell | compactor | EM |
|---|---|---|
| A | `none` | 27.00 |
| C | `svd_tokens` | **0.00** |

两格唯一差异是 compactor。所以 `svd_tokens` 是充分原因，且 `svd_tokens` 被移出主线。

E5a 换上目前最保守、但仍然真实做 K → K' 的算子：把窗口切成完整 256-slot 原生
block，保留最近的 B_out 个，逐 slot 原样复制。不合成任何新向量，reader 见到的
是它本来就训练过要读的 memory。因此 E5a 隔离的是"compact / 写回生命周期本身
能不能活"，而不是"算子好不好"。

### 实现

`methods/prlmc/consolidate.py`：

```python
def recent_block_compact(source, target, block=256):
    layers, k, d = source.shape
    b = k // block
    b_out = target // block
    x = source.view(layers, b, block, d)
    kept = x[:, -b_out:, :, :]
    return kept.reshape(layers, b_out * block, d).contiguous()
```

`window` 与 `target` 必须同时是 `block` 的整数倍。`compact_with_report` 在最终
对 `used - protect` 做 clamp 之后可能把 `window` 打回非整数倍（当最新 block
窄于 `block` 时），因此新增一条显式守卫，返回 `reason_skipped:
block_misaligned:...`，而不是让算子在 try 里抛异常、把原因丢进
`compactor_error`。

配置：`configs/agent_conf/prlmc_e5a.yaml`，相对 E4 的 cell A 只改三行
（`output_dir`、`stats_dir`、`compact_mode: none → recent_blocks`）。

### 结果

`factconsolidation_sh_32k`，n=100，`gate_mode: ones`，`trigger: occupancy`，
`chunk_size 400`，82 次写入。

| 指标 | 基线 | A（none） | C（svd_tokens） | **E5a（recent_blocks）** |
|---|---|---|---|---|
| exact_match | 22.00 | 27.00 | 0.00 | **26.00** |
| f1 | 25.990 | 32.019 | 0.000 | **31.82** |
| substring_exact_match | 24.00 | — | 0.00 | **28.00** |
| rougeL_f1 | 26.355 | 32.101 | 0.000 | **31.90** |

内部信号：82 次写入，5 次压缩，`evicted_blocks 0`，`peak_used 9984`，
`used 8192`，`n_consolidated 2304`，`occupancy 0.667`，
`memory_construction_time 27.6 s`，`query_time_len 1.47 s`，
总时长 815.7 s。

### 结论

**生命周期是活的。** 压缩真实发生（5 次）、真实释放容量（`used` 9984 → 8192）、
且 reader 仍然正常作答（EM 26.00）。

**算子决定了 0 与 26 的差别。** C 与 E5a 的唯一差异是输出向量来自原生 memory
还是来自 SVD 子空间。EM 0.00 → 26.00，与 E3b 的机制判断一致：冻结的 reader
无法解释一种它没见过的 latent 编码。

**代价相对 `none` 是 1 个 EM（27.00 → 26.00）。** n=100 下差值标准误约 6.1
个百分点，这个差**不显著**，不能读成"压缩造成了 1 点损失"。

---

## E5b — `select_blocks`：importance/diversity 打分选择

### 动机

E5a 只用 recency。E5b 换成打分选择，看已有的 importance 信号能不能胜过"只留最新的"。
仍然只选择、不修改：幸存 block 原样复制，且**按原始时序写回，不按分数顺序**。

```text
score_i = w_imp * importance_i + w_recent * recency_i
        + w_goal * goal_relevance_i + w_div * diversity_i
```

diversity 项在 `_greedy_diverse` 里以 max-min 方式贪心加入，不是静态项。
打分权重 `{importance 0.4, recency 0.3, goal 0.2, diversity 0.1}`，偏向 recency。
`goal` 项用 `self.goal_direction`（与 basis/novelty 同一个 `_REFERENCE_LAYER = 0`）。

### 结果

配置 `prlmc_e5b.yaml`，相对 E5a 只改 `compact_mode: recent_blocks → select_blocks`
与输出目录。

| 指标 | A（none） | E5a（recent） | **E5b（select）** |
|---|---|---|---|
| exact_match | 27.00 | 26.00 | **26.00** |
| f1 | 32.019 | 31.82 | **31.82** |
| substring_exact_match | — | 28.00 | **28.00** |
| rougeL_f1 | 32.101 | 31.90 | **31.90** |

内部信号与 E5a **逐项相同**：5 次压缩、`used 8192`、`peak_used 9984`、
`n_consolidated 2304`、`evicted 0`、`occupancy 0.667`、总时长 805.3 s。

逐条查询比对：100/100 条 `output` 字符串完全相同；26 条 EM 正确项集合完全重合
（`a_only 0`、`b_only 0`）。

### 结论

**E5b 与 E5a 在本任务上等价。** 100 条输出逐字相同，不只是总分相同。

**但这不是"selection 比 recency 好"，也不是"打分类信号有效"。** 两者相同更可能
是因为在 19 → 9 的选择里，本轮打分恰好也基本选中了最近的 9 个 block：`goal`
方向来自首个 chunk，`recency` 权重 0.3 且是唯一在窗口内单调变化的项。

**本轮无法区分这两个算子。** 要真正比较，需要构造一个 recency 明显次优的场景
（例如窗口内事实冲突、正确答案在更老的 block 里），否则 E5b 只是 E5a 的复制。
这是 E5c 之前需要先决定的事。

---

## E5c — `select_blocks` + active mask：真正 K→K'

### 动机

E5a/E5b 只选 block、不改 token，但有一个遗留问题：池子里没有用到的 **free region
仍然被 reader 读**。原因有二：

1. filler 用 `_tile_free_region` 把占用内容平铺进去。内容是"真的"，所以 reader
   会把它当成证据。
2. 更根本的是，读路径是 `is_injection=False`，**整个池子都进 attention**，
   不区分哪一段是有效记忆。

E5c 要验证的就是：把 free region 从 attention 里真正拿掉，能不能在不训练的前提下
建立一个 K→K' 的 compaction baseline。

### 实现（Phase 2）

池子宽度**不变**，只改 attention bias。原因见下。

| 位置 | 改动 |
|---|---|
| `config.py` | 新增 `active_memory_mask: bool`、`native_filler_only: bool` |
| `layout.py` | `_sync_active_mask()` 把 `[total]` bool mask 发布到 `model.memory_active_mask` |
| `layout.py` | `_tile_free_region()` 的源在 `native_filler_only` 下限制为**未压缩原生 block** |
| `tools/patch_modeling.py` | 幂等/可回滚地给 untracked 的 modeling fork 打 9 处补丁 |

`_sync_active_mask` 以 `used` 为缓存键，在 `_place`（append）、`_tile_free_region`、
`_evict_oldest`、`compact_with_report`、`reset`、`restore`、`initialize_anchor`
调用。active = Anchor 前缀 `[0, k_anchor)` ∪ 有效后缀 `[total-used, total)`；
free region `[k_anchor, total-used)` 置为 `finfo.min`，SDPA 分配到**恰好 0** 权重。

**为什么不做可变长度前缀切片。** RoPE 偏移和 prefix length 都以固定记忆长度为前提，
把前缀切短会让每个位置整体移位。mask 只改"哪些 key 可被看到"，不动位置。

**打补丁时必须处理的三点**（都踩过）：

1. 原 SDPA 在 `prefix_token_length > 0` 分支里把 `causal_mask` **整体重建成 bool**，
   会丢弃从 `attention_mask` 传下来的加法 bias。所以 bias 必须在**该重建处**折入，
   不能走 `_update_causal_mask`。
2. 只在 prefill 加 bias 不够。decode 步的池子只存在于 KV cache 里，
   `prefix_token_length == 0`，走的是另一条路径。bias 必须同时覆盖 `past_seen + q`。
3. `prefix_token_length` **只在循环内部赋值**。第一版把 bias 构建放在循环之前，
   会直接 `NameError`，已改为在循环内按 `prefix_len` 惰性构建并按 `(prefix_len, key_len)` 缓存。

无 mask 时补丁走**原 bool 分支**，逐字节复现原行为，所以 E5a/E5b 打了补丁也不会变。

### 结果

配置 `prlmc_e5c.yaml` = E5b + `active_memory_mask: true` + `native_filler_only: true`。

| 指标 | E5a（recent） | E5b（select） | **E5c（select + mask）** |
|---|---|---|---|
| exact_match | 26.00 | 26.00 | **23.00** |
| f1 | 31.819 | 31.819 | **29.249** |
| substring_exact_match | 28.00 | 28.00 | **26.00** |
| rougeL_f1 | 31.901 | 31.901 | **29.577** |
| rougeL_recall | 34.333 | 34.333 | **33.833** |
| 总时长 | 815.7 s | 805.3 s | **409.5 s** |

内部状态：5 次压缩、`used 8192`、`peak_used 9984`、`n_consolidated 2304`、
`evicted_blocks 0`、`occupancy 0.667` —— 与 E5a/E5b **完全一致**。
新增 `active_mask_slots 8704`、`active_mask_updates 88`。

**E5c 不再是 E5a 的复制品。** 15/100 条 `parsed_output` 改变，EM 逐条比对：

| | 条数 |
|---|---|
| 两者都对 | 23 |
| 仅 E5a 对 | **3** |
| 仅 E5c 对 | **0** |
| 两者都错 | 74 |

3 条仅 E5a 正确的例子（E5c 给出了更长的句子）：

| query | E5a parsed | E5c parsed | gold |
|---|---|---|---|
| q3 | `methodism` | `morris iemma is affiliated with the religion of method` | `Methodism` |
| q4 | `association football` | `fc slutsk is associated with the sport of association` | `association football` |
| q65 | `catholic church` | `bahawalpur state is affiliated with the religion` | `Catholic Church` |

E5c 的答案是"更完整但不再等于 gold 短答"，所以 EM 掉了 3 分，而 `rougeL_recall`
只从 34.333 掉到 33.833 —— 内容其实还在，只是措辞变长。

### 结论

**active mask 生效了，而且是可测的。** 这是第一次让 free region 真正离开 reader
的 attention，也是第一次 E5 系列出现与 E5a 不同的输出。

**但 mask 本身伴随 -3 EM 的代价，方向必须如实记录。** n=100 时 SE ≈ 6.1 分，
3 分在噪声内，**不能断言 mask 有害**，同样也不能断言无害。当前证据只支持：
mask 改变了 read path，且在长答场景下让 3 条查询从"精确短答"变成"包含答案的长句"。

**K→K' 的物理基础已经成立。** `active_mask_slots 8704 < total 12800`，池子里
4096 个 slot 已确定不参与 attention，同时 accuracy 没有崩（23 vs 26）。这是 E6
`carrier_residual` 的前置条件。

**409.5 s vs 805 s 的时长差不来自 mask。** 日志时间戳显示差异全部在
memorization 阶段（E5a/E5b 约 11m20s，E5c 约 4m36s），query 阶段三者都约 2m13s。
这是 GPFS 数据加载的缓存冷热差异，不是计算量差异。

---

## E5d — `select_blocks` + mask，但 filler 不受限

### 动机

E5c 同时动了**两个**开关：

1. `active_memory_mask: true` —— free region 移出 attention。
2. `native_filler_only: true` —— filler 只允许平铺未压缩的原生 block。

如果 mask 完全生效，filler 就不可读，第 2 个开关应当是**无影响**的。
反过来说，**如果 mask 有漏洞，两个变量就混在一起，E5c 的 -3 分无法归因。**

已知的具体风险点：`output_attentions=True` 会走**未打补丁**的 eager 路径，
bias 在那里被静默丢弃。

E5d 就是为分离这两个开关而跑：**只放开 filler 限制，mask 保持打开。**

### 配置

`prlmc_e5d.yaml` = `prlmc_e5c.yaml` 只改一个开关
（`native_filler_only: true → false`）与输出目录。

| 开关 | E5b | E5c | E5d |
|---|---|---|---|
| `active_memory_mask` | false | true | **true** |
| `native_filler_only` | false | true | **false** |

### 结果

| 指标 | E5b（mask off） | E5c（mask on, filler 原生） | **E5d（mask on, filler 自由）** |
|---|---|---|---|
| exact_match | 26.00 | 23.00 | **24.00** |
| f1 | 31.819 | 29.249 | **30.249** |
| substring_exact_match | 28.00 | 26.00 | **27.00** |
| rougeL_f1 | 31.901 | 29.577 | **30.377** |
| rougeL_recall | 34.333 | 33.833 | **33.833** |
| query_time_len | 1.328 | 1.342 | **1.567** |

内部状态三者**完全一致**：`used 8192`、`peak_used 9984`、5 次压缩、
`evicted 0`；E5c/E5d 的 `active_mask_slots 8704`、`active_mask_updates 88`。

### 逐条比对

| 对比 | `parsed_output` 相同 | EM 逐条标记相同 |
|---|---|---|
| E5b vs E5c | 85/100 | 97/100 |
| E5b vs E5d | 87/100 | 98/100 |
| **E5c vs E5d** | **97/100** | **99/100** |

相对 mask-off 参照（E5b）的不一致对，**全部偏向 mask-off**：

| 运行 | 两者都对 | 仅本运行对 | 仅 E5b 对 | 两者都错 |
|---|---|---|---|---|
| E5c | 23 | **0** | **3** | 74 |
| E5d | 24 | **0** | **2** | 74 |

### 结论

**1. 实现混淆排除。** E5c 与 E5d 只差 `native_filler_only` 一个开关，
结果 97/100 逐字相同、EM 只差 1 条。**filler 装什么内容，在被 mask 之后完全不影响结果。**

**2. 反证 mask 没有泄漏。** E5d 的 filler 仍是重复的真实内容。
若 mask 有漏洞，filler 就会重新可见，E5d 应当退回 E5b 的样子。它没有。

**3. -2/-3 分归因于 mask 本身**，不再是"两个开关混在一起"。

**4. 但这个实验区分不了猜想 1 与猜想 2。** 两个猜想都假设 filler **可见**：

* 猜想 1（重复放大）：filler 携带真实信号，等于给有效内容加了 50% 权重。
* 猜想 2（注意力汇）：filler 只是概率质量的垃圾场，内容无关。

一旦 mask 掉，两者都预测"内容无关"。**E5d 对它们是天生盲的** ——
它是一次**有效性检验**，不是机制实验。要裁决需要：**mask 关掉、filler 换成单个
常量 token**，看 EM 是否仍为 26。仍 26 → 猜想 2；掉到 23 → 猜想 1。

**5. 一致方向值得记一笔，但不足以作为证据。** 两个 mask 运行合计 5 条不一致对，
5/5 全部偏向 mask-off。单侧符号检验 p ≈ 0.031，但两个运行用的是**同一批 100 条
查询，不独立**，所以这个 p 值只能当提示。n=100 时单次运行 SE ≈ 6.1 分。

**6. anchor 份额猜想（猜想 3）本实验测不到。** E5c 与 E5d 都是 mask on，
anchor 份额同为 `512/8704 = 5.9%`；E5b 是 `512/12800 = 4.0%`。
要动这个份额就必须动 mask，E5d 帮不上。

### 附：时长

E5d 用了 1486.1 s（冷缓存），其中 checkpoint 分片加载 4m14s（约 30.6 s/it）。
与 E5c 的 409.5 s 之差全部来自 GPFS 缓存冷热，不是计算量差异。

---

## E6 — `carrier_residual`：原生载体 + 有界学习式残差

### 动机

E4 已经证明合成向量（`svd_tokens`）被冻结的 reader 读不懂。方案第 3 阶段的处方是：
**选出 reader 本来就认识的载体 block，再在其上做小幅学习式融合**，
即 `Y = Y0 + γΔ(X, Y0)`，而不是 `memory → SVD → 人造 latent → reader`。

本运行是这条路径的**接线验证**，不是"融合有效"的证据。

### 实现（Phase 3）

新增 `methods/prlmc/consolidate.py`：

| 组件 | 作用 |
|---|---|
| `_select_block_indices()` | 从 `select_block_compact` 抽出的共享打分函数 |
| `CarrierResidualCompactor` | 载体选择 + 有界残差融合 |
| `_l2()` | 仅用于 q/k 路由表示的行归一化 |

三条性质是**结构性的**，不是调参得来的：

1. **初始即恒等。** `nn.init.zeros_(self.up.weight)`，故 `delta` 恰为零，
   输出与选中的原生载体**逐位相同**。一个刚构造的 compactor 与 `select_blocks`
   不可区分，这让第一次训练步成为严格的改进问题。
2. **改动是残差。** 输出恒为 `Y0 + γΔ`，`Y0` 始终在输出里，reader 见过的 slot 永不被覆盖。
3. **幅度有界。** `γ = γ_max · sigmoid(logit)`，恒 `|γ| < γ_max`，梯度无法越过上限。

路由是 **block 级**（每个输出块对输入块一个注意力分布），不是 slot 级：
slot 级会让算子退化为重新编码，正是要避免的失败模式。
**输出路径上没有 LayerNorm**——那会重新缩放载体本身，破坏"初始即恒等"。

CPU 结构单测（`tools/unit_carrier.py`，10/10 通过）的关键一条：
**`max |carrier − select| = 0.0`**。

### 配置

`prlmc_e6.yaml` = `prlmc_e5c.yaml` 只改 compactor 与输出目录，并新增两个参数：

| 参数 | 值 |
|---|---|
| `compact_mode` | `select_blocks` → **`carrier_residual`** |
| `carrier_rank` | 32 |
| `carrier_gamma_max` | 0.1 |
| `carrier_ckpt` | **未设** |

`carrier_ckpt` 未设是关键：此时 `up` 仍为零初始化，`γ` 接近零，
该运行**在数学上应当等价于 `select_blocks`**。任何偏差都需要解释。

### 结果

| 指标 | E5c（select） | **E6（carrier）** |
|---|---|---|
| exact_match | 23.00 | **24.00** |
| f1 | 29.249 | **30.249** |
| rougeL_f1 | 29.577 | **30.377** |
| rougeL_recall | 33.833 | **33.833** |

内部状态与 E5c 完全一致（`used 8192`、`peak 9984`、5 次压缩、`evicted 0`、
`active_mask_slots 8704`）。时长 451.9 s。

### 逐条比对

| 对比 | `parsed_output` 相同 | EM 标记相同 |
|---|---|---|
| **E5c vs E6** | **98/100** | **99/100** |

| 运行 | 两者都对 | 仅 E5c 对 | 仅 E6 对 | 两者都错 |
|---|---|---|---|---|
| E6 | 23 | **0** | **1** | 76 |

### 结论

**1. 接线是通的。** E6 跑完无失败，5 次压缩正常发生，残差路径没有崩溃也没有
把 accuracy 打到 0——对照 E4 C 的 0.00，这是有意义的。

**2. 但没有达到预期的逐位一致。** 无 `carrier_ckpt` 时应当与 `select_blocks`
完全等价，实测却有 1 条 EM 差异。

最可能的原因是运行间随机性：`cat_memory_and_hiddens` 在
`delta_memory.shape[2] > num_tokens` 分支里用 `torch.randperm`，
而所有配置的 `seed` 都是 `None`。这与"1 条查询差异"的量级吻合。

**但无法排除残差路径有微小偏离**，因为没有独立复跑 E5c 来测量这份噪声基线。
这是本运行唯一未闭合的点。

**3. E6 目前不构成"融合有效"的证据。** 它没有被训练过：`up` 是零矩阵，
`γ ≈ 0.047`，残差项几乎不起作用。E6 ≈ E5c 是**预期行为**，不是发现。
真正的检验要等 Phase 4 训练出 checkpoint 后重跑。

**4. 训练需要显存。** E6 峰值约 18.2 GB（由 OOM 报错反推：已分配 15.10 GiB
再加 3.12 GiB 的申请）。E6a 已在此集群实测跑通训练，见下节。

### 附：一次失败的运行

E6 第一次投递失败了 40 次，全部 OOM，未产出任何结果。原因有二：

1. **重试脚本的显存门槛设低了。** 门槛 14000 MiB，而实测峰值约 18.2 GB，
   于是每次都挑到一张装不下的卡。已提高到 21000 MiB。
2. **在脚本运行期间覆盖了脚本自身。** bash 按字节偏移增量读取脚本，
   覆盖后偏移错位会读到半行，表现为 `genuine: command not found` 与
   `syntax error near unexpected token 'done'`。该次损坏只发生在收尾阶段，
   40 次重试本身都完整执行了。

另一个真实差异：`carrier_residual` 比 `select_blocks` 明显更费显存，
因为它在 fp32 下物化 `blocks`/`carriers`/`context`/`delta`，32 层 × 4096 hidden
下每个都是 1–2 GB。

---

## E6a — 离线单窗口 overfit：训练链验证

### 目的与形态

E6 的 compactor 从未被训练过（`up` 零初始化、`γ ≈ 0`），因此 E6 ≈ E5c 是预期行为。
E6a 用一个窗口做离线 overfit，检验 `CarrierResidualCompactor` 与整条 functional-loss
训练链能否产生参数更新。

本运行**不读 MemoryAgentBench**，不产出 EM/F1 等指标，只产出 loss 与内部状态。
驱动为 `tools/e6a_overfit.py`，在 host `/tmp/` 下执行。

窗口构造要求 probe 的支持事实落在 `select_blocks` **丢弃**的 block 里，
使 selection 可证地丢信息；判定用 `recovery_rate`，见「指标定义」。

### 配置

| 项 | 值 |
|---|---|
| 窗口 | 19 blocks = 4864 slots |
| 压缩目标 | 9 blocks = 2304 slots |
| anchor | 30 行 |
| `LINES_PER_CHUNK` | 20（19 次写入凑满 4864 slots） |
| probes | 20（由窗口内 380 条事实行合成，渲染推理时的 `rag_agent` 模板） |
| stage A | 16 步，`L_gold` 权重 0 |
| stage B | 24 步，`L = L_gold + 0.5L_KD + 0.1L_hidden + 0.05L_carrier + 0.02L_stats` |
| `lr` | 1e-3 |
| `gamma_max` | 0.05（命令行覆盖 YAML 的 0.1） |
| `use_amp` | False |
| GPU | gpu9（v100-2），32768 MB free |
| `max_probe_tokens` | 256（必须大于 ~163 token 的渲染提示词） |

### 首轮结果全部作废

E6a 首轮产出的 `accuracy` 与 `gold_loss` **全部作废**，原因见「已修复的缺陷」第 9 条：
打分跨度落在提示词中间，与 memory 池无关。以下旧表已随之删除：

- 四阶段 accuracy / gold_loss 表（`uncompacted`、`before`、stage A、stage B）
- `acceptance` 字典（`before_accuracy`、`after_accuracy` 均为错跨度产物）
- stage A / stage B 逐项 loss 表
- 内部状态表（该轮以错误目标训练，参数更新不可用）

`gamma`、`residual_ratio`、`route_scale` 等状态量本身是真实读数，但它们描述的是一轮
以错误目标训练的更新，不作为任何结论的依据。

### 硬窗口筛选结果（有效）

跨度修复后重新筛选，`--dry-run` 不训练。首个候选即命中，窗口 `row 3 start 17890`：

| 指标 | 值 |
|---|---|
| 写入 | 19 blocks = 4864 slots |
| selection 丢弃 | 10 blocks |
| 候选 probes | 20 |
| `no-mem` | 0/20 |
| `full`（未压缩满池） | 6/20 |
| `select`（selection 后） | 3/20 |
| **`lost`** | **3** |
| `eligible`（池必需） | 6 |

`carrier_residual == select_blocks at init: True (max |d| = 0.000e+00)`。

被 selection 丢掉、需要靠 fusion 恢复的三条：

```text
Who created Girls?
What sport is Charlie Manuel associated with?
Who is Victor Turner employed by?
```

`weights` 修复生效的直接证据——加权与不加权选出的 dropped 集合不同：

```text
[e6a]   weights change the dropped set: [0,1,2,3,4,5,6,7,8,9] vs [0,1,2,3,4,5,6,8,9,11] (unweighted)
```

### 指标定义

判定不看总体准确率（20 题中只有 3 题 `lost`，信号会被稀释）：

| 指标 | 定义 |
|---|---|
| `eligible` | `full` 对且 `no-mem` 错，即池确实必需 |
| `lost` | `eligible` 且 `select` 错，selection 真的丢了 |
| `recovered` | `lost` 且 `after` 对，fusion 塞了回来 |
| `recovery_rate` | `recovered / lost`，取值 0、1/3、2/3、1 |

`after` 由训练路径产出，`--dry-run` 不产生该值。

### 第一次全跑：40 步更新全部被跳过

第一次完整 E6a 跑完，`before` 与 `after` 逐位相同：

```text
before {'accuracy': 1.0, 'gold_loss': 0.8384018540382385}
after  {'accuracy': 1.0, 'gold_loss': 0.8384018540382385}
```

这两行的 `accuracy` 与 `gold_loss` 数值来自错跨度评分（缺陷 9），数值本身不作数；
但"`before` 与 `after` 逐位相同"这一事实成立，它就是该缺陷的下游证据——
`GradScaler` 每步都跳过更新。

`state_report` 中 `gamma = 0.0023712937` 与初始化值逐位相同，`carrier` 分量 40 步
全为 `0.0`（`up.weight` 恒为 0）。加 `PRLMC_GRAD_TRACE=1` 后：

```text
[grad] step 0 cand_leaf=nan scale=65536.0 | gamma_logit=nan route_logit=nan q.weight=nan k.weight=nan down.weight=nan up.weight=nan
[grad] step 1 cand_leaf=nan scale=32768.0 | gamma_logit=nan route_logit=nan q.weight=nan k.weight=nan down.weight=nan up.weight=nan
```

`scale` 从 65536 降到 32768，即 `GradScaler` 检测到 inf/nan 后每一步都跳过
`optimiser.step()`，参数保持初始化值；该次运行的 loss 数字仍正常输出。

两个独立缺陷：

1. **`consolidate.py:519`（当时）`delta / (delta_rms + 1e-6)`。** `up.weight`
   零初始化 ⇒ 第一步 `delta ≡ 0` ⇒ `delta_rms = 0`。forward 无害（`0/1e-6 = 0`），
   但 backward 的 `d(delta_rms)/d(delta_ij) = delta_ij / (N·delta_rms)` 在原点
   为 `0/0`。该 NaN 传导到除 `gamma_logit` 外的所有 compactor 参数。
2. **`probes.py` 的 `use_amp` 默认 `True`，但路径中无 `autocast`。** backbone 本身
   为 fp16，`GradScaler` 将 loss 放大 65536 倍后 backward，激活梯度超过 fp16 上限
   65504 即 `inf`，`inf * 0 → nan`。

修复后 `[grad]` 全部有限：

```text
[grad] step 0 cand_leaf=1.0039e+01 scale=1.0 | gamma_logit=1.7140e-06 route_logit=0.0 q.weight=0.0 k.weight=0.0 down.weight=0.0 up.weight=1.2535e+00
[grad] step 1 cand_leaf=7.3125e+00 scale=1.0 | gamma_logit=9.7598e-02 route_logit=2.0263e-11 q.weight=2.8726e-08 k.weight=2.8117e-08 down.weight=8.0711e-01 up.weight=3.5428e-01
```

### 硬窗口训练结果（有效，2026-09-25）

缺陷 9–12 全部修复后的完整训练，`--window 3,17890`，stage A 16 步 + stage B 24 步。
命令：

```text
E6A_NEED_MB=21000 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=9 python /tmp/e6a_overfit.py \
  --window 3,17890 --steps-a 16 --steps-b 24 \
  --probes 20 --max-probe-tokens 256 \
  --lr 1e-3 --gamma-max 0.05 --min-probes 4 --min-lost 3 \
  --out /tmp/e6a_report.json
```

筛选指标（`--dry-run` 阶段，与上表逐位一致）：

| 指标 | 值 |
|---|---|
| `no_memory` | 0/20 |
| `uncompacted` | 6/20 |
| `before`（`select_blocks`） | 3/20 |
| `lost` | 3 |
| `identity_at_init` | `True`（`max |d| = 0.000e+00`） |

训练结果：

| 阶段 | `after` | `recovery` | `gold` 首→末 | `kd` 首→末 |
|---|---|---|---|---|
| A（16 步） | 4/20 | **1/3 = 0.333** | 3.8737 → 3.7321 | 1.4617 → 1.0696 |
| B（24 步） | 4/20 | **1/3 = 0.333** | 3.7168 → 3.2787 | 1.0802 → 0.9928 |

`acceptance`：

```text
{'n_eligible': 6, 'n_lost': 3, 'n_recovered': 1, 'recovery_rate': 0.3333333333333333,
 'lost_questions': ['Who created Girls?', 'What sport is Charlie Manuel associated with?',
                    'Who is Victor Turner employed by?'],
 'recovered_questions': ['What sport is Charlie Manuel associated with?'],
 'no_memory_accuracy': 0.0, 'before_accuracy': 0.15, 'after_accuracy': 0.2,
 'uncompacted_accuracy': 0.3}
```

即 `What sport is Charlie Manuel associated with?` 被塞了回来，
`Who created Girls?` 与 `Who is Victor Turner employed by?` 仍丢失。

内部状态（阶段末）：

| 量 | A 末 | B 末 | 预算 / 参照 |
|---|---|---|---|
| `gamma` | 0.0024011 | 0.0024612 | `gamma_max` = 0.05，用到 **4.8%** |
| `residual_ratio` | 0.00072533 | 0.00142481 | 判据要求 < 5%，实测 **0.14%** |
| `route_scale` | 0.018158 | 0.018614 | — |
| `routing_entropy` | 2.9432058 | 2.9432118 | 均匀分布 |
| `forward_calls` | 34 | 83 | — |
| `amp` | 0.0 | 0.0 | — |

`carrier` 分量按步单调上升（A：`0.0 → 3.97e-05`；B：`4.26e-05 → 9.88e-05`），
说明 `up.weight` 确实在离开零初始化；`hidden` 分量全程持平在 ~0.0028，未下降。

**读数解读（不构成机制结论）**：`residual_ratio` 停在 0.14%，即注入张量与原生
`select_blocks` 的差异只占 carrier 的千分之一量级，`gamma` 只用掉预算的 4.8%。
融合项没有真正离开恒等点，因此那 1/3 的恢复不足以归因于 residual fusion。
16 + 24 步、`lr 1e-3` 的有效梯度信号太弱，`kd` 是唯一明显下降的分量。

### Stage B 延长到 220 步（同窗口同超参）

条件与上一节完全相同，只把 Stage B 从 24 步延长到 **220 步**（Stage A 仍 16 步），
固定 `--window 3,17890`，结构未改，loss 权重未调。运行 05:43:58Z → 09:38:07Z，
共 3 h 54 min，`EXIT=0`。命令：

```text
E6A_NEED_MB=21000 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=9 PRLMC_GRAD_TRACE=1 python /tmp/e6a_overfit.py \
  --window 3,17890 --steps-a 16 --steps-b 220 \
  --probes 20 --max-probe-tokens 256 \
  --lr 1e-3 --gamma-max 0.05 --min-probes 4 --min-lost 3 \
  --out /tmp/e6a_report_longB.json
```

筛选指标与前两次逐位一致（`no_memory 0/20`、`uncompacted 6/20`、`before 3/20`、
`lost 3`、`identity_at_init True`）。

#### 两版 Stage B 对照

| 项 | Stage B 24 步 | Stage B 220 步 |
|---|---|---|
| `after` | 4/20 | **6/20** |
| `recovery_rate` | 1/3 = 0.333 | **2/3 = 0.667** |
| `gold` 首→末 | 3.7168 → 3.2787 | 3.6848 → **2.4318** |
| `kd` 首→末 | 1.0802 → 0.9928 | 1.0047 → **0.7586** |
| `hidden` 首→末 | 0.0029897 → 0.0029241 | 0.0028581 → **0.0043209** |
| `carrier` 首→末 | 4.26e-05 → 9.88e-05 | 4.37e-05 → **2.244e-04** |
| `stats` 首→末 | 5.78e-06 → 6.06e-06 | 5.78e-06 → 9.60e-06 |
| `residual_ratio` | 0.0014248 | **0.0026117** |
| `gamma` | 0.0024612 | 0.0030977 |
| `route_scale` | 0.018614 | 0.0267555 |
| `routing_entropy` | 2.9432118 | **2.7599626** |
| `forward_calls` | 83 | 475 |

#### `acceptance`（220 步版）

```text
{'n_eligible': 6, 'n_lost': 3, 'n_recovered': 2, 'recovery_rate': 0.6666666666666666,
 'lost_questions': ['Who created Girls?', 'What sport is Charlie Manuel associated with?',
                    'Who is Victor Turner employed by?'],
 'recovered_questions': ['What sport is Charlie Manuel associated with?',
                         'Who is Victor Turner employed by?'],
 'no_memory_accuracy': 0.0, 'before_accuracy': 0.15, 'after_accuracy': 0.3,
 'uncompacted_accuracy': 0.3}
```

`lost_questions` 列的是全部三条 `lost`，不是未恢复项；本次未恢复的只有
`Who created Girls?`。

#### Stage A 两侧一致

两次运行的 Stage A 完全相同（16 步、`after 4/20`、`recovery 1/3`）：

| 量 | 24 步版 Stage A | 220 步版 Stage A |
|---|---|---|
| `gamma` | 0.0024011049 | 0.0024011049 |
| `residual_ratio` | 0.00072533 | 0.00074491 |
| `route_scale` | 0.018158 | 0.0181656 |
| `routing_entropy` | 2.9432058 | 2.9432061 |
| `forward_calls` | 34 | 34 |
| `carrier` 末 | 3.97e-05 | 4.08e-05 |

`gamma` 逐位相同，`residual_ratio` 有约 2.7% 的相对差异，其余量在浮点噪声范围内。

#### 读数

四个内部量在 220 步下同向移动，且**第一次明确离开恒等点邻域**：
`residual_ratio` 1.83×、`route_scale` 1.44×、`carrier` 2.27×，
`routing_entropy` 从 2.9432 降到 2.7600（路由不再均匀）。
这四点合起来比 `recovery_rate` 本身更强——它说明融合与路由确实被训练推动，
而不是数值噪声。

但两条判据仍未满足：

- `residual_ratio` 0.0026，是「< 5%」判据的**约 1/19**；`gamma` 只用到预算的 **6.2%**。
- 「reader hidden loss 下降」**反向**：`hidden` 从 0.0028581 升到 0.0043209
  （区间最低 0.0026601）。

分母只有 3，`1/3 → 2/3` 是一道题的差别，单看该比值不构成证据；
`after 6/20 == uncompacted 6/20` 同样受制于样本量。

### 附：显存

修复 reader 图与 compactor 图共存后，`[mem] step 0 before student`
从 23.02 GB 降到 21.20 GB。40 步训练期间该值稳定在 21.49–22.70 GB，无 OOM。
延长到 220 步（累计 236 步）期间同样无 OOM。

缺陷 12 修复前的对照：`alloc=22.72GB` 时 backward 需要 3.12 GiB、
仅 3.09 GiB 可用，第一步即 `torch.OutOfMemoryError`。

第一次全跑在此处 OOM：`torch.OutOfMemoryError` 于 stage A 第 0 步的 probe backward，
`GPU 0 has a total capacity of 31.73 GiB of which 68.00 MiB is free`。
当时的账目为：模型 19.57 GB + teacher pass ≈ 20.0 GB（`step 0 before student` 实测
23.02 GB）+ per-layer checkpoint 存下的满宽 layer 输入 3.37 GB + 全池 `pool.grad`
3.36 GB + compactor 活图 3.02 GB。

两条当时的猜测经实测被排除：

1. **注意力分数矩阵**：实测 mem-efficient SDPA 同时接受 bool mask 与 float `-inf` mask，
   forward 峰值 1.24–1.46 GB、backward 1.87 GB，未回退到 math backend。
2. **per-layer checkpoint 无效**：该分支有效，但 fork 在每层循环内重新拼接 12800 前缀，
   每个被存下的 layer 输入为满宽 `[1, 12801, 4096]` fp16 ≈ 105 MB，32 层 ≈ 3.37 GB。

---

## E6b-mini — 多窗口共享 checkpoint：单窗口成绩能否泛化

### 目的与形态

E6a 已经证明**在单个硬窗口上**训练可以把 `recovery_rate` 推到 2/3。E6b-mini 只问一件事：

> 换一个没见过的硬窗口，同一个 checkpoint 还能恢复吗？

这是一次**判定性**实验，不是尺度放大：它与 E6a 的唯一变量是**窗口数量**。
结构、loss、selector、trigger、gate 全部逐字不动——同一个 `CarrierResidualCompactor`、
同一份 `_stage_weights`、同一个 `_select_block_indices`、同样 19→9、`lr 1e-3`、
`gamma_max 0.05`。这样无论结果正负都只有一种解释。

**但该前提有一处未兑现**：每窗口步数同时变了（E6a 220 步/窗口 vs 这里 24 步/窗口）。
这一混淆在本轮成为核心读数限制，见下方「未被排除的竞争假说」。

硬窗口定义沿用 E6a：`uncompacted` 答对、原生 `select_blocks` 丢掉。
**没有为了凑数量塞入普通窗口。**

### 配置

| 项 | 值 |
|---|---|
| 训练窗口 | 8（parquet row 3） |
| held-out 窗口 | 2（parquet row 2） |
| 每窗口 probe 数 | 训练 20，held-out 60 |
| Stage A | 8 窗口 × 4 步 = 32 |
| Stage B | 8 窗口 × 24 步 = 192 |
| 合计 | 224 步，`forward_calls 456` |
| `window_blocks` / `target_blocks` | 19 / 9 |
| `lr` / `gamma_max` | 1e-3 / 0.05 |
| `seed` | 未传入；`torch.manual_seed` 未执行（见缺陷 17） |
| 优化器 | 单个 `AdamW` 实例跨全部窗口与两个阶段复用 |

`probes.py` 为此新增可选参数 `optimiser=None`：不传时行为与原版逐字节相同；
传入时复用调用方实例，使 AdamW 的动量跨窗口存活。否则每个窗口都是重新拟合，
"共享 checkpoint"就是假的。

### 三处设计决策

1. **Stage A 也摊到 8 个窗口。** 方案只固定 Stage B 的 loss，未规定 Stage A。
   Stage A 是"别弄坏 reader"的预热，若只跑单窗口就退化成 E6a 的预拟合，
   "一个共享 checkpoint"只对 B 成立。两个阶段都把步数按窗口均分。

2. **held-out 用 60 probes，训练用 20。** 评估是纯 forward，宽 probe 集不额外付费。
   E6a 的 `lost` 分母只有 3，`recovery_rate` 只能取 0、1/3、2/3、1，
   分辨不出"泛化"与"背下这个窗口"。

3. **训练与验证按 parquet 行分离。** 该文件有 8 行，但**只有 4 个不同 context**
   （0↔4、1↔5、2↔6、3↔7 逐字节相同，正文 `ctx` 摘要为 `8678ca61` 与 `0875f57e`）。
   同行切分等于在训练文本上评估。同行内还要求任意两个窗口的**事实行区间不重叠**。
   筛选循环特意用 `require_other_rows=True`，否则某行一旦凑够窗口就会提前停止，
   静默产出同行的"held-out"。

### 硬窗口筛选结果

11 个候选筛出 10 个硬窗口，用时 332 s，29–62 s/窗口。

| row | start | `uncompacted` | `select_blocks` | `lost` |
|---|---|---|---|---|
| 3 | 17890 | 6/20 | 3/20 | 3 |
| 3 | 17130 | 4/20 | 2/20 | 2 |
| 3 | 16750 | 9/20 | 3/20 | 6 |
| 3 | 14470 | 6/20 | 1/20 | 5 |
| 3 | 17510 | 6/20 | 3/20 | 4 |
| 2 | 3450 | 9/20 | 7/20 | 3 |
| 3 | 14850 | 7/20 | 4/20 | 3 |
| 3 | 16370 | 7/20 | 2/20 | 5 |
| 3 | 15230 | 10/20 | 8/20 | 2 |
| 2 | 3830 | 9/20 | 3/20 | 6 |

每个窗口 `no_memory 0/20` 均成立：答案不可能来自预训练权重，只能来自池。
`carrier_residual == select_blocks at init: True (max |d| = 0.000e+00)`，
所以 `before` 就是原生 selection 的分数。

划分结果：训练 row 3（8 窗口），held-out row 2（2 窗口），`row_separated: True`。

### 结果（2026-09-29 重跑，v100-2 gpu9，`EXIT=0`）

> **前一轮（2026-09-28）的 held-out 数据全部作废**，原因见「已修复的缺陷」第 14–16 条：
> `measure_window` 被传入**已训练**的 compactor 去算 `before`，而 `evaluate_window`
> 用同一个对象算 `after`；前向确定性 + 重建逐位相同 → `before == after`
> → `recovered` 被**结构性固定为 0**。因此当时的 `window_overfit` 与 `0/16`
> 都不是测量结果，是接线缺陷的产物。本轮已重写 `tools/e6b_mini.py` 并重跑。

**`verdict: partial`** —— 训练恢复 7/30，held-out 恢复 **2/16**。

| 集合 | 窗口数 | `lost` | 恢复 | `recovery_rate` | 95% CI |
|---|---|---|---|---|---|
| 训练 | 8 | 30 | 7 | 0.233 | [0.118, 0.409] |
| **held-out** | **2** | **16** | **2** | **0.125** | **[0.035, 0.360]** |

两侧差异检不出：Fisher 精确检验双侧 **p = 0.4628**；改用窗口级
Mann-Whitney（不假设探针独立，n₁=8 / n₂=2）**U = 8.0，z = 0.000**。
两个 95% CI 几乎完全重叠。**计划预设的「训练强、held-out 弱」对比在本轮数据里检不出。**

逐窗口（训练侧极不均匀，8 个窗口里 2 个逐位未变）：

| row | start | `lost` | 恢复 | rate | 答案集是否改变 |
|---|---|---|---|---|---|
| 3 | 17890 | 3 | 1 | 0.333 | 是 |
| 3 | 17130 | 2 | 0 | 0 | 是 |
| 3 | 16750 | 6 | 0 | 0 | **否** |
| 3 | 14470 | 5 | 0 | 0 | **否** |
| 3 | 17510 | 4 | 2 | 0.500 | 是 |
| 3 | 14850 | 3 | 0 | 0 | 是 |
| 3 | 16370 | 5 | 3 | 0.600 | 是 |
| 3 | 15230 | 2 | 1 | 0.500 | 是 |

| row | start | `lost` | 恢复 | rate |
|---|---|---|---|---|
| 2 | 3450 | 7 | 1 | 0.143 |
| 2 | 3830 | 9 | 1 | 0.111 |

held-out 两个窗口**都改变了答案集**，且签名链闭合：
`after == before` 均为 `False`；`trained_as_before == after` 均为 `True`
（即用训练前 compactor 去测量，得到的正是 `after` 的签名——证明 `before`
确实取自未训练状态）。完整 `integrity`：

```text
{'weights_identical': False, 'train_windows_changed': 6, 'held_windows_changed': 2,
 'determinism_checked': 2, 'held_all_unchanged': False, 'ok': True,
 'reason': 'weights changed and at least one window changed an answer'}
```

判定门槛在脚本中先定后用，不看结果再挑：held-out `lost = 0` → `undecidable`；
完整性检查未过 → `unmeasured`（**优先于其他分支**）；
`recovered = 0` → `window_overfit`；`rate ≥ 0.5` 且 `recovered ≥ 2` → `generalises`；
其余 → `partial`。

```text
verdict: {'outcome': 'partial',
          'reason': 'held-out recovery 2/16, below the 0.5 threshold',
          'thresholds': {'generalises_rate': 0.5, 'generalises_min_recovered': 2}}
```

**计划预设的两个实质分支都没有触发。** `generalises` 需要 rate ≥ 0.5 且
recovered ≥ 2，实测 0.125 / 2，未达；`window_overfit` 需要 recovered == 0，
实测 2，未达。落到了兜底的 `partial`——而 `COMPACTOR_PLAN.md` 的判定表
原本没有把这个分支当作真实结局来讨论。

### 训练轨迹

内部量跨 16 个窗口单调上升，**从未回退**：

| 量 | Stage A 首 → 末 | Stage B 首 → 末 | 末值 |
|---|---|---|---|
| `gamma` | 0.002373 → 0.002431 | 0.002502 → **0.003023** | 预算 0.05 的 6.0% |
| `residual_ratio` | 0.000139 → 0.001445 | 0.001960 → **0.002983** | — |
| `route_scale` | 0.017989 → 0.018082 | 0.018546 → **0.023868** | — |
| `routing_entropy` | 2.9432 | 2.9432 → 2.9417 | 均匀上限 ln(19) = 2.9444 的 **0.99907×** |
| `forward_calls`（累计） | 9 → 65 | 113 → 456 | — |

Stage B 的 `gold` 与 `kd` 在 **8/8** 个窗口全部下降，按窗口
`gold` 降幅 5.4%–10.3%、`kd` 降幅 10.3%–18.1%。
`hidden` **没有一致方向**：8 个窗口中 4 个上升、4 个下降，
与 E6a 单窗口的"反向"读数一致——说明它不是一个可靠的验收量。

### 保留性读数（immediate vs final）

E6b-mini 每个训练窗口在自己 24 步结束时**立刻**测一次恢复，再在全轮结束、
后续窗口都训练完之后测第二次。两者的差把"步数不够"与"后来者覆盖前者"分开。
这是 E6a 无法产出的读数。

```text
retention: immediate 8/30 = 0.267 -> final 7/30 = 0.233 over 8 windows (net -1)
  window row 3 start 17890  immediate 1/3 final 1/3 delta +0
  window row 3 start 17130  immediate 0/2 final 0/2 delta +0
  window row 3 start 16750  immediate 0/6 final 0/6 delta +0
  window row 3 start 14470  immediate 0/5 final 0/5 delta +0
  window row 3 start 17510  immediate 2/4 final 2/4 delta +0
  window row 3 start 14850  immediate 0/3 final 0/3 delta +0
  window row 3 start 16370  immediate 4/5 final 3/5 delta -1
  window row 3 start 15230  immediate 1/2 final 1/2 delta +0
  never learned (zero even while latest): [[3, 17130, 2], [3, 16750, 6], [3, 14470, 5], [3, 14850, 3]]
retention control: last window immediate_sig == final_sig: True
retention reading: retained
```

**「后面的窗口覆盖了前面窗口」被否证。** `windows_zeroed_after_own_steps` 为空：
8 个窗口里 7 个的恢复数逐位不变，唯一的变化是窗口 16370 在后续训练中
从 4/5 掉到 3/5（净 −1，不是清零）。四个窗口（17130、16750、14470、14850）
在**自己 24 步刚结束时就已经是 0**，没有任何后来的窗口能把它们压成零。

本轮读数因此把候选约束收敛到**每窗口预算或 compactor 表达能力**二者之一，
而这条测量无法把二者分开：一个窗口在自己还是"最新窗口"时都恢复不出来，
"24 步太少"与"改不动"从分数上看完全一样。

`retention reading` 只作后续说明，**不参与 `verdict`**；判决仍只看 held-out。

### 读数

**本轮不能支持"窗口级过拟合"，也不能支持"泛化成立"。**

先排除结构性零：held-out 2/16，两个窗口各恢复 1 个（row2 3450 的
`In what city did Gamal Abdel Nasser die?`、row2 3830 的
`What sport is Atlético Choloma associated with?`），四处签名断言全部通过。
训练过的 compactor 确实能在**从未见过的窗口**上改变答案并救回被丢弃的事实。
0.125 撑不起"泛化成功"，但它是负结果可被阅读的前提——上一轮的 0/16 不具这个资格。

**过拟合判据未被支持。** 窗口级过拟合要求"训练强、held-out 弱"。实测训练 0.233、
held-out 0.125，Fisher p = 0.4628、窗口级 Mann-Whitney z = 0.000。
但这是**"没检出差异"，不是"两边相等"**：held-out 只有 2 个窗口、16 个 `lost` 探针，
检验功效极低，0.5 的真实差距也未必检得出。用这组数据否证过拟合，
正是把"没检出"当成"不存在"。

**"表达能力不足"作为根因也不能由本轮支持。** `COMPACTOR_PLAN.md` 原用三项证据，
逐项复核后只剩两项：

1. 「训练强 / held-out 弱」——**不成立**（差距检不出，故过拟合也不成立），
   但这只否证过拟合，不正面支持表达力不足。
2. 「过拟合假说无法解释在更多窗口上训练反而把训练侧拉低」——**必须撤回**。
   该现象不存在：窗口 17890 在 E6b-mini 是 `lost 3` 恢复 1 = 1/3，
   与 E6a 24 步的 1/3 **同一水平**。多窗口训练没有损害任何一个窗口。
3. `residual_ratio` 与 `gamma` 单调上涨、量级增长未换来功能恢复——**成立**。
   `residual_ratio` 涨 21×（0.000139 → 0.002983），但 16750/14470 这两个窗口
   的训练后答案集与初始压缩器逐位相同（合计 11 个 `lost` 零响应）。
   量级增长与逐窗口功能响应完全脱钩。

**`loss 降幅是反向指标` 不成立。** Spearman（Stage B 训练损失降幅 vs 恢复率，
n = 8）绝对降幅 **−0.4342**，精确置换检验双侧 **p = 0.2881**；改用相对降幅
**−0.2171**，**p = 0.6119**——两个口径都不显著。
只能说"无正相关"，不能说"反向"。但方向在两轮重跑里都一致：按 Stage B 总损失，
降幅最大的窗口 16750（绝对降幅 0.534）恢复 0/6，降幅最小的窗口 15230
（绝对降幅 0.221）恢复 1/2。样本量不足以下结论。

**未被排除的竞争假说：每窗口步数不足。** 这是本轮最需要写明的读数限制。

| 设定 | 每窗口 Stage B 步数 | `recovery_rate` | `routing_entropy` / 均匀 |
|---|---|---|---|
| E6a 单窗口 | 24 | 1/3 = 0.333 | 2.9432 / 0.99958× |
| E6a 单窗口 | 220 | 2/3 = 0.667 | **2.7600 / 0.93735×** |
| E6b-mini 8 窗口 | 24 | 7/30 = 0.233 | 2.9417 / 0.99907× |

同一窗口、同一超参，步数 24 → 220 把恢复率翻倍，并且**第一次让路由离开均匀分布**。
E6b-mini 把 224 步摊给 8 个窗口，路由熵停在 24 步的单窗口水平（0.99907× vs 0.99958×）。
**"每窗口 24 步预算不足"与"表达能力不足"在本轮数据上不可区分**，
而计划的两项论据（第二项已撤、第一项只否证过拟合）都不足以把二者分开。

保留性读数把第三个候选——"后续窗口覆盖先前窗口"——排除了，
故候选从三个收敛到两个，但这两个仍然不可区分。

一个仍然成立的读数限制：held-out 的 `lost` 为 7 与 9，分母比训练窗口（2–6）大得多，
因为它用了 60 probes 而训练用 20。分母放大是为了让 rate 有分辨力，
也意味着 held-out 测的是更宽的探针集。**2/16 不能单独当作"泛化成立"的证据**，
它只足以推翻"泛化为零"。

### 本轮的变量变动

`L_hidden` 本轮**没有**删除，故"多窗口"与"改 objective"两个变量没有同时变动。
`gate` 与 `trigger` 按方案要求未接入。

一处未被消除的变量：`run_e6b_mini.sh` 没有传 `--seed`，`torch.manual_seed`
因此不执行，`CarrierResidualCompactor` 的 `q/k/down/up` 初始化每次运行都不同。
前后两轮全部 16 个 stage 的 loss 逐条不同（如 Stage B 窗口 1 `4.4756→4.1252`
对 `4.4918→4.1718`），训练侧恢复数 6 → 7 即由该差异产生；
held-out 两侧逐位相同，属巧合而非保证。该事实记录于下方缺陷清单。

### 复现方式

```text
E6A_NEED_MB=21000 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/share/latent/mllm_env/bin/python /tmp/e6b_mini.py \
  --steps-a-per-window 4 --steps-b-per-window 24 \
  --train-windows 8 --eval-windows 2 \
  --probes 20 --eval-probes 60 --max-probe-tokens 256 \
  --lr 1e-3 --gamma-max 0.05 --min-probes 4 --min-lost 2 \
  --max-screen 66 --screen-budget-min 100 \
  --candidates /tmp/e6a_screen_data.json \
  --out /tmp/e6b_mini_report.json --ckpt /tmp/e6b_mini_ckpt.pt
```

驱动 `tools/e6b_mini.py`、启动脚本 `tools/run_e6b_mini.sh`、
完整报告 `runs/e6b_mini_report.json`。运行在 v100-2 gpu9，`EXIT=0`。

### 附：本轮显存

第一次实现在 Stage B 窗口 3 第 3 步 OOM（**缺陷 13**）：`probe_loss.backward()`
处请求 3.12 GiB，`of which 3.12 GiB is free`。修复后本轮到 `EXIT=0`，全程无 OOM。

### 附：与 E6a 的口径差异（不可直接相比）

| 项 | E6a 220 步 | E6b-mini |
|---|---|---|
| 窗口数 | 1 | 8 训练 + 2 held-out |
| 每窗口步数 | 220（Stage B） | 24（Stage B） |
| 训练 probe 数 | 20 | 20 |
| 评估 probe 数 | 20 | 训练 20 / held-out 60 |
| 报告口径 | 单窗口 `recovery_rate` | 汇总 `recovered / lost` |

E6b-mini 的 24 步/窗口是**刻意压低**的：目的是在有限机时内暴露"跨窗口是否成立"，
不是复现 E6a 的绝对水平。因此"7/30 低于 2/3"本身不构成对 E6a 的否证。
但本轮 held-out 非零（2/16），也不构成"泛化失败"的否证——
这正是判为 `partial` 而不是任何二元结论的原因。**该口径差异是下一轮必须消除的混淆**：
每窗口步数需要提到与 E6a 可比的量级，否则窗口数与步数两个变量始终纠缠在一起。

---

## 参数清单

### 三次运行共用的固定参数

| 参数 | 值 | 来源 |
|---|---|---|
| `model_path` | `/share/latent/MemoryLLM-8B` | 基线 |
| `dtype` | `float16` | 基线 |
| `inject_max_tokens` | 512 | 模型 `max_length` |
| `max_new_tokens` | 32 | 基线（`generation_max_length` 优先） |
| `temperature` | 0.0，`do_sample False` | 基线 |
| `chunk_size` | 400 | 留出 bench 模板的约 60 token |
| `k_anchor` | 512 | = 2 块 |
| `anchor_source` | `first_chunk` | 设计文档 |
| `write_mode` | `append` | 刻意固定，隔离变量 |
| `gate_mode` | `novelty` | OURS 默认 |
| `projection` | `none` | OURS 默认 |
| `basis_max_rank` / `rank_max` | 256 | 对齐 `num_tokens` |
| `basis_energy` | 0.95 | 占位 |
| `compact_ratio` | 0.5 | 对齐 |
| `compact_min_slots` | 256 | 1 块 |
| `importance_weights` | goal 1.0, emphasis 1.0, 其余 0.5 | 占位 |

工作容量 = `12800 − 512 = 12288` = 48 块。硬约束：
`tau_k ≤ 1 − 2×256/12288 = 0.958`。

### 各运行的差异（完整）

除下表外，所有运行共用上表的固定参数。`mask` = `active_memory_mask`，
`filler` = `native_filler_only`。

| 运行 | 配置文件 | gate | trigger | consec | compactor | mask | filler | probe | queries |
|---|---|---|---|---|---|---|---|---|---|
| E1 smoke | `prlmc_smoke` | novelty | `occupancy_novelty_rank` | 5 | `svd_tokens` | false | false | false | 2 |
| E2 obs | `prlmc_obs` | novelty | `occupancy_novelty_rank` | 5 | **`none`** | false | false | false | 8 |
| E3 | `prlmc_r1` | novelty | **`occupancy`** | **1** | `svd_tokens` | false | false | false | **0** |
| E4 A | `prlmc_a` | **ones** | `occupancy` | 1 | **`none`** | false | false | false | 0 |
| E4 B | `prlmc_b` | novelty | `occupancy` | 1 | **`none`** | false | false | false | 0 |
| E4 C | `prlmc_c` | **ones** | `occupancy` | 1 | **`svd_tokens`** | false | false | false | 0 |
| E5a | `prlmc_e5a` | ones | `occupancy` | 1 | `recent_blocks` | false | false | false | 0 |
| E5b | `prlmc_e5b` | ones | `occupancy` | 1 | `select_blocks` | false | false | false | 0 |
| E5c | `prlmc_e5c` | ones | `occupancy` | 1 | `select_blocks` | **true** | **true** | false | 0 |
| E5d | `prlmc_e5d` | ones | `occupancy` | 1 | `select_blocks` | true | **false** | false | 0 |
| E6 | `prlmc_e6` | ones | `occupancy` | 1 | **`carrier_residual`** | true | true | false | 0 |
| E7 | — | — | — | — | — | — | — | — | 未开始 |
| 默认 | `prlmc_8b` | novelty | `occupancy_novelty_rank` | 5 | `svd_tokens` | false | false | **true** | 0 |

E6 另加 `carrier_rank: 32`、`carrier_gamma_max: 0.1`，**未设 `carrier_ckpt`**。

### 三组关键对照的变量

| 对照 | 唯一变量 | 结论 |
|---|---|---|
| E4 A (27.00) → C (0.00) | `compact_mode: none → svd_tokens` | 压缩器是充分原因 |
| E4 B (18.00) → D (0.00) | 同上，gate=novelty | 换一条路径复现 |
| E5a ≡ E5b (E5a 26.00) | `recent_blocks → select_blocks` | 逐条 100/100 相同，打分选择未证明有效 |
| E5c → E5d | `native_filler_only: true → false` | 97/100 相同，排除开关混淆 |
| E5c → E6 | `select_blocks → carrier_residual` | 98/100 相同，1 条 EM 差异在噪声内 |

### `compact_probe` 在所有已跑配置中都是 `false`

不是为省时间关掉的，而是它本来就无效：`agent.prlmc_probes` 从未被赋值，
`register_probes()` 没有调用点。默认配置 `prlmc_8b.yaml` 里写的 `true` 同样不起作用。

---

## 已修复的缺陷

E3 之前修复的问题，按发现顺序：

1. **触发器被抢跑（根因）。** `write()` 原顺序为 `_place` → `_observe_capacity`。
   `_place` 在块放不下时自己就会压缩，而"超容量"只能由"需要空间"造成，
   因此触发器看到快照时超容量状态永远已解决。这是 E1 中触发 0/82 的真正机制。
   **修复**：触发器判定移到放置之前，`_place` 的压缩降级为兜底。

2. **novelty 数学口径不一致（三处）。** `summarise_block` 跨 layer 平均；
   candidate 未减 `basis_mean`；`_spectrum`/`_refresh_basis` 用 layer 0 而 candidate 用 32 层平均。
   **修复**：统一到 `_REFERENCE_LAYER = 0`，candidate 减同一 `basis_mean`。

3. **`over_budget` 恒为假。** `_headroom_handled` 位于 `while` 循环体内，
   而循环条件本身要求超容量，故标志恒真、`over_budget` 恒假。
   **修复**：重命名为 `made_room_for_write` 并移入循环体，
   `over_budget` 改为在放置**前**计算，语义为"本次写入请求会超容量"。

4. **hysteresis 从未生效。** `tau_k_exit`/`tau_n_exit` 虽被读取，但 `armed` 只在
   触发的那一次置真，紧接着 `on_compacted()` 就清除，exit 阈值从未在 armed 状态下被读到。
   **修复**：`armed` 在超过 `trigger_consecutive` 后置真、由压缩清除；
   `_place` 兜底压缩也会清除 latch。

5. **rank veto 否决一切。** `occupancy_novelty_rank` 在 `energy_rank >= rank_max`
   或 `rank_utilisation >= 1.0` 时置 `above_n = False`。实测 494/500 满足前者、
   36/500 满足后者，导致 0/500 通过。**修复**：删除该 veto。

6. **novelty 不等号方向写反。** 设计文档为 `rho_bar_t >= tau_N` 才压缩，
   标定脚本曾测 `novelty < tau_n`。**修复**：改为正确方向（500/500，说明该条件为空条件）。

7. **`stats_dir` 位置错误。** YAML 顶层的 `stats_dir` 被 `build_config` 静默忽略，
   统计落到 `outputs/prlmc-obs/prlmc_stats.jsonl` 而非 `outputs/prlmc-obs/stats/`。
   **修复**：`stats_dir` 放入 `prlmc:` 块内。

8. **`KeyError('model')`** —— `agent.py:53` 需要 `model` 与 `temperature` 键。
   **修复**：补入配置文件。

9. **打分跨度落在提示词中间（E6a 全部数字作废的根因）。**
   `score_probes` 与 `_probe_batches` 都写作
   `n_prefix = min(len(stem), width - 1)`。渲染后的 `rag_agent` 提示词长 **163 token**，
   而 `max_probe_tokens` 默认 **64**，于是 `n_prefix = 63`，被打分的位置是
   "answer a question" 里的 `' question'`。该位置只由问题文本决定，与 memory 池无关，
   所以 accuracy 恒为满分、`gold_loss` 也不是答案的 loss。
   **修复**：两处都改为 `n_prefix = len(prefix_ids)`，并在
   `len(prefix_ids) >= max_length` 时直接 `raise`；默认值提到 256。

10. **`weights` 没有接进 compactor 调用。**
    `compressed_pool` 与训练器两处调用 `compactor(...)` 时都漏了 `train_weights`，
    于是选择臂用的是未加权的 importance，与 `_select_block_indices` 内部实际使用的
    0.4 加权 importance 不一致。
    **修复**：三处都按第二个位置参数传入 `train_weights`。
    生效证据是筛窗打印的 dropped 集合发生变化：
    `[0,1,2,3,4,5,6,7,8,9] vs [0,1,2,3,4,5,6,8,9,11] (unweighted)`。

11. **rebuild 写在注入上下文内部（梯度走了另一个 carrier 集合）。**
    `_pool_in_place` 在 `with` 块存续期间把 candidate 写进窗口尾部
    （`new_start = start + window - target`），退出时才 `restore()`。
    而 `manager._memory` 返回的是 `model.memory.data`，即 `source` 是池存储的视图。
    原来的 rebuild 写在 `with` 块内，因此读到的 `source` 已被覆写，且同一份被污染的
    `source` 同时用于三处：`_select_block_indices`（对 19 个 block 全部打分，
    含被覆写的尾部）、`src_summary`/`car_summary` 的路由均值、
    以及尾部 carrier 的 `blocks.index_select`。在该窗口上加权选择恰好丢掉
    block 0–9，于是尾部正是 carrier 集合——**反向传播穿过的是与 forward 不同的
    carrier 集合**。报出的 loss 看起来正常，因为它们来自正确构造的 `candidate_val`。
    **修复**：把 rebuild 与 backward 移出 `with` 块。
    `restore()` 已先执行，`source` 与参数均未变，重建结果是逐位相同的。
    AST 校验：`with` 块内 `cand_leaf.grad is not None` 节点数 **0**。

12. **回滚副本整步占用显存（首次训练在 backward 处 OOM）。**
    `_PoolInjection.__init__` 克隆被顶掉的窗口槽位后把 604 MB 留在 GPU 上，
    但 `restore()` 只在步末读一次，且该张量不参与 autograd。
    与此同时 backward 必须物化全宽 `[32, 12800, 4096]` fp16 的 memory 梯度
    （**3.12 GiB**，读全宽就是必须的），当时只剩 3.09 GiB，**差 30 MB**。
    **修复**：副本存到 CPU，`restore()` 时再传回；每步多一次 604 MB 拷贝，
    峰值让出 604 MB。

13. **多窗口训练在窗口边界因分配器碎片 OOM（E6b-mini 首次运行）。**
     崩在 Stage B 窗口 3 第 3 步的 `probe_loss.backward()`：
     请求 3.12 GiB，`of which 3.12 GiB is free`。

     诊断由数字决定，不是猜的：`reserved` 从 Stage A 窗口 1 第 1 步就冲到
     **33.15 GB**，此后 119 步一直钉在 33.6 GB 天花板；`alloc` 稳定在
     22.73/23.33 GB，**完全持平**——没有增长，故不是泄漏。崩溃瞬间活分配
     27.45 GiB、**缓存 766.85 MiB 未分配**，请求 3.12 GiB、空闲 3.12 GiB。
     那 766 MiB 缓存就是全部缺口。

     E6a 在同一包络内跑完 236 步无 OOM，因为它**全程只有一个窗口**，
     probe 集固定、激活形状固定，分配器很快进入稳定态。E6b-mini 每个窗口的
     probe 由该窗口事实行构造，token 长度逐窗口变化，形状一变，上一个窗口
     的缓存块就对不上新请求。

     **修复**：窗口边界调用 `release()` 清空缓存，把分配器拉回单一形状族的
     稳态；`train_window()` 在 OOM 时先回滚 `compactor.state_dict()` 与
     `optimiser.state_dict()` 再重试**一次**——窗口有 24 步，OOM 可能落在中途，
     不回滚会把部分窗口训练两遍，污染"共享 checkpoint"这个前提。修复后
     `EXIT=0`。

     **同轮排除的误判**：OOM 文本里那句
     `try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 是
     **无条件打印**的。在 v100-2 上实测（`tools/alloc_probe.py`），
     设置该变量后同一句话照样出现，故它不能作为"变量未生效"的证据。

14. **【致命，已修】E6b-mini 的 held-out 度量被结构性固定为 0（`window_overfit` 判决作废）。**
     旧 `tools/e6b_mini.py:729` 把**已训练**的 `compactor` 传进 `measure_window`，
     而该函数正是用它去算 `before`；`evaluate_window` 随后用**同一个对象**算 `after`。
     compactor 前向确定性、窗口重建逐位相同、probe 集复用
     → `before == after` → `recovered = lost & after_correct` 恒为空集
     → **任何训练都无法产出非零 held-out 恢复**。

     这不是"泛化失败"，是度量通道没接上：旧实现的 `0/16` 与
     `verdict: window_overfit` **全部作废**，其下游结论
     （「不是过拟合，是表达能力不足」所依赖的零值证据）随之作废。

     **修复**：`measure_window` 增加独立的 `template` + `initial_state` 参数，
     在函数内构造一份 `before_compactor`（`copy.deepcopy` 模板后
     `load_state_dict(copy.deepcopy(initial_state))`），保证 `before` 永远取自
     训练前状态，与调用方持有的 `compactor` 无关。修复后 held-out 两侧
     `after == before` 均为 `False`、`trained_as_before == after` 均为 `True`。

15. **【已修】held-out 的 `n_lost` 取自错误的对象。**
     旧实现里 held-out 的 `n_lost` 是用**训练后**的 compactor 在 60 probes 上算的，
     而同一窗口在筛选阶段（20 probes、初始 compactor）是 `lost` 3 / 6。
     两个数字口径不同却被并列写进报告，制造了"held-out 分母 7/9 远大于训练窗口 2–6"
     的假象。**修复**后 `n_lost` 一律来自初始 compactor，
     筛选值与报告值重新对齐。

16. **【已修】`recovered == 0` 的判定分支没有完整性闸门。**
     旧 `tools/e6b_mini.py:398` 把 `n_recovered == 0` 直接映射到 `window_overfit`，
     而结构性零恰好落进这一分支——**缺陷被自动翻译成了科学结论**。
     修复后 `verdict` 先查 `integrity["ok"]`，未过则输出 `unmeasured` 并附原因；
     另有三条前置断言：无训练阶段 → `"no training stage ran; nothing was trained"`；
     `state_dict` 逐位相同 → `"every update was discarded (NaN gradients?)"`；
     任何窗口都未改变答案 → `"`after` is the initial compactor, so `recovered` was not measured"`。

     另修复一处会用错的判读：`held_all_unchanged` 只作为附注字符串输出，
     **不折进结果分支**——单个窗口碰巧未变不能否决整轮测量。

17. **【未修】E6b-mini 的 compactor 初始化未被固定。**
     `tools/e6b_mini.py` 只在 `if args.seed is not None:` 分支里调用
     `torch.manual_seed(args.seed)`，而 `tools/run_e6b_mini.sh` 不传 `--seed`，
     故该调用从未执行，`CarrierResidualCompactor` 的
     `q / k / down / up` 四个 `nn.Linear` 每次运行都取不同的随机初始化。

     实测后果：同一脚本、同一窗口集、同一探针集的两轮运行，
     16 个 stage 的 loss 逐条不同（Stage B 窗口 1 `4.4756 → 4.1252` 对
     `4.4918 → 4.1718`；窗口 8 `3.3743 → 3.1534` 对 `3.4012 → 3.1865`）；
     训练侧恢复数 6 → 7 即由该差异产生。
     `final_state` 四项只差第 4–6 位有效数字，但逐窗口答案集签名不同：
     旧轮 `after == before` 为 `True` 的是 16750/14470，新轮也是这两个，
     而 17130 由 `True` 变 `False`（`n_lost` 两轮恒为 `3,2,6,5,4,3,5,2`）。

     held-out 两侧的恢复数与签名在两轮里逐位相同（`2/16`、两个窗口各 1 个），
     这属于巧合，不能当作可复现性的证据。

     该缺陷不影响本轮结论的方向（`partial`、两侧差异检不出两轮都成立），
     但意味着**报告里的逐窗口数字不可按位复现**。

---

## 未解问题

1. **`memory_construction_time` 慢 21 倍**：E3 为 4.9943 s/chunk，基线 0.2366 s。
   与 accuracy 无关，但 `_refresh_basis` 每次写入都做 SVD，新代码又在
   `_maybe_compact` 前额外算一次 spectrum（已用 `carry=pre` 参数消除重复 SVD）。

2. **`encode_seconds` 口径不一致**：E2 为 p50 25.73 s，E3 为 p50 0.1665 s，
   差 150 倍。E3 的 `memory_construction_time` 却比 E2 快 100 倍。
   两次运行的 `encode_seconds` 与 `memory_construction_time` 都不自洽，
   需要核对计时起止点。

3. **`compact_probe` 惰性**：`register_probes`（`agent.py:356`）无调用点，
   `agent.prlmc_probes` 永不存在，`compact_probe: true` 完全不起作用。
   设计文档 3.4 要求的 probe-guided compaction 从未被执行。

4. **`BudgetReport.byte_saving_ratio` 恒为 0.0**：池是固定形状的参数且被整读，
   压缩减少的是可寻址 token 数，不是已分配字节。这是构造性的，不是 bug。

5. **`detective_qa` 基线退化**：n=6，`output_len 2000`，
   `substring_exact_match` 与 `rougeL_recall` 均为 100，不可汇总。

6. **多卡不可用**：`device_map="auto"` 会破坏 `cat_memory_and_hiddens` 的
   逐层全池读取。可行路线是逐层流水线 + 每卡复制池（池仅 3.36 GB fp16），
   属于日级改动。

7. **E6 的 1 条逐条差异未闭合**：无 `carrier_ckpt` 时 `up` 为零矩阵，
   输出应当与 `select_blocks` 逐位相同（CPU 单测为 0.0），但实测与 E5c
   有 1/100 条 EM 差异。最可能是 `torch.randperm` 的随机性——所有配置的
   `seed` 都是 `None`。要闭合需在同一配置下复跑一次 E5c 以测量噪声基线，
   或把 `seed` 固定后重跑 E5c 与 E6。

8. **所有配置的 `seed` 均为 `None`**：这意味着任何逐条比对都混入了运行间
   随机性，1–3 条量级的差异无法归因。n=100 时 SE ≈ 6.1 分，因此平均分的
   排序同样不可靠。**在固定 seed 之前，不要基于逐条差异下结论。**

9. **probe 打分跨度落在提示词中间（E6a 首轮全部指标作废的根因）。**
   `score_probes` 与 `_probe_batches` 都写 `n_prefix = min(len(stem), width - 1)`。
   渲染后的 `rag_agent` 提示词长 **163** token，而 `max_probe_tokens` 默认 **64**，
   于是 `min(163, 63) = 63`，被打分的"答案"是 `' question'`——"answer a question"
   里的那个词，只靠问题文本即可预测。
   后果：accuracy 与 memory 池完全无关，因此

   - `lost` 在 12 个窗口上恒为 0（`select` 必然等于 `full`）；
   - 把 mask 全部置为屏蔽后仍得 20/20，被误读为"mask 失效"；
   - 把 compacted 池里被丢弃的 block 清零后仍得 20/20。

   SDPA 追踪给出决定性反证：`mask-all-off` 屏蔽了全部 12800 个池位置
   （`pool_all_blocked: 64`，`blocked_frac 0.997`），分数仍是 20/20。
   全屏蔽的池必须等价于无记忆（0/20）。**mask 本身没有问题，先前的 `empty-mask`
   异常随之作废。**

   **修复**：`n_prefix = len(prefix_ids)`，且在 `len(prefix_ids) >= max_length`
   时直接 `raise`，不再静默评错 token；`max_probe_tokens` 默认 64 → **256**
   （`tools/e6a_overfit.py`、`methods/prlmc/probes.py` 两处调用点同改）。
   `score_candidate` 与 `train_carrier_compactor` 的默认值同为 256。

   **影响范围**：`_probe_batches` 同时供 trainer 使用，故 E6a 首轮的 `gold_loss`
   也是错跨度的产物，不只是 accuracy。修复后首个窗口即得 `lost = 3`。

   **同轮修复的另两个真缺陷**：
   `compressed_pool` 的 `weights` 未传入 compactor（加权与不加权选出的
   dropped 集合确实不同：`[0..9]` vs `[0,1,2,3,4,5,6,8,9,11]`）；
   `_compacted_pool` 宽度越界（窗口只剩 18 block 时仍请求 19 block，
   `new_used` 变负，产出 13056 宽）。


---

## 复现方式

所有运行均在 `/share/latent/PRLM-C`（MemoryAgentBench 的副本），
`apply_patch.py` 已把 `prlmc` 分支装入宿主 `agent.py` 与 `utils/templates.py`。
宿主无 `git`，每次本地改动都必须重新上传。

```bash
# E1
GPU=6 QUERIES=2 bash run_smoke.sh
# E2
GPU=6 QUERIES=8 bash run_obs.sh
# E3
GPU=6 QUERIES=0 bash run_r1.sh
# E4（三格串行；D 即 E3，不需重跑）
CELL=a GPU=6 QUERIES=0 bash run_2x2.sh
CELL=b GPU=6 QUERIES=0 bash run_2x2.sh
CELL=c GPU=6 QUERIES=0 bash run_2x2.sh
# 或一次串起：bash chain_2x2.sh
# E5a / E5b / E5c
RUN=e5a GPU=6 QUERIES=0 bash run_e5.sh
RUN=e5b GPU=6 QUERIES=0 bash run_e5.sh
RUN=e5c GPU=6 QUERIES=0 bash run_e5.sh
RUN=e5d GPU=6 QUERIES=0 bash run_e5.sh
RUN=e6  GPU=6 QUERIES=0 bash run_e5.sh
```

集群紧张时用重试器（自动挑最空的卡并重试）：

```bash
RUN=e6 QUERIES=0 NEED_MB=21000 bash retry_run.sh
```

**不要在 `retry_run.sh` 运行期间覆盖该文件**：bash 按字节偏移增量读取脚本，
中途覆盖会让下次读取落到半行。E6 第一次投递就是这样在收尾阶段报出
`genuine: command not found` 的。

E5c 需要先给 modeling fork 打补丁（该文件不在本地 git 里，`tools/patch_modeling.py`
是幂等且可回滚的）：

```bash
# 状态检查（只读）
/share/latent/mllm_env/bin/python tools/patch_modeling.py --root /share/latent/PRLM-C --check
# 应用
/share/latent/mllm_env/bin/python tools/patch_modeling.py --root /share/latent/PRLM-C
# 回滚
/share/latent/mllm_env/bin/python tools/patch_modeling.py --root /share/latent/PRLM-C --revert
```

未打补丁时 `active_memory_mask: true` **静默失效**（模型没有该 buffer，
`_sync_active_mask` 直接返回），池子仍按原样读取。

运行结果落在 `outputs/<run>/`，内部信号落在同目录的 `prlmc_stats.jsonl`，
每次查询追加一行，`writes_since_last_query` 含该查询之前的所有写入记录。
压缩事件以 `{"event": "compact", ...}` 独立行写入同一列表。

`tools/calibrate_obs.py` 读取 `prlmc_stats.jsonl` 并打印上述信号分布。
