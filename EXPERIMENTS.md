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
| E4 | 待跑 | 分离 gate 幅值与 compactor | 未开始 |

线性读法：E1 发现触发器不触发 → E2 标定阈值 → E3 修触发器后首次测 accuracy → **发现 accuracy 归零，根因未分离**。

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

### 结论

按设计文档"occupancy-only 跑不好就先修 compactor"的判据，**下一步是修 compactor，
不是继续调 trigger**。触发机制本身已验证可用（问题 1、2 均为正面结果）。

---

## E4 — 待跑：分离 gate 幅值与 compactor

**唯一必要的实验**：新代码 + `compact_mode: none`，`factconsolidation_sh_32k` 全量 100 查询。
配置上只把 `prlmc_r1.yaml` 的 `compact_mode` 改为 `none`，其余不动。

判据：

- 若也是 0.0 → gate 幅值是**第二个独立杀手**，`tau_alloc` 与 gate 定义需要重做。
- 若接近 22.0 → 只有 `svd_tokens` 有问题，gate 修正无害，可直接进 Round 2。

**预估成本**：约 21 分钟（83 chunk × 4.99 s + 100 查询），比 E3 的 1273.7 s 更短。

### Round 2 待办（代码已写，未上传未验证）

按依赖顺序：

1. 修 novelty 数学定义 —— **已完成**：`_REFERENCE_LAYER = 0` 统一 basis/spectrum/candidate
   口径，candidate 减同一个 `basis_mean`。
2. 实现 low-novelty 更新已有 block / high-novelty 才 append —— **代码已写**：
   `_merge_into_block`、`_address`、`allocated_new_slots`、`write_mode: additive`。
   判据为 `rho < tau_alloc OR cosine >= address_threshold`。
3. 重新设计 adaptive trigger —— 围绕 allocation pressure，而非 occupancy ∧ novelty。
4. 接 functional probes —— `register_probes` 需要调用点，否则 `compact_probe` 永远惰性。

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

### 三次运行的差异

| 参数 | E1 smoke | E2 obs | E3 r1 |
|---|---|---|---|
| `trigger` | `occupancy_novelty_rank` | `occupancy_novelty_rank` | **`occupancy`** |
| `tau_k` | 0.80 | 0.0 | 0.80 |
| `tau_k_exit` | 0.60 | 0.0 | 0.0 |
| `tau_n` | 0.885 | 0.50 | 0.50 |
| `tau_n_exit` | 0.880 | 0.30 | 0.30 |
| `trigger_consecutive` | 5 | 5 | **1** |
| `compact_mode` | `svd_tokens` | **`none`** | `svd_tokens` |
| `compact_probe` | false | false | false |
| `structural_ablation` | — | — | **`trigger_occupancy_only`** |
| `queries` | 2 | 8 | **0（不截断）** |

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
```

运行结果落在 `outputs/<run>/`，内部信号落在同目录的 `prlmc_stats.jsonl`，
每次查询追加一行，`writes_since_last_query` 含该查询之前的所有写入记录。
压缩事件以 `{"event": "compact", ...}` 独立行写入同一列表。

`tools/calibrate_obs.py` 读取 `prlmc_stats.jsonl` 并打印上述信号分布。
