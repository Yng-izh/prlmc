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

[← 索引：EXPERIMENTS.md](../EXPERIMENTS.md)
