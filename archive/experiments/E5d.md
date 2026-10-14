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

[← 索引：EXPERIMENTS.md](../EXPERIMENTS.md)
