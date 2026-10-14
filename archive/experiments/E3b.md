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

[← 索引：EXPERIMENTS.md](../EXPERIMENTS.md)
