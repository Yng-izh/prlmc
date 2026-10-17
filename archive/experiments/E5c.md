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

[← 索引：EXPERIMENTS.md](../EXPERIMENTS.md)
