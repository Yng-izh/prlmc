# 01 — V10：保流形槽位嫁接

实现：`methods/prlmc/manifold_graft.py`（算子）、`methods/prlmc/native_skip_merge.py`
（流式适配器 `NativeSkipSlotMergeV10`）。几何常量：`tools/v9_geometry_50block.py`。

## 一句话

保留的载体块只贡献**原生槽**；被丢弃块的内容不是"再编一个槽"，而是取**同块内的原生槽**
做保范数的测地压缩后放进嫁接槽。这样读者看到的每个槽都还在它自己的 K/V 流形上。

## 一次压缩（19 块 → 9 块）做什么

1. **载体块由选择器决定**，V10 不改这一步：`select_window_carrier_blocks` 在窗口内按内容
   相似度与代际年龄挑 9 个块。
2. **核心槽（216/块）**：载体块里按显著度（`‖slot‖ × (0.5+离块质心散度)`）取 top-216，
   逐位复制为输出槽，残差恒为 0。
3. **同块强化**：每个核心槽在**被丢弃块**里找路由相似度最高的一个槽，若 `cos > 阈值`，
   按 `sigmoid(homo_gate_logits)` 缩放做一次保 RMS 的混合（`_geodesic_rms_preserve`）。
   这只会把同源、同语义的槽往载体上"拉近"，不会引入新块。
4. **嫁接槽（40/块，共 9×40=360）**：被丢弃槽按 `leverage = ‖slot‖ × (0.6+散度) ×
   (0.5+桥接相关度) × 新颖度` 排序取前 360，每个嫁接槽：
   - 在**同块**内按路由相似度取邻居（`cos > intra_discard_cluster_cos_thr`），softmax
     （温度 = `cluster_logit_scale`）得到簇上下文；
   - 与簇上下文做 `weight=0.25` 的测地混合，**恢复原槽的逐层 RMS**（关键：键范数不塌）；
   - 再加一个**正交低秩上下文残差**：`down_proj(8组×rank32)` → GELU → `up_proj`，
     投影掉与原槽同向的分量，`tanh` 限幅后按 `gate × ortho_residual_max` 缩放；
   - 最后再做一次保 RMS 归一化。
5. **按源位置排序输出**：核心槽与嫁接槽按它们来自的原生位置交错排列，读者看到的是
   "时间顺序合理"的窗口，而不是"先 216 后 40"的两段拼接。
6. 遥测写进 `merge.last_fusion`：`bridge_slot_offsets`、`slot_source_blocks`、
   `core_slot_count`、`grafted_slots` 等，评测与训练报告都用它做审计。

## 为什么这样设计

- 前几代（v3–v9）在"学一个新的槽向量"上打转：残差预算 .05、低秩投影、门控、原生跳连，
  结果是读者拿到的槽逐渐离开自己的分布，融合读不出增益。
- V10 把自由度收在**同块原生槽的凸/测地组合**上：输出槽要么是原生槽，要么是保范数的
  同块压缩，读者不需要适应新流形。这也是"不改主线、只调内部架构"的最强约束形式。
- 残差预算从 .05 提到 .24，但只作用在嫁接槽（占 15.6%）：`.24/√.156 ≈ .61`，
  正好让 bridge 上限成为有效约束，核心槽（84.4%）逐位不变。

## 可训练参数

`route_q/route_k`（rank 32 路由残差）、`down_proj/up_proj`（8 组 × rank 32 正交上下文）、
`cluster_logit_scale`、`ortho_gate_logits`、`homo_gate_logits`。共 **2,359,313** 个。
`up_proj` 零初始化、`route_k` 零初始化，因此 step 0 精确等于 Selection。
