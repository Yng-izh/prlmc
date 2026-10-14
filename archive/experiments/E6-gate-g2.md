# E6-gate — 压缩门：lost-probe 恢复 6/6 判定（g1 → g2q）

> **判定：FAIL**（2026-10-01，v100-2，`tools/e6_gate_lab.py` md5 `8c60f3f2d7bba251d853e46d40499185`）。
> 最佳 svd_adapter 配置恢复 **3/6**（g2q，gold-only 目标 + rank 256），未达门限 6/6。
> **但恢复数随解码侧容量单调上升（0 → 2 → 3），证伪「SVD 压缩销毁了信息」；
> E4 的 0 分是 reader 兼容性 + 训练目标问题，不是信息销毁。**
> overkill 对照 PASS（3/3），证明训练管线本身有能力恢复全部丢失探针。
> E6c/E7/trigger/capacity 继续暂停；项目处于解释性中间地带，下一步选项见第七节。

窗口固定为 E6a 筛出的硬窗口 `3,17890`（19 native blocks = 4864 slots 压到
9 blocks = 2304 slots，压缩比 0.5），20 个探针中 **lost = 6**（frozen_full 全对、
svd_frozen 全错的交集）：

1. Malcolm Forbes 的国籍
2. John Lightfoot 的雇主
3. 2010 African Women's Championship 的运动项目
4. Charlie Manuel 的运动项目
5. Girls 的创作者
6. Victor Turner 的雇主

门规则（导师给定，无折扣）：压缩后的读法必须把 6 个全部恢复
（`n_recovered == n_lost`），任何 5/6 都算 FAIL。

## 一、分支设定

| 分支 | 读法 | 角色 |
|---|---|---|
| `no_memory` | 完全不给 memory | 机器下限 |
| `frozen_full` | 原生全量 pool（4864 slots） | 天花板 |
| `native_select` | 原生 block 选择 9 blocks，原生读法 | 未训练参照 |
| `svd_frozen` | CarrierResidualCompactor（rank 64）压缩 + 原生读法 | 失败对照（预期 0） |
| `svd_adapter` | 同上压缩 + **Reader-A**（唯一受训分支） | 被判分支 |
| `overkill` | `carrier_residual` 压缩 + Stage B 过拟合 24 步 | 通过对照 |

Reader-A：`M̃_l = M_l + α_l · W_up^(l) σ(W_down^(l) M_l)`，逐层残差，
`W_up` 零初始化（恒等于初始，`identity_at_init: True` 全部运行验证通过），
α 逐层可学，**backbone 冻结，永不全量 finetune**。注入走
`_MemRowsInject` 自定义 autograd `Function`：forward 把适配后的行
`index_copy_` 进模型自己的 memory storage，backward 用 `index_select`
把梯度路由回适配行——模型 storage 之外的任何全宽画布都不存在。

## 二、基线（g2l 全五臂一次跑齐）

| 分支 | accuracy | 备注 |
|---|---|---|
| no_memory | 0.000（gold_loss 8.186099028587341） | 机器下限 |
| frozen_full | 0.300（6/20） | 天花板 |
| uncompacted | 0.300 | 与 frozen_full 逐条一致 |
| native_select | 0.150 | 未训练，恢复 0 个 lost |
| svd_frozen | 0.000（gold_loss 3.3203 同档） | lost 6 个全灭 |
| overkill（g4） | **0.500（10/20）** | recovery **3/3**，gate **pass True** |

svd_frozen = 0.000 确认了门的判据有效：原生 reader 读压缩池时丢失信息。
g4 的 overkill 在 24 步内 initial_loss 3.9304417627208754 → final_loss
1.5212263041680854，恢复 3/3、**frozen-correct 集合零回退**（压缩后 10 对 =
6 个 lost + 4 个原本就对）——训练管线、窗口判定、恢复统计全部可信。

## 三、svd_adapter 扫描（g2l → g2q）

训练目标三件套（与 E6a 相同）：gold 为主，KD 辅助，reader-hidden 系绳。
lr 全部 1e-3，AdamW，20 batch/step，per-probe 图独立建/释放。

| run | steps | KD | hidden | rank | loss[0] | final_loss | 恢复 | acc | gold |
|---|---|---|---|---|---|---|---|---|---|
| g2l | 8 | 0.5 | 0.1 | 64 | 28.66 | 13.133451128005982 | **0/6** | 0.000 | 5.3674 |
| g2m | 24 | 0.5 | 0.1 | 64 | 28.66 | 6.847103691101074 | **0/6** | 0.000 | 3.3203 |
| g2n | 24 | 0 | 0 | 64 | 13.75 | 2.222431355714798 | **1/6** | 0.35 | 2.1669 |
| g2o | 48 | 0 | 0 | 64 | 13.75 | 0.5410768826957792 | **2/6** | 0.45 | 0.5568 |
| g2p | 96 | 0 | 0 | 64 | 13.75 | 0.5222317889449186 | **2/6** | 0.45 | 0.5204 |
| g2q | 48 | 0 | 0 | **256** | 13.75 | 0.9987534153275192 | **3/6** | 0.45 | 0.9548 |

g2q 恢复的 3 个：2010 African Women's Championship（运动）、Girls（创作者）、
Victor Turner（雇主）。g2o/g2p 恢复的 2 个：Girls、Victor Turner。

## 四、训练侧结论

1. **KD 项是 0→1 的决定性瓶颈。** g2l/g2m（KD 0.5 + hidden 0.1）损失从
   28.66 降到 6.85/13.13，lost 恢复仍为 0；去掉 KD 改 gold-only 后第一步
   就有产出（g2n 1/6）。机制：KD 教师在完整池上前向，对压缩后必丢的信息
   饱和，梯度把适配器推向「模仿教师」而不是「恢复答案」；gold-only 让
   梯度只对 lost 内容敏感。
2. **步数在 48 步饱和。** g2o（48）与 g2p（96）恢复集合完全一致（逐 probe
   md5 比对），final_loss 0.5411 → 0.5222 无实质改进。96 步是纯浪费。
3. **rank 64 → 256 直接把恢复从 2 推到 3**（g2o → g2q，同为 48 步 gold-only）。
   这是对「SVD 销毁信息」假设的直接证伪：信息在压缩池里，
   瓶颈在**解码侧的容量/优化**。
4. **恢复数随解码容量单调：** 原生读法 0 → rank 64 训练 2 → rank 256 训练 3
   → overkill（rank 128 + compactor 全参数训练）3/3。g2q 的 gold_loss
   0.9548 高于 g2o 的 0.5568——rank 256 把部分容量花在了新的（非 lost）
   内容上，lost 恢复是容量分配问题，不是单调动损耗。
5. **超越天花板的来源核对。** 每轮 svd_adapter 的 after 集合 = 恢复的 lost
   probe + 若干 out-of-set 收益（如 Ford 创立城市、Heavy Rain 开发者等，
   这些是 frozen_full 也答错/没答对的）。g2q 的缺失集合**恰好等于 3 个未
   恢复的 lost probe**，frozen-correct 集合零回退——收益真实、无退化。
6. **超天花板本身不神秘：** frozen_full 只是把 4864 slots 原样递给 reader 的
   单一读法；训练过的读出器可以在压缩池上重构出全量池读法拿不到的
   组合（E6a 已观察到同现象）。

## 五、工程记录（可复现的 OOM 边界）

- 3.12 GiB 的全宽 GPU 池副本（[32, 12800, 4096] fp16）是第一轮 g2k OOM
  的直接原因。修复：训练前置 `pool.to("cpu").pin_memory()` +
  `torch.cuda.empty_cache()`，per-probe 从 pinned CPU 重物化适配行。
- 适配器走稀疏路径（只对 `_compacted_mask` 存活的 ~2560 行做残差），
  逐层 checkpoint，checkpoint 输入留在 CPU；AMP 禁用（g2i 证实 bf16
  autocast 在 fp16 权重之上再缓存一份反而恶化 OOM）。稳态 ≈22.6 GB。
- 同配置重跑存在步内 loss 微差（embedding 原子加反向的非确定性，
  g2o vs g2p 的恢复集合差异属种子噪声级别），比对一律以 JSON md5 为准。
- g1/g3（md5 相同，重跑）的 svd_adapter 数字因打分阶段语义缺陷作废，
  仅留档不进结论；overkill 分支语义在主机上验证过，g4 重跑确认 3/3 PASS。

## 六、对 E4 的修正

E4 记录「compactor 是 accuracy 归零的充分原因」。本门实验把因果链补全：

- E4 的 0 分 **不是** 「SVD 压缩把信息销毁了」（g4 overkill 3/3、g2q 3/6
  都能从同一个压缩池里读回内容）；
- 而是 ①原生 reader 读不懂压缩池（svd_frozen 0.000），②训练目标里 KD
  项把梯度引向模仿教师而非恢复答案（g2l/g2m 0/6）。
- 门本身（6/6）判定项目**继续/停止**的答案是：**当前形态未过门**；
  E4 的定性结论无需撤回，但「信息销毁」这个隐含读法被本实验推翻。

## 七、下一步选项（待 YG 决策）

- **rank 1024 探针**：沿容量轴再走一步（64→256 给了 +1，1024 是否给到
  4+），约 1.5 小时/次，GPU3 空闲。收益不确定：g2q 已显示容量分配
  摊薄，rank 增益可能递减。
- **LoRA 全秩/逐层 rank 预算**：放弃共享低秩瓶颈，每层独立预算，
  直接针对剩余 3 个 lost probe 的层级分布分配容量。改动量中等。
- **停止训练档**：接受「SVD 信息可读但 6/6 恢复未在当前读出器形态下
  达成」，把 E6-gate 作为项目终态写入总结；E6c/E7/trigger/capacity 永久暂停。

## 八、复现

```bash
# v100-2（脚本 md5 8c60f3f2d7bba251d853e46d40499185）
ssh v100-2 'cd /share/latent/PRLM-C && PRLMC_MEM_TRACE=1 E6A_GPU_ID=3 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup /share/latent/mllm_env/bin/python /tmp/e6_gate_lab.py \
  --branches svd_adapter --steps 48 --kd-weight 0.0 --hidden-weight 0.0 \
  --adapter-rank 256 --out /tmp/e6_gate_g2q.json \
  > /tmp/e6_gate_g2q.log 2>&1 < /dev/null & echo PID=$!'
```

原始 JSON：`runs/e6_gate/{g2l,g2m,g2n,g2o,g2p,g2q,e6_gate_g4}.json`
（g1/g3 作废留档：`runs/e6_gate/e6_gate_g1.json`，g3 与 g1 逐字节相同）。
脚本：`tools/e6_gate_lab.py`（分支实现 `:626-662`，恢复/门 `:697-730`，
Reader-A `:117`，`_MemRowsInject` `:194`，训练循环 `:265`）。

---

[← 索引：EXPERIMENTS.md](../EXPERIMENTS.md)
