# Compactor 路线：从接线到训练

本文记录 2026-09-27 之后确定的下一阶段方案。前序结论见 `EXPERIMENTS.md` 的
「总结」一节：诊断已确立（合成向量是读取失败的充分原因），原生 selection 是安全
基线，active mask 可用，`carrier_residual` 接线完成但尚未训练。

## 项目状态

```text
SVD synthesis       → 已淘汰
native selection    → 安全基线
active mask         → 可用
carrier residual    → 已训练；单窗口 2/3，但 held-out 0/16，表达力不足
adaptive trigger    → 暂停
novelty gate        → 暂停
```

E5a/E5b 都是 EM 26，E5c/E5d/E6 在 23–24；E6 未训练且初始化严格退化为 native
selection。**因此没有理由继续改 trigger 或 selector 权重**，重心收敛到 compactor 本身。

最大的缺口：**还没有真正训练出一个能把被丢弃 block 的信息融合回 carrier、并且提升
功能保持的压缩器。** E6a（2026-09-25/28）把训练链接通并拿到两轮有效读数
（`recovery_rate` 1/3 → 2/3），但 `residual_ratio` 只到 0.26%。E6b-mini（2026-09-28）
把单窗口扩到 8 训练 + 2 held-out，**held-out 恢复 0/16，单窗口成绩不泛化**。
因此缺口不但成立，而且已经定位：问题不在泛化，在表达力。

---

## 代码改动 1：真正的 relative trust region

现状只约束 `gamma`，**没有约束 `||delta||`**。若训练后 `up()` 的 delta norm 增大
100 倍，`gamma_max = 0.1` 仍能产生远大于 carrier 的修改。

改成对 delta 做 RMS 归一后再按 carrier 的 RMS 缩放：

```python
delta = self.up(F.gelu(self.down(context - carrier)))

carrier_rms = carrier.float().pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()
delta_rms = delta.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()

delta = delta / (delta_rms + 1e-6)
delta = delta * carrier_rms

alpha = self.gamma_max * torch.sigmoid(self.gamma_logit)
out = carrier + alpha * delta
```

这样才真正得到：

$$
\mathrm{RMS}(\alpha\Delta) \le \alpha_{\max}\,\mathrm{RMS}(Y_0)
$$

第一轮取 `alpha_max = 0.05`，确认稳定后再试 0.1。这是防止再次跑出 SVD 那类 OOD
memory 的硬保险。

## 代码改动 2：跨 layer 共享 routing

现状 `attn: [L, B_out, B]` 且 `einsum("ljb,lbtd->ljtd")`，即**每一层可以从不同的
source block 融合**。但 MemoryLLM 一个原生 block 是整套 layer state 联合生成的
（layer 0…31 对应同一个 memory event）。逐层独立 routing 可能制造出训练分布里从未
出现过的「跨层拼接记忆」。

改为只在 reference representation 上算 routing：

```python
summary = source[reference_layer].view(B, block, D).mean(dim=1)   # [B, D]
carrier_summary = summary[carrier_idx]                             # [B_out, D]
attn = routing(carrier_summary, summary)                           # [B_out, B]
context = torch.einsum("jb,lbtd->ljtd", attn, source_blocks)
```

同一组 `attn[B_out, B]` 用于全部 32 层，压缩后的 memory block 在所有 layer 上仍对应
同一批源信息。

## 代码改动 3：routing 的 native cosine prior

现状 `q`/`k` 是 learned linear 而 `up = 0`。第一步 `delta = 0`，因此 loss 只对 `up`
有有效梯度，`q`/`k`/`down` 基本收不到有意义的梯度。不是 bug，但训练动力学很绕。

改为先验 + 学习项：

$$
A = A_{\text{native-sim}} + \lambda A_{\text{learned}}, \qquad \lambda_0 = 0
$$

```python
native_logits = (normalize(carrier_summary) @ normalize(summary).T) / temperature
learned_logits = (self.q(carrier_summary) @ self.k(summary).T) / sqrt(rank)
route_scale = torch.sigmoid(self.route_logit)
logits = native_logits + route_scale * learned_logits
attn = softmax(logits, dim=-1)
```

训练开始时至少是「carrier 优先吸收语义接近的 source block」，之后模型再渐进修改 routing。

---

## E6a：单窗口 overfit sanity check

**这是下一步唯一要做的事。** 判断当前 `CarrierResidualCompactor` 与整条
functional-loss 训练链能否工作的最快测试。

设计：

- 取一个真实的 19-block window，压成 9 blocks；
- 准备 16–32 个与窗口事实对应的 **train** probes；
- 只训练这一个窗口。

验收：

```text
训练前：select_blocks probe accuracy = X
训练后：carrier_residual probe accuracy > X
```

同时要求：

```text
generation 仍正常
residual / carrier RMS < 5%
gamma 没撞上限
KD loss 下降
reader hidden loss 下降
```

**如果一个窗口都 overfit 不了，不要跑任何大实验**，优先查训练图、memory
replacement、梯度流、loss 计算。这个实验非常便宜，价值极高。

### E6a 结果（2026-09-24，首轮指标作废；硬窗口已就绪）

首轮全跑的两组数字——`before 1.000 (16/16), gold_loss 0.8384` 与
`stage B gold_loss 0.3761`——**全部作废**，原因不是 loss 权重也不是 probe 集难度，
而是打分跨度错误：`n_prefix = min(len(stem), width - 1)` 被 `max_probe_tokens = 64`
夹住，而渲染后的提示词长 163 token，于是被打分的是提示词中间的一个词，
与 memory 池无关。accuracy 因此恒为满分，`gold_loss` 也不是答案的 loss。
详见 `EXPERIMENTS.md`「已修复的缺陷」第 9 条。

修复跨度后重新筛选硬窗口，首个候选即命中（`row 3 start 17890`）：

| 指标 | 数值 |
|---|---|
| `no-mem` | 0/20 |
| `full`（未压缩） | 6/20 |
| `select`（selection 后） | 3/20 |
| `lost` | **3** |
| `eligible` | 6 |

窗口构造要求支持事实落在 `select_blocks` **丢弃**的 block 里，因此 selection
确实丢信息。判定指标为 `recovery_rate = recovered / lost`，不看总体准确率
（20 题中只有 3 题 `lost`，信号会被稀释）。

首轮"验收条件在准确率上结构性无法满足"的观察仍然成立，但根因已更换：
不是 probe 恰好都是最新事实，而是跨度错误让 accuracy 与池无关。`after > before`
这一判据依旧不能用，仍须落在 `lost` 子集上。

附带确认的一项：`carrier_residual` 在初始化时与 `select_blocks` 逐位相同
（`identity_at_init: True`，`max |d| = 0.000e+00`），这是复用"先安全基线、
后小幅融合"路线的先决条件。

**E6a 尚未在硬窗口上训练。** 下一步唯一要做的事：固定 `--window 3,17890`，
用同一个 `CarrierResidualCompactor`（结构不动）训练，报 `recovery_rate`。

### E6a 硬窗口训练结果（2026-09-25，有效）

缺陷 9–12 全部修复后的首次有效训练。固定 `--window 3,17890`，结构未改，
loss 权重未调，stage A 16 步 + stage B 24 步，`lr 1e-3`，`gamma_max 0.05`。

筛选（`--dry-run`）与上表逐位一致：`no-mem 0/20`、`uncompacted 6/20`、
`before 3/20`、`lost 3`。

| 阶段 | `after` | `recovery_rate` | `gold` 首→末 | `kd` 首→末 |
|---|---|---|---|---|
| A（16 步） | 4/20 | **1/3 = 0.333** | 3.8737 → 3.7321 | 1.4617 → 1.0696 |
| B（24 步） | 4/20 | **1/3 = 0.333** | 3.7168 → 3.2787 | 1.0802 → 0.9928 |

恢复的 probe 是 `What sport is Charlie Manuel associated with?`；
`Who created Girls?` 与 `Who is Victor Turner employed by?` 仍丢失。

内部状态阶段末：`gamma` 0.0024011 → 0.0024612（预算 0.05，**用到 4.8%**）；
`residual_ratio` 0.00072533 → 0.00142481（判据要求 < 5%，**实测 0.14%**）；
`route_scale` 0.018158 → 0.018614；`routing_entropy` 2.9432（仍均匀）；
`hidden` 分量全程持平 ~0.0028，未下降；`amp` 0.0。

**这不是机制成立**：`residual_ratio` 停在 0.14%，注入张量与 `select_blocks`
的差异只占 carrier 的千分之一量级，融合项没有真正离开恒等点。
1/3 的恢复因此不能归因于 residual fusion。训练链本身是通的
（`carrier` 分量单调上升，`up.weight` 已离开零初始化；`kd` 明显下降），
但 16 + 24 步、`lr 1e-3` 下的有效梯度信号不足以推动 `gamma` 用掉预算。

下一步的判据不变：只有当 `residual_ratio` 真正放大、`after` 相对 `before`
在 `lost` 子集上有可重复提升，才能回去做 E7 adaptive trigger 与 E6b 多窗口。

### Stage B 延长到 220 步（2026-09-28）

同一窗口、同一超参，只把 Stage B 从 24 步延长到 220 步（Stage A 仍 16 步）。
运行 3 h 54 min，`EXIT=0`。

| 项 | 24 步 | 220 步 |
|---|---|---|
| `after` | 4/20 | **6/20** |
| `recovery_rate` | 1/3 | **2/3** |
| `residual_ratio` | 0.0014248 | **0.0026117** |
| `gamma` | 0.0024612 | 0.0030977（预算 6.2%） |
| `route_scale` | 0.018614 | 0.0267555 |
| `routing_entropy` | 2.9432118 | **2.7599626** |
| `gold` 首→末 | 3.7168 → 3.2787 | 3.6848 → 2.4318 |
| `kd` 首→末 | 1.0802 → 0.9928 | 1.0047 → 0.7586 |
| `hidden` 首→末 | 0.0029897 → 0.0029241 | 0.0028581 → **0.0043209** |
| `carrier` 首→末 | 4.26e-05 → 9.88e-05 | 4.37e-05 → 2.244e-04 |

`after 6/20` 追平 `uncompacted 6/20`；恢复的是
`What sport is Charlie Manuel associated with?` 与
`Who is Victor Turner employed by?`，未恢复的只有 `Who created Girls?`。
Stage A 两侧 `gamma` 逐位相同（0.0024011049），确认 Stage B 是唯一变量。

**判断**：`residual_ratio` 1.83×、`route_scale` 1.44×、`carrier` 2.27×、
`routing_entropy` 下降——四个量同向移动，第一次明确离开恒等点邻域，
比 `recovery_rate` 本身更能说明融合与路由被推动。
但 `residual_ratio` 仍是「< 5%」判据的约 1/19，`gamma` 只用掉预算 6.2%；
「reader hidden loss 下降」这条验收**反向**（`hidden` 上升）。
分母只有 3，`1/3 → 2/3` 是一道题的差别。因此**仍记为未通过**，
不改结构、不调 loss 权重，下一步仍是让 `residual_ratio` 真正放大。

## E6b-mini：多窗口离线训练（判定性，非放大）

E6a 已证明**单窗口**可以恢复到 2/3。E6b-mini 只问一件事：
换一个没见过的硬窗口，同一个 checkpoint 还能恢复吗。

与 E6a 的唯一变量是**窗口数量**。结构、loss、selector、trigger、gate 逐字不动，
所以结果正负都只有一种解释。8 个训练窗口（row 3）+ 2 个 held-out 窗口（row 2），
每窗口 Stage A 4 步 + Stage B 24 步，共 224 步；训练 20 probes、held-out 60 probes。

### 结果（2026-09-28，v100-2 gpu9，`EXIT=0`）

| 集合 | 窗口数 | `lost` | 恢复 | `recovery_rate` |
|---|---|---|---|---|
| 训练 | 8 | 30 | 7 | 0.233 |
| **held-out** | **2** | **16** | **0** | **0.000** |

`verdict: window_overfit` —— "held-out recovered nothing while training recovered 7/30"。

### 判断：不是过拟合，是表达能力不足

判据是**训练侧自身的分布**。窗口级过拟合的特征是"训练强、held-out 弱"；
实测训练侧只有 23%，且 **8 个窗口中 4 个完全为零**。恢复只发生在 4 个窗口上，
是"在少数窗口上碰巧有效"，不是"学得很好但不能泛化"。
**E6b 到此不是被否证，而是被诊断为还没到能泛化的表达力。**

第二项反证来自步数对照：E6a 单窗口 220 步拿到 2/3，这里每窗口 24 步就崩到 7/30。
过拟合假说无法解释"在更多窗口上训练反而把训练侧拉低"。

第三项是量级与功能的脱钩：`gamma` 与 `residual_ratio` 跨 16 个窗口**单调上涨**
（0.002373 → 0.003025、0.000139 → 0.002993），从未因换窗口塌陷，
说明优化器没在窗口间打架、量级在稳定增长。**但量级增长没有换来功能恢复。**
这是本轮最重要的负面证据：目前的 residual 形式不足以把被丢弃 block 的信息搬回 carrier。

### 读数限制（必须与结论一起引用）

held-out 的 `lost` 为 7 与 9，分母比训练窗口（2–6）大得多，因为它用了 60 probes
而训练用 20。放大分母是为了让 rate 有分辨力，但也使 held-out 测的是更宽的探针集。
**0/16 不能单独当作"泛化完全失败"的终局证据**；它与更强的证据（训练侧仅 23%、
半数为零）方向一致、相互加强。

### 下一步含义

要动的是**架构或目标**，不是继续加窗口、不是调 `gamma`/trigger/selector。
多窗口泛化失败时第一个嫌疑是 `L_hidden`——本轮**没有**删除它，
正是为了不让"多窗口"与"改 objective"两个变量同时变动。

---

## E6b（放大版）：多窗口离线训练

注：以下为 E6a 通过后设想的放大版形态，E6b-mini 已先行做了判定性版本。

不要在线每个 episode 临时训练。离线采样大量 `(memory window X, carrier set Y0,
available probes Q/A)` 组成 compactor training dataset，从训练流随机截
19→15 / 19→12 / 19→9。

一个 batch 本质：

```text
Teacher: uncompressed X   → frozen MemoryLLM → teacher logits / hidden
Student: carrier_residual(X) → frozen MemoryLLM → student logits / hidden
```

只有 compactor 参数更新。

### Loss 分两阶段，不要五项一起从第 1 step 开始

**Stage A（先学「别破坏 reader」）**：

$$
L_A = L_{KD} + 0.1L_{hidden} + 0.1L_{carrier} + 0.02L_{stats}
$$

先不追求超过 teacher，目标是 compressed reader behaviour ≈ uncompressed。

**Stage B（functional fine-tuning）**：

$$
L_B = L_{gold} + 0.5L_{KD} + 0.1L_{hidden} + 0.05L_{carrier} + 0.02L_{stats}
$$

好处：**先解决 representation compatibility，再解决 information selection**，
与整个项目的实验发现一致。

## 固定 selector

E5a 与 E5b 是 **100/100 完全相同输出**，数据未证明 importance/goal/diversity
selector 比 recency 有任何作用。训练 carrier compactor 时先固定：

```text
carrier_idx = select_blocks(...)
```

selector 参数完全不动。研究问题先变成：

> **在完全相同的 9 个 surviving blocks 下，能不能把另外 10 个 dropped blocks 的
> 功能信息融合进去？**

这样 E5c → trained E6 是非常干净的因果比较。若 trained E6 胜出，再回来研究
「应该选哪 9 个 carrier？」。

---

## 评价方式：compaction unit benchmark

直接跑 100 条最终 QA 太粗。每次真实 compaction 记录：

```text
probe_acc_before / probe_acc_select / probe_acc_carrier
KL_before_after
hidden_drift
carrier_rms / residual_rms / residual_ratio
gamma
routing_entropy
K → K'
```

重点画：

$$
\text{functional retention} = \frac{Acc_{\text{compressed}}}{Acc_{\text{before}}}
$$

并测 1 / 3 / 5 / 10 次 compact。真正危险的可能不是一次 compression，而是 residual
compactor 被**反复压缩**后逐步偏离 native manifold。核心结果可能是一张
accuracy–compaction count 曲线，而不只是最终 EM。

## Capacity stress setting（trained compactor 之后）

`sh_32k` 只有 83 chunks，compression 和 eviction 都能活，结果都在 22–27，看不出
compaction 的价值。两条压力轴：

```text
轴 1：K_max   100% / 75% / 50% / 25%
轴 2：horizon 32k / 64k / 128k / 262k
```

最关键的比较只有三个：**Eviction / Native block selection / Trained
CarrierResidual**，全部同 `K_max`、active mask、trigger、`gate=ones`、backbone。

若出现「K 较大时三者差不多，K 较小时 eviction ↓↓↓、selection ↓↓、carrier ↓」，
才算真正验证了核心贡献。

---

## 执行顺序

1. 补真正的 residual norm bound（不要只 bound `gamma`）。
2. routing 改成跨 layer 共享。
3. routing 加 native cosine prior。
4. **跑单窗口 overfit E6a。**
5. 跑多窗口离线训练 E6b：先 compatibility distillation，再 functional fine-tuning。
6. 固定 trained checkpoint，做 selection vs carrier 的 compaction-unit evaluation。
7. 进入小 `K_max` / 长 horizon 的 capacity stress test。
8. **只有 trained carrier residual 明确优于 native selection 后**，才回去做 E7
   adaptive trigger。

## 工程注意

`compact_probe` 在所有已跑配置里实际无效（`agent.prlmc_probes` 从未注册）。
训练**不要依赖这个 runtime flag**；先做独立的 offline compactor trainer + explicit
probe dataset，更干净，也更容易保证 train/test isolation。

## E6a 实现记录：四个必须先解决的坑

E6a 的代码在真正跑通前暴露了四个问题，都记在这里，避免 E6b 重踩。

### 1. 池宽必须保持 12800，不能缩窄

这个 modeling fork 的 `cache_position` 由 **config 常量**推出
（`num_tokens * num_blocks + add_bos_embedding`），不是由实际拼接的 memory 推出。
所以把池缩窄成 `[A,S,R]` 后，`rotary_emb` 仍按 12801 个位置生成 cos/sin，
而 q 更短，`apply_rotary_pos_emb` 直接抛
`The size of tensor a (12) must match the size of tensor b (10252)`。

结论：训练与打分都必须跑 **定宽池**。压缩后的结果是**重排**而不是缩短——
carrier 贴到尾部，被保护的 newest block 紧随其后，腾出的空位由 active mask
屏蔽（`_compacted_pool` / `_compacted_mask`）。这同时让训练和推理的 rotary
偏移完全一致，原先设计文档里担心的 train/inference 不一致因此不存在。

### 2. mask 必须跟着池一起改，否则测的不是压缩

`_compacted_pool` 只重排了 memory tensor。若不同步 `memory_active_mask`，
被丢弃的 slot 仍是 active 且内容还在，reader 会同时看到 carrier 和它替代掉的
block，probe 实际打分的仍是未压缩记忆。`active_mask_override` 负责同步。

### 3. probe 必须用推理时的真实 prompt

FactConsolidation 的每个问题外面套了一段很长的 "knowledge management system"
instruction，模型只在这个格式下回答。用裸 `Question\nAnswer:` 打分，
**未压缩满池也只有 0/16**。这会被误读成记忆坏了，实际是 probe 坏了；
更危险的是 before/after 会是 0 比 0，验收完全无效。`Probe.prompt` 保存
模板渲染后的 prompt，`_probe_batches` 优先用它。

### 4. probe 必须是"窗口负责回答"的事实

FactConsolidation 的规则是"序列号大的更新"，所以正确答案是**最新**的匹配事实。
另外测窗口压缩时，最新事实必须落在窗口内，否则答案在保留区，任何压缩都不影响它，
probe 分辨不出好坏。`probes_from_lines` 用 `first_match` 唯一匹配模板并按 subject
取最新行；模板重叠会导致 `Who is the capital of Great Britain?` 这类无法回答的问题。

修完后 dry run 曾报 `uncompacted = 1.0 (16/16)`、`before = 1.0 (16/16)`；
这两个数值后来确认为打分跨度错误的产物（见「首轮指标作废」），已作废。

### 显存

`consolidate.py` 的 batched einsum 会把 32 层整体转 fp32（2.5 GB 临时 + 1.2 GB 结果），
在 32 GB V100 上 OOM。改成按 layer 循环、`out` 预分配后，同一卡跑通。

### 5. OOM 的真正原因：reader 图和 compactor 图同时存活

第一次全跑在 stage A 第 0 步的 probe backward 里 OOM。逐项测量后的账目：

| 项目 | 大小 |
|---|---|
| 模型 | 19.57 GB |
| teacher pass（logits + 4 层 hidden × 8–16 probe） | 接近 20.0 GB |
| student 前置占用（`[mem] step 0 before student`） | **23.02 GB** |
| per-layer checkpoint 存下的 layer 输入 | 3.37 GB |
| 全池 `pool.grad`（32×12800×4096 fp16） | 3.36 GB |
| compactor 活图 | 3.02 GB |
| 合计 | ≫ 31.73 GB |

每一项单独都放得下，加起来放不下。曾有两条猜测被证伪：

1. **注意力分数矩阵**（以为 additive `-inf` mask 会逼回 math backend，物化
   32×12801×12801 的分数矩阵）。实测 mem-efficient 同时吃 bool 和 float `-inf` mask，
   forward 只要 1.24–1.46 GB，backward 1.87 GB。**证伪。**
2. **per-layer checkpoint 无用**。它确实有用，但 fork 在每一层循环里重新拼
   12800 前缀，所以每个被存下的 layer 输入都是满宽 `[1, 12801, 4096]` fp16
   ≈ 105 MB，32 层 ≈ **3.37 GB**，比它省下的激活还多。

真正的支配项是 **8B reader 图和 compactor 图同时活着**。三条改动把它拆开：

- **`_InjectSlice`（自定义 autograd.Function）**：往不 requires_grad 的 tensor
  里直接赋值会静默丢掉梯度（host 实测 `grad_fn: None`）。必须走 Function。
- **`_PoolInjection.reinject()`**：每个 probe 重新取一个注入节点。共用节点会逼着
  每次 backward 传 `retain_graph=True`，那等于让 16 份 reader 图同时存活，正好抵消
  per-layer checkpoint。每个 probe 用新节点 → reader 图 backward 完立刻释放。
- **梯度隔离在 `cand_leaf` 上**：`candidate` 只有 `[32, 2304, 4096]` fp16 = 604 MB；
  全池 leaf 会分配 3.36 GB 的 `pool.grad`。
- **compactor 图在 reader 之后重建**：先用 `torch.no_grad()` 算出 `candidate_val`
  喂给 reader，等 reader 图全没了，再用同样的 `source` 和参数重建一次 forward 并
  对它 backward。`forward` 是确定性函数且参数没变，重建的 tensor 与单次 forward
  完全一致，梯度也一致。这一步把 3.02 GB 的常驻图从 reader 阶段彻底移走。

改完后 `[mem] step 0 before student` 从 23.02 GB 降到 **21.20 GB**，smoke
（2+2 步、8 probe）完整跑通，stage A 的 step 0 backward 正常完成。

### 6. 第一次全跑"没效果"的真相：梯度是 NaN，训练被静默跳过

第一次完整 E6a 跑完后 `before` 和 `after` 逐位相同（下两行的数值是错跨度产物，
已作废；但「逐位相同」这一事实成立）：

```
before {'accuracy': 1.0, 'gold_loss': 0.8384018540382385}   # 跨度错误的数值
after  {'accuracy': 1.0, 'gold_loss': 0.8384018540382385}   # 跨度错误的数值
```

这不是"训练没用"，是**一个参数都没动**。`state_report` 里 `gamma = 0.0023712937`
与初始化值逐位相同，`carrier` 项 40 步全是 `0.0`（因为 `up.weight` 恒为 0）。

加 `PRLMC_GRAD_TRACE=1` 后看到真实原因：

```
[grad] step 0 cand_leaf=nan scale=65536.0 | gamma_logit=nan route_logit=nan ... up.weight=nan
[grad] step 1 cand_leaf=nan scale=32768.0 | ...全是 nan
```

`scale` 从 65536 掉到 32768 = `GradScaler` 检测到 inf/nan 后**每一步都跳过
`optimiser.step()`**，所以参数保持初始化值，而日志里的 loss 数字看起来仍然正常。
两个独立缺陷叠加：

1. **NaN 源头：`delta / (delta_rms + 1e-6)`。** `up.weight` 零初始化 ⇒ 第一步
   `delta ≡ 0` ⇒ `delta_rms = 0`。forward 无害（`0/1e-6 = 0`），但 backward 需要
   `d(delta_rms)/d(delta_ij) = delta_ij / (N·delta_rms)`，在原点正好是 `0/0`。
   这个 NaN 传导到除 `gamma_logit` 外的所有 compactor 参数（`gamma_logit` 有第二条
   到 loss 的路径才幸免）。
   **修法**：改用 RMSNorm 形式 `delta * rsqrt(delta_var + 1e-6)`。eps 放在根号
   **里面**，原点处梯度增益是有限的 `1/sqrt(1e-6) = 1000`；离开原点后与原式代数等价。
2. **放大器：`use_amp=True` 但没有 `autocast`。** 整条路径没有任何 `autocast`，
   backbone 本身就是 fp16。`GradScaler` 把 loss 乘 65536 再 backward，而 fp16
   激活梯度的上限是 65504，超过就 `inf`，`inf * 0 → nan`。
   **修法**：`use_amp` 默认改为 `False`（没有 autocast 时 scaler 只有害处）。

修后 smoke 梯度全部有限，且 loss 真的在降：

```
[grad] step 0 cand_leaf=1.0039e+01 scale=1.0 | up.weight=1.2535e+00 down.weight=0.0 ...
[grad] step 1 cand_leaf=7.3125e+00 scale=1.0 | up.weight=3.5428e-01 down.weight=8.0711e-01
stage B: gold_loss 0.8384 -> 0.7489   # 跨度错误的数值，仅说明 loss 在降，数值作废
```

**教训（已写进验收）**：overfit 验收不能只看 accuracy。必须在训练后断言
`gamma` / `up.weight` 相对初始化**确实变了**，否则"梯度是 NaN、更新被跳过"会伪装成
"训练无效"，浪费一整轮算力。

该断言在 2026-09-25 的硬窗口训练中通过：`gamma` 从 0.0023713 变到 0.0024612，
`up.weight` 的平方均值从 `0.0` 升到 `9.88e-05`，梯度不再是 NaN。
2026-09-28 的 220 步版进一步确认：`gamma` 到 0.0030977，`up.weight` 平方均值到
`2.244e-04`。剩下的问题是量级——`residual_ratio` 只到 0.26%，见上节。
