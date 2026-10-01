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

[← 索引：EXPERIMENTS.md](../EXPERIMENTS.md)
