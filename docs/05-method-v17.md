# V17：Key 锚定与三段带解耦、密度配额、嫁接注意力校准（代码口径）

本文只写**对当前代码有约束力**的内容。数字与结论见 [03-results.md](03-results.md)；
**本轮全部机制尚未运行**，跑完回填。V13 的设计见 [05-method-v13.md](05-method-v13.md)，
V14 的负结果见 [05-method-v14.md](05-method-v14.md)。

本文出现的所有"≥ +4 题""Harmed ≤ 2"一类的数字都是**运行前写死的门槛**，不是结果。
本轮没有任何一个机制测过。

## 一句话

窗口不换（2,304 槽：1,620 core + 684 graft，冻结 Reader/Writer），在 V13 的校准器上
加四件事，每件事都能在**同一 158 题**上单独开关：把残差拆成 reader 能"看见"的 key
部分和被 mask 掉的 null 部分并分别设上限（于是 key 侧被锁死、value 侧可以放开）；
把 5 个 utility 层分成三个带、逐带一个标量门（出厂层表下第 0 带没有窗口层，见 §2）；
把 1,620 的 core 配额按块内冗余度重新分配、并把块内近重复的腾退槽**折进**保留槽而不是
扔掉；以及在嫁接（graft）槽上做零参数的注意力校准——这一条**在训练窗口上就生效**，
不是推理期才加的臂（见 §4a），配上"随机 core 槽"版反事实 donor 来把增益钉在真实内容上。

## 0. 诚实的边界：基线是 V13 = 67/158

**本轮一切判定都从 67 起算。**

| 臂 | 题数 / 158 |
| --- | --- |
| `Full`（19 块不压缩） | 47 |
| `Selection` | 35 |
| `AnchorOnly` | 56 |
| **`V13`（本轮基线）** | **67** |

三条必须写明的限定：

1. 旧文档里的 **41/158** 是 V13 之前纯选择路线的数字，**不能**当作本轮的参照物；
   引用 41 会把任何一次改动都算成"大幅提升"。
2. 旧计划里的 **46/158 目标已经低于现状**（67）。它作为目标线已作废，本文的判定线
   全部相对 67 表述（净增益、配对检验、损伤上限），不设绝对题数目标。
3. → 任何机制如果只是"回到 47 附近"，那就是负结果；这一点在判定表里写成
   `Trained − NativeGraft ≥ +4`，而不是"达到 46 就算成功"。

另外两条一直没变的限定：单种子；158 题已被本项目反复用于挑机制，所以这四张证据表
是**内部证据**，不构成对外的泛化结论。

## 1. 机制一：Key 锚定 + Value 放开（`methods/prlmc/deep_fusion.py`）

### 为什么分开

读者用 `k = W_K,l · (h / rms(h)) · γ_l` 寻址，用 `v = W_V,l · (h / rms(h)) · γ_l` 取内容。
`W_K,l diag(γ_l)` 的行空间是"读者看得见的 4,096 维里的那一部分"，它的正交补是**改了
也不动 key** 的子空间。给残差 `Δ` 一个每层正交基 `Q_l`（`build_key_bases` 给出），

```
p = Q_l Q_lᵀ Δ                  key 可见部分（读者能看见的那部分改动）
n = Δ − p                       null 部分（key 看不见，value 看得见）
Δ' = clip(p, key_ratio·rms) + clip(n, value_ratio·rms)
clip(x, c) = x · min(1, c / |x|)          逐元素，只有超出才缩
```

`rms` 是该槽位自身的 RMS。这样两个通道各有硬上限：key 侧 10%，value 侧是一个独立的
常数（`None` 时取 0.34，见下）。V13 里两侧共用一个 `max_ratio`，等于"想动内容就必须动
地址"；这里把地址锁住、把内容放开。

`value_ratio=None`（默认）表示**沿用父类的 `max_ratio`**：模块里记录的是构造时的值，
checkpoint 自描述，所以老 checkpoint 的语义不因这行代码改变。

**上限的实际数值必须写清楚**：`None` 不是"没有上限"，而是"等于 `max_ratio`"，而
`max_ratio` 来自 trainer 的 `--max-ratio`，**默认 0.34**——不是 `DeepFusionResidual`
构造器默认的 0.5。`DeepFusionResidual.value_ratio_effective` 是只读属性，直接给出这一
侧真正生效的那个数；checkpoint 的 `geometry["value_ratio_effective"]` 记同一个数。因此
`--v17-value-ratios` 的扫描**必须把 0.34 这个基线列进去**：只发 0.25/0.30/0.40 而不列
0.34，等于拿三个显式值跟一个没写出来的值比较，跑出来的差异无法归因到"放开 value"。

## 2. 机制二：三段带解耦（同文件，`band_mode="three"`）

```
BANDS      = ((0,10), (11,21), (22,31))
BAND_NAMES = ("lexical", "bridging", "readout")
Δ'_l = g_band(l) · Δ_l          g = (band_gates[0], band_gates[1], band_gates[2])
g_0 ≡ 0                         用 mask 实现：参数仍在（为了记录），梯度恒为 0
```

- 5 个 utility 层落在哪个带由 `band_of_layer` / `band_layout(layer_ids)` 给出，后者还
  返回"被强制置零的层"。
- **口径要点：第 0 带在出厂层表下不可观测。** `UTILITY_LAYERS = (12,16,20,24,28)` 里
  **没有任何一层落在第 0 带**（12/16/20 在 bridging，24/28 在 readout）。`band_layout()`
  因此返回 `band0_observable=False`、`gate_parameters_live=(1,2)`：`band_gate[0]` 停在初
  值 0.0，且**永远拿不到梯度**——它不是"学会了取零"，而是从来没有被优化过。
  `band_mode="three"` 在这套层表上只有**两个**活门，不是三个。
- 于是"浅层零修改"这一半子机制在出厂层表上**等于没测**：任何据此写的"带解耦有效"都只
  是在比较 bridging / readout 两带，第 0 带既没有可观测对象、也没有梯度。它只有在
  `layer_ids` 里出现 ≤10 的层时才真正生效；测试就是按这个前提造的（见
  `tools/test_deep_fusion.py` 第 1、2 节）。
- 报告口径：`band_layout()` 的 `band0_observable` / `gate_parameters_live` 必须进报告
  （评测报告与 checkpoint `geometry["band_layout"]` 都记），训练期检查到这里为 False 时
  打警告说明"band 0 覆盖不到窗口层、`band_gate[0]` 是常数"。缺了这两个字段，读者会把
  三个零里的一个当成"被学成零的门"。
- 三段带解耦**只加两个可学标量**（第 0 带的那个被 mask 掉），其余全部复用 V13 的通路。

## 3. 机制三：密度配额 + 块内吸收（`methods/prlmc/evict_window.py`）

总容量**不动**：`len(carriers) * core_per_block = 9 × 180 = 1,620`。动的是这 1,620 怎么花。

### 冗余度 → 配额

```
ρ_b = mean_{i ∈ sample(b)} ( 1 − max_{j ∈ b, j≠i} cos(k_i, k_j) )
```

`sample(b)` 是**等距固定抽样**（`density_sample` 个），不是随机抽——配额必须是窗口的
确定性函数，否则"同一个窗口"两次构建就不一样，单变量对比全部作废。`ρ_b` 接近 1 表示
块里都是互不相同的材料，接近 0 表示块里全是近重复。

配额按 `ρ_b` 比例分配，每块限制在 `[min_core, max_core]`，用整数最大余数法
（`_capped_quota`）保证总和**恰好** 1,620，余数按块序稳定断平。

### 块内吸收（`absorb_cos > 0`）

```
cos(e, c*) > absorb_cos  ⇒  c* ← rms_preserve(c*, c* + α·e)
α = min(absorb_alpha_max, cos · absorb_alpha_max)
```

`e` 是某个 carrier 块**本来要腾退**的槽，`c*` 是**同块**保留的 core 槽。走
`core_fusion.rms_preserve`，所以吸收不会改变目标槽的范数，也不会把窗口的 norm 抬高去
换注意力。`α` 逐槽记录，`absorbed_pairs / absorbed_mean_alpha / absorbed_mean_cos` 进
telemetry。

这一条是"逐字复制"硬校验**唯一允许**的例外：未被吸收的槽仍必须 `torch.equal` 等于其
源位置（代码里就是这么断言的），被吸收的槽则记为 merged。

### 元数据补齐

`meta["pool_block"]`：长度 = 源池槽数（4,864）的整数张量，第 i 项是源位置 i 所属块号。
`source_block` 只覆盖活下来的 2,304 个位置，淘汰池诊断需要的是全池映射。

## 4. 机制四：嫁接注意力校准 + 反事实边际损失

### 4a. 嫁接槽的注意力校准（零参数；训练窗口上就生效，不是"仅推理期臂"）

机制一的那个正交基 `Q_l` 还用来给 graft 槽做一次**不改权重**的压制，只对
`slot_type==1` 动手（core 槽逐位不变）：

```
s_i = tau · s_quantile,i · s_bias
s_quantile,i = min(1, target / ‖k_i‖)        target = Q_q({‖k_j‖ : slot_type_j = 0})
s_bias       = max(0, 1 + bias / max(logit_proxy, ε))
h'_i = h_i − (1 − s_i) · p_i                 只有 key 可见部分被压下，null 部分不动
```

- `norm_quantile = 0`（默认）时 `s_quantile ≡ 1`，`bias = 0` 时 `s_bias ≡ 1`，于是
  `tau = 1` 就是**恒等变换**——中性常量下这条路径是关闭的。
- `logit_proxy` 是 core 槽 key 与抽样 core key 的**平均绝对内积**，是"graft 槽凭什么
  抢到注意力"的一阶代理，报告里就按这个口径记，不当成真实 logit 用。
- 报告字段：`slots_touched / key_scale_mean / key_scale_min / key_norm_before /
  key_norm_after / logit_proxy / tau / bias / norm_quantile / note`。
- **它在训练窗口上就生效**，这是本轮口径的关键修正。三个常数随 checkpoint 走
  （`geometry["graft_attention"]`：`tau / bias / norm_quantile`、`enabled`、
  `applied_in_training=True`、`site`；`enabled` 把 `norm_quantile > 0` 也算进去），
  训练与评测读的是同一份常量，不存在"训练时没阻尼、评测时才加上"的错位：
  - 训练路径：`tools/train_cross_block_calibration.py::damp_graft_keys()` 在**训练窗口**
    上先阻尼、**再**算邻居 context（context 由这套 key 建，顺序反了就等于拿读者看不见的
    键做测量）。中性常量时返回 `applied=False`，窗口原样使用。
  - 评测路径：`methods/prlmc/deep_fusion.py::calibrate` 用同一组常量对每个评测窗口做同
    样的阻尼。
  - 它不改任何权重、不改损失（`training_objective_unchanged`），改的只是窗口本身。
- 因此臂的名字要分清：**`--v17-no-attn` 是关掉机制四的消融臂（`V17NoAttn`）**；
  `--v17-attention`（`V17Attn`）只在**老 checkpoint**（`graft_attention` 常量中性、训练
  时没阻尼）时才有意义——checkpoint 自带阻尼时这一臂与 `V17` 完全重复，评测会把它判为
  重复并跳过，不会悄悄跑一个"开了两次阻尼"的臂。

### 4b. 反事实边际损失（训练期）

V13 的 donor 臂把 graft 槽换成**上一题文档**的内容。本轮把 donor 的内容来源做成开关，
并保持 margin 定义不变：

```
margin = answer_logp(real) − donor_logp
CF     = max(0, gamma − answer_logp(real) + donor_logp)     # 单边 hinge，donor 走 no_grad
```

| `--cf-mode` | graft 槽被换成什么 | 用途 |
| --- | --- | --- |
| `donor`（默认） | 另一份文档的真实 graft 内容 | 与 V13 逐字一致 |
| `random` | **同一文档**里的随机 core 槽，种子 = f(`--cf-random-seed`, 文档 id) | 破坏"内容对应关系"，保留文档分布 |
| `shuffle` | graft 槽之间置换 | 破坏位置，保留内容集合 |

`random` 与 `shuffle` 都不改变张量形状、不改变 donor 侧的 token 数，所以 margin 的
差异只能归因于"内容是不是这一段材料"。另外两个训练期开关同样默认关闭：

- `--kl-topk k > 0`：保护 KL 只在参考分布的 top-k token 上算（其余置 0 后按原
  reduction 归一化，`k ≤ 词表`）；
- `--layer-norm-weights on`：第 0 个 update 之后按 `w_l = 1/(D_init,l + 1e-6)` 归一化到
  均值 1 写进 `module.layer_norm_weights`（buffer，之后不再更新），残差正则改为逐层加权
  累加，而不是全层求和。**逐层权重现在真的落盘**：模块 buffer 故意是非持久化的（持久化
  会给 `state_dict` 加一个键，让所有更早的 checkpoint 严格加载失败），所以训练结束时另
  外写 `payload["layer_norm_weights"]`（逐层权重的 list）与
  `geometry["layer_norm_weights"] = dict(enabled, values, persisted, note)`。评测端可以直接
  从 checkpoint 读回这两个字段，事后能审计"逐层归一化到底把哪一层压得多"，不必再翻训练
  日志——否则这一条只存在于一次 stdout 里，"它做了什么"在文件层面无法复核。

### 邻居侧的一处改动：`sim_floor`

`cross_block_context(..., sim_floor)` 给邻居相似度一个**绝对下界**：低于下界的候选在
top-k 之前置 `-inf`，因此只能被选中、但权重恰好为 0（不是"权重很小"，是不贡献）。
若一整行的合法候选全在下界以下，则**退回该行未加下界的分数**（原 top-k 照常），否则
softmax 会拿到 `-inf` 并产出 NaN。报告固定加两项：`sim_floor` 与
`below_floor_fraction`（= 低于下界的合法候选对 / 全部合法候选对），以及
`floor_fallback_rows`（回退行数）。`sim_floor > 0` 只允许配 `selection="topk"` 且不能与
`csls` 同开（一个是绝对阈值，一个是逐行重心的相对量）。

**下界要按数据定，不能拍一个数。** 报告现在**总是**给出
`neighbour_similarity_quantiles = dict(p01, p10, p50, p90)`（对**被选中的 top-k** 相似
度取分位，也就是下界真正能作用的那批候选；`sim_floor = 0` 时同样存在，因为它是"下界该
落在哪里"的数据，不是"下界开了没有"的记录）。V13 实测池内跨块邻居相似度**均值约
0.663**，所以 `--sim-floor 0.30` 这种量级的下界几乎筛不掉任何候选：预期
`below_floor_fraction` 接近 0、`floor_fallback_rows` 接近 0。因此**这一臂不能当作"过滤
有效/无效"的证据**——它测的是"几乎没有过滤"。要测过滤，下界必须落在分位数上（例如
p10 或 p50），并在报告里同时写出该 run 的 `neighbour_similarity_quantiles`，让读者自己
看出这条线切掉了分布的多少。

## 5. 哪些是参数、哪些是零参数、哪些可以推理期改

| 部件 | 参数 | 推理期能不能改 |
| --- | --- | --- |
| 带门 `band_gates` | 3 个标量（第 0 带被 mask，实际 2 个可学） | `--v17-band-off` 把 `band_mode` 置 off（只改行为，不碰参数） |
| `key_ratio` / `value_ratio` | **零参数**（裁剪常数） | 可改：`--v17-value-ratios` 只改 `value_ratio`，不许改权重；扫描必须含 0.34（`None` 的实际取值），否则基线不在表里 |
| `sim_floor` | 零参数（构造常数） | 可改（同一次评测里换值重建 context），但取值要对着 `neighbour_similarity_quantiles` 定 |
| `graft_attention_calibration` 的 `tau/bias/norm_quantile` | 零参数（常数，随 checkpoint 走） | 训练窗口与评测窗口都用同一组常量；消融是 `--v17-no-attn`（`V17NoAttn`，关掉）；`--v17-attention`（`V17Attn`）只对常量中性的老 checkpoint 有意义 |
| 密度配额 `quota_mode` / `min_core` / `max_core` / `density_sample` | 零参数（Reader key 的确定性函数） | 可改：`--v17-window-density` 重建窗口 |
| `absorb_cos` / `absorb_alpha_max` | 零参数（阈值） | 可改：与 `--v17-merge-cos` 同开 |
| `--cf-mode` / `--cf-random-seed` | 零参数（训练期数据开关） | 不适用（只影响训练） |
| `layer_norm_weights` | 长度 = 窗口层数的 buffer（不是 parameter，也不是每个 utility 层一个） | 不可改（第 0 个 update 后冻结），但 `payload["layer_norm_weights"]` + `geometry["layer_norm_weights"]` 随 checkpoint 落盘，评测可读回审计 |

这条分界是有意的：**推理期能改的只有常数与开关，权重一律不动**。否则
`V17Value30` 这种臂就不再是"同一个 checkpoint 的推理期设定"，而是一次没记录的微调。

## 6. 单变量原则

每个机制都必须在**同一 158 题**上与它的关断版本逐题配对，且两次运行之间只有那一个
变量不同：

| 对比 | 唯一变量 |
| --- | --- |
| `V17` vs `V13` | 校准器类（`CrossBlockResidual` → `DeepFusionResidual`），其余全同 |
| `V17` vs `V17NoBand` | `band_mode` three ↔ off |
| `V17Value25/30/34/40` vs `V17` | 只改 `value_ratio`（0.34 是 `None` 的实际取值，必须出现在扫描里） |
| `V17NoAttn` vs `V17` | 只关嫁接注意力校准（机制四）：`--v17-no-attn` |
| `V17Attn` vs `V17` | 只在老 checkpoint（`graft_attention` 常量中性、训练时没阻尼）上有意义；checkpoint 自带阻尼时这一臂与 `V17` 重复，评测跳过 |
| `V17Density` vs `V17` | 只换窗口配额（`quota_mode`），同一题集、同一 checkpoint |
| `V13Density` vs `V13` | 同上，但在基线臂上，用来把"窗口变了"与"校准器变了"分开 |
| `V17` vs `V13Donor`/`V17Donor` | 内容因果性：graft 内容被换掉 |

`--architecture v13` 且所有新开关取默认值时，训练路径、`config()`、loss 必须与现在
**逐字一致**（新代码一律走 `if` 分支）。V13 那一臂没有动过，是整套对比的地基。

## 7. 判定门槛（门槛，不是结果）

下面每一条都是运行前写死的线。**本轮一条都还没测。**

1. `V17 − V13 ≥ +4` 题：机制一/二带来的是净增益，而不是把 67 打散重排。
2. `V17 − V17Donor ≥ +5` 题：增益依赖 graft 的真实内容（`random`/`shuffle` donor 必须
   回落到 `NativeGraft` 及以下）。
3. 损伤（`Full` 答对而该臂答错的行）不明显高于 V13 的 12。
4. `V17Value30` 与 `V17` 的差异必须能归因：若 `value_ratio` 的各点之间题数不动，说明
   value 通道的上限不是瓶颈，机制一里"放开 value"这一半是空转——那要报负结果，不是
   报"稳定"。扫描里**必须包含 0.34**（`value_ratio=None` 的实际取值），否则比较的是一
   个没列出来的基线。
5. `V17Density` 与固定配额臂在同一题上的差必须存在且方向可解释；若窗口变了而逐题
   预测完全一致，说明配额没有真正改变读者看到的东西。
6. `V17 − V17NoAttn` 必须存在且可归因：若两臂逐题预测完全一致，说明机制四在这套窗口
   与常量下要么中性、要么没有真的改到读者看到的 key，这一条按负结果报，不报"无损"。

上面任一条不齐 ⇒ 按证据链报告负结果，不调参重跑同一批行。

另外两条不是门槛，而是**这一轮测不出来**的东西，报告里必须照写：

- 第 0 带（"浅层零修改"）在出厂层表下 `band0_observable=False`，`band_gate[0]` 永远是
  初值、永远没有梯度（§2）；
- `--sim-floor` 取 0.30 这类远低于分布的量级时 `below_floor_fraction` 接近 0，这一臂
  不构成"过滤有效/无效"的证据（§4）。

## 8. 证据链四张表的口径（`tools/evidence_chain.py`）

四张表全部由 `tools/evidence_chain.py --eval <json...>` 从**已完成的评测报告**里读出来，
不重跑模型、不重算预测；报告里没有的东西写 `MISSING`，不静默跳过。

### 表 1：三臂分离

| 列 | 口径 |
| --- | --- |
| 臂 | `Trained`（本轮 `V17`）、`NativeGraft`、`RandomControl`、`Counterfactual` |
| 臂名对齐 | `NativeGraft` 在报告里的名字是 **`Graft`**（V10 的原生 graft 算子，1,944 core + 360 graft）；**`V11Evict` 不是原生 graft 臂**，它是普通全池标定臂，拿它当"原生基线"会把对照抬错。`tools/evidence_chain.py` 把 `NativeGraft` 解析成 `Graft` 是对的 |
| 题数 / 准确率 / 官方 QA F1 | 每臂各自统计，逐臂打印实际解析到的臂名与行数 |
| PASS 条件 | `Trained ≥ NativeGraft + 4`；两个对照（随机替换候选内容 / 换 donor 内容）都回落到 `NativeGraft` 及以下 |

### 表 2：McNemar 配对

| 项 | 口径 |
| --- | --- |
| `b` | 只有 `Trained` 答对的行 |
| `c` | 只有对照臂答对的行 |
| `net` | `b − c` |
| `p` | 对不一致对 `(b, c)` 的双向精确二项检验 |
| 目标线 | `Harmed ≤ 2`、`Rescued ≥ 13`、`净胜 ≥ +11`、`p < 0.05`，逐项 PASS/FAIL |

### 表 3：证据位置分解

行按支撑事实**在哪里**分组：在**被删块**里 / 在**保留块**里 / 根本不在窗口里 / 未能
定位；再把"`Full` 已经答对"的行单列为**保护组**。救回（`Trained` 对、`NativeGraft` 错）
与损伤（反过来）分别归因到这些组，而不是只报一个总数。

### 表 4：元数据闭环

| 检查 | 期望 |
| --- | --- |
| `protocol.budget` 与逐行 `capacity` | 2,304 |
| 逐行 `core_total + graft_total` | 1,620 + 684 |
| graft 占比 | 684/2,304 = 29.69% |
| `merged_pairs / avoided_evictions / blocks_with_no_merge / mean_cluster_size / folded_slots` | 每行都在 |
| `absorbed_pairs / absorbed_mean_alpha / absorbed_mean_cos` | 开了吸收时每行都在，`α ≤ absorb_alpha_max` |

缺字段 = `MISSING` + 一行原因。**缺失字段不许当成"通过"**。

## 9. 未测清单（写完就不再解释）

- 四个机制全部**未测**：本轮的 `V17`、`V17Value25/30/34/40`、`V17NoBand`、
  `V17NoAttn`、`V13Density`、`V17Density` 一条都还没跑。`V17Attn` 只在
  `graft_attention` 常量中性的老 checkpoint 上有意义；对自带阻尼的新 checkpoint 它是
  重复臂，不算一条独立证据。
- 机制四的**训练期生效**同样未测：`damp_graft_keys()` 的代码路径已经存在，但**没有一次
  带非中性常量的训练**被执行过，所以"训练时先阻尼、再算邻居 context"目前只有代码依据，
  没有数字。
- 第 0 带（"浅层零修改"）在出厂层表下**测不了**：`band_layout()["band0_observable"]`
  为 False，`band_gate[0]` 永远是初值、永远没有梯度（§2）。任何关于"浅层不动"的结论
  在这套层表上都是空的。
- `--sim-floor` 若取 0.30 这类量级，`below_floor_fraction` 会接近 0（V13 邻居相似度
  均值约 0.663），这一臂不能当"过滤有效/无效"的证据（§4）。
- `layer_norm_weights` 现在随 checkpoint 落盘（`payload["layer_norm_weights"]` +
  `geometry["layer_norm_weights"]`），但本轮**没有一次 `--layer-norm-weights on` 的
  训练**，所以"逐层归一化到底做了什么"目前没有可审计的数据。
- 因此本文没有任何收益数字。表格里的 +4 / +5 / ≥13 / ≤2 / +11 / p<0.05 是**门槛**，
  不是结果，也不是预期收益。
- 单种子、同一 158 题、没有独立自然多跳验收集：即使全部过线，这也是内部证据。
- `tools/test_deep_fusion.py` 是 CPU 单测（假 reader、小矩阵），只证明机制的形状与边界
  是对的，不证明它有用。它在本轮**未被运行**（规格禁止运行测试脚本），只做语法自检。

## 10. 代码位置与入口

| 文件 | 内容 |
| --- | --- |
| `methods/prlmc/deep_fusion.py` | `DeepFusionResidual`（key/value 双通道、三段带）、`value_ratio_effective`、`band_layout()`（含 `band0_observable` / `gate_parameters_live`）、`graft_attention_calibration`、`calibrate`、`load_deep` |
| `methods/prlmc/cross_block_calibration.py` | `cross_block_context(..., sim_floor)`（报告固定带 `neighbour_similarity_quantiles`）、`per_layer_relative_delta`、`layer_norm_weights` |
| `methods/prlmc/evict_window.py` | `quota_mode="density"`、`absorb_cos`、`meta["pool_block"]` |
| `methods/prlmc/key_safe_projection.py` | `build_key_bases` / `project_row_space`（key 基的来源，未改动） |
| `tools/train_cross_block_calibration.py` | `--architecture v17` 与全部 V17 开关、`damp_graft_keys()`（机制四的训练期应用点）、`--cf-mode random|shuffle`、`--kl-topk`、`--layer-norm-weights`；`--max-ratio` 默认 0.34 |
| `tools/eval_cross_block_calibration.py` | `--v17 / --v17-value-ratios / --v17-band-off / --v17-attention / --v17-no-attn / --v17-window-density` |
| `tools/evidence_chain.py` | 上述四张表（`NativeGraft` 解析为报告里的 `Graft`） |
| `tools/test_deep_fusion.py` | 纯 CPU 实现检查（11 节） |

```
# 训练（一次一个变量；V13 那一臂不带任何新开关）
# 注意：--sim-floor 要对着报告的 neighbour_similarity_quantiles 定，0.30 近乎不过滤。
python tools/train_cross_block_calibration.py --architecture v17 --key-channel project \
  --key-ratio 0.10 --band-mode three --sim-floor 0.0 --cf-mode random \
  --updates 384 --out runs/cross_block/v17.pt --report runs/cross_block/v17.json

# 评测（同一题集上的全部单变量臂）
# 0.34 是 --max-ratio 的默认值，也就是 value_ratio=None 的实际上限，必须列进扫描；
# --v17-attention 只在老 checkpoint（常量中性）上加，新 checkpoint 上它是重复臂。
python tools/eval_cross_block_calibration.py ... \
  --v17 runs/cross_block/v17.pt --v17-value-ratios 0.25,0.30,0.34,0.40 \
  --v17-band-off --v17-no-attn --v17-window-density \
  --out reports/cross_block/eval_v17_hotpotqa.json

# 证据链（只读 json，不重算预测）
python tools/evidence_chain.py --eval reports/cross_block/eval_v17_hotpotqa.json \
  --out reports/cross_block/evidence_chain_v17.md
```
