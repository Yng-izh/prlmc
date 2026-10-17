## S0b — 系数 oracle 最小训练（C = I + E）

### 目的

S0 的后续门控：在不动 `down.weight`（D）的前提下，只训练 rank×rank 的
系数变换 E（1024 个自由度），检验「系数空间的自由度是否足以救回 lost
probes 而不伤害 retained」。判据预注册：

- **GO**：rescued ≥ ⌈0.4·n_lost⌉ = 6 且 harmed ≤ ⌊0.1·n_retained⌋ = 0
- **NO-GO**：rescued < 6
- **INCONCLUSIVE**：其余情形

### 配置

| 项 | 值 |
|---|---|
| Oracle | `CoefficientOracle` C = I + E，E 零初始化，插在 `gelu(down(·))` 之后、`F.linear(·, basis)` 之前 |
| 基点 | 同 S0：E6d interleave checkpoint，row 3 / start 17890 |
| probes | lost 13 / retained 5（与 S0 同窗口） |
| 训练 | AdamW(lr=3e-3) 只更新 E，60 步，每 5 步测一次；‖E‖ 钳制 max_deviation=0.5 |
| 运行 | v100-3 gpu0（队列脚本接在 partition 任务后自动启动） |

训练信号只有 lost probes 的 margin（rescue 目标）；retained 每次测量
只观测不约束——这是本实验设计的关键局限，结果判读时须记住。

### 结果

**verdict: INCONCLUSIVE**（rescued 7/13 达标，harmed 3/5 超标）。

| step | rescued | harmed | ‖E‖ |
|---|---|---|---|
| 5 | 5 | 1 | 0.451 |
| 10 | 6 | 2 | 0.500 |
| 20 | **9** | 2 | 0.500 |
| 25–55 | 6–8 | 2–3 | 0.500 |
| 60 | 7 | 3 | 0.500 |

- rescued 门达标（7 ≥ 6），峰值 step 20 曾到 9/13。
- harmed 门始终不达标：n_retained=5 时阈值为 0，但 harmed 从未低于 1，
  后期在 2–3 波动。**每 1 个 retained 翻错 = 20% 相对损失**，样本太小，
  这个 harmed 测量本身统计上很脆弱。
- ‖E‖ 从 step 10 起钉死在 0.500：max_deviation 边界全程激活，E 的容量
  被钳制约束住，rescued 曲线的震荡（9→7→8→7→6→8）是「想走更大步但被
  拉回」的形态。

### 判读

1. **系数自由度有真实信号但不充分**：1024 个自由度能稳定救回约一半
   lost（5–9/13），不是噪声。但做不到不伤 retained——rescue 与 harm
   在 E 的同一个移动方向上耦合。
2. **钳制饱和提示容量瓶颈**：‖E‖ 全程顶边界说明优化器想要的 E 比
   0.5 大得多。两个未测的旋钮：放宽 max_deviation（0.5→1.0+）或降
   lr 延长步数。若放宽后 rescued 升而 harmed 不升，说明 harm 来自
   震荡而非方向错误；若 harm 同步升，耦合是结构性的。
3. **对 S1 的含义**：这正是 soft filter 存在的理由。E 是全局系数变换
   （对所有 probe 一视同仁），而 S1 的 `(I+λH)^-1` 是在参数更新上
   惩罚敏感方向位移，粒度更细。**门的结论是「有信号但单靠系数变换
   不够」→ 允许进 S1，但 S1 必须自带 no-harm 约束验证。**
4. **harmed 阈值的修正意见**：⌊0.1·5⌋=0 的阈值过严，等价于要求 harmed
   测量零方差。后续沿用本窗口做对比实验时，建议以 harmed 比例而非
   绝对计数报告，且 n_retained 应扩大（S1 训练驱动把 retained 集从 5
   扩到全量 eligible 未丢 probes）。

### 工件

| 文件 | 说明 |
|---|---|
| `tools/s0b_coefficient_oracle.py` | 驱动（OracleCompactor monkey-patch 复用 checkpoint 循环） |
| `methods/prlmc/functional_subspace.py` | `CoefficientOracle.matrix()` 钳制已修（整体缩放） |
| `runs/s0b_oracle.json` | 完整 history 12 点 + verdict |
