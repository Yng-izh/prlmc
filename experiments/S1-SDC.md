# S1 — SDC：感度定向整固（Sensitivity-Directed Consolidation）

> 状态：**G2 完成（原始版本机制 FAIL）；G2b 完成（方向复现、统计不充分）**，进入 G3 合并臂前需扩大 selection-independent held-out 样本。G1 PASS（2026-10-05，lost 19→9 / ratio 0.474 / p=0.0245）在前。
> 详见 [G0.md](G0.md)、[G1.md](G1.md)、[G2.md](G2.md)、[G2b.md](G2b.md)。
>
> G2b 去掉 native select-dropped 采样条件，按 current/singleton 平衡抽取，discovery 与 held-out 分离。held-out select 7 vs sic 4（降 42.9%，方向复现），但 33 个 eligible 题的 McNemar p=0.371，暂不宣称统计复现。当前最稳妥机制叙事是“功能信号保护独占信息”；版本语义仍只作 G0 动机，不作已证机制。
>
> 上游依据：E6b-mini（逐窗口训练操作器不泛化，train 7/30 vs held-out
> 2/16）、S0（功能敏感子空间真实且强可操作，separation 0.490）、
> S0b（全局系数变换不够，容量钳制饱和）、Readout-probe（B2 few-shot+stop
> 读出协议冻结为全局评测标准，row4 full EM 42）。

## 1. 假设与定位

**核心假设**：固定记忆预算下，用冻结 reader 的功能梯度信号指导潜在记忆
压缩，能在未见 context 与未见问题上超过原生 selection，且已正确问题
的退化受构造性约束。

**模块接口**：冻结 reader，输入原生 memory blocks（[256, 4096]×32 层），
输出更少 blocks。即插即用，不触碰 reader 权重，不依赖逐窗口训练。

**新颖性核查结论**（2026-10-03 检索）：

| 近邻工作 | 压缩对象 | 指导信号 | 与 SDC 的区别 |
|---|---|---|---|
| Task-Aware KV Compression（Corallo et al., arXiv:2503.04973） | KV cache，一次性离线 | 任务文本提示 + few-shot | 选已有 token、文本信号；无持续预算、无合并、无版本覆盖语义 |
| Context Distillation as Latent Memory Mgmt（arXiv:2605.28889） | LoRA 记忆库 | 检索相似度 + self-gating | 离散 adapter 激活问题，非固定池内连续压缩 |
| M+（arXiv:2502.00592，ICML'25） | 不压缩 | 共训 retriever + age | CPU 外存换记忆，不回答"预算逼你丢什么" |
| Fisher/OBD 剪枝谱系（LeCun 1990 起） | 权重，离线 | 任务损失二阶信息 | 对象是永久权重；SDC 作用在会话内临时记忆态，且在线 |
| OGD/GEM/A-GEM（持续学习） | 权重更新方向 | 旧任务梯度正交 | 数学同型（投影正交于敏感方向），但作用于推理期激活态记忆，机制层新用法 |
| AdaCAD / CAD | 解码分布 | JSD 自适应加权 | 读出侧修复，实验中作对照组 |

存活内核：**功能梯度显著性 + 一阶正交写入 + 回滚门**的组合应用于
在线固定预算记忆池整固，且带版本覆盖语义。各成分均有出处，组合点
在机制层。

## 2. 动机：版本盲区论证（G0 要测的东西）

Conflict_Resolution 的 gold 要求"最大序号事实覆盖旧序号"。同一实体
的新旧两条事实（"245. X is a citizen of Y" vs "321. X is a citizen of
Z"）在表征空间是近重复文本——跨层余弦趋近 1，**相似度选择在结构上
无法区分该保谁**。而 reader 功能信号天然带时序：探针问"X 的最新
国籍"时，梯度只落在现行 block，废弃 block 显著性为零。

这是可测量、可作论文 Figure 1 的机制性发现，也是"保留问答行为 ≠
保留向量邻域"的直接证据。**若 G0 测出余弦不接近 1，动机崩塌，整个
方案止损。**

## 3. 方法：三组件

### 3.1 组件 1：功能显著性选择（替代相似度选择）

整固触发（池满）时，对每个候选 block $B_j$：

$$s_j = \sum_{p \in \mathcal{P}} \left\| \frac{\partial \mathcal{L}_p}{\partial B_j} \right\|_F^2$$

$\mathcal{P}$ 为探针集（16–32 条短 QA）。reader 冻结，只对记忆态反传。
高 $s_j$ 原样保留（protected），低 $s_j$ 进合并池。计算量与 S0 诊断同类
（已跑通的反传路径）。

### 3.2 组件 2：感度正交合并写入

被丢弃 block $A$ 的内容写入幸存 block $B$ 时，投影到 $B$ 的不敏感子空间：

$$B' = B + (I - U U^\top)\, f(A), \qquad U = \mathrm{span}\left\{ \frac{\partial \mathcal{L}_p}{\partial B} \right\}_{p \in \mathcal{P}}$$

- $f$：轻量混合器（均值池化 + 线性；E6 的 carrier 降级为可选增强消融）。
- $U$：探针梯度张成的低秩基（秩 ≤ |P|），瘦矩阵乘实现，$UU^\top$ 不物化。
- **一阶保证**：$\langle \partial \mathcal{L}_p / \partial B, \Delta B \rangle = 0$，
  合并对作答行为的影响一阶为零。E6-gate 的 no-harm 从事后检查变成
  构造性硬约束。

### 3.3 组件 3：运行时回滚门

合并后重跑探针；任一条翻转 → 从 touched-block 快照恢复该幸存块 →
降级为纯选择。快照只存被触碰块，内存有界。

### 3.4 探针来源（消融两个变体）

- **P-auto**：写入时轻量抽三元组，模板生成"{entity} 的最新 {relation}"；
- **P-write**：直接用写入 chunk 的 next-token loss（更便宜，无模板依赖）。

探针集是文本元数据（几百 token，不入记忆预算），与 M+ 的 age 元数据
同一合法性等级。

## 4. 与 E6 路线的关系

E6b 失败根源是逐窗口训练操作器不泛化。SDC 免训练，超参只有投影秩与
探针数。E6 资产不废弃：**投影锥内训练 $f$**——投影保证无害，$f$ 只
负责往零空间填多少，训练从危险黑盒变成安全锥内填充器。carrier、
路由、训练链全部转为 SDC 的可选消融臂。

## 5. 预注册实验矩阵

**全局冻结**：B2 读出协议（few-shot + `\n` stop）用于一切评测；
主指标新增 **BRR（行为保持率）**= 压缩前后 reader 对同题给相同答案的
比例（含同错），配对预测直接算，动态区间接近 100。官方 EM 照报。

| 门 | 实验 | 判据 | 假设 |
|---|---|---|---|
| G0 | 版本盲区：现行 vs 废弃 block 跨层余弦 | 余弦显著高（近重复）→ 动机成立；否则止损 | 动机 |
| G1 | 显著性选择 vs 原生相似度选择（ratio 0.5，其余全同） | lost 率降 ≥30%，≥3 context 聚合 McNemar 显著 | 选择准则 |
| G2 | 版本分层：废弃 block 被丢率，显著性 vs 相似度 | ≥2 倍差距 | 版本语义机制 |
| G3 | 探针翻转率 + 回滚触发率 | 翻转 ≤1/16（E6-gate 最好 3/6） | 一阶保证 |
| G4 | 泛化：2 dev context 定参，3 全新 context 评测 | 方向一致（sign test） | 免训练泛化 |
| G5 | 预算曲线 {0.25, 0.5, 0.75} × 方法 | SDC Pareto 支配全部对照 | 主结果 |

对照臂：原生 selection、drop-random、drop-oldest（MemoryLLM 原生）、
E6 carrier_residual、metadata-oracle 版本 GC（上限参照）、full-memory
上界。

**证伪退路**：G1 失败 → 假设按当前形式证伪。G0+G2 仍可单独打包为
"相似度度量的版本盲区"发现型论文（findings 级别）。

## 6. 落地映射

| 文件 | 职责 |
|---|---|
| `tools/sic_probe.py` | 探针收割 + 梯度提取（改造 `s0_functional_sensitivity.py` 反传路径） |
| `methods/prlmc/sic.py` | 显著性打分、低秩 $U$、投影合并、回滚门（新文件，不动 consolidate.py 主干） |
| `tools/readout_probe.py` 模式扩展 | full → SDC 压缩 → 配对评测，B2 协议冻结 |

计算预算：每臂 ≈ 写池 13 chunk + 一次探针反传 + 合并 + 2×100 生成，
单卡约 40 分钟；G1–G5 全矩阵 ≈ 6 臂 × 5 context × 2 种子 ≈ 200 GPU·时。

## 7. 执行顺序

1. **G0（版本盲区余弦测量）**：现有池上直接算，一天出结果，先于一切实现；
2. G0 PASS → 实现 `sic_probe.py` + `sic.py` 最小闭环（G1 所需）；
3. G1 过门 → 扩 G2/G3 机制分析 → G4 泛化 → G5 主结果曲线；
4. 全程 B2 读出协议 + BRR + 官方 EM 双口径记录。

## 8. 论文叙事骨架

Figure 1 = 版本盲区（G0 余弦分布）；Figure 2 = 方法图（三组件）；
主表 = 预算曲线 × 方法（G5）；机制节 = G2 分层 + G3 翻转率；安全节 =
一阶正交解析保证 + 回滚统计。

贡献声明：① 发现相似度压缩在版本化事实流上的结构性盲区；② 提出
reader 功能梯度驱动的免训练整固算子，作答行为一阶保真；③ 在固定
预算持续整固设定下全面超越 selection / 训练式合并，无需逐窗口训练
即泛化到未见 context。
