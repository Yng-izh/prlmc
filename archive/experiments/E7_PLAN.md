# E7 方案：主动知识分工（内容路由稀疏 carrier residual bank）

日期：2026-10-02。本文记录 E7 的完整实验设计，接续 E6d（顺序消融）与
E6e（凶手探针）的结论：遗忘由连续窗口训练的参数整体漂移造成
（interleave 56/160、遗忘事件 6 vs forward 50），不是某组参数定向覆盖；
恢复与内容重叠无关。下一步的问题因此从"谁在覆盖"转为
"能否让不同知识分开承载，使无关更新天然不互相覆盖"。

前序结论见 `EXPERIMENTS.md` 目录（E6c/E6d/E6e 行）。

## 目标与工作假设

目标不是让训练曲线更平滑，而是让不同知识内容对残差参数的更新可以分开承载：
训练新内容时，系统根据内容选择适合的子模块，使已有功能不被无关更新覆盖；
推理和在线压缩时不提供窗口编号、时间顺序或答案标签。

用户提供的同节点结果显示 interleave 在 cum=128 起领先或并列领先，且总分
单调上升（35→36→37→38→38）；forward/reverse 震荡。它支持把交错训练作为
强训练基线，但并不能单独证明功能分工、逐题遗忘/恢复，或窗口之间不存在
单一干扰源。本地 `runs/` 目前只有 E6b/E6c 报告加上 E6d 三臂 JSON
（`e6d_{forward,reverse,interleave}.json`，含逐题 10 点 trace），
用户表中 interleave 128 起的领先已有逐题轨迹可查，但 3-seed 复验仍缺，
本计划把用户提供的表作为已知汇总结果，后续需补齐原始逐题日志与复验。

论文主线仍是：在冻结 reader 的条件下保留 native carriers，再写入有限
residual，以固定有效记忆预算争取相对 native selection 的正净功能收益。
新模块必须服务于这条主线，不能把"专家数增加"或"曲线平稳"本身当作结果。

## 候选模块：内容路由的稀疏 carrier residual bank

令 `Y0 = select_blocks(X)` 是现有 native carrier 输出。用 `K` 个独立的
低秩 residual experts 产生候选修正：

```text
Delta_k = E_k(X, Y0)
Y = Y0 + clip_rms( sum_k pi_k(X, Y0) * Delta_k )
```

第一版 `K=4`，每个 carrier 只激活 top-1 expert。路由特征来自当前压缩输入中
被丢弃来源的摘要与对应 carrier 摘要；不输入窗口 ID、训练轮次、问题、答案或
未来数据。保持当前跨 layer 一致的 source routing、up projection 零初始化
和逐 carrier RMS 上界。先用固定专家数，暂不加入自动扩容、MoRAM 式 rank-1
atom 分解、query-conditioned 路由或多种保护算法。

这是一个可证伪的实验候选，不是已确立的新颖性主张。MoRAM 已有 rank-1
associative atoms 与内容寻址，MoLEM 已有 frozen reasoner 上的
latent-memory experts 与路由，Sparse Memory Finetuning 已有稀疏更新减少
参数干扰。因此，潜在增量只能落在：在 MemoryLLM 冻结 reader 的原生 carrier
上进行内容条件化 residual 修正，并以固定 carrier/slot 预算验证净功能保持。
若后续论文要主张方法新颖，必须再做针对性 prior-work 核验。

## 阶段 A：建立功能干扰图

**问题：窗口/内容能否从输入表征中分出稳定的功能簇？** 如果功能目标本身
高度重叠或冲突，专家路由可能只是在随机隔离梯度。

- 固定共同初始化、同一 carrier、同一 train probes 和评估 probes；使用
  E6c 的 8 个训练窗口，记录每个训练窗口前后所有窗口的逐题输出。
- 对每个窗口分别计算当前 compactor 参数上的 probe-loss 梯度，形成窗口间
  梯度余弦矩阵；另形成"窗口 i 的更新对窗口 j 造成的 rescued/harmed 变化"
  矩阵（E6e 的 conflict_matrix 已给出该矩阵的雏形，1 个 victim × 7 个
  later window，可扩成 8×8 全矩阵）。
- 以 source/carrier 摘要的相似度和梯度/功能干扰分别聚类，比较两种邻接
  关系的一致性；所有特征、阈值只在 train/development contexts 上确定。
- 至少 3 个初始化 seed；另外保留一组独立 context 作复验。每个 seed 改变
  窗口训练排列，避免某一顺序被当作内容结构。

**判读：** 若语义摘要相似度能预测功能干扰/兼容性，内容路由有机制基础；
若二者无关，不能用语义 router 声称在分知识，应考虑按功能梯度或预测变化
路由，或停止专家方向。单个窗口的分数波动不能作为分组证据。

## 阶段 B：Oracle 上界与资源公平

**问题：即使分配完全正确，专家隔离能否缓解干扰？** 先给候选结构一个明确
上界，避免把 router 训练失败误判为架构无用。

| 臂 | 专家分配 | 用途 |
| --- | --- |
| B0 Shared-Interleave | 所有窗口共用当前 compactor；窗口短段轮转 | 当前最佳训练调度基线 |
| B1 Shared-Sequential | 同结构、同曝光量；forward/reverse 顺序各自匹配 seed | 量出纯顺序效应 |
| B2 Expert-Oracle | 按已知训练窗口给专家；窗口身份只用于此上界 | 判断专家隔离在理想路由下是否有收益，不是可部署方案 |
| B3 Expert-Random | 与 B2 同容量，但固定随机分配 | 排除"单纯拆参数/随机分片"带来的变化 |

公平性分两种口径报告：

- **固定 trainable parameters**：专家臂总可训练参数与共享臂一致；共享臂
  对应增大 rank，不能只拿窄共享模块作弱基线。
- **固定 active compute**：每个样本只激活一个 expert；匹配有效训练步数、
  probe 数、reader forward/backward 次数和残差范数上界。另报告完整专家
  bank 的存储开销与单样本激活参数量。

B2 只做诊断上界，不能当最终结果。若 B2 相对 B0 没有稳定提升，停止投入
内容 router；优先排查净功能目标、训练探针或 carrier 表达能力。若 B2 有
收益但 B3 无收益，隔离方式可能有效，进入阶段 C。

## 阶段 C：无窗口 ID 的内容路由

**问题：自动路由能否接近 oracle，并跨窗口/顺序/新 context 保持分工？**

先比较 4 个臂，全部按 interleave 调度训练以隔离架构效应：

| 臂 | 结构 | 可回答的问题 |
| --- | --- | --- |
| C0 | shared compactor | 共享参数基线 |
| C1 | top-1 内容路由 residual experts，`K=4` | 主候选模块 |
| C2 | 相同专家 bank，soft routing | top-1 稀疏隔离是否优于混合共享 |
| C3 | 相同参数预算的 shared 高 rank compactor | 收益是否仅来自容量增加 |

C1 路由只看 dropped-source/carrier 表征；不提供窗口标签。用轻量
load-balancing 项避免全部样本塌缩到单个 expert，但不强迫"每窗口一个
expert"。训练后冻结 router，对路由做审计：窗口内/窗口间 expert 分布、
同语义 paraphrase 的路由一致性、不同语义事实的路由可分性、专家利用率，
以及随机打乱路由后的性能变化。router 的任何阈值和正则权重只用
development contexts 选择。

核心架构主张需同时满足：C1 高于 C0/C3 的 held-out 净收益；低于 B2 的
差距可解释且有界；对窗口顺序置换和新 context 保持收益；路由分布呈现
稳定内容偏好，而不是窗口或 seed 的偶然分组。C1 仅比 C0 平稳但净收益不升，
视为未成功。

## 必须统一的训练与评估协议

- 每臂、每 seed 从 bitwise 相同的初始化开始；固定训练窗口集、候选
  carrier、mask、probe manifest、optimizer steps、每窗口曝光量和总模型
  调用量。B0/C0 与结构臂用完全相同 interleave 排程。
- 先跑 1 seed 做 pipeline/debug，再对有效对比跑至少 3 seeds；结论以独立
  context/episode 作为统计单位。现有 parquet 重复 context 不计作独立样本
  （row 4–7 与 row 0–3 字节级重复，E6c 已确认）。
- 每个 checkpoint 对固定问题集保存逐题答案、correctness、gold NLL 和
  expert route；评价集分 `Q_drop`（selection 丢失但 full 可答）、`Q_keep`
  （native selection 原本正确）及 `Q_other`（保留区/其他内容）。测试
  probes 不用于优化或选超参。
- 报告 `rescued_all`、`harmed_all`、`net_gain=(rescued-harmed)/N`、
  `Q_keep` harm、答案 NLL、free-generation EM/F1、旧题存活率、各 expert
  使用率及参数/显存/延迟成本。Recovery rate 只作诊断指标。
- 在 final checkpoint 做干预：路由置换、expert 输出置零、同一输入替换成
  错误 source、随机路由；若目标事实的恢复不随对应 expert/source 变化，
  不能宣称内容被分工存储。
- FactConsolidation 等存在最新事实更新的数据必须按时间语义评分：新值应
  更新、旧值不应被永久保护。另测只读查询集和冲突更新集。

## 次序与分布外验证

同一方法至少在三种训练次序（forward、reverse、seeded random）下训练，
并在同一 held-out 内容集评估。报告方法 × 顺序交互效应，而不只平均线。
最终将训练、开发、测试按 context/episode、实体与事实做不重叠切分；增加
独立生成的内容流，避免同一 context 的不同窗口造成伪泛化。

另外检验未见过的语义组合与"无法路由"的输入。无可靠 expert match 时应有
native-selection fallback；fallback 只能由训练/开发集定阈值。它的作用是
限制潜在伤害，不作为正收益来源。

## 资源与停止门槛

先对现有 8 窗口、当前 19→9 carrier 设定跑诊断规模，不并行常驻多个完整
GPU 池。K 从 2/4 开始，只有 K=4 过关后才测 K=8；对每种 K 报完整参数与
optimizer state。当前 residual RMS 上界保持不变，避免把路由、rank、幅度
和保护 loss 混成一次实验。

停止/转向条件：

1. 阶段 A 找不到可复现的内容/功能分组关系：先不做语义 router。
2. Oracle 专家在公平预算下不改善 held-out 净收益：先不实现自动路由。
3. 自动路由收益只出现在训练窗口，或明显依赖某个窗口顺序/seed：判定分工
   尚未学成。
4. 只增加 rescued 但 harm 同步或更快增加，net gain 不为正：判定结构失败。
5. 只有参数预算增加才有收益：把结论写成容量效应，不写成主动分工。

## 执行优先级

先补齐当前 interleave 的逐题 checkpoint 轨迹和 3-seed 复验；随后完成
阶段 A 的干扰图与阶段 B 的 oracle/random 上界。只有 oracle 显示专家隔离
值得继续，才实现 C1/C2，并按相同 interleave 进行结构对照。阶段 C 通过后，
才加在线 novelty allocation、专家冻结/扩容或容量压力测试。

## 外部方法定位

- [HESTIA, UAI 2026](https://proceedings.mlr.press/v337/le26a.html)：
  顺序不变优化与 adapter routing。
- [MoRAM, ICML 2026](https://icml.cc/virtual/2026/poster/64295)（[官方实现](https://github.com/Artificer-AI-Lab/MoRAM)）：
  rank-1 内容寻址原子。
- [MoLEM](https://arxiv.org/abs/2605.21951)：latent-memory mixture experts。
- [Sparse Memory Finetuning](https://arxiv.org/abs/2510.15103)（[公开代码](https://github.com/dtunai/continual_learning_via_sparse_memory_finetuning)）：
  稀疏 memory 参数更新。

这些工作更适合作为定位、基线和新颖性边界，不是未经适配即可直接替换
PRLM-C 的实现。
