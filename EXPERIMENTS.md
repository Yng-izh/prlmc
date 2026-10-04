# PRLM-C 实验日志

实现 *残差记忆设计_方案整理* 的完整实验记录。每条记录给出配置、命令、原始数字与结论。

基线为未修改的 MemoryLLM-8B on MemoryAgentBench，参数见 `_mllm_patch/BASELINE.md`。
本文件中的一切数字都在同一环境（`/share/latent/mllm_env`，V100-32GB，`chunk_size 400`，
`inject_max_tokens 512`，`do_sample False`）下取得，可直接对比。

本文件只做索引。每条实验的正文在 `experiments/` 目录下，一实验一文件；本文件不随实验追加内容。

## 目录（experiments/）

| 编号 | 运行 | 目的 | 状态 | 正文 |
|---|---|---|---|---|
| E0 | Baseline Tier 1 | 冻结参照分 | 已完成，5 数据集 | [E0.md](experiments/E0.md) |
| E1 | `prlmc-smoke` | 端到端链路验证 | 已完成 | [E1.md](experiments/E1.md) |
| E2 | `prlmc-obs` | 阈值标定（`compact_mode: none`） | 已完成，500 次写入 | [E2.md](experiments/E2.md) |
| E3 | `prlmc-r1` | occupancy-only 结构消融 | 已完成，**accuracy 全 0** | [E3.md](experiments/E3.md) |
| E3b | 纯代码分析 | 归零的机制根因（不运行） | 已完成 | [E3b.md](experiments/E3b.md) |
| E4 | 2×2 矩阵 | 分离 gate 幅值与 compactor | 已完成，**compactor 是充分原因** | [E4.md](experiments/E4.md) |
| E5a | `recent_blocks` | 只保留原生 block 时 reader 是否正常 | 已完成，**EM 26.00** | [E5a.md](experiments/E5a.md) |
| E5b | `select_blocks` | 打分选择能否胜过纯 recency | 已完成，**与 E5a 逐条相同** | [E5b.md](experiments/E5b.md) |
| E5c | E5b + active mask | 真正 K→K' 的无训练 compaction baseline | 已完成，**EM 23.00**（与 E5a 不再相同） | [E5c.md](experiments/E5c.md) |
| E5d | E5c，但 filler 不受限 | 排除 E5c 的两个开关混淆 | 已完成，**EM 24.00**，与 E5c 逐条 97/100 相同 | [E5d.md](experiments/E5d.md) |
| E6 | `carrier_residual` | learned fusion 是否超过 selection | 已完成，**EM 24.00**；无 ckpt，初始化即恒等，故仅验证接线 | [E6.md](experiments/E6.md) |
| E6a | 离线单窗口 overfit | 训练链能否跑通（不读 MemoryAgentBench 指标） | 已完成，单窗口 220 步恢复 2/3 | [E6a.md](experiments/E6a.md) |
| E6b-mini | 离线多窗口共享 checkpoint | 单窗口恢复能否泛化到未见窗口 | 已完成（**2026-09-29 重跑，前一轮作废**），训练 7/30 = 0.233、held-out 2/16 = 0.125，判为 `partial` | [E6b-mini.md](experiments/E6b-mini.md) |
| E6c | 训练档容量压力测试 | 50% 预算下训练过的 compactor 三臂对比 | **训练侧已完成**（2026-09-30，8×96 步），训练 5/30 = 0.167、held-out 3/16 = 0.188，retention reading 翻转为 `overwrites`；三臂 bench 未运行 | [E6c.md](experiments/E6c.md) |
| E6-gate | 压缩门（g1→g2q） | lost-probe 恢复是否 6/6，判定项目继续/停止 | **FAIL**（2026-10-01，最佳 3/6 @ rank 256）；SVD 信息可读，瓶颈在解码侧容量/优化，非信息销毁 | [E6-gate-g2.md](experiments/E6-gate-g2.md) |
| E6d | 三臂顺序消融（种子 0） | 连续窗口训练是否是遗忘原因 | 已完成（2026-10-02），interleave **56/160（+30）** vs forward 37（+11）vs reverse 33（+7），遗忘事件 6 vs 50 vs 42 | [E6d.md](experiments/E6d.md) |
| E6e | 凶手探针 | 遗忘发生时哪组参数在动 | 已完成（2026-10-02），预注册三选项全不成立，判为漂移压过信号的临界翻转，非定向覆盖 | [E6e.md](experiments/E6e.md) |
| E7 | 内容路由稀疏 residual experts | 主动知识分工（阶段 A 干扰图 → B oracle 上界 → C 内容路由） | 设计定稿（2026-10-02，正文 [E7_PLAN.md](E7_PLAN.md)），未开始运行 | [E7_PLAN.md](E7_PLAN.md) |
| E7a | 阶段 A 干扰图（seed 0） | 8 窗梯度可分性与单步功能干扰 | 已完成（2026-10-02），梯度余弦 0.05–0.32 无冲突无重复，单步 harmed=0、rescue 集中在 2 道边界题，corr(cos,rescue)≈0；一阶近似下无定向干扰，与 E6e 漂移结论一致 | [E7a.md](experiments/E7a.md) |
| E7a-24 | 阶段 A 深度扩展 + interleave 3-seed | 24 步功能矩阵；E6d 净增跨 seed 稳定性 | 已完成（2026-10-03，干扰矩阵 2/3 seed：seed 1 被 OOM 挤死待重跑），interleave 复现 +24~+30/forget_ev≤12；24 步 harmed=5 且两 seed 逐条相同、效应不对称（17130 整行零、16370 共振带）、rescue:harmed=31:5 | [E7a-24.md](experiments/E7a-24.md) |
| E7a-sim | 阶段 A 收尾判读 | 内容相似度能否预测功能干扰（router 机制前提） | 已完成（2026-10-03），内容带宽 0.9976–0.9991（同 context 8 窗在 reader 表征中为同一点），与梯度/功能信号零相关，ARI<0 → **停止门槛 1 触发：不做语义 router**，B 阶段 oracle 改测纯窗口隔离 | [E7a-sim.md](experiments/E7a-sim.md) |
| S0 | 残差子空间方案诊断 1：功能敏感子空间扰动验证 | lost probes 的 margin Jacobian 子空间是否真实承载可操作功能信号（soft filter/hard projection 的目标是否真实） | 已完成（2026-10-03），以 E6d interleave ckpt 为基点，sketch rank 13 满秩；敏感方向救回 6.625/13 lost、低敏感 0.25、随机 0，separation 0.490 单调上升 → **PASS**；附带确认初始 compactor（up=0）是数学死区（rank 0），S0 必须用 warm-up ckpt | [S0.md](experiments/S0.md) |
| S0-分区 | 敏感子空间内部功能结构 | 子空间是否按内容主题分区 | 已完成（2026-10-03），非主题聚类：**公共轴 6 探针**（被 7–8/8 方向救，两两余弦 0.63）+ 专属尾迹 5 + 不可达 2（方向存在但增益不够）；Gram top1 占 58.8%、top2 72.6%；S0b 救回的 7 个恰为公共轴+1 尾迹 | [S0.md](experiments/S0.md) |
| S0b | 系数 oracle 最小训练（C=I+E） | rank×rank 系数自由度能否救 lost 而不伤 retained | 已完成（2026-10-03），**INCONCLUSIVE**：rescued 7/13 达标（峰值 step20 达 9），harmed 3/5 超标（阈值 0 过严，n_retained=5 统计脆弱）；‖E‖ 全程钉 0.500 → 容量钳制饱和；判读为「有信号但全局系数变换不够」→ 允许进 S1 但 S1 须自带 no-harm 验证 | [S0b.md](experiments/S0b.md) |
| Native-QA | 数据集原生问答成对对照 | 完整写入、未压缩记忆 vs 同状态手动压缩记忆 vs 原生 full-rewrite；真实生成、官方 EM/SUB 重算 | 已完成（2026-10-03）：官方口径 row0 三臂 5/6/7 EM、row4 30/31/13 EM；压缩 vs 未压缩净差 ±1/100 为噪声级，full-rewrite 仅 row4 大幅掉分；根因=ckpt 来自 row3 窗口、监督与评测任务错位 | [Native-QA.md](experiments/Native-QA.md) |
| Native-QA-warmup | 任务匹配功能监督训练 | row0 原生 24 题 gold+KD 训 96 步，零重叠 row4 100 题评测 | 已完成（2026-10-03）：loss 3.95→3.01 有效，但 EM 31/100 与未训练 selection ckpt 逐题相同（γ 仅用预算 5.2%）；**判定读出/格式瓶颈 >> 压缩损失，停止压缩算子侧变体** | [Native-QA.md](experiments/Native-QA.md) |
| Readout-probe | 读出修复三臂并行（A 指令对抗 / B few-shot 锚定 / C reader 注意力增益） | 同一 row4 无损池上 10 组对照，官方两层打分 + McNemar | 已完成（2026-10-03）：**B2 few-shot+stop 唯一显著，EM 30→42（+15/−3，p=0.0075）**，增益全为先验压制题；无 stop 的 B 官方 EM 崩到 1（续写 Question/Answer 对劫持 parse 层）；C attn×8 +11/−6 不显著且增益与 B2 仅重叠 7/49；组合 BC 39<42 无增益；row0 验证无增益无损（5/97→5/97 排除泄漏），跨行未证实 | [Native-QA.md](experiments/Native-QA.md) |

## 固定参考文件

| 内容 | 正文 |
|---|---|
| 全部超参数清单 | [experiments/params.md](experiments/params.md) |
| 已修复的缺陷 | [experiments/fixes.md](experiments/fixes.md) |
| 未解问题 | [experiments/open-questions.md](experiments/open-questions.md) |
| 复现方式 | [experiments/reproduce.md](experiments/reproduce.md) |

线性读法：E1 发现触发器不触发 → E2 标定阈值 → E3 修触发器后首次测 accuracy → **发现 accuracy 归零，根因未分离** → E4 定位到 compactor → E5a/E5b 用原生 block 选择把 accuracy 恢复到 26.00 → E5c 打开 active mask，free region 真正移出 reader 视野，EM 23.00 → E5d 放开 filler 限制以排除开关混淆，与 E5c 逐条 97/100 相同 → E6 把 compactor 换成 `carrier_residual`，因未训练而与 E5c 逐条 98/100 相同 → E6a 用离线单窗口 overfit 验证训练链；首轮打分跨度错误使全部指标作废，修复后筛出 `lost = 3` 的硬窗口并训练到恢复 2/3 → E6b-mini 把单窗口扩到 8 训练 + 2 held-out；**首轮因度量缺陷把 held-out 分子结构性固定为 0，`window_overfit` 与 0/16 全部作废；修复后重跑得训练 7/30、held-out 2/16，两侧无统计差异（Fisher p = 0.4628）** → E6c 把每窗口 Stage B 步数提到 96，训练 5/30、held-out 3/16，retention reading 从 `retained` 翻转为 `overwrites`——**schedule 才是约束，不是每窗口预算** → E6-gate 在固定硬窗口上跑压缩门（压缩后恢复率 6/6 才算过）：**FAIL，最佳 3/6**；恢复数随解码侧容量（0→64→256 rank）单调上升 0→2→3，证伪「SVD 销毁信息」，但 6/6 门未过，项目进入解释性中间地带 → E6d 固定种子把训练顺序单独切出来：interleave（每窗 4 步轮转）**56/160 净增 +30** vs forward +11 vs reverse +7，遗忘事件 6 vs 50 vs 42——连续窗口训练是遗忘主因，可被纯调度消除；3-seed 复验（E7a-24）确认 +24~+30、forget_ev≤12 稳定 → E6e 凶手探针：预注册的三个定向覆盖假设全部不成立，遗忘是参数漂移压过信号的临界翻转 → E7 设计定稿（阶段 A 干扰图 → B oracle 上界 → C 内容路由，含停止门槛）→ E7a 干扰图 1 步版：梯度余弦 0.05–0.32 无冲突无重复，单步 harmed=0——一阶近似下无定向干扰 → E7a-24 深度扩展：24 步 harmed=5 且跨 seed 逐条相同、效应高度不对称（17130 整行惰性、16370 共振带），干扰结构真实但以 rescue 为主（31:5），1 步版的"无干扰"是深度不足的伪象 → E7a-sim 内容相似度判读：**同 context 8 窗在 reader 表征中为同一点（带宽 0.9976–0.9991），与梯度/功能信号零相关，ARI<0——停止门槛 1 触发，不做语义 router**；阶段 B 的 oracle 臂改为测纯窗口隔离，C 阶段唯一存活的 router 候选是功能梯度路由 → S0 残差子空间诊断 1：以 E6d interleave ckpt 为基点做扰动验证，敏感方向救回 6.625/13 lost（separation 0.490 单调上升）而低敏感/随机方向几乎救不回，**功能敏感子空间真实且强可操作，PASS**；附带确认初始 compactor 是数学死区。

## 交叉引用约定

`methods/prlmc/consolidate.py`、`COMPACTOR_PLAN.md` 等文件里的 `EXPERIMENTS.md E4`、
「缺陷 14–16」一类引用按编号指向上表对应文件；缺陷编号见 [experiments/fixes.md](experiments/fixes.md)。
