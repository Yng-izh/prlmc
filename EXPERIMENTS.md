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
| E7 | E6 + adaptive trigger | 恢复完整 Ours | 未开始，依赖 E6 的训练 | — |

## 固定参考文件

| 内容 | 正文 |
|---|---|
| 全部超参数清单 | [experiments/params.md](experiments/params.md) |
| 已修复的缺陷 | [experiments/fixes.md](experiments/fixes.md) |
| 未解问题 | [experiments/open-questions.md](experiments/open-questions.md) |
| 复现方式 | [experiments/reproduce.md](experiments/reproduce.md) |

线性读法：E1 发现触发器不触发 → E2 标定阈值 → E3 修触发器后首次测 accuracy → **发现 accuracy 归零，根因未分离** → E4 定位到 compactor → E5a/E5b 用原生 block 选择把 accuracy 恢复到 26.00 → E5c 打开 active mask，free region 真正移出 reader 视野，EM 23.00 → E5d 放开 filler 限制以排除开关混淆，与 E5c 逐条 97/100 相同 → E6 把 compactor 换成 `carrier_residual`，因未训练而与 E5c 逐条 98/100 相同 → E6a 用离线单窗口 overfit 验证训练链；首轮打分跨度错误使全部指标作废，修复后筛出 `lost = 3` 的硬窗口并训练到恢复 2/3 → E6b-mini 把单窗口扩到 8 训练 + 2 held-out；**首轮因度量缺陷把 held-out 分子结构性固定为 0，`window_overfit` 与 0/16 全部作废；修复后重跑得训练 7/30、held-out 2/16，两侧无统计差异（Fisher p = 0.4628）** → E6c 把每窗口 Stage B 步数提到 96，训练 5/30、held-out 3/16，retention reading 从 `retained` 翻转为 `overwrites`——**schedule 才是约束，不是每窗口预算** → E6-gate 在固定硬窗口上跑压缩门（压缩后恢复率 6/6 才算过）：**FAIL，最佳 3/6**；恢复数随解码侧容量（0→64→256 rank）单调上升 0→2→3，证伪「SVD 销毁信息」，但 6/6 门未过，项目进入解释性中间地带 → E7 与 trigger 依赖的完整训练档暂缓。

## 交叉引用约定

`methods/prlmc/consolidate.py`、`COMPACTOR_PLAN.md` 等文件里的 `EXPERIMENTS.md E4`、
「缺陷 14–16」一类引用按编号指向上表对应文件；缺陷编号见 [experiments/fixes.md](experiments/fixes.md)。
