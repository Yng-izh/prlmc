## E6e — 凶手探针：遗忘发生时，compactor 哪组参数在动

### 目的

E6d 证明连续窗口训练造成遗忘、交错可消除，但"哪部分参数在覆盖别人的记忆"
仍未回答。本轮对训练过程做参数组摄影：victim 窗口（3:17890）先训满
96 步，之后每个后续窗口按 4 步一切片训练，每片结束立即重测 victim 的
20 道 probe，同时拍摄三组参数（routing / residual / strength）的快照。
问题预注册：训练后面窗口时，是哪一组参数的变化让前面已恢复的题重新答错？

参数组划分：

| 组 | 参数 | 语义 |
|---|---|---|
| routing | `q.weight`, `k.weight`, `route_logit` | 丢弃的信息投递到哪些槽位 |
| residual | `down.weight`, `up.weight` | 注入什么内容（up 零初始化） |
| strength | `gamma_logit` | 注入多强 |

预注册判读表：routing 动得最多且行为跟随 → 投递错位（misdelivery）；
residual 动得最多且 routing 稳定 → 内容腐蚀（content corruption）；
两组都动 → 联合干扰。距离用对称形式
`||A−B|| / (||A||+||B||+eps)`，避免零初始化 up.weight 的分母爆炸。

### 配置

| 项 | 值 |
|---|---|
| victim | 3:17890（Stage A 4 步 + Stage B 96 步，训完 victim 5/6 恢复、20 题 probe 对 5） |
| later windows | E6c 其余 7 窗，每窗 96 步、4 步一切片，共 168 个切片 |
| 快照数 | 169（victim 训完 1 张 + 每切片 1 张） |
| seed / lr / 结构 | 与 E6d 完全相同（seed 0） |
| 输出 | `/tmp/e6e_probe_report.json` → `runs/e6e_probe_report.json` |

### 结果

冲突矩阵（每窗训练前后 victim 对题数）：

| 窗口 | before → after | delta | 归类 |
|---|---|---|---|
| 3:17130 | 5 → 4 | −1 | drop |
| 3:16750 | 4 → 3 | −1 | drop |
| 3:14470 | 3 → 4 | +1 | gain |
| 3:17510 | 4 → 4 | 0 | flat |
| 3:14850 | 4 → 3 | −1 | drop |
| 3:16370 | 3 → 5 | +2 | gain |
| 3:15230 | 5 → 5 | 0 | flat |

**逐题翻转点定位（3 道动态题，其余 17 题全程稳定）**：

| 题目 | 翻转轨迹（翻转所在窗口） |
|---|---|
| Victor Turner（受雇于谁） | 错@3:17130 → 对@3:17510 → 错@3:14850 → 对@3:16370 → 错@3:15230 → 对@3:15230（6 次翻转，末态对） |
| Charlie Manuel（运动） | 错@3:16750 → 对@3:14470 → 错@3:17510 → 对@3:14850（4 次，末态对） |
| John Lightfoot（受雇于谁） | 错@3:14850 → 对@3:16370（2 次，末态对） |

**chunk 级检验（翻转 chunk vs 无翻转 chunk）**：

| 组 | 翻转 chunk 平均位移 (n=11) | 无翻转 chunk 平均位移 (n=157) |
|---|---|---|
| routing | 0.0132 | 0.0129 |
| residual | **0.0425** | 0.0332 |
| strength | 0.0012 | 0.0013 |

corr(victim_delta, routing_d) = −0.101；corr(victim_delta, residual_d) = −0.058；
corr(victim_delta, strength_d) = −0.022。

**窗口级检验（窗口终局 delta vs 距 victim 的累计参数距离）**：
corr(window_delta, routing_total) = 0.434；residual_total = 0.466；
strength_total = 0.472。三者同量级、全部为弱正相关——累计参数漂移越大
的窗口发生在越靠后（参数本就离 victim 越远），这一相关主要由时间顺序驱动，
不区分 drop/gain：drop 窗（14850）与 gain 窗（16370）的累计距离几乎相同
（routing 0.796 vs 0.807、residual 0.875 vs 0.887）。

### 判读

**预注册三选项全部不成立，改判"随机漂移下的碰运气恢复"。** 判读表假设
遗忘由某一参数组主导；实测三个 drop 点的 routing 位移（0.0437/0.0161/
0.0096）与 gain 点（0.0175/0.0054）同分布，residual 在翻转 chunk 略高
（0.0425 vs 0.0332，+28%）但方向混杂——residual 位移既造 drop 也造 gain，
符号无法预测（chunk 级相关 −0.058，≈0）。3 道动态题的翻转像布朗运动：
同一窗口内先错后对（3:14850 内 516 错、520 对；3:15230 内 712 错、732 对）。

**遗忘的机制是漂移压过信号，不是定向覆盖。** 后续窗口训练把 compactor
参数整体推离 victim 时刻（800 步后 routing 0.83、residual 0.90 的对称距离），
victim 相关题的答案在这个漂移场里处于临界态——对错由漂移的瞬时方向决定。
这与 E6d 的调度结论拼上：interleave 每 4 步就把梯度拉回均衡，参数轨道
不被任何单窗口长时间牵引，翻转率 6 次对 50 次。

**恢复与内容重叠无关。** victim 的 294 条 probe 与任何后续窗口 probe 的
交集为 0（已验证），但 3:16370 训练后 Victor Turner/John Lightfoot
两题恢复——共享的 residual 通路（32 维 bottleneck）恰好被推向 victim
所需子空间，属于偶然协同，不是记忆内容复用。

**工程含义。** 若目标是保留，约束参数漂移（如 EWC/漂移正则）或交错调度
（E6d，已验证）都可行；若定向修复，则"哪个参数组在害人"不成立，
该问"参数离 victim 状态多远"——受 distance-from-victim 支配的临界
恢复才是实测行为。

### 复现方式

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
/share/latent/mllm_env/bin/python /tmp/e6e_probe.py \
  --chunk 4 --out /tmp/e6e_probe_report.json
```

运行于 v100-2 gpu8，PID 51287，2026-10-01 21:59 UTC 启动、2026-10-02
05:35 UTC 写盘（约 7.6 h，教师缓存命中 125+）。脚本
`tools/e6e_probe.py`（md5 `e805fe1e35555926b224550087e12d00`）。

### 附：产物

| 文件 | 大小 | md5 |
|---|---|---|
| `runs/e6e_probe_report.json` | 474936 B | `a2fc7f38ffa3d0d115d9ec083ecfc525` |

报告结构：design（含预注册判读表）、baseline（各窗 probe 基线）、
photographs（169 张快照：每题对错 + 三组参数 md5 + gamma/route_scale/
residual_ratio/routing_entropy）、conflict_matrix、correlations
（每 chunk 的 victim 变化 + 三组参数距 victim 距离与逐 chunk 位移）。
