# V11：Reader 感知的潜在记忆整合（代码口径）

本文只写**对当前代码有约束力**的内容。数字与结论见 [03-results.md](03-results.md)；
本轮的实验报告尚未产生，等跑完再回填。

## 一句话

在固定的 2,304 个潜在记忆槽位下，用**冻结 Reader 自己读得出的东西**决定保留哪些槽位，
把载体块内部的同质冗余**折叠**成容量让给被删块，再用**全池有界残差**把 1,620 个 core 与
684 个 graft 一起校准。

```
载体块 9 × 256 = 2,304  --折叠-->  9 × 180 = 1,620 core
被删块 10 × 256 = 2,560  --选槽-->  684 graft
                                    2,304 槽位（不变）
```

## 三个模块

### 模块 1 `methods/prlmc/reader_utility.py`

替换 `disc_leverage`（只看槽位自己：范数、块内离散度、与保留槽的相似度）。新分数用冻结 Reader
的真实投影与真实 RoPE：

```
U_i = Σ_{l∈{12,16,20,24,28}} Σ_h α_i^{(l,h)} · ‖v_i^{(l,h)}‖₂
k_i = W_K·RMSNorm(h_i)    v_i = W_V·RMSNorm(h_i)    q_t = W_Q·RMSNorm(h_t)
α = softmax_slots(q·kᵀ/√128)
```

- 投影用 `key_safe_projection.effective_projection`，**decoder LoRA 增量已折叠**
  （`build_key_bases` 已用真实 `k_proj` 校验过这份折叠）。
- RMSNorm 直接调 Reader 自己的 `input_layernorm`（fp32 输入），eps 与转型顺序都是部署口径。
- 32 个 query head / 8 个 KV head 的分组按 `h → h//4` 处理，cos/sin 来自 Reader 自己的
  `rotary_emb`。
- **位置口径**：候选按它在 19 块源窗口里**当前所在的位置**打分，问句放在整窗之后。目的位置是
  选择结果本身决定的，按目的位置打分是循环定义；时间顺序重排保持幸存槽的相对次序。
  `--utility-geometry compressed` 是敏感性检查用的旁路，不在主口径里。
- **查询来源**：训练侧问题组成的固定探针库（默认 16 条 MuSiQue 训练问题），在任何测试行之前
  一次性编码并缓存进 checkpoint 的 `probe_ids`。测试问题与参考答案都不进入这个分数。
- **去冗余**：`mmr_select(score - penalty·max_sim_to_chosen)`，避免 684 个槽位全是同一类内容。
- graft 的最终分数 = 归一化 `U` × `(1 - w + w·novelty)`，`novelty = 1 - max cos(key_candidate,
  key_core)`：Reader 喜欢但与已保留槽重复的候选要被折价。

### 模块 2 `methods/prlmc/homogeneous_merge.py`

载体块 256 → 180：只在**同一个载体块内部**折叠，且必须**同时**通过多层 Reader key 相似度与
value 一致性两道阈值（只过 key 会把两个 attention 相同但写的值不同的事实混成一个）。

- 簇代表 = 簇内效用最高的槽位；簇大小上限 3。
- 阈值默认 **key 0.72 / value 0.60**（首版 0.80/0.70 时每块只折 18–33 个槽、其余 46–58 个靠淘汰补齐，
  B 与 D 的差别因此不足以支撑“归并优于淘汰”的结论；放松后仍需核对 `blocks_where_quota_came_from_eviction`）。
- 折叠用有界加权融合 + 保范：`w = min(blend_weight·(size-1), max_blend)`，然后按代表槽的
  逐层 RMS 还原。
- **不假设每块都有 76 对可折叠槽位**：报告逐块给出 `merged_pairs`（实测）、`evicted`（不够时
  直接淘汰的数量）、`splits`（折叠过头再拆回）、`blocks_where_quota_came_from_eviction`。
- `evict_block` 是同容量普通淘汰对照（arm D），只换这一步。

### 模块 3 `FullPoolResidual`（在 `methods/prlmc/anchor_residual.py`）

```
h' = RMSPreserve(h, h + Δh)
Δh = f_θ(h, Δpos, log|C|, t)
条件 = [Δpos/window, log|C|/1.5, is_graft]   （零初始化 FiLM）
```

- **Graft 半边继承 AnchorOnly checkpoint**（`down` / `up_graft` / `gate_logit`），
  **Core 半边 `up_core` 零初始化**，FiLM 零初始化。因此 step 0：core 逐位不变、
  graft 逐位等于 AnchorOnly —— 训练脚本里这两条是断言，不是口号。
- 幅度上界与 AnchorOnly 完全一致：`gate·rms·tanh(raw/rms)`，`max_ratio` 同样是 0.5。
- 条件让校准能区分「折叠过的槽位」和「逐字原生槽位」，而不需要被告知自己在哪个臂。

## 训练目标

```
L = CE(answer) + w_cf·CF + w_protect·KL(calibrated ‖ reference) + w_reg·R
```

| 项 | 定义 | 关键约束 |
| --- | --- | --- |
| CE | 答案交叉熵，首 token（实体）权重 `--first-token-weight` | 相对权重均值归一化 |
| CF | `max(0, γ - logP(y|M_true) + logP(y|M_irrel))` | 只在 `logP(Full) - logP(窗口) > margin` 的行上计算；`M_irrel` = 同一窗口、684 个 graft 换成**上一行的文档**；donor 前向在 `no_grad` 下算，hinge 是单侧压力 |
| protect | `KL(calibrated ‖ reference)` | reference = **初始模块**下的 V11 窗口（graft 带 AnchorOnly、core 未动），此处 KL 恰为 0，所以它只会阻止 core 被拿去换 graft，不会把运行推回去撤销自己的起点；按文档缓存 |
| R | `mean((‖Δh‖/rms(h))²)` | 逐层分块累加，保持图内 |

记录项：`cf_effective_steps` / `cf_effective_fraction`（真正满足反事实条件的步数），
`identity_check`（core 逐位 + graft 等于 AnchorOnly），stage-1/stage-2 梯度可达性检查。

## 评测七个臂

| 臂 | 窗口 |
| --- | --- |
| `Full` | 19 块不压缩 |
| `Selection` | 9 个载体块原样 |
| `Graft` | V10 原生嫁接，1,944 + 360 |
| `AnchorOnly` | 上面 + 冻结 AnchorResidual = **56/158 基线** |
| `V11` | 模块 1+2+3，1,620 + 684，B 的权重 |
| `V11Donor` | 同 B 的窗口几何与校准，684 个 graft 换成**另一篇文档** |
| `V11Evict` | 同选槽、同校准，载体按 Reader 效用直接淘汰到 1,620（D 自己的权重） |

三条判定线（研究推进门槛，不是显著性门槛）：

1. `V11 > AnchorOnly`：整套机制有净收益。
2. `V11 > V11Donor ≥ 5`：收益可归因于**真实被删块内容**。前四轮所有机制都过了第 1 条、
   死在第一条上，所以它不可省。
3. `V11 > V11Evict`：折叠本身有价值，而不是"多 684 个槽位 + 全池校准"就够。

逐样本硬校验：2,304 容量 / 1,620+684 拆分 / 全局时间顺序 / 未折叠槽逐字复制 / 折叠槽保住
代表槽的逐层 RMS / graft 只来自被删块 / 两半都被校准动过 / RMS 还原成立 / donor 与 D 的窗口
确实与 B 不同 / 四条老臂与参考报告逐题一致。

## 已实现但不在本轮开启

- Reader K-Adapter、层组解耦、新的候选融合分支：本轮不实现。
- 跨随机种子 3 次：先跑一个种子确认实现正确，再补。
- `--grad-accum 8`（方案里的 effective batch 8）：已实现，默认 1；8 的代价是 8 倍墙钟。
