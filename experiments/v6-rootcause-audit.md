# v6 根因审计 — 5 个病灶的代码级核实（2026-10-18）

审计人：auditor（任务 t1）。仓库：v100-2:/share/latent/PRLM-C。全部为静态代码/日志核对，未运行 GPU 任务。
每条给出结论、file:line 证据与代码摘录。与声明不符之处在末尾汇总（§6）。

## 病灶 1：逐槽位单位 RMS 归一化 + step-0 native_gain 梯度 1.4e5

**结论：归一化声明属实（两个文件均在）；"梯度 1.4e5" 的声明只有实验笔记一句孤证，在仓库所有已存训练产物中均不可复现——判定为未经留痕的断言，且与已存日志量级（≤0.115）矛盾。**

### 1a. gated_slot_merge.py 归一化存在 — 属实

`methods/prlmc/gated_slot_merge.py:173-180`：

```python
def _group_delta(self, probabilities, indices, source_layer, carrier_layer,
                 down_weight, up_weight, group_gain=None):
    values = F.linear(source_layer.float().flatten(0, 1), down_weight)
    context = (values[indices] * probabilities.unsqueeze(-1)).sum(1)
    carrier = carrier_layer.float().flatten(0, 1)
    delta = F.linear(F.gelu(context - F.linear(carrier, down_weight)), up_weight)
    return (delta * torch.rsqrt(delta.square().mean(-1, keepdim=True) + _DELTA_VAR_EPS)
            * carrier.square().mean(-1, keepdim=True).sqrt())
```

即 `delta * rsqrt(mean(delta²)+eps) * carrier_rms`，逐槽位（最后一维 mean，keepdim）单位 RMS 归一化后按该槽位 carrier 自身 RMS 缩放，与声明一致。同构代码更早出现于 `methods/prlmc/consolidate.py:535-537`（CarrierResidualCompactor._layer_delta）与 `methods/prlmc/slot_merge.py:59-63`（_slot_delta），是该仓库所有融合算子的统一模式。

### 1b. native_skip_merge.py 归一化存在 — 属实

`methods/prlmc/native_skip_merge.py:64-93`：

```python
def _group_delta(self, probabilities, indices, source_layer, carrier_layer,
                 down_weight, up_weight, group_gain=None):
    adapted = super()._group_delta(probabilities, indices, source_layer,
                                  carrier_layer, down_weight, up_weight)
    ...
    carrier_rms = carrier.square().mean(-1, keepdim=True).sqrt()
    ...
    native = (torch.cat(chunks) - carrier) * confidence
    gain = self.native_gain if group_gain is None else group_gain
    combined = adapted + gain * native
    return combined * torch.rsqrt(combined.square().mean(-1, keepdim=True) + _DELTA_VAR_EPS) * carrier_rms
```

注意：native 支路在求和前**没有**中间归一化（注释也明说），归一化作用在 `combined`（adapted+gain*native）整体上。`native_gain` 是每层组一个符号增益：`native_skip_merge.py:48-52`（`self.native_gain = nn.Parameter(torch.zeros(self.num_groups))`，默认 `num_groups=8`，`gated_slot_merge.py:96`），不是单一全局标量。

### 1c. 梯度 1.4e5 的日志出处 — **不存在**

唯一出处是实验笔记：`experiments/Natural-merge-v5-routing-generalization.md:56`：

> 原生支路在 step 0 的梯度量级比学习支路大（`native_gain` ≈1.4e5 vs `up_weight` ≈8e2）……

该笔记未引用任何日志文件。仓库内全部训练产物核对结果：

- v5 run `runs/natural_merge_v5_20261018_1050/Dsup.train.v5plan.json`：`updates[0].gradient_norms_before_clip` 仅记录 `{down_weight 0.0, up_weight 1.966, q/k/slot_gate/rel_gate 全 0}`，**键里没有 native_gain**；全部 64 个 update 亦无。
- v4 run `runs/natural_merge_v4_dd_20261015_0108/Dnative.train.json`：`updates[0].gradient_norms_before_clip.native_gain = 0.0`，64 步中该值最大 0.115（step0 0.00448 → 逐步上升）。v4 Dsup step0 `up_weight = 37.4`。
- `runs/natural_merge_v5_20261018_1050/Dsup.train.log` 只有 10 行且是旧 plan 被拒的 Traceback；`grep -rniE '1\.4e|1\.4e\+05' logs/ runs/ reports/ docs/` 除上述笔记一句外无任何命中。

**判定**：1.4e5 无法从仓库复现，且已存的 Dnative per-param 梯度范数（≤0.115）与之相差 6 个数量级；"up_weight ≈8e2" 与已存 step0 值（v5 1.97、v4 37.4）同样不符。该断言应视为未留痕/口径不明的孤证，下游不得引用；若方案 B 需以它为动机，必须先重跑并落盘 per-param 梯度。

机制层面零初始化自归一化的梯度放大路径本身是真实的：`up_weight` 零初始化（`gated_slot_merge.py:117`、`consolidate.py:473`）+ `combined` 自 RMS（`rsqrt(0+1e-6)=1000`）确实会产生 1/sqrt(eps) 量级的首步增益（`consolidate.py:543-556` 有专门注释），但具体数值无证据。

## 病灶 2：32 层共享标量 alpha 与全局 _cap(residual_rms_ratio≤0.05) 的相互作用

**结论：属实，但"共享标量"需修正——v3/v5 门控路径的 alpha 是逐槽位向量（跨 32 层共享同一份），v1/CarrierResidual 路径才是真标量。全局 _cap 的确以单一 scale 联合缩放全部 32 层、全部槽位，并与 alpha 级联。**

- alpha（v3/v5 门控）：`gated_slot_merge.py:279,287`
  ```python
  alpha = self.gamma_max * torch.sigmoid(gate_logits)   # [slots]，逐槽位
  for layer in range(self.num_layers):
      ...
      outputs.append((carriers[layer].float().flatten(0, 1) + alpha * delta).to(source.dtype))
  ```
  同一份 alpha 用于所有 32 层（循环体内不随 layer 变化），浅层/深层无法表达不同幅度；`last_gamma` 记录的是 `alpha.mean()`（:289）。
- alpha（v1 真标量）：`methods/prlmc/consolidate.py:502-505`（`gamma = gamma_max * sigmoid(gamma_logit)`，单一参数）；`slot_merge.py:95,100`、`merge.py:99,132` 均为 `alpha = self.gamma` 后跨层复用。
- 全局 _cap：`gated_slot_merge.py:47-84`。对 candidate 与全部 carriers 累加能量后算**一个**标量 scale：
  ```python
  limit = max(0., self.global_rms_budget - margin)
  scale = (limit * torch.sqrt(carrier_sq.clamp_min(1e-24)) /
           torch.sqrt(residual_sq + 1e-24)).clamp(max=1.)
  ```
  budget_scope 注释自证："all layers and output slots; measured in stored dtype"（:71-73），超限直接 raise（:70-72）。
- 相互作用：`forward` 中 `_cap(candidate, carriers)` 在 alpha 乘加之后调用（`gated_slot_merge.py:298`，candidate 已含 `alpha*delta`）。若 alpha 已把 stored residual 推近上限，_cap 会用同一个 scale 把所有层、所有槽位等比压回 ≤0.05——门控学到的层间/槽间相对差异在幅度维度被全局一次缩放抹平，只有方向差异幸存。这就是"32 层共享幅度 + 全局硬帽"互相锁死的机制，声明成立。

## 病灶 3：consolidate.py / slot_merge.py 的盲压缩路径

**结论：属实。载体选择与融合值计算均不接触任何 query/问题信号，是写时（write-time）盲压缩。**

- `methods/prlmc/slot_merge.py:1-3` 模块 docstring 直言：
  > Query-independent slot fusion into native carriers, with cross-layer routing. … No question enters this module.
- 载体选择 `methods/prlmc/consolidate.py:204-267`（`_select_block_indices`）：score 仅由 importance（第 16 层 block 均值的去均值范数，:243）、recency（:249）、goal_direction（可选外部方向，:251-257）与 greedy diversity（:266，`_greedy_diverse`）组成；`source_allow_mask` 只能剔除、不能引入读时信息。无任何 reader 查询向量参与。
- 融合值：`slot_merge.py:97-99` query/keys 均来自 carriers 与 discarded blocks 自身：
  ```python
  query = _l2(self.q(_l2(carriers[ref].detach().float().flatten(0, 1))))
  keys  = _l2(self.k(_l2(fusion[ref].detach().float().flatten(0, 1))))
  ```
- 唯一的读时出口是 `goal_direction` 与 `select_weights` 参数（由上层传入；v6 方案 A 的槽位级择优若要读时感知，需要经这两个口或新增入口）。

## 病灶 4：docs/04 中 Selection 25.93 vs native50 25.94 的数据一致性

**结论：表内算术一致（两行均值复算吻合），但文档自身存在一处未清理的自相矛盾段落，须在 v6 文档中显式作废。**

- `docs/04-longbench-eval.md:66-74` 16k 三臂表：Selection 五任务 35.01/26.33/21.03/25.44/21.85，均值 129.66/5 = **25.932 ≈ 25.93** ✓；native50 35.80/27.71/20.00/25.77/20.43，均值 129.71/5 = **25.942 ≈ 25.94** ✓；Dsup 27.43 ✓。引用句"打平（25.93 vs 25.94）"（:75）成立。
- 融合子集表（:77-85）：触发数合计 124+25+191+4+151 = 495 ✓，加权 +3.02 与正文一致。
- **矛盾**：同文件 :96-97（"16k 结果（2026-10-17）"节）仍写：
  > 16k 下多数 LongBench 样本文档不足 10240 write tokens，**根本不进融合路径**
  而 :78-83 的"结论修正"节已明确推翻它（hotpotqa 124/200、musique 191/200、narrativeqa 151/200 进融合）。:97 是旧结论残留，与逐题 prediction 逐字相同的证据（:84-85）冲突。结论以"结论修正"节为准，:96-97 段落应在 v6 中标注作废。

## 病灶 5：docs/01-problem-discovery.md 的 E4/S1c/E7a 证据

**结论：E4 与 E7a 在 docs/01 有逐字记录；S1c 不在 docs/01，而在 docs/02-method-exploration.md——用户声明的文件归属有误。且两处均为归档索引摘录，正文（experiments/E4.md 等）已声明缺失，不可作为独立证据源复核。**

- E4：`docs/01-problem-discovery.md:9`：
  > E4 定位到 compactor（**compactor 是充分原因**）
  以及 `methods/prlmc/consolidate.py:50` 把 svd_tokens 基线的失败归到 E4、:534 提到 "the E4 failure mode in disguise"。docs/01:22 证据位置：`outputs/prlmc-{smoke,obs,r1,e5a,…}`，正文摘自 `archive/2026H1_legacy/root/EXPERIMENTS.md`；docs/01:5 明确声明逐实验正文不存在、"结论数字逐字摘自归档索引"。
- S1c：`docs/01-problem-discovery.md` 全文 24 行，`grep -n 'S1c' docs/01-problem-discovery.md` 无命中。S1c 证据在 `docs/02-method-exploration.md:3`（S1c-wash 单窗口 PASS、S1c-global-oracle 预注册门 FAIL 停线、rescue_gap=0 等完整结论）与 `docs/00-overview.md:10`。**声明与代码/文档不符。**
- E7a：`docs/01-problem-discovery.md:9`：
  > E7a 干扰图（梯度余弦 0.05–0.32，单步 harmed=0）→ E7a-24 深度扩展（harmed=5 跨 seed 逐条相同；rescue:harmed=31:5）→ E7a-sim（同 context 8 窗在 reader 表征带宽 0.9976–0.9991，与梯度/功能信号零相关，ARI<0——**停止门槛 1 触发，不做语义 router**）
  存在且自洽；但同样是索引摘录级证据（正文缺失），且 E7a-sim 的"不做语义 router"结论对 v6 方案 A/B 的相似度路由设计是直接约束，应纳入 v6 设计输入。

## §6 与声明不符之处汇总

1. **[高] "step-0 native_gain 梯度 1.4e5" 无日志出处**（§1c）：唯一来源 experiments/Natural-merge-v5-routing-generalization.md:56，无引用文件；已存 Dnative.train.json 的 native_gain 梯度范数 step0=0.0、全程 max=0.115，v5 产物根本未记录该键。数值声明不可采信。
2. **[中] "32 层共享标量 alpha" 措辞不精确**（§2）：v3/v5 是逐槽位门控向量跨层共享，真标量只在 v1 路径（gamma_logit）。病灶实质（跨层共享幅度 + 全局硬帽联合缩放）成立。
3. **[中] S1c 证据不在 docs/01**（§5）：在 docs/02-method-exploration.md:3。
4. **[低] docs/04:96-97 残留旧结论**（§4）："16k 多数样本根本不进融合路径"与同文件 :78-85 的结论修正及融合子集数据直接矛盾，需作废标注。
5. **[低] docs/01 所有结论均为归档索引摘录**，正文缺失（docs/01:5 自认），单独引用需注明证据等级。
