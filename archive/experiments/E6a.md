## E6a — 离线单窗口 overfit：训练链验证

### 目的与形态

E6 的 compactor 从未被训练过（`up` 零初始化、`γ ≈ 0`），因此 E6 ≈ E5c 是预期行为。
E6a 用一个窗口做离线 overfit，检验 `CarrierResidualCompactor` 与整条 functional-loss
训练链能否产生参数更新。

本运行**不读 MemoryAgentBench**，不产出 EM/F1 等指标，只产出 loss 与内部状态。
驱动为 `tools/e6a_overfit.py`，在 host `/tmp/` 下执行。

窗口构造要求 probe 的支持事实落在 `select_blocks` **丢弃**的 block 里，
使 selection 可证地丢信息；判定用 `recovery_rate`，见「指标定义」。

### 配置

| 项 | 值 |
|---|---|
| 窗口 | 19 blocks = 4864 slots |
| 压缩目标 | 9 blocks = 2304 slots |
| anchor | 30 行 |
| `LINES_PER_CHUNK` | 20（19 次写入凑满 4864 slots） |
| probes | 20（由窗口内 380 条事实行合成，渲染推理时的 `rag_agent` 模板） |
| stage A | 16 步，`L_gold` 权重 0 |
| stage B | 24 步，`L = L_gold + 0.5L_KD + 0.1L_hidden + 0.05L_carrier + 0.02L_stats` |
| `lr` | 1e-3 |
| `gamma_max` | 0.05（命令行覆盖 YAML 的 0.1） |
| `use_amp` | False |
| GPU | gpu9（v100-2），32768 MB free |
| `max_probe_tokens` | 256（必须大于 ~163 token 的渲染提示词） |

### 首轮结果全部作废

E6a 首轮产出的 `accuracy` 与 `gold_loss` **全部作废**，原因见「已修复的缺陷」第 9 条：
打分跨度落在提示词中间，与 memory 池无关。以下旧表已随之删除：

- 四阶段 accuracy / gold_loss 表（`uncompacted`、`before`、stage A、stage B）
- `acceptance` 字典（`before_accuracy`、`after_accuracy` 均为错跨度产物）
- stage A / stage B 逐项 loss 表
- 内部状态表（该轮以错误目标训练，参数更新不可用）

`gamma`、`residual_ratio`、`route_scale` 等状态量本身是真实读数，但它们描述的是一轮
以错误目标训练的更新，不作为任何结论的依据。

### 硬窗口筛选结果（有效）

跨度修复后重新筛选，`--dry-run` 不训练。首个候选即命中，窗口 `row 3 start 17890`：

| 指标 | 值 |
|---|---|
| 写入 | 19 blocks = 4864 slots |
| selection 丢弃 | 10 blocks |
| 候选 probes | 20 |
| `no-mem` | 0/20 |
| `full`（未压缩满池） | 6/20 |
| `select`（selection 后） | 3/20 |
| **`lost`** | **3** |
| `eligible`（池必需） | 6 |

`carrier_residual == select_blocks at init: True (max |d| = 0.000e+00)`。

被 selection 丢掉、需要靠 fusion 恢复的三条：

```text
Who created Girls?
What sport is Charlie Manuel associated with?
Who is Victor Turner employed by?
```

`weights` 修复生效的直接证据——加权与不加权选出的 dropped 集合不同：

```text
[e6a]   weights change the dropped set: [0,1,2,3,4,5,6,7,8,9] vs [0,1,2,3,4,5,6,8,9,11] (unweighted)
```

### 指标定义

判定不看总体准确率（20 题中只有 3 题 `lost`，信号会被稀释）：

| 指标 | 定义 |
|---|---|
| `eligible` | `full` 对且 `no-mem` 错，即池确实必需 |
| `lost` | `eligible` 且 `select` 错，selection 真的丢了 |
| `recovered` | `lost` 且 `after` 对，fusion 塞了回来 |
| `recovery_rate` | `recovered / lost`，取值 0、1/3、2/3、1 |

`after` 由训练路径产出，`--dry-run` 不产生该值。

### 第一次全跑：40 步更新全部被跳过

第一次完整 E6a 跑完，`before` 与 `after` 逐位相同：

```text
before {'accuracy': 1.0, 'gold_loss': 0.8384018540382385}
after  {'accuracy': 1.0, 'gold_loss': 0.8384018540382385}
```

这两行的 `accuracy` 与 `gold_loss` 数值来自错跨度评分（缺陷 9），数值本身不作数；
但"`before` 与 `after` 逐位相同"这一事实成立，它就是该缺陷的下游证据——
`GradScaler` 每步都跳过更新。

`state_report` 中 `gamma = 0.0023712937` 与初始化值逐位相同，`carrier` 分量 40 步
全为 `0.0`（`up.weight` 恒为 0）。加 `PRLMC_GRAD_TRACE=1` 后：

```text
[grad] step 0 cand_leaf=nan scale=65536.0 | gamma_logit=nan route_logit=nan q.weight=nan k.weight=nan down.weight=nan up.weight=nan
[grad] step 1 cand_leaf=nan scale=32768.0 | gamma_logit=nan route_logit=nan q.weight=nan k.weight=nan down.weight=nan up.weight=nan
```

`scale` 从 65536 降到 32768，即 `GradScaler` 检测到 inf/nan 后每一步都跳过
`optimiser.step()`，参数保持初始化值；该次运行的 loss 数字仍正常输出。

两个独立缺陷：

1. **`consolidate.py:519`（当时）`delta / (delta_rms + 1e-6)`。** `up.weight`
   零初始化 ⇒ 第一步 `delta ≡ 0` ⇒ `delta_rms = 0`。forward 无害（`0/1e-6 = 0`），
   但 backward 的 `d(delta_rms)/d(delta_ij) = delta_ij / (N·delta_rms)` 在原点
   为 `0/0`。该 NaN 传导到除 `gamma_logit` 外的所有 compactor 参数。
2. **`probes.py` 的 `use_amp` 默认 `True`，但路径中无 `autocast`。** backbone 本身
   为 fp16，`GradScaler` 将 loss 放大 65536 倍后 backward，激活梯度超过 fp16 上限
   65504 即 `inf`，`inf * 0 → nan`。

修复后 `[grad]` 全部有限：

```text
[grad] step 0 cand_leaf=1.0039e+01 scale=1.0 | gamma_logit=1.7140e-06 route_logit=0.0 q.weight=0.0 k.weight=0.0 down.weight=0.0 up.weight=1.2535e+00
[grad] step 1 cand_leaf=7.3125e+00 scale=1.0 | gamma_logit=9.7598e-02 route_logit=2.0263e-11 q.weight=2.8726e-08 k.weight=2.8117e-08 down.weight=8.0711e-01 up.weight=3.5428e-01
```

### 硬窗口训练结果（有效，2026-09-25）

缺陷 9–12 全部修复后的完整训练，`--window 3,17890`，stage A 16 步 + stage B 24 步。
命令：

```text
E6A_NEED_MB=21000 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=9 python /tmp/e6a_overfit.py \
  --window 3,17890 --steps-a 16 --steps-b 24 \
  --probes 20 --max-probe-tokens 256 \
  --lr 1e-3 --gamma-max 0.05 --min-probes 4 --min-lost 3 \
  --out /tmp/e6a_report.json
```

筛选指标（`--dry-run` 阶段，与上表逐位一致）：

| 指标 | 值 |
|---|---|
| `no_memory` | 0/20 |
| `uncompacted` | 6/20 |
| `before`（`select_blocks`） | 3/20 |
| `lost` | 3 |
| `identity_at_init` | `True`（`max |d| = 0.000e+00`） |

训练结果：

| 阶段 | `after` | `recovery` | `gold` 首→末 | `kd` 首→末 |
|---|---|---|---|---|
| A（16 步） | 4/20 | **1/3 = 0.333** | 3.8737 → 3.7321 | 1.4617 → 1.0696 |
| B（24 步） | 4/20 | **1/3 = 0.333** | 3.7168 → 3.2787 | 1.0802 → 0.9928 |

`acceptance`：

```text
{'n_eligible': 6, 'n_lost': 3, 'n_recovered': 1, 'recovery_rate': 0.3333333333333333,
 'lost_questions': ['Who created Girls?', 'What sport is Charlie Manuel associated with?',
                    'Who is Victor Turner employed by?'],
 'recovered_questions': ['What sport is Charlie Manuel associated with?'],
 'no_memory_accuracy': 0.0, 'before_accuracy': 0.15, 'after_accuracy': 0.2,
 'uncompacted_accuracy': 0.3}
```

即 `What sport is Charlie Manuel associated with?` 被塞了回来，
`Who created Girls?` 与 `Who is Victor Turner employed by?` 仍丢失。

内部状态（阶段末）：

| 量 | A 末 | B 末 | 预算 / 参照 |
|---|---|---|---|
| `gamma` | 0.0024011 | 0.0024612 | `gamma_max` = 0.05，用到 **4.8%** |
| `residual_ratio` | 0.00072533 | 0.00142481 | 判据要求 < 5%，实测 **0.14%** |
| `route_scale` | 0.018158 | 0.018614 | — |
| `routing_entropy` | 2.9432058 | 2.9432118 | 均匀分布 |
| `forward_calls` | 34 | 83 | — |
| `amp` | 0.0 | 0.0 | — |

`carrier` 分量按步单调上升（A：`0.0 → 3.97e-05`；B：`4.26e-05 → 9.88e-05`），
说明 `up.weight` 确实在离开零初始化；`hidden` 分量全程持平在 ~0.0028，未下降。

**读数解读（不构成机制结论）**：`residual_ratio` 停在 0.14%，即注入张量与原生
`select_blocks` 的差异只占 carrier 的千分之一量级，`gamma` 只用掉预算的 4.8%。
融合项没有真正离开恒等点，因此那 1/3 的恢复不足以归因于 residual fusion。
16 + 24 步、`lr 1e-3` 的有效梯度信号太弱，`kd` 是唯一明显下降的分量。

### Stage B 延长到 220 步（同窗口同超参）

条件与上一节完全相同，只把 Stage B 从 24 步延长到 **220 步**（Stage A 仍 16 步），
固定 `--window 3,17890`，结构未改，loss 权重未调。运行 05:43:58Z → 09:38:07Z，
共 3 h 54 min，`EXIT=0`。命令：

```text
E6A_NEED_MB=21000 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=9 PRLMC_GRAD_TRACE=1 python /tmp/e6a_overfit.py \
  --window 3,17890 --steps-a 16 --steps-b 220 \
  --probes 20 --max-probe-tokens 256 \
  --lr 1e-3 --gamma-max 0.05 --min-probes 4 --min-lost 3 \
  --out /tmp/e6a_report_longB.json
```

筛选指标与前两次逐位一致（`no_memory 0/20`、`uncompacted 6/20`、`before 3/20`、
`lost 3`、`identity_at_init True`）。

#### 两版 Stage B 对照

| 项 | Stage B 24 步 | Stage B 220 步 |
|---|---|---|
| `after` | 4/20 | **6/20** |
| `recovery_rate` | 1/3 = 0.333 | **2/3 = 0.667** |
| `gold` 首→末 | 3.7168 → 3.2787 | 3.6848 → **2.4318** |
| `kd` 首→末 | 1.0802 → 0.9928 | 1.0047 → **0.7586** |
| `hidden` 首→末 | 0.0029897 → 0.0029241 | 0.0028581 → **0.0043209** |
| `carrier` 首→末 | 4.26e-05 → 9.88e-05 | 4.37e-05 → **2.244e-04** |
| `stats` 首→末 | 5.78e-06 → 6.06e-06 | 5.78e-06 → 9.60e-06 |
| `residual_ratio` | 0.0014248 | **0.0026117** |
| `gamma` | 0.0024612 | 0.0030977 |
| `route_scale` | 0.018614 | 0.0267555 |
| `routing_entropy` | 2.9432118 | **2.7599626** |
| `forward_calls` | 83 | 475 |

#### `acceptance`（220 步版）

```text
{'n_eligible': 6, 'n_lost': 3, 'n_recovered': 2, 'recovery_rate': 0.6666666666666666,
 'lost_questions': ['Who created Girls?', 'What sport is Charlie Manuel associated with?',
                    'Who is Victor Turner employed by?'],
 'recovered_questions': ['What sport is Charlie Manuel associated with?',
                         'Who is Victor Turner employed by?'],
 'no_memory_accuracy': 0.0, 'before_accuracy': 0.15, 'after_accuracy': 0.3,
 'uncompacted_accuracy': 0.3}
```

`lost_questions` 列的是全部三条 `lost`，不是未恢复项；本次未恢复的只有
`Who created Girls?`。

#### Stage A 两侧一致

两次运行的 Stage A 完全相同（16 步、`after 4/20`、`recovery 1/3`）：

| 量 | 24 步版 Stage A | 220 步版 Stage A |
|---|---|---|
| `gamma` | 0.0024011049 | 0.0024011049 |
| `residual_ratio` | 0.00072533 | 0.00074491 |
| `route_scale` | 0.018158 | 0.0181656 |
| `routing_entropy` | 2.9432058 | 2.9432061 |
| `forward_calls` | 34 | 34 |
| `carrier` 末 | 3.97e-05 | 4.08e-05 |

`gamma` 逐位相同，`residual_ratio` 有约 2.7% 的相对差异，其余量在浮点噪声范围内。

#### 读数

四个内部量在 220 步下同向移动，且**第一次明确离开恒等点邻域**：
`residual_ratio` 1.83×、`route_scale` 1.44×、`carrier` 2.27×，
`routing_entropy` 从 2.9432 降到 2.7600（路由不再均匀）。
这四点合起来比 `recovery_rate` 本身更强——它说明融合与路由确实被训练推动，
而不是数值噪声。

但两条判据仍未满足：

- `residual_ratio` 0.0026，是「< 5%」判据的**约 1/19**；`gamma` 只用到预算的 **6.2%**。
- 「reader hidden loss 下降」**反向**：`hidden` 从 0.0028581 升到 0.0043209
  （区间最低 0.0026601）。

分母只有 3，`1/3 → 2/3` 是一道题的差别，单看该比值不构成证据；
`after 6/20 == uncompacted 6/20` 同样受制于样本量。

### 附：显存

修复 reader 图与 compactor 图共存后，`[mem] step 0 before student`
从 23.02 GB 降到 21.20 GB。40 步训练期间该值稳定在 21.49–22.70 GB，无 OOM。
延长到 220 步（累计 236 步）期间同样无 OOM。

缺陷 12 修复前的对照：`alloc=22.72GB` 时 backward 需要 3.12 GiB、
仅 3.09 GiB 可用，第一步即 `torch.OutOfMemoryError`。

第一次全跑在此处 OOM：`torch.OutOfMemoryError` 于 stage A 第 0 步的 probe backward，
`GPU 0 has a total capacity of 31.73 GiB of which 68.00 MiB is free`。
当时的账目为：模型 19.57 GB + teacher pass ≈ 20.0 GB（`step 0 before student` 实测
23.02 GB）+ per-layer checkpoint 存下的满宽 layer 输入 3.37 GB + 全池 `pool.grad`
3.36 GB + compactor 活图 3.02 GB。

两条当时的猜测经实测被排除：

1. **注意力分数矩阵**：实测 mem-efficient SDPA 同时接受 bool mask 与 float `-inf` mask，
   forward 峰值 1.24–1.46 GB、backward 1.87 GB，未回退到 math backend。
2. **per-layer checkpoint 无效**：该分支有效，但 fork 在每层循环内重新拼接 12800 前缀，
   每个被存下的 layer 输入为满宽 `[1, 12801, 4096]` fp16 ≈ 105 MB，32 层 ≈ 3.37 GB。

---

[← 索引：EXPERIMENTS.md](../EXPERIMENTS.md)
