# v7 天花板实测：压缩几乎不损失 F1，融合算子没有可恢复空间

日期：2026-10-21 | 代码基线：`22cf7a9`（本次新增 `tools/ceiling_report.py`、`PRLMC_READER_PATH`）
状态：**hotpotqa 部分结果（96 配对样本）+ narrativeqa 部分结果（90 样本），已按用户指令中止**
产物：`runs/longbench_prlmc_v7_eval/`（Selection/SharedMerge 部分行）、`runs/longbench_prlmc_v7_oracle/`（Full50 部分行）

---

## 0. 结论

1. **在 hotpotqa 上，把 20 块压成 9 块几乎不损失任何 F1。** 无压缩 Full50 = 33.35%，同预算 Selection = 33.25%，差值 **+0.10pp**，配对 bootstrap 95% CI **[−6.21pp, +6.53pp]**。
2. **因此"恢复被丢弃信息"的天花板就是 +0.1pp**（任何融合算子都不可能超过信息全保留）。v7 实测 −0.47pp（CI [−6.37pp, +5.52pp]），W/L/T = 11/16/69，即**未检测到任何效应**。
3. **v5 宣称的 hotpotqa +1.70pp 高于该天花板**（v5 `t2_final_stats`：sel 35.01 / sm 36.70 / CI [−2.72, 8.05]），逻辑上不可能是真实恢复效应。
4. **v7 的架构工作（分代配额、tanh 软界、稀疏高 SNR 门控、原生支路解耦）在一个效应量为零的空间里优化**，因此"效果平淡"不是算子实现问题。
5. **评测设置本身有六个缺陷**（§2），其中"缺少无压缩天花板参照"和"统计功效不足"是导致此前所有结论不可信的根本原因。

---

## 1. 实测数据

### 1.1 hotpotqa（96 配对样本，13/9 几何，16k / tail 2048 / reset 协议）

| 臂 | `official_qa_f1_truncated` | 相对 Selection |
|---|---|---|
| Selection（同预算选择器） | 33.25% | — |
| **Full50（无压缩，天花板）** | **33.35%** | **+0.10pp**，CI [−6.21, +6.53] |
| SharedMerge v7（DsupV7 ckpt `11d36751…`） | 32.78% | −0.47pp，CI [−6.37, +5.52] |

配对 W/L/T：天花板 vs 选择器 14/17/65（65 平局）；v7 vs 选择器 11/16/69。

### 1.2 历史无压缩读数（hotpotqa，200 样本全量）

`runs/longbench_native50_v5metric_20261018_1050/`，同一套 truncated F1 口径：

| 协议 | F1 |
|---|---|
| `default/`（样本间不重置内存） | 35.80% |
| `reset/`（每样本重置，与 v5/v7 同协议） | **34.32%** |

对照 v5 `t2_final_stats.json`：Selection 35.01%、SharedMerge 36.70%。**reset 口径下 Selection 反而比无压缩高 0.7pp**——压缩损失在噪声内为零。

### 1.3 narrativeqa（部分，90 样本）

Full50 = 22.76%；v5 全量 Selection 21.85%、SharedMerge 22.60%。**不同 run、未配对**，仅作方向参考：同样没有显示出可恢复空间。

### 1.4 统计功效

hotpotqa 配对差值 per-sample sd = **0.3256**。要在 80% power、α=0.05 下检测 **1.7pp** 效应，需要 **n ≈ 2880 样本**。既有 200 样本评测的 CI 宽度约 ±5–8pp，**从来没有能力证伪或证实 1–2pp 级效应**。

---

## 2. 评测设置的六个缺陷

| # | 缺陷 | 后果 | 修正 |
|---|---|---|---|
| D1 | **没有无压缩天花板参照** | 无法区分"算子弱"与"根本没有可恢复信息"；v5 的 +1.70pp 就是这样被当成涨点 | 每个任务必须同时跑 Full（`--capacity-blocks 50`） |
| D2 | **统计功效不足** | 200 样本 CI ±5–8pp，1–2pp 效应不可判定 | 报告 `required_n`；n 不足时只能声明"未检测到" |
| D3 | **两套评分口径混用** | 同一 run 的 hotpotqa：raw F1 9.44% vs truncated 35.80%，差 3.8 倍 | 统一 truncated；raw 只作附注 |
| D4 | **几何不同轴** | v5 19/9 每样本最多压 1 次；v7 13/9 压 2–4 次（分布 {0:15,1:10,2:12,3:21,4:37}），绝对分不可比 | 只比同几何的 Δ；跨几何只比相对 Selection 的增益 |
| D5 | **内存初始化协议不统一** | `default` vs `reset` 在 hotpotqa 上差 1.5pp（35.80 vs 34.32），足以伪造或抹掉一个"涨点" | 固定 `--reset-memory`，并写入 identity |
| D6 | **reader 加载路径硬编码 GPFS** | `MODEL_PATH` 指向 37GB FP32，每行白耗 ~12 分钟；10 行 = 2 小时 | 新增 `PRLMC_READER_PATH`，指向 `/dev/shm/s1d_reader_fp16`（加载 ~4 秒），路径写入 identity |

---

## 3. 修正后的评测协议（强制）

1. **三臂同跑**：每个任务必须同时产出 Selection / Treatment / Ceiling（Full50），同 seed、同几何、同 `--reset-memory`。缺 Ceiling 的行不得进入结论。
2. **只认配对 CI**：所有效应量报 paired bootstrap 95% CI（`tools/ceiling_report.py`，resamples 2000, seed 0）。
3. **效应必须落在天花板内**：`effect.mean > headroom.high` 直接判为不可信（工具输出 `effect_exceeds_headroom`）。
4. **headroom 覆盖 0 即停止调参**：工具输出 `headroom_absent` 时，说明该任务没有可恢复信息，继续改算子没有意义。
5. **样本量**：以 `required_n` 为准；达不到就只声明"未检测到效应"，不声明涨点。
6. **口径唯一**：`official_qa_f1_truncated`；raw F1 只作附注，不得跨口径比较。

---

## 4. 复现命令

```bash
# 天花板（无压缩）
python tools/longbench_memoryllm_eval.py \
  --data-dir /share/latent/benchmarks/LongBench/data \
  --model-path /dev/shm/s1d_reader_fp16 \
  --capacity-blocks 50 --total-tokens 16384 --tail-tokens 2048 \
  --protocol-mode prlmc_aligned --reset-memory \
  --cuda-device 8 --tasks hotpotqa \
  --out runs/longbench_prlmc_v7_oracle/full50_hotpotqa.json

# 同预算选择器 / 融合臂
PRLMC_READER_PATH=/dev/shm/s1d_reader_fp16 \
python tools/longbench_prlmc_eval.py --arm Selection \
  --budget-blocks 20 --window-blocks 13 --target-blocks 9 \
  --tasks hotpotqa --total-tokens 16384 --tail-tokens 2048 --cuda-device 8 \
  --out runs/longbench_prlmc_v7_eval/Selection_hotpotqa.json \
  --data-dir /share/latent/benchmarks/LongBench/data

# 天花板裁决
python tools/ceiling_report.py \
  --selection runs/longbench_prlmc_v7_eval/Selection_hotpotqa.json \
  --ceiling   runs/longbench_prlmc_v7_oracle/full50_hotpotqa.json \
  --treatment runs/longbench_prlmc_v7_eval/SharedMerge_hotpotqa.json \
  --allow-partial
```

---

## 5. 对主线的影响

- **不能再用"恢复被丢弃信息"作为命题**：hotpotqa 上该信息量为零，narrativeqa 亦无迹象。
- 可选的重构方向（需另行立项，不在本文件决定）：
  1. **效率叙事**：用 9 块达到 Full50 的水平——但既然 Selection 已经做到，需要证明的是"更少块/更小预算仍保持"，即把预算压到 8/4 块，那里 Selection 会掉，融合才有空间。
  2. **换任务族**：找压缩确实有害的设置（多跳证据分散、长答案细节密集），先用 `ceiling_report.py` 筛出 `headroom` 显著为正的任务，再谈算子。
  3. **提高预算压力**：在 budget 10 / 8 / 4 上重做 headroom 扫描，确定"从第几块开始压缩才真的伤"。
- **当前不建议**继续在 20/13/9 上做架构消融：该几何下 headroom ≈ 0，任何消融都只能测到噪声。
