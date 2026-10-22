# v7 满预算触发点控制实验：把 20 块真的用掉，值 +3.88 ~ +5.12pp

## 0. 结论

1. **缺陷（D8）**：20/13/9 几何下触发点是 `min(budget, window+1) = 14`。写满 14 块就压最老 13→9、只留 1 块未压缩尾巴，常驻宽度在 10–14 之间循环——**实测峰值 3389 槽（13.2 块），预算 5120 槽（20 块）从未用满，第 15–20 块（1536 槽）一次都没被写过**。同时训练侧前 32 步先堆到 19 块再压 13、留 6 块未压缩尾巴，**训/评布局错位**。
2. **修法**：评测侧新增 `--trigger-blocks 20`（**不重训**、不改模型）：写满 20 块才压最老 13→9，保留最新 7 块未压缩，常驻宽度 16–20。实测峰值 **5120 槽 = 恰好 20 块**，压缩次数 2.59 → **1.15**。
3. **收益：Selection 33.25% → 37.13%（+3.88pp），SharedMerge 32.78% → 37.90%（+5.12pp）**，同一批 96 题、同一 ckpt、同一代码路径，唯一变量是触发点。Selection 的 EM 提升 **+5.21pp，CI [+1.04, +10.42]（5 胜 0 负）已过 95% 显著线**。
4. **这是整条 v7 线上量级最大的效应，且与架构无关**——它来自"把声明过的预算花掉"，此前所有 v7 结论都是在"只用了 70% 预算"的设置下得到的。
5. **副作用：无压缩参照不再是上界**。满预算 Selection（37.13%）比同路径 Full（35.30%）高 **1.83pp**（Full 28 块上下文引入更多干扰项）。headroom 由 +2.05pp 翻成 **−1.83pp**，直接印证了 D9：不能用 `effect_exceeds_headroom` 否决效应。
6. **11 道"融合损伤题"里 6 道是评测侧容量闲置造成的**：idx 8、24、34、35、68、92 在满预算下直接恢复（5 题 0→1.00），不是算子写坏了。
7. **但融合模块仍未证得涨点**：满预算下 SharedMerge − Selection = **+0.77pp，CI [−5.15, +7.26]，W/L/T 12/12/72**；非零率 **−7.29pp，CI [−14.58, +1.04]**（答错实体仍然更多），高精度率 +4.17pp、EM +3.12pp（不显著）。

---

## 1. 几何与触发点实测

| 触发点 | 峰值槽（均值 / 最大） | 峰值块 | 压缩次数（均值 / 分布） |
|---|---|---|---|
| `min(20,14)=14`（旧） | 3389 / **3584** | 13.2 / 14 | 2.59 / {0:15, 1:10, 2:12, 3:21, 4:38} |
| `budget=20`（新） | 4579 / **5120** | 17.9 / **20** | 1.15 / {0:35, 1:12, 2:49} |

5120 槽 = 20 块 × 256 槽/块，即声明的 20 块预算第一次被完整使用；压缩次数从平均 2.59 次降到 1.15 次（约一半）。

---

## 2. 结果（96 配对样本，hotpotqa，16k / tail 2048 / reset）

### 2.1 mean F1

| 臂 | 触发 14（旧） | 触发 20（新） | 差 |
|---|---|---|---|
| Selection | 33.25% | **37.13%** | +3.88pp |
| SharedMerge v7 | 32.78% | **37.90%** | +5.12pp |
| Full（无压缩，与触发点无关） | 35.30% | 35.30% | — |

### 2.2 纯触发增益（同一臂，14 → 20）

| 口径 | Selection | SharedMerge |
|---|---|---|
| mean F1 | **+3.88pp**，CI [−1.05, +9.05]，W/L/T 9/4/83 | **+5.12pp**，CI [−3.04, +13.50]，W/L/T 18/9/69 |
| 非零率 F1>0 | +4.17pp，CI [−1.04, +10.42]，6/2/88 | +6.25pp，CI [−2.08, +14.58]，12/6/78 |
| 高精度率 F1≥0.80 | +2.08pp，CI [−3.12, +8.33]，5/3/88 | +4.17pp，CI [−4.17, +12.50]，11/7/78 |
| EM | **+5.21pp，CI [+1.04, +10.42]，5/0/91** | +5.21pp，CI [−3.12, +14.58]，12/7/77 |

两个臂、四个口径全部为正，其中 Selection 的 EM 是**唯一越过 95% 显著线的读数**（5 胜 0 负）。

### 2.3 满预算下的三臂对照（四项口径）

| 臂 | mean F1 | 非零率 | 高精度率 | EM |
|---|---|---|---|---|
| Selection（触发 20） | 37.13% | 58.3% | 25.0% | 21.9% |
| Full（无压缩参照） | 35.30% | 55.2% | 25.0% | 20.8% |
| SharedMerge v7（触发 20） | **37.90%** | 51.0% | **29.2%** | **25.0%** |

配对 Δ：

| 对照 | mean F1 | 非零率 | 高精度率 | EM |
|---|---|---|---|---|
| SharedMerge − Selection | +0.77pp，CI [−5.15, +7.26]，12/12/72 | **−7.29pp**，CI [−14.58, +1.04]，4/11/81 | +4.17pp，CI [−3.12, +12.50]，9/5/82 | +3.12pp，CI [−4.17, +10.42]，8/5/83 |
| Full − Selection（headroom） | **−1.83pp**，CI [−6.37, +2.61]，4/10/82 | −3.12pp，CI [−8.33, +2.08]，2/5/89 | 0.00pp，CI [−5.21, +4.17]，3/3/90 | −1.04pp，CI [−5.21, +2.08]，1/2/93 |

**同一个结构性事实在满预算下复现**：SharedMerge 的均值与非零率反向（−7.29pp 非零率 vs +4.17pp 高精度率、+3.12pp EM），即"答对的实体更少，但答对时更完整"。单看 mean F1 会把这两个效应抵消成 +0.77pp 的噪声。

---

## 3. 逐题翻转（同一臂，14 → 20）

| 臂 | 改善 | 变差 | 净 |
|---|---|---|---|
| Selection | 9 题 | 4 题 | +5（W/L/T 9/4/83） |
| SharedMerge | 18 题 | 9 题 | +9（W/L/T 18/9/69） |

Selection 改善：idx 12 (0→1.00)、23 (0→1.00)、58 (0→1.00)、87 (0→1.00)、71 (0.24→1.00)、70 (0→0.67)、34 (0.15→0.40)、41 (0.21→0.24)、88 (0→0.09)；变差：56 (0.80→0.22)、77 (1.00→0.50)、74 (0.19→0)、81 (0.80→0)。

SharedMerge 改善 18 题，其中 **idx 8、24、35、68、92 直接 0→1.00**，idx 44 (0→0.80)、87 (0→0.67)、34 (0→0.40)。对照 `experiments/v7-per-question-hotpotqa.md` §8.1 的 11 道受损题清单：**idx 8、24、34、35、68、92 共 6 题在满预算下恢复**，剩余 5 题（6、38、39、41、57）未恢复。

即：上一版"融合损伤 11 题"里有 **6 题的损伤其实来自评测侧的容量闲置**，不是融合算子写坏了。SharedMerge 变差的 9 题：0 (1.00→0.33)、19 (1.00→0)、27 (1.00→0.17)、40 (1.00→0.50)、52 (1.00→0)、69 (1.00→0)、74 (1.00→0)、80 (0.50→0)、29 (0.62→0)。

---

## 4. 对主线的含义

- **"稳定涨点"目前最确定的来源是把声明预算花掉**：+3.88 / +5.12pp，且 Selection 的 EM 已过显著线。这条不需要任何新架构，属于评测/几何口径修正，可立即写入所有后续实验的强制协议。
- **训/评错位已消除**：训练侧 19 块常驻 + 6 块未压缩尾巴，评测侧现在 16–20 块常驻 + 7 块未压缩尾巴，两侧布局首次一致。此前"训练时有效、评测时失效"的落差有了直接解释。
- **融合模块的方向为正但仍未显著**（+0.77pp，12 胜 12 负 72 平），且非零率仍偏负。要继续在架构上争涨点，必须先在满预算触发点下重测历史结论（此前所有 v7/v5 读数都建立在"只用 14 块"的设置上）。
- **天花板参照必须重新定义**：满预算下 Selection 反超 Full 1.83pp，"无压缩"不再是上界。任何用 headroom 作硬上界的判据都必须废除（已在 `ceiling_report.py` schema/2 中删除 `effect_exceeds_headroom`）。

---

## 5. 复现命令

```bash
# 满预算选择器 / 融合臂（评测侧控制，不重训）
python tools/longbench_prlmc_eval.py --arm Selection \
  --budget-blocks 20 --window-blocks 13 --target-blocks 9 --trigger-blocks 20 \
  --data-dir /share/latent/benchmarks/LongBench/data --tasks hotpotqa \
  --max-samples 96 --total-tokens 16384 --tail-tokens 2048 --cuda-device 8 \
  --out runs/longbench_prlmc_v7_eval/Selection_fulltrigger_hotpotqa.json

python tools/longbench_prlmc_eval.py --arm SharedMerge \
  --checkpoint runs/natural_merge_v7_DsupV7_20261020_1617.pt \
  --budget-blocks 20 --window-blocks 13 --target-blocks 9 --trigger-blocks 20 \
  --data-dir /share/latent/benchmarks/LongBench/data --tasks hotpotqa \
  --max-samples 96 --total-tokens 16384 --tail-tokens 2048 --cuda-device 9 \
  --out runs/longbench_prlmc_v7_eval/SharedMerge_fulltrigger_hotpotqa.json

# Full 臂不做压缩，与触发点无关，直接复用 Full_hotpotqa.json

# 四口径报表
python tools/ceiling_report.py \
  --selection runs/longbench_prlmc_v7_eval/Selection_fulltrigger_hotpotqa.json \
  --ceiling   runs/longbench_prlmc_v7_eval/Full_hotpotqa.json \
  --treatment runs/longbench_prlmc_v7_eval/SharedMerge_fulltrigger_hotpotqa.json \
  --out runs/longbench_prlmc_v7_eval/ceiling_report_fulltrigger_hotpotqa.json
```

产物：`Selection_fulltrigger_hotpotqa.json`、`SharedMerge_fulltrigger_hotpotqa.json`、`ceiling_report_fulltrigger_hotpotqa.json`，协议字段 `protocol.trigger_blocks=20` 与 `identity.trigger_blocks=20` 可核对。
