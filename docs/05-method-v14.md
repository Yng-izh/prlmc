# V14：互补关系感知的跨块校准（代码口径）

本文只写**对当前代码有约束力**的内容。V13 的设计与结论见
[05-method-v13.md](05-method-v13.md)，数字见 [03-results.md](03-results.md) 第 15 节。

**已跑完：V14 65/158，比 V13 的 67 少 2 题（2 救 4 伤，p ≈ 0.69），判定为负结果。**
下面"判定线"一节是运行前写死的，保留原文以便对照：三条里第 1 条不过，第 2、3 条过
（对 donor +8、损伤 11 低于 V13 的 12）。因此本次改动测的不是"互补关系这个想法"，
而是它的一个具体实现（2 同类 + 2 异类、等权、加成一个 context、进同一个 rank-64 瓶颈）——
两组的身份在聚合那一步被平均掉了，要继续这条线必须先改聚合与注入通路。

## 一句话

窗口、预算、容量分配、校准器结构、损失、训练步数、cross 学习率**全部不动**，
只改一件事：每个槽位的 4 个跨块邻居**从哪来**。

| | V13（`--neighbor-mode topk`） | V14（`--neighbor-mode complement`） |
|---|---|---|
| 邻居选择 | 全部跨块里 Key 余弦 top-4，**不看槽位类型** | 同类 Top-2（同 core / 同 graft）+ 异类 Top-2（跨 core/graft 两侧） |
| 邻居数量 | 4 | 4（`--k-same 2 --k-cross 2`） |
| 组内权重 | 4 个邻居一个 softmax | 每组各一个 softmax，两组总权重各 0.5 |
| 其余全部 | — | 完全相同 |

## 为什么改这个

V13 的机制有效（67/158），但它不区分"原本保留的知识"和"从删除块搬进来的新知识"。
2,304 个槽位里 1,620 个 core 承载保留内容、684 个 graft 承载新增内容，而 V13 给每个
槽位找的是"最相似的 4 个槽位"——一个 core 槽位的 top-4 完全可能都是 core，这样它的
校准条件里根本没有它必须一起读懂的那些新内容。

这**只是从代码提出来的假设**，V13 的报告里没有这个统计。所以 V14 同时补上测量：

```
neighbour_type_cross_fraction       邻居里"异类"的占比（跨 core/graft 的另一侧）
core_to_graft_neighbour_fraction    core 槽位的邻居里 graft 的比例
graft_to_core_neighbour_fraction    graft 槽位的邻居里 core 的比例
cross_weight_mass_mean              异类邻居实际拿到的权重份额
```

最后一项是防自欺的：如果只改选择、但 softmax 让同类邻居拿走了几乎全部权重，
V14 就在暗中退化成"我自己的 top-2"，而臂的编号还会显示它跑了。等质量分组
（每组各 0.5）就是为了不让这种情况发生。

## 改了哪些代码

| 文件 | 改动 |
|---|---|
| `methods/prlmc/cross_block_calibration.py` | `cross_block_context(..., mode, k_same, k_cross)`：`topk` 与 `complement` 两种邻居构造；每组单独 softmax、组权重等分；报告里加 5 个来源统计。`CrossBlockResidual` 新增 `neighbor_mode / k_same / k_cross`（进 `config()`，进 checkpoint），新增 `neighbours()` 作为唯一入口——调用点不许自己拼 context |
| `tools/train_cross_block_calibration.py` | `--neighbor-mode / --k-same / --k-cross`，训练前校验 `k_same + k_cross == k_neighbors`；`calibrate_with_reg` 改用 `module.neighbours()`；`module_report` 与 checkpoint `config`/`geometry` 记录邻居规则 |
| `tools/eval_cross_block_calibration.py` | 新臂 `V14` / `V14Donor`（`--v14`）：**复用 V13 的同一个窗口**再校准一次，所以 V13 与 V14 之间只有邻居规则一个变量；新增 `versus_v13`、`versus_v14` 与 `neighbour_composition`（两种模式的来源统计）；`thresholds` 加 `complement_over_topk ≥ 4`、`v14_content_causality ≥ 5`；顺手修掉 `thresholds.total.achieved` 只报 plain 臂（61）的口径问题，加 `achieved_by_arm` |

## 训练与评测怎么跑

训练（从同一个 AnchorOnly 初始化，384 更新，其余与 V13 逐项相同）：

```
python tools/train_cross_block_calibration.py ... \
    --cross on --neighbor-mode complement --k-same 2 --k-cross 2 \
    --cross-lr 3e-3 --updates 384 --out runs/cross_block/complement.pt
```

评测（十臂，一条命令内跑完 V13 与 V14 的逐题配对）：

```
python tools/eval_cross_block_calibration.py ... \
    --v11 runs/cross_block/plain.pt \
    --v11-evict runs/cross_block/cross_lr.pt \
    --v14 runs/cross_block/complement.pt \
    --out reports/cross_block/eval_v14_hotpotqa.json
```

## 硬约束（代码里已经拦住）

- `--v14` 必须是 `complement` 模式，`--v11-evict` 必须是 `topk` 模式，且两边 `k` 相同，
  否则直接报错——两个臂必须只差邻居来源。
- `--v14` 与 `--v11-evict` 记录的配额与效用层必须一致：它们共用同一个窗口。
- V14 校准后的窗口必须与 V13 不同；一模一样说明规则没生效，不许当成"没涨点"。
- V14 的 donor 臂必须存在：内容因果性不能拿 V13 的 donor 代替。
- 上下文仍然只能来自**其他块**（同块屏蔽），邻居块分布逐行记录。

## 判定线（运行前写死）

1. `V14 − V13 ≥ +4`：邻居来源是瓶颈，改它有用。
2. `V14 − V14Donor ≥ +5`：涨点仍然来自 graft 的真实内容，不是噪声。
3. 损伤（Full 答对而被该臂答错的题数）不明显高于 V13 的 12。

三条不齐 ⇒ 邻居来源不是当前瓶颈，下一轮按计划改"逐层跨块条件"，而不是继续拧
top-k、残差上限或学习率。

## 已知限定

- 单种子；V13 的 +6 本身是超参换来的（`--cross-lr 3e-3` 是在同一批 158 行上看着惰性
  那一次选的），V14 沿用同一个学习率，这个限定不变。
- 没有独立自然多跳验收集；V14 的结论同样只在这 158 行上成立。
- `complement` 的"等质量分组"是一个设计选择，不是自然结论：另一种选择（4 个邻居一个
  softmax）大概率退化成同类 top-2，所以没有把它作为对照臂——但这也意味着本臂测的是
  "2 同类 + 2 异类、等权重"这个具体方案，不是"互补关系"这个抽象想法。
