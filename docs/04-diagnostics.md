# 04 — 问题定位实验（A/B/C/D）

目的只有一个：**为什么以前融合有收益，现在 V10 和 Selection 打平**。
到底是融合没起作用、融合方式有问题，还是多轮压缩把收益破坏了。

## 四个臂

| 实验 | 做法 | 定位的问题 | 命令要点 |
| --- | --- | --- | --- |
| A | 只保留块，不融合 | 没有融合时的上限 | `--arm Selection` |
| B | 每轮都融合（当前 V10） | 当前方法的净增益 | `--arm SharedMerge --fusion-policy all` |
| C | 前面全部 Selection，**只有最后一轮**融合 | 是不是反复融合把信息磨掉了 | `--arm SharedMerge --fusion-policy last` |
| D | 同 C，但算子只做原生槽位替换 | V10 的加工（凝聚/正交/Core 强化）是否破坏原生信息 | C 的开关 + `--operator-variant native` |

C 和 D 的比较对象是 **A**，不是 B：A/C/D 在最后一次压缩之前的轨迹完全一致，
所以 `C − A` 就是"最后一次融合"的净效果，`D − C` 就是"额外加工"的净效果。

**两臂必须同载体**：给 C/D 传 `--carrier-schedule <B 的报告>`，让前面每一次压缩都用 B 记录下来的
载体块（Selection 只保留、不融合）。这样 A/C/D 的窗口内容和 B 一致，唯一变量是融合本身。

## 实验 D 到底关掉了什么

`--operator-variant native` 把 `ManifoldGraftedSlotMerge.operator_mode` 设成 `"native"`：

- 核心槽（216/块）：直接复制载体槽，**不做同块同源强化**；
- 嫁接槽（40/块）：直接复制被丢弃块里挑中的原生槽，**不做同块凝聚、不做正交上下文残差**；
- 载体选择与嫁接槽的挑选规则不变（所以 D 和 C 的差别只在加工）。

## 证据位置检查（每个样本都记录）

只跑 C 还不够：如果最后一次被丢弃的窗口里根本没有题目需要的信息，C 的阴性结果没有解释力。
所以评测对每个样本额外记录 `last_merge`：

```json
{"before_write": 190,
 "window_writes": [...], "kept_writes": [...], "dropped_writes": [...],
 "window_token_range": [97280, 107008], "document_fraction": [0.772, 0.849],
 "answer_text_in_window": true, "answer_text_in_kept": false, "answer_text_in_dropped": true}
```

这是**文本级代理**，不是证明：它说的是"正确选项的措辞有没有出现在被丢弃的那段原文里"。
三个都是 false 时，说明这个样本的证据不在最后窗口，C 的结果对它不具解释力，应单独剔除再看。
它也不能证明"救回的槽位被读者用上了"——那需要另一组实验。

## 先看概率，而不是只看对错

`tools/diagnose_ab.py` 直接读已有报告（不跑模型），把每题四个选项的 logit 拿出来比较：

```bash
python tools/diagnose_ab.py \
  reports/main_table/eval_selection_50b_n20.json \
  reports/main_table/eval_v10_50b_n20.json --labels Selection V10
```

**已经在 20 题上得到的第一个结论（2026-11-04）**：

| 臂 | 正确率 | P(正确选项) | 选项 logit 间距 | P ≥ 0.5 的题 |
| --- | --- | --- | --- | --- |
| Selection | 4/20 | 0.237 | 0.91 | 0/20 |
| V10 | 4/20 | 0.246 | 1.02 | 0/20 |

四选一的随机水平是 0.25。也就是说**读者对正确选项几乎没有偏好**：20 题里没有一题的概率过半，
平均间距约 1 个 logit。融合把 19/20 题的 P(正确) 挪动了一点点（9 涨 10 跌），
但只改了 2 题的预测字母，0 题改对。

这条数字比"4/20 打平"重要得多：它说明**在这个 128k / 50 块设置下，读者基本没读到证据**，
融合有没有用根本不是当前的主要矛盾。C 实验要回答的"多轮融合是否损伤"仍然值得做，
但在此之前应先确认这个设置本身是否给了读者任何可读的信号。

## 执行顺序

1. `diagnose_ab.py` 对比 A/B（已有报告，不跑模型）——已完成。
2. C：`--arm SharedMerge --fusion-policy last --carrier-schedule <B 报告>`，同 20 题，与 A 比。
3. D：C 无收益或表现异常时再做，`--operator-variant native`。
4. 历史 HotpotQA 复现：只有当上面三步都无法解释时，才回溯旧版本，确认当初的增益来自融合还是选块。
