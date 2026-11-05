# 04 — 读主表报告时的两个约定

主表（`tools/eval_lbv2.py`）除了分数，还记录两样东西。它们不影响结果，但决定一条数字能不能被解释。

## 1. `--operator-variant native` 到底关掉了什么

把 `ManifoldGraftedSlotMerge.operator_mode` 设成 `"native"`：

- 核心槽（216/块）：直接复制载体槽，**不做同块同源强化**；
- 嫁接槽（40/块）：直接复制被丢弃块里挑中的原生槽，**不做同块凝聚、不做正交上下文残差**；
- 载体选择与嫁接槽的挑选规则不变。

也就是说这个开关只关"加工"，不关"选槽"。定位实验 1、2、3、1A 全部跑在这个模式上
（结论见 [03-results.md](03-results.md)）。

## 2. `last_merge`：每个样本都记录这次压缩丢了什么

只跑一个臂还不够：如果最后一次被丢弃的窗口里根本没有题目需要的信息，阴性结果就没有解释力。
所以评测对每个样本额外记录：

```json
{"before_write": 240,
 "window_writes": [...], "kept_writes": [...], "dropped_writes": [...],
 "window_contiguous": false,
 "dropped_write_span": [166, 208], "dropped_document_fraction": [0.675, 0.85],
 "dropped_token_share": 0.041,
 "answer_wording_in_window": 3, "answer_wording_in_kept": 0,
 "answer_wording_in_dropped": 3, "answer_wording_distinctive": true}
```

这是**文本级代理，不是证明**。三个要点：

- `window_contiguous=false` 是常态：第一次压缩之后，窗口是"前几轮活下来的载体 + 最新写入"，
  索引不连续。所以位置区间按 `dropped_writes` 报（`dropped_write_span` 是首尾包络，
  `dropped_token_share` 是被丢弃块实际占的文本比例，通常只有几个百分点）。
- 三个计数是**整词短语匹配**的次数（`None` = 集合为空）。单 token 的选项（`China`、`1945`、`2`）
  在长文档里到处都是，命中没有区分度，所以同时报 `answer_wording_distinctive`；
  它和 `answer_wording_in_dropped` 都为真才说明这次丢弃真的丢掉了证据。
- `dropped` 指"这些块没有留下自己的载体"，**不等于内容丢了**：被丢弃块仍有 40/256 的槽会作为
  嫁接槽进入输出。所以 `answer_wording_in_dropped` 真、`answer_wording_in_kept` 假时，
  只能说明"证据所在的块被降级成了嫁接来源"，不能说明它完全消失了。

它也不能证明"救回的槽位被读者用上了"——那需要另一组实验。
