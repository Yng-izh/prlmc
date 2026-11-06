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

## 3. 两个基础设施事实（2026-11-05 实测，影响怎么训模型）

### 3.1 这个 reader 在 train 模式下不确定，原因是 LoRA dropout

reader 里挂着 **384 个 `nn.Dropout`，全部 p = 0.1**（`lora_dropout`），对应的 384 个 `lora_B`
张量 99.999% 非零——LoRA 是活的，所以它的 dropout 会改输出。同一个池子、同一个输入：

| 模式 | 两次前向的答案 CE | CUDA RNG |
| --- | --- | --- |
| eval | 0.877455 / 0.877455 | 不动 |
| train | 0.880165 / 0.840025 | 前进 |
| train，把 p 全置 0 | 0.877455 / 0.877455 | 不动，**与 eval 逐位相同** |

**`tools/train.py` 的零 dropout 守卫只检查 `config.attention_dropout`，漏了
`lora_config['lora_dropout']`。** 所以 V10 那 704 步是在 10% LoRA dropout 噪声下训完的。
不影响已训出的 checkpoint（评测走 eval 模式，dropout 关闭），但任何需要"训练时的前向 == 评测时
的前向"的实验，都必须先把所有 `Dropout.p` 置 0，再切 `train()`。

### 3.2 反传必须给全宽记忆梯度，只有梯度检查点能装下

一次反传要物化 `32 × 12800 × 4096` fp16 = **3.12 GiB** 的记忆梯度。40 GB 卡上，
3,000 槽级的窗口 + 这份梯度 + 激活，不检查点就是 OOM（实测差 1.5–2 GB）。
而这个 fork 把检查点开在 `self.training` 上，所以**只能先置零 dropout 再进 train 模式**——
两件事绑在一起。实验 2 的训练脚本（`tools/train_source_compaction.py`）就是按这个顺序做的，
峰值 33.4 GB、6.1 s/步。

### 3.3 `methods/prlmc/probes.py` 曾缺 `@contextmanager`

2026-11-05 整理仓库时，`differentiable_memory` / `active_mask_override` / `_pool_in_place`
三个生成器丢了 `@contextmanager` 装饰器，`tools/train.py` 的 V10 训练路径因此**跑不起来**
（`with _pool_in_place(...)` 会 TypeError）。已补回。任何人改动这个文件后，至少跑一次
`python -c "from methods.prlmc.probes import _pool_in_place; print(hasattr(_pool_in_place, '__wrapped__'))"`
确认装饰器还在。
