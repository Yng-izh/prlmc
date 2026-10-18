# 自然融合 v5（Dsup v5，grouped_gated_sparse_slot_v5）

## 目的

v4 Dsup 在 16k 上已经稳定高于 Selection（hotpotqa +2.83、musique +3.58、2wikimqa +0.85、qasper +0.07、narrativeqa +0.15，official_qa_f1_truncated，n=200），但增益集中在多跳任务，Qasper/NarrativeQA 接近完全抵消（逐题配对 2 胜 0 负 / 27 胜 25 负）。本轮**不改论文主线方案**：结构主张、监督形式、预算、评测口径全部保持，只调内部路由与门控，目标是让同一份增益跨任务、跨域稳定。

## 四处改动

| # | 改动 | 原状（v4） | 现状（v5） | 为什么 |
|---|---|---|---|---|
| 1 | 几何门控 | `slot_gate` 只看槽位隐状态 | 块内 z-score 的 3 个标定无关特征（路由 top-1 logit、top-k 概率锐度、与块质心冗余度）经零初始化 `rel_gate(3→1)` 加进门控 | 隐状态门控无法判断「这个载体槽位是否真的匹配到了源内容」，域外隐状态水平会漂移 |
| 2 | 路由先验混合 | 只学 32-D 低秩投影的余弦 | `0.6·cos(原生 4096-D) + 0.4·cos(学习投影)`，除温度 0.04；原生视角为常数项不反传 | 低秩投影在域外退化时，预训练表示托住路由；梯度仍只走学习视角 |
| 3 | 原生增益 | 单个全局标量 | `num_groups` 个有符号增益；原生支路取 top-2 最锐槽位全宽源均值减载体，按路由置信度缩放 | 浅层寻址、深层读出是两种需求，一个标量无法同时满足；4096-D 平均会糊掉稀有实体/数字词元 |
| 4 | 载体选择层 | `_SELECT_LAYER = _REFERENCE_LAYER = 0` | 两者都是 16，与路由层 `reference_layer=16` 一致 | 选择器、目标方向、路由读取跨流形，是域内增益不能迁移的原因之一 |

另有监督项 `protection_anchor_weight = 0.10`：对 Full 答案起始词元分布做 KL 锚定，教师是无梯度 Selection 前向，只在保护组激活。

## 结构不变量（CPU 测试守，不得破坏）

1. step 0 输出与 Selection 逐位相同；
2. 全局残差 RMS 预算 ≤ 0.05；
3. `source_allow_mask` 消融精确退化为 Selection；
4. `state_digest` 稳定（只含 state_dict）；
5. coverage mass 求和为 1。

## 配置

`tools/prepare_natural_merge_v4.py` 的 `TRAIN_V4`：

```text
variants.Dsup.architecture      grouped_gated_sparse_slot_v5   local_gate_max 0.20
variants.Dnative.architecture   grouped_gated_sparse_native_v5 local_gate_max 0.20
groups 8   topk 8   reference_layer 16   initial_gate 0.02
global_rms_budget 0.05   native_prior_weight 0.60
route_mix_temperature 0.04   balance_strength 0.15
protection_anchor_weight 0.10   gradient_clip 1.0
```

v4 结构名已从 `tools/natural_merge_runtime.py` 注册表移除：v4 checkpoint 用新代码加载会直接报 `unknown shared merge architecture`，避免被静默重新解释。

## 执行顺序

1. `CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tools -p "test_*.py"`（当前 78 项通过）；
2. native50 基线按原协议重跑 hotpotqa：`tools/longbench_memoryllm_eval.py ... --reset-memory --total-tokens 16384`，复核绝对水平异常偏低（hotpotqa 8.83 vs 公开 ≈33+）属于协议问题还是方法问题；
3. v5 训练：`tools/natural_merge_v4_train.py --plan <v4 plan> --variant Dsup --cuda-device N --out runs/<tag> --checkpoint-out runs/<tag>/Dsup.train.pt`；
4. 冻结评测：16k 的 `Selection` 与 `SharedMerge` 两臂全量重跑。**Selection 臂必须重跑**，因为 `_SELECT_LAYER` 由 0 改为 16，不重跑就与 Dsup 臂不同轴。8k 不必重跑两臂：该长度下压缩不触发、两臂 prediction 逐字相同（已实测）；要扩大融合覆盖面应提高输入上限（`--total-tokens 32768`，需先放开工具的 `choices`）。

## 已知风险

- 原生支路在 step 0 的梯度量级比学习支路大（`native_gain` ≈1.4e5 vs `up_weight` ≈8e2），来源是零初始化时 `combined` 的自 RMS 归一化。Adam 的更新对整体梯度缩放不敏感，且训练全部 fp32、无 GradScaler，因此不改写；若后续引入混合精度需重新评估。
- `reference_layer=16` 要求 reader 至少 17 层；更小读者由 `_reference_layer_index()` 夹取，CPU 契约靠这一点跑通，但小于 17 层的真实 reader 不在承诺范围内。
