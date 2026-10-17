# G2b — 多样探针复验（selection-independent probe check）

- 日期：2026-10-07
- 状态：**方向复现，统计不充分**
- 代码：`tools/g2b_diverse.py`
- 运行：`runs/g2b_report.json`、`runs/g2b_run.log`
- 目的：回应 G2 的混淆风险，不覆盖原始 G2 FAIL

## 为什么需要 G2b

G1/G2 的原始探针有一个会影响机制解释的条件：只从 native `select` 会丢的块
里收题。G2 进一步发现，select 丢的 19 题里 18 题属于 singleton 独占事实。
这可能是方法效果，也可能是题目分布被筛选器塑形。

G2b 去掉这个条件：探针先从整个窗口收集，再按事实类型分层抽样。每个
context 的 discovery 和 held-out 使用不重合的事实行；梯度只看 discovery，
结论只看 held-out。每个分层尽量等量抽取 current 与 singleton；row 3 的
current 事实太少，因此按可用数量自适应，避免复制同一事实充当独立样本。

这里不人为制作 superseded gold 题目：Conflict_Resolution 的冻结规则是
最新事实优先，旧事实没有合法的 gold answer。superseded 仍保留在 G2 的
块级分层统计中。

## 结果

| split | select lost | sic lost | 降幅 | sic/select |
|---|---:|---:|---:|---:|
| discovery | 7 | 6 | 14.3% | 0.857 |
| **held-out** | **7** | **4** | **42.9%** | **0.571** |

held-out 逐 context：

- row 0：select 3，sic 0
- row 1：select 3，sic 3
- row 2：select 0，sic 0
- row 3：select 1，sic 1

held-out eligible 共 33 题。配对差异 b=4（select 独丢）/ c=1（sic 独丢），
McNemar 连续性校正 p=0.371。方向与 G1 一致，但样本太小，统计证据没有
达到显著线。

## 判读

G2b 给出两个同时成立的结论：

1. **G1 的效果方向没有依赖原始的 select-dropped 采样条件。** 换成
   selection-independent、current/singleton 平衡、discovery/held-out 分离
   后，sic 仍比 select 少丢 43% 的 held-out eligible 题。
2. **当前证据不足以宣称稳健统计复现。** 只有 33 个 eligible 题，p=0.371；
   这个结果支持“值得扩大样本继续测”，不支持“G1 已在多样题集上再次显著”。

因此原始 G2 的机制结论仍按预注册纪律保留：版本区分假设 FAIL。更稳妥的
当前叙事是：sic 的优势在多样探针上方向复现，最强的已验证机制仍是
singleton/独占信息保护；版本语义只作为 G0 的动机背景，不作为已证实机制。

## 下一步

G3 前先补一轮更大的 selection-independent held-out 集：扩展窗口起点和
context，而不是在同一事实行上生成更多相关问法。目标至少 100 个 eligible
held-out 题，之后再决定 G2b 是否能升级为正式统计门。sicm 合并臂的 G3
实现与 G2b 结果相互独立，可以继续，但报告中保留这项样本量限制。
