# DnativeV7 vs Selection 完整 200 题对比（hotpotqa, 2026-10-23）

## 配置（固定）
MemoryLLM-8B，20 块预算，几何 w13→9，触发 20，seed 固定。
checkpoint: runs/natural_merge_v7_DnativeV7_20261022_1350.pt（仅 Dnative 臂）。
Dnative 前 96 题复用既有 run（--resume 续跑至 200），Selection 全新跑 200。

## 结果

| | 前 96 题（参与调参） | 后 104 题（held-out） | 全集 200 |
|---|---|---|---|
| Selection | 37.13% | 35.41% | 36.23% |
| DnativeV7 | 41.39% | 36.32% | 38.75% |
| Δ | +4.26pp | +0.91pp | +2.52pp |

后 104 配对比较：Dnative 8 胜 / 87 平 / 9 负；
配对差均值 +0.91pp，95% CI [−4.3, +6.1]（仅 17 题有差异，不显著）。

## 38.75% 相对 41.39% 下降的归因

1. **两臂同步下降**（Selection −1.72pp，Dnative −5.07pp）→ 后 104 题本身更难，
   不是 Dnative 特有退化。
2. **下降全部来自零分率上升**：Dnative 零分 42→51（44%→49%），
   非零题均值 73.6→71.3（−2.3pp，噪声级）。前 96 题有 54 道非零题，
   后 104 只有 53 道（53/104 vs 54/96），可得分题密度更低。
3. **本征失败率**：前 96 题 Full（50 块无压缩）也有 43 题零分（45%）；
   按该率推算后 104 应有 ~46 题本征零分，Dnative 实测 51，
   压缩额外损失仅 ~5 题。
4. **排除项**：context 长度（57.5k vs 55.7k 字符）、答案相对位置
   （29% vs 26%）、截断（均为 0）、压缩次数（1.15 vs 1.09）两半相当。
5. **调参红利**：+4.26pp 中含前 96 题多轮方案迭代的拟合成分；
   held-out 104 上增益方向为正但接近噪声边缘。

## 结论
- 41.39% 是"最易的 96 题"上的数字，全集预期应报 38.75%（200）/ 36.32%（held-out 104）。
- Dnative 相对优势的结构在 held-out 上保持（方向为正），但单任务 104 题不足以
  达到显著；建议跨任务（2wikimqa 等）复制验证。
- 另见 reports/block_ortho_flip9_20261023/：窗口几何（w13→19）是比融合算子
  更大的杠杆（9 题子集上 4/9 恢复 vs 1/9）。

## 文件
- SharedMerge_Dnative_trigger20_hotpotqa_full200.{json,log}
- Selection_trigger20_hotpotqa_full200.{json,log}
（Selection 臂曾被共享 GPU 上的外部进程三次挤 OOM，最终在同一 GPU 空闲后跑完。）
