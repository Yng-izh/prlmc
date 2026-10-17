# 03 — 自然融合主线（v1 → v2 A/B/C → v3 C/D）

目标：不再用人工模板 episode，而是在原始 SQuAD context 构成的连续写入流上训练并评测共享融合模块；沿用 MemoryLLM reader，冻结后评测。当前待运行方案是 **v3 的 C/D 对照**。

## v1（自然跨块融合首轮）——未通过门槛

runs/natural_merge_initial_v1.results.md：64/64 次更新（94.6 分钟），开发 E1 完成。93 道开发问题：Full 35 / Selection 26 / 新 SharedMerge 27 / 新 CarrierOnlyFusion 27（作者答案包含率 37.63% / 27.96% / 29.03% / 29.03%）。上一轮开发 E1 未通过门槛；历史失败结果与原始证据保留在 runs/。

## v2（A/B/C 三臂）——未通过推进门槛

A=块级 SharedMemoryMerge（平均原始答案 CE）、B=A+压缩损失加权 CE+Full 答案位置 KL、C=新 SlotMemoryMerge；B−A 对应监督变化，C−B 对应结构变化。协议正文已归档：archive/2026H1_legacy/experiments/Natural-merge-v2-ABC.md。

三组均完成 64 次更新、训练内生成诊断和开发 E1。93 道开发问题上 Full/Selection/A/B/C 分别 35/26/27/27/27；三组相对 Selection 新增同一道题，无损伤，但均未恢复 Full-correct/Selection-wrong 的 9 道压缩损失题。非载体完整融合−屏蔽融合净收益均为零，三组都未通过推进门槛，未启动正式 E1/E2/E3。原始证据：runs/natural_merge_v2_abc_20261014_023355_r2/（comparison.json、逐题 e1_development.json、日志）。

## v3（C/D 固定对照）——已停线

协议：experiments/Natural-merge-v3-CD.md。同一训练监督、总体残差预算及课程下，C（已有稠密槽位融合 budgeted_dense_slot_v3）vs D（分组稀疏门控 grouped_gated_sparse_slot_v3：第 16 层 Top-8 丢弃槽位注意力、8 组 rank32、逐槽位门控上限 .20）。总残差上限 RMS 比率 ≤ .05，两者 step 0 精确等价于 Selection。数据沿用 v2 固定 312 context/1402 QA/8 流；损失 = 恢复组 CE + 0.5·KL(Full‖Shared) + 保护组 ReLU(CE_Shared−CE_Selection)。开发推进门槛：Full 最低可读性通过、Shared−Selection 净差严格为正、非载体完整融合−屏蔽融合净差严格为正；正式阶段仍需 freeze，E2 context-bootstrap CI 下界严格大于零后进入 E3。

代码与 CPU 测试已完成但 GPU 未运行；v3 线已被 v4 取代。

## v4（Dsup，grouped_gated_sparse_slot_v4）——当前方案

协议：experiments/Natural-merge-v4-D-supervision-native.md。在 v3 结构线上收敛到单一 Dsup 臂（分组稀疏门控，第 16 层 Top-8 丢弃槽位注意力、8 组 rank32、逐槽位门控上限 .20，总残差 RMS ≤ .05），接入原生跳连监督。训练入口 tools/natural_merge_v4_train.py；冻结评测经 tools/natural_merge_runtime.py；公开基准经 tools/longbench_prlmc_eval.py（见 docs/04）。

8k LongBench 结果与 native50 基线对比见 reports/longbench_prlmc_v4_8k_analysis.md；16k 评测进行中。

## 数据与隔离边界

原始 SQuAD 下载入口 tools/stage_natural_merge_data.py；v1 计划准备要求已观察问题集、匹配的公开干扰段落和固定 NaturalQA 计划，不能省略隔离审计。数据证书：reports/natural_merge_implementation_20261013/。
