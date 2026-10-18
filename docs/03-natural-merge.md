# 03 — 自然融合主线（v1 → v2 A/B/C → v3 C/D → v4 Dsup → v5 Dsup）

目标：不再用人工模板 episode，而是在原始 SQuAD context 构成的连续写入流上训练并评测共享融合模块；沿用 MemoryLLM reader，冻结后评测。当前方案是 **v5（Dsup v5，代码就绪、GPU 未跑）**。

## v1（自然跨块融合首轮）——未通过门槛

runs/natural_merge_initial_v1.results.md：64/64 次更新（94.6 分钟），开发 E1 完成。93 道开发问题：Full 35 / Selection 26 / 新 SharedMerge 27 / 新 CarrierOnlyFusion 27（作者答案包含率 37.63% / 27.96% / 29.03% / 29.03%）。上一轮开发 E1 未通过门槛；历史失败结果与原始证据保留在 runs/。

## v2（A/B/C 三臂）——未通过推进门槛

A=块级 SharedMemoryMerge（平均原始答案 CE）、B=A+压缩损失加权 CE+Full 答案位置 KL、C=新 SlotMemoryMerge；B−A 对应监督变化，C−B 对应结构变化。协议正文已归档：archive/2026H1_legacy/experiments/Natural-merge-v2-ABC.md。

三组均完成 64 次更新、训练内生成诊断和开发 E1。93 道开发问题上 Full/Selection/A/B/C 分别 35/26/27/27/27；三组相对 Selection 新增同一道题，无损伤，但均未恢复 Full-correct/Selection-wrong 的 9 道压缩损失题。非载体完整融合−屏蔽融合净收益均为零，三组都未通过推进门槛，未启动正式 E1/E2/E3。原始证据：runs/natural_merge_v2_abc_20261014_023355_r2/（comparison.json、逐题 e1_development.json、日志）。

## v3（C/D 固定对照）——已停线

协议：experiments/Natural-merge-v3-CD.md。同一训练监督、总体残差预算及课程下，C（已有稠密槽位融合 budgeted_dense_slot_v3）vs D（分组稀疏门控 grouped_gated_sparse_slot_v3：第 16 层 Top-8 丢弃槽位注意力、8 组 rank32、逐槽位门控上限 .20）。总残差上限 RMS 比率 ≤ .05，两者 step 0 精确等价于 Selection。数据沿用 v2 固定 312 context/1402 QA/8 流；损失 = 恢复组 CE + 0.5·KL(Full‖Shared) + 保护组 ReLU(CE_Shared−CE_Selection)。开发推进门槛：Full 最低可读性通过、Shared−Selection 净差严格为正、非载体完整融合−屏蔽融合净差严格为正；正式阶段仍需 freeze，E2 context-bootstrap CI 下界严格大于零后进入 E3。

代码与 CPU 测试已完成但 GPU 未运行；v3 线已被 v4 取代。v3 结构名仍可在运行时构建，但其内部路由实现在 v5 中统一升级；v3 从未 GPU 训练，没有 checkpoint 受影响。

## v4（Dsup，grouped_gated_sparse_slot_v4）——结果即记录，代码已退役

协议：experiments/Natural-merge-v4-D-supervision-native.md。在 v3 结构线上收敛到单一 Dsup 臂（分组稀疏门控，第 16 层 Top-8 丢弃槽位注意力、8 组 rank32、逐槽位门控上限 .20，总残差 RMS ≤ .05），接入原生跳连监督。训练入口 tools/natural_merge_v4_train.py；冻结评测经 tools/natural_merge_runtime.py；公开基准经 tools/longbench_prlmc_eval.py（见 docs/04）。

8k 结果见 reports/longbench_prlmc_v4_8k_analysis.md，16k 结果见 docs/04。v4 结构名已从运行时注册表移除：旧 checkpoint 用新代码加载会直接报错，不会被静默重新解释；已有数字仍按下表作为记录。

## v5（Dsup v5，grouped_gated_sparse_slot_v5）——当前方案，代码就绪、GPU 未跑

协议：experiments/Natural-merge-v5-routing-generalization.md。主线结构、监督、预算和评测口径全部不变，只改内部路由与门控，目标是让 16k 上已经出现的增益跨任务、跨域稳定。四处改动：

1. **分段几何门控**：门控输入从「槽位自身隐状态」改为三个标定无关的几何特征——路由亲和度 top-1 logit、top-8 概率锐度 p_max−p_min、与块质心的冗余度 cos(slot, centroid)——在块内 z-score 后经零初始化 `rel_gate(3→1)` 加到 `slot_gate` 上。作用：门控能判断「这个槽位是否真的匹配到了源内容」，不再只依赖域内隐状态的绝对水平。
2. **路由先验混合**：路由 logit = 0.6·cos(原生 4096-D) + 0.4·cos(学习 32-D 投影)，再除以温度 0.04；原生视角为常数项（不反传），学习视角才有梯度。作用：低秩投影在域外会退化，原生表示先验托住路由。
3. **逐层组有符号原生增益**：`native_gain` 从单个标量改为 `num_groups` 个（浅层寻址、深层读出，两类需求不再被同一标量绑定）；原生支路取 top-2 最锐槽位的全宽源均值减载体，按路由置信度缩放。
4. **载体选择层对齐**：`consolidate._SELECT_LAYER` 与 `layout._REFERENCE_LAYER` 从 0 改为 16，与路由层 `reference_layer=16` 一致。作用：选择器、载体、目标方向与路由读取不再跨流形。

保持不变的结构不变量（有 CPU 测试守）：step 0 输出与 Selection 逐位相同；全局残差 RMS 预算 ≤ 0.05；`source_allow_mask` 消融精确退化为 Selection；`state_digest` 稳定；coverage mass 求和为 1。

额外监督项：`protection_anchor_weight=0.10`，对 Full 答案起始词元分布做 KL 锚定（教师为无梯度 Selection 前向），防止保护组在域外把输出拉离原生分布。

因为 `_SELECT_LAYER` 由 0 改为 16，**Selection 臂也必须重跑**，否则与 Dsup 臂不同轴。

**必须先知道的评测事实**：压缩仅在逐样本写前缀 ≥ 10240 tokens（20 block × 512）时触发。8192 输入下前缀上限 6144，`compression_count` 恒为 0，两臂 prediction 逐字相同（已逐题实测，见 docs/04）；16384 下按任务不同，触发比例从 4/200（qasper）、25/200（2wikimqa）到 124/200（hotpotqa）、151/200（narrativeqa）、191/200（musique）不等；两臂在未触发样本上 prediction 逐字相同，所以全集 Δ 完全由融合子集贡献，v4 的 16k 融合子集加权增益是 +3.02（详见 docs/04）。覆盖低的两个任务不是方法上限，是评测覆盖不足。v5 的每次评测都要单独报告融合子集（`compression_count>0`）的分数与样本数。

## 数据与隔离边界

原始 SQuAD 下载入口 tools/stage_natural_merge_data.py；v1 计划准备要求已观察问题集、匹配的公开干扰段落和固定 NaturalQA 计划，不能省略隔离审计。数据证书：reports/natural_merge_implementation_20261013/。
