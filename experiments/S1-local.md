# S1-local：Protected Functional Subspace Residual

## 冻结协议

本实验验证模块机制，不验证泛化，也不提供论文主表结果。

固定 S0 的 row 3 / start 17890、19→9 blocks 和 E6d interleave 检查点。原始 60 道 dropped-block 探针仅用于复现 S0 的 13 个 lost；保护探针扩展到同一个已写入窗口内全部可生成的有效事实问答，仍遵守全文 latest-fact、窗口归属、去重、答案长度和官方 QUERY_TEMPLATE 规则。保护集合严格为 `full正确且native select正确`。不将 `full正确但select错误` 冒充 retained，不增加 fullK 分支。

重新计算 lost 和 retained 在同一运行状态、同一 D0 上的逐题 gold-answer CE Jacobian。S0 保存行只用于核实原 lost 问题集合与输入来源，不以集合相同代替运行池一致性证明。

`U_L = orth(J_L^T)`，`U_K = orth(J_K^T)`，`U_safe = orth[(I - U_K U_K^T) U_L]`。小矩阵薄 SVD 使用 CPU float64；绝不构造 131072×131072 矩阵。保护仅是一阶 CE 不变条件，不预先声称有限步 EM 无损。

## 五臂比较

- native select：仅保留原生 carriers，零 residual。
- unconstrained residual：从现有检查点出发，自由更新 down.weight 位移。
- S0b coefficient oracle：复用原 C=I+E 实现，位移上限 0.5。
- subspace residual：D=D0+reshape(U_L alpha)。
- protected subspace residual：D=D0+reshape(U_safe alpha)。

现有检查点原状态另作为 step-0 诊断，而非第六个方法变体。四个训练臂共享 lost CE 目标、AdamW lr=0.003、60 steps、seed=0。三个参数位移臂共享 Frobenius 上限 1.0（对应 S0 扰动 scale=1），所有参数空间臂的其余参数冻结。oracle 的 0.5 是系数空间上限，不能解释为与参数空间 1.0 等价。每 5 步测量，最终第 60 步判门，中间峰值仅供诊断。

## 指标与门槛

主 rescue 分母固定为复现出的 S0 13 个 lost。harm 对全部 `full正确且select正确` 统计，同时单列 D0 原本正确、训练后变错的新增 harm；不混淆部署相对 select 的破坏与新增训练破坏。报告完整逐题正确性、rescue、harm 数/率、净收益与 teacher-forced token-exact accuracy。本地 token-exact accuracy 不称作官方生成 EM。

运行前冻结的保守操作门槛：retained 至少 10 道；protected 最终 rescue≥7；相对 select 的 harm 严格低于重跑的 coefficient oracle 且至多为其一半；harm 不高于 unconstrained 和 U_L；相对 select 的 rescued-minus-harmed 净收益≥0。任何一项未过即 NO-GO，不启动 S1-generalization。该计数门是工程判断，不冒充统计显著性检验。

短程验证与正式实验使用独立输出与新运行池。短程验证的结果标为 SMOKE-ONLY，无方法结论。

## 运行

```bash
/share/latent/mllm_env/bin/python -u tools/s1_local_protected.py \
  --carrier-ckpt runs/e6d_interleave_ckpt.pt \
  --s0-rows runs/s0_jacobian_rows.pt \
  --cuda-device 9 --steps 60 --measure-every 5 \
  --out runs/s1_local_protected.json
```

短程验证额外设置 `--steps 2 --measure-every 1 --smoke` 并使用独立输出名。输入 SHA256 写入 JSON；每次测量原子保存，保留 basis 和各臂最终权重用于复放。

## 当前状态

实现与数值验证完成；首轮 GPU 短程验证在基线阶段主动终止，尚未产生训练结论。独立代码审查发现：原 S0b 外置 oracle 的系数参数未被内部 compactor 的检查点开关扫描到。S1-local 将同一个 oracle 注册到内部 arm，启用逐层 checkpoint，不解冻其他参数、不改变前向公式与实验门槛；新增 CPU 回归测试核实检查点调用与系数梯度。扩展探针复用 `probe_indices` 的实际窗口边界过滤。修正后从独立输出名重新运行短验证。尚无 S1-local 方法结果。

S1-local 过门后才执行：train 窗口建基和 interleave 训练→冻结→held-out 窗口无监督压缩评测。held-out 题的答案与 Jacobian 不参与建基、优化或模型选择。S1-paper 主表仅 native selection / unconstrained residual / subspace residual / protected subspace residual，预算 0.25/0.5/0.75；SIC、semantic router、reader adapter、SVD retuning 仅保留历史诊断。
