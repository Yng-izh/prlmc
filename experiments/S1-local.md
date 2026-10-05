# S1-local：Protected Functional Subspace Residual

## 冻结协议

本实验验证模块机制，不验证泛化，也不提供论文主表结果。

固定 S0 的 row 3 / start 17890、19→9 blocks 和 E6d interleave 检查点。保留原始 60 道 dropped-block 探针作为 rescue 候选；保护探针扩展到同一个已写入窗口内全部可生成的有效事实问答，仍遵守全文 latest-fact、窗口归属、去重、答案长度和官方 QUERY_TEMPLATE 规则。保护集合严格为 `full正确且native select正确`。不将 `full正确但select错误` 冒充 retained，不增加 fullK 分支。

**结果前执行修正：**实际 reader 配置是 attention_dropout=0，但 LoRA dropout=0.1。原 S0/S0b 的 `model.train()` 同时启用了 LoRA dropout，冻结权重未关闭这一随机性。S1 保留原 seed=0 窗口写入流程，写入结束后将所有 dropout 子模块置为 eval，父级仍 train 以启用梯度检查点。原 60 题按 no-memory→full→checkpoint 重测确定性基线；旧 13 题逐项验证仍为 lost。第四轮短验证确认旧 13 题全部仍丢失，额外新增 `Who founded Tiantai?`。正式比较固定原 13 题为 rescue 目标和主分母，新增题只列诊断，不用于达到 rescue≥7；若原 13 题中有任何一题已不再 lost，停在基线审计而非扩大或替换主分母。短验证当前 14 题只用于完整路径压力检查，不作方法结论。原始题单在扩展评分中重复测量，一致性未过则停在基线阶段。rescue≥7 和所有 harm 门槛保持不变；这些执行修正发生于正式五臂训练结果之前。

重新计算本轮 lost 和 retained 在同一运行状态、同一 D0 上的逐题 gold-answer CE Jacobian。S0 保存行仅保留历史来源，不用于本轮建基。

`U_L = orth(J_L^T)`，`U_K = orth(J_K^T)`，`U_safe = orth[(I - U_K U_K^T) U_L]`。小矩阵薄 SVD 使用 CPU float64；绝不构造 131072×131072 矩阵。保护仅是一阶 CE 不变条件，不预先声称有限步 EM 无损。

## 五臂比较

- native select：仅保留原生 carriers，零 residual。
- unconstrained residual：从现有检查点出发，自由更新 down.weight 位移。
- S0b coefficient oracle：复用原 C=I+E 实现，位移上限 0.5。
- subspace residual：D=D0+reshape(U_L alpha)。
- protected subspace residual：D=D0+reshape(U_safe alpha)。

现有检查点原状态另作为 step-0 诊断，而非第六个方法变体。四个训练臂共享 lost CE 目标、AdamW lr=0.003、60 steps、seed=0。三个参数位移臂共享 Frobenius 上限 1.0（对应 S0 扰动 scale=1），所有参数空间臂的其余参数冻结。oracle 的 0.5 是系数空间上限，不能解释为与参数空间 1.0 等价。每 5 步测量，最终第 60 步判门，中间峰值仅供诊断。

## 指标与门槛

主 rescue 分母固定为原 S0 13 题，并逐项验证它们在本轮确定性基线仍为 lost；额外确定性 lost 只作诊断。harm 对全部 `full正确且select正确` 统计，同时单列保护集合内 D0 原本正确、训练后变错的新增 harm，并额外报告全部 296 题中 D0 原本正确、训练后变错的数量和逐题列表；不混淆部署相对 select 的破坏与新增训练破坏。报告完整逐题正确性、rescue、harm 数/率、净收益与 teacher-forced token-exact accuracy。本地 token-exact accuracy 不称作官方生成 EM。

运行前冻结的保守操作门槛：retained 至少 10 道；protected 最终 rescue≥7；相对 select 的 harm 严格低于重跑的 coefficient oracle 且至多为其一半；harm 不高于 unconstrained 和 U_L；相对 select 的 rescued-minus-harmed 净收益≥0。任何一项未过即 NO-GO，不启动 S1-generalization。该计数门是工程判断，不冒充统计显著性检验。

短程验证与正式实验使用独立输出与新运行池。短程验证的结果标为 SMOKE-ONLY，无方法结论。

## 确定性基线审计（非方法结果）

第四轮短验证：原始题单 60，扩展题单 296；确定性 lost 总数 14，原 S0 13 题全部仍 lost，仅新增 `Who founded Tiantai?`。保护集合 full∧native select 正确共 75 题，而现有 D0 检查点只保住其中 31 题，基线相对 select 已损失 44 题。后续同时报告相对 select 的总破坏和相对 D0 的新增破坏，避免把既有错误归因给更新，也避免用“新增 harm 少”掩盖系统仍低于 native select。

这批数字仅是基线，尚无正式 60 步方法结论。快照见 `runs/s1_local_smoke_v4_snapshot.json`，实际 reader 配置见 `runs/s1_reader_config.json`。旧 13 个 rescue 目标与 75 个 select-retained 中有 7 题重叠：lost 指 D0 丢失，retained 指 select 保住，两套参照的含义不同；不通过偷偷删保护题或改分母消除该事实。

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

实现与数值验证完成；首轮 GPU 短程验证在基线阶段主动终止，尚未产生训练结论。独立代码审查发现：原 S0b 外置 oracle 的系数参数未被内部 compactor 的检查点开关扫描到。S1-local 将同一个 oracle 注册到内部 arm，启用逐层 checkpoint，不解冻其他参数、不改变前向公式与实验门槛；新增 CPU 回归测试核实检查点调用与系数梯度。扩展探针复用 `probe_indices` 的实际窗口边界过滤。修正后的第二轮短验证完成窗口构建与 296 题基线评分，但在 S0 lost 集合一致性断言处退出，尚未建基或训练。CPU 检查证实：原 60 题与扩展集合中同名题的 gold 差异为 0；原 `/tmp` 输入与持久备份 SHA256 完全一致。失配原因尚待定位，该现象不构成方法失败结论，也不使用旧 rows 强行继续。

第三轮在原始基线审计阶段主动停止：实际配置与 reader 源码确认 PEFT 两个 adapter 使用 LoRA dropout=0.1，原训练态评分存在随机性。改为窗口写入后仅关闭 dropout 子模块，再以确定性评分、Jacobian、训练和重复评分一致性检查继续；父模型训练态及梯度检查点保留，所有 reader 权重冻结。7 项测试通过，包含 dropout 关闭后父级仍 train、前向可重复且梯度有效的回归测试。第四轮确定性短验证完成全部五臂路径，所有训练臂零更新正确性与 D0 一致、梯度有限、权重检查点成功保存，结论标为 `SMOKE-ONLY`。压力检查采用 14 题；basis ranks=14/75/7，保护正交误差 6.31e-15。两步分数不用于方法判门。正式驱动已锁定原 13 题，主分母契约与完整回归测试共 9 项全部通过；正式 60 步运行待启动。

S1-local 过门后才执行：train 窗口建基和 interleave 训练→冻结→held-out 窗口无监督压缩评测。held-out 题的答案与 Jacobian 不参与建基、优化或模型选择。S1-paper 主表仅 native selection / unconstrained residual / subspace residual / protected subspace residual，预算 0.25/0.5/0.75；SIC、semantic router、reader adapter、SVD retuning 仅保留历史诊断。
