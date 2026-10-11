# S1d stage 2 protected oracle

- **状态**：代码与协议已冻结；GPU 运行在原实验节点执行，结果以独立校验器确认后更新。
- **用途**：开发阶段诊断对照，不参与论文方法结果，不作为 held-out 或泛化证据。
- **窗口与基**：固定 `(3,17510)`；沿用 `runs/s1d_selfsup_rows_v2.pt` 的 per-window selfsup SVD-13 basis；不使用 QA 重建 basis。
- **可训练参数**：仅 13 个 `alpha`；reader、carrier、basis 冻结；`D_new = D0 + reshape(U_selfsup @ alpha)`。
- **恢复监督**：`rescue_d0_wrong` 的真实 QA 第一答案别名 answer-token CE，prompt token 全部 mask。
- **保护监督**：所有 D0 原本答对的题。先用 D0 计算固定参考 CE `ce_d0(q)`，每步加入
  `mean_q ReLU(ce_current(q) - ce_d0(q))`。
- **固定权重**：`lambda_protect = 1.0`，只做一次诊断，不扫描权重。
- **优化**：AdamW，lr=0.003，weight decay=0.01，60 步，每 10 步测量，位移预算 1.0，seed=0。
- **判读**：仍以 TF `new_recovery / d0_harmed / net_vs_d0` 为主；GEN 只记录，不作为决定性指标。与原 oracle 的唯一实验变量是保护 CE 惩罚。

## 运行状态

当前工作站没有历史 `/share/latent/mllm_env` 且没有 CUDA；驱动已同步到原 `v100-2` 节点的 `/share/latent/PRLM-C`，使用 GPU1 执行。报告生成后运行：

```bash
python3 tools/test_s1d_stage2_dev.py
python3 tools/verify_s1d_stage2_oracle.py
python3 tools/verify_s1d_stage2_oracle_protected.py
```

目标报告：`runs/s1d_stage2_oracle_protected.w17510.json`。

## 显存修复与重跑

32GB V100 上原训练路径在第一次 backward 达到约 31.6GiB 后 OOM；未发现该卡被其他 GPU 进程抢占。保留图的 QA 路径改为逐题重建候选池与梯度累加，并使用 `torch.autograd.graph.save_on_cpu(pin_memory=True)` 卸载保存张量。候选 memory 与 active mask 上下文覆盖完整 backward；保护权重、cohort 均值损失、优化器与 13D basis 保持原定义。

本次验证恢复 `qa_max_tokens=256`，cohort 为 rescue=43、rescue_d0_wrong=38、全部 D0-correct protection=79（初始日志的 protect=60 仅指 retained 保护子集）。128-token 的失败尝试不作为可比较结果。单步 smoke 已完成全部 117 道训练题的 backward，报告 status=complete，loss=3.7742887547，alpha norm=0.0108166523，TF recovery/harm/net=0/0/0；full/select/D0 基线与原 oracle 逐项一致，basis 未变。正式实验执行 60 步，GEN 使用 `--skip-generation` 跳过，最终仅判读 TF 指标；单步结果不是最终实验结论。

上述 smoke 证明显存卸载可完成完整训练步。后续独立代码复查发现 QA 内外层重复使用非可重入的 memory 上下文；正式运行在 step 2 早期停止并归档，去除内层重复绑定后从 step 1 重跑。新增直接调用真实 `train_answer_arm` 的小模型回归，验证多题固定权重均值梯度、hinge 开关、候选池梯度路由、checkpoint 重算读 memory、basis 保持与异常恢复，测试 `PASS 10/10`。早期中断报告不作为最终实验结果。

## 解释边界

保护项若降低 harm，说明此前损伤确实可由目标函数中的 no-harm 约束部分缓解；只有在 recovery 仍保持且 net 改善时，才能支持“目标错配是主要瓶颈”的解释。若保护项仅牺牲 recovery 而没有正 net，说明固定 13D 基内的恢复/保持权衡仍然是主要限制。无论结果如何，这仍是单窗口、固定基、有限步诊断，不构成方法主结果或数学上界。
