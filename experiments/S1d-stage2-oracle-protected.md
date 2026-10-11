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

## 解释边界

保护项若降低 harm，说明此前损伤确实可由目标函数中的 no-harm 约束部分缓解；只有在 recovery 仍保持且 net 改善时，才能支持“目标错配是主要瓶颈”的解释。若保护项仅牺牲 recovery 而没有正 net，说明固定 13D 基内的恢复/保持权衡仍然是主要限制。无论结果如何，这仍是单窗口、固定基、有限步诊断，不构成方法主结果或数学上界。
