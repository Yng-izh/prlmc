# S1d stage 2 dev — 无答案残差训练结果

- **状态**：开发窗口完成；当前结论为 **INSUFFICIENT，TF 有积极信号**。
- **日期**：2026-10-10
- **目标**：验证每窗口冻结的自监督 13D 基，在不使用 QA 答案梯度的条件下优化 `D_new = D0 + reshape(U_window @ alpha)`，是否能恢复压缩损失的功能。
- **窗口**：`(3,16370)`、`(3,17510)`；按 stage-1 几何表现选择，不按 QA 结果选择。
- **训练**：事实行末 4 token 自回归 CE；60 steps；AdamW；lr=0.003；weight decay=0.01；位移范数上限 1.0；每 10 步 TF，末步 GEN。
- **评分纪律**：`new_recovery` 只统计 `rescue_d0_wrong = full ∧ ¬selection ∧ ¬D0` 中训练后答对的题；`d0_harmed` 统计所有 D0 原本答对而训练后答错的题；`net = recovery - harmed`。因此 net 不是全题正确数差。

## 结果

| 窗口 | arm | new recovery | D0 harmed | net | 最终位移范数 |
|---|---|---:|---:|---:|---:|
| 3,16370 | selfsup-13d | 8 | 1 | **+7** | 0.374 |
| 3,16370 | random-13d | 0 | 0 | 0 | 0.653 |
| 3,16370 | unconstrained | 10 | 3 | **+7** | 1.000 |
| 3,17510 | selfsup-13d | 5 | 5 | 0 | 0.271 |
| 3,17510 | random-13d | 0 | 1 | -1 | 0.686 |
| 3,17510 | unconstrained | 4 | 7 | -3 | 1.000 |
| **合计** | **selfsup-13d** | **13** | **6** | **+7** | - |
| **合计** | **random-13d** | **0** | **1** | **-1** | - |
| **合计** | **unconstrained** | **14** | **10** | **+4** | - |

解释：selfsup-13d 在两个开发窗口都优于 random-13d；在 `(3,16370)` 以较小位移取得与 unconstrained 相同的净增益，在 `(3,17510)` 恢复与损伤相抵，跨窗口稳定性不足。两窗口仍不足以支持泛化结论。

## 评测状态修复

首轮运行因 `identity-at-init` 翻转停止。诊断证明 generation 评测会污染后续 reader 评测；模型配置还包含 `lora_dropout=0.1`，旧版 `model.train()` 会重新激活此前关闭的 LoRA dropout。修复后的 reader evaluation：

- 逐模块保存并恢复 `training` 状态；
- 保存并恢复 CPU/CUDA RNG 状态；
- generation 前后检查 active mask 和完整 memory；若 generation 写 memory，退出时恢复完整快照；
- TF reader 显式 `use_cache=False`；generation 显式传 `attention_mask`；
- 保留严格参数级与功能级 identity-at-init 断言。

修复后四个切片均通过 generation-sandwich 和 identity-at-init，均完成 60 steps。独立校验脚本 `tools/verify_s1d_stage2_dev.py` 对四份 JSON 做逐题指标、驱动 hash、协议、测量点、位移预算和跨切片基线一致性检查。

## GEN 结果边界

所有 GEN 整段精确匹配为 0，当前不可作为功能结论。原文显示模型常在答案后继续输出解释、下一条问答或重复文本；例如生成 `Sigmund Freud` 后继续输出 explanation。因此 GEN 的整段 EM 与 TF answer-token exact 不是同一指标，当前应将 GEN 结果标为 **不可判读**，不能将其解读为“没有恢复”。正式 held-out 前需要单独冻结生成答案抽取/评分协议。

## 产物

- `runs/s1d_stage2_dev.w16370.sr.json`
- `runs/s1d_stage2_dev.w16370.u.json`
- `runs/s1d_stage2_dev.w17510.sr.json`
- `runs/s1d_stage2_dev.w17510.u.json`
- `tools/s1d_stage2_dev.py`
- `tools/test_s1d_stage2_dev.py`
- `tools/verify_s1d_stage2_dev.py`

四份报告使用同一修复后驱动 SHA：`39bbbcc8e4657c00c04dc1d683a929e6b4c533f7d03b6ebf5ae9859aa80b3c5e`。依赖 checkpoint、self-supervised rows、gold rows 的 SHA 已写入各 JSON 报告。
