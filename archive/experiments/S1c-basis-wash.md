# S1c — Basis Wash: 修正图上重建 U_L 的四臂对照

**状态**：formal 完成，门判定 **PASS**（2026-10-07，V100-32GB，torch 2.5.1+cu124）

## 动机

S1 修复（`s1_soft_local.py`）证明：历史驱动的 `.data` 视图别名污染了 13D 子空间臂的**梯度**，
但从未回答别名是否也污染了 **basis 本身**（13 行 Jacobian 是在别名图上算的）。
若 basis 被污染，则"功能子空间"的正结论可能只是伪影。本实验在修复后的图上
以 clone-source 路径重算 13 行 Jacobian 重建 U_L'，用四臂对照直接裁决。

## 设计

单窗口 (row 3, start 17890)，296 探针，与 `s1_soft_local` 完全相同的协议
（AdamW lr 3e-3 wd 0.01、budget 1.0、60 步、每 5 步测、seed 0、identity-at-init）。
四臂唯一差异是 basis：

| 臂 | basis | 可训练自由度 |
|---|---|---|
| corrected_ul | 本实验重算（clone source + 块外 backward） | 13 |
| old_ul | 历史 `s1_local_protected.basis.pt`（`3af25e44…`） | 13 |
| random_13d | `qr(randn)` seed 0 随机正交基 | 13 |
| unconstrained | 无基，全 down.weight 位移 | 131072 |

预注册门（PASS = 前三者全真）：`protect_harmed(corrected) ≤ 1` ∧ `net(corrected) ≥ 8`
∧ `rescue(corrected) − rescue(random) ≥ 5` ∧ `net(corrected) − net(random) ≥ 5`。
FAIL：`net < 8` 或对 random 两差距均 <3。边界 3–5：random 换 seed 1/2 重抽，corrected
须胜最好一抽 ≥3。

## 结果

| 臂 | original | R | T | P_harm | K_harm | net |
|---|---|---|---|---|---|---|
| corrected_ul | 7 | **15** | 38 | 0 | 6 | **+9** |
| old_ul | 7 | 14 | 38 | 0 | 6 | +8 |
| random_13d | 0 | 4 | 0 | 0 | 44 | −40 |
| unconstrained | 10 | 10 | 28 | 1 | 17 | −7 |

门核算：P_harm 0≤1 ✓；net +9≥8 ✓；vs random rescue +11≥5 ✓ net +49≥5 ✓ → **PASS**。

basis 审计：corrected 与 old 主角度全部 ≤0.005 rad，max|cos| ≈ 1.0。
历史 unconstrained 在修复图上复现历史数字（R10/harm17/net−7，逐位一致）。

## 结论

1. **basis 未受别名污染**：别名只通过 backward recompute 读突变 source 起作用，
   而 Jacobian 行的 backward 在 `_pool_in_place` 块外执行，restore 已还原字节；
   主角度 ~0 从几何上证实了这一点。
2. **13D 方向承载功能，不是任意低维都行**：random-13D 彻底崩溃（net −40，
   rescue 只有 4），说明子空间的**方向**是功能性的，排除"低维即有效"的别名解释。
3. corrected (net+9) ≥ old (net+8)：干净梯度对最终结果无副作用，S1 的正结论在
   全链路干净图上成立。

## 工件

- 报告：`runs/s1c_wash.json`（sha `12538514…`）
- basis：`runs/s1c_wash.basis.pt`（corrected/old/random 三基 + 13 行 Jacobian，sha `07fa8b78…`）
- 四臂 checkpoint：`runs/s1c_wash.{corrected_ul,old_ul,random_13d,unconstrained}.pt`
- smoke：`runs/s1c_wash_smoke.json`
- 驱动：`tools/s1c_wash.py`（sha `836d53e6…`）；校验：`tools/verify_s1c_wash.py`
- 测试：`tools/test_s1c_wash.py`、`tools/test_verify_s1c_wash.py`（10 项全绿）
- 门禁链：checkpoint `a9fe2ea3…` → 历史 basis `3af25e44…` → 修复报告 `e707b139…`
  → 梯度审计 `63059678…` → pinned 驱动 `9ec75980…`

复现：`python tools/verify_s1c_wash.py --report runs/s1c_wash.json --basis runs/s1c_wash.basis.pt --require-complete --require-gate PASS`

## 下游

门 PASS → 阶段 2（梯度几何审计，R/P/T 共向性诊断）继续；
停止条件①（corrected 不胜 random）未触发。
