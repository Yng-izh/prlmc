# S1c — Global Oracle: 跨窗口 U_global 13D 约束的 held-out 裁决

**状态**：formal 完成，预注册门 **FAIL** → 停线（2026-10-09，V100-32GB，torch 2.5.1+cu124）

## 动机

阶段 1（[S1c-basis-wash.md](S1c-basis-wash.md)）证明 13D 功能子空间在**单窗口内**
方向承载功能（corrected net+9 vs random −40）。但单窗口 U_L 是用该窗口自己的
rescue 监督建的——它可能只是"该窗口答案的编码器"，不携带跨窗口可迁移结构。
阶段 3 层 1 的问题：**跨 8 个训练窗口汇总的 rescue 梯度主方向，在从未见过的
held-out 窗口上是否仍优于随机方向？**

若是，说明存在窗口无关的全局功能方向，层 2（冻结 U_global + 残差预测器 α）有
立项基础；若否，13D 约束在跨窗口设定下不成立，主方法停线。

## 设计

**U_global 构建**：8 个训练窗口（同 S1c 家族的 `(3, 17890/17130/16750/14470/17510/14850/16370/15230)`）
逐窗对 rescue 题（full ∧ ¬select ∧ ¬d0，与 wash 同定义）算 clone-source 干净
Jacobian 行，合计 240 行，拼接 SVD 取前 13 右奇异向量 → `U_global` (131072×13)，
energy_fraction 0.6425。

**held-out 裁决**：两个从未参与建基的窗口 `(2,3450)` 与 `(2,3830)`，各跑四臂
（AdamW lr 3e-3 wd 0.01、budget 1.0、60 步、每 5 步测、seed 0、identity-at-init）：

| 臂 | basis | 可训练自由度 |
|---|---|---|
| global_13d | U_global（训练窗口汇总 SVD） | 13 |
| random_13d | `qr(randn)` seed 0 | 13 |
| unconstrained | 无基 | 131072 |
| local_13d | 该 held-out 窗口自己的 rescue 行 SVD（参照上限） | 13 |

α 优化可用金答案（oracle diagnostic，报告 `oracle_alpha: true`，
`diagnostic_only: true`）——本实验只裁决"方向是否有信息"，不裁决可学习性。

**预注册门**（PASS 全真）：`net ≥ +3` ∧ `rescue(global) − rescue(random) ≥ 3`
∧ `net(global) − net(random) ≥ 3` ∧ `net(global) ≥ net(unc)`。
FAIL：`net < 0` ∨ 两 gap 均 ≤0 ∨ `net < net(unc) − 2`。边界 → 扩展次级
held-out `(1,1930)/(1,1550)`。

## 结果

两窗口汇总（oracle α，各臂 step 60）：

| 臂 | rescue | K_harm | net |
|---|---|---|---|
| global_13d | 25 | 21 | **+4** |
| random_13d | 25 | 32 | −7 |
| unconstrained | 29 | 18 | **+11** |
| local_13d | 28 | 20 | +8 |

分窗口：

| 窗口 | 臂 | R | harm | net |
|---|---|---|---|---|
| (2,3450) | global_13d | 8 | 5 | +3 |
| | random_13d | 6 | 20 | −14 |
| | unconstrained | 10 | 6 | +4 |
| | local_13d | 8 | 6 | +2 |
| (2,3830) | global_13d | 17 | 16 | +1 |
| | random_13d | 19 | 12 | +7 |
| | unconstrained | 19 | 12 | +7 |
| | local_13d | 20 | 14 | +6 |

门核算：net=4≥3 ✓；但 rescue_gap=25−25=**0**（<3 ✗）、net_gap=11≥3 ✓、
net(4) ≥ net_unc(11) ✗ → **FAIL**（两条独立失败：rescue_gap 与 unconstrained 对比）。

窗口 (2,3830) 是关键证据：该窗口 random_13d 反常强（R=19，flat 至 step 60），
global_13d 的 R=17 甚至低于 random——U_global 方向在该窗口没有区分度。

## 结论

1. **跨窗口迁移不成立**：8 窗口汇总的 rescue 梯度主方向（energy 0.6425）在
   held-out 窗口上与随机 13D 无 rescue 区分度（gap 0）。单窗口的正结论
   （wash：rescue +11）依赖窗口自身监督，不携带跨窗口结构。
2. **13D 约束是净损失**：unconstrained（net+11）与 local_13d（+8）均高于
   global_13d（+4）。跨窗口设定下，把位移限制在任何预建低维基上都劣于
   不限制或窗口内自建。
3. **停止条件③触发：主方法停线**。secondary 扩展（FAIL 非 MARGIN 不必跑）、
   层 2（冻结 U_global + f_θ 残差）取消——其前提"U_global 是可冻结的有效基"
   已被本门否定。

## 方法学注记

- 窗口 (2,3450) 四臂数据来自 reader dropout 修复**前**的运行（该窗口
  identity-at-init 恰好通过，数据有效）；窗口 (2,3830) 为修复后运行。两窗口
  协议一致，dropout 只影响评估稳定性不影响训练监督。修复本身见驱动
  `disable_reader_dropout` 调用（wash 驱动 178 行同款）。
- resume 机制验证可用：`(2,3450)` 经 resume 恢复进最终报告，报告记录
  `resumed_from_driver_sha256`。

## 工件

- 报告：`runs/s1c_global_oracle.json`（sha `a5057236…`）
- U_global：`runs/s1c_global_oracle.uglobal.pt`（sha `2aee6f4f…`）
- 240 行缓存：`runs/s1c_global_rows.pt`（sha `73754926…`，远端留存不入库）
- 日志：`runs/s1c_global_oracle.log`
- 驱动：`tools/s1c_global_oracle.py`（sha `7cdb5d0f…`）；测试：
  `tools/test_s1c_global_oracle.py`（11 项）；校验：
  `tools/verify_s1c_global_oracle.py`（51 项全 PASS，含哈希链、臂完整性、
  门算术重算、U 正交 5.6e-16、energy 0.6425 复算）
- 溯源：wash basis `07fa8b78…` → wash report `12538514…` → checkpoint
  `a9fe2ea3…` → 梯度审计 `63059678…`

## 下游

无。预注册停止条件③触发，S1c 主线（13D 功能子空间）在跨窗口设定下停线；
S1-soft-local 单窗口结论保持原判。
