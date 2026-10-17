# S1c — Geometry Audit: R/P/T 梯度与 13D 子空间的几何关系

**状态**：分析完成，预注册判定 **共向（co-directional）→ 不解耦**（2026-10-08）

## 动机

阶段 1（[S1c-basis-wash.md](S1c-basis-wash.md)）确认 13D 功能子空间的方向承载
功能。但训练只监督 rescue；protect/repair 的零损伤是否只是位移预算的副作用？
若是，预注册解耦方案（shared/private/conflict 分解）值得做；若三个 cohort 的
梯度本来就共向，解耦就是多余的复杂度。

## 设计

对 111 题（R36 + P31 + T44）逐题计算干净梯度行 `g_i = d(CE_i)/d(down.weight)`
（clone-source 修正路径，与阶段 1 corrected basis 同源），投影到修正 13D 子空间：
`z_i = U_L'^T g_i`。分析量：cohort 均值方向的 13D 投影余弦、全空间余弦、投影能量
占比。预注册：`cos(ḡ_R, ḡ_P) > 0.3` → "耦合即共享收益，不解耦"，否则解耦立项。

原草案的 "conflict energy < 10%" 与 cos 定义冲突（conflict 就是 |cos|），按预注册
精神以 co-directionality 为准 operative 判据，conflict 幅度记录在案。

## 结果

13D 投影空间 cohort 均值余弦：

| | rescue | protect | repair |
|---|---|---|---|
| rescue | 1.0 | **0.9498** | 0.9252 |
| protect | 0.9498 | 1.0 | 0.8643 |
| repair | 0.9252 | 0.8643 | 1.0 |

全空间余弦 0.76–0.83；投影能量占比 rescue 0.44 / protect 0.30 / repair 0.41。
逐题拆分（grad/proj/residual 范数）在报告 `per_question` 字段。

## 结论

**cos(g_R, g_P) = 0.9498 ≫ 0.3 阈值 → 共向，不解耦。**

1. 三个 cohort 的梯度在功能子空间内几乎同向：沿 rescue 方向移动同时降低
   protect/repair 的 CE——rescue 监督顺带压低保护损伤是**一阶原理**，
   S1 的 P_harm=0 不是预算的意外副产品。
2. 投影能量占比 ~0.3–0.4：三个 cohort 都有约六成能量在 13D 子空间外，
   但这些正交分量共享同向的主结构，解耦收益空间小。
3. **停止条件②触发：不做 shared/private/conflict 分解，零代码。**

## 工件

- 报告：`runs/s1c_geometry_audit.json`（sha `22b71a0e…`）
- 梯度行缓存：`runs/s1c_geometry_rows.pt`（111×131072 fp32，sha `1183a4bf…`）
- 驱动：`tools/s1c_geometry_audit.py`；测试：`tools/test_s1c_geometry_audit.py`（4 项）
- 溯源：阶段 1 basis `07fa8b78…` → checkpoint `a9fe2ea3…` → 梯度审计 `63059678…`

## 下游

门放行 → 阶段 3 层 1（跨窗口 U_global oracle 验证）继续。
