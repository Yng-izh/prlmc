# S1d-selfsup-basis — 无答案窗口自监督建基可行性（stage 1，v1+v2）

- **状态**: stage 1 完成，gate PASS（预注册门槛），稳健性双标未全达；stage 2 dev 已完成，见 [S1d-stage2-dev.md](S1d-stage2-dev.md)，当前 INSUFFICIENT（TF 有积极信号，GEN 未形成有效判据）
- **日期**: 2026-10-09
- **问题**: S1c 链证明固定全局 13D 基跨窗口失败（窗口子空间近正交，align 0.256）。S1d-B 问：用**压缩前可观测、无答案标签**的窗口事实构造监督（事实行尾部遮罩补全 CE），其 Jacobian 行 SVD 出的每窗口 13D 基，能否覆盖 gold rescue 梯度子空间明显超过 LOWO 固定全局基（≈0.32）
- **判定**: **gate 通过**：mean gRho 0.5116 ≥ 0.20 且 ratio 1.6226 ≥ 1.5（v2 扩池后）。用户稳健性双标（ratio ≥ 1.65 ∧ 平均有效秩 ≥ 13）未全达：ratio 差 1.7%，有效秩均值 7.41

## 协议（预注册，v2 扩池版本）

- 8 个建基窗口 = s1c TRAIN_WINDOWS（row 3）；held-out (2,3450)(2,3830) 不触碰
- 监督 = 事实行去句号后尾部 MASK_SPAN=4 token 遮罩，reader 在 D0 压缩池（E6d interleave ckpt `a9fe2ea3…`）下做补全 CE，对 `down.weight` 求梯度 → 一行/事实行
- **行池（v2，用户批准的扩池）**：dropped 块全部事实行（移除模板过滤）优先，latest-owner 去重（同 (slot, subject) 取最新序号行，与 `probes_from_lines` 冲突消解语义一致），不足 40 有效行才用 retained 块回填并单独记录比例；短行（<2×MASK_SPAN+2=10 token）跳过继续拉候选
- 基 = 行池 uncentered SVD 前 13 右奇异向量（**v2 修正为 SVD 序**；v1 误用 QR 序，见"缺陷与修正"）
- 靶 = `s1c_global_rows.pt`（sha `73754926…`）各窗口 rescue_d0_wrong gold-CE 行
- gate：mean gRho ≥ 0.20 且 ≥ 1.5 × LOWO r13（LOWO 参照 `s1c_lowo_coverage.json`）
- held-out QA 答案集合从未参与建基；gold 行只作覆盖率诊断靶

## v1 → v2 差异与扩池动机

v1（`runs/s1d_selfsup_basis.json`，driver `c2879f32…`）用模板过滤 + 23–31 行/窗口，gate 擦线通过（ratio 1.5003，超出 0.02%），熵有效秩 3.92–9.58 < 13，max 行间余弦 0.88–0.99——正是用户指出的"13 条高相关行 ≠ 13 个有效方向"。用户批准扩池重跑（多卡并行）：移除模板过滤、latest-owner 去重、有效行拉满 40、辅助回填单独记账。

## 结果

**v2（8/8 窗口各 40 有效行，全部来自 dropped primary 池，aux 回填 = 0）**：

| 窗口 | gRho | uRho(gold上界) | LOWO | align | 有效秩 |
|---|---|---|---|---|---|
| 3,14470 | 0.4817 | 0.9509 | 0.3108 | 0.4392 | 11.56 |
| 3,14850 | 0.5909 | 0.9411 | 0.2393 | 0.4397 | 6.76 |
| 3,15230 | 0.4225 | 0.9722 | 0.2497 | 0.4306 | 7.65 |
| 3,16370 | 0.6960 | 0.9873 | 0.4536 | 0.5601 | 5.56 |
| 3,16750 | 0.5073 | 0.9555 | 0.3692 | 0.4303 | 5.65 |
| 3,17130 | 0.4560 | 0.9261 | 0.3301 | 0.4351 | 8.07 |
| 3,17510 | 0.3700 | 0.8881 | 0.2574 | 0.4626 | 8.42 |
| 3,17890 | 0.5688 | 0.9322 | 0.3126 | 0.4953 | 5.64 |
| **均值** | **0.5116** | **0.944** | **0.3153** | **0.452** | **7.41** |

- ratio vs LOWO = **1.6226**（v1 擦线 1.5003 → 8.5% margin）
- 8/8 窗口 gRho > 各自 LOWO；rank 5→13 覆盖仍单调增长（+0.08–0.15），top-13 携带行池能量 89.8–95.3%
- v1/v2 重叠 203 行，同码重跑最坏 rel diff 1.24e-2（V100 fp16 前向非确定性，cos >0.99999）

**扩池收益分解**：v1 行按 v2 的 SVD 序重算 gRho 均值 ≈0.488 → 0.5116 中扩池贡献 ≈+0.024，排序修正 ≈+0.005。真收益但不大。

**有效秩天花板证据**：40 行时 sigma1/sigma13 = 4.6–9.0，primary 池深 199+ 行/窗口只取了 40；有效秩 6.7→7.41 的微增说明这是**同窗口 dropped 块事实行梯度方向天然聚集**，继续加行大概率无效。gold 行自身有效秩 4.2–14.5（uRho 0.89–0.99 同样低秩），两侧条件数同量级。

## 缺陷与修正

1. **v1 orth() 用 QR 序**：`torch.linalg.qr(rows.t())` 的前 13 列不是 top-13 奇异子空间（v1 缓存实测 uRho 0.59–0.89 QR vs 0.89–0.99 SVD）。v2 改为 `s1_local_protected.orth` 的 SVD 语义（vh 按奇异值排序），与 s1c 链（oracle local_13d、LOWO）一致。v1 JSON 数字按 QR 序作废重读。
2. **v2 部署事故**：一次坏编辑（argparse 变量名混写）使首 launch 崩于参数解析，未耗 GPU 时间。
3. **短行拒绝**：移除模板过滤后行池出现 7–9 token 行，长度门槛拒为 None 触发"None 即报错"；改为跳过短行继续拉候选（每窗口实际跳过 9–22 行）。
4. **v1 一致性校验过严**：位相等改为记录最坏相对偏差（fp16 非确定性）。

## 执行细节

- 双卡分片（GPU9 全空 + GPU0  piggyback 445 MiB 占用、32 GB 卡），`--shard k --num-shards 2`，各 4 窗口，~25 分钟（单卡约 45–50 分钟的一半）
- 产物：`runs/s1d_selfsup_basis_v2.json`（sha `67ccce64…`）、`runs/s1d_selfsup_rows_v2.pt`（`a453b421…`）、shard0/1（`d34cc063…`/`20fbe1fa…`）、驱动 `tools/s1d_selfsup_basis_v2.py`（`83ae8608…`，双端逐位一致）、测试 `tools/test_s1d_selfsup_basis_v2.py` 8/8 双端 PASS
- v1 产物保留不动：`runs/s1d_selfsup_basis.json`、`runs/s1d_selfsup_rows.pt`、`tools/s1d_selfsup_basis.py`（`c2879f32…`）、测试 7/7

## 判读与停线条件（承接 S1d 预注册）

- gate 通过 → stage 2（D_new = D0 + U_window·α，无答案监督优化 α）资格成立
- 稳健性双标未全达的解读：有效秩缺口是数据模态固有属性而非采样不足（证据见上）；若 stage 2 只有 gold oracle 有收益而自监督无收益 → 停止训练 fθ 预测器
