# 02 — 方法探索线（S0/S0b、S1、S1c、S1d、G0–G2c）

**结局：全部 NO-GO / FAIL / 停线。** S1c 跨窗口裁决 FAIL 后，13D 功能子空间主线停线：单窗口内（S1c-wash）corrected 13D 臂 net+9、random 崩溃 −40，方向承载功能成立；但 8 训练窗口 240 行 rescue 梯度 SVD 得到的 U_global（energy 0.6425）在 2 个 held-out 窗口四臂裁决中：global 25/21/+4、random 25/32/−7、unconstrained 29/18/+11、local 28/20/+8——rescue_gap=0 且 net 低于 unconstrained−2，预注册门 **FAIL**。单窗口正结论依赖窗口自身监督，不携带跨窗口结构；层 2（冻结 U_global + 残差预测器）的前提被否定，取消立项。S1-local hard-nullspace NO-GO、S1-soft-local NO-IMPROVEMENT 维持原判。

本线索引到的协议正文（约 17 篇）已全部归档，路径见下表；正文内容未改写。

## 逐篇索引（archive/2026H1_legacy/experiments/ 下）

| 文档 | 结论/状态 |
|---|---|
| S1-soft-local.md | 固定 13 维 UL 的单边有限 CE 保护；teacher-forced 口径；正文含冻结 basis SHA256 与训练监督定义 |
| S1c-global-oracle.md | 跨窗口 U_global 13D 约束 held-out 裁决：formal 完成，预注册门 **FAIL** → 停线（2026-10-09） |
| S1c-lowo-coverage.md | LOWO 梯度能量覆盖率审计：**窗口差异主导，非秩不足** |
| S1d-two-lines-v1.md | 方法线 + 生成应用线并行协议（2026-10-12） |
| S1d-two-lines-v2-results.md | 四卡两线最终汇总：M1 net+1 未过、M2 net+9 过方法门；生成线 A1/A2 C−Selection +9/+3；方法门槛未全部通过 |
| S1d-C-dimensions-v1.md | C 维度消融 20D/24D/28D |
| S1d-C-risk-protection-v1.md | C 组风险感知保护固定方案（原 C 弱窗口 R=6/H=5/net+1；强窗口 R=10/H=1/net+9） |
| S1d-controlled-C13-development.md | 冻结 C13 独立受控开发 episode 生成评测 |
| S1d-experiment1-fact-v3.md | 实验1 受损事实建基交叉对照：弱窗口最佳净收益 +1，低于 +5 门槛，不进入实验2 |
| S1d-experiment1-fact-cross.md / S1d-experiment1-v3-fleet.md | 实验1 实现与并行运行记录 |
| S1d-native-write-drop-development.md | 原生写入/淘汰策略受控开发（16 测试 episode 封存不评测） |
| S1d-projected-C-review.md | C+RiskProtect 后 projected-C 实现审查 |
| S1d-shared-E6d-baselines-training.md | 共享 CarrierResidualCompactor 候选：独立共享训练与冻结后验证 |
| S1d-shared-compactor-zero-training.md | 零训练跨 episode 评测（第一步：加载 E6d interleave checkpoint） |

## 与当前主线的衔接

S0（功能敏感子空间 PASS）、S0b（INCONCLUSIVE）、S1c-wash（单窗口 PASS）确认了「方向承载功能」这一前提，但跨窗口泛化失败。共享 CarrierResidualCompactor（S1d-shared-* 一支）成为后续自然融合与 public retention 主线的模块基础（见 03/05）。人工 24 episode 与 45.31% 结果仅作附录机制诊断，不进入论文主表。

## 边界与证据

仅两个已见开发窗口、单 seed；生成问题为事实模板构造，按官方 EM 规则计分，不是原生官方题集成绩；16 个测试 episode 封存，不得依据应用成绩扫描 prompt、λ 或筛选测试子集。证据：runs/、outputs/、reports/s1d_round_20261013/。S0/S0b/S1-local/G 系列协议正文缺失（见 01 的正文缺失声明），证据见 runs/ 与 outputs/。
