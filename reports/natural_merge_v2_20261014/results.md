# 自然融合 v2 A/B/C：最终开发裁决

2026-10-14，v100-1。A/B/C 均完成全新 64-update 训练、固定训练窗口生成诊断和开发 E1；最终队列为 complete。三组未通过预定开发推进门槛，没有启动正式 E1/E2/E3。

## 开发 E1

使用同一个已观察的开发写入流：19 个原始 SQuAD context、全部 93 道原始 QA，测试时共享模块和 reader 冻结。表内主指标为作者 case-sensitive 答案包含率，不是 normalized EM。

| 方法 | 正确 / 93 | 作者答案包含率 |
| --- | ---: | ---: |
| Full | 35 | 37.63% |
| Selection | 26 | 27.96% |
| A：块级融合 + 原 CE | 27 | 29.03% |
| B：块级融合 + 加权 CE / Full KL | 27 | 29.03% |
| C：槽位融合 + 与 B 相同监督 | 27 | 29.03% |

A/B/C 的 SharedMerge 逐题正误相同，均相对 Selection 新增 `5726449f1125e71900ae1928`，损伤 0，净增 1，增益 1.08 个百分点。context bootstrap 95% CI 为 [0, 3.125] 个百分点；B−A、C−B 的正误净差均为 0。逐题正误相同不代表生成 token 或模块参数完全相同。

Full-correct / Selection-wrong 集合共 9 题，三组恢复均为 0。新增题不属于这个压缩损失恢复集合，因此不能将它作为成功恢复被压缩丢弃知识的证据。

| 组别 | 完整融合正确数 | 屏蔽融合正确数 | 非载体 44 题上的完整−屏蔽净差 | 开发门槛 |
| --- | ---: | ---: | ---: | --- |
| A | 27 | 27 | 0 | 未通过 |
| B | 27 | 26 | 0 | 未通过 |
| C | 27 | 26 | 0 | 未通过 |

B/C 在全体问题上的完整−屏蔽差异来自载体来源题，不满足非载体知识融合门槛。Full 的最低可读性通过，Shared−Selection 为正，但非载体净差未严格为正，所以三组都不能进入正式评测。评分与门槛沿用预定方案，没有放宽。

## 训练内生成诊断

固定第一条训练流的前 19 个 context，全部 88 道原始 QA：Full 47/88（53.41%），Selection 28/88（31.82%），A/B/C 均 29/88（32.95%）。三组均只相对 Selection 新增同一道题，非载体完整−屏蔽净差仍为 0。这是训练内拟合诊断，不作为独立泛化或正式门槛结果。

## 执行与证据

三组使用同一 312-context、1402-QA 训练计划，reader、数据、rank32、19→9 预算、64 次更新、生成和评分协议固定。A/B 同结构，B/C 同监督；三组使用最后 checkpoint，没有挑选中间模型。训练约 122–125 分钟，训练内生成约 32–33 分钟，开发 E1 各约 34.4 分钟，三组并发。

首次 smoke 的 reader backward OOM 发生在参数更新前，失败证据保留。重跑使用相同的 source CPU 暂存/VJP 修复，三组 smoke 成功，正式队列每组 smoke/train/fit/eval 均退出 0。执行成功与方法验收失败分别报告。

- `completed_run/comparison.json`：服务器最终汇总，原样保留。
- `completed_run/A|B|C.e1_development.json`：完整开发原文标识、问题、答案、生成 token、预测、评分、预算与冻结审计。
- `completed_run/A|B|C.training_fit.json`：完整训练内生成；明确标为 train diagnosis。
- `completed_run/A|B|C.train.json`、`.train.log`、`.train.pt`：64 次训练轨迹、日志和最终共享模块 checkpoint。
- `completed_run/launcher.json`：完整 PID、GPU、启动命令与阶段退出码。
- `development_plan.json`、`train_plan.json`：实际固定评测与训练计划。
- `source_snapshot/`：服务器运行时源码快照，作为来源证据，不是另一个实验入口。
- `initial_smoke_failure/`、根目录 `*.initial_smoke_failure.json`：首次失败记录。
- `REMOTE_ARTIFACT_MANIFEST.json`、`SHA256SUMS.json`、`validation.json`：远程路径、归档文件哈希及本地重算结果。
- `generation_results.csv`、`training_results.csv`：便于浏览的最终数据表。

早期 `status.md` 和根目录 `launcher.json` 是启动时快照，保持原样；最新状态以本文件和 `completed_run/launcher.json` 为准。原始文件中的绝对路径与哈希不因归档而改写。

## 结论与边界

本轮新监督与槽位融合结构在这一个开发场景上没有改善主要生成结果，没有证明可读的非载体知识转移。正式自然任务稳定收益和跨任务迁移仍未得到验证。该开发集已经参与方法开发，93 道题来自同一个写入场景；不能将题目数量当作 93 个独立实验。

结果说明本次固定训练配置未达目标，不足以认定所有融合架构或自然任务均无效。后续研究应依据这个负面结果另立方案，不能通过放宽门槛或将训练内成绩替代正式结果来推进。
