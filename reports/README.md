# 实验报告索引

## 当前主线

| 目录 | 内容与证据范围 |
| --- | --- |
| [natural_merge_v2_20261014](natural_merge_v2_20261014/status.md) | A/B/C 固定自然训练计划、启动、OOM 失败与修复后 smoke 记录；最终生成结果另行归档 |
| [natural_merge_initial_20261013](natural_merge_initial_20261013/results.md) | 上一轮自然共享训练与开发 E1 原始预测、checkpoint、门槛失败裁决 |
| [natural_merge_implementation_20261013](natural_merge_implementation_20261013/data_suite.json) | 自然 E1/E2/E3 suite 哈希、准备与原始数据校验证书 |
| [public_retention_setup_20261013](public_retention_setup_20261013/preparation.json) | 公开原始数据与作者协议的准备记录 |
| [public_retention_squad_interim_20261013](public_retention_squad_interim_20261013/analysis_snapshot.json) | 已观察 SQuAD author512 的中间分析快照，不是新正式测试 |

当前任务目录在服务器 `runs/natural_merge_v2_abc_20261014_023355_r2/`。这里的启动快照不会自动随服务器进度更新；评估是否跑完应检查实际 launcher 和最终逐题报告。

## 历史诊断

| 目录 | 内容与限制 |
| --- | --- |
| [s1d_round_20261013](s1d_round_20261013/summary.json) | 开发窗口 C13、保护、扩维与生成结果；同一 context 的方法开发证据 |
| [s1d_controlled_dev_20261013](s1d_controlled_dev_20261013/summary.json) | 人造受控开发 episode；不代表公开自然任务或跨任务泛化 |
| [s1d_exp1_fact_v3](s1d_exp1_fact_v3/) | 早期事实目标实验原始归档 |

历史原始 JSON、日志、源码快照和 SHA256 清单原样保留，包括负面结果和执行失败。旧报告内部的路径、文档链接及“未提交”等状态描述是当时快照，未随仓库整理重写；历史文档现集中于 [archive/experiments](../archive/experiments/)，旧执行产物可在清理前 Git 提交找到。

论文主要结果应来自冻结模块在公开原始数据、相同有效预算和预定评分协议上的正式评测。训练内 fit、人造 episode、已观察开发集与 smoke 都需单独标注。
