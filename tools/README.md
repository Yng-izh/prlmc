# 工具入口

当前实验按 `experiments/Natural-merge-v2-ABC.md` 执行。下面的自然融合入口保留原文件路径与内容，清理未改变 checkpoint 的源码哈希口径。

## 当前自然融合 A/B/C

| 入口 | 用途 |
| --- | --- |
| `stage_natural_merge_data.py` | 下载固定版本的原始 SQuAD，核对 Git/LFS 与文件哈希 |
| `prepare_natural_merge.py` | 构建文章隔离的 v1 训练与 E1/E2/E3 计划 |
| `verify_natural_merge_data.py` | 原文、QA、写入与数据隔离独立校验 |
| `prepare_natural_merge_v2.py` | 为固定训练 context 恢复全部原始 QA，生成 v2 计划 |
| `run_natural_merge_v2.py` | 调度 A/B/C：smoke → 64-update 训练 → fit → 开发 E1 |
| `natural_merge_v2_train.py` | 固定课程和监督下的共享训练 |
| `natural_merge_v2_training_fit.py` | 固定训练窗口的生成诊断，不作泛化评测 |
| `natural_merge_eval.py` | 冻结 checkpoint 的开发 E1、正式 E1/E2/E3 评测 |
| `summarize_natural_merge_v2.py` | 独立重算三组开发成绩、配对收益与门槛 |
| `verify_natural_merge.py` | 逐题评分、预算、载体、冻结状态与置信区间校验 |
| `freeze_natural_merge.py` | 开发门槛通过后冻结模型、代码和正式评测计划 |
| `run_natural_merge_parallel.py` | 正式评测按完整 context 分片与合并 |
| `natural_merge_runtime.py` | 共享 writer、布局、模块恢复与评分 |
| `test_natural_merge.py` / `test_natural_merge_v2.py` | v1/v2 执行契约的 CPU 测试 |

`natural_merge_train.py` 保留供上一轮自然训练复现及当前测试调用。当前 A/B/C 训练使用 v2 入口。正式评测仍需通过预定冻结门槛。

## 公开记忆保持与模型缓存

- `prepare_public_retention.py`、`public_retention_eval.py`、`run_public_retention_parallel.py`、`verify_public_retention.py`、`test_public_retention.py`：原始 SQuAD/NaturalQA 作者协议和共享读写基础。
- `stage_s1d_reader_cache.py`、`verify_s1d_cache_values.py`：生成并核对共享 FP16 reader 缓存，保留历史命名。
- `native_qa_compare.py`：原始 QA、原文分片和结果写入等公共工具。

## 保留的兼容依赖

部分 `e6a_*`、`s0*`、`s1*`、`readout_probe.py` 和共享 E6d 工具仍被当前公开评测或兼容 checkpoint 的代码引用。它们保持原路径，避免改变现有源码清单或破坏旧 checkpoint 恢复。这些文件的存在不代表要重启旧的人造数据、局部 alpha 或保护实验。

已删除的不再被当前链路引用的扫描、审计、监视和旧启动器，详见 `archive/cleanup-20261014.json`。旧入口的完整运行环境与数据按清理前 Git 提交恢复，不建议混用新代码和旧冻结证书。
