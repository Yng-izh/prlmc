# 工具入口

当前最新方案按 `experiments/Natural-merge-v4-D-supervision-native.md` 执行；v4 已完成 Dsup/Dnative 开发 E1，v3/v2 的协议和归档证据保留。

## 已完成的 v4 Dsup/Dnative

| 入口 | 用途 |
| --- | --- |
| `prepare_natural_merge_v4.py` | 封存实际 EOS、前导空格答案边界和全部原始 QA 覆盖的 v4 计划 |
| `natural_merge_v4_objective.py` | 内容 CE、Full 内容 KL、载体保护和独立 EOS CE |
| `natural_merge_v4_train.py` | Dsup/Dnative 共用的 fresh 64-update 训练 |
| `natural_merge_v4_training_fit.py` | 固定训练窗口完整生成诊断 |
| `run_natural_merge_v4.py` | Dsup/Dnative 双卡队列，到开发 E1 为止 |
| `summarize_natural_merge_v4.py` | Dsup/Dnative、Selection 对照及预设开发门槛 |
| `test_natural_merge_v4.py` | v4 监督、EOS、梯度桥、native skip、预算和 checkpoint 契约测试 |

v4 实际运行结果保存在服务器 `runs/natural_merge_v4_dd_20261015_0108/`。Dsup 通过开发门槛；Dnative 的全维直通支路未改善结果，暂不进入正式评测。

## 已完成的 v3 C/D

| 入口 | 用途 |
| --- | --- |
| `prepare_natural_merge_v3.py` | 从固定 v2 训练计划重新编码原始答案，封存 C/D 共同协议 |
| `natural_merge_v3_objective.py` | 非载体恢复、载体保护、双学习率与循环原生来源跟踪 |
| `natural_merge_v3_train.py` | C/D 共用的 fresh 64-update 训练 |
| `natural_merge_v3_training_fit.py` | 固定训练窗口完整生成诊断 |
| `run_natural_merge_v3.py` | C/D 单卡排队或双卡并行，到开发 E1 为止 |
| `summarize_natural_merge_v3.py` | D−C、Selection 对照、压缩损失恢复及原定门槛 |
| `test_natural_merge_v3.py` | 预算、梯度桥、源屏蔽、循环分组及计划一致性 CPU 测试 |

新 architecture 已接入 `natural_merge_runtime.py`，冻结评测复用 `natural_merge_eval.py`。v3 不修改旧计划、结果或 source snapshot；新增源码会改变当前 manifest，旧冻结证书需配合原归档源码使用。

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
