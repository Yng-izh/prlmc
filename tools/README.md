# 工具入口

当前主线为**自然融合 v4（Dsup）训练链**与 **LongBench PRLMC v4 评测链**。

## v4 训练链

| 入口 | 用途 |
| --- | --- |
| `stage_natural_merge_data.py` | 下载固定版本的原始 SQuAD，核对 Git/LFS 与文件哈希 |
| `prepare_natural_merge.py` | 构建文章隔离的 v1 训练与评测计划 |
| `prepare_natural_merge_v4.py` | v4 计划：重编码原始答案，封存 Dsup 协议 |
| `natural_merge_v4_objective.py` | 非载体恢复、载体保护、双学习率与原生来源跟踪 |
| `natural_merge_v4_train.py` | Dsup fresh 64-update 训练 |
| `natural_merge_v4_training_fit.py` | 固定训练窗口完整生成诊断 |
| `run_natural_merge_v4.py` | 单卡/多卡调度：smoke → 训练 → fit → 开发 E1 |
| `summarize_natural_merge_v4.py` | 成绩、配对收益与门槛汇总 |
| `verify_natural_merge.py` / `verify_natural_merge_data.py` | 逐题评分、预算、冻结状态与数据隔离校验 |
| `freeze_natural_merge.py` | 门槛通过后冻结模型、代码和评测计划 |
| `natural_merge_eval.py` / `natural_merge_runtime.py` | 冻结 checkpoint 评测与共享运行时 |
| `run_natural_merge_parallel.py` | 正式评测按完整 context 分片与合并 |
| `test_natural_merge*.py` | 执行契约 CPU 测试（v1/v2/v3/v4 链共用依赖） |

v4 链复用 v2/v3 的 `prepare_natural_merge_v2/v3.py`、`natural_merge_v2_train.py`、`natural_merge_v3_objective.py` 作为依赖（保留在仓库）。

## LongBench 评测链（当前公开基准）

| 入口 | 用途 |
| --- | --- |
| `longbench_memoryllm_eval.py` | MemoryLLM native50 基线逐题评测 |
| `longbench_prlmc_eval.py` | PRLMC v4 三臂（native50/Dsup/Selection）8k/16k 评测 |
| `run_longbench_native50_parallel.py` | native50 基线分片并行 |
| `test_longbench_memoryllm_eval.py` | 评测契约 CPU 测试 |

## 公共依赖

- `prepare_public_retention.py`、`public_retention_eval.py`、`run_public_retention_parallel.py`、`verify_public_retention.py`、`test_public_retention.py`：原始 SQuAD/NaturalQA 作者协议和共享读写基础。
- `native_qa_compare.py`：原始 QA、原文分片和结果写入等公共工具。
- `natural_merge_train.py`：上一轮自然训练入口，供当前测试调用。
- `agent.py`（methods/prlmc/）：`native_qa_compare.py` 的 backbone 加载依赖。

## 已移出 git 的旧线

S0/S1/S1c/S1d、e6a、readout_probe、residual_subspace_diag、v2/v3 独立运行与汇总入口等旧线文件保留在磁盘与 archive/2026H1_legacy/，可从历史 Git 提交恢复；它们不再被当前链路引用（v4 依赖的 6 个 v2/v3 文件除外）。
