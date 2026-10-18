# PRLM-C：共享潜在记忆融合

在冻结的 MemoryLLM reader 上，学习有界、内容相关的共享残差压缩模块，将多个记忆块中的信息融合进保留的载体。目标是在公开原始任务上，以相同有效记忆槽位预算改善知识保留与读取；评测时模块完全冻结，不进行逐场景训练。

当前主线方案是**自然记忆融合 v5（Dsup v5，grouped_gated_sparse_slot_v5）**：v4 的结构与监督不变，只改内部路由与门控（几何门控、原生路由先验、逐层组原生增益、载体选择层对齐到第 16 层），追求跨域稳定涨点。训练入口 `tools/natural_merge_v4_train.py`，协议见[实验文档](experiments/Natural-merge-v5-routing-generalization.md)。公开长文档评测使用 **LongBench PRLMC**（native50 基线 + Dsup/Selection 对照，8k/16k），入口 `tools/longbench_prlmc_eval.py`，见 docs/04。v4 的 8k/16k 数字仍是现行结果记录。

## 文档导航

- [00 导览](docs/00-overview.md) —— 研究时间线与当前主线
- [03 自然融合主线](docs/03-natural-merge.md) —— v1 → v2 A/B/C → v3 C/D → v4 Dsup → v5 Dsup
- [04 LongBench 评测](docs/04-longbench-eval.md) —— native50 基线 + PRLMC Dsup/Selection 双臂（当前公开评测）
- 01/02/05 为历史线结论索引；旧线完整代码与文档在 archive/2026H1_legacy/（不随 git 发布）

## 仓库结构

```text
methods/prlmc/               共享融合模块、载体选择、记忆布局
methods/memoryllm_modeling/  MemoryLLM reader 实现
tools/                       v5 数据准备、训练、冻结评测、LongBench 评测与公共依赖
experiments/                 主线协议（v5 Dsup、LongBench v4/v5、native50）
docs/                        研究时间线与各阶段结论
reports/                     主线分析报告与数据校验证书
references/                  MemoryLLM 作者评测脚本及来源记录
data/                        固定公开数据版本及哈希清单（不入 git）
runs/ runs_results/ logs/    运行证据（不入 git）
```

## 运行

使用已有 MemoryLLM 环境（PyTorch、Transformers、safetensors、PyYAML、NumPy），准备本地 reader checkpoint 与 tokenizer。

LongBench 评测（当前公开基准入口）：

```bash
python tools/longbench_prlmc_eval.py \
  --data-dir /path/to/LongBench/data \
  --out runs/longbench_prlmc_v5/<tag> \
  --total-tokens 8192 --cuda-device 0 --arm SharedMerge \
  --checkpoint runs/<v5 train>/Dsup.train.pt --tasks hotpotqa --resume
```

v5 训练与冻结评测按 [experiments/Natural-merge-v5-routing-generalization.md](experiments/Natural-merge-v5-routing-generalization.md) 执行；`--arm Selection` 不传 `--checkpoint`。数据准备入口 `tools/stage_natural_merge_data.py` 与 `tools/prepare_natural_merge*.py`。

## CPU 验证

```bash
python -m unittest discover -s tools -p test_natural_merge*.py
python -m unittest discover -s tools -p test_public_retention.py
python -m unittest discover -s tools -p test_longbench_memoryllm_eval.py
```

这些测试检查数据隔离、冻结 reader、梯度桥、源屏蔽、连续合并、评分和汇总等执行契约。GPU 训练效果需要实际逐题生成评测，不能由 CPU 测试替代。

## 历史结果与证据

v1/v2 A/B/C、v3 C/D、v4 Dsup、S0/S1/S1c/S1d、public retention 等旧线实验均未推进、已停线或已退役，结论索引见 docs/；原始证据保留在磁盘 runs/、runs_results/、reports/ 与 archive/（不入 git）。旧执行文件可从历史 Git 提交恢复。
