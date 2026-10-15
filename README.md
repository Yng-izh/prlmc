# PRLM-C：共享潜在记忆融合

在冻结的 MemoryLLM reader 上，学习有界、内容相关的共享残差压缩模块，将多个记忆块中的信息融合进保留的载体。目标是在公开原始任务上，以相同有效记忆槽位预算改善知识保留与读取；评测时模块完全冻结，不进行逐场景训练。

当前最新方案是 **自然记忆融合 v4 的 Dsup/Dnative 对照**：Dsup 修复答案边界、真实 EOS 监督并覆盖全部原始训练 QA；Dnative 在相同监督下增加一条全维源表示直通支路。两者均使用 64 次更新和 5% 总体残差预算。v4 已完成 GPU 训练和开发 E1；Dsup 通过开发门槛，Dnative 暂不保留。见[设计与运行命令](experiments/Natural-merge-v4-D-supervision-native.md)。

数据使用原始 SQuAD context、question 和 answer，训练与开发、正式评测按文章隔离。人造 episode 和逐窗口 C13 已退出主方法，仅保留历史诊断证据。

## 已完成的 v2 实验

| 组别 | 结构 | 训练目标 |
| --- | --- | --- |
| A | 块级 `SharedMemoryMerge` | 原始答案 CE |
| B | 与 A 相同 | 压缩损失加权 CE + Full 答案位置 KL |
| C | 槽位交叉注意力 `SlotMemoryMerge` | 与 B 相同 |

三组固定同一 reader、312 个原始训练 context、1402 道原始 QA、rank=32、残差上限 0.05、19→9 块压缩和 64 次更新。只使用最后 checkpoint，生成协议及推进门槛预先固定。B−A 检验监督变化，C−B 检验整体融合结构变化。

- [完整 A/B/C 协议与运行命令](experiments/Natural-merge-v2-ABC.md)
- [E1/E2/E3 数据划分、评测与冻结门槛](experiments/Natural-cross-block-merge-design.md)
- [本轮启动记录](reports/natural_merge_v2_20261014/status.md)
- [报告索引与证据边界](reports/README.md)

三组均已完成 64 次更新、训练内生成诊断和开发 E1。93 道开发问题上，Full / Selection / A / B / C 分别答对 35 / 26 / 27 / 27 / 27；三组相对 Selection 新增同一道题，无损伤，但均未恢复 Full-correct / Selection-wrong 的 9 道压缩损失题。非载体完整融合−屏蔽融合净收益均为零，三组都未通过推进门槛，未启动正式 E1/E2/E3。见[本轮结果与限制](reports/natural_merge_v2_20261014/results.md)和[原始对比](reports/natural_merge_v2_20261014/completed_run/comparison.json)。目前尚未证明正式自然任务上的稳定收益或跨任务泛化。

上一轮自然融合开发 E1 同样未通过门槛，见[原始裁决](reports/natural_merge_initial_20261013/results.md)。历史失败结果与本轮原始证据均保留。

## 仓库结构

```text
methods/prlmc/          共享融合模块、载体选择、记忆布局和兼容组件
methods/memoryllm_modeling/  用于这些实验的 MemoryLLM reader 实现
tools/                  数据准备、训练、冻结评测、验证和必要兼容依赖
experiments/            当前自然融合 v1/v2/v3/v4 协议
data/                   固定公开数据版本及哈希清单；原始数据另行下载
reports/                本轮记录和历史原始报告、逐题预测与校验证书
references/             MemoryLLM 作者评测脚本及来源记录
archive/                旧实验文档、历史导航及本次清理清单
```

[工具入口](tools/README.md)区分当前实验与兼容依赖。历史报告和 source snapshot 保留原始内容及哈希；其中的运行时路径描述当时环境，不代表当前入口。

## 运行

使用已有 MemoryLLM 环境（PyTorch、Transformers、safetensors、PyYAML、NumPy），准备本地 reader checkpoint 与 tokenizer。MemoryLLM reader 权重、原始公开数据和大规模临时运行文件不随仓库发布；已归档的小型共享模块 checkpoint 另见报告。

已有 `data/natural_merge_v1/` 和 `data/natural_merge_v4/train.json` 时，在具备足够显存的服务器运行 v4：

```bash
python tools/run_natural_merge_v4.py \
  --train-plan data/natural_merge_v4/train.json \
  --dev-plan data/natural_merge_v1/E1-development.json \
  --model-path /dev/shm/s1d_reader_fp16 \
  --devices 7,9 --variants Dsup,Dnative \
  --out-dir runs/natural_merge_v4_dd
```

v4 运行器依次执行 smoke、64 次全新训练、训练内诊断和开发 E1，并保存逐题结果。当前服务器归档结果位于 `runs/natural_merge_v4_dd_20261015_0108/`；Dsup 开发 E1 为 39/93，Selection 为 26/93，Dnative 为 36/93。正式 E1/E2/E3 尚未由该运行器自动启动。

如需复现已归档的 v2 A/B/C 对照，使用原 v2 入口：

```bash
python tools/run_natural_merge_v2.py \
  --train-plan data/natural_merge_v2/train.json \
  --dev-plan data/natural_merge_v1/E1-development.json \
  --model-path /dev/shm/s1d_reader_fp16 \
  --devices 1,2,3 --variants A,B,C \
  --out-dir runs/natural_merge_v2_abc
```

卡号和模型路径按实际环境设置。输出目录必须不存在，运行器会检查每卡至少 28000 MiB 空闲；已启动任务不要重复执行。每组依次运行 smoke、全新训练、训练内生成诊断、开发 E1，然后生成对比报告。训练内生成不作为泛化证据；运行器不会自动开启正式 E1/E2/E3。

首次准备数据及逐组运行请按对应[完整协议](experiments/Natural-merge-v4-D-supervision-native.md)执行。原始 SQuAD 下载入口为 `tools/stage_natural_merge_data.py`；v1 计划准备还要求已观察问题集、匹配的公开干扰段落和固定 NaturalQA 计划，不能省略这些输入来绕过隔离审计。

## CPU 验证

```bash
python -m unittest discover -s tools -p 'test_natural_merge*.py'
python -m unittest discover -s tools -p 'test_public_retention.py'
```

这些测试检查数据隔离、冻结 reader、梯度桥、源屏蔽、连续合并、评分和汇总等执行契约。GPU 训练效果需要实际逐题生成评测，不能由 CPU 测试替代。

## 预算与历史结果

Selection 与共享融合使用相同有效槽位预算。物理 reader 记忆池仍保持模型大小，因此有效槽位减少不等于节省物理显存；NativeBudget 的初始化、布局及淘汰策略也不同，只作为系统级对照。

旧受控数据、C13、保护、扩维等实验归入[历史索引](archive/README.md)，不作为公开自然任务有效性的主证据。被移除的旧执行文件可从清理前 Git 提交恢复；本次清理没有修改正在运行的服务器代码、计划或进程。
