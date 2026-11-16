# PRLM-C: 冻结 MemoryLLM reader 上的跨块记忆校准

**目标(冻结)**:不大改论文主线,通过调节压缩算子的内部结构,在固定记忆容量下得到稳定涨点。
**主张(冻结)**:相同、有限的潜在记忆容量下,跨块信息能提高长文档中知识的保留与可读取能力。
**核心指标**:`Ours − Selection`。两臂的写入流、载体块、容量、解码与评分相同,只差压缩算子。

MemoryLLM-8B 的 reader/writer 全部冻结,只有压缩算子可训练。
一次压缩把 19 块(每块 256 槽)压成 9 块,总容量 2,304 槽。

## 当前主线(只保留这四个版本)

| 版本 | 做什么 | HotpotQA 158 题答对 | 状态 |
| --- | --- | --- | --- |
| V10 | 同块内原生槽测地压缩,嫁接 360 槽 | 41 | 基线,`full_v10.pt` |
| V11Evict | 窗口 = 1,620 载体槽 + 684 被删块槽(Reader 效用选槽,逐字复制),全池残差校准 | 61 | 窗口结构 |
| **V13** | V11Evict 窗口 + 校准器额外读取其他块里 key 余弦 top-4 的槽位内容 | **67** | **当前最好** |
| V17-density | V13 上不训练,按块内冗余重分配 core 配额 | 66 | 与 V13 无差别 |

参照:Full(19 块不压缩)47,Selection(只保留 9 个载体块)35,AnchorOnly 56。
V14、V15、V16、V13+ 等变体全部不如 V13,代码保留为开关,结果只记在总表里。

**一句话结论**:V13 的 +6(对 V11Evict)p=0.07,换成别人的 graft 内容后掉 13 题(p=0.024),
所以收益依赖真实搬入的内容。但随机邻居只比真邻居低 5 题(p=0.18),
"跨块邻居"本身的贡献还没有显著性。LongBench v2 主表没有测到涨点。
详细数字、判定线与限定见 [docs/06-experiment-ledger.md](docs/06-experiment-ledger.md)。

## 文档导航

| 文件 | 内容 |
| --- | --- |
| [docs/00-overview.md](docs/00-overview.md) | 目标、主张、方法演进的因果链 |
| [docs/06-experiment-ledger.md](docs/06-experiment-ledger.md) | **实验总表**:每个版本改了什么、几题、结论、为什么淘汰 |
| [docs/01-method-v10.md](docs/01-method-v10.md) | V10 算子细节 |
| [docs/05-method-v13.md](docs/05-method-v13.md) | V11Evict 窗口与 V13 校准器(代码口径) |
| [docs/02-eval-protocol.md](docs/02-eval-protocol.md) | LongBench v2 评测口径 |
| [docs/03-results.md](docs/03-results.md) | 早期实验 1 到 8 的详细记录(附录) |
| [docs/04-diagnostics.md](docs/04-diagnostics.md) | LongBench v2 的 A/B/C/D 问题定位 |
| [tools/README.md](tools/README.md) | 工具入口 |

## 目录结构

```text
methods/memoryllm_modeling/    冻结的 MemoryLLM reader
methods/prlmc/                 压缩算子
  consolidate.py layout.py importance.py   载体选择与记忆布局(V10 起)
  manifold_graft.py                        V10 算子
  evict_window.py reader_utility.py        V11Evict 窗口与 Reader 效用选槽
  cross_block_calibration.py               V13 校准器(含 V14 到 V16 的开关)
  anchor_residual.py core_fusion.py ...    V13 依赖的早期模块
tools/train.py                 V10 训练
tools/train_cross_block_calibration.py   V11Evict / V13 训练(--cross on|off)
tools/eval_cross_block_calibration.py    HotpotQA 158 题多臂评测(主评测)
tools/eval_lbv2.py             LongBench v2 评测
tools/run_free_v17_evals.sh    不训练的 V17 评测
v13_lr_chain.sh                产出 V13=67 的训练加评测链
reports/cross_block/           V13 / V17 的评测报告
reports/main_table/            LongBench v2 主表
reports/diagnostics/           实验 1 到 8 的原始报告
runs/                          checkpoint(只保留 4 个,见下)
```

保留的 checkpoint:`runs/natural_merge_v9/full_v10.pt`(V10,sha256 `a4d1f6e7…47134`)、
`runs/anchor_residual/anchor_only_rank64.pt`(锚点初始化)、`runs/cross_block/plain.pt`(V11Evict 校准器)、
`runs/cross_block/cross_lr.pt`(V13)。

## 运行

环境:`/data/zhangjiayang/yyz/env/bin/python`(py3.10, torch 2.5.1, transformers 4.46.3),
reader 缓存 `/dev/shm/s1d_reader_fp16`(重启后需要重新放置)。

```bash
# 复现 V13 = 67/158(训练约 384 更新,再评测)
bash v13_lr_chain.sh

# 邻居消融(随机 / 最差邻居)
bash diag_chain.sh

# 不训练的 V17 评测
bash tools/run_free_v17_evals.sh
```

V10 训练、LongBench v2 评测命令见 [tools/README.md](tools/README.md)。

## 哈希锁定(重要)

`full_v10.pt` 记录了算子文件的 sha256。`consolidate.py`、`layout.py`、`importance.py`、
`slot_merge.py`、`gated_slot_merge.py`、`native_skip_merge.py` 不能改动或删除,否则
`validate_merge_checkpoint` 会报错。命名里的 `v9`、`DnativeV9` 同理,是被 sealed plan 钉死的字符串。

## 仓库整理记录

2026-11-16 删除了 v1 到 v9 的历史代码与产物、V14 到 V16 及 V13+ 的 checkpoint 与大部分报告(保留 `reports/cross_block/eval_v16.json`,它是 V17 评测的参考报告)、
`reports/` 下旧目录,共 190 项(约 3.7GB)。目录不是 git 仓库,删除不可恢复。
被删项清单在 [docs/_deleted_20261116.txt](docs/_deleted_20261116.txt)。
V14 到 V16 的结论仍记在实验总表里,需要时可用 `v14/v15/v16` 开关重训。
