# PRLM-C — 冻结 MemoryLLM reader 上的共享记忆融合

**目标（冻结）**：不大改论文主线方案，通过调节内部架构实现具有强泛化性的稳定涨点。
**主张（冻结）**：在相同、有限的潜在记忆容量下，通过跨块信息融合，提高长文档中知识的保留和可读取能力。
**核心指标**：`Ours − Selection`。两臂的写入流、载体块、容量、解码与评分完全相同，唯一差别是压缩窗口是否经过融合算子。

本仓库只保留**当前版本**：V10 保流形槽位嫁接（manifold-preserving slot grafting）。
v1–v9 各代实现、旧 baseline 评测器、旧报告与旧运行目录已从仓库移除，需要时从 git 历史取回。

## 方法（V10）

冻结的 MemoryLLM-8B 读侧不动，只有压缩算子可训练。一次 19→9 压缩时：

1. 载体块（9 个）由选择器在窗口内按内容与代际年龄挑出，V10 不改变载体选择；
2. 每个 256 槽的块被切成 **216 个核心槽 + 40 个嫁接槽**；核心槽逐位复制载体，硬掩码保证残差为 0；
3. 40 个嫁接槽是**同块内原生槽的测地压缩**（保范数、保流形），每次 merge 重新推导，不跨块写入；
4. 全局残差 RMS 预算 0.24，且只花在嫁接槽上（`.24/sqrt(.156) ≈ .61` 触到 bridge 上限），核心槽始终是原生槽。

可训练参数 2,359,313（`down_proj/up_proj/route_q/route_k/cluster_logit_scale/ortho_gate_logits/homo_gate_logits`），
checkpoint `runs/natural_merge_v9/full_v10.pt`，sha256 `a4d1f6e77d899e40d4d411c43311fd175d65f74ec4768725985ef17acfd47134`。
设计细节见 [docs/01-method-v10.md](docs/01-method-v10.md)。

## 结果（当前证据）

| 实验 | 样本 | 准确率（cue 口径） | 备注 |
| --- | --- | --- | --- |
| MemoryLLM-8B 原生 50 块 | 503 | **18.7%** | 无压缩，官方 judge |
| V10 vs Selection，50 块预算 | 20 | **20.0% vs 20.0%** | 答对的是同 4 题；12/20 预测文本不同 |

定位实验（HotpotQA 158 题配对）的完整数字、以及每个配对的救回/弄坏与 McNemar p，
在 [docs/03-results.md](docs/03-results.md)。

结论：**主表上没测到涨点**。157 题上 V10 与原生基线统计不可区分（±6pp）；20 题配对比较是平局，
样本量只够排除 ±18pp 的差异。

更值得注意的一条：两臂对**正确选项**的概率都只有 0.24 左右（四选一随机是 0.25），
20 题里没有一题过半。融合把 19/20 题的概率挪动了一点，但没有一题因此改对 ——
说明在这个 128k / 50 块设置下读者基本没读到证据，融合是否有用还不是主要矛盾。

实验 1B（把删掉的槽先融进 core，读出蒸馏监督）是**负结果**：158 题上 `Fusion` 23.42 vs
`Graft` 25.95（1 救 5 伤，p = 0.22）。它把读出距离拉近了 20%，准确率反而掉 2.5 pp —— 训练目标被
优化了，但和"答对"脱钩。

实验 2（**来源端浓缩**）是**第一个统计显著的正结果**：不改容量、不改载体、不动 1,944 个 core，
只让 360 个嫁接槽各自从"自己来源块里未被选中的槽"聚合一个残差（0.52 M 模块，用真实答案交叉熵
训练，MuSiQue 训 / HotpotQA 测）。

| 158 题 | 答对 | 准确率 | QA F1 |
| --- | --- | --- | --- |
| `Full`（不压缩） | 47 | 29.75% | 30.40 |
| `Graft`（原生嫁接） | 41 | 25.95% | 27.05 |
| **`Fusion`（来源端浓缩）** | **57** | **36.08%** | **32.82** |

`Fusion − Graft`：23 救 / 7 伤，净 **+16**，McNemar **p = 0.005**。证据在被删块里的 40 题从
7/40 涨到 16/40。**同结构、同残差幅度、但不训练的随机对照只 +1 题（p = 1.00），
训练版比随机版多对 15 题（p = 0.008）** —— 收益来自学习，不是扰动。

**但收益不来自那 2,200 个候选槽位**：把每个锚点读的候选全部换成另一篇文档的（不重训、同 158 题），
`Counterfactual` 仍是 **57 题 / 36.08% / F1 32.82**，与训练版 **0 救 0 伤、逐题 F1 相同**（p = 1.00）。
机制探针把原因量了出来：候选变了 130%，`context` 只变 19%，残差 5.0%，窗口 1.5%，答案 0。

再把候选路径**整条删掉**（只让锚点自己产生残差，参数量相同、训练完全一致）：`AnchorOnly` 得到
**56 题 / 35.44% / F1 33.35**，与训练版 **差 1 题、p = 1.00**，对 `Graft` 仍 +15 题（p = 0.011）。

| 权重 | 候选 | 158 题正确 |
| --- | --- | --- |
| 随机 | 自己的 | 42 |
| 训练 | 自己的 | 57 |
| 训练 | 别人的 | 57 |
| 训练 | **没有** | **56** |

**起作用的是 0.52 M 参数学到的"锚点自调整"，不是候选里的知识** —— "跨块信息融合"这个说法目前
没有证据，能写的是"相同容量下提高嫁接槽位的可读取性"，而且有一个更简单、更强的实现。
详见 [docs/03-results.md](docs/03-results.md) 第 6、7 节。

实验 3（**在冻结的校准之上再读一次候选**）把这个结论钉死了：冻结 `AnchorOnly`，只在它上面加一个
只读本块 **Top-4** 候选（原生 4,096 维混合、向量范数限幅）的分支，158 题掉到 **50 / 31.65%**，
对 56 题基线 **净 −6（10 救 16 伤，p = 0.33）**；同一个分支换成读**别的文档**的候选反而回到
**53**。47 道 `Full` 答对的题被分支改错 21 道（`AnchorOnly` 是 13 道）。丢弃块证据的 40 题上，
读别人的候选（40.0%）比读自己的（32.5%）更好。

| 做法 | 158 题正确 |
| --- | --- |
| 候选全读（实验 2） | 57 |
| 候选不读（Anchor-Only） | 56 |
| 候选换掉（反事实） | 57 |
| 候选只读 Top-4，加在冻结校准之上 | **50** |
| Top-4 但读别人的 | 53 |

**"把未入选槽位的知识浓缩进锚点"这条路，三个独立机制都没走通**（全注意力、删掉、稀疏原生宽度
+ 范数限幅），第三个方向还是负的。论文主线因此收束到第一层：跨块搬运后的原生槽位如何适配冻结
Reader。详见 [docs/03-results.md](docs/03-results.md) 第 8 节。

实验 4（**Key-Safe 投影**）把"破坏寻址"这一条单独拎出来验证：在实验 3 的权重上，**不重训**，只把
残差限制到 Reader 的 key 子空间零空间里并保持槽位模长。实测 key 变化从 **1.63e-1 降到 5.3e-4**
（小 308 倍，已到 fp16 池的舍入地板），value 仍变 14%。结果：**损伤从 21 题降到 15 题，保护组误伤
从 9 降到 3，总答对 50 → 54，但仍没超过 56**；而且真候选与无关候选变得**完全等价**（54 vs 54，
0 救 0 伤，预测只差 3/158，实验 3 是 38/158）。

也就是说：实验 3 里那点内容依赖，绝大部分是**通过寻址通道**传到答案的——而那正是破坏已有知识的
通道。冻结寻址之后，value 通道几乎传不动内容。详见 [docs/03-results.md](docs/03-results.md) 第 9 节。

反常识的一条：按"压缩后记忆像不像原始记忆"（读出距离）衡量，实验 2 **差了 74%**，而 1B 是近了
20% 但准确率更差。两次方向相反、结论一致：**读出距离不是知识可用性的代理**。

完整数字、分组、对照与保留见 [docs/03-results.md](docs/03-results.md) 第 4–9 节、
[reports/exp2_source_compaction/](reports/exp2_source_compaction/)、
[reports/anchor_only/](reports/anchor_only/) 与
[reports/anchor_fusion/](reports/anchor_fusion/)。

口径与限制见 [docs/02-eval-protocol.md](docs/02-eval-protocol.md)，
读报告时的两个约定（`--operator-variant native`、`last_merge` 记录）见
[docs/04-diagnostics.md](docs/04-diagnostics.md)。

## 仓库结构

```text
methods/prlmc/                 压缩算子、载体选择、记忆布局、可微注入
methods/prlmc/core_fusion.py   1B 的 core 融合模块（0.52 M 参数）
methods/prlmc/source_compaction.py  2 的来源端浓缩模块（0.52 M 参数）
methods/prlmc/anchor_residual.py  Anchor-Only 的锚点残差模块（同参数量的 0.52 M）
methods/prlmc/anchor_fusion.py    冻结校准 + 稀疏 Top-4 候选分支（可训练 0.52 M）
methods/memoryllm_modeling/    MemoryLLM reader（冻结）
tools/plan.py                  训练计划构建与校验（sealed plan）
tools/objective.py             V10 训练目标（恢复/蒸馏/保护/锚定/EOS）
tools/train.py                 训练入口（704 更新）
tools/train_core_fusion.py     1B 训练入口（只训融合模块，读出蒸馏）
tools/eval_learned_fusion.py   1B 评测入口（Graft vs Fusion，六臂）
tools/train_source_compaction.py  浓缩训练入口（--variant compaction / anchor-only）
tools/eval_source_compaction.py   2 评测入口（Graft vs Fusion，四臂）
tools/eval_anchor_residual.py     Anchor-Only 评测入口（五臂，含 Fusion 对照）
tools/eval_anchor_fusion.py       实验 3 评测入口（六臂，含 Donor 对照）
tools/randomize_compaction.py     2 的随机对照（同幅度、不训练）
tools/eval_counterfactual_candidates.py  2 的反事实消融（候选换成另一篇文档的）
tools/probe_compaction_sensitivity.py    2 的机制探针（候选→残差→窗口的逐级衰减）
tools/runtime.py               流式运行时（写入、触发、merge、装载）
tools/stream_eval.py           流式记忆与参考臂（Selection/Recent/Full）
tools/v9_geometry_50block.py   50 块几何常量（被哈希锁定的算子直接 import）
tools/eval_lbv2.py             LongBench v2 双口径评测入口（A/B/C/D 四臂）
tools/diagnose_ab.py           读已有报告对比逐题选项概率（不跑模型）
tools/eval_core.py             LongBench 8k/16k 评测入口（v2 复用其运行时）
tools/lbv2_data.py             LBv2 数据与官方模板
tools/native_eval.py           MemoryLLM-8B 原生基线评测
tools/io_utils.py              sha/digest/atomic_json/tokenizer manifest
configs/agent_conf/prlmc_e6.yaml  运行时 manager 配置
reports/main_table/            主表结果与证据
reports/diagnostics/           定位实验 1A 的报告（也是后续实验的 --verify-against 参考）
reports/exp2_source_compaction/ 实验 2 的训练记录、日志、主结果、随机对照与反事实消融
reports/anchor_only/           Anchor-Only 的训练记录与 158 题主结果
reports/anchor_fusion/         实验 3 的训练记录与 158 题主结果
docs/                          目标、方法、评测口径、结果
```

## 运行

环境：`python3.10 + torch 2.5.1 + transformers 4.46.3`，本地 MemoryLLM-8B fp16 cache（`/dev/shm/s1d_reader_fp16`）。

训练（16 条流，50 块预算，704 更新）：

```bash
python tools/train.py \
  --plan data/natural_merge_v9_50block/train.json \
  --model-path /dev/shm/s1d_reader_fp16 \
  --cuda-device 0 --variant DnativeV9 --qa-batch 1 \
  --out reports/main_table/full_v10.json \
  --checkpoint-out runs/natural_merge_v9/full_v10.pt
```

评测（两臂必须用同一份数据与同一组载体；`--carrier-schedule` 让 Selection 复现 V10 的载体序列）：

```bash
python tools/eval_lbv2.py --arm SharedMerge --checkpoint runs/natural_merge_v9/full_v10.pt \
  --data-json <lbv2/data.json> --budget-blocks 50 --cuda-device 0 --out <v10.json>
python tools/eval_lbv2.py --arm Selection --carrier-schedule <v10.json> \
  --data-json <lbv2/data.json> --budget-blocks 50 --cuda-device 0 --out <selection.json>
```

## 命名与哈希锁定（重要）

V10 算子由 `methods/prlmc/manifold_graft.py` 实现，但在训练计划、架构名与 checkpoint 里沿用
`natural_merge_v9_training_plan/1` / `grouped_gated_sparse_native_v9` / `DnativeV9`。
这些字符串被 sealed plan 的 `content_sha256` 与 checkpoint 的 `training` 记录钉死，改名会让已有
checkpoint 无法校验，因此保持原样。

`tools/runtime.py` 的 `CRITICAL_MERGE_FILES` 列出 checkpoint 要求逐字节一致的算子文件：
`consolidate.py`、`layout.py`、`importance.py`、`slot_merge.py`、`gated_slot_merge.py`、`native_skip_merge.py`。
其中 `slot_merge.py` / `gated_slot_merge.py` 是早期算子，**不在 V10 路径上**，但 `full_v10.pt` 的
source manifest 记录了它们的哈希，删除或改动会让 checkpoint 校验失败（`validate_merge_checkpoint`
会直接报错），所以它们保留在仓库里。`native_skip_merge.py` 同时承载 V10 的流式适配器。
