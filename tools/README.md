# 工具入口（V10 单一版本）

| 文件 | 用途 |
| --- | --- |
| `io_utils.py` | `sha` / `digest` / `atomic_json` / `tokenizer_manifest` |
| `plan.py` | sealed 训练计划的构建与校验；`sealed_training()` 供 checkpoint 校验读取封存记录 |
| `objective.py` | V10 训练目标：恢复 CE、Full 教师 KL、保护 hinge、首 token 锚定、EOS；含可微注入用到的共享辅助 |
| `train.py` | 训练入口（704 更新，`--resume`、原子保存、滚动 `_latest.pt`） |
| `runtime.py` | 流式运行时：`MergeStream`、`install`、`load_runtime`、几何解析、checkpoint 校验 |
| `stream_eval.py` | 流式记忆与参考臂（`Selection` / `Recent` / `Full` / `NativeDefault`） |
| `v9_geometry_50block.py` | 50 块几何常量（`GEOMETRY_50B`、`TRIGGER_BLOCKS`、`DISTILL_LAYERS`） |
| `eval_lbv2.py` | LongBench v2 双口径评测（主表入口） |
| `eval_core.py` | LongBench 8k/16k 评测（v2 复用其运行时与协议记录） |
| `lbv2_data.py` | LBv2 数据加载、官方模板与官方答案抽取 |
| `native_eval.py` | MemoryLLM-8B 原生基线评测（50 块池、不压缩） |
| `eval_graft_recovery.py` | 实验 1：原生槽位嫁接的知识恢复/损伤（Full / Selection / Graft，19→9，单次压缩） |
| `eval_replace_reorder.py` | 实验 2：替换 vs 重排的损伤归因（Full / Selection / Replace / Graft，B 与 C 的槽位向量完全相同） |
| `eval_evict_only.py` | 实验 3：Evict-Only，把替换损伤拆成"删槽"和"插槽"（Full / Selection / Evict / Replace） |
| `eval_graft_span.py` | 实验 1A：嫁接来源端的 Span-4 成组选择（Full / Selection / Graft / Span，只改来源侧） |
| `train_core_fusion.py` | 实验 1B 训练端：只训 0.52 M 的 core 融合模块，目标是 Reader 读出蒸馏（不碰答案） |
| `eval_learned_fusion.py` | 实验 1B：学习式多对一融合 vs 原生嫁接（Full / Selection / Graft / Fusion / Graft2 / Fusion2） |
| `train_source_compaction.py` | 浓缩训练端（`--variant compaction` / `anchor-only` / `anchor-fusion`，后者可加 `--fusion-mode key-safe`）：只训 0.52 M 模块，目标是**真实答案交叉熵** + KL 保护 |
| `eval_source_compaction.py` | 实验 2：来源端浓缩 vs 原生嫁接（Full / Selection / Graft / Fusion，单次压缩） |
| `eval_anchor_residual.py` | Anchor-Only：删掉候选路径后还剩多少（Full / Selection / Graft / Fusion / AnchorOnly） |
| `eval_budget_reallocation.py` | 实验 5：只改 Core/Graft 配额（1944/360 → 1440/864 → 1296/1008），同 2304 容量、同载体、冻结 Reader；每配置跑 Native Graft + 已训练 AnchorOnly（不重训），并按运行时配额校验容量/来源/时间顺序/逐字 core |
| `eval_anchor_fusion.py` | 冻结锚点校准后再试候选：稀疏 Top-4 + 原生维度混合（Full / Selection / Graft / AnchorOnly / AnchorFusion / AnchorFusionDonor）；按 checkpoint 的 `fusion_mode` 重建算子，`key-safe` 会实测 key/value 变化 |
| `test_key_safe.py` | Key-Safe 合并的数学检查（CPU，18 项）：正交基、零空间、保范、初始逐位一致、实测 key/value 变化、LoRA 折叠与错误折叠拒绝 |
| `randomize_compaction.py` | 实验 2 的随机对照：同结构、同残差幅度、不训练，用来排除"随便扰动就有用" |
| `eval_counterfactual_candidates.py` | 实验 2 反事实消融：候选槽位换成另一篇文档的，不重训，五臂同进程（Full / Selection / Graft / Fusion / Counterfactual） |
| `eval_bridge_routing.py` | 实验 8（V12）：桥接路由 + 动态容量分配，不训练任何参数（Full / Selection / Graft / AnchorOnly / V11NativeGraft / Bridge）；逐题配对、三组子集、分配诊断，并按预定三分支规则机械判定 |
| `probe_compaction_sensitivity.py` | 实验 2 机制探针：把候选 → context → raw → 残差 → 窗口的衰减逐级量出来（不跑 reader 生成） |

`v9_geometry_50block.py` 的文件名是历史遗留：`methods/prlmc/consolidate.py` 被 checkpoint 哈希锁定，
它直接 `from tools.v9_geometry_50block import GEOMETRY_50B`，因此该文件名不能改。

## 训练

```bash
python tools/train.py --plan data/natural_merge_v9_50block/train.json \
  --model-path /dev/shm/s1d_reader_fp16 --cuda-device 0 \
  --variant DnativeV9 --qa-batch 1 \
  --out reports/main_table/full_v10.json \
  --checkpoint-out runs/natural_merge_v9/full_v10.pt
```

常用开关：`--smoke-updates N`（冒烟）、`--preflight`（6 个事件的结构检查）、
`--save-every N`（小 checkpoint 间隔）、`--resume <ckpt>`（权重+优化器+池状态续跑）、
`--no-reader-checkpointing`（显存够时关掉重算）。

## 评测

```bash
# V10 臂
python tools/eval_lbv2.py --arm SharedMerge \
  --checkpoint runs/natural_merge_v9/full_v10.pt \
  --data-json /data/.../LongBench-v2/data.json \
  --budget-blocks 50 --cuda-device 0 --out reports/main_table/<v10>.json

# Selection 臂：复现 V10 的载体序列，唯一差别是没有融合
python tools/eval_lbv2.py --arm Selection \
  --carrier-schedule reports/main_table/<v10>.json \
  --data-json /data/.../LongBench-v2/data.json \
  --budget-blocks 50 --cuda-device 0 --out reports/main_table/<selection>.json
```

`--max-samples N` 跑子集；`--rescore` 只重算已有报告的评分（不跑模型）；
`--strict-code-hash` 默认开启，算子文件与 checkpoint 记录不一致会直接报错。

## 问题定位（A/B/C/D）

四个臂必须同题、同口径。`--sample-ids` 是固定的 20 题清单（`configs/lbv2_sample_ids_n20.json`，
取自 A/B 两份报告的 `samples[].id`，顺序一致），
`--length all --total-tokens 128000 --tail-tokens 2048` 与 A/B 的协议一致；
少了 `--length all`（默认是 `medium`）会选到另外 20 道题。

```bash
# A：只保留、不融合（复现 B 的载体序列）。C 要跟它比，所以必须用同一份代码重跑一次
python tools/eval_lbv2.py --arm Selection \
  --carrier-schedule reports/main_table/<B>.json \
  --data-json <lbv2/data.json> --length all --total-tokens 128000 --tail-tokens 2048 \
  --budget-blocks 50 --sample-ids configs/lbv2_sample_ids_n20.json --cuda-device 0 --out <A>.json

# C：只有最后一次压缩融合（前面全部 Selection，且沿用 B 的载体序列）
python tools/eval_lbv2.py --arm SharedMerge --fusion-policy last \
  --carrier-schedule reports/main_table/<B>.json --checkpoint runs/natural_merge_v9/full_v10.pt \
  --data-json <lbv2/data.json> --length all --total-tokens 128000 --tail-tokens 2048 \
  --budget-blocks 50 --sample-ids configs/lbv2_sample_ids_n20.json --cuda-device 0 --out <C>.json

# D：同 C，但算子只做原生槽位替换（不做凝聚/正交/Core 强化）
python tools/eval_lbv2.py --arm SharedMerge --fusion-policy last --operator-variant native ...

# 不跑模型，直接对比已有报告的逐题选项概率（按题目 id 配对，不按顺序）
python tools/diagnose_ab.py <A>.json <C>.json --labels Selection C
```

`--fusion-policy last` 和 `--operator-variant native` 只对 `--arm SharedMerge` 有效，其它臂会直接报错；
`--carrier-schedule` 缺任何一题的条目也会报错点名，不会静默退回自己的选块。

每个样本会额外记录 `last_merge`：最后一次融合保留了哪些写入块、丢弃了哪些、
被丢弃的那段落在文档的哪个比例区间、正确选项的措辞是否出现在里面。
注意 `dropped` 指的是"这些块没有留下自己的载体"——被丢弃块仍有 40/256 的槽会被嫁接进输出，
所以它不等于"内容丢了"；单token 选项（`China`、`1945`）的匹配也不具区分度，
`answer_wording_distinctive` 会标出来。设计说明见 [docs/04-diagnostics.md](../docs/04-diagnostics.md)。

## 实验 1：原生槽位嫁接的知识恢复能力

回答两个问题：Selection 删掉含证据的块之后，原生槽位嫁接能不能把答题能力救回来（Rescued）；
嫁接要占掉保留块自己的 40 个槽，会不会反而弄坏原本能答的题（Harmed）。

每个样本只写 **19 个块**（9,728 token 的文档前缀），**只压缩一次** 19→9，然后从同一份
压缩前记忆上读三次：`Full`（19 块 4,864 槽，不压缩）、`Selection`（9 块 2,304 槽）、
`Graft`（同样 9 个载体，216 原生 core 槽 + 40 个从被丢弃块直接复制的原生槽 = 2,304 槽）。
载体集合由 `select_window_carrier_blocks` 在压缩前窗口上**只算一次**，A/B 共用，所以两者只差
"哪些原生槽进入 Reader"。`Graft` 用 `operator_mode="native"`：不做同块凝聚、不做正交上下文增量、
不做 core 强化。

```bash
# 先看会跑哪些题、证据落在哪些块（只加载 tokenizer，不占 GPU、不写文件）
python tools/eval_graft_recovery.py --dry-run \
  --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
  --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
  --model-path /dev/shm/s1d_reader_fp16

# 正式跑（三个臂在同一次前向里，不需要 carrier-schedule）
python tools/eval_graft_recovery.py \
  --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
  --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
  --model-path /dev/shm/s1d_reader_fp16 \
  --checkpoint runs/natural_merge_v9/full_v10.pt \
  --cuda-device 0 --out reports/diagnostics/graft_recovery_hotpotqa.json

# 不跑模型，重新分组已有报告（--f1-threshold 可以换判对阈值，不用重跑）
python tools/eval_graft_recovery.py --analyze reports/diagnostics/graft_recovery_hotpotqa.json \
  --f1-threshold 0.3
```

口径说明：

- 写入流只有**正文前缀**，官方模板里前置的那句 instruction 被去掉，这样第 b 块正好覆盖正文
  `[512b, 512(b+1))` token，证据的 字符→token→块 映射是精确的；后置 instruction + 问题 + `Answer:`
  作为 query，三个臂用完全相同的 query token。
- 上下文不足 19 块的题**排除并计数**，不填充——填出来的窗口是另一个实验。
- "证据"用 HotpotQA dev 的 `supporting_facts` 支持句定位，不用答案串匹配（答案串只作旁证，
  记在 `evidence.answer_string_blocks`）。支持句先归一化再回映射到原始字符偏移，
  经 tokenizer offset mapping 落到块号。
- `Graft` 的 360 个嫁接槽来源记在 `provenance.graft_slots_by_source_block`（按窗口块号），
  可以回答"含证据的被丢弃块到底有没有贡献嫁接槽"。
- 筛选用了答案和证据位置，所以分组的准确率是**机制诊断**，不是数据集成绩；
  报告里 `protocol.screening_caveat` 写死了这句话，全量列 `arms` 才是可比的数字。

## 实验 2：替换和重排究竟损伤了多少知识

实验 1 只测了 `Selection → Graft` 的**联合**效果，无法区分损伤来自哪一步：
(i) 每个载体让出 40 个自己的槽、被外来内容覆盖；还是 (ii) 槽位向量完全相同，但
Reader 读到的**顺序**变了（V10 按来源时间输出，嫁接槽夹在载体槽中间，不在载体原来的位置上）。

这里加一个 B 组，它是 C 组输出的**纯置换**，所以 B、C 持有字面相同的 2,304 个槽位向量：

| 臂 | 做法 |
|---|---|
| `Full` | 19 块 4,864 槽，不压缩（参考） |
| `Selection` | 9 个载体块原样（基线 A） |
| `Replace` | 每个载体保留自己的 216 个 core 槽**在原位置**，腾出的 40 个位置填 V10 的嫁接槽（B） |
| `Graft` | V10 算子的原始输出，即按来源时间排列（C，等同实验 1 的 Graft） |

于是 `A→B` = 替换的净影响（删除 40 个载体槽 + 引入 40 个嫁接槽的**联合**效应），
`B→C` = 纯重排的影响（向量集合相同），`A→C` = 复核实验 1。

B 不是重新实现一遍压缩，而是 `graft_output.index_select(1, permutation)`；`permutation`
在 `plan_inplace_replace` 里断言是 `range(2304)` 的双射，并且**完整存进报告**
（`layout.permutation`，不只是摘要），所以跑完之后可以只凭 C 臂的输出把 B 复原出来，
不用再上 GPU。360 个嫁接槽**各自送给离它最近的、还有空位的载体**（按来源顺序贪心分配，
平局取较小载体号），再填进该载体升序的空位——这样每个载体拿到的是它自己邻域里被丢弃的内容，
也就是"原位替换"该有的样子（早先按来源顺序每载体 40 个的规则单调但不局部，实测会把 0 号块的
内容塞进 10 号载体的空位）。这条规则只决定"哪个外来向量进哪个块"，不决定记忆里有哪些向量。

```bash
# 先看会跑哪些题（只加载 tokenizer，不占 GPU、不写文件）
python tools/eval_replace_reorder.py --dry-run \
  --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
  --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
  --model-path /dev/shm/s1d_reader_fp16

# 不需要 GPU 的自检：置换、阶段映射、McNemar 统计
python tools/eval_replace_reorder.py --self-test

# 正式跑（四个臂共享同一次压缩；--verify-against 逐题比对一份已完成的报告）
python tools/eval_replace_reorder.py \
  --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
  --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
  --model-path /dev/shm/s1d_reader_fp16 \
  --checkpoint runs/natural_merge_v9/full_v10.pt \
  --verify-against reports/diagnostics/graft_span_hotpotqa.json \
  --cuda-device 0 --out reports/diagnostics/replace_reorder_hotpotqa.json

# 不跑模型，重新分组已有报告
python tools/eval_replace_reorder.py --analyze reports/diagnostics/replace_reorder_hotpotqa.json
```

口径说明：

- **最关键的指标是保护组的 Harmed 归因**：`summary.protection_group.harmed_stages` 把
  "Full 对、Selection 对、Graft 错"的题分成 `broken_at_replace`（B 就已经错，替换阶段损伤）
  和 `broken_at_reorder`（B 对、C 错，只有重排损伤）；`harmed_detail` 与 `transitions` 给出
  逐题明细（`selection` / `replace` / `graft` 三个 0/1）。
- 除新增的 B 组外，其余全部沿用实验 1：同一批 158 道有效题、冻结 Reader、单次 19→9、
  同样的 9 个载体、同样的 top-216 core、同样的 360 个嫁接槽、`operator_mode="native"`、
  不训练。`--verify-against` 会在跑之前就打开参照报告（路径写错不会等到跑完才炸），
  并且**双向**核对行集合：跑的是子集也会被判 `passed=false`，不会伪装成 100% 一致。
  若在 `Full`/`Selection`/`Graft` 三个臂上不是逐题一致，那是本驱动的 bug，不是结果。
- `A→B` 是**联合**效应，不能凭这一个差值再区分"删除损失"和"嫁接收益"，报告里
  `protocol.attribution_note` 写死了这句话。
- 显著性用**双侧精确 McNemar**，只对不一致对计数；`mcnemar_p` 就在每个 `pairs` 条目里。
- 筛选用了答案和证据位置，分组准确率是**机制诊断**，不是数据集成绩。

## 实验 3：Evict-Only —— 损伤到底来自"删槽"还是"插槽"

实验 2 已经证明保护组 13 题里有 4 题是在 `Replace` 阶段变错的，但"替换"同时做了两件事：
(i) 把每个载体自己的 40 个 Bottom 槽**删掉**了；(ii) 又把 360 个嫁接槽**插**进了腾出的位置。
这一个差值分不开这两件事，所以本实验只新增一个臂：

| 臂 | 做法 | 可读槽位 |
|---|---|---|
| `Full` | 19 块 4,864 槽，不压缩（参考） | 4,864 |
| `Selection` | 9 个载体块原样（A） | 2,304 |
| `Evict` | **和 A 完全相同的张量**，只把这 360 个槽从 Reader 的注意力里摘掉，不插入任何外来槽位（B） | 1,944 |
| `Replace` | 删掉同样的 360 个槽，再把 360 个嫁接槽填进腾出的位置（C，等同实验 2 的 B） | 2,304 |

于是 `A→B` = **纯删除**的影响（记忆里没有任何外来向量），`B→C` = **纯插入**的影响
（载体相同、位置相同，只是把 360 个嫁接向量加了进来并解除屏蔽），`A→C` = 复核实验 2。

B 臂不是"清空再重排"：它调用和 A 臂**同一个** `gr.install_selection`，然后清掉
`model.memory_active_mask` 里对应的 360 位。所以两件事被逐题硬校验：

- 逐题用 `torch.equal` 直接比对 A 臂和 B 臂装进池子里的那段窗口（形状、dtype、每一个 bit 都
  必须相同，结果记在 `eviction.mask.tensor_equal_to_selection`），证明 A、B 的张量一模一样，
  "掩码"是唯一变量。`fingerprints` 只当记录，**不当证据**：它是全池的稀疏均值，实验 2 里四个
  臂的指纹全都在 1e-4 之内，拿它当门禁等于用巧合做证明；
- 每个载体保留的 216 个槽仍在**原来的块内位置**，不存在重新打包这个隐藏变量。

为什么用 Mask 而不是补零或重排：Reader 通过 `memory_active_mask` 给非活跃槽位打
`finfo.min` 的注意力偏置，槽位等于不存在；补零但继续让它被 attend 到是另一种干扰，会把
"删槽"和"看见零向量"混在一起。掩码是**非连续**的（9 个块里各 40 个洞），而管理器在
`used` 变动时会自己重算掩码，所以掩码在 install **之后**才施加，并且生成之后再查一次
活跃槽位数（每题的 `eviction.mask.active_after` 必须是 1,944），否则说明掩码被覆盖了。

```bash
# 先看会跑哪些题（不占 GPU、不写文件）
python tools/eval_evict_only.py --dry-run \
  --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
  --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
  --model-path /dev/shm/s1d_reader_fp16

# 不需要 GPU 的自检：淘汰集合、掩码算术、阶段映射、McNemar 统计
python tools/eval_evict_only.py --self-test

# 正式跑（--verify-against 逐题比对一份已完成的报告）
python tools/eval_evict_only.py \
  --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
  --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
  --model-path /dev/shm/s1d_reader_fp16 \
  --checkpoint runs/natural_merge_v9/full_v10.pt \
  --verify-against reports/diagnostics/graft_span_hotpotqa.json \
  --cuda-device 0 --out reports/diagnostics/evict_only_hotpotqa.json

# 不跑模型，重新分组已有报告（含 --f1-threshold）
python tools/eval_evict_only.py --analyze reports/diagnostics/evict_only_hotpotqa.json
```

判据表（每题的 A/B/C 三个 0/1 组合，八种情形全部命名，不会漏掉任何一种）：

| 组合 | `summary` 里的名字 | 说明什么 |
|---|---|---|
| A=1 B=0 C=0 | `broken_at_evict` | 删槽就已经把这题弄坏了，外来槽位无辜 |
| A=1 B=1 C=0 | `broken_at_insert` | 删槽没事，插槽才弄坏 → 疑点在外来内容或它的位置分配 |
| A=1 B=0 C=1 | `broken_at_evict_recovered_at_insert` | 删槽伤了，嫁接槽又救回了可回答性 |
| A=0 B=0 C=1 | `fixed_at_insert` | 只有插入带来了净收益 |
| A=0 B=1 C=1 / A=0 B=1 C=0 | `fixed_at_evict` / `fixed_at_evict_broken_at_insert` | 删槽反而修好了（后者又被插槽弄坏） |
| A=1 B=1 C=1 / A=0 B=0 C=0 | `unchanged_correct` / `unchanged_wrong` | 什么都没发生 |

`summary.protection_group` 就是实验 2 那 13 题（`Full` 对、`Selection` 对、证据块在窗口内），
给出 `harmed_by_evict` / `harmed_by_insert` / `harmed_by_replace` 三个计数、`harmed_detail`
（每题 A/B/C 的 0/1 和 QA F1、`p_gold`）以及 `verdict` 一句话结论。若有 `--verify-against`，
还会核对这 13 个 id 和实验 2 报告里的保护组是否**完全一致**（`matches_reference`）。

口径说明：

- **容量不等**：B 读 1,944 个槽，A/C 读 2,304 个。这是故意的——它就是为了"只减不增"，
  所以 `A→B` 是**机制诊断**，不是等容量主表比较；B 只支持一个**否定**结论：某题在 B 就已经错，
  那外来槽位就不是肇事者。这句话写死在 `protocol.capacity_caveat` 里。
- 除新增的 B 臂外全部沿用实验 2：同一批 158 道题、冻结 Reader、单次 19→9、同样的 9 个载体、
  同样的 top-216 core、同样的 360 个嫁接槽、`operator_mode="native"`、不训练。
  `--verify-against` 在开跑前就打开参照报告，并**双向**核对行集合，跑子集会被判 `passed=false`。
- 淘汰集合是从 C 臂自己的 `permutation` 反推出来的（`evicted_window_positions`），不是另写
  一条规则，所以 B 删的 360 个位置和 C 插的 360 个位置**必然**是同一批，跑完还能用报告的
  `eviction.positions` 复算。
- 显著性仍是**双侧精确 McNemar**；筛选用了答案和证据位置，分组准确率是**机制诊断**，不是数据集成绩。

## 实验 1A：嫁接来源端的 Span-4 成组选择

实验 1、2、3 都固定在 V10 原来的来源选法上：把 10 个删除块的每个槽单独打分，取全局 top-360。
这 360 个槽落在哪就是哪，所以记忆里的嫁接侧是一把散点。这个实验**只改来源侧**：

| 臂 | 来源选择 | 输出 |
|---|---|---|
| `Full` | 不压缩，19 块 4,864 槽 | — |
| `Selection` | 9 个载体块原样 | 2,304 |
| `Graft` | V10 训练出来的规则：10 个删除块的**单点** leverage 全局 top-360 | 2,304 |
| `Span` | 同样这 10 个块，按**互不重叠的 4 槽 span 均值**排序，取 top-90 组 = 同样 360 槽，但落成 90 段连续的 4 连槽 | 2,304 |

其余全部冻死：同一批 158 题、冻结 Reader、同样 9 个载体、**原来的单点 top-216 core**、
360 个嫁接槽、2,304 输出槽、100% 原生复制、现有全局时间顺序、不训练。

一次前向里**跑两次压缩**（point、span4），两次都读同一个压缩前快照，所以 `Span − Graft`
只差"选了哪些来源槽"。驱动每题硬校验两件事：两次压缩的 **core 位置集合必须完全相同**
（证明只动了来源侧），两个嫁接位置集合必须**不同**（否则这个臂就等于 point 臂）。
`span_structure()` 再从 telemetry 反推：360 个嫁接位置必须正好构成 90 段完整的 4 连槽，
有一段不完整就报错——半截 span 说明算子选的不是它打分的组。

```bash
python tools/eval_graft_span.py --self-test          # 不需要 GPU
python tools/eval_graft_span.py --dry-run --data-dir ... --task hotpotqa \
  --supporting-facts-glob '...' --model-path /dev/shm/s1d_reader_fp16

python tools/eval_graft_span.py \
  --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
  --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
  --model-path /dev/shm/s1d_reader_fp16 \
  --checkpoint runs/natural_merge_v9/full_v10.pt \
  --verify-against <另一份已完成的 Full/Selection/Graft 报告> \
  --cuda-device 0 --out reports/diagnostics/graft_span_hotpotqa.json
```

看三个数：

- `summary.changed_group`：压缩会改变对错的**那 16 题**（实验 1/2 里是 11 救 5 伤），
  `rescued_by_span` / `harmed_by_span` 和 point 的 11/5 直接对照，`stages` 是八种情形的计数。
- `summary.dropped_evidence_group`：**40 道证据落在被删块里**的题，四个臂的准确率/F1 加
  `span_minus_point` 配对。
- `summary.net`：`rescued − harmed` 对 Selection 的净值，point 与 span 并排放（实验 1 报的 +6 就是
  这个量）。

`--verify-against` 在这里不是走过场：算子新加了 `graft_source` 分支，`Graft` 臂必须和已有报告
**逐题一致**，否则这个实验比较的就是两个不同实现。`operator_change` 写进了 protocol，
`operator_matches_checkpoint` 会是 `false`（`manifold_graft.py` 不在 checkpoint 的哈希锁定列表里）。

## 实验 1B：学习式多对一 core 融合 vs 原生嫁接

实验 1、2、3、1A 测的都是"怎么挑槽、怎么排槽"。这个实验测的是**删掉的槽能不能不丢**：
原生嫁接会删掉每个载体块 256 个槽里的 40 个（216 core + 40 嫁接），那 40 个槽里的东西
一旦被覆盖就没了。B 臂在删除**之前**，用一个小残差模块把这 40 个槽融合进**同一个块**
留下的 216 个槽，然后照原样删、照原样嫁接。

| 臂 | 压缩方式 | 输出槽 |
|---|---|---|
| `Full` | 不压缩，19 块 4,864 槽 | — |
| `Selection` | 9 个载体块原样 | 2,304 |
| `Graft` | V10 原生嫁接（A）：删 40、嫁 360，全部逐字复制 | 2,304 |
| `Fusion` | 同 A，但先把 40 个待删槽融合进本块的 216 个 core，再删再嫁接 | 2,304 |
| `Graft2` | A 的**第二轮**：把 A 自己的 9 块 + 后 10 块原生块拼成 19 块，再压一次 | 2,304 |
| `Fusion2` | B 的第二轮：从 B 自己的第一轮结果继续，再压一次 | 2,304 |

`Graft2` / `Fusion2` 只跑 `available_blocks ≥ 29` 的题（158 题里有 88 题）；两轮臂各自
延续**自己**的第一轮记忆，所以第二轮比的是"融合过的记忆再压一次还站不站得住"。

融合模块（`methods/prlmc/core_fusion.py`，rank 32 时 524,289 参数）只做一件事：每个 core 槽
对本块那 40 个待删槽做一次同块 cross-attention，取回一个残差。残差有三重约束（都来自 V10）：
`tanh` 把它压在本槽 RMS 的 `max_ratio` 倍以内、sigmoid 门再缩放一次、加完之后**恢复原槽 RMS**。
`out_proj` 零初始化，所以训练起点上 `Fusion` 与 `Graft` **逐位相同**——这一点在训练脚本里
开跑前实测一次（`identity_check`），不是写在注释里。

**不训练 Reader、不训练 Writer、不训练 V10 算子。** 训练只发生在 `train_core_fusion.py`：
在 musique（与评测的 HotpotQA 不同任务，196/200 篇够 19 块）上，把"压缩后窗口的 Reader 读出"
往"压缩前窗口的 Reader 读出"上拉，query 只取**问题**的最后 8 个 token
（`prompt_hidden_states` 只跑 prompt，不跑答案），所以目标里没有任何答案信息。

`--distill-layers` 决定监督哪几层：

| 值 | 含义 | 结果 |
|---|---|---|
| `all`（默认） | 32 层全监督 | **发散**：验证集 −28%，前 13 层读出距离被放大 1.1–4.8 倍 |
| `v10` | 只监督 8/16/24/31 | 收敛到 +14.5%；158 题上 taps +20.1%、其余 28 层 −10.6%，**准确率比 `Graft` 低 2.53 pp** |

发散不是实现 bug，是目标函数的问题：损失是各层距离的**绝对**平均，而前几层的距离本来只有
0.001 量级，把它搞坏几乎不花钱，后几层距离大、改好一点就赚回来 —— 优化器于是拿前几层去换
后几层。要修就得给每层一个自己的开关，或者把损失改成逐层相对量；两者都还没做。

更值得记住的是 `v10` 那一行：读出距离确实被优化了 20%，准确率却反着走。**这个目标不等于
"答对"**，所以调模块之前得先换目标。逐层与准确率数字见
[docs/03-results.md](../docs/03-results.md) 第 4 节。

```bash
# 1) 训练融合模块（约 3.5 s/更新，整轮 384 更新约 30 min）
python tools/train_core_fusion.py --dry-run \
  --data-dir /data/.../LongBench/extracted/data --train-task musique

python tools/train_core_fusion.py --updates 384 --distill-layers all \
  --data-dir /data/.../LongBench/extracted/data --train-task musique \
  --model-path /dev/shm/s1d_reader_fp16 \
  --checkpoint runs/natural_merge_v9/full_v10.pt \
  --out runs/core_fusion/fusion_all32.pt \
  --report runs/core_fusion/fusion_all32.json

# 2) 评测
python tools/eval_learned_fusion.py --self-test          # 不需要 GPU

python tools/eval_learned_fusion.py \
  --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
  --supporting-facts-glob '/data/.../hotpot_qa/hotpot_dev_distractor_v1.parquet' \
  --model-path /dev/shm/s1d_reader_fp16 \
  --checkpoint runs/natural_merge_v9/full_v10.pt \
  --fusion runs/core_fusion/fusion_all32.pt \
  --verify-against reports/diagnostics/graft_span_hotpotqa.json \
  --cuda-device 0 --out reports/diagnostics/learned_fusion_hotpotqa.json
```

`--per-layer` 把读出距离从 8 个探针层扩到 32 层，结果进 `summary.readout.per_layer`
（每层的 graft / fusion 距离与 `removed`）。8 层平均会掩盖"拿一组层换另一组层"的模块，
这个表就是用来暴露它的；它只用于诊断，2 题就够，不要拿它当准确率证据。

驱动每题硬校验"只动了 core"这件事，而不是在文档里声明：

- `Fusion` 的 360 个嫁接槽与 `Graft` 的**逐位相同**（`torch.equal`，360 列）；
- core / 嫁接的**位置**和 360 配额不变，每个载体**正好删 40 个**，且这 40 个不在输出里；
- core 确实变了（否则这个臂什么都没测）；
- 两臂装进 Reader 后的 `active_fingerprint` 不同（证明真的换了池子）。

看四个数：

- `summary.damage`：`Full` 答对的题里各臂丢了几个——B 相对 A 少丢几个，就是"原有知识损伤是否减少"。
- `summary.multi_round_group`：88 题上的 `Fusion2 − Graft2`、`Fusion2 − Fusion`、`Graft2 − Graft`。
- `summary.pairs.Fusion_minus_Graft`：配对准确率、救回/弄坏、精确 McNemar。
- `summary.readout`：读出距离。`fusion_removes` 是**监督过的层**上融合消掉了多少距离，
  `fusion_removes_held_out` 是**从没优化过的 4/12/20/28 层**。前者降后者不降 = 只拟合了探针；
  两个都不降 = 没学到东西。这个数和准确率无关，是机制层面的证据。
  用 `--distill-layers all` 训出来的 checkpoint 没有 held-out 层，报告会把
  `held_out` 与 `fusion_removes_held_out` 置空并在 `note` 里说明——这时候只有准确率算数。

`--verify-against` 在这里同样不是走过场：`Full` / `Selection` / `Graft` 三臂必须和 1A 的报告
逐题一致（1A 之后 `manifold_graft.py` 已与 checkpoint 清单不同，`operator_matches_checkpoint`
会是 `false`，靠这个比对兜底）。

## 实验 2：来源端浓缩 vs 原生嫁接

实验 1、2、3、1A 测"怎么挑槽、怎么排槽"，1B 测"删掉的槽能不能先融进 core"。实验 2 换一头：
原生嫁接从被扔掉的 10 个块（2,560 槽）里挑 360 个原样复制，**剩下 2,200 个槽彻底浪费**；
实验 2 让这 360 个锚点各自从**自己来源块里未被选中的槽**聚合一个残差。

| 臂 | 压缩方式 | 输出槽 |
|---|---|---|
| `Full` | 不压缩，19 块 4,864 槽 | — |
| `Selection` | 9 个载体块原样 | 2,304 |
| `Graft` | V10 原生嫁接：1,944 core + 360 逐字复制 | 2,304 |
| `Fusion` | 同 `Graft`，但 360 个嫁接槽 = 原生锚点 + 学习式残差 | 2,304 |

**容量、位置、顺序、配额全不变，core 逐字节原样**——与 1B 正好互为镜像（1B 动 core、不动嫁接）。

模块（`methods/prlmc/source_compaction.py`，rank 32 时 524,289 参数）只做一件事：每个锚点对
**本块**未被选中的槽做一次 cross-attention，取回一个残差。残差三重约束（同 1B）：`tanh` 压在
锚点 RMS 的 `max_ratio` 倍以内、sigmoid 门再缩放、加完**恢复原 RMS**。`out_proj` 零初始化，
所以训练起点上 `Fusion` 与 `Graft` **逐位相同**（训练脚本开跑前实测 `window_bit_identical`）。

```bash
# 1) 训练（384 步约 45 min，6 s/步，峰值 33 GB）
python tools/train_source_compaction.py --dry-run \
  --data-dir /data/.../LongBench/extracted/data --train-task musique

python tools/train_source_compaction.py --updates 384 --lr 1e-3 --protect-weight 0.1 \
  --data-dir /data/.../LongBench/extracted/data --train-task musique \
  --model-path /dev/shm/s1d_reader_fp16 \
  --checkpoint runs/natural_merge_v9/full_v10.pt \
  --out runs/source_compaction/compaction_rank32.pt \
  --report runs/source_compaction/compaction_rank32.json

# 2) 随机对照（同结构、同残差幅度、不训练）
python tools/randomize_compaction.py \
  --data-dir /data/.../LongBench/extracted/data --train-task musique \
  --model-path /dev/shm/s1d_reader_fp16 \
  --v10-checkpoint runs/natural_merge_v9/full_v10.pt \
  --trained runs/source_compaction/compaction_rank32.pt \
  --out runs/source_compaction/random_rank32.pt

# 3) 评测（158 题约 39 min）
python tools/eval_source_compaction.py --self-test        # 不需要 GPU
python tools/eval_source_compaction.py \
  --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
  --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
  --model-path /dev/shm/s1d_reader_fp16 \
  --checkpoint runs/natural_merge_v9/full_v10.pt \
  --compaction runs/source_compaction/compaction_rank32.pt \
  --verify-against reports/diagnostics/graft_span_hotpotqa.json \
  --cuda-device 0 --out reports/diagnostics/source_compaction_hotpotqa.json
```

训练端有两处**必须**按顺序做，否则要么不收敛、要么 OOM（细节见
[docs/04-diagnostics.md](../docs/04-diagnostics.md) 第 3 节）：

1. **先把所有 `nn.Dropout` 的 `p` 置 0，再 `model.train()`**。reader 有 384 个 p=0.1 的
   `lora_dropout`，`tools/train.py` 的守卫只查 `attention_dropout`，漏了它们；不置零的话
   train 模式的前向和评测时不一致，而且不确定。
2. **然后开梯度检查点**。反传要物化 3.12 GiB 的全宽记忆梯度，不检查点在 40 GB 卡上装不下。

梯度路径不是假设出来的：第 1 步只有 `out_proj` 该有梯度（残差还是零），第 2 步四个投影必须
全部有梯度，脚本会在那里直接报错而不是安静地训练一个被 detach 的模块。

驱动硬校验的是**"只动了嫁接槽"**：

- `Fusion` 的 1,944 个 core 槽与 `Graft` **逐位相同**（`torch.equal`）；
- 360 个嫁接槽确实变了，且每个锚点的候选只来自它**自己那一块**、且都不是锚点；
- 两臂装进 Reader 后的 `active_fingerprint` 不同（证明真的换了池子）。

`--verify-against` 同样要求 `Full` / `Selection` / `Graft` 三臂与 1A 逐题一致——2026-11-06 那
两次 158 题运行都是 158/158 一致。

随机对照是这个实验的命门，`randomize_compaction.py` 的标定逻辑：用训练 checkpoint 在**一篇真实
文档**上测出残差幅度（0.2636），然后对随机 `out_proj` 做二分缩放直到幅度相同（实测误差 3e-8）。
门的初值被开到 0.6（上限 0.30）只是为了让二分能取到目标值，方向仍然是随机的。

### 反事实消融：收益到底靠不靠那 2,200 个槽位

`eval_source_compaction.py` 只能回答"加上候选有没有用"，回答不了"是候选里的**知识**有用，还是
随便动一动锚点就有用"。`tools/eval_counterfactual_candidates.py` 做的是后者：同一份权重、同一批
158 题、同一套锚点，**不重训**，只把每个锚点读的候选槽位换成**另一篇文档**丢弃块里同样数量的
槽位（评测顺序上的上一行；两个窗口都正好丢 10 块，所以第 k 个丢弃块对第 k 个丢弃块）。

同一进程里五臂全跑，所以每个对比逐题配对，而且四臂必须与实验 2 的报告逐题一致——`--verify-against`
除了 `Full`/`Selection`/`Graft`，还额外校验 `Fusion` 臂，因为它是反事实要减掉的那一臂。

```bash
python tools/eval_counterfactual_candidates.py --self-test      # 不需要 GPU

python tools/eval_counterfactual_candidates.py \
  --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
  --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
  --model-path /dev/shm/s1d_reader_fp16 \
  --checkpoint runs/natural_merge_v9/full_v10.pt \
  --compaction runs/source_compaction/compaction_rank32.pt \
  --verify-against reports/exp2_source_compaction/eval_trained_hotpotqa.json \
  --cuda-device 0 --out reports/diagnostics/counterfactual_hotpotqa.json
```

结果：反事实 **0 救 0 伤**，158 题 QA F1 逐题相同（见
[docs/03-results.md](../docs/03-results.md) 第 6 节）。所以还得再问一句"替换到底有没有进到模块里"，
这就是第二个工具。

`tools/probe_compaction_sensitivity.py` 不跑 reader 生成，只在 7 篇真实文档上把同一次替换逐级拆开
（候选 → `context` → `raw` → 残差 → 窗口），并记录注意力熵/最大权重与 `tanh` 饱和比例。它的
`shuffled` 臂是**接线检查**而不是敏感性测量：一行候选同时带着自己的 key 和 value，行序置换会一起
搬走，输出按构造完全不变（实测 0.0）；不变才说明探针接对了。

```bash
python tools/probe_compaction_sensitivity.py \
  --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
  --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
  --model-path /dev/shm/s1d_reader_fp16 \
  --checkpoint runs/natural_merge_v9/full_v10.pt \
  --compaction runs/source_compaction/compaction_rank32.pt \
  --cuda-device 0 --max-samples 8 \
  --out reports/exp2_source_compaction/probe_candidate_sensitivity.json
```

衰减链：候选 130% → `context` 19% → `raw` 16% → 残差 5.0% → 窗口 1.5% → 答案 0。

## Anchor-Only：把候选路径删掉，还剩多少

反事实消融证明"换掉候选没用"，但没回答"能不能不要候选"。这个实验把整条候选路径删掉：

| 臂 | 压缩方式 | 输出槽 |
|---|---|---|
| `Full` | 不压缩，19 块 4,864 槽 | — |
| `Selection` | 9 个载体块原样 | 2,304 |
| `Graft` | V10 原生嫁接：1,944 core + 360 逐字复制 | 2,304 |
| `Fusion` | 同 `Graft`，360 个嫁接槽 = 锚点 + 读本块剩余槽的残差 | 2,304 |
| `AnchorOnly` | 同 `Graft`，360 个嫁接槽 = 锚点 + **只看锚点自己**的残差 | 2,304 |

模块（`methods/prlmc/anchor_residual.py`）是 `hidden -> rank -> hidden` 的 gelu 瓶颈，
`forward(anchors)` **只有一个参数**——候选路径不是"读了但没用"，是代码里没有这条路。rank 取 64
是为了让参数量与实验 2 的模块**逐位相同**：

| | 结构 | 参数量 |
| --- | --- | --- |
| `SourceCompaction` | 4 × (32 × 4096) + 1 | 524,289 |
| `AnchorResidual` | 2 × (64 × 4096) + 1 | 524,289 |

两个 run 走同一个训练脚本，所以数据、384 步、学习率、损失、种子、梯度检查都是同一段代码：

```bash
# 训练（384 步约 44 min，6.1 s/步，峰值 29.9 GB）
python tools/train_source_compaction.py --variant anchor-only --smoke \
  --data-dir /data/.../LongBench/extracted/data --train-task musique \
  --model-path /dev/shm/s1d_reader_fp16 \
  --checkpoint runs/natural_merge_v9/full_v10.pt \
  --out runs/anchor_residual/smoke.pt --report runs/anchor_residual/smoke.json

python tools/train_source_compaction.py --variant anchor-only --updates 384 \
  --lr 1e-3 --protect-weight 0.1 \
  --data-dir /data/.../LongBench/extracted/data --train-task musique \
  --model-path /dev/shm/s1d_reader_fp16 \
  --checkpoint runs/natural_merge_v9/full_v10.pt \
  --out runs/anchor_residual/anchor_only_rank64.pt \
  --report runs/anchor_residual/anchor_only_rank64.json

# 评测（158 题约 52 min，五臂同进程）
python tools/eval_anchor_residual.py --self-test        # 不需要 GPU
python tools/eval_anchor_residual.py \
  --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
  --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
  --model-path /dev/shm/s1d_reader_fp16 \
  --checkpoint runs/natural_merge_v9/full_v10.pt \
  --compaction runs/source_compaction/compaction_rank32.pt \
  --anchor runs/anchor_residual/anchor_only_rank64.pt \
  --verify-against reports/exp2_source_compaction/eval_trained_hotpotqa.json \
  --cuda-device 0 --out reports/diagnostics/anchor_only_hotpotqa.json
```

结果：`AnchorOnly` **56/158（35.44%）** vs `Fusion` 57/158（36.08%），6 救 7 伤、净 −1、p = 1.000；
对 `Graft` 23 救 8 伤（p = 0.011）。详见 [docs/03-results.md](../docs/03-results.md) 第 7 节。

驱动硬校验四件事：

- 四条老臂与实验 2 的报告**逐题一致**（`--verify-against` 额外校验 `Fusion` 臂）；
- `AnchorOnly` 的 1,944 个 core 槽与 `Graft` 逐位相同，360 个嫁接槽确实变了；
- 模块自报 `candidate_slots == 0`，否则直接报错（路径没删干净就不许跑）；
- 两个 checkpoint 的参数量必须相等，否则拒绝比较（否则容量和架构混在一起）。

## 实验 8（V12）：桥接路由 + 动态容量分配

这一轮**不训练任何参数**，也不加载任何 checkpoint（除冻结算子与 AnchorOnly）。窗口由
`methods/prlmc/bridge_routing.py` 里的两个 router 现场构造，共用一次 Reader 读数：

| 臂 | 窗口 |
| --- | --- |
| `V11NativeGraft` | 每载体块效用前 180 逐字 + 684 个被删块槽（V11 的固定配额，去掉校准） |
| `Bridge` | 每载体块保底 140 + 1,044 个名额由 3,604 个候选全局竞争（效用 + 桥接分数，MMR） |

两个窗口都是**原生搬运**：装进去的每个槽位都是源槽位的逐位拷贝，驱动用 `torch.equal` 校验。

```bash
# 先看会跑哪些题、几个臂（只读 tokenizer，不占 GPU）
python tools/eval_bridge_routing.py --dry-run   --data-dir /data/.../LongBench/extracted/data --model-path /dev/shm/s1d_reader_fp16

# 正式跑（六臂同进程；前四条臂逐题复现参考报告，否则整轮作废）
python tools/eval_bridge_routing.py   --data-dir /data/.../LongBench/extracted/data --task hotpotqa   --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet'   --model-path /dev/shm/s1d_reader_fp16   --checkpoint runs/natural_merge_v9/full_v10.pt   --anchor runs/anchor_residual/anchor_only_rank64.pt   --verify-against reports/anchor_fusion/eval_anchor_fusion_hotpotqa.json   --cuda-device 0 --out reports/bridge_routing/eval_v12_hotpotqa.json
```

参数都是显式的、落在报告 `protocol` 里：`--protected-per-block 140`（保底）、
`--seeds-per-block 8`（桥接 seed）、`--bridge-weight 0.5`（效用与桥接的凸混合权重，
置 0 即"同分配、不桥接"）、`--core-per-block/--graft-per-block`（臂 B 的固定配额）、
`--probe-task musique --probe-count 16`（Reader 效用与桥接分数共用的固定探针库）。

四条硬校验：每条臂恰好 2,304 槽；全局时间序且源位置唯一；每个槽位逐位等于它声称的来源槽；
臂 C 的保底恰好 9 × 140 且每个 protected 槽来自自己的载体块、`admitted_from_dropped +
admitted_from_carriers == 1,044`。报告还会给出分配诊断（每个被删块实际进多少、有多少载体槽
被反抢回来、桥接分数与效用的相关系数、`admitted` 与 `rejected` 的平均桥接分数）。

判定线三条，写进 `summary.decision`：`Bridge − V11NativeGraft ≥ +5` 且损伤不超过基线 →
路由站得住；救回多于改错但净收益不到 5 → 先改保底值；持平或更差 → 停。

注意：本轮没有 donor 对照臂，所以正结果只能归因到"保底 + 全局竞争 + 桥接项"这一整包
（报告 `protocol.known_limits` 里也写着）。
