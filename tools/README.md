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

# 正式跑（四个臂共享同一次压缩；--verify-against 逐题比对实验 1 的答案）
python tools/eval_replace_reorder.py \
  --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
  --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
  --model-path /dev/shm/s1d_reader_fp16 \
  --checkpoint runs/natural_merge_v9/full_v10.pt \
  --verify-against reports/diagnostics/graft_recovery_hotpotqa.json \
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

# 正式跑（--verify-against 逐题比对实验 2 的 Full/Selection/Replace）
python tools/eval_evict_only.py \
  --data-dir /data/.../LongBench/extracted/data --task hotpotqa \
  --supporting-facts-glob '/data/.../hotpot_qa/distractor/validation-*.parquet' \
  --model-path /dev/shm/s1d_reader_fp16 \
  --checkpoint runs/natural_merge_v9/full_v10.pt \
  --verify-against reports/diagnostics/replace_reorder_hotpotqa.json \
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
