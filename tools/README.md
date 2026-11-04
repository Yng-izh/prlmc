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

# 不跑模型，重新分组已有报告
python tools/eval_graft_recovery.py --analyze reports/diagnostics/graft_recovery_hotpotqa.json
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
