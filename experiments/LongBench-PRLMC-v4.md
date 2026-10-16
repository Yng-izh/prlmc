# LongBench：PRLM-C 自然记忆融合 v4（Dsup）评测

本实验在 LongBench 上评测 PRLM-C 自然记忆融合方案：20-block 托管预算内的
19→9 块 slot 融合（Dsup，`grouped_gated_sparse_slot_v4`），与 native50
基线使用完全相同的输入协议与生成协议。

## 固定协议

- 与 `LongBench-MemoryLLM-8B-native50.md` 相同的部分：
  - 数据：`/share/latent/benchmarks/LongBench/data` 官方 `test` 文件；
    任务 HotpotQA、2WikiMQA、MuSiQue、Qasper、MultiFieldQA-en、NarrativeQA。
  - 输入长度 8,192 / 16,384 tokens；前 `total-2,048` 按最多 512 tokens 写入，
    最后 2,048 tokens 直接作为 query；greedy 解码；max_new_tokens 32/64/128；
    尾部截断；每个样本重置 seed 0 与初始 memory；不训练、无 optimizer。
- 方法侧差异：
  - 写入路径：`inject_memory(update_memory=False)` 取 delta，由
    `ProtectedResidualMemory` 管理器放置；物理池保持 50-block（12,800 slots），
    工作预算 20 blocks（5,120 slots），free region 由 active mask 隐藏。
  - 压缩：写入将超出 20-block 预算前，把最旧 19 blocks（4,864 slots）融合为
    9 blocks（2,304 slots）；融合模块 = Dsup checkpoint
    `runs/natural_merge_v4_dd_20261015_0108/Dsup.train.pt`
    （sha256 `70733390…`，局部门控上限 0.20，全局 RMS 预算 0.05）。
  - 无原生随机淘汰（`drop_memory` 路径不触发）。

## 对照臂

`--arm Selection`：同一预算/窗口/目标，用纯 importance-weighted 块选择器
替换融合模块，即 v4 开发流中 SharedMerge vs Selection 的同轴对照。

## 评分

每题同时记录：

- `official_qa_f1`：原始输出对官方 QA F1（与 native50 报告同口径，可比）。
- `official_qa_f1_truncated`：第一个 `\n` 前截断后的 QA F1（官方 LongBench
  后处理口径）。MemoryLLM 解码器会在答案后继续生成
  `\n\nQuestion: ...`，raw F1 被该解码伪影压低，truncated 分数才反映记忆质量。

汇总 `summary[task].qa_f1` 以 truncated 口径为主指标，`qa_f1_raw` 附带。

## 入口

`tools/longbench_prlmc_eval.py`：

```bash
python tools/longbench_prlmc_eval.py \
  --data-dir /share/latent/benchmarks/LongBench/data \
  --out runs/longbench_prlmc_v4_dsup_8k.json \
  --total-tokens 8192 --cuda-device 0            # 方法臂
python tools/longbench_prlmc_eval.py \
  --data-dir /share/latent/benchmarks/LongBench/data \
  --out runs/longbench_prlmc_v4_selection_8k.json \
  --total-tokens 8192 --cuda-device 0 --arm Selection   # 对照臂
```

运行前记录：checkpoint 与全部 SOURCE_FILES 的 sha256（自动写入
`identity.source_manifest`）、LongBench data zip SHA、tokenizer SHA、
PyTorch/Transformers 版本、每题压缩事件（次数、selected_blocks、融合 γ）。
