# LongBench：MemoryLLM-8B 原生 50-block 评测

本实验只评测当前 MemoryLLM-8B checkpoint 的原生 50-block 容量，不运行 20-block 对照，也不使用 PRLM-C Selection 或共享融合模块。

## 固定协议

- 数据：`/share/latent/benchmarks/LongBench/data` 的官方 `test` 文件。
- 任务：HotpotQA、2WikiMQA、MuSiQue、Qasper、MultiFieldQA-en、NarrativeQA。
- 输入长度：分别运行 8,192 和 16,384 tokens。
- 每个样本：前 `total_tokens - 2,048` tokens 按最多 512 tokens 写入；最后 2,048 tokens 直接作为 query。
- 每次写入：MemoryLLM 原生产生 256 latent tokens，并使用 checkpoint 自带随机淘汰。
- 生成：greedy，`do_sample=False`、`num_beams=1`；生成长度依任务为 32/64/128 tokens。
- 每个样本开始前恢复 checkpoint 初始 memory；不训练、不创建 optimizer。
- 主指标：LongBench 官方 QA F1，逐题结果完整保存。

入口：`tools/longbench_memoryllm_eval.py`。`--capacity-blocks` 只接受 `50`，用于防止误跑其他容量协议。

示例（仅用于用户下令后的执行）：

```bash
python tools/longbench_memoryllm_eval.py \
  --data-dir /share/latent/benchmarks/LongBench/data \
  --model-path /share/latent/MemoryLLM-8B \
  --total-tokens 8192 --capacity-blocks 50 --cuda-device 0 \
  --out runs/longbench_memoryllm_native50_8k.json
```

仓库中固定了官方 LongBench prompt 和 QA F1 实现对应的任务规则；实际运行前应记录 LongBench data zip SHA256、模型 config/tokenizer SHA256、GPU、PyTorch/Transformers 版本和输入截断统计。
