# 06 — 论文主表与统一评测协议（冻结稿）

状态：协议冻结，数值待测。本文件只登记"要跑什么、怎么记"，不登记尚未跑出的数字。

## 1. 主表（Table 1）

Table 1. Performance under a Unified Historical Memory Budget

| Method | LongBench 6-QA F1 | LooGLE Short / Long QA | InfiniteBench En.QA / En.MC | LongBench v2 Medium Acc |
| --- | --- | --- | --- | --- |
| MemoryLLM-8B | 待测 | 待测 | 待测 | **18.7** |
| M+ | 待测 | 待测 | 待测 | 待测 |
| StreamingLLM | 待测 | 待测 | 待测 | 待测 |
| KIVI | 待测 | 待测 | 待测 | 待测 |
| ReST-KV | 待测 | 待测 | 待测 | 待测 |
| PRLM-C (Ours) | 待测 | 待测 | 待测 | 待测 |

定位：LongBench 6-QA 为常规能力对照；LooGLE、InfiniteBench、LongBench v2 为超预算、多轮融合的主验证。
各 benchmark 的单项任务分数进详细表，不做跨任务宏平均。

### 1.1 Baseline 定位与竞争强度（冻结）

| Baseline | 论文与年份 | 方法定位 | 对我们论文的竞争强度 |
| --- | --- | --- | --- |
| MemoryLLM | ICML 2024 | 原生 latent memory，历史记忆持续更新 | 必须保留的基础对照 |
| M+ | ICML 2025 | Latent memory + 长期记忆检索 | 强，直接竞争者 |
| StreamingLLM | ICLR 2024 | Attention sinks + 滑动窗口 | 经典，但长程知识保留较弱 |
| KIVI | ICML 2024 | 2-bit KV Cache 量化 | 强，属于另一条技术路线 |
| ReST-KV | ICLR 2026 | 基于输出重构的 KV Cache 淘汰 | 强，较新的竞争者 |

H₂O 已从主表移出（改为 StreamingLLM）；H₂O 只剩两条可用实现路径，且都依赖 `flash_attn`，与本轮容量口径的匹配度也低于上述五个。

## 2. 内部对照副表（证明融合增益的唯一依据）

| 方法 | 想验证的内容 |
| --- | --- |
| Native MemoryLLM | 原生记忆容量与淘汰能力 |
| Recent-only | 仅保留近期记忆是否足够 |
| Selection | 有效选块本身的收益 |
| Selection with V9 carriers | V9 选块策略的收益 |
| PRLM-C V9 | 被删除块 slot 融合的额外收益 |

硬约束：同一 reader、同一预算、同一问题、同一选块结果。只有 V9 − Selection 的差值才能归因于跨块融合。
多轮融合题单独汇报，避免被全体平均掩盖。

## 3. 统一容量与调度

| 项 | 值 |
| --- | --- |
| 主预算 | 50 blocks |
| latent slots | 50 × 256 = 12,800 |
| 记忆表示 | 32 layers × 4096 dims × FP16 |
| 历史记忆预算 | 3.125 GiB |
| 每次 native write | 512 tokens |
| 压缩触发 | 达到 50-block 容量后、下一次写入前 |
| Reader | 冻结 MemoryLLM-8B，测试时完全不更新 |
| 问题输入 | 文档流式处理结束后再给问题 |
| 结果记录 | 官方评分 + 逐题记录 + 压缩次数 + 资源统计 |

50 blocks 需超过 25,600 写入 tokens 才产生第一次容量压缩；不足此长度的输入不得写成"验证了 50-block 融合"。
外部 KV 方法按真实 K/V 占用 + 元数据计容量；M+ 需计入全部持久化 LTM 与索引。做不到等预算的进"非等预算参考表"。

## 4. 各 benchmark 官方任务与指标

| Benchmark | 任务 | 指标 | 官方来源 |
| --- | --- | --- | --- |
| LongBench | hotpotqa / 2wikimqa / musique / qasper / multifieldqa_en / narrativeqa | QA F1（首行截断后） | THUDM/LongBench @ 2e00731 |
| LooGLE | shortdep_qa / longdep_qa | ROUGE-L（官方自动指标） | bigai-nlco/LooGLE |
| InfiniteBench | longbook_qa_eng (En.QA) / longbook_choice_eng (En.MC) | QA F1 / Accuracy | OpenBMB/InfiniteBench |
| LongBench v2 | Medium 组 | Accuracy（官方 `The correct answer is (X)` 抽取后判分） | THUDM/LongBench @ 2e00731 |

## 5. 数据落盘位置（服务器）

| 数据 | 路径 | 大小 |
| --- | --- | --- |
| LongBench v1 | `datasets/LongBench/extracted/data/*.jsonl` | 已解压 |
| LongBench v2 | `datasets/LongBench-v2/data.json` | 465,490,535 B |
| LooGLE | `datasets/LooGLE/shortdep_qa.jsonl`, `longdep_qa.jsonl` | 199,281,301 / 134,979,019 B |
| InfiniteBench | `datasets/InfiniteBench/longbook_qa_eng.jsonl`, `longbook_choice_eng.jsonl` | 298,297,185 / 185,904,631 B |

## 6. 已知偏差登记

| 项 | 官方 | 本仓库 | 处理 |
| --- | --- | --- | --- |
| 输入预算 | 120,000 tokens，中间截断 | 16,384 tokens（8192/16384/32768 可选），中间截断 | 在 protocol 块显式记录，不宣称等价官方设置 |
| 解码 | temperature 0.1 | greedy，seed 0 | 记录为确定性偏差，所有方法一致 |
| 停止符 | 128 tokens | 128 tokens + 换行 EOS | 记录 |
| 压缩几何 | 无（原生淘汰） | 20-block 开发预算 / 50-block 主预算 | 训练与评测几何必须一致后才可进主表 |

## 7. 已产出的主表格子

### MemoryLLM-8B × LongBench v2（全 503 题，128k 预算）

| 项 | 值 |
| --- | --- |
| Overall | 18.7 |
| Easy (n=192) | 21.4 |
| Hard (n=311) | 17.0 |
| Short (n=180) | 15.0 |
| Medium (n=215) | 21.4 |
| Long (n=108) | 19.4 |

运行条件：`--length all --total-tokens 128000 --tail-tokens 2048`，greedy，seed 0，
native `inject_memory(update_memory=True)`，不加载任何 checkpoint，不清空初始残差池。
503/503 题完成，耗时 4.95 h（单卡 A100-PCIE-40GB），峰值显存 22.0 GiB。
长文档每题写入 243 个 512-token chunk = 覆盖 62,208 slot ≈ 内存绕 4.9 圈，多轮淘汰被真实触发。
逐题记录：`reports/main_table/memoryllm8b_lbv2_all_128k.json`。

解读限制：原生 reader 在超长上下文下大量输出固定字母（`A` 占绝对多数），该行是 Native 下限，
不能单独作为"融合无效"的证据；它的作用是内部对照副表中被比较的基线。

## 8. 进度

- [x] 四套数据落盘
- [x] LongBench v2 评测入口（`tools/longbench_v2_eval.py`，官方 prompt/抽取/判分）
- [x] MemoryLLM-8B × LongBench v2 全组（18.7 Overall / 21.4 Medium）
- [ ] LooGLE、InfiniteBench 的 MemoryLLM-8B 基线
- [ ] 20-block 几何对齐后的 V9 vs Selection 关键对照
- [ ] 50-block 主线（LooGLE + InfiniteBench）
- [ ] 五个外部 baseline + 完整主表
