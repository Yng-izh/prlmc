# Native-QA — 原生问答压缩前后成对对照

状态：carrier residual、full-rewrite 两组 GPU 评测已完成；原生问题功能 warmup 正在运行。

## 目标与数据

直接使用 Conflict_Resolution parquet 的原始 context、questions、answers 和
metadata.qa_pair_ids。已通过只读数据结构检查核实这些字段，answers 为逐题
答案别名数组。保留原始事实序号、说明和顺序，不生成新问题，不删除冲突事实。

## 驱动

`tools/native_qa_compare.py` 是独立入口，不导入有自动选卡副作用的 S0 驱动。
使用基准 `utils.templates.get_template` 的 factconsolidation memorization/query
模板。context 按包含模板后的 token 预算无损分块，各块拼接等于原文；查询
不截断，不拼接 gold 答案。生成使用 eval、inference_mode 和 do_sample=False。

严格对照流程：

1. 写入全部 context，禁用自动压缩，强制 append；检查每次写入。
2. 对所有原生问题生成完整记忆答案，保存逐题预测和答案别名。
3. 在同一已写入状态上做一次手动压缩；最新块与 anchor 按现有 manager 契约保护。
4. 使用完全相同的问题、模板、生成预算和指标评测压缩记忆。
5. 汇总 normalized alias EM、lost、gained、retained、net_change；保存实际释放
   槽数、压缩报告和压缩前后 memory stats。请求 ratio 是被选压缩窗口的比例，
   不是整个池的最终比例；实际槽数以报告为准。

select_blocks 是默认压缩对照。carrier_residual 必须显式给训练后的
--carrier-ckpt，避免 up=0 初始化冒充已训练候选。该脚本仅评测，不训练或
使用答案选择压缩方向。可显式 --limit-questions 限制原生题目的前缀子集，
报告总题数与实际题数；默认使用全部题目。

## 完整记忆的容量门

row 3 的完整 context 实测约 1,118,123 字符。固定宽度 MemoryLLM 池不应被
默认视为能容纳全部 append-only 写入。脚本先按每块最大占一 block 做保守
容量检查；超出时写 status=capacity_exceeded，退出码为 2，不输出准确率。
该上界可能偏保守，报告明确标注 conservative admission。

任何实际 dropped write、驱逐、自动压缩都会使完整记忆臂失效，结果记录为
invalid_full_memory。压缩未释放槽则记录 compaction_skipped。只有 completed
状态带成对 summary。完整记忆在这里指“所有 context 块均写入并保留，尚未
显式压缩”，不是“编码没有信息损失”的宣称。

因此：此驱动适用于能够完整装入固定池的原生 episode。长 context 的正式
评测还需决定独立的容量方案（例如可验证的扩展池，或明确标成有驱逐的原生
流式基线）；本轮不自动改池宽、不截取 context、不把流式基线称为完整记忆。

## 指标边界

最终报告统一使用 MemoryAgentBench 官方 `default_post_process` 规则：同时比较原始生成文本与 `Answer:` 解析结果，经过 `normalize_answer` 后计算 EM，并计算 gold 是否出现在预测中的 SUB。驱动保存逐题原始输出，所有结果可离线重算。

驱动内的 `normalized_alias_exact_match` 只作为运行时占位指标，不能用于跨臂比较；本轮已按官方规则重新计算。

## 已完成基线

| row | full-rewrite 原生基座 | 我们的 full-memory | carrier residual 压缩后 |
|---|---:|---:|---:|
| 0 | 7 EM / 7 SUB | 5 / 7 | 6 / 8 |
| 4 | 13 EM / 13 SUB | 30 / 34 | 31 / 34 |

`full-rewrite` 每次写入都替换整个 working pool，因此不是“无压缩 append-only”上界；它是 MemoryLLM 原生整池 rewrite 对照。row 0/4 使用完全相同的 context，但问题集合没有重叠。

## Native-QA 功能 warmup（已完成，2026-10-03）

方案：row 0 的 24 道原生问题做 gold+KD 监督，训练 carrier residual 96 步
Stage B（实际压缩形状 6→3 blocks），checkpoint 存
`runs/native_qa_functional_warmup.pt`。评测在同一 context、问题零重叠的
row 4 的 100 道原生问题上，官方 EM/SUB 口径。

| 臂（row 4, n=100） | EM | SUB |
|---|---:|---:|
| full-memory（无压缩参照） | 30 | 34 |
| 压缩后（旧 row3 窗口 ckpt） | 31 | 34 |
| 压缩后（本次功能 warmup ckpt） | 31 | 34 |

训练本身有效：loss 3.95→3.01（gold 3.93→2.89）。但 EM 逐题判定与旧
ckpt 完全一致（31/100，相异 0 题）；89/100 题预测逐字符相同，11 题差异
全部是措辞/实体层面的随机扰动（如 SteamOS 开发者 Valve→Microsoft、
Endymion 作者换错实体），双向翻车，无系统性增益。

内部状态佐证瓶颈不在训练量：γ 只用到预算的 5.2%（0.00262/0.05），
residual_ratio 0.0013，与 E6a long-B 的结论一致——弱梯度信号下残差
几乎不离开 select_blocks 恒等点。

**结论**：任务匹配的功能监督（gold CE + KD + reader-hidden）在 96 步内
未能让压缩臂超过未训练的 selection 基线。结合全臂读出瓶颈（full-memory
row 4 仅 30/100，格式问题普遍：模板回声、错实体、先解释后模板），瓶颈
分层判定为：**读出/格式 >> 压缩损失**。在 reader 读出问题解决之前，压缩
算子层面的任何改进（换基、换路由、加深度）都无法在 native-QA EM 上体现。
下一步方向不是更多压缩算子变体，而是修生成格式（stop 序列、few-shot
锚定或官方解析适配）。


## 验证与工件

- `tools/unit_native_qa_compare.py`：9 项 CPU 回归通过，包括字段/ID 对齐、
  多答案别名、无损切块及模板预算、EM、成对指标和原子保存失败保护。
- full-rewrite 与 carrier residual GPU 写入、生成和压缩流程均已完成；原始 JSON
  已保存到 `runs/native_qa_rewrite_row{0,4}.json` 与 `runs/native_qa_row{0,4}.json`。
- 官方 scorer 重算已全部完成，含功能 warmup（`runs/native_qa_functional_warmup.json`，
  checkpoint `runs/native_qa_functional_warmup.pt`）。
- JSON 先完整序列化，再临时文件 fsync + replace；避免再次留下半截结果。
