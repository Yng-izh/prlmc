# Native-QA — 原生问答压缩前后成对对照

状态：脚本准备完成，尚未运行 GPU 评测。

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

当前指标是明确命名的 normalized_alias_exact_match：小写、去 ASCII 标点、
去英文冠词、合并空白，匹配任一原生 gold alias。未核实与基准官方 scorer
完全等价，因此不将其称为官方 benchmark 分数。保存逐题原始输出与答案，
可之后用官方 scorer 重评分。

本对照不含 no-memory 臂，故 gained 不代表已排除预训练知识贡献；全记忆
答对也不代表该问题必然依赖 memory。未在本实验上调参或筛选 hard probes。

## 验证与工件

- `tools/unit_native_qa_compare.py`：9 项 CPU 回归通过，包括字段/ID 对齐、
  多答案别名、无损切块及模板预算、EM、成对指标和原子保存失败保护。
- CLI --help 验证通过；未加载 MemoryLLM，未启动 GPU 实验。
- JSON 先完整序列化，再临时文件 fsync + replace；避免再次留下半截结果。
- GPU 生成、实际记忆写入/压缩和官方指标一致性尚待运行验证。
