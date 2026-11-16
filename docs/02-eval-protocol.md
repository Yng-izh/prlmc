# 02 — 评测口径

入口：`tools/eval_lbv2.py`（LongBench v2，128k，四选一）。运行时复用 `tools/eval_core.py`。

## 输入协议

- 文档总预算 128,000 token，`doc_truncation="middle"`；尾部 2,048 token 直接作为 query 前缀；
  其余按 512 token 一块写入记忆。
- 预算 50 块（`--budget-blocks 50`），压缩几何 `window=19 → target=9`，触发点 50 块
  （`compaction_trigger`：预算大于 window+1 时先填满预算再压，避免旧的
  `min(budget, window+1)` 把 50 块预算偷偷压成 20 块）。
- 解码：greedy，`num_beams=1`，`max_new_tokens=128`，seed 0。
- 提示词：官方 0shot 模板（`tools/lbv2_data.py build_prompt`，与基线一致），
  **末尾追加 `ANSWER_CUE = "\nThe correct answer is ("`**。原因是 base 读者会顺着模板续写
  （"For example, ..."），补一个未闭合的句子把续写变成选项，代价为零训练。

## 评分：三套口径同时记录

| 字段 | 含义 | 用途 |
| --- | --- | --- |
| `letter` / `correct` | 官方 `The correct answer is (X)` 正则直接作用于续写 | 与官方/基线对齐 |
| `cue_letter` / `cue_correct` | 把 `ANSWER_CUE` 拼回续写后再走官方正则 | 当前提示词下的有效口径 |
| `option_letter` / `option_correct` | 在 `ANSWER_CUE` 之后那一个位置上直接比较 A/B/C/D 的 logit | 最高分辨率、100% 可解析 |

因为提示词已经以 `ANSWER_CUE` 结尾，官方正则作用在"裸续写"上几乎必然失败
（`official_extraction_rate` 0.05），所以 `correct` 恒为 0，`cue_correct`/`option_correct`
才是当前有信息量的数字。

## 两臂必须同载体

`--carrier-schedule <ours.json>`：Selection 臂不重新选载体，而是逐事件复现 V10 报告里记录的
`compression_schedule`（每次压缩保留下来的块索引）。这样两臂的写入流、容量、载体、解码完全一致，
唯一差别是压缩窗口有没有经过融合算子 —— 这才是 `Ours − Selection`。

## 与原生基线不可直接比较

`reports/main_table/memoryllm8b_lbv2_all_128k.json` 是 **原生 50 块池、不压缩** 的结果，
评分是官方 `extract_answer + exact match judge`（18.7%，n=503）。当前 PRLM-C 两臂是
**50 块工作预算 + 19→9 压缩**，评分是 cue/option 口径。两者在容量与评分上都不是同一把尺子，
所以结果表里必须分开写，不能相减。

要把三者放到同一把尺子上，需要一次改动：去掉 `ANSWER_CUE`（或对裸续写用官方正则），
让两臂都走官方评分，再和原生基线比。
