# main_table — 当前结果与证据

所有文件都来自**当前版本（V10）**的运行，字段见各自报告内的 `protocol` / `identity`。

| 文件 | n | 臂 | 口径 | 结果 |
| --- | --- | --- | --- | --- |
| `memoryllm8b_lbv2_all_128k.json` | 503 | 原生 MemoryLLM-8B | 官方 extract + judge | 18.7%（easy 21.4 / hard 17.0） |
| `eval_v10_50b_n20.json` | 20 | V10 | cue / option | 20.0% / 20.0% |
| `eval_selection_50b_n20.json` | 20 | Selection（同载体） | cue / option | 20.0% / 20.0% |
| `full_v10.training.json` | — | 训练 | — | 704 更新，4.0 h，峰值 38.3 GB |
| `logs/full_v10.log` | — | 训练日志 | — | — |

## 读这些数字时要注意

1. **不同行不能相减**：原生基线是"50 块池、不压缩、官方 judge"；PRLM-C 两臂是
   "50 块工作预算、19→9 压缩、cue/option 口径"。
2. `correct`（官方正则作用在裸续写）恒为 0，因为提示词以 `ANSWER_CUE` 结尾；有效口径是
   `cue_correct` 与 `option_correct`。
3. 曾经那次 157/503 题的中断运行（`eval_v10_lbv2_50b_SharedMerge.partial.json`）已移出仓库，
   它的 18.5% 记在 [docs/03-results.md](../../docs/03-results.md)；那个数按 `cue_correct`
   逐题重算，不是报告自带汇总。
4. 20 题的 ±18pp 误差意味着那一对比较只能说明"没有崩"，不能说明"有效"。

## 复现要点

- checkpoint：`runs/natural_merge_v9/full_v10.pt`，sha256 `a4d1f6e7…47134`（不入 git，9.5 MB）。
- 数据：LongBench v2 `data.json`，sha256 `15d61c22…7fe2`。
- 两臂必须用同一份 `data.json` 与同一组载体（`--carrier-schedule`）。
