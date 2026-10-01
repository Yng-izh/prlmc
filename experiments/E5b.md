## E5b — `select_blocks`：importance/diversity 打分选择

### 动机

E5a 只用 recency。E5b 换成打分选择，看已有的 importance 信号能不能胜过"只留最新的"。
仍然只选择、不修改：幸存 block 原样复制，且**按原始时序写回，不按分数顺序**。

```text
score_i = w_imp * importance_i + w_recent * recency_i
        + w_goal * goal_relevance_i + w_div * diversity_i
```

diversity 项在 `_greedy_diverse` 里以 max-min 方式贪心加入，不是静态项。
打分权重 `{importance 0.4, recency 0.3, goal 0.2, diversity 0.1}`，偏向 recency。
`goal` 项用 `self.goal_direction`（与 basis/novelty 同一个 `_REFERENCE_LAYER = 0`）。

### 结果

配置 `prlmc_e5b.yaml`，相对 E5a 只改 `compact_mode: recent_blocks → select_blocks`
与输出目录。

| 指标 | A（none） | E5a（recent） | **E5b（select）** |
|---|---|---|---|
| exact_match | 27.00 | 26.00 | **26.00** |
| f1 | 32.019 | 31.82 | **31.82** |
| substring_exact_match | — | 28.00 | **28.00** |
| rougeL_f1 | 32.101 | 31.90 | **31.90** |

内部信号与 E5a **逐项相同**：5 次压缩、`used 8192`、`peak_used 9984`、
`n_consolidated 2304`、`evicted 0`、`occupancy 0.667`、总时长 805.3 s。

逐条查询比对：100/100 条 `output` 字符串完全相同；26 条 EM 正确项集合完全重合
（`a_only 0`、`b_only 0`）。

### 结论

**E5b 与 E5a 在本任务上等价。** 100 条输出逐字相同，不只是总分相同。

**但这不是"selection 比 recency 好"，也不是"打分类信号有效"。** 两者相同更可能
是因为在 19 → 9 的选择里，本轮打分恰好也基本选中了最近的 9 个 block：`goal`
方向来自首个 chunk，`recency` 权重 0.3 且是唯一在窗口内单调变化的项。

**本轮无法区分这两个算子。** 要真正比较，需要构造一个 recency 明显次优的场景
（例如窗口内事实冲突、正确答案在更老的 block 里），否则 E5b 只是 E5a 的复制。
这是 E5c 之前需要先决定的事。

---

[← 索引：EXPERIMENTS.md](../EXPERIMENTS.md)
