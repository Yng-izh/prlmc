## 未解问题

1. **`memory_construction_time` 慢 21 倍**：E3 为 4.9943 s/chunk，基线 0.2366 s。
   与 accuracy 无关，但 `_refresh_basis` 每次写入都做 SVD，新代码又在
   `_maybe_compact` 前额外算一次 spectrum（已用 `carry=pre` 参数消除重复 SVD）。

2. **`encode_seconds` 口径不一致**：E2 为 p50 25.73 s，E3 为 p50 0.1665 s，
   差 150 倍。E3 的 `memory_construction_time` 却比 E2 快 100 倍。
   两次运行的 `encode_seconds` 与 `memory_construction_time` 都不自洽，
   需要核对计时起止点。

3. **`compact_probe` 惰性**：`register_probes`（`agent.py:356`）无调用点，
   `agent.prlmc_probes` 永不存在，`compact_probe: true` 完全不起作用。
   设计文档 3.4 要求的 probe-guided compaction 从未被执行。

4. **`BudgetReport.byte_saving_ratio` 恒为 0.0**：池是固定形状的参数且被整读，
   压缩减少的是可寻址 token 数，不是已分配字节。这是构造性的，不是 bug。

5. **`detective_qa` 基线退化**：n=6，`output_len 2000`，
   `substring_exact_match` 与 `rougeL_recall` 均为 100，不可汇总。

6. **多卡不可用**：`device_map="auto"` 会破坏 `cat_memory_and_hiddens` 的
   逐层全池读取。可行路线是逐层流水线 + 每卡复制池（池仅 3.36 GB fp16），
   属于日级改动。

7. **E6 的 1 条逐条差异未闭合**：无 `carrier_ckpt` 时 `up` 为零矩阵，
   输出应当与 `select_blocks` 逐位相同（CPU 单测为 0.0），但实测与 E5c
   有 1/100 条 EM 差异。最可能是 `torch.randperm` 的随机性——所有配置的
   `seed` 都是 `None`。要闭合需在同一配置下复跑一次 E5c 以测量噪声基线，
   或把 `seed` 固定后重跑 E5c 与 E6。

8. **所有配置的 `seed` 均为 `None`**：这意味着任何逐条比对都混入了运行间
   随机性，1–3 条量级的差异无法归因。n=100 时 SE ≈ 6.1 分，因此平均分的
   排序同样不可靠。**在固定 seed 之前，不要基于逐条差异下结论。**

9. **probe 打分跨度落在提示词中间（E6a 首轮全部指标作废的根因）。**
   `score_probes` 与 `_probe_batches` 都写 `n_prefix = min(len(stem), width - 1)`。
   渲染后的 `rag_agent` 提示词长 **163** token，而 `max_probe_tokens` 默认 **64**，
   于是 `min(163, 63) = 63`，被打分的"答案"是 `' question'`——"answer a question"
   里的那个词，只靠问题文本即可预测。
   后果：accuracy 与 memory 池完全无关，因此

   - `lost` 在 12 个窗口上恒为 0（`select` 必然等于 `full`）；
   - 把 mask 全部置为屏蔽后仍得 20/20，被误读为"mask 失效"；
   - 把 compacted 池里被丢弃的 block 清零后仍得 20/20。

   SDPA 追踪给出决定性反证：`mask-all-off` 屏蔽了全部 12800 个池位置
   （`pool_all_blocked: 64`，`blocked_frac 0.997`），分数仍是 20/20。
   全屏蔽的池必须等价于无记忆（0/20）。**mask 本身没有问题，先前的 `empty-mask`
   异常随之作废。**

   **修复**：`n_prefix = len(prefix_ids)`，且在 `len(prefix_ids) >= max_length`
   时直接 `raise`，不再静默评错 token；`max_probe_tokens` 默认 64 → **256**
   （`tools/e6a_overfit.py`、`methods/prlmc/probes.py` 两处调用点同改）。
   `score_candidate` 与 `train_carrier_compactor` 的默认值同为 256。

   **影响范围**：`_probe_batches` 同时供 trainer 使用，故 E6a 首轮的 `gold_loss`
   也是错跨度的产物，不只是 accuracy。修复后首个窗口即得 `lost = 3`。

   **同轮修复的另两个真缺陷**：
   `compressed_pool` 的 `weights` 未传入 compactor（加权与不加权选出的
   dropped 集合确实不同：`[0..9]` vs `[0,1,2,3,4,5,6,8,9,11]`）；
   `_compacted_pool` 宽度越界（窗口只剩 18 block 时仍请求 19 block，
   `new_used` 变负，产出 13056 宽）。

10. **E7a 干扰矩阵的 seed 1 缺失**：GPU0 被宿主机进程挤占，seed 1 的
    24 步矩阵在功能臂 3/8 处 OOM 死亡。seed 0/2 的矩阵 64 格仅 3 格差、
    harmed 逐条相同，判读不受影响；seed 1 重跑在 GPU9 进行中
    （PID 68484），产物 `/tmp/e7a_interference_seed1_s24.json`。

11. **内容相似度带宽只在单 context 上测过**：E7a-sim 判"停止门槛 1
    触发"依据的是 row 3 同一篇文章的 8 窗——事实模板同质（人物-运动、
    机构-城市），reader 表征带宽 0.9976-0.9991。跨 context（不同 row
    的窗口）带宽未测。该判决限定于当前 fixture；阶段 B/C 若要复活
    内容路由，前提是先证跨 context 带宽非平。

12. **checkpoint 的 `initialized=1` 与纯文本前向不兼容**
    （E7a-sim 过程中发现）：加载即置位的 buffer 使每次前向把
    12824 slot memory 池拼进 cache_position，短文本批直接维度错位
    崩溃（23 vs 12824）。复位 `initialized=0` 后才是纯文本编码路径。
    任何想复用 reader 做 embedding 的后续实验都会踩到这颗雷。

13. **宿主机 GPU 竞争**：v100-2 的卡被容器外进程长期占用
    （每卡 9-18 GB，容器内不可见不可杀），单卡 fp16 reader（16 GB）
    已无处安放，E7a-sim 被迫双卡分片 + `lm_head→Identity`。后续
    训练实验需按 13 GB 可用预算规划，或等待窗口期。

---

[← 索引：EXPERIMENTS.md](../EXPERIMENTS.md)
