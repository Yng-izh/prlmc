## 已修复的缺陷

E3 之前修复的问题，按发现顺序：

1. **触发器被抢跑（根因）。** `write()` 原顺序为 `_place` → `_observe_capacity`。
   `_place` 在块放不下时自己就会压缩，而"超容量"只能由"需要空间"造成，
   因此触发器看到快照时超容量状态永远已解决。这是 E1 中触发 0/82 的真正机制。
   **修复**：触发器判定移到放置之前，`_place` 的压缩降级为兜底。

2. **novelty 数学口径不一致（三处）。** `summarise_block` 跨 layer 平均；
   candidate 未减 `basis_mean`；`_spectrum`/`_refresh_basis` 用 layer 0 而 candidate 用 32 层平均。
   **修复**：统一到 `_REFERENCE_LAYER = 0`，candidate 减同一 `basis_mean`。

3. **`over_budget` 恒为假。** `_headroom_handled` 位于 `while` 循环体内，
   而循环条件本身要求超容量，故标志恒真、`over_budget` 恒假。
   **修复**：重命名为 `made_room_for_write` 并移入循环体，
   `over_budget` 改为在放置**前**计算，语义为"本次写入请求会超容量"。

4. **hysteresis 从未生效。** `tau_k_exit`/`tau_n_exit` 虽被读取，但 `armed` 只在
   触发的那一次置真，紧接着 `on_compacted()` 就清除，exit 阈值从未在 armed 状态下被读到。
   **修复**：`armed` 在超过 `trigger_consecutive` 后置真、由压缩清除；
   `_place` 兜底压缩也会清除 latch。

5. **rank veto 否决一切。** `occupancy_novelty_rank` 在 `energy_rank >= rank_max`
   或 `rank_utilisation >= 1.0` 时置 `above_n = False`。实测 494/500 满足前者、
   36/500 满足后者，导致 0/500 通过。**修复**：删除该 veto。

6. **novelty 不等号方向写反。** 设计文档为 `rho_bar_t >= tau_N` 才压缩，
   标定脚本曾测 `novelty < tau_n`。**修复**：改为正确方向（500/500，说明该条件为空条件）。

7. **`stats_dir` 位置错误。** YAML 顶层的 `stats_dir` 被 `build_config` 静默忽略，
   统计落到 `outputs/prlmc-obs/prlmc_stats.jsonl` 而非 `outputs/prlmc-obs/stats/`。
   **修复**：`stats_dir` 放入 `prlmc:` 块内。

8. **`KeyError('model')`** —— `agent.py:53` 需要 `model` 与 `temperature` 键。
   **修复**：补入配置文件。

9. **打分跨度落在提示词中间（E6a 全部数字作废的根因）。**
   `score_probes` 与 `_probe_batches` 都写作
   `n_prefix = min(len(stem), width - 1)`。渲染后的 `rag_agent` 提示词长 **163 token**，
   而 `max_probe_tokens` 默认 **64**，于是 `n_prefix = 63`，被打分的位置是
   "answer a question" 里的 `' question'`。该位置只由问题文本决定，与 memory 池无关，
   所以 accuracy 恒为满分、`gold_loss` 也不是答案的 loss。
   **修复**：两处都改为 `n_prefix = len(prefix_ids)`，并在
   `len(prefix_ids) >= max_length` 时直接 `raise`；默认值提到 256。

10. **`weights` 没有接进 compactor 调用。**
    `compressed_pool` 与训练器两处调用 `compactor(...)` 时都漏了 `train_weights`，
    于是选择臂用的是未加权的 importance，与 `_select_block_indices` 内部实际使用的
    0.4 加权 importance 不一致。
    **修复**：三处都按第二个位置参数传入 `train_weights`。
    生效证据是筛窗打印的 dropped 集合发生变化：
    `[0,1,2,3,4,5,6,7,8,9] vs [0,1,2,3,4,5,6,8,9,11] (unweighted)`。

11. **rebuild 写在注入上下文内部（梯度走了另一个 carrier 集合）。**
    `_pool_in_place` 在 `with` 块存续期间把 candidate 写进窗口尾部
    （`new_start = start + window - target`），退出时才 `restore()`。
    而 `manager._memory` 返回的是 `model.memory.data`，即 `source` 是池存储的视图。
    原来的 rebuild 写在 `with` 块内，因此读到的 `source` 已被覆写，且同一份被污染的
    `source` 同时用于三处：`_select_block_indices`（对 19 个 block 全部打分，
    含被覆写的尾部）、`src_summary`/`car_summary` 的路由均值、
    以及尾部 carrier 的 `blocks.index_select`。在该窗口上加权选择恰好丢掉
    block 0–9，于是尾部正是 carrier 集合——**反向传播穿过的是与 forward 不同的
    carrier 集合**。报出的 loss 看起来正常，因为它们来自正确构造的 `candidate_val`。
    **修复**：把 rebuild 与 backward 移出 `with` 块。
    `restore()` 已先执行，`source` 与参数均未变，重建结果是逐位相同的。
    AST 校验：`with` 块内 `cand_leaf.grad is not None` 节点数 **0**。

12. **回滚副本整步占用显存（首次训练在 backward 处 OOM）。**
    `_PoolInjection.__init__` 克隆被顶掉的窗口槽位后把 604 MB 留在 GPU 上，
    但 `restore()` 只在步末读一次，且该张量不参与 autograd。
    与此同时 backward 必须物化全宽 `[32, 12800, 4096]` fp16 的 memory 梯度
    （**3.12 GiB**，读全宽就是必须的），当时只剩 3.09 GiB，**差 30 MB**。
    **修复**：副本存到 CPU，`restore()` 时再传回；每步多一次 604 MB 拷贝，
    峰值让出 604 MB。

13. **多窗口训练在窗口边界因分配器碎片 OOM（E6b-mini 首次运行）。**
     崩在 Stage B 窗口 3 第 3 步的 `probe_loss.backward()`：
     请求 3.12 GiB，`of which 3.12 GiB is free`。

     诊断由数字决定，不是猜的：`reserved` 从 Stage A 窗口 1 第 1 步就冲到
     **33.15 GB**，此后 119 步一直钉在 33.6 GB 天花板；`alloc` 稳定在
     22.73/23.33 GB，**完全持平**——没有增长，故不是泄漏。崩溃瞬间活分配
     27.45 GiB、**缓存 766.85 MiB 未分配**，请求 3.12 GiB、空闲 3.12 GiB。
     那 766 MiB 缓存就是全部缺口。

     E6a 在同一包络内跑完 236 步无 OOM，因为它**全程只有一个窗口**，
     probe 集固定、激活形状固定，分配器很快进入稳定态。E6b-mini 每个窗口的
     probe 由该窗口事实行构造，token 长度逐窗口变化，形状一变，上一个窗口
     的缓存块就对不上新请求。

     **修复**：窗口边界调用 `release()` 清空缓存，把分配器拉回单一形状族的
     稳态；`train_window()` 在 OOM 时先回滚 `compactor.state_dict()` 与
     `optimiser.state_dict()` 再重试**一次**——窗口有 24 步，OOM 可能落在中途，
     不回滚会把部分窗口训练两遍，污染"共享 checkpoint"这个前提。修复后
     `EXIT=0`。

     **同轮排除的误判**：OOM 文本里那句
     `try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 是
     **无条件打印**的。在 v100-2 上实测（`tools/alloc_probe.py`），
     设置该变量后同一句话照样出现，故它不能作为"变量未生效"的证据。

14. **【致命，已修】E6b-mini 的 held-out 度量被结构性固定为 0（`window_overfit` 判决作废）。**
     旧 `tools/e6b_mini.py:729` 把**已训练**的 `compactor` 传进 `measure_window`，
     而该函数正是用它去算 `before`；`evaluate_window` 随后用**同一个对象**算 `after`。
     compactor 前向确定性、窗口重建逐位相同、probe 集复用
     → `before == after` → `recovered = lost & after_correct` 恒为空集
     → **任何训练都无法产出非零 held-out 恢复**。

     这不是"泛化失败"，是度量通道没接上：旧实现的 `0/16` 与
     `verdict: window_overfit` **全部作废**，其下游结论
     （「不是过拟合，是表达能力不足」所依赖的零值证据）随之作废。

     **修复**：`measure_window` 增加独立的 `template` + `initial_state` 参数，
     在函数内构造一份 `before_compactor`（`copy.deepcopy` 模板后
     `load_state_dict(copy.deepcopy(initial_state))`），保证 `before` 永远取自
     训练前状态，与调用方持有的 `compactor` 无关。修复后 held-out 两侧
     `after == before` 均为 `False`、`trained_as_before == after` 均为 `True`。

15. **【已修】held-out 的 `n_lost` 取自错误的对象。**
     旧实现里 held-out 的 `n_lost` 是用**训练后**的 compactor 在 60 probes 上算的，
     而同一窗口在筛选阶段（20 probes、初始 compactor）是 `lost` 3 / 6。
     两个数字口径不同却被并列写进报告，制造了"held-out 分母 7/9 远大于训练窗口 2–6"
     的假象。**修复**后 `n_lost` 一律来自初始 compactor，
     筛选值与报告值重新对齐。

16. **【已修】`recovered == 0` 的判定分支没有完整性闸门。**
     旧 `tools/e6b_mini.py:398` 把 `n_recovered == 0` 直接映射到 `window_overfit`，
     而结构性零恰好落进这一分支——**缺陷被自动翻译成了科学结论**。
     修复后 `verdict` 先查 `integrity["ok"]`，未过则输出 `unmeasured` 并附原因；
     另有三条前置断言：无训练阶段 → `"no training stage ran; nothing was trained"`；
     `state_dict` 逐位相同 → `"every update was discarded (NaN gradients?)"`；
     任何窗口都未改变答案 → `"`after` is the initial compactor, so `recovered` was not measured"`。

     另修复一处会用错的判读：`held_all_unchanged` 只作为附注字符串输出，
     **不折进结果分支**——单个窗口碰巧未变不能否决整轮测量。

17. **【未修】E6b-mini 的 compactor 初始化未被固定。**
     `tools/e6b_mini.py` 只在 `if args.seed is not None:` 分支里调用
     `torch.manual_seed(args.seed)`，而 `tools/run_e6b_mini.sh` 不传 `--seed`，
     故该调用从未执行，`CarrierResidualCompactor` 的
     `q / k / down / up` 四个 `nn.Linear` 每次运行都取不同的随机初始化。

     实测后果：同一脚本、同一窗口集、同一探针集的两轮运行，
     16 个 stage 的 loss 逐条不同（Stage B 窗口 1 `4.4756 → 4.1252` 对
     `4.4918 → 4.1718`；窗口 8 `3.3743 → 3.1534` 对 `3.4012 → 3.1865`）；
     训练侧恢复数 6 → 7 即由该差异产生。
     `final_state` 四项只差第 4–6 位有效数字，但逐窗口答案集签名不同：
     旧轮 `after == before` 为 `True` 的是 16750/14470，新轮也是这两个，
     而 17130 由 `True` 变 `False`（`n_lost` 两轮恒为 `3,2,6,5,4,3,5,2`）。

     held-out 两侧的恢复数与签名在两轮里逐位相同（`2/16`、两个窗口各 1 个），
     这属于巧合，不能当作可复现性的证据。

     该缺陷不影响本轮结论的方向（`partial`、两侧差异检不出两轮都成立），
     但意味着**报告里的逐窗口数字不可按位复现**。

     **E6c 延续该状态**：`tools/run_e6c_train.sh` 同样不传 `--seed`，
     训练侧逐窗口数字同样不可按位复现；held-out 3/16 的方向性结论不受影响。

18. **【已修，2026-09-30】`carrier_ckpt` 加载失败被吞掉，carrier 臂静默退化成驱逐臂。**
     `layout.py` 的 `compact_with_report` 用宽 `except` 包住压缩算子调用，
     保证一次坏 forward 不弄死整轮长跑。副作用：`_carrier_compactor` 里
     `torch.load` 抛出的任何错误（文件缺失、外来 state_dict、形状不匹配）
     在同一条路径上被吞成 `reason_skipped: compactor_error: ...`，
     写入没释放任何槽位，`_place` 落到 `_evict_oldest`——
     带 `carrier_ckpt` 的运行**表面上在跑 carrier 臂，实际上在跑驱逐臂**，
     而这恰恰是三臂对比要区分的东西。

     **修复**（`methods/prlmc/layout.py`，+95/−12）：
     (a) 构造函数里若配置了 `carrier_ckpt` 则**立即**加载，把所有失败模式
     变成启动错误而不是运行中静默跳过；
     (b) 抽出 `_load_carrier_state`，对文件不可读、非 state_dict、键不匹配
     （missing/unexpected 各自列出）、非张量项、形状不匹配分别给出
     指名文件的 `RuntimeError`，最终 `load_state_dict(..., strict=True)`；
     (c) 新增 `compactor_errors` 计数器，每次算子抛错时 +1 并进入 `stats()`，
     挂到每条 stats 行上；`reset()` 刻意不清零，持久性失败不会被重置掩盖；
     (d) `gamma_max` 是普通属性而非参数，state_dict 不含它、也无法从文件恢复，
     若 YAML 值与训练值不同会静默缩放所有学到的残差——加载后打印
     `gamma_max × sigmoid(gamma_logit) = effective gamma`，把该风险留在日志里。

     验证：`tools/unit_carrier.py` 14/14 通过；10/10 主机探针（逐一触发
     缺文件/坏键/坏形状并确认 RuntimeError）；一次带 `carrier_ckpt` 的
     运行时探针确认启动即加载。E6c 三臂 bench 的 carrier 臂使用该修复后的
     `layout.py`（md5 `a1e23be610b88617194448db5376ebfe`）。

     训练侧不受影响：`tools/e6b_mini.py` 直接构造 compactor、不走
     `ProtectedResidualMemory` 的这条路径。

---

[← 索引：EXPERIMENTS.md](../EXPERIMENTS.md)
